"""v9「土地と建物を分ける町」の月ループ。

設計の正は `docs/world_design_v9.md`。
**既存の `src/simulation.py`・`src/sim_v8*.py`・`src/field_v8*.py`・
`src/llm_client_factory.py` には一切触らない**（読み取り専用で借りるだけ）。

このファイルの責務は v8 系と同じ3つだけ:
  1. 各主体に「今月あなたに見えているもの」を配る
  2. LLM の返した答えを **そのまま** 帳簿に記帳する（解釈・補正・代行をしない）
  3. 記録を残す（理由の一言は原文のまま・判定も分類もしない）

やっていないこと（意図的に）:
  - 誰かの行動を条件分岐で決める
  - 「〜なら売る」「〜%の確率で」といったパラメータ
  - 答えが無かったときに既定値（出さない／売らない／家）を書き込むこと
  - 土地と建物のどちらかを狙わせる誘導
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from .field_v9 import (ACQUIRER_NAME, HOME, KIND_BOTH, KIND_BUILDING,  # noqa: F401
                       KIND_LAND, KIND_VALUES, LIST_BOTH, LIST_BUILDING,
                       LIST_LAND, LIST_NO, LIST_TO_KIND, LIST_VALUES,
                       MAX_OFFER_CHARS, MAX_REASON_CHARS, NO_ANSWER, NOT_ASKED,
                       SELL_NO, SELL_YES, RegistryV9, acquirer_schema_v9,
                       adjacency_v9, build_absentee_prefix_v9,
                       build_acquirer_prefix_v9, build_acquirer_prompt_v9,
                       build_common_prefix_v9, build_decide_prompt_v9,
                       build_plan_prompt_v9, build_scene_prompt_v9,
                       decide_schema_v9, delivered_offer_v9, load_personas_v9,
                       plan_schema_v9, rotate, scene_schema_v8, sell_order,
                       transfer_notice)
from .llm_client_factory import UsageMeter
from .sim_v8 import parse_json
from .sim_v8b import CostLimitReached
from .sim_v8d import DEFAULT_REQUEST_TIMEOUT_SEC, TimeoutGeminiClient

logger = logging.getLogger(__name__)

RESULT_SOLD = "売った"
RESULT_NOT_SOLD = "売らなかった"
RESULT_NO_ANSWER = "答えが返らなかった"


class MockV9Client:
    """API を叩かずに配線と集計を通すためだけの client。

    **世界の挙動を作り込まない**：返す値はスキーマを満たす最小限で、選択は seed 固定の
    乱数で決める（実験の結果としては一切使わない）。
    区画と種別だけは帳簿を見て**実行できる組み合わせ**を選ぶ（配線を通すため。
    mock 専用の便宜であって、実APIのX社には登記簿しか渡らない）。
    """

    def __init__(self, seed: int = 42, usage: Optional[UsageMeter] = None,
                 reg: Optional[RegistryV9] = None, send_rate: float = 0.6,
                 sell_rate: float = 0.10):
        self.rnd = random.Random(seed)
        self.usage = usage or UsageMeter()
        self.reg = reg
        self.send_rate = send_rate
        self.sell_rate = sell_rate
        self.cache_created = 0
        self.cache_failed = 0
        self.timeout_retries = 0
        self.timeout_giveups = 0

    def close_caches(self) -> int:
        return 0

    def count_tokens(self, text: str) -> int:
        return max(1, int(len(text) * 0.9))

    def _pick_offer(self, name: str) -> Optional[Tuple[str, str]]:
        if self.reg is None:
            return None
        aid = self.reg.id_of_name.get(name)
        if aid is None:
            return None
        cand = [(p, k) for p in self.reg.parcels_owned(aid) for k in KIND_VALUES
                if self.reg.can_offer(aid, p, k)]
        return self.rnd.choice(cand) if cand else None

    def generate(self, system_prompt: str, user_prompt: str,
                 schema: Optional[Dict[str, Any]] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, tag: str = "agent") -> str:
        self.usage.add(tag, input_tokens=len(user_prompt) // 3, output_tokens=55)
        props = (schema or {}).get("properties", {})
        if "offers" in props:
            item = props["offers"]["items"]["properties"]
            rows = []
            for n in item["to"]["enum"]:
                send = self.rnd.random() < self.send_rate
                pick = self._pick_offer(n) if send else None
                parcel, kind = (pick if pick
                                else (self.rnd.choice(item["parcel"]["enum"]),
                                      KIND_LAND))
                row = {"to": n, "send": send, "parcel": parcel, "kind": kind,
                       "text": (f"{parcel}の{kind}を譲り受けたい。" if send else "")}
                if "reason" in item:
                    row["reason"] = ("登記簿の並びから" if self.rnd.random() < 0.7
                                     else "")
                rows.append(row)
            return json.dumps({"offers": rows}, ensure_ascii=False)
        if "go" in props:
            return json.dumps({"thought": "今月のこと。",
                               "go": self.rnd.choice(props["go"]["enum"])},
                              ensure_ascii=False)
        if "listings" in props or "sell" in props:
            out: Dict[str, Any] = {"thought": "決めた。"}
            if "listings" in props:
                out["listings"] = {
                    p: self.rnd.choice(spec["enum"])
                    for p, spec in props["listings"]["properties"].items()}
                out["listing_reasons"] = {
                    p: ("家の事情で" if self.rnd.random() < 0.7 else "")
                    for p in props["listings"]["properties"]}
            if "sell" in props:
                out["sell"] = (SELL_YES if self.rnd.random() < self.sell_rate
                               else SELL_NO)
                out["sell_reason"] = ("条件が分からないから"
                                      if self.rnd.random() < 0.7 else "")
            return json.dumps(out, ensure_ascii=False)
        if "text" in props:
            others = props["talk_to"]["items"]["enum"]
            return json.dumps({"thought": "話すか。", "text": "最近どうですか。",
                               "talk_to": [self.rnd.choice(others)] if others else []},
                              ensure_ascii=False)
        return "{}"


# 月ごとに書き出すもの（属性名 → ファイル名）。**上書き**で書く。
CHECKPOINT_FILES_V9 = (
    ("monthly", "monthly.json"),
    ("offers", "offers.json"),
    ("decisions", "decisions.json"),
    ("plans", "plans.json"),
    ("utterances", "utterances.json"),
    ("listings", "listings.json"),
    ("deliveries", "deliveries.json"),
    ("acquirer_raw", "acquirer_raw.json"),
    ("ledger_by_step", "ledger_by_step.json"),
    ("notices", "notices.json"),
)


class SimulationV9:

    def __init__(self, cfg: Dict[str, Any], run_dir: str):
        if cfg.get("scenario_version") != "field_v9":
            raise ValueError("config の scenario_version が field_v9 ではない")
        self.cfg = cfg
        self.run_dir = run_dir
        self.n_steps = int(cfg["steps"])
        self.chat = bool(cfg.get("chat", True))
        self.scene_rounds = int(cfg.get("scene_rounds", 1))
        self.chunk = int(cfg.get("acquirer_chunk", 10))
        self.seed = int(cfg.get("seed", 42))
        self.workers = int(cfg["llm"].get("parallel_workers", 4))
        self.acquirer_reason = bool(cfg.get("acquirer_reason", True))

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pf = str(cfg["personas_file"])
        if not os.path.isabs(pf):
            pf = os.path.join(root, pf)
        self.agents, self.parcels = load_personas_v9(pf)
        self.reg = RegistryV9(self.agents, self.parcels)

        venues = cfg.get("social", {}).get("venues", [])
        self.venue_labels = {str(v["id"]): str(v["label"]) for v in venues}
        if len(set(self.venue_labels.values())) != len(self.venue_labels):
            raise ValueError("場所の名前が重複している")
        self.venue_order = [str(v["id"]) for v in venues]
        self.all_labels = [self.venue_labels[v] for v in self.venue_order]
        self.label_to_venue = {lb: vid for vid, lb in self.venue_labels.items()}
        # 隣近所（開始時の使用者どうし）。行き先に関わらず毎月ひと言が届く。
        self.neighbours = adjacency_v9(self.reg)
        self.neighbour_names = {aid: [self.reg.name_of[x] for x in nbs]
                                for aid, nbs in self.neighbours.items()}

        self.usage = UsageMeter()
        provider = str(cfg["llm"].get("provider", "mock")).lower()
        if provider == "mock":
            self.client = MockV9Client(seed=self.seed, usage=self.usage,
                                       reg=self.reg)
        else:
            self.client = TimeoutGeminiClient(
                request_timeout_sec=float(cfg["llm"].get(
                    "request_timeout_sec", DEFAULT_REQUEST_TIMEOUT_SEC)),
                model=str(cfg["llm"].get("model", "gemini-2.5-flash-lite")),
                temperature=float(cfg["llm"].get("temperature", 0.75)),
                max_tokens=int(cfg["llm"].get("max_tokens", 2200)),
                enable_cache=bool(cfg["llm"].get("enable_cache", False)),
                usage=self.usage,
                parallel_workers=int(cfg["llm"].get("parallel_workers", 4)))
        self.max_tokens = int(cfg["llm"].get("max_tokens", 2200))
        self.max_cost_usd = float(cfg.get("max_cost_usd", 0) or 0)
        self.stopped_by_cost = False
        self.partial_month: Optional[int] = None

        n_parcels = len(self.parcels)
        self.common_prefix = build_common_prefix_v9(cfg, self.agents, n_parcels)
        self.absentee_prefix = build_absentee_prefix_v9(cfg, n_parcels)
        self.acquirer_prefix = build_acquirer_prefix_v9(cfg, self.reg)
        self.common_prefix_tokens: Optional[int] = None

        # 主体ごとの持ち越し内心
        self.thought: Dict[str, str] = {str(a["id"]): "" for a in self.agents}
        # 記録
        self.offers: List[Dict[str, Any]] = []
        self.plans: List[Dict[str, Any]] = []
        self.utterances: List[Dict[str, Any]] = []
        self.decisions: List[Dict[str, Any]] = []
        self.monthly: List[Dict[str, Any]] = []
        self.deliveries: List[Dict[str, Any]] = []
        self.listings: List[Dict[str, Any]] = []
        self.acquirer_raw: List[Dict[str, Any]] = []
        self.ledger_by_step: List[Dict[str, Any]] = []
        self.notices: List[Dict[str, Any]] = []
        self.left_agents: List[Dict[str, Any]] = []
        self.parse_fail: List[Dict[str, Any]] = []
        self.listed_by_step: Dict[int, List[Tuple[str, str]]] = {}
        # 健全性カウンタ（欠損を既定値に丸めないための見張り）
        self.no_answer = 0
        self.truncated = 0
        self.invalid_venue = 0
        self.plan_no_answer = 0
        self.invalid_listing = 0
        self.listing_missing = 0
        self.invalid_sell = 0
        self.sell_no_answer = 0
        self.acquirer_missing_targets = 0
        self.acquirer_dup_rows = 0
        self.acquirer_off_range = 0
        self.acquirer_chunk_fail = 0
        self.acquirer_empty_text = 0
        self.acquirer_invalid_offer = 0
        self.acquirer_missing_parcel = 0
        self.declines_delivered = 0
        self.reason_counts: Dict[str, int] = {
            "listing_total": 0, "listing_blank": 0,
            "sell_total": 0, "sell_blank": 0,
            "acquirer_total": 0, "acquirer_blank": 0,
        }
        self.checkpoint_dir = os.path.join(run_dir, "checkpoint")
        self.checkpoints_written = 0

    # -- LLM -------------------------------------------------------------------

    def _call(self, items: List[Tuple[str, str, str, Dict[str, Any], str]],
              tag: str) -> Dict[str, str]:
        """items = [(key, system_prompt, user_prompt, schema, tag_suffix)]"""
        if not items:
            return {}

        def one(it):
            key, system, up, schema, sfx = it
            raw = self.client.generate(system, up, schema=schema,
                                       max_tokens=self.max_tokens,
                                       tag=f"v9:{tag}{sfx}")
            return key, raw

        out: Dict[str, str] = {}
        workers = max(1, min(self.workers, len(items)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for key, raw in ex.map(one, items):
                out[key] = raw
        return out

    def _read(self, key: str, raw: str, step: int,
              tag: str) -> Optional[Dict[str, Any]]:
        act, truncated = parse_json(raw)
        if truncated:
            self.truncated += 1
        if act is None:
            self.parse_fail.append({"step": step, "who": key, "tag": tag,
                                    "raw": (raw or "")[:2000]})
            return None
        return act

    def _cost_so_far(self) -> float:
        usage = self.usage.as_dict()
        price = (self.cfg.get("cost", {}).get("price_table", {})
                 .get(self.cfg["llm"].get("model", ""), {}))
        billed_in = max(0, usage["input_tokens"] - usage["cached_tokens"])
        return (billed_in / 1e6 * float(price.get("input", 0.0))
                + usage["cached_tokens"] / 1e6 * float(price.get("cache_read", 0.0))
                + usage["output_tokens"] / 1e6 * float(price.get("output", 0.0)))

    def _guard_cost(self) -> None:
        if self.max_cost_usd > 0 and self._cost_so_far() >= self.max_cost_usd:
            raise CostLimitReached()

    def _system_for(self, aid: str) -> str:
        return (self.common_prefix if self.reg.is_resident(aid)
                else self.absentee_prefix)

    # -- 所有権が動いたことの通知（月初・事実1行） -------------------------------

    def _notices_for(self, step: int) -> Dict[str, List[str]]:
        """先月末に所有権が移った区画の**使用者**に届く事実1行（施主 10:54(5)）。"""
        out: Dict[str, List[str]] = {}
        for t in self.reg.transfers:
            if int(t["step"]) != step - 1:
                continue
            user = self.reg.user_of(t["parcel"])
            if user is None or user == t["agent_id"]:
                continue
            for kind in t["moved"]:
                line = transfer_notice(t["parcel"], kind,
                                       t["before"].get(kind, ""), ACQUIRER_NAME)
                out.setdefault(user, []).append(line)
                self.notices.append({"step": step, "to": user,
                                     "to_name": self.reg.name_of[user],
                                     "parcel": t["parcel"], "kind": kind,
                                     "text": line})
        return out

    # -- X社の月次 --------------------------------------------------------------

    def _acquirer_turn(self, step: int) -> Dict[str, Dict[str, Any]]:
        risk = self.reg.risk_set()
        if not risk:
            return {}
        names = [self.reg.name_of[aid] for aid in risk]
        chunks = [names[i:i + self.chunk] for i in range(0, len(names), self.chunk)]
        listed_rows = [f"{p}（{LIST_TO_KIND[v]}）"
                       for _aid, p, v in self.listed_by_step.get(step - 1, [])]
        items = []
        for n, targets in enumerate(chunks, 1):
            parcels: List[str] = []
            for name in targets:
                parcels += self.reg.parcels_owned(self.reg.id_of_name[name])
            up = build_acquirer_prompt_v9(self.reg, step, self.n_steps, targets,
                                          self.offers, listed_rows, parcels,
                                          n, len(chunks),
                                          with_reason=self.acquirer_reason)
            items.append((f"X{n}", self.acquirer_prefix, up,
                          acquirer_schema_v9(targets, parcels,
                                             self.acquirer_reason), ":acquirer"))
        raws = self._call(items, "acquirer")
        for k in sorted(raws):
            self.acquirer_raw.append({"step": step, "chunk": k,
                                      "raw": (raws[k] or "")[:8000]})

        sent: Dict[str, Dict[str, Any]] = {}
        seen: set = set()
        for n, targets in enumerate(chunks, 1):
            act = self._read(f"X{n}", raws.get(f"X{n}", ""), step, "acquirer")
            if act is None:
                self.acquirer_chunk_fail += 1
                continue
            for row in (act.get("offers") or []):
                to = str(row.get("to", "") or "")
                if to not in targets:
                    self.acquirer_off_range += 1
                    continue
                if to in seen:
                    self.acquirer_dup_rows += 1
                seen.add(to)
                if self.acquirer_reason:
                    self.reason_counts["acquirer_total"] += 1
                    if not str(row.get("reason", "") or "").strip():
                        self.reason_counts["acquirer_blank"] += 1
                text = str(row.get("text", "") or "").strip()
                if bool(row.get("send")) and not text:
                    self.acquirer_empty_text += 1
                if not bool(row.get("send")) or not text:
                    continue
                parcel = str(row.get("parcel", "") or "").strip()
                kind = str(row.get("kind", "") or "").strip()
                if not parcel or kind not in KIND_VALUES:
                    self.acquirer_missing_parcel += 1
                    continue
                aid = self.reg.id_of_name[to]
                # 世界は**実行できない提示を配らない**（数えるだけ・X社には教えない）
                if not self.reg.can_offer(aid, parcel, kind):
                    self.acquirer_invalid_offer += 1
                    continue
                sent[aid] = {
                    "parcel": parcel, "kind": kind,
                    "text": text[:MAX_OFFER_CHARS],
                    "delivered": delivered_offer_v9(text[:MAX_OFFER_CHARS]),
                    "reason": str(row.get("reason", "") or "").strip()[:MAX_REASON_CHARS],
                }
        self.acquirer_missing_targets += len([x for x in names if x not in seen])
        for aid, o in sent.items():
            self.offers.append({
                "step": step, "to": self.reg.name_of[aid], "to_id": aid,
                "to_in_town": self.reg.is_resident(aid),
                "parcel": o["parcel"], "kind": o["kind"], "text": o["text"],
                "delivered": o["delivered"], "reason": o["reason"],
                "result": RESULT_NOT_SOLD, "accepted": False,
                "decline_reason": ""})
        return sent

    # -- 月初の行き先（町にいる人だけ） -------------------------------------------

    def _plan_turn(self, step: int, offers: Dict[str, Dict[str, Any]],
                   notices: Dict[str, List[str]]) -> Dict[str, str]:
        town = self.reg.in_town_ids()
        items = []
        for aid in town:
            a = self.reg.by_id[aid]
            up = build_plan_prompt_v9(a, self.reg, step, self.n_steps,
                                      self.all_labels, self.thought[aid],
                                      offers.get(aid), notices.get(aid),
                                      neighbours=self.neighbour_names.get(aid))
            items.append((aid, self.common_prefix, up,
                          plan_schema_v9(self.all_labels), ":plan"))
        raws = self._call(items, "plan")

        go: Dict[str, str] = {}
        for aid in town:
            a = self.reg.by_id[aid]
            act = self._read(aid, raws.get(aid, ""), step, "plan")
            where = NO_ANSWER      # 既定で埋めない（答えが無かった事実を残す）
            if act is not None:
                self.thought[aid] = str(act.get("thought", "") or "")[:600]
                value = str(act.get("go", "") or "").strip()
                if value in self.label_to_venue:
                    where = self.label_to_venue[value]
                elif value == HOME:
                    where = HOME
                else:
                    self.invalid_venue += 1
            if where == NO_ANSWER:
                self.plan_no_answer += 1
            go[aid] = where
            self.plans.append({"step": step, "agent_id": aid, "name": a["name"],
                               "go": where,
                               "go_label": (NO_ANSWER if where == NO_ANSWER
                                            else self.venue_labels.get(where, HOME)),
                               "thought": self.thought[aid]})
        return go

    # -- 場の集まり --------------------------------------------------------------

    def _scene_turn(self, step: int,
                    go: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
        town = set(self.reg.in_town_ids())
        heard: Dict[str, List[Dict[str, Any]]] = {aid: [] for aid in town}
        if not self.chat:
            return heard
        groups: Dict[str, List[str]] = {}
        for aid, venue in go.items():
            if venue in (HOME, NO_ANSWER) or aid not in town:
                continue
            groups.setdefault(venue, []).append(aid)
        groups = {v: sorted(m) for v, m in groups.items() if len(m) >= 2}
        if not groups:
            return heard
        absent = {aid for aid, v in go.items() if v == NO_ANSWER}

        for rnd in range(1, self.scene_rounds + 1):
            items = []
            ctx: Dict[str, Tuple[str, List[str]]] = {}
            for venue, members in sorted(groups.items()):
                present = [self.reg.name_of[m] for m in members]
                for aid in members:
                    a = self.reg.by_id[aid]
                    up = build_scene_prompt_v9(a, self.reg, step, self.n_steps,
                                               self.thought[aid],
                                               self.venue_labels[venue], present)
                    items.append((aid, self.common_prefix, up,
                                  scene_schema_v8(present, a["name"]), ":scene"))
                    ctx[aid] = (venue, members)
            raws = self._call(items, f"scene_r{rnd}")

            spoken: List[Dict[str, Any]] = []
            for aid in sorted(ctx):
                a = self.reg.by_id[aid]
                venue, members = ctx[aid]
                act = self._read(aid, raws.get(aid, ""), step, "scene")
                if act is None:
                    continue
                self.thought[aid] = str(act.get("thought", "") or "")[:600]
                text = str(act.get("text", "") or "").strip()
                present_names = [self.reg.name_of[m] for m in members]
                raw_to = [str(t) for t in (act.get("talk_to") or [])]
                talk_to = [t for t in raw_to
                           if t in present_names and t != a["name"]]
                if not text:
                    continue
                row = {"step": step, "round": rnd, "venue": venue,
                       "venue_label": self.venue_labels[venue],
                       "from_id": aid, "from": a["name"], "text": text,
                       "thought": self.thought[aid], "talk_to": talk_to,
                       "heard_by": [n for n in present_names if n != a["name"]]}
                self.utterances.append(row)
                spoken.append(row)

            # 配送は2経路（v8 と同じ）。重複は1回に畳む。
            for row in spoken:
                _venue, members = ctx[row["from_id"]]
                routes: Dict[str, str] = {}
                for mid in members:
                    if mid != row["from_id"]:
                        routes.setdefault(mid, "居合わせ")
                for mid in self.neighbours.get(row["from_id"], []):
                    if mid != row["from_id"]:
                        routes.setdefault(mid, "隣近所")
                for mid, route in routes.items():
                    if mid not in town or mid in absent:
                        continue
                    heard[mid].append({**row, "route": route})
                    self.deliveries.append(
                        {"step": step, "to": mid, "to_name": self.reg.name_of[mid],
                         "from": row["from"], "from_id": row["from_id"],
                         "route": route, "venue_label": row["venue_label"],
                         "text": row["text"]})
        return heard

    # -- 月末の問い ---------------------------------------------------------------

    def _decide_turn(self, step: int, offers: Dict[str, Dict[str, Any]],
                     heard: Dict[str, List[Dict[str, Any]]],
                     notices: Dict[str, List[str]]
                     ) -> Tuple[List[str], List[str], List[Tuple[str, str, str]]]:
        """月末の問い（①持ち物ごとの出品 ②2択の売る/売らない）。

        返り値 = (売ると答えた人, 売買の答えが返らなかった人, [(人, 区画, 出品の答え)])
        """
        risk = self.reg.risk_set()
        items = []
        opts: Dict[str, List[Tuple[str, List[str]]]] = {}
        sorders: Dict[str, List[str]] = {}
        for aid in risk:
            a = self.reg.by_id[aid]
            in_town = self.reg.is_resident(aid)
            offer = offers.get(aid)
            lo = [(p, rotate(self.reg.listing_options(aid, p),
                             int(a["index"]) * 7 + i, step))
                  for i, p in enumerate(self.reg.parcels_owned(aid))]
            opts[aid] = lo
            so = sell_order(int(a["index"]), step) if offer else None
            if so:
                sorders[aid] = so
            up = build_decide_prompt_v9(
                a, self.reg, step, self.n_steps, self.thought[aid], offer,
                heard.get(aid, []), listing_options=lo, sell_order_=so,
                notices=notices.get(aid),
                neighbours=(self.neighbour_names.get(aid) if in_town else None),
                in_town=in_town)
            items.append((aid, self._system_for(aid), up,
                          decide_schema_v9(lo, so), ":decide"))
        raws = self._call(items, "decide")

        sellers: List[str] = []
        sell_blanks: List[str] = []
        listers: List[Tuple[str, str, str]] = []
        for aid in risk:
            a = self.reg.by_id[aid]
            offer = offers.get(aid)
            act = self._read(aid, raws.get(aid, ""), step, "decide")
            rec: Dict[str, Any] = {
                "step": step, "agent_id": aid, "name": a["name"],
                "in_town": self.reg.is_resident(aid),
                "listings": {p: NO_ANSWER for p, _o in opts[aid]},
                "listing_reasons": {p: "" for p, _o in opts[aid]},
                "sell": (NO_ANSWER if offer else NOT_ASKED), "sell_reason": "",
                "thought": "",
                "offer": (offer["delivered"] if offer else ""),
                "offer_parcel": (offer["parcel"] if offer else ""),
                "offer_kind": (offer["kind"] if offer else ""),
                "listing_options": {p: o for p, o in opts[aid]},
                "sell_order": sorders.get(aid, []),
                "heard": len(heard.get(aid, [])),
            }
            if act is None:
                # **既定値で埋めない**。答えが無かった事実を残す。
                self.no_answer += 1
                if offer:
                    self.sell_no_answer += 1
                    sell_blanks.append(aid)
                self.decisions.append(rec)
                continue
            thought = str(act.get("thought", "") or "")
            self.thought[aid] = thought[:600]
            rec["thought"] = thought

            if opts[aid]:
                answers = act.get("listings") or {}
                if not isinstance(answers, dict):
                    answers = {}
                reasons = act.get("listing_reasons") or {}
                if not isinstance(reasons, dict):
                    reasons = {}
                for parcel, _allowed in opts[aid]:
                    r = str(reasons.get(parcel, "") or "").strip()[:MAX_REASON_CHARS]
                    rec["listing_reasons"][parcel] = r
                    self.reason_counts["listing_total"] += 1
                    if not r:
                        self.reason_counts["listing_blank"] += 1
                for parcel, allowed in opts[aid]:
                    v = str(answers.get(parcel, "") or "").strip()
                    if v in allowed:
                        rec["listings"][parcel] = v
                        if v != LIST_NO:
                            listers.append((aid, parcel, v))
                    elif v:
                        self.invalid_listing += 1
                        self.no_answer += 1
                    else:
                        self.listing_missing += 1
                        self.no_answer += 1

            if offer:
                sell = str(act.get("sell", "") or "").strip()
                rec["sell_reason"] = str(
                    act.get("sell_reason", "") or "").strip()[:MAX_REASON_CHARS]
                self.reason_counts["sell_total"] += 1
                if not rec["sell_reason"]:
                    self.reason_counts["sell_blank"] += 1
                if sell in (SELL_YES, SELL_NO):
                    rec["sell"] = sell
                    if sell == SELL_YES:
                        sellers.append(aid)
                else:
                    self.sell_no_answer += 1
                    self.invalid_sell += 1
                    sell_blanks.append(aid)
            self.decisions.append(rec)
        return sellers, sell_blanks, listers

    # -- 月 ------------------------------------------------------------------------

    def _step(self, step: int) -> None:
        notices = self._notices_for(step)
        offers = self._acquirer_turn(step)
        self._guard_cost()
        go = self._plan_turn(step, offers, notices)
        self._guard_cost()
        heard = self._scene_turn(step, go)
        self._guard_cost()
        sellers, sell_blanks, listers = self._decide_turn(step, offers, heard,
                                                          notices)

        # 所有権が動くのは②で「売る」と答えたときだけ（出品では動かない）
        moved_rows: List[Dict[str, Any]] = []
        for aid in sellers:
            o = offers[aid]
            row = self.reg.apply_transfer(aid, o["parcel"], o["kind"], step)
            moved_rows.append(row)
            if row["left_town"]:
                self.left_agents.append({"step": step, "agent_id": aid,
                                         "name": self.reg.name_of[aid],
                                         "parcel": o["parcel"],
                                         "still_owns": self.reg.parcels_owned(aid)})

        # 断りの一言＝「売らない」と答えた人が書いた理由。**その提示にだけ**紐づける。
        decline = {d["agent_id"]: str(d.get("sell_reason", "") or "")
                   for d in self.decisions
                   if d["step"] == step and d.get("sell") == SELL_NO}
        for o in self.offers:
            if o["step"] != step:
                continue
            if o["to_id"] in sellers:
                o["result"], o["accepted"] = RESULT_SOLD, True
            elif o["to_id"] in sell_blanks:
                o["result"], o["accepted"] = RESULT_NO_ANSWER, False
            else:
                note = decline.get(o["to_id"], "")
                if note:
                    o["decline_reason"] = note
                    self.declines_delivered += 1

        # 出品の記録（その月かぎり・翌月にX社が見る）。
        # 同じ月に売れて所有権が動いた区画は、翌月の公の一覧には残さない。
        sold_parcels = {(r["agent_id"], r["parcel"]) for r in moved_rows}
        self.listed_by_step[step] = [(aid, p, v) for aid, p, v in listers
                                     if (aid, p) not in sold_parcels]
        for aid, parcel, v in listers:
            self.listings.append({"step": step, "agent_id": aid,
                                  "name": self.reg.name_of[aid],
                                  "in_town": self.reg.is_resident(aid),
                                  "parcel": parcel, "listing": v,
                                  "kind": LIST_TO_KIND[v],
                                  "sold_same_month": (aid, parcel) in sold_parcels})

        self.ledger_by_step.append({"step": step, "rows": self.reg.ledger_rows()})

        risk = self.reg.risk_set()
        by_venue = {self.venue_labels[v]: sum(1 for w in go.values() if w == v)
                    for v in self.venue_order}
        by_venue[HOME] = sum(1 for w in go.values() if w == HOME)
        na_go = sum(1 for w in go.values() if w == NO_ANSWER)
        if na_go:
            by_venue[NO_ANSWER] = na_go
        heard_counts = [len(v) for v in heard.values()] or [0]
        offers_in_town = sum(1 for aid in offers if self.reg.is_resident(aid))
        sold_in_town = sum(1 for aid in sellers if self.reg.is_resident(aid))
        self.monthly.append({
            "step": step,
            "offers_sent": len(offers),
            "offers_to_in_town": offers_in_town,
            "offers_to_absentee": len(offers) - offers_in_town,
            "offers_by_kind": {k: sum(1 for o in offers.values()
                                      if o["kind"] == k) for k in KIND_VALUES},
            "listed_this_month": len(listers),
            "listed_by_kind": {k: sum(1 for _a, _p, v in listers
                                      if LIST_TO_KIND[v] == k)
                               for k in KIND_VALUES},
            "sold_this_month": len(sellers),
            "accepted_this_month": len(sellers),
            "sold_in_town": sold_in_town,
            "sold_absentee": len(sellers) - sold_in_town,
            "sold_by_kind": {k: sum(1 for r in moved_rows if r["kind"] == k)
                             for k in KIND_VALUES},
            "parcels_this_month": len(moved_rows),
            "parcels_cum": len(self.reg.acquired_parcels()),
            "land_cum": len(self.reg.acquired_land()),
            "building_cum": len(self.reg.acquired_buildings()),
            "both_cum": len(self.reg.acquired_both()),
            "left_this_month": sum(1 for r in moved_rows if r["left_town"]),
            "left_cum": len(self.reg.left_ids()),
            "in_town": len(self.reg.in_town_ids()),
            "absentee_owners": len(self.reg.absentee_owner_ids()),
            "no_user_parcels": sum(1 for p in self.reg.parcel_names
                                   if self.reg.user_of(p) is None),
            "risk_set": len(risk),
            "notices": sum(1 for n in self.notices if n["step"] == step),
            "attended": sum(1 for v in go.values()
                            if v not in (HOME, NO_ANSWER)),
            "no_answer_go": na_go,
            "by_venue": by_venue,
            "utterances": sum(1 for u in self.utterances if u["step"] == step),
            "declines_with_reason": sum(1 for o in self.offers
                                        if o["step"] == step
                                        and o.get("decline_reason")),
            "heard_mean": round(sum(heard_counts) / len(heard_counts), 2),
            "heard_max": max(heard_counts),
            "heard_min": min(heard_counts),
        })
        try:
            self._checkpoint(step)
        except Exception as e:  # noqa: BLE001
            logger.warning("チェックポイントの書き出しに失敗（続行）: %s", e)

    # -- チェックポイント ------------------------------------------------------------

    def _checkpoint(self, step: int) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        for attr, name in CHECKPOINT_FILES_V9:
            obj = getattr(self, attr, None)
            if obj is None:
                continue
            tmp = os.path.join(self.checkpoint_dir, name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            os.replace(tmp, os.path.join(self.checkpoint_dir, name))
        tmp = os.path.join(self.checkpoint_dir, "transfers.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.reg.transfers, f, ensure_ascii=False, indent=2)
        os.replace(tmp, os.path.join(self.checkpoint_dir, "transfers.json"))
        m = self.monthly[-1] if self.monthly else {}
        row = {"step": step, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "cost_usd": round(self._cost_so_far(), 4),
               "offers_sent": m.get("offers_sent", 0),
               "listed_this_month": m.get("listed_this_month", 0),
               "sold_this_month": m.get("sold_this_month", 0),
               "land_cum": m.get("land_cum", 0),
               "building_cum": m.get("building_cum", 0),
               "both_cum": m.get("both_cum", 0),
               "left_cum": m.get("left_cum", 0),
               "utterances_cum": len(self.utterances),
               "timeout_retries": getattr(self.client, "timeout_retries", 0),
               "timeout_giveups": getattr(self.client, "timeout_giveups", 0)}
        with open(os.path.join(self.checkpoint_dir, "log.jsonl"), "a",
                  encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.checkpoints_written += 1
        print(f"[v9] m{step:02d}/{self.n_steps} 提示{row['offers_sent']} "
              f"出品{row['listed_this_month']} 売却{row['sold_this_month']} "
              f"土地{row['land_cum']}/建物{row['building_cum']}/両方{row['both_cum']} "
              f"退場{row['left_cum']} ${row['cost_usd']:.4f}", flush=True)

    # -- run ----------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        t0 = time.time()
        try:
            self.common_prefix_tokens = int(
                self.client.count_tokens(self.common_prefix))
        except Exception as e:  # noqa: BLE001
            logger.warning("共通前置きのトークン実測に失敗: %s", e)
        for step in range(1, self.n_steps + 1):
            try:
                self._step(step)
            except CostLimitReached:
                self.stopped_by_cost = True
                self.partial_month = step
                logger.error("費用上限 $%.4f に達したので第%d月の途中で止める（$%.4f）",
                             self.max_cost_usd, step, self._cost_so_far())
                break
            m = self.monthly[-1]
            logger.info("m%02d 提示%d 出品%d 売却%d 土地%d 建物%d 退場%d $%.4f",
                        step, m["offers_sent"], m["listed_this_month"],
                        m["sold_this_month"], m["land_cum"], m["building_cum"],
                        m["left_cum"], self._cost_so_far())
            if self.max_cost_usd > 0 and self._cost_so_far() >= self.max_cost_usd:
                self.stopped_by_cost = True
                logger.error("費用上限 $%.4f に達したので第%d月の終わりで止める",
                             self.max_cost_usd, step)
                break
        elapsed = time.time() - t0
        try:
            self.client.close_caches()
        except Exception as e:  # noqa: BLE001
            logger.warning("キャッシュ削除に失敗（TTLで消える）: %s", e)
        return self._finalize(elapsed)

    # -- 集計と出力 -----------------------------------------------------------------

    def _dump_timeline(self) -> None:
        out_dir = os.path.join(self.run_dir, "timeline_v9")
        os.makedirs(out_dir, exist_ok=True)
        plans = {(r["step"], r["agent_id"]): r for r in self.plans}
        said: Dict[Any, List[Dict[str, Any]]] = {}
        for u in self.utterances:
            said.setdefault((u["step"], u["from_id"]), []).append(u)
        got: Dict[Any, List[Dict[str, Any]]] = {}
        for d in self.deliveries:
            got.setdefault((d["step"], d["to"]), []).append(d)
        dec = {(d["step"], d["agent_id"]): d for d in self.decisions}
        offers = {(o["step"], o["to_id"]): o for o in self.offers}
        notes: Dict[Any, List[str]] = {}
        for n in self.notices:
            notes.setdefault((n["step"], n["to"]), []).append(n["text"])

        index = []
        for a in self.agents:
            aid = str(a["id"])
            months = []
            for step in range(1, len(self.monthly) + 1):
                pl = plans.get((step, aid), {})
                d = dec.get((step, aid))
                o = offers.get((step, aid))
                months.append({
                    "month": step,
                    "thought_at_plan": pl.get("thought", ""),
                    "went": pl.get("go_label", NOT_ASKED),
                    "notices": notes.get((step, aid), []),
                    "said": [{"text": u["text"], "at": u["venue_label"],
                              "to": u["talk_to"], "thought": u.get("thought", "")}
                             for u in said.get((step, aid), [])],
                    "heard": [{"from": g["from"], "route": g["route"],
                               "at": g["venue_label"], "text": g["text"]}
                              for g in got.get((step, aid), [])],
                    "offer": (o["text"] if o else ""),
                    "offer_parcel": (o["parcel"] if o else ""),
                    "offer_kind": (o["kind"] if o else ""),
                    "listings": (d.get("listings") if d else {}),
                    "listing_reasons": (d.get("listing_reasons", {}) if d else {}),
                    "sell": (d["sell"] if d else NOT_ASKED),
                    "sell_reason": (d.get("sell_reason", "") if d else ""),
                    "thought_at_decision": (d["thought"] if d else ""),
                })
            sold = [t for t in self.reg.transfers if t["agent_id"] == aid]
            doc = {"agent_id": aid, "name": a["name"], "role": a["role_label"],
                   "district": a.get("district", ""),
                   "resident_at_start": bool(a.get("resident", True)),
                   "sellable": bool(a.get("sellable", True)),
                   "owned_at_start": [p["name"] for p in self.parcels
                                      if p["land"] == aid or p.get("bld_owner") == aid],
                   "used_at_start": [p["name"] for p in self.parcels
                                     if p.get("tenant") == aid],
                   "neighbours": self.neighbour_names.get(aid, []),
                   "left_month": self.reg.left_month.get(aid),
                   "sold": [{"month": t["step"], "parcel": t["parcel"],
                             "kind": t["kind"]} for t in sold],
                   "months": months}
            with open(os.path.join(out_dir, aid + ".json"), "w",
                      encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
            index.append({"agent_id": aid, "name": a["name"],
                          "role": a["role_label"],
                          "resident_at_start": bool(a.get("resident", True)),
                          "sellable": bool(a.get("sellable", True)),
                          "left_month": self.reg.left_month.get(aid),
                          "sold_count": len(sold),
                          "file": "timeline_v9/" + aid + ".json"})
        with open(os.path.join(self.run_dir, "timeline_index.json"), "w",
                  encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    def _finalize(self, elapsed: float) -> Dict[str, Any]:
        usage = self.usage.as_dict()
        cost = self._cost_so_far()
        sellable_parcels = len([p for p in self.parcels
                                if self.reg.by_id[str(p["land"])].get("sellable", True)])
        in_town_offers = sum(1 for o in self.offers if o["to_in_town"])
        in_town_sold = sum(1 for o in self.offers
                           if o["to_in_town"] and o["result"] == RESULT_SOLD)
        abs_offers = len(self.offers) - in_town_offers
        abs_sold = sum(1 for o in self.offers
                       if not o["to_in_town"] and o["result"] == RESULT_SOLD)
        summary: Dict[str, Any] = {
            "run_name": self.cfg.get("run_name"),
            "scenario_version": "field_v9",
            "chat": self.chat,
            "scene_rounds": self.scene_rounds,
            "seed": self.seed,
            "steps": self.n_steps,
            "months_run": len(self.monthly),
            "stopped_by_cost": self.stopped_by_cost,
            "partial_month": self.partial_month,
            "max_cost_usd": self.max_cost_usd,
            "agents": len(self.agents),
            "residents_at_start": sum(1 for a in self.agents
                                      if a.get("resident", True)),
            "absentee_owners_at_start": sum(1 for a in self.agents
                                            if not a.get("resident", True)),
            "parcels_total": len(self.parcels),
            "sellable_parcels": sellable_parcels,
            "buildings_total": sum(1 for p in self.parcels if p["building"]),
            "land_only_parcels": sum(1 for p in self.parcels if not p["building"]),
            "acquired_parcels": len(self.reg.acquired_parcels()),
            "acquired_land": len(self.reg.acquired_land()),
            "acquired_buildings": len(self.reg.acquired_buildings()),
            "acquired_both": len(self.reg.acquired_both()),
            "transfers_total": len(self.reg.transfers),
            "sold_agents": len({t["agent_id"] for t in self.reg.transfers}),
            "left_agents": len(self.reg.left_ids()),
            "in_town_end": len(self.reg.in_town_ids()),
            "absentee_owners_end": len(self.reg.absentee_owner_ids()),
            "no_user_parcels_end": sum(1 for p in self.reg.parcel_names
                                       if self.reg.user_of(p) is None),
            "offers_total": len(self.offers),
            "offers_accepted": sum(1 for o in self.offers
                                   if o["result"] == RESULT_SOLD),
            "offers_declined": sum(1 for o in self.offers
                                   if o["result"] == RESULT_NOT_SOLD),
            "offers_no_answer": sum(1 for o in self.offers
                                    if o["result"] == RESULT_NO_ANSWER),
            "offers_to_in_town": in_town_offers,
            "offers_to_absentee": abs_offers,
            "sold_from_in_town": in_town_sold,
            "sold_from_absentee": abs_sold,
            "accept_rate_in_town": (round(in_town_sold / in_town_offers, 4)
                                    if in_town_offers else 0.0),
            "accept_rate_absentee": (round(abs_sold / abs_offers, 4)
                                     if abs_offers else 0.0),
            "offers_by_kind": {k: sum(1 for o in self.offers if o["kind"] == k)
                               for k in KIND_VALUES},
            "sold_by_kind": {k: sum(1 for t in self.reg.transfers
                                    if t["kind"] == k) for k in KIND_VALUES},
            "listings_total": len(self.listings),
            "listings_by_kind": {k: sum(1 for r in self.listings
                                        if r["kind"] == k) for k in KIND_VALUES},
            "listing_choice_counts": {
                v: sum(1 for d in self.decisions
                       for _p, x in d["listings"].items() if x == v)
                for v in LIST_VALUES + [NO_ANSWER]},
            "notices_total": len(self.notices),
            "utterances_total": len(self.utterances),
            "no_answer": self.no_answer,
            "plan_no_answer": self.plan_no_answer,
            "sell_no_answer": self.sell_no_answer,
            "invalid_listing": self.invalid_listing,
            "listing_missing": self.listing_missing,
            "invalid_sell": self.invalid_sell,
            "invalid_venue": self.invalid_venue,
            "truncated": self.truncated,
            "parse_fail": len(self.parse_fail),
            "acquirer_missing_targets": self.acquirer_missing_targets,
            "acquirer_dup_rows": self.acquirer_dup_rows,
            "acquirer_off_range": self.acquirer_off_range,
            "acquirer_chunk_fail": self.acquirer_chunk_fail,
            "acquirer_empty_text": self.acquirer_empty_text,
            "acquirer_invalid_offer": self.acquirer_invalid_offer,
            "acquirer_missing_parcel": self.acquirer_missing_parcel,
            "declines_delivered": self.declines_delivered,
            "reason_counts": dict(self.reason_counts),
            "common_prefix_tokens": self.common_prefix_tokens,
            "cached_ratio": (usage["cached_tokens"] / usage["input_tokens"]
                             if usage["input_tokens"] else 0.0),
            "checkpoints_written": self.checkpoints_written,
            "request_timeout_sec": getattr(self.client, "request_timeout_sec", None),
            "timeout_retries": getattr(self.client, "timeout_retries", 0),
            "timeout_giveups": getattr(self.client, "timeout_giveups", 0),
            "elapsed_sec": round(elapsed, 1),
            "usage": usage,
            "cost_usd": round(cost, 4),
        }
        for key, total, blank in (("listing", "listing_total", "listing_blank"),
                                  ("sell", "sell_total", "sell_blank"),
                                  ("acquirer", "acquirer_total", "acquirer_blank")):
            t = self.reason_counts[total]
            summary[f"reason_written_{key}"] = t - self.reason_counts[blank]
            summary[f"reason_rate_{key}"] = (
                round((t - self.reason_counts[blank]) / t, 4) if t else 0.0)

        os.makedirs(self.run_dir, exist_ok=True)

        def dump(name: str, obj: Any) -> None:
            with open(os.path.join(self.run_dir, name), "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)

        dump("summary.json", summary)
        dump("monthly.json", self.monthly)
        dump("offers.json", self.offers)
        dump("decisions.json", self.decisions)
        dump("plans.json", self.plans)
        dump("utterances.json", self.utterances)
        dump("deliveries.json", self.deliveries)
        dump("listings.json", self.listings)
        dump("notices.json", self.notices)
        dump("transfers.json", self.reg.transfers)
        dump("left_agents.json", self.left_agents)
        dump("neighbours.json", self.neighbours)
        dump("acquirer_raw.json", self.acquirer_raw)
        dump("ledger_by_step.json", self.ledger_by_step)
        dump("ledger_final.json", self.reg.ledger_rows())
        if self.parse_fail:
            dump("parse_fail.json", self.parse_fail)
        for name, text in (("common_prefix.txt", self.common_prefix),
                           ("absentee_prefix.txt", self.absentee_prefix),
                           ("acquirer_prefix.txt", self.acquirer_prefix)):
            with open(os.path.join(self.run_dir, name), "w", encoding="utf-8") as f:
                f.write(text)
        self._dump_timeline()
        return summary


__all__ = ["SimulationV9", "MockV9Client", "CostLimitReached",
           "RESULT_SOLD", "RESULT_NOT_SOLD", "RESULT_NO_ANSWER"]
