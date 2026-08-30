"""v8「最小の町」の月ループ。

設計の正は `docs/world_design_v8_minimal.md`。
**既存の `src/simulation.py` には一切触らない**（v8 はここに独立した月ループを持つ）。

このファイルの責務は3つだけ:
  1. 各主体に「今月あなたに見えているもの」を配る
  2. LLM の返した答えを **そのまま** 登記簿に記帳する（解釈・補正・代行をしない）
  3. 記録を残す

やっていないこと（意図的に）:
  - 誰かの行動を条件分岐で決める
  - 「〜なら売る」「〜%の確率で」といったパラメータ
  - 答えが無かったときに「売らない」を代わりに書き込むこと
    （答えが無かった月は "no_answer" として別に数える＝健全性ゲート）
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from .field_v8 import (ACQUIRER_NAME, DECIDE_KEEP, DECIDE_SELL, HOME,
                       MAX_OFFER_CHARS, RegistryV8, acquirer_schema_v8,
                       adjacency_v8, build_acquirer_prefix_v8,
                       build_acquirer_prompt_v8, build_common_prefix_v8,
                       build_decide_prompt_v8, build_plan_prompt_v8,
                       build_scene_prompt_v8, decide_order, decide_schema_v8,
                       load_personas_v8, plan_schema_v8, scene_schema_v8)
from .llm_client_factory import UsageMeter, create_llm_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 応答の読み取り（v5 の _parse_action と同じ考え方・補正はしない）
# ---------------------------------------------------------------------------

def repair_truncated_json(raw: str) -> Optional[Dict[str, Any]]:
    """閉じ括弧が足りないだけの JSON を閉じて読む（中身は足さない）。"""
    s = (raw or "").strip()
    if not s:
        return None
    depth_c = s.count("{") - s.count("}")
    depth_b = s.count("[") - s.count("]")
    if depth_c <= 0 and depth_b <= 0:
        return None
    fixed = s
    if fixed.count('"') % 2 == 1:
        fixed += '"'
    fixed += "]" * max(0, depth_b) + "}" * max(0, depth_c)
    try:
        return json.loads(fixed)
    except Exception:
        return None


def parse_json(raw: str) -> Tuple[Optional[Dict[str, Any]], bool]:
    """(dict, truncated) を返す。読めなければ (None, False)。"""
    s = (raw or "").strip()
    if not s:
        return None, False
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    try:
        return json.loads(s), False
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        try:
            return json.loads(m.group(0)), False
        except Exception:
            pass
    fixed = repair_truncated_json(s)
    if fixed is not None:
        return fixed, True
    return None, False


# ---------------------------------------------------------------------------
# v8 専用の mock（既存の MockClient には触らない）
# ---------------------------------------------------------------------------

class MockV8Client:
    """API を叩かずに配線と集計を通すためだけの client。

    **世界の挙動を作り込まない**：返す値はスキーマを満たす最小限で、
    2択は seed 固定の乱数で決める（実験の結果としては一切使わない）。
    """

    def __init__(self, seed: int = 42, usage: Optional[UsageMeter] = None,
                 sell_rate: float = 0.05):
        self.rnd = random.Random(seed)
        self.usage = usage or UsageMeter()
        self.sell_rate = sell_rate
        self.cache_created = 0
        self.cache_failed = 0
        self.max_token_finishes = 0
        self.batch_kinds = set()

    def close_caches(self) -> int:
        return 0

    def count_tokens(self, text: str) -> int:
        # 実測ではない目安（日本語はおよそ1文字≒1トークン弱）。
        return max(1, int(len(text) * 0.9))

    def generate(self, system_prompt: str, user_prompt: str,
                 schema: Optional[Dict[str, Any]] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, tag: str = "agent") -> str:
        self.usage.add(tag, input_tokens=len(user_prompt) // 3,
                       output_tokens=40)
        props = (schema or {}).get("properties", {})
        if "offers" in props:
            enum = props["offers"]["items"]["properties"]["to"]["enum"]
            offers = [{"to": n, "send": True,
                       "text": f"{n}のご都合に合わせます。住み続けていただいて構いません。"}
                      for n in enum]
            return json.dumps({"offers": offers}, ensure_ascii=False)
        if "go" in props:
            choices = props["go"]["enum"]
            return json.dumps({"thought": "今月のこと。",
                               "go": self.rnd.choice(choices)}, ensure_ascii=False)
        if "decision" in props:
            d = DECIDE_SELL if self.rnd.random() < self.sell_rate else DECIDE_KEEP
            return json.dumps({"thought": "決めた。", "decision": d}, ensure_ascii=False)
        if "text" in props:
            others = props["talk_to"]["items"]["enum"]
            return json.dumps({"thought": "話すか。", "text": "最近どうですか。",
                               "talk_to": [self.rnd.choice(others)] if others else []},
                              ensure_ascii=False)
        return "{}"

    def generate_many(self, items, tag: str = "agent", kind: str = "agents",
                      job_key: Optional[str] = None):
        return [self.generate(it["system_prompt"], it["user_prompt"],
                              schema=it.get("schema"),
                              max_tokens=it.get("max_tokens"),
                              tag=it.get("tag") or tag) for it in items]


# ---------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------

class SimulationV8:

    def __init__(self, cfg: Dict[str, Any], run_dir: str):
        if cfg.get("scenario_version") != "field_v8":
            raise ValueError("config の scenario_version が field_v8 ではない")
        self.cfg = cfg
        self.run_dir = run_dir
        self.n_steps = int(cfg["steps"])
        self.chat = bool(cfg.get("chat", True))
        self.scene_rounds = int(cfg.get("scene_rounds", 1))
        self.chunk = int(cfg.get("acquirer_chunk", 10))
        self.seed = int(cfg.get("seed", 42))
        self.workers = int(cfg["llm"].get("parallel_workers", 8))

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pf = str(cfg["personas_file"])
        if not os.path.isabs(pf):
            pf = os.path.join(root, pf)
        self.agents = load_personas_v8(pf)
        self.reg = RegistryV8(self.agents)

        venues = cfg.get("social", {}).get("venues", [])
        self.venue_labels = {str(v["id"]): str(v["label"]) for v in venues}
        if len(set(self.venue_labels.values())) != len(self.venue_labels):
            raise ValueError("場所の名前が重複している")
        # 場所は4つだけで、**全員がどこへでも行ける**（施主決定 2026-08-29 21:47）。
        self.venue_order = [str(v["id"]) for v in venues]
        self.all_labels = [self.venue_labels[v] for v in self.venue_order]
        self.label_to_venue = {lb: vid for vid, lb in self.venue_labels.items()}
        # 隣近所（不動産が隣り合う持ち主どうし）。行き先に関わらず毎月ひと言が届く。
        self.neighbours = adjacency_v8(self.agents)
        self.neighbour_names = {aid: [self.reg.name_of[x] for x in nbs]
                                for aid, nbs in self.neighbours.items()}

        self.usage = UsageMeter()
        if str(cfg["llm"].get("provider", "mock")).lower() == "mock":
            self.client = MockV8Client(seed=self.seed, usage=self.usage)
        else:
            self.client = create_llm_client({**cfg["llm"], "seed": self.seed},
                                            self.usage)
        self.max_tokens = int(cfg["llm"].get("max_tokens", 2200))
        # 手元集計の費用がこれを超えたら、その月の終わりで走行を止める（施主の絶対上限）。
        # 0 以下なら無制限（mock 用）。Codex 走行前レビュー2巡目の指摘で入れた安全弁。
        self.max_cost_usd = float(cfg.get("max_cost_usd", 0) or 0)
        self.stopped_by_cost = False

        # 全コール共通の前置き（＝キャッシュに載る唯一の塊）
        self.common_prefix = build_common_prefix_v8(cfg, self.agents)
        # X社は別の前置き（公の情報だけ。会場・会話の仕組み・「記録を見ていない」を渡さない
        # ＝Codex 走行前レビュー 2026-08-29 の指摘）。
        self.acquirer_prefix = build_acquirer_prefix_v8(cfg, self.agents)
        self.common_prefix_tokens: Optional[int] = None

        # 主体ごとの持ち越し内心
        self.thought: Dict[str, str] = {str(a["id"]): "" for a in self.agents}
        # 記録
        self.offers: List[Dict[str, Any]] = []        # X社が出した提示（原文）
        self.plans: List[Dict[str, Any]] = []         # 月ごとの行き先
        self.utterances: List[Dict[str, Any]] = []    # 発話（原文）
        self.decisions: List[Dict[str, Any]] = []     # 月末の2択（原文の内心つき）
        self.monthly: List[Dict[str, Any]] = []       # 月別の集計
        self.deliveries: List[Dict[str, Any]] = []    # 誰に何がどの経路で届いたか
        self.parse_fail: List[Dict[str, Any]] = []
        self.no_answer = 0
        self.truncated = 0
        self.invalid_venue = 0

    # -- LLM ---------------------------------------------------------------

    def _call(self, items: List[Tuple[str, str, Dict[str, Any], str]],
              tag: str) -> Dict[str, str]:
        """items = [(key, user_prompt, schema, tag_suffix)] → {key: raw}"""
        if not items:
            return {}

        def one(it):
            key, up, schema, sfx = it
            system = (self.acquirer_prefix if sfx == ":acquirer"
                      else self.common_prefix)
            raw = self.client.generate(system, up, schema=schema,
                                       max_tokens=self.max_tokens,
                                       tag=f"v8:{tag}{sfx}")
            return key, raw

        out: Dict[str, str] = {}
        workers = max(1, min(self.workers, len(items)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for key, raw in ex.map(one, items):
                out[key] = raw
        return out

    def _read(self, key: str, raw: str, step: int, tag: str) -> Optional[Dict[str, Any]]:
        act, truncated = parse_json(raw)
        if truncated:
            self.truncated += 1
        if act is None:
            self.parse_fail.append({"step": step, "who": key, "tag": tag,
                                    "raw": (raw or "")[:2000]})
            return None
        return act

    # -- 月 ------------------------------------------------------------------

    def _acquirer_turn(self, step: int) -> Dict[str, str]:
        """X社の月次。未売却の持ち主ごとに提示を出すか決める。"""
        risk = self.reg.risk_set()
        if not risk:
            return {}
        names = [self.reg.name_of[aid] for aid in risk]
        chunks = [names[i:i + self.chunk] for i in range(0, len(names), self.chunk)]
        history = [{"step": o["step"], "to": o["to"], "text": o["text"],
                    "result": o["result"]} for o in self.offers]
        items = []
        for n, targets in enumerate(chunks, 1):
            up = build_acquirer_prompt_v8(self.reg, step, self.n_steps, targets,
                                          history, n, len(chunks))
            items.append((f"X{n}", up, acquirer_schema_v8(targets), ":acquirer"))
        raws = self._call(items, "acquirer")

        sent: Dict[str, str] = {}
        for n, targets in enumerate(chunks, 1):
            act = self._read(f"X{n}", raws.get(f"X{n}", ""), step, "acquirer")
            if act is None:
                continue
            for row in (act.get("offers") or []):
                to = str(row.get("to", "") or "")
                if to not in targets:
                    continue
                text = str(row.get("text", "") or "").strip()
                if not bool(row.get("send")) or not text:
                    continue
                sent[self.reg.id_of_name[to]] = text[:MAX_OFFER_CHARS]
        for aid, text in sent.items():
            # result は「応じた／応じなかった／答えが返らなかった」の3状態。
            # 欠損を「応じなかった」に丸めない（Codex 走行前レビューの指摘）。
            self.offers.append({"step": step, "to": self.reg.name_of[aid],
                                "to_id": aid, "text": text,
                                "result": "応じなかった", "accepted": False})
        return sent

    def _plan_turn(self, step: int, offers: Dict[str, str]) -> Dict[str, str]:
        items = []
        for a in self.agents:
            aid = str(a["id"])
            up = build_plan_prompt_v8(a, self.reg, step, self.n_steps,
                                      self.all_labels, self.thought[aid],
                                      offers.get(aid),
                                      neighbours=self.neighbour_names.get(aid))
            items.append((aid, up, plan_schema_v8(self.all_labels), ":plan"))
        raws = self._call(items, "plan")

        go: Dict[str, str] = {}
        for a in self.agents:
            aid = str(a["id"])
            act = self._read(aid, raws.get(aid, ""), step, "plan")
            where = HOME
            if act is not None:
                self.thought[aid] = str(act.get("thought", "") or "")[:600]
                value = str(act.get("go", "") or "").strip()
                if value in self.label_to_venue:
                    where = self.label_to_venue[value]
                elif value and value != HOME:
                    self.invalid_venue += 1
            go[aid] = where
            self.plans.append({"step": step, "agent_id": aid,
                               "name": a["name"], "go": where,
                               "go_label": self.venue_labels.get(where, HOME),
                               "thought": self.thought[aid]})
        return go

    def _scene_turn(self, step: int, go: Dict[str, str],
                    offers: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
        """月1回の集まり。同じ会場に2人以上いれば会話が成立する。"""
        heard: Dict[str, List[Dict[str, Any]]] = {str(a["id"]): [] for a in self.agents}
        if not self.chat:
            return heard
        groups: Dict[str, List[str]] = {}
        for aid, venue in go.items():
            if venue == HOME:
                continue
            groups.setdefault(venue, []).append(aid)
        groups = {v: sorted(m) for v, m in groups.items() if len(m) >= 2}
        if not groups:
            return heard

        for rnd in range(1, self.scene_rounds + 1):
            items = []
            ctx: Dict[str, Tuple[str, List[str]]] = {}
            for venue, members in sorted(groups.items()):
                present = [self.reg.name_of[m] for m in members]
                for aid in members:
                    a = self.reg.by_id[aid]
                    up = build_scene_prompt_v8(a, self.reg, step, self.n_steps,
                                               self.thought[aid],
                                               self.venue_labels[venue], present)
                    items.append((aid, up, scene_schema_v8(present, a["name"]),
                                  ":scene"))
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

            # 配送は2経路（施主決定 2026-08-29 21:53）。重複は1回に畳む。
            #   ① 居合わせ＝その月に同じ場所へ行った全員
            #   ② 隣近所＝不動産が隣り合う持ち主（行き先に関わらず毎月届く）
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
                    # 経路によって「どこで言われたか」が見えるかどうかが変わる
                    # （隣近所には場所を渡さない・Codex 走行前レビュー2巡目）
                    heard[mid].append({**row, "route": route})
                    self.deliveries.append(
                        {"step": step, "to": mid, "to_name": self.reg.name_of[mid],
                         "from": row["from"], "from_id": row["from_id"],
                         "route": route, "venue_label": row["venue_label"],
                         "text": row["text"]})
        return heard

    def _decide_turn(self, step: int, offers: Dict[str, str],
                     heard: Dict[str, List[Dict[str, Any]]]
                     ) -> Tuple[List[str], List[str]]:
        """月末の2択。未売却の持ち主全員に等しく問う（集まりに行ったかは関係ない）。"""
        risk = self.reg.risk_set()
        items = []
        orders: Dict[str, List[str]] = {}
        for aid in risk:
            a = self.reg.by_id[aid]
            order = decide_order(int(a["index"]), step)
            orders[aid] = order
            up = build_decide_prompt_v8(a, self.reg, step, self.n_steps,
                                        self.thought[aid], offers.get(aid),
                                        heard.get(aid, []), order=order,
                                        neighbours=self.neighbour_names.get(aid))
            items.append((aid, up, decide_schema_v8(order), ":decide"))
        raws = self._call(items, "decide")

        sellers: List[str] = []
        blanks: List[str] = []
        for aid in risk:
            a = self.reg.by_id[aid]
            act = self._read(aid, raws.get(aid, ""), step, "decide")
            if act is None:
                # **「売らない」で埋めない**。答えが無かったという事実を残す。
                self.no_answer += 1
                blanks.append(aid)
                self.decisions.append({"step": step, "agent_id": aid,
                                       "name": a["name"], "decision": "no_answer",
                                       "thought": "", "offer": offers.get(aid, ""),
                                       "order": orders[aid],
                                       "heard": len(heard.get(aid, []))})
                continue
            thought = str(act.get("thought", "") or "")
            self.thought[aid] = thought[:600]
            decision = str(act.get("decision", "") or "").strip()
            if decision not in (DECIDE_SELL, DECIDE_KEEP):
                self.no_answer += 1
                blanks.append(aid)
                decision = "no_answer"
            self.decisions.append({"step": step, "agent_id": aid, "name": a["name"],
                                   "decision": decision, "thought": thought,
                                   "offer": offers.get(aid, ""),
                                   "order": orders[aid],
                                   "heard": len(heard.get(aid, []))})
            if decision == DECIDE_SELL:
                sellers.append(aid)
        return sellers, blanks

    def _step(self, step: int) -> None:
        offers = self._acquirer_turn(step)
        go = self._plan_turn(step, offers)
        heard = self._scene_turn(step, go, offers)
        sellers, no_answer_ids = self._decide_turn(step, offers, heard)

        moved: List[str] = []
        for aid in sellers:
            moved += self.reg.apply_sale(aid, step)
        # 提示の結果を X社 の履歴に反映する（3状態・欠損を「応じなかった」に丸めない）
        for o in self.offers:
            if o["step"] != step:
                continue
            if o["to_id"] in sellers:
                o["result"], o["accepted"] = "応じた", True
            elif o["to_id"] in no_answer_ids:
                o["result"], o["accepted"] = "答えが返らなかった", False

        risk = self.reg.risk_set()
        # 場所別の来訪人数（分布・施主指示 2026-08-29 21:55）
        by_venue = {self.venue_labels[v]: sum(1 for w in go.values() if w == v)
                    for v in self.venue_order}
        by_venue[HOME] = sum(1 for w in go.values() if w == HOME)
        heard_counts = [len(heard.get(str(a["id"]), [])) for a in self.agents]
        self.monthly.append({
            "step": step,
            "offers_sent": len(offers),
            "sold_this_month": len(sellers),
            "parcels_this_month": len(moved),
            "sold_cum": len(self.reg.sold_ids()),
            "parcels_cum": len(self.reg.acquired_parcels()),
            "risk_set": len(risk),
            "attended": sum(1 for v in go.values() if v != HOME),
            "by_venue": by_venue,
            "utterances": sum(1 for u in self.utterances if u["step"] == step),
            "heard_mean": round(sum(heard_counts) / len(heard_counts), 2),
            "heard_max": max(heard_counts) if heard_counts else 0,
            "heard_min": min(heard_counts) if heard_counts else 0,
        })

    # -- run -----------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        t0 = time.time()
        try:
            self.common_prefix_tokens = int(
                self.client.count_tokens(self.common_prefix))
        except Exception as e:  # noqa: BLE001
            logger.warning("共通前置きのトークン実測に失敗: %s", e)
        for step in range(1, self.n_steps + 1):
            self._step(step)
            m = self.monthly[-1]
            logger.info("m%02d 提示%d 売却%d 累計%d区画 残り%d人 $%.4f",
                        step, m["offers_sent"], m["sold_this_month"],
                        m["parcels_cum"], m["risk_set"], self._cost_so_far())
            if self.max_cost_usd > 0 and self._cost_so_far() >= self.max_cost_usd:
                self.stopped_by_cost = True
                logger.error("費用上限 $%.4f に達したので第%d月で止める",
                             self.max_cost_usd, step)
                break
        elapsed = time.time() - t0
        try:
            self.client.close_caches()
        except Exception as e:  # noqa: BLE001
            logger.warning("キャッシュ削除に失敗（TTLで消える）: %s", e)
        return self._finalize(elapsed)

    def _dump_timeline(self) -> None:
        """主体ごとの月順タイムライン（施主指示 2026-08-29 22:11）。

        1人1ファイル＝`timeline_v8/<id>.json`、一覧は `timeline_index.json`。
        観測されたものをそのまま月順に並べるだけで、判定・分類はしない。
        """
        out_dir = os.path.join(self.run_dir, "timeline_v8")
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

        index = []
        for a in self.agents:
            aid = str(a["id"])
            months = []
            for step in range(1, self.n_steps + 1):
                pl = plans.get((step, aid), {})
                d = dec.get((step, aid))
                o = offers.get((step, aid))
                months.append({
                    "month": step,
                    "thought_at_plan": pl.get("thought", ""),
                    "went": pl.get("go_label", HOME),
                    "said": [{"text": u["text"], "at": u["venue_label"],
                              "to": u["talk_to"], "thought": u.get("thought", "")}
                             for u in said.get((step, aid), [])],
                    "heard": [{"from": g["from"], "route": g["route"],
                               "at": g["venue_label"], "text": g["text"]}
                              for g in got.get((step, aid), [])],
                    "offer": (o["text"] if o else ""),
                    "decision": (d["decision"] if d else "問われていない"),
                    "thought_at_decision": (d["thought"] if d else ""),
                })
            doc = {
                "agent_id": aid, "name": a["name"], "role": a["role_label"],
                "district": a.get("district", ""),
                "holdings": list(a["holdings"]),
                "sellable": bool(a.get("sellable", True)),
                "neighbours": self.neighbour_names.get(aid, []),
                "sold_month": self.reg.sold_month[aid],
                "months": months,
            }
            with open(os.path.join(out_dir, aid + ".json"), "w",
                      encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
            index.append({"agent_id": aid, "name": a["name"],
                          "role": a["role_label"],
                          "sellable": bool(a.get("sellable", True)),
                          "sold_month": self.reg.sold_month[aid],
                          "file": "timeline_v8/" + aid + ".json"})
        with open(os.path.join(self.run_dir, "timeline_index.json"), "w",
                  encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    def _cost_so_far(self) -> float:
        """手元集計の費用（走行中に何度でも呼べる）。"""
        usage = self.usage.as_dict()
        price = (self.cfg.get("cost", {}).get("price_table", {})
                 .get(self.cfg["llm"].get("model", ""), {}))
        billed_in = max(0, usage["input_tokens"] - usage["cached_tokens"])
        return (billed_in / 1e6 * float(price.get("input", 0.0))
                + usage["cached_tokens"] / 1e6 * float(price.get("cache_read", 0.0))
                + usage["output_tokens"] / 1e6 * float(price.get("output", 0.0)))

    def _finalize(self, elapsed: float) -> Dict[str, Any]:
        usage = self.usage.as_dict()
        price = (self.cfg.get("cost", {}).get("price_table", {})
                 .get(self.cfg["llm"].get("model", ""), {}))
        billed_in = max(0, usage["input_tokens"] - usage["cached_tokens"])
        cost = (billed_in / 1e6 * float(price.get("input", 0.0))
                + usage["cached_tokens"] / 1e6 * float(price.get("cache_read", 0.0))
                + usage["output_tokens"] / 1e6 * float(price.get("output", 0.0)))
        n_sellable = len(self.reg.sellable_ids)
        summary = {
            "run_name": self.cfg.get("run_name"),
            "scenario_version": "field_v8",
            "chat": self.chat,
            "scene_rounds": self.scene_rounds,
            "seed": self.seed,
            "steps": self.n_steps,
            "months_run": len(self.monthly),
            "stopped_by_cost": self.stopped_by_cost,
            "max_cost_usd": self.max_cost_usd,
            "agents": len(self.agents),
            "sellable_agents": n_sellable,
            "parcels_total": len(self.reg.owner_of),
            "sellable_parcels": sum(len(a["holdings"]) for a in self.agents
                                    if a.get("sellable", True)),
            "acquired_parcels": len(self.reg.acquired_parcels()),
            "sold_agents": len(self.reg.sold_ids()),
            "final_unsold_ratio": (len(self.reg.risk_set()) / n_sellable
                                   if n_sellable else 0.0),
            "offers_total": len(self.offers),
            "offers_accepted": sum(1 for o in self.offers if o["result"] == "応じた"),
            "offers_declined": sum(1 for o in self.offers
                                   if o["result"] == "応じなかった"),
            "offers_no_answer": sum(1 for o in self.offers
                                    if o["result"] == "答えが返らなかった"),
            "heard_per_person_month": (
                round(sum(m["heard_mean"] for m in self.monthly) / len(self.monthly), 2)
                if self.monthly else 0.0),
            "attended_mean": (
                round(sum(m["attended"] for m in self.monthly) / len(self.monthly), 2)
                if self.monthly else 0.0),
            "no_answer": self.no_answer,
            "truncated": self.truncated,
            "invalid_venue": self.invalid_venue,
            "parse_fail": len(self.parse_fail),
            "common_prefix_tokens": self.common_prefix_tokens,
            "cache_created": getattr(self.client, "cache_created", 0),
            "cache_failed": getattr(self.client, "cache_failed", 0),
            "cached_ratio": (usage["cached_tokens"] / usage["input_tokens"]
                             if usage["input_tokens"] else 0.0),
            "elapsed_sec": round(elapsed, 1),
            "usage": usage,
            "cost_usd": round(cost, 4),
        }
        os.makedirs(self.run_dir, exist_ok=True)

        def dump(name: str, obj: Any) -> None:
            with open(os.path.join(self.run_dir, name), "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)

        dump("summary.json", summary)
        dump("monthly.json", self.monthly)
        dump("offers.json", self.offers)
        dump("decisions.json", self.decisions)
        dump("utterances.json", self.utterances)
        dump("plans.json", self.plans)
        dump("transfers.json", self.reg.transfers)
        dump("neighbours.json", self.neighbours)
        dump("deliveries.json", self.deliveries)
        self._dump_timeline()
        with open(os.path.join(self.run_dir, "acquirer_prefix.txt"), "w",
                  encoding="utf-8") as f:
            f.write(self.acquirer_prefix)
        if self.parse_fail:
            dump("parse_fail.json", self.parse_fail)
        with open(os.path.join(self.run_dir, "common_prefix.txt"), "w",
                  encoding="utf-8") as f:
            f.write(self.common_prefix)
        return summary


__all__ = ["SimulationV8", "MockV8Client", "parse_json", "repair_truncated_json"]
