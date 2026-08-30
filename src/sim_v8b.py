"""v8b「売りに出す町」の月ループ（v8 からの差分だけ）。

設計の正は `docs/world_design_v8b.md`。
**既存の `src/simulation.py`・`src/sim_v8.py` には一切触らない**
（`SimulationV8` を **継承** して、X社ターンと月末の問いだけ差し替える）。

このファイルの責務は v8 と同じ3つだけ:
  1. 各主体に「今月あなたに見えているもの」を配る
  2. LLM の返した答えを **そのまま** 登記簿に記帳する（解釈・補正・代行をしない）
  3. 記録を残す

やっていないこと（意図的に）:
  - 誰かの行動を条件分岐で決める
  - 「〜なら売る」「〜%の確率で」といったパラメータ
  - 答えが無かったときに「出さない」「応じない」を代わりに書き込むこと
    （答えが無かった月は "no_answer" として別に数える＝健全性ゲート）
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from .field_v8b import (ACQUIRER_NAME, HOME, LIST_NO, LIST_YES, MAX_OFFER_CHARS,
                        NO_ANSWER, NOT_ASKED, RESP_NO, RESP_YES,
                        acquirer_schema_v8b, build_acquirer_prefix_v8b,
                        build_acquirer_prompt_v8b, build_common_prefix_v8b,
                        build_decide_prompt_v8b, build_plan_prompt_v8b,
                        build_scene_prompt_v8b, decide_schema_v8b,
                        listing_order, plan_schema_v8, respond_order,
                        scene_schema_v8)
from .sim_v8 import MockV8Client, SimulationV8

logger = logging.getLogger(__name__)


class CostLimitReached(Exception):
    """手元集計の費用が上限に達したので、その場で走行をやめる合図。

    v8 は月末にしか費用を見ていなかったので、1か月ぶんまるごと超過しうる
    （Codex 走行前レビュー2巡目の指摘＝唯一の走行NG項目）。v8b は**場面と場面の
    あいだ**でも見て、超えたらその月を捨てて止まる。
    途中で捨てた月は `monthly` に入らない＝`months_run` に数えない。
    """


class MockV8BClient(MockV8Client):
    """v8b 用の mock（配線と集計を通すためだけのもの）。

    **世界の挙動を作り込まない**：返す値はスキーマを満たす最小限で、
    2つの問いは seed 固定の乱数で決める（実験の結果としては一切使わない）。
    """

    def __init__(self, seed: int = 42, usage=None, list_rate: float = 0.15,
                 respond_rate: float = 0.10, send_rate: float = 0.6):
        super().__init__(seed=seed, usage=usage)
        self.list_rate = list_rate
        self.respond_rate = respond_rate
        self.send_rate = send_rate

    def generate(self, system_prompt: str, user_prompt: str,
                 schema: Optional[Dict[str, Any]] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, tag: str = "agent") -> str:
        self.usage.add(tag, input_tokens=len(user_prompt) // 3, output_tokens=40)
        props = (schema or {}).get("properties", {})
        if "offers" in props:
            enum = props["offers"]["items"]["properties"]["to"]["enum"]
            offers = []
            for n in enum:
                send = self.rnd.random() < self.send_rate
                offers.append({"to": n, "send": send,
                               "text": (f"{n}のご都合に合わせて手続きを進めます。"
                                        if send else "")})
            return json.dumps({"offers": offers}, ensure_ascii=False)
        if "go" in props:
            choices = props["go"]["enum"]
            return json.dumps({"thought": "今月のこと。",
                               "go": self.rnd.choice(choices)}, ensure_ascii=False)
        if "listing" in props:
            out: Dict[str, Any] = {
                "thought": "決めた。",
                "listing": (LIST_YES if self.rnd.random() < self.list_rate
                            else LIST_NO),
            }
            if "respond" in props:
                out["respond"] = (RESP_YES if self.rnd.random() < self.respond_rate
                                  else RESP_NO)
            return json.dumps(out, ensure_ascii=False)
        if "text" in props:
            others = props["talk_to"]["items"]["enum"]
            return json.dumps({"thought": "話すか。", "text": "最近どうですか。",
                               "talk_to": [self.rnd.choice(others)] if others else []},
                              ensure_ascii=False)
        return "{}"


class SimulationV8B(SimulationV8):

    def __init__(self, cfg: Dict[str, Any], run_dir: str):
        if cfg.get("scenario_version") != "field_v8b":
            raise ValueError("config の scenario_version が field_v8b ではない")
        # 親は field_v8 を要求するので、版名だけ差し替えて渡し、あとで戻す。
        super().__init__({**cfg, "scenario_version": "field_v8"}, run_dir)
        self.cfg = cfg
        # 前置きは v8b のもので置き換える（住民側にX社の名前も命題も出ない）
        self.common_prefix = build_common_prefix_v8b(cfg, self.agents)
        self.acquirer_prefix = build_acquirer_prefix_v8b(cfg, self.agents)
        if str(cfg["llm"].get("provider", "mock")).lower() == "mock":
            self.client = MockV8BClient(seed=self.seed, usage=self.usage)
        # v8b の記録
        self.listings: List[Dict[str, Any]] = []          # 出品（人×月）
        self.listed_by_step: Dict[int, List[str]] = {}    # 月 → 出した人のid
        self.respond_no_answer = 0
        self.invalid_listing = 0
        self.invalid_respond = 0
        # X社の応答の取りこぼしを黙って「提示なし」に丸めないための健全性カウンタ
        # （Codex 走行前レビュー 2026-08-29 の指摘）。
        self.acquirer_missing_targets = 0   # 応答に現れなかった相手（判断が返っていない）
        self.acquirer_dup_rows = 0          # 同じ相手が2回以上現れた
        self.acquirer_off_range = 0         # 今回の対象でない相手が現れた
        self.acquirer_chunk_fail = 0        # 塊まるごと読めなかった回数
        self.partial_month: Optional[int] = None   # 費用で途中で捨てた月
        self.acquirer_empty_text = 0        # send=true なのに条件文が空だった
        self.acquirer_raw: List[Dict[str, Any]] = []   # X社の応答の原文（観測用）

    # -- 費用の歯止め ----------------------------------------------------------

    def _guard_cost(self) -> None:
        if self.max_cost_usd > 0 and self._cost_so_far() >= self.max_cost_usd:
            raise CostLimitReached()

    # -- 月 ------------------------------------------------------------------

    def _acquirer_turn(self, step: int) -> Dict[str, str]:
        """X社の月次。登記簿＋**前月末の出品一覧**＋畳んだ履歴を見て判断する。"""
        risk = self.reg.risk_set()
        if not risk:
            return {}
        names = [self.reg.name_of[aid] for aid in risk]
        chunks = [names[i:i + self.chunk] for i in range(0, len(names), self.chunk)]
        listed_names = [self.reg.name_of[aid]
                        for aid in self.listed_by_step.get(step - 1, [])]
        items = []
        for n, targets in enumerate(chunks, 1):
            up = build_acquirer_prompt_v8b(self.reg, step, self.n_steps, targets,
                                           self.offers, listed_names, n, len(chunks))
            items.append((f"X{n}", up, acquirer_schema_v8b(targets), ":acquirer"))
        raws = self._call(items, "acquirer")
        for k in sorted(raws):
            self.acquirer_raw.append({"step": step, "chunk": k,
                                      "raw": (raws[k] or "")[:8000]})

        sent: Dict[str, str] = {}
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
                text = str(row.get("text", "") or "").strip()
                if bool(row.get("send")) and not text:
                    self.acquirer_empty_text += 1
                if not bool(row.get("send")) or not text:
                    continue
                sent[self.reg.id_of_name[to]] = text[:MAX_OFFER_CHARS]
        # 応答に現れなかった相手は「出さないと決めた」のではなく「判断が返らなかった」。
        # 数えて健全性ゲートに出す（提示なしと同じ扱いで進めるが、事実は残す）。
        self.acquirer_missing_targets += len([x for x in names if x not in seen])
        for aid, text in sent.items():
            # result は「応じた／応じなかった／答えが返らなかった」の3状態。
            # 欠損を「応じなかった」に丸めない。
            self.offers.append({"step": step, "to": self.reg.name_of[aid],
                                "to_id": aid, "text": text,
                                "result": "応じなかった", "accepted": False})
        return sent

    def _plan_turn(self, step: int, offers: Dict[str, str]) -> Dict[str, str]:
        """月初の思考と行き先（文面は v8b のビルダを使う＝中身は v8 と同じ）。"""
        items = []
        for a in self.agents:
            aid = str(a["id"])
            up = build_plan_prompt_v8b(a, self.reg, step, self.n_steps,
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

    def _scene_prompt(self, a, step, venue_label, present):
        return build_scene_prompt_v8b(a, self.reg, step, self.n_steps,
                                      self.thought[str(a["id"])], venue_label,
                                      present)

    def _scene_turn(self, step: int, go: Dict[str, str],
                    offers: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
        """月1回の集まり（v8 と同一の仕組み。文面だけ v8b のビルダを使う）。"""
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
                    up = self._scene_prompt(a, step, self.venue_labels[venue],
                                            present)
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
                    heard[mid].append({**row, "route": route})
                    self.deliveries.append(
                        {"step": step, "to": mid, "to_name": self.reg.name_of[mid],
                         "from": row["from"], "from_id": row["from_id"],
                         "route": route, "venue_label": row["venue_label"],
                         "text": row["text"]})
        return heard

    def _decide_turn(self, step: int, offers: Dict[str, str],
                     heard: Dict[str, List[Dict[str, Any]]]
                     ) -> Tuple[List[str], List[str], List[str]]:
        """月末の問い。

        ①「今月、自分の不動産を売りに出すか」＝未売却の持ち主**全員**に聞く
          （X社の名前は出ない）。
        ②「X社の条件に応じるか」＝**その月に条件が届いた人だけ**に、同じコールで聞く。

        返り値 = (応じた人, 応じるを答えなかった人, 出品した人)
        """
        risk = self.reg.risk_set()
        prev = set(self.listed_by_step.get(step - 1, []))
        items = []
        lorders: Dict[str, List[str]] = {}
        rorders: Dict[str, List[str]] = {}
        for aid in risk:
            a = self.reg.by_id[aid]
            offer = offers.get(aid)
            lo = listing_order(int(a["index"]), step)
            ro = respond_order(int(a["index"]), step) if offer else None
            lorders[aid] = lo
            if ro:
                rorders[aid] = ro
            up = build_decide_prompt_v8b(a, self.reg, step, self.n_steps,
                                         self.thought[aid], offer,
                                         heard.get(aid, []), list_order=lo,
                                         resp_order=ro,
                                         neighbours=self.neighbour_names.get(aid))
            items.append((aid, up, decide_schema_v8b(lo, ro), ":decide"))
        raws = self._call(items, "decide")

        accepters: List[str] = []
        respond_blanks: List[str] = []
        listers: List[str] = []
        for aid in risk:
            a = self.reg.by_id[aid]
            offer = offers.get(aid, "")
            act = self._read(aid, raws.get(aid, ""), step, "decide")
            rec = {"step": step, "agent_id": aid, "name": a["name"],
                   "listing": NO_ANSWER,
                   "respond": (NO_ANSWER if offer else NOT_ASKED),
                   "thought": "", "offer": offer,
                   "listing_order": lorders[aid],
                   "respond_order": rorders.get(aid, []),
                   "listed_last_month": aid in prev,
                   "heard": len(heard.get(aid, []))}
            if act is None:
                # **「出さない」「応じない」で埋めない**。答えが無かった事実を残す。
                self.no_answer += 1
                if offer:
                    self.respond_no_answer += 1
                    respond_blanks.append(aid)
                self.decisions.append(rec)
                continue
            thought = str(act.get("thought", "") or "")
            self.thought[aid] = thought[:600]
            rec["thought"] = thought

            listing = str(act.get("listing", "") or "").strip()
            if listing in (LIST_YES, LIST_NO):
                rec["listing"] = listing
                if listing == LIST_YES:
                    listers.append(aid)
            else:
                self.no_answer += 1
                self.invalid_listing += 1

            if offer:
                respond = str(act.get("respond", "") or "").strip()
                if respond in (RESP_YES, RESP_NO):
                    rec["respond"] = respond
                    if respond == RESP_YES:
                        accepters.append(aid)
                else:
                    self.respond_no_answer += 1
                    self.invalid_respond += 1
                    respond_blanks.append(aid)
            self.decisions.append(rec)
        return accepters, respond_blanks, listers

    def _step(self, step: int) -> None:
        offers = self._acquirer_turn(step)
        self._guard_cost()
        go = self._plan_turn(step, offers)
        self._guard_cost()
        heard = self._scene_turn(step, go, offers)
        self._guard_cost()
        accepters, respond_blanks, listers = self._decide_turn(step, offers, heard)

        # 名義が動くのは②で応じたときだけ（出品では動かない）
        moved: List[str] = []
        for aid in accepters:
            moved += self.reg.apply_sale(aid, step)
        for o in self.offers:
            if o["step"] != step:
                continue
            if o["to_id"] in accepters:
                o["result"], o["accepted"] = "応じた", True
            elif o["to_id"] in respond_blanks:
                o["result"], o["accepted"] = "答えが返らなかった", False

        # 出品の記録（その月かぎり・翌月にX社が見る）。
        # 同じ月に「出す」と答え、かつ提示に「応じる」と答えた人は、その月末に名義が
        # X社へ移っている＝翌月の公の出品一覧には残らない（Codex 走行前レビューの指摘）。
        # 出品したという事実は listings に残す（sold_same_month で印を付ける）。
        acc = set(accepters)
        self.listed_by_step[step] = [aid for aid in listers if aid not in acc]
        for aid in listers:
            self.listings.append({"step": step, "agent_id": aid,
                                  "name": self.reg.name_of[aid],
                                  "sold_same_month": aid in acc})

        risk = self.reg.risk_set()
        by_venue = {self.venue_labels[v]: sum(1 for w in go.values() if w == v)
                    for v in self.venue_order}
        by_venue[HOME] = sum(1 for w in go.values() if w == HOME)
        heard_counts = [len(heard.get(str(a["id"]), [])) for a in self.agents]
        self.monthly.append({
            "step": step,
            "offers_sent": len(offers),
            "listed_this_month": len(listers),
            "accepted_this_month": len(accepters),
            "sold_this_month": len(accepters),
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

    # -- run -------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """v8 の run と同じだが、**月の途中でも費用の上限で止まる**。

        最悪でも「場面ひとつぶん」しか超えない
        （30コール × 出力上限2,200トークン ≒ $0.034。0.55 + 0.034 < 0.6）。
        """
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
            logger.info("m%02d 提示%d 出品%d 応諾%d 累計%d区画 残り%d人 $%.4f",
                        step, m["offers_sent"], m["listed_this_month"],
                        m["accepted_this_month"], m["parcels_cum"],
                        m["risk_set"], self._cost_so_far())
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

    # -- 集計と出力 ------------------------------------------------------------

    def listing_outcomes(self) -> Dict[str, int]:
        """「売れない家」の勘定（設計 §1-5・決定論）。

        第 t 月の出品に対する X社 の反応は第 t+1 月に現れる（X社は月初に動くため）。
        """
        offered = {(o["step"], o["to_id"]) for o in self.offers}
        accepted = {(o["step"], o["to_id"]) for o in self.offers if o["accepted"]}
        out = {"listings_total": len(self.listings),
               "listing_sold_same_month": 0,
               "listing_no_offer_next": 0,
               "listing_offer_not_accepted_next": 0,
               "listing_accepted_next": 0,
               "listing_no_next_month": 0}
        months_run = len(self.monthly)
        for row in self.listings:
            t, aid = int(row["step"]), str(row["agent_id"])
            if row.get("sold_same_month"):
                # 出した同じ月に提示に応じて名義が移った＝翌月を待たずに片付いた
                out["listing_sold_same_month"] += 1
            elif t + 1 > months_run:
                out["listing_no_next_month"] += 1
            elif (t + 1, aid) not in offered:
                out["listing_no_offer_next"] += 1
            elif (t + 1, aid) in accepted:
                out["listing_accepted_next"] += 1
            else:
                out["listing_offer_not_accepted_next"] += 1
        out["unsold_listings"] = (out["listings_total"]
                                  - out["listing_accepted_next"]
                                  - out["listing_sold_same_month"])
        return out

    def _dump_timeline(self) -> None:
        """主体ごとの月順タイムライン（v8 と同じ形＋v8b の2つの答え）。"""
        out_dir = os.path.join(self.run_dir, "timeline_v8b")
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
            for step in range(1, len(self.monthly) + 1):
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
                    "listing": (d["listing"] if d else NOT_ASKED),
                    "respond": (d["respond"] if d else NOT_ASKED),
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
                          "file": "timeline_v8b/" + aid + ".json"})
        with open(os.path.join(self.run_dir, "timeline_index.json"), "w",
                  encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    def _finalize(self, elapsed: float) -> Dict[str, Any]:
        summary = super()._finalize(elapsed)
        summary["scenario_version"] = "field_v8b"
        summary.update(self.listing_outcomes())
        summary["respond_no_answer"] = self.respond_no_answer
        summary["acquirer_missing_targets"] = self.acquirer_missing_targets
        summary["acquirer_dup_rows"] = self.acquirer_dup_rows
        summary["acquirer_off_range"] = self.acquirer_off_range
        summary["acquirer_chunk_fail"] = self.acquirer_chunk_fail
        summary["partial_month"] = self.partial_month
        summary["acquirer_empty_text"] = self.acquirer_empty_text
        summary["invalid_listing"] = self.invalid_listing
        summary["invalid_respond"] = self.invalid_respond
        summary["offers_to_listed"] = sum(
            1 for o in self.offers
            if o["to_id"] in set(self.listed_by_step.get(int(o["step"]) - 1, [])))
        summary["offers_to_unlisted"] = (summary["offers_total"]
                                         - summary["offers_to_listed"])
        summary["listed_agents_ever"] = len({r["agent_id"] for r in self.listings})

        def dump(name: str, obj: Any) -> None:
            with open(os.path.join(self.run_dir, name), "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)

        dump("listings.json", self.listings)
        dump("acquirer_raw.json", self.acquirer_raw)
        dump("summary.json", summary)
        return summary


__all__ = ["SimulationV8B", "MockV8BClient"]
