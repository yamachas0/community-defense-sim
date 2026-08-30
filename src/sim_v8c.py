"""v8c「売る／売らないと、その理由」の月ループ（v8b からの差分だけ）。

設計の正は `docs/world_design_v8c.md`。
**既存の `src/simulation.py`・`src/sim_v8.py`・`src/sim_v8b.py` には一切触らない**
（`SimulationV8B` を **継承** して、理由欄・売買の語・断りの一言だけ差し替える）。

このファイルの責務は v8b と同じ3つだけ:
  1. 各主体に「今月あなたに見えているもの」を配る
  2. LLM の返した答えを **そのまま** 登記簿に記帳する（解釈・補正・代行をしない）
  3. 記録を残す（理由の一言は原文のまま・判定も分類もしない）

やっていないこと（意図的に）:
  - 誰かの行動を条件分岐で決める
  - 「〜なら売る」「〜%の確率で」といったパラメータ
  - 答えが無かったときに「出さない」「売らない」を代わりに書き込むこと
  - 理由を書かせるための促し（空欄は空欄のまま数える）
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from .field_v8c import (ACQUIRER_INTRO_V8C, ACQUIRER_NAME, HOME,  # noqa: F401
                        LIST_NO, LIST_YES,
                        MAX_OFFER_CHARS, MAX_REASON_CHARS, NO_ANSWER,
                        NOT_ASKED, SELL_NO, SELL_YES, acquirer_schema_v8c,
                        build_acquirer_prefix_v8c, build_acquirer_prompt_v8c,
                        build_common_prefix_v8c, build_decide_prompt_v8c,
                        build_plan_prompt_v8c, build_scene_prompt_v8b,
                        decide_schema_v8c, delivered_offer_v8c, listing_order,
                        plan_schema_v8c, scene_schema_v8, sell_order)
from .sim_v8b import CostLimitReached, MockV8BClient, SimulationV8B

logger = logging.getLogger(__name__)

RESULT_SOLD = "売った"
RESULT_NOT_SOLD = "売らなかった"
RESULT_NO_ANSWER = "答えが返らなかった"


class MockV8CClient(MockV8BClient):
    """v8c 用の mock（配線と集計を通すためだけのもの）。

    **世界の挙動を作り込まない**：返す値はスキーマを満たす最小限で、
    2つの問いは seed 固定の乱数で決める（実験の結果としては一切使わない）。
    理由欄は一部を空文字にして「書かない自由」の経路も通す。
    """

    def generate(self, system_prompt: str, user_prompt: str,
                 schema: Optional[Dict[str, Any]] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, tag: str = "agent") -> str:
        self.usage.add(tag, input_tokens=len(user_prompt) // 3, output_tokens=45)
        props = (schema or {}).get("properties", {})
        if "offers" in props:
            item_props = props["offers"]["items"]["properties"]
            enum = item_props["to"]["enum"]
            offers = []
            for n in enum:
                send = self.rnd.random() < self.send_rate
                row = {"to": n, "send": send,
                       "text": (f"{n}の不動産を譲り受けたい。" if send else "")}
                if "reason" in item_props:
                    row["reason"] = ("登記簿の並びから" if self.rnd.random() < 0.7
                                     else "")
                offers.append(row)
            return json.dumps({"offers": offers}, ensure_ascii=False)
        if "go" in props:
            choices = props["go"]["enum"]
            return json.dumps({"thought": "今月のこと。",
                               "go": self.rnd.choice(choices),
                               "reason": ("用事があるから"
                                          if self.rnd.random() < 0.7 else "")},
                              ensure_ascii=False)
        if "listing" in props:
            out: Dict[str, Any] = {
                "thought": "決めた。",
                "listing": (LIST_YES if self.rnd.random() < self.list_rate
                            else LIST_NO),
                "listing_reason": ("家の事情で" if self.rnd.random() < 0.7 else ""),
            }
            if "sell" in props:
                out["sell"] = (SELL_YES if self.rnd.random() < self.respond_rate
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


class SimulationV8C(SimulationV8B):

    def __init__(self, cfg: Dict[str, Any], run_dir: str):
        if cfg.get("scenario_version") != "field_v8c":
            raise ValueError("config の scenario_version が field_v8c ではない")
        # 親（v8b）は field_v8b を要求するので、版名だけ差し替えて渡し、あとで戻す。
        super().__init__({**cfg, "scenario_version": "field_v8b"}, run_dir)
        self.cfg = cfg
        # 前置きは v8c のもので置き換える（住民側にX社の名前も命題も出ない）
        self.common_prefix = build_common_prefix_v8c(cfg, self.agents)
        self.acquirer_prefix = build_acquirer_prefix_v8c(cfg, self.agents)
        if str(cfg["llm"].get("provider", "mock")).lower() == "mock":
            self.client = MockV8CClient(seed=self.seed, usage=self.usage)
        # X社の理由欄（費用が上限に収まらないときの唯一の削りしろ・世界は変えない）
        self.acquirer_reason = bool(cfg.get("acquirer_reason", True))
        # 理由の記入率（空欄は空欄のまま数える）
        self.reason_counts: Dict[str, int] = {
            "plan_total": 0, "plan_blank": 0,
            "listing_total": 0, "listing_blank": 0,
            "sell_total": 0, "sell_blank": 0,
            "acquirer_total": 0, "acquirer_blank": 0,
        }
        self.declines_delivered = 0   # X社に届いた断りの一言の件数

    # -- 月 ------------------------------------------------------------------

    def _acquirer_turn(self, step: int) -> Dict[str, str]:
        """X社の月次。登記簿＋前月末の出品一覧＋畳んだ履歴（断りの一言つき）を見る。"""
        risk = self.reg.risk_set()
        if not risk:
            return {}
        names = [self.reg.name_of[aid] for aid in risk]
        chunks = [names[i:i + self.chunk] for i in range(0, len(names), self.chunk)]
        listed_names = [self.reg.name_of[aid]
                        for aid in self.listed_by_step.get(step - 1, [])]
        items = []
        for n, targets in enumerate(chunks, 1):
            up = build_acquirer_prompt_v8c(self.reg, step, self.n_steps, targets,
                                           self.offers, listed_names, n, len(chunks),
                                           with_reason=self.acquirer_reason)
            items.append((f"X{n}", up,
                          acquirer_schema_v8c(targets, self.acquirer_reason),
                          ":acquirer"))
        raws = self._call(items, "acquirer")
        for k in sorted(raws):
            self.acquirer_raw.append({"step": step, "chunk": k,
                                      "raw": (raws[k] or "")[:8000]})

        sent: Dict[str, str] = {}
        raw_text: Dict[str, str] = {}
        reasons: Dict[str, str] = {}
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
                aid = self.reg.id_of_name[to]
                # 相手に届く形＝自己紹介の1行を世界が添える（施主 00:56）。
                # X社が書いた条件文そのもの（raw）は offers に別に残す。
                sent[aid] = delivered_offer_v8c(text[:MAX_OFFER_CHARS])
                raw_text[aid] = text[:MAX_OFFER_CHARS]
                reasons[aid] = str(row.get("reason", "") or "").strip()[:MAX_REASON_CHARS]
        self.acquirer_missing_targets += len([x for x in names if x not in seen])
        for aid in sent:
            text = raw_text[aid]
            # result は「売った／売らなかった／答えが返らなかった」の3状態。
            # 欠損を「売らなかった」に丸めない。
            self.offers.append({"step": step, "to": self.reg.name_of[aid],
                                "to_id": aid, "text": text,
                                "delivered": delivered_offer_v8c(text),
                                "reason": reasons.get(aid, ""),
                                "result": RESULT_NOT_SOLD, "accepted": False,
                                "decline_reason": ""})
        return sent

    def _plan_turn(self, step: int, offers: Dict[str, str]) -> Dict[str, str]:
        """月初の思考と行き先＋理由の一言。"""
        items = []
        for a in self.agents:
            aid = str(a["id"])
            up = build_plan_prompt_v8c(a, self.reg, step, self.n_steps,
                                       self.all_labels, self.thought[aid],
                                       offers.get(aid),
                                       neighbours=self.neighbour_names.get(aid))
            items.append((aid, up, plan_schema_v8c(self.all_labels), ":plan"))
        raws = self._call(items, "plan")

        go: Dict[str, str] = {}
        for a in self.agents:
            aid = str(a["id"])
            act = self._read(aid, raws.get(aid, ""), step, "plan")
            where = HOME
            reason = ""
            if act is not None:
                self.thought[aid] = str(act.get("thought", "") or "")[:600]
                reason = str(act.get("reason", "") or "").strip()[:MAX_REASON_CHARS]
                self.reason_counts["plan_total"] += 1
                if not reason:
                    self.reason_counts["plan_blank"] += 1
                value = str(act.get("go", "") or "").strip()
                if value in self.label_to_venue:
                    where = self.label_to_venue[value]
                elif value and value != HOME:
                    self.invalid_venue += 1
            go[aid] = where
            self.plans.append({"step": step, "agent_id": aid,
                               "name": a["name"], "go": where,
                               "go_label": self.venue_labels.get(where, HOME),
                               "reason": reason,
                               "thought": self.thought[aid]})
        return go

    def _scene_prompt(self, a, step, venue_label, present):
        # 発話は判断ではないので理由欄は無い（v8b の場面をそのまま使う）。
        return build_scene_prompt_v8b(a, self.reg, step, self.n_steps,
                                      self.thought[str(a["id"])], venue_label,
                                      present)

    def _decide_turn(self, step: int, offers: Dict[str, str],
                     heard: Dict[str, List[Dict[str, Any]]]
                     ) -> Tuple[List[str], List[str], List[str]]:
        """月末の問い（①出す／出さない ②売る／売らない・それぞれ理由の一言つき）。

        返り値 = (売ると答えた人, 売買の答えが返らなかった人, 出品した人)
        """
        risk = self.reg.risk_set()
        prev = set(self.listed_by_step.get(step - 1, []))
        items = []
        lorders: Dict[str, List[str]] = {}
        sorders: Dict[str, List[str]] = {}
        for aid in risk:
            a = self.reg.by_id[aid]
            offer = offers.get(aid)
            lo = listing_order(int(a["index"]), step)
            so = sell_order(int(a["index"]), step) if offer else None
            lorders[aid] = lo
            if so:
                sorders[aid] = so
            up = build_decide_prompt_v8c(a, self.reg, step, self.n_steps,
                                         self.thought[aid], offer,
                                         heard.get(aid, []), list_order=lo,
                                         sell_order_=so,
                                         neighbours=self.neighbour_names.get(aid))
            items.append((aid, up, decide_schema_v8c(lo, so), ":decide"))
        raws = self._call(items, "decide")

        sellers: List[str] = []
        sell_blanks: List[str] = []
        listers: List[str] = []
        for aid in risk:
            a = self.reg.by_id[aid]
            offer = offers.get(aid, "")
            act = self._read(aid, raws.get(aid, ""), step, "decide")
            rec = {"step": step, "agent_id": aid, "name": a["name"],
                   "listing": NO_ANSWER, "listing_reason": "",
                   "sell": (NO_ANSWER if offer else NOT_ASKED),
                   "sell_reason": "",
                   "thought": "", "offer": offer,
                   "listing_order": lorders[aid],
                   "sell_order": sorders.get(aid, []),
                   "listed_last_month": aid in prev,
                   "heard": len(heard.get(aid, []))}
            if act is None:
                # **「出さない」「売らない」で埋めない**。答えが無かった事実を残す。
                self.no_answer += 1
                if offer:
                    self.respond_no_answer += 1
                    sell_blanks.append(aid)
                self.decisions.append(rec)
                continue
            thought = str(act.get("thought", "") or "")
            self.thought[aid] = thought[:600]
            rec["thought"] = thought

            listing = str(act.get("listing", "") or "").strip()
            rec["listing_reason"] = str(
                act.get("listing_reason", "") or "").strip()[:MAX_REASON_CHARS]
            self.reason_counts["listing_total"] += 1
            if not rec["listing_reason"]:
                self.reason_counts["listing_blank"] += 1
            if listing in (LIST_YES, LIST_NO):
                rec["listing"] = listing
                if listing == LIST_YES:
                    listers.append(aid)
            else:
                self.no_answer += 1
                self.invalid_listing += 1

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
                    self.respond_no_answer += 1
                    self.invalid_respond += 1
                    sell_blanks.append(aid)
            self.decisions.append(rec)
        return sellers, sell_blanks, listers

    def _step(self, step: int) -> None:
        """v8b の月と同じ順序。差は結果の語と、断りの一言をX社の履歴に載せること。"""
        offers = self._acquirer_turn(step)
        self._guard_cost()
        go = self._plan_turn(step, offers)
        self._guard_cost()
        heard = self._scene_turn(step, go, offers)
        self._guard_cost()
        sellers, sell_blanks, listers = self._decide_turn(step, offers, heard)

        # 名義が動くのは②で「売る」と答えたときだけ（出品では動かない）
        moved: List[str] = []
        for aid in sellers:
            moved += self.reg.apply_sale(aid, step)

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

        # 出品の記録（その月かぎり・翌月にX社が見る）。同じ月に名義が移った人は板に残さない。
        acc = set(sellers)
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
            "accepted_this_month": len(sellers),
            "sold_this_month": len(sellers),
            "parcels_this_month": len(moved),
            "sold_cum": len(self.reg.sold_ids()),
            "parcels_cum": len(self.reg.acquired_parcels()),
            "risk_set": len(risk),
            "attended": sum(1 for v in go.values() if v != HOME),
            "by_venue": by_venue,
            "utterances": sum(1 for u in self.utterances if u["step"] == step),
            "declines_with_reason": sum(1 for o in self.offers
                                        if o["step"] == step
                                        and o.get("decline_reason")),
            "heard_mean": round(sum(heard_counts) / len(heard_counts), 2),
            "heard_max": max(heard_counts) if heard_counts else 0,
            "heard_min": min(heard_counts) if heard_counts else 0,
        })

    # -- 集計と出力 ------------------------------------------------------------

    def _dump_timeline(self) -> None:
        """主体ごとの月順タイムライン（v8b の形＋2つの答えの理由）。"""
        out_dir = os.path.join(self.run_dir, "timeline_v8c")
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
                    "go_reason": pl.get("reason", ""),
                    "said": [{"text": u["text"], "at": u["venue_label"],
                              "to": u["talk_to"], "thought": u.get("thought", "")}
                             for u in said.get((step, aid), [])],
                    "heard": [{"from": g["from"], "route": g["route"],
                               "at": g["venue_label"], "text": g["text"]}
                              for g in got.get((step, aid), [])],
                    "offer": (o["text"] if o else ""),
                    "listing": (d["listing"] if d else NOT_ASKED),
                    "listing_reason": (d.get("listing_reason", "") if d else ""),
                    "sell": (d["sell"] if d else NOT_ASKED),
                    "sell_reason": (d.get("sell_reason", "") if d else ""),
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
                          "file": "timeline_v8c/" + aid + ".json"})
        with open(os.path.join(self.run_dir, "timeline_index.json"), "w",
                  encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    def _finalize(self, elapsed: float) -> Dict[str, Any]:
        summary = super()._finalize(elapsed)
        summary["scenario_version"] = "field_v8c"
        # 結果の語が v8b と違う（応じた→売った）ので、3状態を数え直す。
        summary["offers_accepted"] = sum(1 for o in self.offers
                                         if o["result"] == RESULT_SOLD)
        summary["offers_declined"] = sum(1 for o in self.offers
                                         if o["result"] == RESULT_NOT_SOLD)
        summary["offers_no_answer"] = sum(1 for o in self.offers
                                          if o["result"] == RESULT_NO_ANSWER)
        summary["acquirer_reason"] = self.acquirer_reason
        summary["reason_counts"] = dict(self.reason_counts)
        summary["declines_delivered"] = self.declines_delivered
        summary["decline_reason_offers"] = sum(1 for o in self.offers
                                               if o.get("decline_reason"))
        for key, total, blank in (("plan", "plan_total", "plan_blank"),
                                  ("listing", "listing_total", "listing_blank"),
                                  ("sell", "sell_total", "sell_blank"),
                                  ("acquirer", "acquirer_total", "acquirer_blank")):
            t = self.reason_counts[total]
            summary[f"reason_written_{key}"] = t - self.reason_counts[blank]
            summary[f"reason_rate_{key}"] = (
                round((t - self.reason_counts[blank]) / t, 4) if t else 0.0)

        def dump(name: str, obj: Any) -> None:
            with open(os.path.join(self.run_dir, name), "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)

        # offers.json は親が先に書いているので、断りの一言つきで書き直す。
        dump("offers.json", self.offers)
        dump("plans.json", self.plans)
        dump("decisions.json", self.decisions)
        dump("summary.json", summary)
        return summary


__all__ = ["SimulationV8C", "MockV8CClient", "CostLimitReached",
           "RESULT_SOLD", "RESULT_NOT_SOLD", "RESULT_NO_ANSWER"]
