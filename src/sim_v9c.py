"""v9c「借りている人にも出口がある町」の月ループ — v9b からの差分だけ。

設計の正は `docs/world_design_v9c.md`。
**`src/sim_v9.py`・`src/sim_v9b.py`・`src/field_v9*.py`・v8系には一切触らない**
（読み取り専用で借りて、差分だけを上書きする）。

v9b との差:
  1. 名簿が `configs/personas_v9c.yaml`（町にいない所有者14人のペルソナが厚い）
  2. 借りて使っている人にも毎月「今月、A市を出るか」を聞く（出た人は退場・区画は使用者なし）
  3. 借りて使っている人と家主のあいだに月1回の一言（任意・世界は中身を見ない・X社には見えない）
  4. 毎月の user プロンプトの冒頭に経過した時間（第N月・開始から N-1 か月・年齢）

A（実行できない約束の門番）と B（X社の設定2行）は v9b をそのまま継承する。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from .field_v9 import (ACQUIRER_NAME, HOME, KIND_VALUES, LIST_NO,  # noqa: F401
                       LIST_TO_KIND, LIST_VALUES, MAX_REASON_CHARS, NO_ANSWER,
                       NOT_ASKED, SELL_NO, SELL_YES, RegistryV9,
                       build_decide_prompt_v9, load_personas_v9, rotate,
                       sell_order, transfer_notice)
from .field_v9c import (LEAVE_NO, LEAVE_YES, MAX_LINE_CHARS,  # noqa: F401
                        build_absentee_prefix_v9c, build_common_prefix_v9c,
                        build_tenant_prompt_v9c, decide_schema_v9c,
                        landlord_line_row, landlord_of, landlord_reply_block,
                        leave_order, tenant_ids, tenant_line_row,
                        tenant_parcels, tenant_schema_v9c, time_header,
                        vacancy_notice)
from .sim_v9 import RESULT_NO_ANSWER, RESULT_NOT_SOLD, RESULT_SOLD
from .sim_v9b import MockV9BClient, SimulationV9B

logger = logging.getLogger(__name__)


class RegistryV9C(RegistryV9):
    """v9 の帳簿に「使う人がいなくなった区画」を足しただけ。

    借りて使っていた人がA市を出ると、その区画は**使う人がいなくなる**
    （家主が自動で使い始めることはない＝設計 §2）。
    """

    def __init__(self, agents, parcels):
        super().__init__(agents, parcels)
        self.vacated: set = set()

    def user_of(self, parcel: str) -> Optional[str]:
        if str(parcel) in self.vacated:
            return None
        return super().user_of(parcel)


class MockV9CClient(MockV9BClient):
    """mock 専用。借りて使っている人の答えと一言も返せるようにするだけ。"""

    def generate(self, system_prompt: str, user_prompt: str,
                 schema: Optional[Dict[str, Any]] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, tag: str = "agent") -> str:
        props = (schema or {}).get("properties", {})
        if "leave" in props:
            self.usage.add(tag, input_tokens=len(user_prompt) // 3,
                           output_tokens=40)
            out: Dict[str, Any] = {
                "thought": "今月のこと。",
                "leave": (LEAVE_YES if self.rnd.random() < 0.03 else LEAVE_NO),
                "leave_reason": ("暮らしの事情で" if self.rnd.random() < 0.7 else ""),
            }
            if "to_landlord" in props:
                out["to_landlord"] = ("今月も変わりありません。"
                                      if self.rnd.random() < 0.5 else "")
            return json.dumps(out, ensure_ascii=False)
        raw = super().generate(system_prompt, user_prompt, schema=schema,
                               temperature=temperature, max_tokens=max_tokens,
                               tag=tag)
        if "to_tenant" in props:
            obj = json.loads(raw)
            obj["to_tenant"] = {
                p: ("承知しました。" if self.rnd.random() < 0.5 else "")
                for p in props["to_tenant"]["properties"]}
            return json.dumps(obj, ensure_ascii=False)
        return raw


class SimulationV9C(SimulationV9B):

    def __init__(self, cfg: Dict[str, Any], run_dir: str):
        if cfg.get("scenario_version") != "field_v9c":
            raise ValueError("config の scenario_version が field_v9c ではない")
        inner = dict(cfg)
        inner["scenario_version"] = "field_v9b"   # 親の検算を通すためだけ
        super().__init__(inner, run_dir)
        self.cfg["scenario_version"] = "field_v9c"

        # 帳簿を v9c 版に差し替える（使う人がいなくなった区画を持てる）
        self.reg = RegistryV9C(self.agents, self.parcels)
        from .field_v9 import adjacency_v9
        self.neighbours = adjacency_v9(self.reg)
        self.neighbour_names = {aid: [self.reg.name_of[x] for x in nbs]
                                for aid, nbs in self.neighbours.items()}
        n_parcels = len(self.parcels)
        self.common_prefix = build_common_prefix_v9c(cfg, self.agents, n_parcels)
        self.absentee_prefix = build_absentee_prefix_v9c(cfg, n_parcels)
        from .field_v9b import build_acquirer_prefix_v9b
        self.acquirer_prefix = build_acquirer_prefix_v9b(cfg, self.reg)
        if str(cfg["llm"].get("provider", "mock")).lower() == "mock":
            self.client = MockV9CClient(seed=self.seed, usage=self.usage,
                                        reg=self.reg)

        self._cur_step = 1
        self._roster_left: List[str] = []
        # 借りて使っている人の月末の答えと、一言のやり取り
        self.tenant_decisions: List[Dict[str, Any]] = []
        self.left_tenants: List[Dict[str, Any]] = []
        self.messages: List[Dict[str, Any]] = []      # 一言（両方向）全件
        self.vacancy_notices: List[Dict[str, Any]] = []
        # 翌月に配るもの（書いた時点で宛先を確定して持つ＝走行前レビューの必須指摘）
        self._pending_to_landlord: List[Dict[str, Any]] = []
        self._pending_to_tenant: List[Dict[str, Any]] = []
        # X社が所有者になっている区画で使う人がいなくなった事実（翌月X社に返す）
        self._x_vacancy: Dict[int, List[str]] = {}
        self._skip_checkpoint = False
        self.tenant_no_answer = 0
        self.invalid_leave = 0
        self.undelivered_lines = 0

    # -- 経過した時間を全部の user プロンプトの冒頭に置く（差分4） -------------------

    def _call(self, items: List[Tuple[str, str, str, Dict[str, Any], str]],
              tag: str) -> Dict[str, str]:
        out = []
        for key, system, up, schema, sfx in items:
            agent = self.reg.by_id.get(key)
            head = time_header(self._cur_step, agent)
            if tag == "acquirer":
                # X社は命題の行が先頭（v9 の規律）。その次の行に置く。
                first, _, rest = up.partition("\n")
                vac = self._x_vacancy.get(self._cur_step) or []
                block = ""
                if vac:
                    block = ("\n[あなたが所有する区画に起きたこと]\n"
                             + "\n".join(f"  {x}" for x in vac))
                up = f"{first}\n{head}{block}\n{rest}"
            else:
                up = f"{head}\n{up}"
            out.append((key, system, up, schema, sfx))
        return super()._call(out, tag)

    # -- 通知（所有権が移った事実＋使う人がいなくなった事実） -----------------------

    def _notices_for(self, step: int) -> Dict[str, List[str]]:
        out = super()._notices_for(step)
        for row in self.left_tenants:
            if int(row["step"]) != step - 1:
                continue
            for parcel in row["parcels"]:
                who = row["landlord_id"].get(parcel)
                line = vacancy_notice(parcel, row["name"])
                if who is None:
                    # 退場した時点の所有者が X社 だった区画。事実は所有者に届く
                    # （走行前レビューの必須指摘 2026-08-30）。
                    self._x_vacancy.setdefault(step, []).append(line)
                    self.vacancy_notices.append(
                        {"step": step, "to": ACQUIRER_NAME,
                         "to_name": ACQUIRER_NAME, "parcel": parcel,
                         "tenant": row["name"], "text": line})
                    continue
                out.setdefault(who, []).append(line)
                self.vacancy_notices.append(
                    {"step": step, "to": who,
                     "to_name": self.reg.name_of.get(who, who),
                     "parcel": parcel, "tenant": row["name"], "text": line})
        return out

    # -- 月末の問い（所有者＋借りて使っている人） ----------------------------------

    def _decide_turn(self, step: int, offers: Dict[str, Dict[str, Any]],
                     heard: Dict[str, List[Dict[str, Any]]],
                     notices: Dict[str, List[str]]
                     ) -> Tuple[List[str], List[str], List[Tuple[str, str, str]]]:
        """v9 の月末の問いに、借りて使っている人の問いと一言を足したもの。"""
        risk = self.reg.risk_set()
        tenants = tenant_ids(self.reg)
        items = []
        opts: Dict[str, List[Tuple[str, List[str]]]] = {}
        sorders: Dict[str, List[str]] = {}
        replies: Dict[str, List[str]] = {}      # 家主 → 返事を書ける区画
        torders: Dict[str, List[str]] = {}
        tparcels: Dict[str, List[str]] = {}
        tlines: Dict[str, bool] = {}

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
            # 借りて使っている人から届いた一言（宛先が自分のものだけ）
            inbox_lines = [m for m in self._pending_to_landlord
                           if m["to_id"] == aid]
            mine = sorted({m["parcel"] for m in inbox_lines})
            replies[aid] = mine
            up = build_decide_prompt_v9(
                a, self.reg, step, self.n_steps, self.thought[aid], offer,
                heard.get(aid, []), listing_options=lo, sell_order_=so,
                notices=(list(notices.get(aid) or [])
                         + [tenant_line_row(m["parcel"], m["from_name"],
                                            m["text"]) for m in inbox_lines]),
                neighbours=(self.neighbour_names.get(aid) if in_town else None),
                in_town=in_town)
            if mine:
                up += "\n" + "\n".join(landlord_reply_block(mine))
            items.append((aid, self._system_for(aid), up,
                          decide_schema_v9c(lo, so, mine), ":decide"))

        for aid in tenants:
            a = self.reg.by_id[aid]
            ps = tenant_parcels(self.reg, aid)
            tparcels[aid] = ps
            order = leave_order(int(a["index"]), step)
            torders[aid] = order
            got_rows = [m for m in self._pending_to_tenant if m["to_id"] == aid]
            got = [landlord_line_row(m["parcel"], m["from_name"], m["text"])
                   for m in got_rows]
            # 家主がいる区画が1つも無ければ（すべてX社のもの）一言の欄は出さない
            has_landlord = any(landlord_of(self.reg, p) for p in ps)
            tlines[aid] = has_landlord
            up = build_tenant_prompt_v9c(
                a, self.reg, step, self.n_steps, self.thought[aid], ps, order,
                heard=heard.get(aid, []), notices=notices.get(aid), lines=got,
                with_line=has_landlord)
            items.append((aid, self._system_for(aid), up,
                          tenant_schema_v9c(order, has_landlord), ":tenant"))

        raws = self._call(items, "decide")

        # 一言は毎月配り切る（次の月のぶんをここから作り直す）
        delivered_to_tenant = {m["to_id"] for m in self._pending_to_tenant}
        self._pending_to_landlord = []
        self._pending_to_tenant = []

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
                "tenant_lines_in": len(replies.get(aid) or []),
            }
            if act is None:
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

            # 家主の返事（任意・世界は中身を見ない）
            out_lines = act.get("to_tenant") or {}
            if not isinstance(out_lines, dict):
                out_lines = {}
            for parcel in (replies.get(aid) or []):
                txt = str(out_lines.get(parcel, "") or "").strip()[:MAX_LINE_CHARS]
                if not txt:
                    continue
                to_id = self.reg.tenant_of.get(parcel)
                if to_id is None or not self.reg.is_resident(to_id):
                    self.undelivered_lines += 1   # 相手がもういない＝配らない
                    continue
                self._pending_to_tenant.append(
                    {"step": step, "parcel": parcel, "to_id": to_id,
                     "from_name": a["name"], "text": txt})
                self.messages.append(
                    {"step": step, "direction": "家主→借りて使っている人",
                     "parcel": parcel, "from": a["name"],
                     "to": self.reg.name_of[to_id], "text": txt})
            self.decisions.append(rec)

        # 借りて使っている人
        for aid in tenants:
            a = self.reg.by_id[aid]
            ps = tparcels[aid]
            act = self._read(aid, raws.get(aid, ""), step, "tenant")
            rec = {"step": step, "agent_id": aid, "name": a["name"],
                   "parcels": ps, "leave": NO_ANSWER, "leave_reason": "",
                   "thought": "", "leave_order": torders[aid],
                   "line_out": "",
                   "line_in": (1 if aid in delivered_to_tenant else 0),
                   "heard": len(heard.get(aid, []))}
            if act is None:
                self.no_answer += 1
                self.tenant_no_answer += 1
                self.tenant_decisions.append(rec)
                continue
            thought = str(act.get("thought", "") or "")
            self.thought[aid] = thought[:600]
            rec["thought"] = thought
            v = str(act.get("leave", "") or "").strip()
            rec["leave_reason"] = str(
                act.get("leave_reason", "") or "").strip()[:MAX_REASON_CHARS]
            if v in (LEAVE_YES, LEAVE_NO):
                rec["leave"] = v
            else:
                # **既定値で埋めない**。答えが無かった事実を残す。
                self.tenant_no_answer += 1
                self.invalid_leave += 1
            txt = str(act.get("to_landlord", "") or "").strip()[:MAX_LINE_CHARS]
            if txt and tlines.get(aid):
                rec["line_out"] = txt
                for p in ps:
                    to_id = landlord_of(self.reg, p)
                    if to_id is None:
                        self.undelivered_lines += 1   # 家主がX社＝私信は渡さない
                        continue
                    self._pending_to_landlord.append(
                        {"step": step, "parcel": p, "to_id": to_id,
                         "from_name": a["name"], "text": txt})
                    self.messages.append(
                        {"step": step, "direction": "借りて使っている人→家主",
                         "parcel": p, "from": a["name"],
                         "to": self.reg.name_of[to_id], "text": txt})
            self.tenant_decisions.append(rec)
        return sellers, sell_blanks, listers

    # -- 月 --------------------------------------------------------------------

    def _step(self, step: int) -> None:
        self._cur_step = step
        # 名簿の「（転出）」は町を出た人が増えたときだけ作り直す（それ以外は同じ塊）
        left_now = sorted(self.reg.left_ids())
        if left_now != self._roster_left:
            self._roster_left = left_now
            self.common_prefix = build_common_prefix_v9c(
                self.cfg, self.agents, len(self.parcels), left_now)
        self._skip_checkpoint = True
        try:
            super()._step(step)
        finally:
            self._skip_checkpoint = False
        # 借りて使っている人の退場は、その月の売買のあとに反映する
        leaving = [r for r in self.tenant_decisions
                   if r["step"] == step and r["leave"] == LEAVE_YES]
        for r in leaving:
            aid = r["agent_id"]
            if self.reg.left_month.get(aid) is not None:
                continue
            ps = [p for p in r["parcels"] if self.reg.tenant_of.get(p) == aid]
            self.reg.left_month[aid] = int(step)
            landlords = {p: landlord_of(self.reg, p) for p in ps}
            for p in ps:
                self.reg.vacated.add(p)
                # 借りて使う人の欄も空にする（登記簿に、もう町にいない人を載せない）
                self.reg.tenant_of[p] = None
            self.left_tenants.append(
                {"step": step, "agent_id": aid, "name": self.reg.name_of[aid],
                 "parcels": ps, "reason": r["leave_reason"],
                 "landlord_id": landlords})
            self.left_agents.append(
                {"step": step, "agent_id": aid, "name": self.reg.name_of[aid],
                 "parcel": "・".join(ps), "still_owns": [],
                 "how": "借りて使っていた人がA市を出た"})
        if leaving:
            # 退場のあとの帳簿を、その月の記録として残し直す
            self.ledger_by_step[-1] = {"step": step,
                                       "rows": self.reg.ledger_rows()}
        m = self.monthly[-1] if self.monthly else None
        if m is not None and m.get("step") == step:
            m["tenant_calls"] = len([r for r in self.tenant_decisions
                                     if r["step"] == step])
            m["tenant_left_this_month"] = len(leaving)
            m["tenant_left_cum"] = len(self.left_tenants)
            m["messages_this_month"] = len([x for x in self.messages
                                            if x["step"] == step])
            m["left_cum"] = len(self.reg.left_ids())
            m["in_town"] = len(self.reg.in_town_ids())
            m["no_user_parcels"] = sum(1 for p in self.reg.parcel_names
                                       if self.reg.user_of(p) is None)
        try:
            # 退場と一言を含めた状態でもう一度書き出す（落ちても取り戻せるように）
            self._checkpoint(step)
        except Exception as e:  # noqa: BLE001
            logger.warning("チェックポイントの書き出しに失敗（続行）: %s", e)

    # -- 書き出し ----------------------------------------------------------------

    def _checkpoint(self, step: int) -> None:
        # 親の _step からの呼び出しは飛ばす＝退場を反映したあとに月1回だけ書く
        # （走行前レビューの必須指摘 2026-08-30・二重計上を止める）。
        if self._skip_checkpoint:
            return
        super()._checkpoint(step)
        for name, obj in (("tenant_decisions.json", self.tenant_decisions),
                          ("messages.json", self.messages),
                          ("left_tenants.json", self.left_tenants),
                          ("vacancy_notices.json", self.vacancy_notices)):
            tmp = os.path.join(self.checkpoint_dir, name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            os.replace(tmp, os.path.join(self.checkpoint_dir, name))

    def _finalize(self, elapsed: float) -> Dict[str, Any]:
        summary = super()._finalize(elapsed)
        summary["scenario_version"] = "field_v9c"
        t_out = [m for m in self.messages if m["direction"].startswith("借りて")]
        t_in = [m for m in self.messages if m["direction"].startswith("家主")]
        summary["tenant_calls_total"] = len(self.tenant_decisions)
        summary["tenant_left_total"] = len(self.left_tenants)
        summary["tenant_no_answer"] = self.tenant_no_answer
        summary["invalid_leave"] = self.invalid_leave
        summary["tenant_leave_counts"] = {
            v: sum(1 for r in self.tenant_decisions if r["leave"] == v)
            for v in (LEAVE_YES, LEAVE_NO, NO_ANSWER)}
        summary["messages_total"] = len(self.messages)
        summary["messages_tenant_to_landlord"] = len(t_out)
        summary["messages_landlord_to_tenant"] = len(t_in)
        summary["vacancy_notices_total"] = len(self.vacancy_notices)
        summary["vacancy_notices_to_acquirer"] = sum(
            1 for x in self.vacancy_notices if x["to"] == ACQUIRER_NAME)
        summary["undelivered_lines"] = self.undelivered_lines
        summary["vacated_parcels_end"] = sorted(self.reg.vacated)
        for name, obj in (("tenant_decisions.json", self.tenant_decisions),
                          ("messages.json", self.messages),
                          ("left_tenants.json", self.left_tenants),
                          ("vacancy_notices.json", self.vacancy_notices)):
            with open(os.path.join(self.run_dir, name), "w",
                      encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.run_dir, "summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.run_dir, "common_prefix.txt"), "w",
                  encoding="utf-8") as f:
            f.write(self.common_prefix)
        return summary


__all__ = ["SimulationV9C", "RegistryV9C", "MockV9CClient"]
