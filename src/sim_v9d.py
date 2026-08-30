"""v9d「ものさしとしてのお金」の月ループ — v9c からの差分だけ。

設計の正は `docs/world_design_v9d.md`。
**v9c までのファイルには一切触らない**（読み取り専用で借りて、差分だけを上書きする）。

v9c との差:
  1. 評価額（公開）を帳簿に置く（走行前に凍結した表からしか作らない）
  2. X社の提示に金額が必須（金額はX社が決める）
  3. X社の資金は有限（町の全評価額の51%）。残額を超える提示は世界が配らない
  4. 売った人の記録に金額が入る（財布は記帳だけ・何も起こさない）
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from .field_v9 import (ACQUIRER_NAME, KIND_VALUES, LIST_TO_KIND,  # noqa: F401
                       MAX_OFFER_CHARS, MAX_REASON_CHARS, delivered_offer_v9)
from .field_v9b import (UNDELIVERED_PROMISE, UNDELIVERED_RIGHTS,
                        undeliverable_promise)
from .field_v9d import (UNDELIVERED_BUDGET, acquirer_schema_v9d,
                        build_absentee_prefix_v9d, build_acquirer_prefix_v9d,
                        build_acquirer_prompt_v9d, build_common_prefix_v9d,
                        build_valuation, money_block_for_owner,
                        offer_amount_row, total_value, value_of, yen)
from .sim_v9 import RESULT_NOT_SOLD
from .sim_v9c import MockV9CClient, SimulationV9C

logger = logging.getLogger(__name__)

BUDGET_SHARE = 0.51      # 施主決定＝町の全評価額の 51%


class MockV9DClient(MockV9CClient):
    """mock 専用。金額の欄を埋めるだけ（世界の挙動は作り込まない）。"""

    def generate(self, system_prompt: str, user_prompt: str,
                 schema: Optional[Dict[str, Any]] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, tag: str = "agent") -> str:
        raw = super().generate(system_prompt, user_prompt, schema=schema,
                               temperature=temperature, max_tokens=max_tokens,
                               tag=tag)
        props = (schema or {}).get("properties", {})
        if "offers" not in props:
            return raw
        obj = json.loads(raw)
        for row in obj.get("offers", []):
            row["amount"] = (int(self.rnd.choice([3, 5, 8, 12, 20, 40]) * 1_000_000)
                             if row.get("send") else 0)
        return json.dumps(obj, ensure_ascii=False)


class SimulationV9D(SimulationV9C):

    def __init__(self, cfg: Dict[str, Any], run_dir: str):
        if cfg.get("scenario_version") != "field_v9d":
            raise ValueError("config の scenario_version が field_v9d ではない")
        inner = dict(cfg)
        inner["scenario_version"] = "field_v9c"
        super().__init__(inner, run_dir)
        self.cfg["scenario_version"] = "field_v9d"

        self.valuation = build_valuation(self.parcels)
        self.total_value = total_value(self.valuation)
        self.budget_total = int(round(self.total_value * BUDGET_SHARE / 100_000)
                                * 100_000)
        self.spent = 0
        self.wallet: Dict[str, int] = {str(a["id"]): 0 for a in self.agents}
        self.payments: List[Dict[str, Any]] = []
        self.acquirer_over_budget = 0
        self.acquirer_bad_amount = 0
        # チェックポイントは「支払いを記帳したあと」に月1回だけ書く
        # （走行前レビューの必須指摘 2026-08-30）。
        self._defer_checkpoint = False

        n_parcels = len(self.parcels)
        self.common_prefix = build_common_prefix_v9d(cfg, self.agents, n_parcels)
        self.absentee_prefix = build_absentee_prefix_v9d(cfg, n_parcels)
        self.acquirer_prefix = build_acquirer_prefix_v9d(cfg, self.reg,
                                                         self.budget_total)
        if str(cfg["llm"].get("provider", "mock")).lower() == "mock":
            self.client = MockV9DClient(seed=self.seed, usage=self.usage,
                                        reg=self.reg)

    @property
    def budget_left(self) -> int:
        return self.budget_total - self.spent

    # -- 名簿を作り直すときも v9d の前置きで --------------------------------------

    def _step(self, step: int) -> None:
        self._cur_step = step
        left_now = sorted(self.reg.left_ids())
        if left_now != self._roster_left:
            self._roster_left = left_now
            self.common_prefix = build_common_prefix_v9d(
                self.cfg, self.agents, len(self.parcels), left_now)
            self._roster_rebuilt = True
        # v9c の月ループをそのまま使う（名簿の作り直しは上で v9d 版として済ませてある）
        self._defer_checkpoint = True
        try:
            SimulationV9C._step(self, step)
        finally:
            self._defer_checkpoint = False
        # 売れた分の支払いを記帳する（v9c の _step が売買を確定させたあと）
        self._settle(step)
        m = self.monthly[-1] if self.monthly else None
        if m is not None and m.get("step") == step:
            m["spent_cum"] = self.spent
            m["budget_left"] = self.budget_left
            m["paid_this_month"] = sum(int(x["amount"]) for x in self.payments
                                       if x["step"] == step)
        try:
            self._checkpoint(step)
        except Exception as e:  # noqa: BLE001
            logger.warning("チェックポイントの書き出しに失敗（続行）: %s", e)

    def _settle(self, step: int) -> None:
        """その月に成立した売買の金額を、売った人の記録に入れる（記帳だけ）。"""
        for o in self.offers:
            if o["step"] != step or not o.get("accepted") or o.get("paid"):
                continue
            amount = int(o.get("amount", 0))
            o["paid"] = True
            self.spent += amount
            self.wallet[o["to_id"]] = self.wallet.get(o["to_id"], 0) + amount
            self.payments.append(
                {"step": step, "to": o["to"], "to_id": o["to_id"],
                 "parcel": o["parcel"], "kind": o["kind"], "amount": amount,
                 "valuation": value_of(self.valuation, o["parcel"], o["kind"]),
                 "budget_left_after": self.budget_left})

    # -- X社の月次（v9b/v9c の門番＋金額と資金） ----------------------------------

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
            up = build_acquirer_prompt_v9d(
                self.reg, self.valuation, step, self.n_steps, targets,
                self.offers, listed_rows, parcels, n, len(chunks),
                self.budget_left, self.budget_total, self.spent,
                with_reason=self.acquirer_reason, undelivered=self.undelivered)
            items.append((f"X{n}", self.acquirer_prefix, up,
                          acquirer_schema_v9d(targets, parcels,
                                              self.acquirer_reason), ":acquirer"))
        raws = self._call(items, "acquirer")
        for k in sorted(raws):
            self.acquirer_raw.append({"step": step, "chunk": k,
                                      "raw": (raws[k] or "")[:8000]})

        sent: Dict[str, Dict[str, Any]] = {}
        seen: set = set()
        committed = 0     # その月に配った提示の合計（全部成約しても資金を超えない）
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
                text = text[:MAX_OFFER_CHARS]
                reason = str(row.get("reason", "") or "").strip()[:MAX_REASON_CHARS]
                try:
                    amount = int(row.get("amount", 0) or 0)
                except (TypeError, ValueError):
                    amount = -1
                if not self.reg.can_offer(aid, parcel, kind):
                    self.acquirer_invalid_offer += 1
                    self._record_undelivered(step, to, parcel, kind, text,
                                             reason, UNDELIVERED_RIGHTS,
                                             amount)
                    continue
                if undeliverable_promise(text):
                    self.acquirer_undeliverable_promise += 1
                    self._record_undelivered(step, to, parcel, kind, text,
                                             reason, UNDELIVERED_PROMISE,
                                             amount)
                    continue
                if amount <= 0:
                    # 金額が無い提示は成り立たない（既定値で埋めない）
                    self.acquirer_bad_amount += 1
                    continue
                if amount > self.budget_left - committed:
                    self.acquirer_over_budget += 1
                    self._record_undelivered(step, to, parcel, kind, text,
                                             reason, UNDELIVERED_BUDGET, amount)
                    continue
                committed += amount
                sent[aid] = {"parcel": parcel, "kind": kind, "text": text,
                             "amount": amount,
                             "delivered": delivered_offer_v9(text),
                             "reason": reason}
        self.acquirer_missing_targets += len([x for x in names if x not in seen])
        for aid, o in sent.items():
            v = value_of(self.valuation, o["parcel"], o["kind"])
            self.offers.append({
                "step": step, "to": self.reg.name_of[aid], "to_id": aid,
                "to_in_town": self.reg.is_resident(aid),
                "parcel": o["parcel"], "kind": o["kind"], "text": o["text"],
                "amount": o["amount"], "valuation": v,
                "ratio": (round(o["amount"] / v, 4) if v else None),
                "delivered": o["delivered"], "reason": o["reason"],
                "result": RESULT_NOT_SOLD, "accepted": False, "paid": False,
                "decline_reason": ""})
        return sent

    def _record_undelivered(self, step, to, parcel, kind, text, reason, why,
                            amount=0):
        row = {"step": step, "to": to, "parcel": parcel, "kind": kind,
               "text": text, "reason": reason, "why": why,
               "amount": int(amount),
               "valuation": value_of(self.valuation, parcel, kind)
               if parcel in self.valuation else None}
        self.undelivered.append(row)

    # -- 月末の問い（金額の事実を並べる） -----------------------------------------

    def _decide_turn(self, step, offers, heard, notices):
        """v9c の月末の問いに、評価額と提示額の事実を足す（比べる語は書かない）。"""
        extra: Dict[str, List[str]] = {}
        for aid, o in offers.items():
            extra.setdefault(aid, []).append(
                offer_amount_row(self.valuation, o["parcel"], o["kind"],
                                 int(o["amount"])))
        merged = dict(notices)
        for aid, rows in extra.items():
            merged[aid] = list(merged.get(aid) or []) + rows
        self._money_blocks = {
            aid: money_block_for_owner(
                self.valuation, self.reg, aid,
                sorted(set(self.reg.parcels_owned(aid))
                       | ({offers[aid]["parcel"]} if aid in offers else set())),
                self.wallet.get(aid, 0))
            for aid in self.reg.risk_set()}
        return super()._decide_turn(step, offers, heard, merged)

    def _call(self, items, tag):
        """月末の問いのプロンプトに、その人の評価額と記録の金額を足す。"""
        if tag == "decide" and getattr(self, "_money_blocks", None):
            out = []
            for key, system, up, schema, sfx in items:
                block = self._money_blocks.get(key)
                if block and sfx == ":decide":
                    up = up + "\n\n" + "\n".join(block)
                out.append((key, system, up, schema, sfx))
            items = out
        return super()._call(items, tag)

    # -- 書き出し ----------------------------------------------------------------

    def _checkpoint(self, step: int) -> None:
        # 支払いを記帳する前の呼び出し（v9c の月ループの中）は飛ばす
        if self._defer_checkpoint:
            return
        super()._checkpoint(step)
        if self._skip_checkpoint:
            return
        tmp = os.path.join(self.checkpoint_dir, "payments.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.payments, f, ensure_ascii=False, indent=2)
        os.replace(tmp, os.path.join(self.checkpoint_dir, "payments.json"))

    def _finalize(self, elapsed: float) -> Dict[str, Any]:
        summary = super()._finalize(elapsed)
        summary["scenario_version"] = "field_v9d"
        summary["valuation_total"] = self.total_value
        summary["budget_total"] = self.budget_total
        summary["budget_share"] = BUDGET_SHARE
        summary["spent_total"] = self.spent
        summary["budget_left"] = self.budget_left
        summary["acquirer_over_budget"] = self.acquirer_over_budget
        summary["acquirer_bad_amount"] = self.acquirer_bad_amount
        paid = [o for o in self.offers if o.get("accepted")]
        summary["paid_deals"] = len(paid)
        ratios = [o["ratio"] for o in self.offers if o.get("ratio") is not None]
        summary["offer_ratio_mean"] = (round(sum(ratios) / len(ratios), 4)
                                       if ratios else None)
        summary["wallets_nonzero"] = {self.reg.name_of[a]: v
                                      for a, v in self.wallet.items() if v}
        for name, obj in (("payments.json", self.payments),
                          ("valuation.json", self.valuation),
                          ("wallets.json", {self.reg.name_of[a]: v
                                            for a, v in self.wallet.items()})):
            with open(os.path.join(self.run_dir, name), "w",
                      encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.run_dir, "summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary


__all__ = ["SimulationV9D", "MockV9DClient", "BUDGET_SHARE"]
