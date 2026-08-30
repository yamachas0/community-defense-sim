"""v9b「実行される約束だけが届く町」の月ループ — v9 からの差分だけ。

設計の正は `docs/world_design_v9b.md`。
**`src/sim_v9.py`・`src/field_v9.py`・v8 系のファイルには一切触らない**
（読み取り専用で借りて、差分だけを上書きする）。

v9 との差は2つだけ:
  A. X社の提示に「この世界で実行される仕組みが無い約束」が含まれていたら、
     世界はそれを配らない（件数を数え、翌月 X社 に事実と理由だけを返す）。
  B. X社の system プロンプトに施主指定の事実2行が足されている。

町の人の側の処理（行き先・会話・出品・売買・理由の一言）は1行も変えていない。
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional

from .field_v9 import KIND_VALUES, MAX_OFFER_CHARS, MAX_REASON_CHARS, RegistryV9
from .field_v9b import (UNDELIVERED_PROMISE, UNDELIVERED_RIGHTS,
                        acquirer_schema_v9, build_acquirer_prefix_v9b,
                        build_acquirer_prompt_v9b, promise_word_hits,
                        undeliverable_promise)
from .sim_v9 import (CHECKPOINT_FILES_V9, RESULT_NOT_SOLD,  # noqa: F401
                     MockV9Client, SimulationV9)

logger = logging.getLogger(__name__)


class MockV9BClient(MockV9Client):
    """mock 専用。A の配線を通すために、条件文に**わざと**実行できない約束を混ぜる。

    実験の結果には一切使わない（実APIのX社は自分で文を書く）。混ぜる語と割合は
    seed 固定の決定論で、世界の側の判定規則は一切変えない。
    """

    PROMISE_SAMPLES = (
        "改修費用は当社が負担します。",
        "地域の活動を支援します。",
        "建物の管理をお引き受けします。",
        "移転先をご紹介します。",
    )
    SAFE_SAMPLES = (
        "所有権の移転後も、あなたは借家として今までどおり住み続けられます。",
        "当社は不動産管理等は行いません。",
        "所有権の移転だけをお願いしたい。",
    )

    def __init__(self, *args, promise_rate: float = 0.35, **kwargs):
        super().__init__(*args, **kwargs)
        self.promise_rate = promise_rate
        self._prnd = random.Random(20260830)

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
            if not row.get("send"):
                continue
            if self._prnd.random() < self.promise_rate:
                row["text"] += self._prnd.choice(self.PROMISE_SAMPLES)
            else:
                row["text"] += self._prnd.choice(self.SAFE_SAMPLES)
        return json.dumps(obj, ensure_ascii=False)


class SimulationV9B(SimulationV9):
    """v9 の月ループそのまま。X社の月次だけを差し替える。"""

    def __init__(self, cfg: Dict[str, Any], run_dir: str):
        if cfg.get("scenario_version") != "field_v9b":
            raise ValueError("config の scenario_version が field_v9b ではない")
        inner = dict(cfg)
        inner["scenario_version"] = "field_v9"   # 親の検算を通すためだけ
        super().__init__(inner, run_dir)
        self.cfg["scenario_version"] = "field_v9b"
        # B: X社の前置きだけを差し替える（住民側の前置きは触らない）
        self.acquirer_prefix = build_acquirer_prefix_v9b(cfg, self.reg)
        # mock のときだけ A の配線を通せる client に入れ替える
        if str(cfg["llm"].get("provider", "mock")).lower() == "mock":
            self.client = MockV9BClient(seed=self.seed, usage=self.usage,
                                        reg=self.reg)
        # A: 配られなかった提示（理由つき・全件そのまま保存）
        self.undelivered: List[Dict[str, Any]] = []
        self.acquirer_undeliverable_promise = 0

    # -- X社の月次（v9 の写し＋ A の門番） ---------------------------------------

    def _acquirer_turn(self, step: int) -> Dict[str, Dict[str, Any]]:
        risk = self.reg.risk_set()
        if not risk:
            return {}
        names = [self.reg.name_of[aid] for aid in risk]
        chunks = [names[i:i + self.chunk] for i in range(0, len(names), self.chunk)]
        from .field_v9 import LIST_TO_KIND
        listed_rows = [f"{p}（{LIST_TO_KIND[v]}）"
                       for _aid, p, v in self.listed_by_step.get(step - 1, [])]
        items = []
        for n, targets in enumerate(chunks, 1):
            parcels: List[str] = []
            for name in targets:
                parcels += self.reg.parcels_owned(self.reg.id_of_name[name])
            up = build_acquirer_prompt_v9b(self.reg, step, self.n_steps, targets,
                                           self.offers, listed_rows, parcels,
                                           n, len(chunks),
                                           with_reason=self.acquirer_reason,
                                           undelivered=self.undelivered)
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
                text = text[:MAX_OFFER_CHARS]
                reason = str(row.get("reason", "") or "").strip()[:MAX_REASON_CHARS]
                # ①区画・種別の門番（v9 と同じ）
                if not self.reg.can_offer(aid, parcel, kind):
                    self.acquirer_invalid_offer += 1
                    self._record_undelivered(step, to, parcel, kind, text,
                                             reason, UNDELIVERED_RIGHTS)
                    continue
                # ②約束の門番（v9b で足したもの・条件文にだけ掛かる）
                if undeliverable_promise(text):
                    self.acquirer_undeliverable_promise += 1
                    self._record_undelivered(step, to, parcel, kind, text,
                                             reason, UNDELIVERED_PROMISE)
                    continue
                from .field_v9 import delivered_offer_v9
                sent[aid] = {"parcel": parcel, "kind": kind, "text": text,
                             "delivered": delivered_offer_v9(text),
                             "reason": reason}
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

    def _record_undelivered(self, step: int, to: str, parcel: str, kind: str,
                            text: str, reason: str, why: str) -> None:
        """配られなかった提示を全件そのまま残す（分類も判定もしない）。"""
        self.undelivered.append({
            "step": step, "to": to, "parcel": parcel, "kind": kind,
            "text": text, "reason": reason, "why": why,
            "words": promise_word_hits(text)})

    # -- 書き出し（v9 の内容＋ undelivered.json） ----------------------------------

    def _checkpoint(self, step: int) -> None:
        super()._checkpoint(step)
        tmp = os.path.join(self.checkpoint_dir, "undelivered.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.undelivered, f, ensure_ascii=False, indent=2)
        os.replace(tmp, os.path.join(self.checkpoint_dir, "undelivered.json"))
        u_step = sum(1 for u in self.undelivered if u["step"] == step)
        print(f"[v9b] m{step:02d}/{self.n_steps} 届かなかった提示 {u_step}"
              f"（累計 {len(self.undelivered)}）", flush=True)

    def _finalize(self, elapsed: float) -> Dict[str, Any]:
        summary = super()._finalize(elapsed)
        summary["scenario_version"] = "field_v9b"
        summary["acquirer_undeliverable_promise"] = \
            self.acquirer_undeliverable_promise
        summary["undelivered_total"] = len(self.undelivered)
        summary["undelivered_by_reason"] = {
            UNDELIVERED_RIGHTS: sum(1 for u in self.undelivered
                                    if u["why"] == UNDELIVERED_RIGHTS),
            UNDELIVERED_PROMISE: sum(1 for u in self.undelivered
                                     if u["why"] == UNDELIVERED_PROMISE)}
        with open(os.path.join(self.run_dir, "summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.run_dir, "undelivered.json"), "w",
                  encoding="utf-8") as f:
            json.dump(self.undelivered, f, ensure_ascii=False, indent=2)
        return summary


__all__ = ["SimulationV9B", "MockV9BClient"]
