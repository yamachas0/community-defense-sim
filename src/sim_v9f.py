"""v9f「積んででも買えと言われた買い手」の月ループ — v9e からの差分だけ。

設計の正は `docs/world_design_v9f.md`。
**v9e までのファイルには一切触らない。** 世界は v9e と1文字も違わない。
違うのは X社 の user プロンプトの**先頭の命題の行だけ**である。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

from . import sim_v9d as _m9d
from .field_v9f import ACQUIRER_MANDATE_V9F, build_acquirer_prompt_v9f
from .sim_v9d import SimulationV9D
from .sim_v9e import SimulationV9E

logger = logging.getLogger(__name__)


class SimulationV9F(SimulationV9E):

    def __init__(self, cfg: Dict[str, Any], run_dir: str):
        if cfg.get("scenario_version") != "field_v9f":
            raise ValueError("config の scenario_version が field_v9f ではない")
        inner = dict(cfg)
        inner["scenario_version"] = "field_v9e"
        super().__init__(inner, run_dir)
        self.cfg["scenario_version"] = "field_v9f"

    def _acquirer_turn(self, step: int) -> Dict[str, Dict[str, Any]]:
        # v9d の月次処理をそのまま呼び、プロンプトだけ v9f のものにする
        # （v9e の差し替えは通さない＝命題の行が二重に置き換わらないように）。
        orig = _m9d.build_acquirer_prompt_v9d
        _m9d.build_acquirer_prompt_v9d = build_acquirer_prompt_v9f
        try:
            return SimulationV9D._acquirer_turn(self, step)
        finally:
            _m9d.build_acquirer_prompt_v9d = orig

    def _finalize(self, elapsed: float) -> Dict[str, Any]:
        summary = super()._finalize(elapsed)
        summary["scenario_version"] = "field_v9f"
        summary["mandate"] = ACQUIRER_MANDATE_V9F
        # 「積んだ」提示＝評価額の1.2倍以上（走行前に決めた線引き）
        piled = [o for o in self.offers if (o.get("ratio") or 0) >= 1.2]
        summary["piled_offers"] = len(piled)
        summary["piled_offers_accepted"] = len([o for o in piled if o["accepted"]])
        summary["piled_accept_rate"] = (
            round(len([o for o in piled if o["accepted"]]) / len(piled), 4)
            if piled else None)
        # 同じ相手・同じ区画・同じ種別への再提示で、金額をどう動かしたか
        seen: Dict[tuple, int] = {}
        ups, downs, same, deltas = 0, 0, 0, []
        rows = []
        for o in sorted(self.offers, key=lambda x: x["step"]):
            key = (o["to_id"], o["parcel"], o["kind"])
            prev = seen.get(key)
            amt = int(o["amount"])
            if prev is not None:
                d = amt - prev
                if d > 0:
                    ups += 1
                    deltas.append(d)
                elif d < 0:
                    downs += 1
                else:
                    same += 1
                rows.append({"step": o["step"], "to": o["to"],
                             "parcel": o["parcel"], "kind": o["kind"],
                             "prev_amount": prev, "amount": amt, "delta": d,
                             "ratio": o.get("ratio"), "accepted": o["accepted"]})
            seen[key] = amt
        ds = sorted(deltas)
        summary["reoffers_total"] = len(rows)
        summary["reoffers_amount_up"] = ups
        summary["reoffers_amount_down"] = downs
        summary["reoffers_amount_same"] = same
        summary["reoffer_up_median_yen"] = (ds[len(ds) // 2] if ds else None)
        summary["reoffer_up_max_yen"] = (ds[-1] if ds else None)
        summary["reoffers_accepted"] = len([r for r in rows if r["accepted"]])
        with open(os.path.join(self.run_dir, "reoffers.json"), "w",
                  encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.run_dir, "summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary


__all__ = ["SimulationV9F"]
