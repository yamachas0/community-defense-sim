"""v9g「全員にフルアタック」の月ループ — v9f からの差分だけ。

設計の正は `docs/world_design_v9g.md`。
**v9f までのファイルには一切触らない。** 差は2点だけ:

  1. X社の資金＝**町の全評価額の100%**（`budget_share` を config から読む）
  2. 命題＝毎月全員に提示せよ／断られたら上限なく吊り上げてよい／面積の過半／使い切れ

月内に配る提示の合計が残額を超えない規則・門番3種・世界の側は v9f のまま。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from . import sim_v9d as _m9d
from .field_v9d import build_acquirer_prefix_v9d, total_value
from .field_v9g import ACQUIRER_MANDATE_V9G, build_acquirer_prompt_v9g
from .sim_v9d import SimulationV9D
from .sim_v9f import SimulationV9F

logger = logging.getLogger(__name__)


class SimulationV9G(SimulationV9F):

    def __init__(self, cfg: Dict[str, Any], run_dir: str):
        if cfg.get("scenario_version") != "field_v9g":
            raise ValueError("config の scenario_version が field_v9g ではない")
        inner = dict(cfg)
        inner["scenario_version"] = "field_v9f"
        super().__init__(inner, run_dir)
        self.cfg["scenario_version"] = "field_v9g"
        # 資金の割合を config から読む（v9d〜v9f は 0.51 固定だった）
        self.budget_share = float(cfg.get("budget_share", 0.51))
        self.budget_total = int(round(total_value(self.valuation)
                                      * self.budget_share / 100_000) * 100_000)
        self.acquirer_prefix = build_acquirer_prefix_v9d(cfg, self.reg,
                                                         self.budget_total)
        self.coverage: List[Dict[str, Any]] = []

    def _acquirer_turn(self, step: int) -> Dict[str, Dict[str, Any]]:
        orig = _m9d.build_acquirer_prompt_v9d
        _m9d.build_acquirer_prompt_v9d = build_acquirer_prompt_v9g
        try:
            sent = SimulationV9D._acquirer_turn(self, step)
        finally:
            _m9d.build_acquirer_prompt_v9d = orig
        # 網羅率＝その月に提示が届いた人 ÷ 提示を出せる相手（所有権を持つ売れる主体）
        risk = self.reg.risk_set()
        self.coverage.append({
            "step": step, "owners": len(risk), "delivered_to": len(sent),
            "rate": (round(len(sent) / len(risk), 4) if risk else None)})
        return sent

    def _finalize(self, elapsed: float) -> Dict[str, Any]:
        summary = super()._finalize(elapsed)
        summary["scenario_version"] = "field_v9g"
        summary["mandate"] = ACQUIRER_MANDATE_V9G
        summary["budget_share"] = self.budget_share
        rates = [c["rate"] for c in self.coverage if c["rate"] is not None]
        summary["coverage_mean"] = (round(sum(rates) / len(rates), 4)
                                    if rates else None)
        summary["coverage_full_months"] = len([c for c in self.coverage
                                               if c["rate"] == 1.0])
        with open(os.path.join(self.run_dir, "coverage.json"), "w",
                  encoding="utf-8") as f:
            json.dump(self.coverage, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.run_dir, "summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary


__all__ = ["SimulationV9G"]
