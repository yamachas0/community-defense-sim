"""v9e「過半を目指せと言われた買い手」の月ループ — v9d からの差分だけ。

設計の正は `docs/world_design_v9e.md`。
**v9d までのファイルには一切触らない。** 世界は v9d と1文字も違わない。
違うのは X社 に渡す user プロンプトだけである:

  1. 命題が施主の新しい文（評価額と面積の両方で過半・資金は使い切る前提）
  2. 登記簿に**区画の面積**が載る（評価額と同じく公開されている事実）
  3. **いまの取得割合**（評価額・土地の面積）が事実として毎月返る

町の人・名簿・門番・会話・出品・資金51%・月内の提示合計の規則は v9d のまま。
そのため月次処理は v9d の `_acquirer_turn` を**そのまま**呼び、
プロンプトを組み立てる関数だけを v9e のものに差し替える（配線の重複を作らない）。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from . import sim_v9d as _m9d
from .field_v9e import (ACQUIRER_MANDATE_V9E, acquired_area,  # noqa: F401
                        acquired_value, build_acquirer_prompt_v9e,
                        total_area)
from .field_v9d import total_value
from .sim_v9d import MockV9DClient, SimulationV9D  # noqa: F401

logger = logging.getLogger(__name__)


class SimulationV9E(SimulationV9D):

    def __init__(self, cfg: Dict[str, Any], run_dir: str):
        if cfg.get("scenario_version") != "field_v9e":
            raise ValueError("config の scenario_version が field_v9e ではない")
        inner = dict(cfg)
        inner["scenario_version"] = "field_v9d"
        super().__init__(inner, run_dir)
        self.cfg["scenario_version"] = "field_v9e"
        self.total_area = total_area(self.valuation)
        self.progress: List[Dict[str, Any]] = []

    # -- X社の月次（v9d の配線そのまま・プロンプトだけ v9e） ---------------------

    def _acquirer_turn(self, step: int) -> Dict[str, Dict[str, Any]]:
        orig = _m9d.build_acquirer_prompt_v9d
        _m9d.build_acquirer_prompt_v9d = build_acquirer_prompt_v9e
        try:
            return super()._acquirer_turn(step)
        finally:
            _m9d.build_acquirer_prompt_v9d = orig

    def _step(self, step: int) -> None:
        super()._step(step)
        av = acquired_value(self.reg, self.valuation)
        aa = acquired_area(self.reg, self.valuation)
        row = {"step": step, "acquired_value": av, "acquired_area": aa,
               "value_share": round(av / self.total_value, 4),
               "area_share": round(aa / self.total_area, 4),
               "spent_cum": self.spent, "budget_left": self.budget_left,
               "budget_used_share": round(self.spent / self.budget_total, 4)}
        self.progress.append(row)
        m = self.monthly[-1] if self.monthly else None
        if m is not None and m.get("step") == step:
            m.update({k: row[k] for k in ("acquired_value", "acquired_area",
                                          "value_share", "area_share",
                                          "budget_used_share")})

    def _finalize(self, elapsed: float) -> Dict[str, Any]:
        summary = super()._finalize(elapsed)
        av = acquired_value(self.reg, self.valuation)
        aa = acquired_area(self.reg, self.valuation)
        summary["scenario_version"] = "field_v9e"
        summary["mandate"] = ACQUIRER_MANDATE_V9E
        summary["total_area_m2"] = self.total_area
        summary["acquired_value_end"] = av
        summary["acquired_area_m2_end"] = aa
        summary["value_share_end"] = round(av / self.total_value, 4)
        summary["area_share_end"] = round(aa / self.total_area, 4)
        summary["budget_used_share"] = round(self.spent / self.budget_total, 4)
        summary["reached_majority_value"] = av * 2 > self.total_value
        summary["reached_majority_area"] = aa * 2 > self.total_area
        with open(os.path.join(self.run_dir, "progress.json"), "w",
                  encoding="utf-8") as f:
            json.dump(self.progress, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.run_dir, "summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary


__all__ = ["SimulationV9E"]
