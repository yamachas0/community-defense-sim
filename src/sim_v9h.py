"""v9h「目標を町に明言する買い手」の月ループ — v9f からの差分だけ。

設計の正は `docs/world_design_v9h.md`。v9g は不採用（ボツ）。
**v9g までのファイルには一切触らない。** 世界は v9f と1文字も違わない。
違うのは2つだけ:

  1. 町の人に届く手紙の冒頭に、**世界が必ず**「私どもは、この街の不動産の過半の取得を
     目指しています。」を添える（X社の自由文ではない・毎通・全員に同じ）。
  2. X社の設定文に、整合のための事実を1行足す（公言していること）。

命題・資金51%・門番3種・月内の提示合計の規則・町の人のプロンプトは v9f のまま。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

from . import sim_v9d as _m9d
from .field_v9h import (DECLARATION_V9H, build_acquirer_prefix_v9h,
                        delivered_offer_v9h)
from .sim_v9f import SimulationV9F

logger = logging.getLogger(__name__)


class SimulationV9H(SimulationV9F):

    def __init__(self, cfg: Dict[str, Any], run_dir: str):
        if cfg.get("scenario_version") != "field_v9h":
            raise ValueError("config の scenario_version が field_v9h ではない")
        inner = dict(cfg)
        inner["scenario_version"] = "field_v9f"
        super().__init__(inner, run_dir)
        self.cfg["scenario_version"] = "field_v9h"
        # X社の前置きに「公言している」の1行だけ足す（資金・設定文Bは v9f のまま）
        self.acquirer_prefix = build_acquirer_prefix_v9h(cfg, self.reg,
                                                         self.budget_total)

    def _acquirer_turn(self, step: int) -> Dict[str, Dict[str, Any]]:
        # 届く文の組み立てだけを v9h のものにする（v9f の月次処理はそのまま使う）。
        orig = _m9d.delivered_offer_v9
        _m9d.delivered_offer_v9 = delivered_offer_v9h
        try:
            return super()._acquirer_turn(step)
        finally:
            _m9d.delivered_offer_v9 = orig

    def _finalize(self, elapsed: float) -> Dict[str, Any]:
        summary = super()._finalize(elapsed)
        summary["scenario_version"] = "field_v9h"
        summary["declaration"] = DECLARATION_V9H
        delivered = [o for o in self.offers if o.get("delivered")]
        summary["offers_with_declaration"] = len(
            [o for o in delivered if DECLARATION_V9H in o["delivered"]])
        summary["offers_delivered_total"] = len(delivered)
        with open(os.path.join(self.run_dir, "summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary


__all__ = ["SimulationV9H"]
