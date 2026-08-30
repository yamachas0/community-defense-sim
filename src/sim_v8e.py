"""v8e の月ループ（v8d からの差分だけ）。

設計＝`src/field_v8e.py` の docstring。**v1〜v8d には一切触らない**
（`SimulationV8D` を継承して、自己紹介の1行と出自の事実だけ差し替える）。

差分は **X社まわりの文字列2つだけ** で、月の進み方・住民のプロンプト・判定・記録は
v8d と1文字も違わない。`acquirer_intro_mode: overseas` にすると v8d と完全に同じ文面になる。
"""

from __future__ import annotations

import threading
from typing import Any, Dict

from .field_v8e import (ACQUIRER_INTRO_CITY_OUTSIDE,  # noqa: F401
                        ACQUIRER_INTRO_OVERSEAS, ACQUIRER_ORIGIN_FACT_V8E,
                        build_acquirer_prefix_v8e, build_acquirer_prompt_v8e,
                        delivered_offer_v8e, intro_mode, intro_of)
from .sim_v8d import (MockV8DClient, SimulationV8D,  # noqa: F401
                      TimeoutGeminiClient)


# 文面差し替えをプロセス内で直列化する（1プロセス1走行が前提・保険）。
_PATCH_LOCK = threading.Lock()


class MockV8EClient(MockV8DClient):
    """v8e 用の mock（v8d と同じ・別名で置くだけ）。"""


class SimulationV8E(SimulationV8D):

    def __init__(self, cfg: Dict[str, Any], run_dir: str):
        if cfg.get("scenario_version") != "field_v8e":
            raise ValueError("config の scenario_version が field_v8e ではない")
        super().__init__({**cfg, "scenario_version": "field_v8d"}, run_dir)
        self.cfg = cfg
        self.intro_mode = intro_mode(cfg)
        self.intro = intro_of(cfg)
        # X社の前置きだけ差し替える（住民の共通前置きは v8d のまま＝1文字も変えない）
        self.acquirer_prefix = build_acquirer_prefix_v8e(cfg, self.agents)
        if str(cfg["llm"].get("provider", "mock")).lower() == "mock":
            self.client = MockV8EClient(seed=self.seed, usage=self.usage)

    def _acquirer_turn(self, step: int) -> Dict[str, str]:
        """X社の月次。**自己紹介の1行の文字列だけ**を差し替えて親を使う。

        差し替えはモジュール属性なので、**同じプロセスで複数の走行を同時に回さない**
        （Codex レビュー 2026-08-30 の指摘）。1プロセス1走行が前提で、
        月内の並列（スレッドプール）はこの差し替えの内側で完結する。念のため
        プロセス内のロックで直列化する。
        """
        import src.sim_v8c as _v8c
        prompt_orig = _v8c.build_acquirer_prompt_v8c
        deliver_orig = _v8c.delivered_offer_v8c
        intro = self.intro

        def _prompt(*a, **k):
            return build_acquirer_prompt_v8e(*a, intro=intro, **k)

        def _deliver(text):
            return delivered_offer_v8e(text, intro)

        with _PATCH_LOCK:
            _v8c.build_acquirer_prompt_v8c = _prompt
            _v8c.delivered_offer_v8c = _deliver
            try:
                return super()._acquirer_turn(step)
            finally:
                _v8c.build_acquirer_prompt_v8c = prompt_orig
                _v8c.delivered_offer_v8c = deliver_orig

    def _finalize(self, elapsed: float) -> Dict[str, Any]:
        summary = super()._finalize(elapsed)
        summary["scenario_version"] = "field_v8e"
        summary["acquirer_intro_mode"] = self.intro_mode
        summary["acquirer_intro"] = self.intro
        import json
        import os
        with open(os.path.join(self.run_dir, "summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary


__all__ = ["SimulationV8E", "MockV8EClient"]
