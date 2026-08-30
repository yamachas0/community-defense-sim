"""v9h「目標を町に明言する買い手」の試験。

設計の正は `docs/world_design_v9h.md`（施主確定 2026-08-30 実時刻20:12）。
固定するのは v9f との差分2つだけ＝**明言の1行が全通に入ること**と、
**それ以外が v9f と1文字も違わないこと**。
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import field_v9h as F  # noqa: E402
from src.field_v9 import ACQUIRER_INTRO_V8C, delivered_offer_v9  # noqa: E402
from src.field_v9f import ACQUIRER_MANDATE_V9F  # noqa: E402
from src.sim_v9f import SimulationV9F  # noqa: E402
from src.sim_v9h import SimulationV9H  # noqa: E402

CONFIG_H = os.path.join(ROOT, "configs", "config_field_v9h.yaml")
CONFIG_F = os.path.join(ROOT, "configs", "config_field_v9f.yaml")
DESIGN = os.path.join(ROOT, "docs", "world_design_v9h.md")


def load_cfg(path=CONFIG_H, steps=2, provider="mock"):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["steps"] = steps
    cfg["llm"] = dict(cfg["llm"], provider=provider)
    return cfg


def test_declaration_is_the_owner_wording():
    assert F.DECLARATION_V9H == "私どもは、この街の不動産の過半の取得を目指しています。"


def test_declaration_is_frozen_in_the_design_doc():
    with open(DESIGN, encoding="utf-8") as f:
        assert F.DECLARATION_V9H in f.read()


def test_delivered_text_has_intro_then_declaration_then_offer():
    out = F.delivered_offer_v9h("湯坂上の古家を譲り受けたい。")
    assert out == (ACQUIRER_INTRO_V8C + F.DECLARATION_V9H
                   + "湯坂上の古家を譲り受けたい。")
    assert out.index(ACQUIRER_INTRO_V8C) < out.index(F.DECLARATION_V9H)
    assert out.index(F.DECLARATION_V9H) < out.index("湯坂上の古家を譲り受けたい。")
    # v9f（明言なし）との差は、この1行だけ
    assert out.replace(F.DECLARATION_V9H, "") == delivered_offer_v9(
        "湯坂上の古家を譲り受けたい。")


def test_empty_text_stays_empty():
    assert F.delivered_offer_v9h("") == ""
    assert F.delivered_offer_v9h(None) == ""


def test_acquirer_prefix_states_the_declaration(tmp_path):
    sim = SimulationV9H(load_cfg(steps=1), str(tmp_path))
    assert F.DECLARATION_V9H in sim.acquirer_prefix
    assert "あなたはこの目標を町に公言している。" in sim.acquirer_prefix


def test_town_prompts_are_untouched(tmp_path):
    h = SimulationV9H(load_cfg(CONFIG_H, steps=1), str(tmp_path / "h"))
    f = SimulationV9F(load_cfg(CONFIG_F, steps=1), str(tmp_path / "f"))
    assert h.common_prefix == f.common_prefix
    assert h.absentee_prefix == f.absentee_prefix
    assert h.valuation == f.valuation
    assert h.budget_total == f.budget_total == 1_171_800_000   # 資金は51%のまま
    # X社の前置きは明言の1行だけが違う
    assert h.acquirer_prefix.replace(F.ACQUIRER_DECLARATION_FACT_V9H, "") == \
        f.acquirer_prefix


def test_mandate_is_still_v9f(tmp_path):
    """命題は v9f のまま（v9g の「全員に提示」「上限なく吊り上げ」は使わない）。"""
    from src.field_v9f import build_acquirer_prompt_v9f
    from src.field_v9 import RegistryV9, load_personas_v9
    from src.field_v9d import build_valuation
    agents, parcels = load_personas_v9(
        os.path.join(ROOT, "configs", "personas_v9c.yaml"))
    reg = RegistryV9(agents, parcels)
    val = build_valuation(parcels)
    up = build_acquirer_prompt_v9f(reg, val, 7, 36, ["湯坂上のご夫婦"], [], [],
                                   ["湯坂上の古家"], 1, 1, 1_000_000_000,
                                   1_171_800_000, 171_800_000)
    assert up.split("\n")[0] == ACQUIRER_MANDATE_V9F
    assert "毎月、A市の所有者全員に提示せよ" not in up
    assert "上限なく金額を吊り上げてよい" not in up


def test_v9f_simulation_rejects_v9h_config(tmp_path):
    with pytest.raises(ValueError):
        SimulationV9F(load_cfg(CONFIG_H), str(tmp_path))


@pytest.fixture(scope="module")
def mock_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("v9h")
    sim = SimulationV9H(load_cfg(steps=2), str(out))
    return sim, sim.run(), str(out)


def test_every_delivered_offer_carries_the_declaration(mock_run):
    sim, summary, _o = mock_run
    assert summary["offers_delivered_total"] > 0
    assert summary["offers_with_declaration"] == summary["offers_delivered_total"]
    for o in sim.offers:
        assert o["delivered"].startswith(ACQUIRER_INTRO_V8C + F.DECLARATION_V9H)
        # X社が書いた条件文はそのまま残っている
        assert o["text"] in o["delivered"]


def test_declaration_is_not_in_the_town_side_world_text(mock_run):
    sim, _s, _o = mock_run
    assert F.DECLARATION_V9H not in sim.common_prefix
    assert F.DECLARATION_V9H not in sim.absentee_prefix


def test_builder_and_delivery_are_restored(tmp_path):
    import src.sim_v9d as m9d
    from src.field_v9 import delivered_offer_v9 as orig_del
    from src.field_v9d import build_acquirer_prompt_v9d as orig_build
    sim = SimulationV9H(load_cfg(steps=1), str(tmp_path))
    sim.run()
    assert m9d.delivered_offer_v9 is orig_del
    assert m9d.build_acquirer_prompt_v9d is orig_build
