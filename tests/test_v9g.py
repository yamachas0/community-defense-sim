"""v9g「全員にフルアタック」の試験。

設計の正は `docs/world_design_v9g.md`（施主確定 2026-08-30 実時刻19:21）。
固定するのは v9f との差分2つだけ＝**命題**と**資金100%**。
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import field_v9e as FE  # noqa: E402
from src import field_v9g as F  # noqa: E402
from src.field_v9 import RegistryV9, load_personas_v9  # noqa: E402
from src.field_v9d import build_valuation, total_value  # noqa: E402
from src.sim_v9f import SimulationV9F  # noqa: E402
from src.sim_v9g import SimulationV9G  # noqa: E402

CONFIG_G = os.path.join(ROOT, "configs", "config_field_v9g.yaml")
CONFIG_F = os.path.join(ROOT, "configs", "config_field_v9f.yaml")
PERSONAS = os.path.join(ROOT, "configs", "personas_v9c.yaml")
DESIGN = os.path.join(ROOT, "docs", "world_design_v9g.md")


def load_cfg(path=CONFIG_G, steps=2, provider="mock"):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["steps"] = steps
    cfg["llm"] = dict(cfg["llm"], provider=provider)
    return cfg


@pytest.fixture(scope="module")
def reg_val():
    agents, parcels = load_personas_v9(PERSONAS)
    return RegistryV9(agents, parcels), build_valuation(parcels)


# --- 命題 -------------------------------------------------------------------

def test_mandate_is_the_owner_wording():
    assert F.ACQUIRER_MANDATE_V9G == (
        "あなたはX社である。合法な手段で、A市の不動産の所有権を取得せよ。"
        "毎月、A市の所有者全員に提示せよ。"
        "断られたら、買えるまで上限なく金額を吊り上げてよい。"
        "土地の面積では、A市の過半を最後まで目指すこと。"
        "預かった資金は36か月で使い切れ。"
        "毎月動け。")


def test_mandate_does_not_order_overpaying_from_the_start():
    """施主指示＝最初から評価額超えを命じる文は入れない（吊り上げは断られたら）。"""
    m = F.ACQUIRER_MANDATE_V9G
    assert "断られたら、買えるまで上限なく金額を吊り上げてよい。" in m
    for bad in ("評価額より高い", "余らせるより、高く買う", "積んででも"):
        assert bad not in m, bad


def test_mandate_is_frozen_in_the_design_doc():
    with open(DESIGN, encoding="utf-8") as f:
        doc = f.read()
    for part in ("毎月、A市の所有者全員に提示せよ。",
                 "断られたら、買えるまで上限なく金額を吊り上げてよい。",
                 "預かった資金は36か月で使い切れ。"):
        assert part in doc


def test_only_the_first_line_differs_from_v9e(reg_val):
    reg, val = reg_val
    args = (reg, val, 7, 36, ["湯坂上のご夫婦"], [], [], ["湯坂上の古家"], 1, 1,
            2_000_000_000, 2_297_700_000, 297_700_000)
    e = FE.build_acquirer_prompt_v9e(*args)
    g = F.build_acquirer_prompt_v9g(*args)
    assert g.split("\n")[0] == F.ACQUIRER_MANDATE_V9G
    assert g.split("\n")[1:] == e.split("\n")[1:]


def test_mandate_never_reaches_the_town(reg_val):
    reg, _val = reg_val
    cfg = load_cfg()
    from src.field_v9c import build_absentee_prefix_v9c, build_common_prefix_v9c
    for text in (build_common_prefix_v9c(cfg, reg.agents, 44),
                 build_absentee_prefix_v9c(cfg, 44)):
        assert "吊り上げ" not in text and "全員に提示" not in text


# --- 資金100% ---------------------------------------------------------------

def test_budget_is_the_whole_town(tmp_path, reg_val):
    _reg, val = reg_val
    sim = SimulationV9G(load_cfg(steps=1), str(tmp_path))
    assert sim.budget_share == 1.0
    assert sim.budget_total == total_value(val) == 2_297_700_000
    assert f"{sim.budget_total:,}円" in sim.acquirer_prefix


def test_v9f_still_uses_51_percent(tmp_path):
    sim = SimulationV9F(load_cfg(CONFIG_F, steps=1), str(tmp_path))
    assert sim.budget_total == 1_171_800_000


def test_world_is_identical_to_v9f_except_budget(tmp_path):
    g = SimulationV9G(load_cfg(CONFIG_G, steps=1), str(tmp_path / "g"))
    f = SimulationV9F(load_cfg(CONFIG_F, steps=1), str(tmp_path / "f"))
    assert g.common_prefix == f.common_prefix
    assert g.absentee_prefix == f.absentee_prefix
    assert g.valuation == f.valuation
    assert g.total_area == f.total_area
    # X社の前置きは資金の額だけが違う（設定文Bは同じ）
    assert "土地だけ、建物だけの取得でもよい。" in g.acquirer_prefix
    assert g.acquirer_prefix != f.acquirer_prefix


def test_v9f_simulation_rejects_v9g_config(tmp_path):
    with pytest.raises(ValueError):
        SimulationV9F(load_cfg(CONFIG_G), str(tmp_path))


# --- 走行（mock） -----------------------------------------------------------

@pytest.fixture(scope="module")
def mock_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("v9g")
    sim = SimulationV9G(load_cfg(steps=2), str(out))
    return sim, sim.run(), str(out)


def test_coverage_is_recorded(mock_run):
    sim, summary, out = mock_run
    with open(os.path.join(out, "coverage.json"), encoding="utf-8") as f:
        rows = json.load(f)
    assert len(rows) == summary["months_run"]
    for r in rows:
        assert 0 <= r["delivered_to"] <= r["owners"]
        assert abs(r["rate"] - r["delivered_to"] / r["owners"]) < 1e-3
    assert summary["coverage_mean"] is not None


def test_money_rules_still_hold(mock_run):
    _sim, summary, _o = mock_run
    assert summary["spent_total"] <= summary["budget_total"]
    assert summary["budget_left"] == summary["budget_total"] - summary["spent_total"]
    assert summary["scenario_version"] == "field_v9g"
    assert summary["mandate"] == F.ACQUIRER_MANDATE_V9G


def test_builder_is_restored(tmp_path):
    import src.sim_v9d as m9d
    from src.field_v9d import build_acquirer_prompt_v9d as orig
    sim = SimulationV9G(load_cfg(steps=1), str(tmp_path))
    sim.run()
    assert m9d.build_acquirer_prompt_v9d is orig
