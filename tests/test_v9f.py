"""v9f「積んででも買えと言われた買い手」の試験。

設計の正は `docs/world_design_v9f.md`（施主確定 2026-08-30 実時刻17:02）。
固定するのは v9e との差分だけ＝**命題の行だけが違うこと**。
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
from src import field_v9f as F  # noqa: E402
from src.field_v9 import RegistryV9, load_personas_v9  # noqa: E402
from src.field_v9d import build_valuation  # noqa: E402
from src.sim_v9e import SimulationV9E  # noqa: E402
from src.sim_v9f import SimulationV9F  # noqa: E402

CONFIG_F = os.path.join(ROOT, "configs", "config_field_v9f.yaml")
CONFIG_E = os.path.join(ROOT, "configs", "config_field_v9e.yaml")
PERSONAS = os.path.join(ROOT, "configs", "personas_v9c.yaml")
DESIGN = os.path.join(ROOT, "docs", "world_design_v9f.md")


def load_cfg(path=CONFIG_F, steps=3, provider="mock"):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["steps"] = steps
    cfg["llm"] = dict(cfg["llm"], provider=provider)
    return cfg


@pytest.fixture(scope="module")
def reg_val():
    agents, parcels = load_personas_v9(PERSONAS)
    return RegistryV9(agents, parcels), build_valuation(parcels)


def test_mandate_is_the_owner_wording():
    assert F.ACQUIRER_MANDATE_V9F == (
        "あなたはX社である。合法な手段で、A市の不動産の所有権を取得せよ。"
        "資金の範囲で、できるだけ多くの評価額と面積を取得すること。"
        "土地の面積では、A市の過半を最後まで目指すこと。"
        "買えるかどうかは相手次第だが、買えるまで、金額を含む条件を変えて働きかけ続けること。"
        "預かった資金は36か月で使い切ることが前提であり、余らせるより、高く買うことを選べ。"
        "毎月動け。")


def test_mandate_has_no_impossible_target():
    """17:02 の初案（過半＋積んででも買え）は算数上不可能だったので使わない。"""
    m = F.ACQUIRER_MANDATE_V9F
    assert "評価額と面積の両方で過半" not in m      # 算数上不可能な目標は与えない
    assert "土地の面積では、A市の過半" in m          # 面積の過半だけは目標に残す
    assert "できるだけ多くの評価額と面積" in m
    assert "買えるまで" in m and "余らせるより、高く買うことを選べ" in m


def test_mandate_is_frozen_in_the_design_doc():
    with open(DESIGN, encoding="utf-8") as f:
        doc = f.read()
    for part in ("資金の範囲で、できるだけ多くの評価額と面積を取得すること。",
                 "土地の面積では、A市の過半を最後まで目指すこと。",
                 "買えるまで、金額を含む条件を変えて働きかけ続けること。",
                 "余らせるより、高く買うことを選べ。"):
        assert part in doc


def test_mandate_has_no_number_or_target_parcel():
    m = F.ACQUIRER_MANDATE_V9F
    for bad in ("倍", "%", "％", "円", "ホテル", "旅館", "誰から", "区画を"):
        assert bad not in m, bad


def test_only_the_first_line_differs_from_v9e(reg_val):
    reg, val = reg_val
    args = (reg, val, 7, 36, ["湯坂上のご夫婦"], [], [], ["湯坂上の古家"], 1, 1,
            1_000_000_000, 1_171_800_000, 171_800_000)
    e = FE.build_acquirer_prompt_v9e(*args)
    f = F.build_acquirer_prompt_v9f(*args)
    assert f.split("\n")[0] == F.ACQUIRER_MANDATE_V9F
    assert e.split("\n")[0] == FE.ACQUIRER_MANDATE_V9E
    assert f.split("\n")[1:] == e.split("\n")[1:]      # 2行目以降は完全一致


def test_mandate_never_reaches_the_town(reg_val):
    reg, _val = reg_val
    cfg = load_cfg()
    from src.field_v9c import build_absentee_prefix_v9c, build_common_prefix_v9c
    from src.field_v9d import build_acquirer_prefix_v9d
    for text in (build_common_prefix_v9c(cfg, reg.agents, 44),
                 build_absentee_prefix_v9c(cfg, 44),
                 build_acquirer_prefix_v9d(cfg, reg, 1)):
        assert "積んで" not in text
        assert F.ACQUIRER_MANDATE_V9F not in text


def test_world_is_identical_to_v9e(tmp_path):
    f = SimulationV9F(load_cfg(CONFIG_F, steps=1), str(tmp_path / "f"))
    e = SimulationV9E(load_cfg(CONFIG_E, steps=1), str(tmp_path / "e"))
    assert f.common_prefix == e.common_prefix
    assert f.absentee_prefix == e.absentee_prefix
    assert f.acquirer_prefix == e.acquirer_prefix
    assert f.valuation == e.valuation
    assert f.budget_total == e.budget_total
    assert f.total_area == e.total_area


def test_v9e_simulation_rejects_v9f_config(tmp_path):
    with pytest.raises(ValueError):
        SimulationV9E(load_cfg(CONFIG_F), str(tmp_path))


def test_builder_is_restored(tmp_path):
    import src.sim_v9d as m9d
    from src.field_v9d import build_acquirer_prompt_v9d as orig
    sim = SimulationV9F(load_cfg(steps=1), str(tmp_path))
    sim.run()
    assert m9d.build_acquirer_prompt_v9d is orig


@pytest.fixture(scope="module")
def mock_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("v9f")
    sim = SimulationV9F(load_cfg(steps=3), str(out))
    return sim, sim.run(), str(out)


def test_reoffers_are_counted(mock_run):
    sim, summary, out = mock_run
    with open(os.path.join(out, "reoffers.json"), encoding="utf-8") as f:
        rows = json.load(f)
    assert len(rows) == summary["reoffers_total"]
    assert (summary["reoffers_amount_up"] + summary["reoffers_amount_down"]
            + summary["reoffers_amount_same"]) == len(rows)
    for r in rows:
        assert r["delta"] == r["amount"] - r["prev_amount"]


def test_piled_offers_are_counted(mock_run):
    sim, summary, _o = mock_run
    piled = [o for o in sim.offers if (o.get("ratio") or 0) >= 1.2]
    assert summary["piled_offers"] == len(piled)
    assert summary["piled_offers_accepted"] == len([o for o in piled
                                                    if o["accepted"]])
    if piled:
        assert abs(summary["piled_accept_rate"]
                   - summary["piled_offers_accepted"] / len(piled)) < 1e-6


def test_v9e_observations_still_there(mock_run):
    _sim, summary, _o = mock_run
    for k in ("value_share_end", "area_share_end", "reached_majority_value",
              "reached_majority_area", "budget_used_share", "mandate"):
        assert k in summary
    assert summary["scenario_version"] == "field_v9f"
    assert summary["mandate"] == F.ACQUIRER_MANDATE_V9F


def test_money_rules_still_hold(mock_run):
    _sim, summary, _o = mock_run
    assert summary["spent_total"] <= summary["budget_total"]
    assert summary["budget_left"] == summary["budget_total"] - summary["spent_total"]
