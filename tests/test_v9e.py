"""v9e「過半を目指せと言われた買い手」の試験。

設計の正は `docs/world_design_v9e.md`（施主確定 2026-08-30 19:32／19:34）。
固定するのは v9d との差分だけ:

  - 命題が施主の文のままであること（価格の指示を含まないこと）
  - 登記簿に面積が載ること・取得割合と残り月数の事実行があること
  - 取得割合の計算（評価額・土地の面積）が定義どおりであること
  - **世界の側（町の人の前置き・X社の設定文・門番・資金）が v9d と1文字も違わないこと**
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import field_v9d as FD  # noqa: E402
from src import field_v9e as F  # noqa: E402
from src.field_v9 import load_personas_v9, RegistryV9  # noqa: E402
from src.sim_v9d import SimulationV9D  # noqa: E402
from src.sim_v9e import SimulationV9E  # noqa: E402

CONFIG_E = os.path.join(ROOT, "configs", "config_field_v9e.yaml")
CONFIG_D = os.path.join(ROOT, "configs", "config_field_v9d.yaml")
PERSONAS = os.path.join(ROOT, "configs", "personas_v9c.yaml")
DESIGN = os.path.join(ROOT, "docs", "world_design_v9e.md")


def load_cfg(path=CONFIG_E, steps=3, provider="mock"):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["steps"] = steps
    cfg["llm"] = dict(cfg["llm"], provider=provider)
    return cfg


@pytest.fixture(scope="module")
def reg_val():
    agents, parcels = load_personas_v9(PERSONAS)
    return RegistryV9(agents, parcels), FD.build_valuation(parcels)


# --- 命題 -------------------------------------------------------------------

def test_mandate_is_the_owner_wording():
    assert F.ACQUIRER_MANDATE_V9E == (
        "あなたはX社である。合法な手段で、A市の不動産の所有権を取得せよ。"
        "評価額と面積の両方で過半を、最後まで目指すこと。"
        "預かった資金は使い切ってよい。"
        "同じやり方で進まないときは、次の手を打つこと。"
        "毎月動け。")


def test_mandate_has_no_price_instruction():
    """19:34 の差し替え＝価格の指示は書かない（目標と許可だけ）。"""
    m = F.ACQUIRER_MANDATE_V9E
    for bad in ("積ん", "上乗せ", "高く", "安く", "евро", "余らせる", "使い切ることが前提",
                "評価額の", "倍", "何円", "値上げ"):
        assert bad not in m, bad
    assert "過半" in m and "使い切ってよい" in m
    assert "次の手を打つこと" in m          # 中身は書かない（19:36 追加）


def test_mandate_is_frozen_in_the_design_doc():
    with open(DESIGN, encoding="utf-8") as f:
        doc = f.read()
    for part in ("評価額と面積の両方で過半を、最後まで目指すこと。",
                 "預かった資金は使い切ってよい。",
                 "同じやり方で進まないときは、次の手を打つこと。"):
        assert part in doc


def test_mandate_only_reaches_the_acquirer(reg_val):
    reg, val = reg_val
    cfg = load_cfg()
    from src.field_v9d import build_acquirer_prefix_v9d
    from src.field_v9c import build_common_prefix_v9c, build_absentee_prefix_v9c
    pre = build_acquirer_prefix_v9d(cfg, reg, 1)
    assert F.ACQUIRER_MANDATE_V9E not in pre        # system 側には置かない
    assert "過半" not in build_common_prefix_v9c(cfg, reg.agents, 44)
    assert "過半" not in build_absentee_prefix_v9c(cfg, 44)


# --- 面積と取得割合 ---------------------------------------------------------

def test_total_area_matches_the_frozen_table(reg_val):
    _reg, val = reg_val
    assert F.total_area(val) == 18960


def test_area_share_counts_land_ownership_only(reg_val):
    reg, val = reg_val
    assert F.acquired_area(reg, val) == 0
    p = "駅前通りの家"
    reg.building_of[p] = "X社"                       # 建物だけでは面積は増えない
    assert F.acquired_area(reg, val) == 0
    assert F.acquired_value(reg, val) == val[p]["building"]
    reg.land_of[p] = "X社"                           # 土地を持つと面積が増える
    assert F.acquired_area(reg, val) == val[p]["area"]
    reg.building_of[p] = None
    reg.land_of[p] = "B"


def test_progress_rows_are_facts_only(reg_val):
    reg, val = reg_val
    rows = F.progress_rows_v9e(reg, val, 16, 36)
    text = "\n".join(rows)
    assert "残りの月 … 21か月" in text
    assert "%" in text and "m2" in text
    for bad in ("急", "あと", "目標", "べき", "足りない", "遅れ", "ペース"):
        assert bad not in text, bad


def test_ledger_shows_area_and_value(reg_val):
    reg, val = reg_val
    rows = F.ledger_rows_with_value_and_area(reg, val)
    assert len(rows) == 44
    assert any("（面積 180m2）" in r for r in rows)
    assert all("評価額" in r for r in rows)


def test_prompt_has_mandate_first_and_the_new_facts(reg_val):
    reg, val = reg_val
    up = F.build_acquirer_prompt_v9e(reg, val, 7, 36, ["湯坂上のご夫婦"], [], [],
                                     ["湯坂上の古家"], 1, 1, 1_000_000_000,
                                     1_171_800_000, 171_800_000)
    assert up.split("\n")[0] == F.ACQUIRER_MANDATE_V9E
    assert "[いまあなたが持っている割合]" in up
    assert "残りの月 … 30か月" in up
    assert "面積 180m2" in up


# --- 世界の側が v9d と同じであること -----------------------------------------

def test_world_prompts_are_identical_to_v9d(tmp_path):
    e = SimulationV9E(load_cfg(CONFIG_E, steps=1), str(tmp_path / "e"))
    d = SimulationV9D(load_cfg(CONFIG_D, steps=1), str(tmp_path / "d"))
    assert e.common_prefix == d.common_prefix
    assert e.absentee_prefix == d.absentee_prefix
    assert e.acquirer_prefix == d.acquirer_prefix       # 設定文（B）も同じ
    assert e.valuation == d.valuation
    assert e.budget_total == d.budget_total
    assert e.total_value == d.total_value


def test_v9d_prompt_builder_is_restored_after_the_turn(tmp_path):
    """v9d の月次処理を借りているので、差し替えた関数を必ず戻す。"""
    import src.sim_v9d as m9d
    from src.field_v9d import build_acquirer_prompt_v9d as orig
    sim = SimulationV9E(load_cfg(steps=1), str(tmp_path))
    sim.run()
    assert m9d.build_acquirer_prompt_v9d is orig


def test_v9d_simulation_rejects_v9e_config(tmp_path):
    with pytest.raises(ValueError):
        SimulationV9D(load_cfg(CONFIG_E), str(tmp_path))


# --- 走行（mock） -----------------------------------------------------------

@pytest.fixture(scope="module")
def mock_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("v9e")
    sim = SimulationV9E(load_cfg(steps=3), str(out))
    return sim, sim.run(), str(out)


def test_progress_is_recorded_every_month(mock_run):
    sim, summary, out = mock_run
    with open(os.path.join(out, "progress.json"), encoding="utf-8") as f:
        rows = json.load(f)
    assert len(rows) == summary["months_run"] == len(sim.monthly)
    for r in rows:
        assert 0.0 <= r["value_share"] <= 1.0
        assert 0.0 <= r["area_share"] <= 1.0
        assert r["budget_left"] == summary["budget_total"] - r["spent_cum"]
    assert rows[-1]["value_share"] == summary["value_share_end"]
    assert rows[-1]["area_share"] == summary["area_share_end"]


def test_majority_flags_match_the_shares(mock_run):
    _sim, summary, _o = mock_run
    assert summary["reached_majority_value"] == (summary["value_share_end"] > 0.5)
    assert summary["reached_majority_area"] == (summary["area_share_end"] > 0.5)
    assert summary["scenario_version"] == "field_v9e"
    assert summary["mandate"] == F.ACQUIRER_MANDATE_V9E


def test_money_rules_still_hold(mock_run):
    _sim, summary, _o = mock_run
    assert summary["spent_total"] <= summary["budget_total"]
    assert summary["budget_left"] == summary["budget_total"] - summary["spent_total"]
    assert summary["acquirer_bad_amount"] == 0


# --- 走行前レビュー（2026-08-30）の推奨による回帰試験 ------------------------

def test_builder_is_restored_even_if_the_turn_raises(tmp_path):
    """親の月次処理が落ちても、差し替えた関数は必ず戻る。"""
    import src.sim_v9d as m9d
    from src.field_v9d import build_acquirer_prompt_v9d as orig
    from src.sim_v9d import SimulationV9D as _D

    sim = SimulationV9E(load_cfg(steps=1), str(tmp_path))

    def boom(self, step):
        raise RuntimeError("わざと落とす")

    saved = _D._acquirer_turn
    _D._acquirer_turn = boom
    try:
        with pytest.raises(RuntimeError):
            sim._acquirer_turn(1)
    finally:
        _D._acquirer_turn = saved
    assert m9d.build_acquirer_prompt_v9d is orig


def test_shares_agree_across_progress_monthly_and_summary(mock_run):
    sim, summary, _o = mock_run
    last_m = sim.monthly[-1]
    last_p = sim.progress[-1]
    assert last_m["value_share"] == last_p["value_share"] == summary["value_share_end"]
    assert last_m["area_share"] == last_p["area_share"] == summary["area_share_end"]
    assert last_m["acquired_value"] == summary["acquired_value_end"]
    assert last_m["acquired_area"] == summary["acquired_area_m2_end"]
