"""v9d「ものさしとしてのお金」の試験。

設計の正は `docs/world_design_v9d.md`（施主確定 2026-08-30 16:33）。
固定するのは v9c との差分だけ:

  - 評価額が凍結した表からしか作られない（決定論・44区画・公開）
  - X社の資金＝町の全評価額の51%。残額を超える提示は世界が配らない
  - 月内に配った提示の合計も資金を超えない（全部成約しても超過しない）
  - 売った金額が売った人の記録に入る（財布は記帳だけ）
  - 世界の文に「得／損／高い／安い／相場より」等の比べる語を書かない
  - v9c の骨格（借りて使う人の出口・一言・時間・門番）が生きている
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import field_v9d as F  # noqa: E402
from src.field_v9 import load_personas_v9  # noqa: E402
from src.sim_v9c import SimulationV9C  # noqa: E402
from src.sim_v9d import BUDGET_SHARE, MockV9DClient, SimulationV9D  # noqa: E402

CONFIG = os.path.join(ROOT, "configs", "config_field_v9d.yaml")
PERSONAS = os.path.join(ROOT, "configs", "personas_v9c.yaml")
# 「取得」は施主文言に含まれる語なので、比べる語としては単独形だけを見る
COMPARE_WORDS = ["お得", "得する", "損", "高い", "安い", "相場より", "有利",
                 "不利", "お買い得", "妥当", "割安", "割高", "値打ち"]


def load_cfg(steps=3, provider="mock"):
    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["steps"] = steps
    cfg["llm"] = dict(cfg["llm"], provider=provider)
    return cfg


@pytest.fixture(scope="module")
def book():
    return load_personas_v9(PERSONAS)


@pytest.fixture(scope="module")
def val(book):
    return F.build_valuation(book[1])


# --- 評価額 -----------------------------------------------------------------

def test_every_parcel_has_a_value(book, val):
    agents, parcels = book
    assert len(val) == len(parcels) == 44
    for p in parcels:
        row = val[p["name"]]
        assert row["land"] > 0
        if p["building"]:
            assert row["building"] > 0
        else:
            assert row["building"] == 0
        assert row["both"] == row["land"] + row["building"]


def test_valuation_is_deterministic(book):
    a = F.build_valuation(book[1])
    b = F.build_valuation(book[1])
    assert a == b


def test_valuation_uses_only_the_frozen_tables(book):
    assert set(F.PARCEL_USE) == {p["name"] for p in book[1]}
    assert set(F.PARCEL_USE.values()) <= set(F.USE_SPEC)
    assert set(F.DISTRICT_TIER.values()) <= set(F.LAND_TIER)


def test_land_tiers_are_five_and_ordered():
    v = list(F.LAND_TIER.values())
    assert len(F.LAND_TIER) == 5
    assert v == sorted(v, reverse=True)


def test_totals_match_the_design(val):
    assert F.total_value(val) == 2_297_700_000


# --- 資金 -------------------------------------------------------------------

@pytest.fixture(scope="module")
def mock_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("v9d")
    sim = SimulationV9D(load_cfg(steps=4), str(out))
    summary = sim.run()
    return sim, summary, str(out)


def test_budget_is_51_percent(mock_run):
    sim, summary, _o = mock_run
    assert BUDGET_SHARE == 0.51
    assert summary["budget_total"] == int(
        round(summary["valuation_total"] * 0.51 / 100_000) * 100_000)
    assert summary["budget_total"] > summary["valuation_total"] * 0.5


def test_spending_never_exceeds_the_budget(mock_run):
    sim, summary, _o = mock_run
    assert summary["spent_total"] <= summary["budget_total"]
    assert summary["budget_left"] == summary["budget_total"] - summary["spent_total"]
    assert summary["budget_left"] >= 0


def test_month_total_of_delivered_offers_fits_in_the_budget(tmp_path):
    """同じ月に配った提示が全部成約しても資金を超えない。"""
    sim = SimulationV9D(load_cfg(steps=2), str(tmp_path))
    sim.run()
    by_step = {}
    for o in sim.offers:
        by_step.setdefault(o["step"], []).append(int(o["amount"]))
    spent = 0
    for step in sorted(by_step):
        assert sum(by_step[step]) <= sim.budget_total - spent
        spent += sum(int(o["amount"]) for o in sim.offers
                     if o["step"] == step and o["accepted"])


def test_over_budget_offers_are_not_delivered(tmp_path):
    sim = SimulationV9D(load_cfg(steps=1), str(tmp_path))
    sim.budget_total = 1_000_000        # わざと足りなくする
    sim.run()
    assert sim.acquirer_over_budget > 0
    assert all(int(o["amount"]) <= 1_000_000 for o in sim.offers)
    rows = [u for u in sim.undelivered if u["why"] == F.UNDELIVERED_BUDGET]
    assert rows and all(int(r["amount"]) > 1_000_000 for r in rows)


# --- 財布 -------------------------------------------------------------------

def test_wallet_records_the_amount_paid(mock_run):
    sim, summary, out = mock_run
    with open(os.path.join(out, "payments.json"), encoding="utf-8") as f:
        pays = json.load(f)
    assert len(pays) == summary["paid_deals"]
    total = 0
    for p in pays:
        assert p["amount"] > 0
        assert p["valuation"] > 0
        total += p["amount"]
    assert total == summary["spent_total"]
    with open(os.path.join(out, "wallets.json"), encoding="utf-8") as f:
        w = json.load(f)
    assert sum(w.values()) == summary["spent_total"]


def test_wallets_start_at_zero(tmp_path):
    sim = SimulationV9D(load_cfg(steps=1), str(tmp_path))
    assert set(sim.wallet.values()) == {0}


def test_ratio_is_recorded_on_every_offer(mock_run):
    sim, _s, _o = mock_run
    for o in sim.offers:
        assert o["valuation"] > 0
        assert abs(o["ratio"] - o["amount"] / o["valuation"]) < 1e-3


# --- 世界の文 ---------------------------------------------------------------

def test_world_text_has_no_comparing_words(mock_run):
    sim, _s, _o = mock_run
    blobs = [sim.common_prefix, sim.absentee_prefix, sim.acquirer_prefix,
             F.MONEY_FACTS, F.ACQUIRER_MONEY_FACTS]
    for b in blobs:
        for w in COMPARE_WORDS:
            assert w not in b, w


def test_offer_amount_row_only_puts_two_numbers(val):
    row = F.offer_amount_row(val, "駅前通りの家", "土地", 12_000_000)
    assert "12,000,000円" in row and "評価額" in row
    for w in COMPARE_WORDS + ["より", "倍"]:
        assert w not in row


def test_money_facts_say_money_does_nothing_else():
    assert "賃料はない" in F.MONEY_FACTS
    assert "維持費も税もない" in F.MONEY_FACTS
    assert "お金で何かを買う仕組みもない" in F.MONEY_FACTS


def test_acquirer_sees_valuations_and_budget(mock_run):
    sim, _s, _o = mock_run
    blob = json.dumps(sim.acquirer_raw, ensure_ascii=False)
    assert "評価額" in sim.acquirer_prefix
    assert F.yen(sim.budget_total) in sim.acquirer_prefix
    assert isinstance(blob, str)


def test_residents_see_the_valuation_of_their_own_parcels(mock_run):
    sim, _s, _o = mock_run
    rows = F.money_block_for_owner(sim.valuation, sim.reg, "A",
                                   sim.reg.parcels_owned("A"), 0)
    assert rows[0] == "[公開されている評価額]"
    assert any("評価額" not in r and "円" in r for r in rows[1:])


# --- v9c の骨格が生きている --------------------------------------------------

def test_v9c_features_still_there(mock_run):
    _sim, summary, _o = mock_run
    for k in ("tenant_calls_total", "messages_total", "undelivered_total",
              "vacancy_notices_total"):
        assert k in summary
    assert summary["scenario_version"] == "field_v9d"


def test_v9c_simulation_rejects_v9d_config(tmp_path):
    with pytest.raises(ValueError):
        SimulationV9C(load_cfg(), str(tmp_path))


def test_config_keeps_the_v9c_roster():
    cfg = load_cfg()
    assert cfg["personas_file"] == "configs/personas_v9c.yaml"
    assert cfg["seed"] == 85 and cfg["llm"]["temperature"] == 0.75


# --- 走行前レビュー（2026-08-30）の指摘に対する回帰試験 ----------------------

def test_checkpoint_is_written_after_the_payment_is_recorded(tmp_path):
    """途中終了しても、その月の支払いがチェックポイントに入っている。"""
    sim = SimulationV9D(load_cfg(steps=3), str(tmp_path))
    summary = sim.run()
    cp = os.path.join(str(tmp_path), "checkpoint", "payments.json")
    with open(cp, encoding="utf-8") as f:
        rows = json.load(f)
    assert len(rows) == summary["paid_deals"] == len(sim.payments)
    assert sum(int(x["amount"]) for x in rows) == summary["spent_total"]
    with open(os.path.join(str(tmp_path), "checkpoint", "log.jsonl"),
              encoding="utf-8") as f:
        lines = [x for x in f.read().split("\n") if x.strip()]
    assert len(lines) == summary["months_run"]


def test_money_conservation_every_month(tmp_path):
    """資金の不変条件＝はじめの金額 ＝ 残り ＋ 支払った合計 ＝ 残り ＋ 財布の合計。"""
    sim = SimulationV9D(load_cfg(steps=3), str(tmp_path))
    summary = sim.run()
    assert summary["budget_total"] == summary["budget_left"] + summary["spent_total"]
    assert sum(sim.wallet.values()) == summary["spent_total"]
    for m in sim.monthly:
        assert m["budget_left"] == summary["budget_total"] - m["spent_cum"]
        assert m["budget_left"] >= 0
    cum = 0
    for m in sim.monthly:
        cum += m["paid_this_month"]
        assert cum == m["spent_cum"]
