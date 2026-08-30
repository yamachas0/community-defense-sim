"""v8e の試験（準備だけ・実APIでは走らせない）。

固定するのは「**海外→市外の置換と、X社の前置きに足した出自の事実1つ以外に差が無い**」
ことだけである（施主 2026-08-30 01:37 の準備指示）。
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.field_v8d import (build_acquirer_prefix_v8d,  # noqa: E402
                           build_acquirer_prompt_v8d, build_common_prefix_v8d,
                           build_decide_prompt_v8d, build_plan_prompt_v8d,
                           load_personas_v8, RegistryV8, SELL_NO, SELL_YES)
from src.field_v8e import (ACQUIRER_INTRO_CITY_OUTSIDE,  # noqa: E402
                           ACQUIRER_INTRO_OVERSEAS, ACQUIRER_ORIGIN_FACT_V8E,
                           build_acquirer_prefix_v8e, build_acquirer_prompt_v8e,
                           delivered_offer_v8e, intro_of)
from src.sim_v8e import SimulationV8E  # noqa: E402

PERSONAS = os.path.join(ROOT, "configs", "personas_v8.yaml")
CFG = os.path.join(ROOT, "configs", "config_field_v8e.yaml")
CFG_V8D = os.path.join(ROOT, "configs", "config_field_v8d.yaml")


@pytest.fixture(scope="module")
def agents():
    return load_personas_v8(PERSONAS)


@pytest.fixture(scope="module")
def cfg():
    with open(CFG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run(tmp_path, steps=3, **over):
    with open(CFG, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    c["steps"] = steps
    c["llm"] = {**c["llm"], "provider": "mock"}
    c.update(over)
    sim = SimulationV8E(c, str(tmp_path))
    return sim, sim.run()


def test_intro_line_is_the_only_word_change(cfg):
    assert ACQUIRER_INTRO_OVERSEAS == "私どもは海外の不動産投資会社です。"
    assert ACQUIRER_INTRO_CITY_OUTSIDE == "私どもは市外の不動産投資会社です。"
    assert intro_of(cfg) == ACQUIRER_INTRO_CITY_OUTSIDE
    assert (ACQUIRER_INTRO_CITY_OUTSIDE.replace("市外", "海外")
            == ACQUIRER_INTRO_OVERSEAS)
    assert delivered_offer_v8e("譲り受けたい。", ACQUIRER_INTRO_CITY_OUTSIDE) == (
        "私どもは市外の不動産投資会社です。譲り受けたい。")
    assert delivered_offer_v8e("", ACQUIRER_INTRO_CITY_OUTSIDE) == ""


def test_acquirer_prefix_adds_only_the_origin_fact(agents, cfg):
    x = build_acquirer_prefix_v8e(cfg, agents)
    base = build_acquirer_prefix_v8d(cfg, agents)
    assert x.replace(ACQUIRER_ORIGIN_FACT_V8E, "") == base
    assert "あなたの実体は海外の投資会社である。" in x
    assert "問われたときに限り、そのことを text に書いてよい。" in x
    # 当為（毎月書け・書くな）は足さない
    for word in ("必ず", "書け", "書くな", "隠せ", "偽", "せよ"):
        assert word not in ACQUIRER_ORIGIN_FACT_V8E


def test_acquirer_prompt_only_swaps_the_intro(agents, cfg):
    reg = RegistryV8(agents)
    a = build_acquirer_prompt_v8d(reg, 1, 36, [agents[0]["name"]], [], [], 1, 1)
    b = build_acquirer_prompt_v8e(reg, 1, 36, [agents[0]["name"]], [], [], 1, 1,
                                  intro=ACQUIRER_INTRO_CITY_OUTSIDE)
    assert b == a.replace(ACQUIRER_INTRO_OVERSEAS, ACQUIRER_INTRO_CITY_OUTSIDE)
    assert "海外" not in b and "市外" in b
    assert "text にはこの1行を書かない" in b
    # 住民の原文（断りの一言）に同じ語が出ても、そこは書き換えない
    offers = [{"step": 1, "to": agents[0]["name"], "text": "譲ってほしい。",
               "result": "売らなかった",
               "decline_reason": "海外の投資会社だと聞いたため"}]
    c = build_acquirer_prompt_v8e(reg, 2, 36, [agents[0]["name"]], offers, [], 1, 1,
                                  intro=ACQUIRER_INTRO_CITY_OUTSIDE)
    assert "相手の一言:「海外の投資会社だと聞いたため」" in c
    assert ACQUIRER_INTRO_CITY_OUTSIDE in c
    assert ACQUIRER_INTRO_OVERSEAS not in c


def test_resident_side_is_untouched(agents, cfg):
    """住民のプロンプトは v8d と1文字も違わない（自己紹介は届いた提示にだけ乗る）。"""
    reg = RegistryV8(agents)
    labels = [str(v["label"]) for v in cfg["social"]["venues"]]
    assert build_common_prefix_v8d(cfg, agents) == build_common_prefix_v8d(
        {**cfg, "acquirer_intro_mode": "overseas"}, agents)
    text = build_common_prefix_v8d(cfg, agents)
    for word in ("海外", "市外", "投資会社"):
        assert word not in text
    plan = build_plan_prompt_v8d(agents[0], reg, 1, 36, labels, "", None)
    dec = build_decide_prompt_v8d(agents[0], reg, 1, 36, "", "条件", [],
                                  sell_order_=[SELL_NO, SELL_YES])
    for t in (plan, dec):
        assert "海外" not in t and "実体" not in t


def test_overseas_mode_is_byte_identical_to_v8d(agents, cfg):
    over = {**cfg, "acquirer_intro_mode": "overseas"}
    reg = RegistryV8(agents)
    assert build_acquirer_prefix_v8e(over, agents) == build_acquirer_prefix_v8d(
        over, agents)
    assert build_acquirer_prompt_v8e(reg, 1, 36, [agents[0]["name"]], [], [], 1, 1,
                                     intro=ACQUIRER_INTRO_OVERSEAS) == \
        build_acquirer_prompt_v8d(reg, 1, 36, [agents[0]["name"]], [], [], 1, 1)


def test_mock_run_delivers_the_city_outside_line(tmp_path):
    sim, s = _run(tmp_path, steps=3)
    assert s["scenario_version"] == "field_v8e"
    assert s["acquirer_intro_mode"] == "city_outside"
    assert sim.offers
    for o in sim.offers:
        assert o["delivered"] == ACQUIRER_INTRO_CITY_OUTSIDE + o["text"]
        assert not o["text"].startswith(ACQUIRER_INTRO_CITY_OUTSIDE)
    told = [d for d in sim.decisions if d["offer"]]
    assert told and all(d["offer"].startswith(ACQUIRER_INTRO_CITY_OUTSIDE)
                        for d in told)
    assert s["months_run"] == 3 and s["parse_fail"] == 0
    assert s["checkpoints_written"] == 3
    ap = open(os.path.join(str(tmp_path), "acquirer_prefix.txt"),
              encoding="utf-8").read()
    assert "あなたの実体は海外の投資会社である。" in ap


def test_mock_run_in_overseas_mode_matches_v8d_shape(tmp_path):
    sim, s = _run(tmp_path, steps=2, acquirer_intro_mode="overseas")
    assert s["acquirer_intro_mode"] == "overseas"
    for o in sim.offers:
        assert o["delivered"] == ACQUIRER_INTRO_OVERSEAS + o["text"]


def test_config_differs_from_v8d_only_where_declared():
    with open(CFG, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    with open(CFG_V8D, encoding="utf-8") as f:
        b = yaml.safe_load(f)
    assert c["scenario_version"] == "field_v8e"
    assert c["acquirer_intro_mode"] == "city_outside"
    assert c["seed"] == b["seed"] == 85 and c["steps"] == b["steps"] == 36
    assert c["world"] == b["world"] and c["social"] == b["social"]
    assert c["llm"] == b["llm"]
    # 上限は施主決定 $1（1本あたり）。最悪の超過を足しても $1 を割らない。
    assert c["max_cost_usd"] == 0.94 and c["max_cost_usd"] + 0.034 < 1.0
    for k in ("run_name", "scenario_version", "acquirer_intro_mode",
              "max_cost_usd"):
        c.pop(k, None), b.pop(k, None)
    assert c == b, "宣言していない差が config に入っている"


def test_v8d_files_are_untouched(agents, cfg):
    from src.field_v8d import ACQUIRER_FACTS_V8D
    assert "第1月の開始時点で、あなたは A市に不動産を持っていない。" in ACQUIRER_FACTS_V8D
    x = build_acquirer_prefix_v8d(cfg, agents)
    assert ACQUIRER_INTRO_OVERSEAS in build_acquirer_prompt_v8d(
        RegistryV8(agents), 1, 36, [agents[0]["name"]], [], [], 1, 1)
    assert "あなたの実体" not in x
