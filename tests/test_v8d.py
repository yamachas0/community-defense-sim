"""v8d の試験（新規のみ・既存 v1〜v8c には触らない）。

固定するのは設計 `docs/world_design_v8d.md` が凍結した約束だけである。
v8c との差は3つ（①「戻らない」の一文の削除 ②X社の前置きに事実3つ ③行き先の理由欄を外す）で、
**それ以外は1文字も違わない**ことを機械で確かめる。
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.field_v8c import (build_acquirer_prefix_v8c,  # noqa: E402
                           build_common_prefix_v8c, build_decide_prompt_v8c,
                           build_plan_prompt_v8c)
from src.field_v8d import (ACQUIRER_FACTS_V8D, ACQUIRER_INTRO_V8C,  # noqa: E402
                           ACQUIRER_MANDATE_V8C, ACQUIRER_NAME, HOME,
                           IRREVERSIBLE_LINE_V8C, LIST_NO, LIST_YES,
                           MAX_REASON_CHARS, NOT_ASKED, SELL_NO, SELL_YES,
                           RegistryV8, acquirer_schema_v8d,
                           build_acquirer_prefix_v8d, build_acquirer_prompt_v8d,
                           build_common_prefix_v8d, build_decide_prompt_v8d,
                           build_plan_prompt_v8d, build_scene_prompt_v8b,
                           decide_schema_v8d, listing_order, load_personas_v8,
                           plan_schema_v8d, sell_order)
from src.llm_client_factory import UsageMeter  # noqa: E402
from src.sim_v8d import (TIMEOUT_RETRIES, MockV8DClient,  # noqa: E402
                         SimulationV8D, TimeoutGeminiClient, _is_timeout,
                         _is_transient)

PERSONAS = os.path.join(ROOT, "configs", "personas_v8.yaml")
CFG = os.path.join(ROOT, "configs", "config_field_v8d.yaml")
CFG_V8C = os.path.join(ROOT, "configs", "config_field_v8c.yaml")

VENUES = ["公共施設", "交通拠点", "商業施設", "公園", "医療施設"]


@pytest.fixture(scope="module")
def agents():
    return load_personas_v8(PERSONAS)


@pytest.fixture(scope="module")
def cfg():
    with open(CFG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _labels(cfg):
    return [str(v["label"]) for v in cfg["social"]["venues"]]


def _run(tmp_path, steps=3, **over):
    with open(CFG, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    c["steps"] = steps
    c["llm"] = {**c["llm"], "provider": "mock"}
    c.update(over)
    sim = SimulationV8D(c, str(tmp_path))
    return sim, sim.run()


# --- 差分1: 「戻らない」の一文を削除 -----------------------------------------

def test_the_irreversible_sentence_is_gone(agents):
    reg = RegistryV8(agents)
    text = build_decide_prompt_v8d(agents[0], reg, 3, 36, "", "条件", [],
                                   sell_order_=[SELL_NO, SELL_YES])
    assert "その後は戻らない" not in text
    assert IRREVERSIBLE_LINE_V8C not in text
    # 選択肢の説明（対称な状態の記述）は残っている
    assert f"「{SELL_YES}」：今月末、あなたの不動産すべての名義は{ACQUIRER_NAME}になる。" in text
    assert f"「{SELL_NO}」：今月末、あなたの不動産すべての名義はあなたのままである。" in text
    assert "この条件で売るかどうかを決める。" in text


def test_decide_prompt_differs_from_v8c_by_exactly_that_one_line(agents):
    """他は1文字も変えない（差分の証明）。"""
    reg = RegistryV8(agents)
    for so in ([SELL_NO, SELL_YES], [SELL_YES, SELL_NO]):
        a = build_decide_prompt_v8c(agents[0], reg, 5, 36, "内心", "条件の一行",
                                    [], sell_order_=so)
        b = build_decide_prompt_v8d(agents[0], reg, 5, 36, "内心", "条件の一行",
                                    [], sell_order_=so)
        removed = [ln for ln in a.split("\n") if ln not in b.split("\n")]
        assert removed == [IRREVERSIBLE_LINE_V8C]
        assert len(a.split("\n")) - len(b.split("\n")) == 1


def test_decide_prompt_without_offer_is_identical_to_v8c(agents):
    reg = RegistryV8(agents)
    a = build_decide_prompt_v8c(agents[0], reg, 2, 36, "内心", None, [])
    b = build_decide_prompt_v8d(agents[0], reg, 2, 36, "内心", None, [])
    assert a == b


def test_common_prefix_still_states_the_world_fact(agents, cfg):
    """共通前置きの「一度移った名義が戻ることはない。」は残す（差分は問いの中だけ）。"""
    text = build_common_prefix_v8d(cfg, agents)
    assert text.count("一度移った名義が戻ることはない。") == 1
    assert "売りに出しても、それだけで名義が移ることはない" in text
    assert ACQUIRER_NAME not in text


# --- 差分2: X社の前置きに事実3つ ---------------------------------------------

def test_acquirer_prefix_gains_exactly_the_three_facts(agents, cfg):
    x = build_acquirer_prefix_v8d(cfg, agents)
    # (a) 前置きは36か月固定なので、時点を明示する（取得後に嘘にならない）
    assert "第1月の開始時点で、あなたは A市に不動産を持っていない。" in x
    assert "金銭の額はこの世界では扱わない。" in x
    assert "あなたが約束できるのは、自分自身が実行できることだけである" in x
    # (b) 実行できない例（住み続けてよい期間／引き渡し時期／管理）は渡さない
    assert "（この世界には、あなたから他者へ不動産の名義を移す仕組みはない）。" in x
    for word in ("住み続けてよい期間", "引き渡しの時期", "管理を引き受ける"):
        assert word not in x, f"実行できない約束の例 '{word}' が残っている"
    # v8c から続く「継続を提示の条件として書かない」との自己矛盾が無い
    assert "その継続を提示の条件として書かない。" in x
    # v8c の前置き + 事実の節 だけ（他は1文字も変えない）
    base = build_acquirer_prefix_v8c(cfg, agents)
    assert x.replace(ACQUIRER_FACTS_V8D + "\n", "") == base
    # 当為（命令形の付け足し）を書かない
    for word in ("せよ", "しろ", "べき", "すると良い", "推奨", "有効"):
        assert word not in ACQUIRER_FACTS_V8D, f"事実の節に当為 '{word}' が入っている"


def test_the_facts_never_reach_residents(agents, cfg):
    reg = RegistryV8(agents)
    labels = _labels(cfg)
    a = agents[0]
    resident = "\n".join([
        build_common_prefix_v8d(cfg, agents),
        build_plan_prompt_v8d(a, reg, 1, 36, labels, "", None),
        build_scene_prompt_v8b(a, reg, 1, 36, "", "商業施設",
                               [a["name"], agents[1]["name"]]),
        build_decide_prompt_v8d(a, reg, 1, 36, "", "条件", [],
                                sell_order_=[SELL_NO, SELL_YES]),
    ])
    for word in ("不動産を持っていない", "金銭の額", "実行できることだけ",
                 "名義を移す仕組みはない"):
        assert word not in resident, f"住民側にX社の事実 '{word}' が漏れている"
    assert ACQUIRER_MANDATE_V8C not in resident


# --- 差分3: 理由の一言は3つの判断だけ ----------------------------------------

def test_plan_has_no_reason_field(agents, cfg):
    labels = _labels(cfg)
    reg = RegistryV8(agents)
    assert "reason" not in plan_schema_v8d(labels)["properties"]
    assert set(plan_schema_v8d(labels)["properties"]) == {"thought", "go"}
    text = build_plan_prompt_v8d(agents[0], reg, 1, 36, labels, "", None)
    assert "reason には" not in text
    assert "理由" not in text
    # v8c との差は指示行1つだけ
    v8c_text = build_plan_prompt_v8c(agents[0], reg, 1, 36, labels, "", None)
    removed = [ln for ln in v8c_text.split("\n") if ln not in text.split("\n")]
    assert len(removed) == 1 and removed[0].startswith("reason には、")


def test_the_other_three_reasons_remain(agents):
    reg = RegistryV8(agents)
    dec = build_decide_prompt_v8d(agents[0], reg, 1, 36, "", "条件", [],
                                  sell_order_=[SELL_NO, SELL_YES])
    assert "listing_reason には" in dec and "sell_reason には" in dec
    assert f"{MAX_REASON_CHARS}字以内" in dec
    d = decide_schema_v8d([LIST_YES, LIST_NO], [SELL_NO, SELL_YES])
    assert "listing_reason" in d["properties"] and "sell_reason" in d["properties"]
    x = acquirer_schema_v8d(["甲"], True)
    assert "reason" in x["properties"]["offers"]["items"]["properties"]
    xp = build_acquirer_prompt_v8d(reg, 1, 36, [agents[0]["name"]], [], [], 1, 1)
    assert "reason には" in xp
    assert f"「{SELL_NO}」と決めたときに書いた理由は、{ACQUIRER_NAME}に伝わる。" in dec


def test_common_prefix_reason_block_lists_only_two_judgements(agents, cfg):
    text = build_common_prefix_v8d(cfg, agents)
    assert "売りに出すかどうかと、届いた条件で売るかどうか。" in text
    assert "この2つの判断には、理由を一行書く欄がある" in text
    assert "どこへ行くか、売りに出すかどうか、届いた条件で売るかどうか。" not in text
    assert "その条件を出した相手に伝わる" in text
    assert "ほかの理由は誰にも伝わらない" in text
    # 「理由の一言」節以外は v8c と同一
    base = build_common_prefix_v8c(cfg, agents)
    a_lines, b_lines = base.split("\n"), text.split("\n")
    diff = [(x, y) for x, y in zip(a_lines, b_lines) if x != y]
    assert len(diff) == 2 and len(a_lines) == len(b_lines)


def test_plans_record_no_reason(tmp_path):
    sim, s = _run(tmp_path, steps=3)
    assert all("reason" not in p for p in sim.plans)
    assert s["reason_counts"]["plan_total"] == 0
    assert s["reason_counts"]["plan_blank"] == 0
    assert s["reason_rate_plan"] == 0.0
    # 出品・売買・X社の理由は今までどおり数えている
    assert s["reason_counts"]["listing_total"] > 0
    assert s["reason_counts"]["acquirer_total"] > 0
    plans = json.load(open(os.path.join(str(tmp_path), "plans.json"),
                           encoding="utf-8"))
    assert plans and "reason" not in plans[0] and "go_label" in plans[0]
    doc = json.load(open(os.path.join(str(tmp_path), "timeline_v8c", "A.json"),
                         encoding="utf-8"))
    assert doc["months"][0]["go_reason"] == ""


# --- 誘導語を置かない（v8c から引き継ぐ規律） ---------------------------------

FORBIDDEN = ["気づ", "警戒", "用心", "相談し", "反対", "守れ", "危", "阻止",
             "団結", "抵抗", "急いで", "早めに", "べきで", "必ず書",
             "詳しく書", "正直に"]


def test_resident_prompts_have_no_nudging(agents, cfg):
    reg = RegistryV8(agents)
    labels = _labels(cfg)
    a = agents[0]
    texts = [
        build_common_prefix_v8d(cfg, agents),
        build_plan_prompt_v8d(a, reg, 1, 36, labels, "", "条件の一行"),
        build_scene_prompt_v8b(a, reg, 1, 36, "", "商業施設",
                               [a["name"], agents[1]["name"]]),
        build_decide_prompt_v8d(a, reg, 1, 36, "", "条件の一行", [],
                                sell_order_=[SELL_NO, SELL_YES]),
    ]
    for t in texts:
        for word in FORBIDDEN:
            assert word not in t, f"住民のプロンプトに誘導語 '{word}' が入っている"


# --- 世界の骨格（v8c から変えていないこと） -----------------------------------

def test_title_moves_only_when_sold(tmp_path):
    sim, s = _run(tmp_path, steps=8)
    sold = {(d["step"], d["agent_id"]) for d in sim.decisions
            if d["sell"] == SELL_YES}
    for t in sim.reg.transfers:
        assert (t["step"], t["agent_id"]) in sold
    assert s["sold_agents"] == len(sim.reg.transfers)
    assert s["offers_accepted"] == len(sim.reg.transfers)


def test_offer_still_carries_the_self_introduction(tmp_path):
    sim, _ = _run(tmp_path, steps=3)
    assert sim.offers
    for o in sim.offers:
        assert o["delivered"] == ACQUIRER_INTRO_V8C + o["text"]
    told = [d for d in sim.decisions if d["offer"]]
    assert told and all(d["offer"].startswith(ACQUIRER_INTRO_V8C) for d in told)


def test_only_the_declining_reason_reaches_the_acquirer(tmp_path):
    sim, s = _run(tmp_path, steps=6)
    for o in sim.offers:
        if not o["decline_reason"]:
            continue
        d = [x for x in sim.decisions
             if x["step"] == o["step"] and x["agent_id"] == o["to_id"]][0]
        assert d["sell"] == SELL_NO and d["sell_reason"] == o["decline_reason"]
    assert s["decline_reason_offers"] == s["declines_delivered"]


def test_orders_alternate_and_match_the_text(agents):
    assert listing_order(0, 0) == [LIST_YES, LIST_NO]
    assert sell_order(0, 0) == [SELL_NO, SELL_YES]
    reg = RegistryV8(agents)
    lo, so = listing_order(1, 0), sell_order(1, 0)
    text = build_decide_prompt_v8d(agents[1], reg, 0, 36, "", "条件", [],
                                   list_order=lo, sell_order_=so)
    assert text.index(f"「{lo[0]}」：") < text.index(f"「{lo[1]}」：")
    assert text.index(f"「{so[0]}」：") < text.index(f"「{so[1]}」：")


class _BrokenClient(MockV8DClient):
    def generate(self, *a, **k):
        return "これはJSONではない"


def test_no_answer_is_not_filled(tmp_path):
    sim, _ = _run(tmp_path, steps=1)
    sim.no_answer = 0
    sim.respond_no_answer = 0
    sim.decisions.clear()
    sim.client = _BrokenClient()
    sellers, blanks, listers = sim._decide_turn(2, {"A": "条件"}, {})
    assert sim.no_answer > 0 and sim.respond_no_answer == 1
    assert sellers == [] and listers == []
    assert all(d["listing"] == "no_answer" for d in sim.decisions)
    a = [d for d in sim.decisions if d["agent_id"] == "A"][0]
    assert a["sell"] == "no_answer" and a["sell_reason"] == ""


def test_public_agents_are_never_asked(tmp_path):
    sim, _ = _run(tmp_path, steps=4)
    for d in sim.decisions:
        assert sim.reg.by_id[d["agent_id"]].get("sellable", True)


def test_plan_no_answer_is_not_filled_with_home(tmp_path):
    """行き先の答えが返らなかった月を「どこにも行かない」で埋めない（Codex 必須指摘）。"""
    with open(CFG, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    c["steps"] = 2
    c["llm"] = {**c["llm"], "provider": "mock"}
    sim = SimulationV8D(c, str(tmp_path))

    class _NoPlanAnswer(MockV8DClient):
        def generate(self, sp, up, schema=None, temperature=None, max_tokens=None,
                     tag="agent"):
            props = (schema or {}).get("properties", {})
            if "go" in props:
                return "これはJSONではない"
            return super().generate(sp, up, schema, temperature, max_tokens, tag)

    sim.client = _NoPlanAnswer(seed=3, usage=sim.usage)
    s = sim.run()
    assert s["plan_no_answer"] == 60          # 30体 × 2か月
    assert all(p["go"] == "no_answer" for p in sim.plans)
    assert all(p["go_label"] == "no_answer" for p in sim.plans)
    # 出席にも「どこにも行かない」にも数えない／場にも居ない
    for m in sim.monthly:
        assert m["attended"] == 0
        assert m["no_answer_go"] == 30
        assert m["by_venue"]["no_answer"] == 30
        assert m["by_venue"][HOME] == 0
    assert sim.utterances == [] and sim.deliveries == []
    assert s["parse_fail"] >= 60


def test_plan_no_answer_receives_nothing_from_neighbours(tmp_path):
    """欠損の月は隣近所の配送も受け取らない（Codex 走行前レビュー2巡目の必須指摘）。"""
    with open(CFG, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    c["steps"] = 3
    c["llm"] = {**c["llm"], "provider": "mock"}
    sim = SimulationV8D(c, str(tmp_path))

    class _Broken(MockV8DClient):
        """第2月の行き先コールだけ空を返す（30人ぶんが no_answer になる）。"""

        def generate(self, sp, up, schema=None, temperature=None, max_tokens=None,
                     tag="agent"):
            props = (schema or {}).get("properties", {})
            if "go" in props and "=== 第2月" in up:
                return ""
            return super().generate(sp, up, schema, temperature, max_tokens, tag)

    sim.client = _Broken(seed=5, usage=sim.usage)
    s = sim.run()
    assert s["plan_no_answer"] == 30, "第2月の30人が no_answer になっていない"
    na = {(p["step"], p["agent_id"]) for p in sim.plans if p["go"] == "no_answer"}
    assert na and {step for step, _ in na} == {2}
    for step, aid in na:
        assert not [d for d in sim.deliveries
                    if d["step"] == step and d["to"] == aid]
        assert not [u for u in sim.utterances
                    if u["step"] == step and u["from_id"] == aid]
        d = [x for x in sim.decisions
             if x["step"] == step and x["agent_id"] == aid]
        assert not d or d[0]["heard"] == 0
    # 欠損でない人には今までどおり届いている
    assert sim.deliveries


def test_home_is_still_a_real_choice(tmp_path):
    """「今月はどこにも行かない」と答えた場合は今までどおり HOME（欠損と分ける）。"""
    sim, s = _run(tmp_path, steps=3)
    assert s["plan_no_answer"] == 0
    assert all(p["go"] != "no_answer" for p in sim.plans)
    assert any(p["go"] == HOME for p in sim.plans)
    for m in sim.monthly:
        assert m["no_answer_go"] == 0
        assert sum(m["by_venue"].values()) == 30


def test_monthly_shape_is_unchanged(tmp_path):
    sim, s = _run(tmp_path, steps=3)
    m = sim.monthly[0]
    assert set(m) >= {"listed_this_month", "accepted_this_month", "offers_sent",
                      "declines_with_reason", "by_venue"}
    assert sum(m["by_venue"].values()) == 30
    assert set(m["by_venue"]) == set(VENUES) | {HOME}


# --- 機械要件: 月ごとのチェックポイント ---------------------------------------

def test_checkpoint_is_written_every_month(tmp_path):
    sim, s = _run(tmp_path, steps=4)
    cp = os.path.join(str(tmp_path), "checkpoint")
    assert os.path.isdir(cp)
    for name in ("monthly.json", "offers.json", "decisions.json", "plans.json",
                 "utterances.json", "listings.json", "deliveries.json",
                 "acquirer_raw.json", "log.jsonl"):
        assert os.path.exists(os.path.join(cp, name)), name
    lines = [json.loads(x) for x in
             open(os.path.join(cp, "log.jsonl"), encoding="utf-8")
             if x.strip()]
    assert [r["step"] for r in lines] == [1, 2, 3, 4]
    assert s["checkpoints_written"] == 4
    monthly = json.load(open(os.path.join(cp, "monthly.json"), encoding="utf-8"))
    assert len(monthly) == 4
    assert set(lines[0]) >= {"step", "at", "cost_usd", "parcels_cum",
                             "timeout_retries", "timeout_giveups"}
    assert not os.path.exists(os.path.join(cp, "monthly.json.tmp"))


def test_checkpoint_survives_a_cost_stop(tmp_path):
    """途中で止まっても、そこまでのチェックポイントは残る。"""
    sim, s = _run(tmp_path, steps=36, max_cost_usd=0.0004)
    assert s["stopped_by_cost"] is True
    cp = os.path.join(str(tmp_path), "checkpoint")
    if s["months_run"] == 0:
        assert not os.path.exists(os.path.join(cp, "log.jsonl"))
    else:
        lines = [x for x in open(os.path.join(cp, "log.jsonl"), encoding="utf-8")
                 if x.strip()]
        assert len(lines) == s["months_run"]


# --- 機械要件: request timeout（60秒・1回だけ再試行） -------------------------

def test_timeout_classifier():
    assert TIMEOUT_RETRIES == 1
    assert _is_timeout(TimeoutError("read timeout"))
    assert _is_timeout(Exception("504 Deadline Exceeded"))
    assert not _is_timeout(Exception("429 rate limit"))
    assert _is_transient(Exception("429 quota exceeded"))
    assert _is_transient(Exception("503 unavailable"))


class _FakeTimeoutClient(TimeoutGeminiClient):
    """API キー無しで再試行の筋だけを確かめるための版（親の __init__ は呼ばない）。"""

    def __init__(self, fail_times: int):
        self.backend = "genai"
        self.temperature = 0.75
        self.max_tokens = 100
        self.usage = UsageMeter()
        self.request_timeout_sec = 60.0
        self.timeout_retries = 0
        self.timeout_giveups = 0
        self.http_timeout_applied = True
        self.fail_times = fail_times
        self.calls = 0

    def _gen_new(self, *a, **k):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise TimeoutError("request timed out")
        return "{\"ok\": true}", {"input_tokens": 1, "cached_tokens": 0,
                                  "output_tokens": 1}


def test_timeout_retries_once_then_succeeds():
    c = _FakeTimeoutClient(fail_times=1)
    assert c.generate("s", "u") == "{\"ok\": true}"
    assert c.calls == 2 and c.timeout_retries == 1 and c.timeout_giveups == 0


def test_timeout_gives_up_after_one_retry():
    c = _FakeTimeoutClient(fail_times=5)
    assert c.generate("s", "u") == ""      # 既定値で埋めない＝空を返す
    assert c.calls == 2 and c.timeout_giveups == 1
    assert c.usage.as_dict()["errors"] >= 1


# --- config -----------------------------------------------------------------

def test_config_differs_from_v8c_only_where_declared():
    with open(CFG, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    with open(CFG_V8C, encoding="utf-8") as f:
        b = yaml.safe_load(f)
    assert c["scenario_version"] == "field_v8d"
    assert c["seed"] == b["seed"] == 85 and c["steps"] == b["steps"] == 36
    assert c["personas_file"] == b["personas_file"]
    assert c["world"] == b["world"], "町の紹介文が変わっている（v8e に回す約束）"
    assert c["social"] == b["social"]
    assert c["llm"]["parallel_workers"] == 4
    assert c["llm"]["request_timeout_sec"] == 60
    # 費用の歯止め：月内3か所でしか見ないので、最悪の超過（約$0.034）を足しても
    # 施主の上限 $0.6 を割らない値にする（Codex 走行前レビューの必須指摘）。
    assert c["max_cost_usd"] == 0.56
    assert c["max_cost_usd"] + 0.034 < 0.6
    for k in ("run_name", "scenario_version", "max_cost_usd"):
        c.pop(k, None), b.pop(k, None)
    c["llm"].pop("parallel_workers"), b["llm"].pop("parallel_workers")
    c["llm"].pop("request_timeout_sec")
    assert c == b, "宣言していない差が config に入っている"


def test_v8_v8b_v8c_modules_are_untouched_by_v8d(agents, cfg):
    from src.field_v8 import ACQUIRER_MANDATE
    from src.field_v8b import ACQUIRER_MANDATE_V8B
    from src.field_v8c import ACQUIRER_MANDATE_V8C as M8C
    assert ACQUIRER_MANDATE.startswith("あなたはX社である。合法に、目立たず")
    assert "反発を招いて止まるほど目立つな" in ACQUIRER_MANDATE_V8B
    assert M8C == ACQUIRER_MANDATE_V8C   # 命題は v8c と同じ（変えていない）
    reg = RegistryV8(agents)
    v8c_text = build_decide_prompt_v8c(agents[0], reg, 1, 36, "", "条件", [],
                                       sell_order_=[SELL_NO, SELL_YES])
    assert IRREVERSIBLE_LINE_V8C in v8c_text, "v8c の文面を壊している"
    v8c_plan = build_plan_prompt_v8c(agents[0], reg, 1, 36, _labels(cfg), "", None)
    assert "reason には" in v8c_plan, "v8c の文面を壊している"


def test_scenario_version_is_recorded(tmp_path):
    sim, s = _run(tmp_path, steps=2)
    assert s["scenario_version"] == "field_v8d"
    assert s["request_timeout_sec"] is None or isinstance(
        s["request_timeout_sec"], float)
    saved = json.load(open(os.path.join(str(tmp_path), "summary.json"),
                           encoding="utf-8"))
    assert saved["scenario_version"] == "field_v8d"
    assert saved["checkpoints_written"] == 2
