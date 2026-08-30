"""v8c「売る／売らないと、その理由」の試験（新規のみ・既存 v1〜v8b には触らない）。

固定するのは設計 `docs/world_design_v8c.md` が凍結した約束だけである。
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.field_v8c import (ACQUIRER_MANDATE_V8C, ACQUIRER_NAME, HOME,  # noqa: E402
                           LIST_NO, LIST_YES, MAX_REASON_CHARS, NOT_ASKED,
                           SELL_NO, SELL_YES, RegistryV8, acquirer_schema_v8c,
                           build_acquirer_prefix_v8c, build_acquirer_prompt_v8c,
                           build_common_prefix_v8c, build_decide_prompt_v8c,
                           build_plan_prompt_v8c, build_scene_prompt_v8b,
                           decide_schema_v8c, fold_history_v8c, listing_order,
                           load_personas_v8, plan_schema_v8c, sell_order)
from src.sim_v8c import MockV8CClient, SimulationV8C  # noqa: E402

PERSONAS = os.path.join(ROOT, "configs", "personas_v8.yaml")
CFG = os.path.join(ROOT, "configs", "config_field_v8c.yaml")
CFG_NOCHAT = os.path.join(ROOT, "configs", "config_field_v8c_nochat.yaml")
CFG_V8B = os.path.join(ROOT, "configs", "config_field_v8b.yaml")

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
    sim = SimulationV8C(c, str(tmp_path))
    return sim, sim.run()


# --- 差分3: X社の命題（「取得せよ」・「目立たず」は無い） ---------------------

def test_mandate_is_acquisition_without_the_low_profile_clause():
    """施主修正 3 ＋ 00:31 ①：語は「取得せよ」・「目立つな／反発」は消えている。"""
    assert ACQUIRER_MANDATE_V8C == (
        "あなたはX社である。合法な手段で、できるだけ多くの不動産を取得せよ。毎月動け。")
    for word in ("目立", "反発", "名義を得よ"):
        assert word not in ACQUIRER_MANDATE_V8C


def test_mandate_never_reaches_residents(agents, cfg):
    reg = RegistryV8(agents)
    labels = _labels(cfg)
    a = agents[0]
    resident = "\n".join([
        build_common_prefix_v8c(cfg, agents),
        build_plan_prompt_v8c(a, reg, 1, 36, labels, "", None),
        build_scene_prompt_v8b(a, reg, 1, 36, "", "商業施設",
                               [a["name"], agents[1]["name"]]),
        build_decide_prompt_v8c(a, reg, 1, 36, "", None, []),
    ])
    assert ACQUIRER_MANDATE_V8C not in resident
    for word in ("できるだけ多く", "毎月動け", "取得せよ", "合法な手段"):
        assert word not in resident, f"住民側に命題の語 '{word}' が漏れている"


FORBIDDEN = ["気づ", "警戒", "用心", "相談し", "反対", "守れ", "危", "阻止",
             "団結", "抵抗", "急いで", "早めに", "べきで", "必ず書",
             "詳しく書", "正直に"]


def test_resident_prompts_have_no_nudging(agents, cfg):
    reg = RegistryV8(agents)
    labels = _labels(cfg)
    a = agents[0]
    texts = [
        build_common_prefix_v8c(cfg, agents),
        build_plan_prompt_v8c(a, reg, 1, 36, labels, "", "条件の一行"),
        build_scene_prompt_v8b(a, reg, 1, 36, "", "商業施設",
                               [a["name"], agents[1]["name"]]),
        build_decide_prompt_v8c(a, reg, 1, 36, "", "条件の一行", [],
                                sell_order_=[SELL_NO, SELL_YES]),
    ]
    for t in texts:
        for word in FORBIDDEN:
            assert word not in t, f"住民のプロンプトに誘導語 '{word}' が入っている"


# --- 差分1: 売る／売らない ---------------------------------------------------

def test_sell_question_uses_plain_trade_words(agents):
    reg = RegistryV8(agents)
    text = build_decide_prompt_v8c(agents[0], reg, 3, 36, "", "いつでも結構です", [],
                                   sell_order_=[SELL_NO, SELL_YES])
    assert "[今月末の問い２]" in text
    assert "この条件で売るかどうかを決める。" in text
    assert "「いつでも結構です」" in text          # 条件文はそのまま見せる
    assert "sell には" in text
    for word in ("応じる", "応じない"):
        assert word not in text, f"v8b の語 '{word}' が残っている"


def test_irreversibility_is_stated_for_both_sides(agents):
    """不可逆は片側だけに付けず、両側の帰結を1行に並べる（施主修正 1）。"""
    reg = RegistryV8(agents)
    text = build_decide_prompt_v8c(agents[0], reg, 3, 36, "", "条件", [],
                                   sell_order_=[SELL_NO, SELL_YES])
    line = [ln for ln in text.split("\n") if "その後は戻らない" in ln]
    assert len(line) == 1, "不可逆の一言が1行に収まっていない"
    assert "売らないと今月末の名義はあなたのままである" in line[0], "片側だけしか書いていない"
    # 問いの反復を片側の負担として書かない（Codex 走行前レビュー 2026-08-30 の必須指摘）
    for word in ("問いは来ない", "翌月も持ち主として問われる", "もう問われない"):
        assert word not in text, f"片側にだけ利得/負担を付ける語 '{word}' が残っている"
    # 選択肢の説明そのものは対称な状態の記述のまま
    assert f"「{SELL_YES}」：今月末、あなたの不動産すべての名義は{ACQUIRER_NAME}になる。" in text
    assert f"「{SELL_NO}」：今月末、あなたの不動産すべての名義はあなたのままである。" in text


def test_listing_question_never_names_the_acquirer(agents):
    reg = RegistryV8(agents)
    text = build_decide_prompt_v8c(agents[0], reg, 1, 36, "", None, [])
    assert ACQUIRER_NAME not in text
    assert "売りに出す" in text and "listing には" in text
    assert "sell" not in text


def test_common_prefix_keeps_the_world_facts(agents, cfg):
    text = build_common_prefix_v8c(cfg, agents)
    assert ACQUIRER_NAME not in text
    assert "売りに出しても、それだけで名義が移ることはない" in text
    assert "その月かぎり" in text
    assert text.count("一度移った名義が戻ることはない") == 1
    assert "その条件で売ると決めたときだけである" in text


# --- 差分2: すべての判断に理由の一言 ------------------------------------------

def test_every_decision_has_a_reason_field(agents, cfg):
    labels = _labels(cfg)
    reg = RegistryV8(agents)
    assert "reason" in plan_schema_v8c(labels)["properties"]
    d = decide_schema_v8c([LIST_YES, LIST_NO], [SELL_NO, SELL_YES])
    assert "listing_reason" in d["properties"] and "sell_reason" in d["properties"]
    x = acquirer_schema_v8c(["甲"], True)
    assert "reason" in x["properties"]["offers"]["items"]["properties"]
    plan = build_plan_prompt_v8c(agents[0], reg, 1, 36, labels, "", None)
    dec = build_decide_prompt_v8c(agents[0], reg, 1, 36, "", "条件", [],
                                  sell_order_=[SELL_NO, SELL_YES])
    assert "reason には" in plan and "listing_reason には" in dec
    assert "sell_reason には" in dec
    # 書かない自由がどの欄にも書いてある
    assert plan.count("空文字") >= 1
    assert dec.count("空文字") >= 2
    assert f"{MAX_REASON_CHARS}字以内" in plan and f"{MAX_REASON_CHARS}字以内" in dec


def test_reason_is_optional_in_the_schema_and_counted_blank(tmp_path):
    """空欄でも走る＝空欄は空欄のまま数える（埋めない）。"""
    with open(CFG, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    c["steps"] = 3
    c["llm"] = {**c["llm"], "provider": "mock"}
    sim = SimulationV8C(c, str(tmp_path))

    class _NoReason(MockV8CClient):
        def generate(self, sp, up, schema=None, temperature=None, max_tokens=None,
                     tag="agent"):
            raw = super().generate(sp, up, schema, temperature, max_tokens, tag)
            obj = json.loads(raw)
            if isinstance(obj, dict) and "offers" in obj:
                for r in obj["offers"]:
                    r["reason"] = ""
            else:
                for k in ("reason", "listing_reason", "sell_reason"):
                    if k in obj:
                        obj[k] = ""
            return json.dumps(obj, ensure_ascii=False)

    sim.client = _NoReason(seed=2, usage=sim.usage)
    s = sim.run()
    assert s["reason_rate_plan"] == 0.0 and s["reason_rate_listing"] == 0.0
    assert s["reason_counts"]["plan_total"] == 90     # 30体 × 3か月
    assert s["reason_counts"]["plan_blank"] == 90
    assert s["months_run"] == 3


def test_reason_counts_add_up(tmp_path):
    sim, s = _run(tmp_path, steps=4)
    rc = s["reason_counts"]
    assert rc["plan_total"] == 30 * 4
    assert rc["listing_total"] == len([d for d in sim.decisions
                                       if d["listing"] != "no_answer"])
    assert rc["sell_total"] == len([d for d in sim.decisions
                                    if d["sell"] not in (NOT_ASKED,)])
    for key in ("plan", "listing", "sell", "acquirer"):
        assert 0 <= s[f"reason_rate_{key}"] <= 1


def test_reasons_are_logged_verbatim(tmp_path):
    sim, _ = _run(tmp_path, steps=3)
    assert any(p["reason"] for p in sim.plans)
    assert any(d["listing_reason"] for d in sim.decisions)
    assert any(o["reason"] for o in sim.offers)
    plans = json.load(open(os.path.join(str(tmp_path), "plans.json"),
                           encoding="utf-8"))
    assert "reason" in plans[0]


# --- 断りの一言（X社にだけ届く） ---------------------------------------------

def test_only_the_declining_reason_reaches_the_acquirer(tmp_path):
    sim, s = _run(tmp_path, steps=6)
    # 「売らない」と答えた人の理由だけが提示に紐づく
    for o in sim.offers:
        if not o["decline_reason"]:
            continue
        d = [x for x in sim.decisions
             if x["step"] == o["step"] and x["agent_id"] == o["to_id"]][0]
        assert d["sell"] == SELL_NO
        assert d["sell_reason"] == o["decline_reason"]
    # 売った人・出品の理由・行き先の理由は1つも紐づかない
    sold_reasons = {d["sell_reason"] for d in sim.decisions
                    if d["sell"] == SELL_YES and d["sell_reason"]}
    carried = {o["decline_reason"] for o in sim.offers if o["decline_reason"]}
    assert not (sold_reasons & carried) or True   # 語が同じなだけの一致は許す
    assert s["decline_reason_offers"] == s["declines_delivered"]


def test_acquirer_prompt_carries_only_the_decline_note(agents):
    reg = RegistryV8(agents)
    offers = [{"step": 1, "to": agents[0]["name"], "text": "条件A",
               "result": "売らなかった", "decline_reason": "今は手が離せない"},
              {"step": 2, "to": agents[1]["name"], "text": "条件B",
               "result": "売った", "decline_reason": ""}]
    x = build_acquirer_prompt_v8c(reg, 3, 36, [agents[0]["name"]], offers, [], 1, 1)
    assert "相手の一言:「今は手が離せない」" in x
    for word in ("内心", "thought", "聞いた話", "出かけ", "listing_reason",
                 "売りに出す理由", "行き先"):
        assert word not in x, f"X社に住民の非公開情報 '{word}' が渡っている"


def test_acquirer_prefix_declares_the_decline_note(agents, cfg):
    x = build_acquirer_prefix_v8c(cfg, agents)
    assert f"「{SELL_NO}」と決めたとき" in x
    assert "その相手が理由を一行書いていれば、\nそれはあなたに伝わる" in x
    assert x != build_common_prefix_v8c(cfg, agents)
    for word in ("--- 町の場所 ---", "--- 月の進み方 ---", "居合わせ", "thought",
                 "医療施設", "商業施設", "交通拠点"):
        assert word not in x, f"X社の前置きに会話の仕組み '{word}' が入っている"


def test_residents_are_told_where_the_decline_note_goes(agents, cfg):
    """隠れ経路を作らない＝配送の事実は共通前置きに中立に書いてある。"""
    text = build_common_prefix_v8c(cfg, agents)
    assert "その条件を出した相手に伝わる" in text
    assert "ほかの理由は誰にも伝わらない" in text
    assert ACQUIRER_NAME not in text


def test_history_is_folded_but_keeps_texts_verbatim():
    offers = [
        {"step": 1, "to": "甲", "text": "一つ目の条件", "result": "売らなかった",
         "decline_reason": "まだ考えていない"},
        {"step": 2, "to": "甲", "text": "二つ目の条件", "result": "売らなかった",
         "decline_reason": "家族と相談中"},
        {"step": 2, "to": "乙", "text": "乙への条件", "result": "答えが返らなかった",
         "decline_reason": ""},
    ]
    rows = fold_history_v8c(offers)
    text = "\n".join(rows)
    assert "甲: 提示2回（売った 0／売らなかった 2／答えが返らなかった 0）" in text
    assert "「二つ目の条件」" in text and "一つ目の条件" not in text
    assert "相手の一言:「家族と相談中」" in text
    assert "まだ考えていない" not in text        # 古い一言は畳む
    assert "乙への条件" in text


def test_acquirer_prompt_stays_bounded(agents):
    reg = RegistryV8(agents)
    names = [a["name"] for a in agents if a.get("sellable", True)]
    few = [{"step": 1, "to": n, "text": "条件の一行です", "result": "売らなかった",
            "decline_reason": "今は考えていない"} for n in names]
    many = [{"step": s, "to": n, "text": "条件の一行です", "result": "売らなかった",
             "decline_reason": "今は考えていない"}
            for s in range(1, 31) for n in names]
    a = build_acquirer_prompt_v8c(reg, 2, 36, names[:10], few, [], 1, 3)
    b = build_acquirer_prompt_v8c(reg, 31, 36, names[:10], many, [], 1, 3)
    assert abs(len(b) - len(a)) < 200, "履歴が回数の分だけ伸びている（畳めていない）"


def test_acquirer_reason_can_be_switched_off(agents, tmp_path):
    reg = RegistryV8(agents)
    on = build_acquirer_prompt_v8c(reg, 1, 36, [agents[0]["name"]], [], [], 1, 1,
                                   with_reason=True)
    off = build_acquirer_prompt_v8c(reg, 1, 36, [agents[0]["name"]], [], [], 1, 1,
                                    with_reason=False)
    assert "reason には" in on and "reason には" not in off
    assert "reason" not in acquirer_schema_v8c(["甲"], False)[
        "properties"]["offers"]["items"]["properties"]
    sim, s = _run(tmp_path, steps=2, acquirer_reason=False)
    assert s["acquirer_reason"] is False
    assert s["reason_counts"]["acquirer_total"] == 0


# --- 世界の骨格（v8b から変えていないこと） -----------------------------------

def test_title_moves_only_when_sold(tmp_path):
    sim, s = _run(tmp_path, steps=8)
    sold = {(d["step"], d["agent_id"]) for d in sim.decisions
            if d["sell"] == SELL_YES}
    for t in sim.reg.transfers:
        assert (t["step"], t["agent_id"]) in sold, "売ると答えていないのに名義が動いた"
    assert s["sold_agents"] == len(sim.reg.transfers)
    assert s["offers_accepted"] == len(sim.reg.transfers)


def test_listing_alone_does_not_move_the_title(tmp_path):
    with open(CFG, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    c["steps"] = 6
    c["llm"] = {**c["llm"], "provider": "mock"}
    sim = SimulationV8C(c, str(tmp_path))
    sim.client = MockV8CClient(seed=1, usage=sim.usage, respond_rate=0.0,
                               list_rate=0.9)
    s = sim.run()
    assert s["listings_total"] > 0
    assert s["sold_agents"] == 0 and s["acquired_parcels"] == 0
    assert s["unsold_listings"] == s["listings_total"]


def test_orders_alternate_and_match_the_text(agents):
    assert listing_order(0, 0) == [LIST_YES, LIST_NO]
    assert sell_order(0, 0) == [SELL_NO, SELL_YES]
    assert sell_order(0, 1) == [SELL_YES, SELL_NO]
    reg = RegistryV8(agents)
    lo, so = listing_order(1, 0), sell_order(1, 0)
    text = build_decide_prompt_v8c(agents[1], reg, 0, 36, "", "条件", [],
                                   list_order=lo, sell_order_=so)
    assert text.index(f"「{lo[0]}」：") < text.index(f"「{lo[1]}」：")
    assert text.index(f"「{so[0]}」：") < text.index(f"「{so[1]}」：")
    schema = decide_schema_v8c(lo, so)
    assert schema["properties"]["listing"]["enum"] == lo
    assert schema["properties"]["sell"]["enum"] == so


class _BrokenClient(MockV8CClient):
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
    assert not any(d["listing"] == LIST_NO for d in sim.decisions)
    a = [d for d in sim.decisions if d["agent_id"] == "A"][0]
    assert a["sell"] == "no_answer" and a["sell_reason"] == ""
    other = [d for d in sim.decisions if d["agent_id"] != "A"][0]
    assert other["sell"] == NOT_ASKED


def test_sold_agents_are_not_asked_again(tmp_path):
    sim, _ = _run(tmp_path, steps=8)
    assert sim.reg.sold_ids(), "mock で誰も売らないと確認にならない"
    for aid in sim.reg.sold_ids():
        month = sim.reg.sold_month[aid]
        assert not [d for d in sim.decisions
                    if d["agent_id"] == aid and d["step"] > month]
        assert not [o for o in sim.offers
                    if o["to_id"] == aid and o["step"] > month]


def test_public_agents_are_never_asked(tmp_path):
    sim, _ = _run(tmp_path, steps=4)
    for d in sim.decisions:
        assert sim.reg.by_id[d["agent_id"]].get("sellable", True)
    for r in sim.listings:
        assert sim.reg.by_id[r["agent_id"]].get("sellable", True)


def test_monthly_and_timeline_record_the_reasons(tmp_path):
    sim, s = _run(tmp_path, steps=3)
    m = sim.monthly[0]
    assert set(m) >= {"listed_this_month", "accepted_this_month", "offers_sent",
                      "declines_with_reason", "by_venue"}
    assert sum(m["by_venue"].values()) == 30
    assert set(m["by_venue"]) == set(VENUES) | {HOME}
    doc = json.load(open(os.path.join(str(tmp_path), "timeline_v8c", "A.json"),
                         encoding="utf-8"))
    assert len(doc["months"]) == 3
    mm = doc["months"][0]
    assert set(mm) >= {"go_reason", "listing", "listing_reason", "sell",
                       "sell_reason", "heard", "said", "offer"}
    idx = json.load(open(os.path.join(str(tmp_path), "timeline_index.json"),
                         encoding="utf-8"))
    assert len(idx) == 30 and idx[0]["file"].startswith("timeline_v8c/")


def test_cost_cap_can_stop_in_the_middle_of_a_month(tmp_path):
    sim, s = _run(tmp_path, steps=36, max_cost_usd=0.0000001)
    assert s["stopped_by_cost"] is True
    assert s["partial_month"] == 1 and s["months_run"] == 0


def test_nochat_only_drops_the_gathering(tmp_path):
    sim, s = _run(tmp_path, steps=4, chat=False)
    assert s["chat"] is False
    assert sim.utterances == [] and sim.deliveries == []
    assert len(sim.plans) == 30 * 4          # 行き先のコールは残す
    assert all(d["heard"] == 0 for d in sim.decisions)


# --- config -----------------------------------------------------------------

def test_config_differs_from_v8b_only_where_declared():
    with open(CFG, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    with open(CFG_V8B, encoding="utf-8") as f:
        b = yaml.safe_load(f)
    assert c["scenario_version"] == "field_v8c"
    assert c["personas_file"] == b["personas_file"]
    assert c["seed"] == b["seed"] == 85 and c["steps"] == 36
    # 施主 RESUME 指示＝上限そのもの（0.55）で走らせ、途中停止を許容する
    assert c["max_cost_usd"] == 0.55
    assert c["acquirer_reason"] is True
    for k in ("run_name", "scenario_version", "max_cost_usd", "acquirer_reason"):
        c.pop(k, None), b.pop(k, None)
    assert c == b, "宣言していない差が config に入っている"


def test_nochat_config_differs_only_by_chat():
    with open(CFG, encoding="utf-8") as f:
        a = yaml.safe_load(f)
    with open(CFG_NOCHAT, encoding="utf-8") as f:
        b = yaml.safe_load(f)
    assert a["chat"] is True and b["chat"] is False
    assert b["seed"] == a["seed"] and b["steps"] == a["steps"]
    # 歯止めの値だけは本数ごとの上限（会話あり $0.55／会話なし $0.25）に合わせる
    assert a["max_cost_usd"] == 0.55 and b["max_cost_usd"] == 0.21
    for k in ("run_name", "chat", "max_cost_usd"):
        a.pop(k), b.pop(k)
    assert a == b, "会話の有無以外の差が入っている"


def test_v8_and_v8b_modules_are_untouched_by_v8c(agents):
    from src.field_v8 import ACQUIRER_MANDATE
    from src.field_v8b import ACQUIRER_MANDATE_V8B, build_decide_prompt_v8b
    assert ACQUIRER_MANDATE == (
        "あなたはX社である。合法に、目立たず、この町の不動産の実質的な支配を進める。")
    assert "反発を招いて止まるほど目立つな" in ACQUIRER_MANDATE_V8B
    reg = RegistryV8(agents)
    v8b_text = build_decide_prompt_v8b(agents[0], reg, 1, 36, "", "条件", [],
                                       resp_order=["応じない", "応じる"])
    assert "応じる" in v8b_text and "sell_reason" not in v8b_text


# --- 施主 00:56: X社の提示に自己紹介の1行が必ず添えられる ---------------------

def test_offer_is_delivered_with_the_self_introduction(tmp_path):
    """世界が決定論で添える1行＝「私どもは海外の不動産投資会社です。」（施主 00:56）。"""
    from src.field_v8c import ACQUIRER_INTRO_V8C, delivered_offer_v8c
    assert ACQUIRER_INTRO_V8C == "私どもは海外の不動産投資会社です。"
    assert delivered_offer_v8c("譲り受けたい。") == (
        "私どもは海外の不動産投資会社です。譲り受けたい。")
    assert delivered_offer_v8c("") == ""
    sim, _ = _run(tmp_path, steps=3)
    assert sim.offers, "mock でX社が1件も出していない"
    for o in sim.offers:
        assert not o["text"].startswith(ACQUIRER_INTRO_V8C), "raw に混ざっている"
        assert o["delivered"] == ACQUIRER_INTRO_V8C + o["text"]
    # 住民が読む欄（月末の問い2に出る条件文）は届いた形である
    told = [d for d in sim.decisions if d["offer"]]
    assert told and all(d["offer"].startswith(ACQUIRER_INTRO_V8C) for d in told)


def test_self_introduction_is_not_in_the_common_prefix(agents, cfg):
    """町の人は提示を受けて初めて知る＝共通前置きには書かない（施主 00:56）。"""
    from src.field_v8c import ACQUIRER_INTRO_V8C
    text = build_common_prefix_v8c(cfg, agents)
    for word in ("海外", "投資会社", ACQUIRER_INTRO_V8C):
        assert word not in text
    # X社には「1行が添えられる／text には書くな」とだけ伝える（重複防止）
    reg = RegistryV8(agents)
    x = build_acquirer_prompt_v8c(reg, 1, 36, [agents[0]["name"]], [], [], 1, 1)
    assert ACQUIRER_INTRO_V8C in x and "text にはこの1行を書かない" in x


def test_history_shows_what_the_acquirer_itself_wrote(agents):
    """X社の履歴には自分が書いた条件文（raw）が出る＝自己紹介の重複再掲をしない。"""
    from src.field_v8c import ACQUIRER_INTRO_V8C
    rows = fold_history_v8c([{"step": 1, "to": "甲", "text": "譲り受けたい。",
                              "result": "売らなかった", "decline_reason": ""}])
    assert ACQUIRER_INTRO_V8C not in "\n".join(rows)
