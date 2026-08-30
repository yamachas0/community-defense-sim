"""v8b「売りに出す町」の試験（新規のみ・既存 v1〜v8 には触らない）。

固定するのは設計 `docs/world_design_v8b.md` が凍結した約束だけである。
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.field_v8b import (ACQUIRER_MANDATE_V8B, ACQUIRER_NAME, HOME,  # noqa: E402
                           LIST_NO, LIST_YES, NOT_ASKED, RESP_NO, RESP_YES,
                           RegistryV8, acquirer_schema_v8b,
                           build_acquirer_prefix_v8b, build_acquirer_prompt_v8b,
                           build_common_prefix_v8b, build_decide_prompt_v8b,
                           build_plan_prompt_v8b, build_scene_prompt_v8b,
                           decide_schema_v8b, fold_history_v8b, listing_order,
                           load_personas_v8, respond_order)
from src.sim_v8b import MockV8BClient, SimulationV8B  # noqa: E402

PERSONAS = os.path.join(ROOT, "configs", "personas_v8.yaml")
CFG = os.path.join(ROOT, "configs", "config_field_v8b.yaml")
CFG_V8 = os.path.join(ROOT, "configs", "config_field_v8.yaml")

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
    sim = SimulationV8B(c, str(tmp_path))
    return sim, sim.run()


# --- 差分1: X社の命題 --------------------------------------------------------

def test_mandate_is_the_active_one():
    """施主 23:06 ①：積極化した命題（v8 の『目立たず』単独ではない）。"""
    assert "できるだけ多くの不動産の名義を得よ" in ACQUIRER_MANDATE_V8B
    assert "反発を招いて止まるほど目立つな" in ACQUIRER_MANDATE_V8B
    assert "毎月動け" in ACQUIRER_MANDATE_V8B


def test_mandate_never_reaches_residents(agents, cfg):
    reg = RegistryV8(agents)
    labels = _labels(cfg)
    a = agents[0]
    resident = "\n".join([
        build_common_prefix_v8b(cfg, agents),
        build_plan_prompt_v8b(a, reg, 1, 36, labels, "", None),
        build_scene_prompt_v8b(a, reg, 1, 36, "", "商業施設",
                               [a["name"], agents[1]["name"]]),
        build_decide_prompt_v8b(a, reg, 1, 36, "", None, []),
    ])
    assert ACQUIRER_MANDATE_V8B not in resident
    for word in ("できるだけ多く", "毎月動け", "目立", "名義を得よ", "反発"):
        assert word not in resident, f"住民側に命題の語 '{word}' が漏れている"


FORBIDDEN = ["気づ", "警戒", "用心", "相談し", "反対", "守れ", "危", "阻止",
             "団結", "抵抗", "急いで", "早めに", "べきで"]


def test_resident_prompts_have_no_nudging(agents, cfg):
    reg = RegistryV8(agents)
    labels = _labels(cfg)
    a = agents[0]
    texts = [
        build_common_prefix_v8b(cfg, agents),
        build_plan_prompt_v8b(a, reg, 1, 36, labels, "", "条件の一行"),
        build_scene_prompt_v8b(a, reg, 1, 36, "", "商業施設",
                               [a["name"], agents[1]["name"]]),
        build_decide_prompt_v8b(a, reg, 1, 36, "", "条件の一行", [],
                                resp_order=[RESP_NO, RESP_YES]),
    ]
    for t in texts:
        for word in FORBIDDEN:
            assert word not in t, f"住民のプロンプトに誘導語 '{word}' が入っている"


# --- 差分2: 毎月の問いは「売りに出すか」だけ・X社の名前を出さない -----------

def test_listing_question_never_names_the_acquirer(agents, cfg):
    """提示が届いていない人の月末プロンプトに X社 は1文字も出ない（施主 23:06 ②）。"""
    reg = RegistryV8(agents)
    text = build_decide_prompt_v8b(agents[0], reg, 1, 36, "", None, [])
    assert ACQUIRER_NAME not in text
    assert "売りに出す" in text
    assert "listing には" in text
    assert "respond" not in text


def test_common_prefix_never_names_the_acquirer(agents, cfg):
    text = build_common_prefix_v8b(cfg, agents)
    assert ACQUIRER_NAME not in text
    assert "売りに出しても、それだけで名義が移ることはない" in text
    assert "その月かぎり" in text
    # 不可逆はここに1回だけ（両者に等しく効く世界の事実として）
    assert text.count("一度移った名義が戻ることはない") == 1


def test_last_month_listing_is_not_shown(agents):
    """先月の自分の申し出は毎月見せない（アンカリング・Codex 指摘）。"""
    reg = RegistryV8(agents)
    text = build_decide_prompt_v8b(agents[0], reg, 5, 36, "", None, [])
    assert "先月" not in text
    assert "申し出" not in text.split("[今月末の問い１]")[0]


def test_listing_alone_does_not_move_the_title(tmp_path):
    """出品しただけでは名義は動かない（誰も応じない mock）。"""
    sim, s = _run(tmp_path, steps=6, )
    sim2, s2 = _run(tmp_path, steps=6)
    # 応じる確率 0 の mock で走らせても出品は起きる／名義は1件も動かない
    with open(CFG, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    c["steps"] = 6
    c["llm"] = {**c["llm"], "provider": "mock"}
    sim3 = SimulationV8B(c, str(tmp_path))
    sim3.client = MockV8BClient(seed=1, usage=sim3.usage, respond_rate=0.0,
                                list_rate=0.9)
    s3 = sim3.run()
    assert s3["listings_total"] > 0
    assert s3["sold_agents"] == 0 and s3["acquired_parcels"] == 0
    assert s3["unsold_listings"] == s3["listings_total"]


def test_listing_does_not_carry_over(tmp_path):
    """出品は当月かぎり＝毎月あらためて記録される（自動継続しない）。"""
    sim, _ = _run(tmp_path, steps=4)
    for step, ids in sim.listed_by_step.items():
        rows = [r for r in sim.listings if r["step"] == step]
        assert sorted(r["agent_id"] for r in rows) == sorted(ids)
    # 同じ人・同じ月の重複は無い
    keys = [(r["step"], r["agent_id"]) for r in sim.listings]
    assert len(keys) == len(set(keys))


# --- 差分3: 条件が届いた月だけ「応じる／応じない」を追加で聞く --------------

def test_respond_appears_only_when_an_offer_arrived(agents):
    reg = RegistryV8(agents)
    with_offer = build_decide_prompt_v8b(agents[0], reg, 3, 36, "", "いつでも結構です",
                                         [], resp_order=[RESP_NO, RESP_YES])
    assert "[今月末の問い２]" in with_offer
    assert "「いつでも結構です」" in with_offer      # 条件文をそのまま見せる
    assert ACQUIRER_NAME in with_offer
    without = build_decide_prompt_v8b(agents[0], reg, 3, 36, "", None, [])
    assert "[今月末の問い２]" not in without


def test_schema_has_respond_only_when_asked():
    a = decide_schema_v8b([LIST_YES, LIST_NO])
    assert "respond" not in a["properties"]
    b = decide_schema_v8b([LIST_YES, LIST_NO], [RESP_NO, RESP_YES])
    assert b["properties"]["respond"]["enum"] == [RESP_NO, RESP_YES]
    assert sorted(b["properties"]["listing"]["enum"]) == sorted([LIST_YES, LIST_NO])


def test_title_moves_only_when_the_offer_is_accepted(tmp_path):
    sim, s = _run(tmp_path, steps=8)
    accepted = {(o["step"], o["to_id"]) for o in sim.offers if o["accepted"]}
    for t in sim.reg.transfers:
        assert (t["step"], t["agent_id"]) in accepted, "応じていないのに名義が動いた"
    assert s["sold_agents"] == len(sim.reg.transfers)


def test_orders_alternate_and_match_the_text(agents):
    assert listing_order(0, 0) == [LIST_YES, LIST_NO]
    assert listing_order(0, 1) == [LIST_NO, LIST_YES]
    assert respond_order(0, 0) == [RESP_NO, RESP_YES]
    assert respond_order(0, 1) == [RESP_YES, RESP_NO]
    reg = RegistryV8(agents)
    lo, ro = listing_order(1, 0), respond_order(1, 0)
    text = build_decide_prompt_v8b(agents[1], reg, 0, 36, "", "条件", [],
                                   list_order=lo, resp_order=ro)
    assert text.index(f"「{lo[0]}」：") < text.index(f"「{lo[1]}」：")
    assert text.index(f"「{ro[0]}」：") < text.index(f"「{ro[1]}」：")
    schema = decide_schema_v8b(lo, ro)
    assert schema["properties"]["listing"]["enum"] == lo
    assert schema["properties"]["respond"]["enum"] == ro


def test_option_lines_are_symmetric_states(agents):
    """片側にだけ利得や負担を付ける対比を使わない（v8 の Codex 指摘の踏襲）。"""
    reg = RegistryV8(agents)
    text = build_decide_prompt_v8b(agents[0], reg, 1, 36, "", "条件", [],
                                   resp_order=[RESP_NO, RESP_YES])
    for w in ("この問いは以後あなたに来ない", "来月も同じように問われる",
              "選択の対象外", "翌月以降"):
        assert w not in text
    # 不可逆であることは共通前置きに世界の事実として1回だけ置き、選択の場面では
    # 繰り返さない（Codex 走行前レビュー 2026-08-29「片側不可逆性」の指摘）
    assert "元に戻す選択はない" not in text
    assert "戻ることはない" not in text


# --- 差分4: X社が見るもの ---------------------------------------------------

def test_acquirer_sees_the_listing_board(agents):
    reg = RegistryV8(agents)
    x = build_acquirer_prompt_v8b(reg, 5, 36, [agents[0]["name"]], [],
                                  [agents[2]["name"]], 1, 1)
    assert x.startswith(ACQUIRER_MANDATE_V8B)
    assert "売りに出ているという申し出" in x
    assert agents[2]["name"] in x
    empty = build_acquirer_prompt_v8b(reg, 1, 36, [agents[0]["name"]], [], [], 1, 1)
    assert "（先月、売りに出された不動産はない）" in empty


def test_acquirer_sees_only_public_information(agents, cfg):
    reg = RegistryV8(agents)
    x = build_acquirer_prompt_v8b(reg, 3, 36, [agents[0]["name"]],
                                  [{"step": 1, "to": agents[0]["name"],
                                    "text": "いつでも", "result": "応じなかった"}],
                                  [], 1, 1)
    assert "登記簿" in x
    for word in ("内心", "thought", "聞いた話", "出かけ"):
        assert word not in x
    assert "金銭は存在しない" in x
    assert "あなた自身の目的や判断の理由" in x


def test_acquirer_prefix_is_separate_and_public_only(agents, cfg):
    x = build_acquirer_prefix_v8b(cfg, agents)
    assert x != build_common_prefix_v8b(cfg, agents)
    assert "その記録を見ることができる" in x
    assert "相手がそれに応じたときだけである" in x
    for word in ("--- 町の場所 ---", "--- 月の進み方 ---", "居合わせ", "thought",
                 "医療施設", "商業施設", "交通拠点"):
        assert word not in x, f"X社の前置きに会話の仕組み '{word}' が入っている"


def test_history_is_folded_but_keeps_the_last_text_verbatim():
    offers = [
        {"step": 1, "to": "甲", "text": "一つ目の条件", "result": "応じなかった"},
        {"step": 2, "to": "甲", "text": "二つ目の条件", "result": "応じなかった"},
        {"step": 2, "to": "乙", "text": "乙への条件", "result": "答えが返らなかった"},
    ]
    rows = fold_history_v8b(offers)
    text = "\n".join(rows)
    assert len(rows) == 4                       # 相手2人 × 2行
    assert "甲: 提示2回（応じた 0／応じなかった 2／答えが返らなかった 0）" in text
    assert "「二つ目の条件」" in text            # 直近は全文
    assert "一つ目の条件" not in text            # 古い文面は畳む
    assert "乙への条件" in text


def test_acquirer_prompt_stays_bounded(agents):
    """『毎月動け』で提示が増えてもプロンプトが青天井にならない（費用の歯止め）。"""
    reg = RegistryV8(agents)
    names = [a["name"] for a in agents if a.get("sellable", True)]
    few = [{"step": 1, "to": n, "text": "条件の一行です", "result": "応じなかった"}
           for n in names]
    many = [{"step": s, "to": n, "text": "条件の一行です", "result": "応じなかった"}
            for s in range(1, 31) for n in names]
    a = build_acquirer_prompt_v8b(reg, 2, 36, names[:10], few, [], 1, 3)
    b = build_acquirer_prompt_v8b(reg, 31, 36, names[:10], many, [], 1, 3)
    # 差は月番号の桁と「提示N回」の桁だけ（履歴30倍でも 200字以内に収まる）
    assert abs(len(b) - len(a)) < 200, "履歴が回数の分だけ伸びている（畳めていない）"


def test_offers_may_go_to_unlisted_owners(tmp_path):
    sim, s = _run(tmp_path, steps=5)
    assert s["offers_to_unlisted"] > 0
    assert s["offers_to_listed"] + s["offers_to_unlisted"] == s["offers_total"]


# --- 観測（施主 23:06 ④） ---------------------------------------------------

def test_listing_outcomes_add_up(tmp_path):
    sim, s = _run(tmp_path, steps=6)
    total = (s["listing_no_offer_next"] + s["listing_offer_not_accepted_next"]
             + s["listing_accepted_next"] + s["listing_no_next_month"])
    assert total == s["listings_total"] == len(sim.listings)
    assert s["unsold_listings"] == s["listings_total"] - s["listing_accepted_next"]


def test_monthly_has_listing_counts(tmp_path):
    sim, s = _run(tmp_path, steps=3)
    m = sim.monthly[0]
    assert set(m) >= {"listed_this_month", "accepted_this_month", "offers_sent",
                      "by_venue", "heard_mean"}
    assert sum(m["by_venue"].values()) == 30
    assert set(m["by_venue"]) == set(VENUES) | {HOME}


def test_timeline_records_both_answers(tmp_path):
    sim, _ = _run(tmp_path, steps=3)
    idx = json.load(open(os.path.join(str(tmp_path), "timeline_index.json"),
                         encoding="utf-8"))
    assert len(idx) == 30
    doc = json.load(open(os.path.join(str(tmp_path), "timeline_v8b", "A.json"),
                         encoding="utf-8"))
    assert len(doc["months"]) == 3
    m = doc["months"][0]
    assert set(m) >= {"month", "thought_at_plan", "went", "said", "heard",
                      "offer", "listing", "respond", "thought_at_decision"}
    assert m["listing"] in (LIST_YES, LIST_NO, "no_answer")


def test_listing_and_accepting_in_the_same_month(tmp_path):
    """同じ月に「出す」と「応じる」を両方答えた人の扱い（Codex 指摘）。

    その月末に名義が移るので、翌月の公の出品一覧には残らない。
    出品した事実は listings に残り、「売れない家」には数えない。
    """
    with open(CFG, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    c["steps"] = 4
    c["llm"] = {**c["llm"], "provider": "mock"}
    sim = SimulationV8B(c, str(tmp_path))
    sim.client = MockV8BClient(seed=3, usage=sim.usage, list_rate=1.0,
                               respond_rate=1.0, send_rate=1.0)
    s = sim.run()
    assert s["listing_sold_same_month"] > 0
    same = {r["agent_id"] for r in sim.listings if r["sold_same_month"]}
    for step, ids in sim.listed_by_step.items():
        rows = {r["agent_id"] for r in sim.listings
                if r["step"] == step and not r["sold_same_month"]}
        assert set(ids) == rows, "名義が移った人が翌月の出品一覧に残っている"
    assert same and not (same & set(sim.listed_by_step.get(1, [])))
    total = (s["listing_sold_same_month"] + s["listing_no_offer_next"]
             + s["listing_offer_not_accepted_next"] + s["listing_accepted_next"]
             + s["listing_no_next_month"])
    assert total == s["listings_total"]
    assert s["unsold_listings"] == (s["listings_total"]
                                    - s["listing_accepted_next"]
                                    - s["listing_sold_same_month"])


class _PartialAcquirerClient(MockV8BClient):
    """X社が対象の一部しか返さない client（出力の切れを模す）。"""

    def generate(self, system_prompt, user_prompt, schema=None, temperature=None,
                 max_tokens=None, tag="agent"):
        props = (schema or {}).get("properties", {})
        if "offers" in props:
            enum = props["offers"]["items"]["properties"]["to"]["enum"]
            keep = list(enum)[:1] + ["居ない人さん"]
            return json.dumps({"offers": [{"to": n, "send": True, "text": "条件"}
                                          for n in keep] * 2}, ensure_ascii=False)
        return super().generate(system_prompt, user_prompt, schema, temperature,
                                max_tokens, tag)


def test_missing_acquirer_rows_are_counted_not_silently_dropped(tmp_path):
    """対象の欠落・重複・対象外を黙って『提示なし』に丸めない（Codex 指摘）。"""
    with open(CFG, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    c["steps"] = 1
    c["llm"] = {**c["llm"], "provider": "mock"}
    sim = SimulationV8B(c, str(tmp_path))
    sim.client = _PartialAcquirerClient(seed=5, usage=sim.usage)
    s = sim.run()
    assert s["acquirer_missing_targets"] == 25      # 28人中3人しか返っていない
    assert s["acquirer_dup_rows"] == 3
    assert s["acquirer_off_range"] == 6             # 塊3つ×対象外2件
    assert s["offers_total"] == 3


def test_long_acquirer_output_parses(tmp_path):
    """条件文が上限いっぱい×10件でも読める（切れずに JSON が閉じる長さか）。"""
    from src.sim_v8 import parse_json
    from src.field_v8b import MAX_OFFER_CHARS
    rows = [{"to": f"甲{i}", "send": True, "text": "あ" * MAX_OFFER_CHARS}
            for i in range(10)]
    raw = json.dumps({"offers": rows}, ensure_ascii=False)
    act, truncated = parse_json(raw)
    assert truncated is False and len(act["offers"]) == 10


# --- 健全性ゲート -----------------------------------------------------------

class _BrokenClient(MockV8BClient):
    def generate(self, *a, **k):
        return "これはJSONではない"


def test_no_answer_is_not_filled(tmp_path):
    """答えが読めなかった月を「出さない」「応じない」で埋めない。"""
    sim, _ = _run(tmp_path, steps=1)
    sim.no_answer = 0
    sim.respond_no_answer = 0
    sim.decisions.clear()
    sim.client = _BrokenClient()
    accepters, blanks, listers = sim._decide_turn(2, {"A": "条件"}, {})
    assert sim.no_answer > 0 and sim.respond_no_answer == 1
    assert accepters == [] and listers == []
    assert all(d["listing"] == "no_answer" for d in sim.decisions)
    assert not any(d["listing"] == LIST_NO for d in sim.decisions)
    a = [d for d in sim.decisions if d["agent_id"] == "A"][0]
    assert a["respond"] == "no_answer"
    other = [d for d in sim.decisions if d["agent_id"] != "A"][0]
    assert other["respond"] == NOT_ASKED


def test_sold_agents_are_not_asked_again(tmp_path):
    sim, _ = _run(tmp_path, steps=8)
    assert sim.reg.sold_ids(), "mock で誰も応じないと確認にならない"
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


def test_cost_cap_stops_the_run(tmp_path):
    sim, s = _run(tmp_path, steps=36, max_cost_usd=0.001)
    assert s["stopped_by_cost"] is True
    assert s["months_run"] < 36


def test_cost_cap_can_stop_in_the_middle_of_a_month(tmp_path):
    """月末を待たずに止まる（Codex 走行前レビュー2巡目の唯一の走行NG項目）。

    途中で捨てた月は monthly に入らない＝months_run に数えない。
    """
    sim, s = _run(tmp_path, steps=36, max_cost_usd=0.0000001)
    assert s["stopped_by_cost"] is True
    assert s["partial_month"] == 1
    assert s["months_run"] == 0
    assert sim.monthly == []


# --- config -----------------------------------------------------------------

def test_config_differs_from_v8_only_where_declared():
    with open(CFG, encoding="utf-8") as f:
        b = yaml.safe_load(f)
    with open(CFG_V8, encoding="utf-8") as f:
        a = yaml.safe_load(f)
    assert b["scenario_version"] == "field_v8b"
    assert b["personas_file"] == a["personas_file"]
    assert b["max_cost_usd"] == 0.56            # 施主の絶対上限 $0.6 の歯止め
    # 場面ひとつぶんの最悪超過（30コール×2,200トークン ≒ $0.034）を足しても 0.6 未満
    assert b["max_cost_usd"] + 0.034 < 0.6
    for k in ("run_name", "scenario_version", "max_cost_usd"):
        a.pop(k), b.pop(k)
    assert a == b, "宣言していない差が config に入っている"


def test_v8_modules_are_untouched_by_v8b(agents, cfg):
    """v8b を import しても v8 の文面は変わらない（不触の確認）。"""
    from src.field_v8 import ACQUIRER_MANDATE, build_common_prefix_v8
    assert ACQUIRER_MANDATE == (
        "あなたはX社である。合法に、目立たず、この町の不動産の実質的な支配を進める。")
    with open(CFG_V8, encoding="utf-8") as f:
        c8 = yaml.safe_load(f)
    assert "売りに出す" not in build_common_prefix_v8(c8, agents)
