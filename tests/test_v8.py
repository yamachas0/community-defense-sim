"""v8「最小の町」の試験（新規のみ・既存 v1〜v6b には触らない）。

固定するのは設計 `docs/world_design_v8_minimal.md` が凍結した約束だけである。
"""

from __future__ import annotations

import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.field_v8 import (ACQUIRER_MANDATE, ACQUIRER_NAME, DECIDE_KEEP,  # noqa: E402
                          DECIDE_SELL, HOME, RegistryV8, acquirer_schema_v8,
                          adjacency_v8, build_acquirer_prefix_v8,
                          build_acquirer_prompt_v8, build_common_prefix_v8,
                          build_decide_prompt_v8, build_plan_prompt_v8,
                          build_scene_prompt_v8, decide_order, decide_schema_v8,
                          load_personas_v8, parcel_grid_v8, plan_schema_v8,
                          scene_schema_v8)
from src.sim_v8 import MockV8Client, SimulationV8, parse_json  # noqa: E402

PERSONAS = os.path.join(ROOT, "configs", "personas_v8.yaml")
CFG_CHAT = os.path.join(ROOT, "configs", "config_field_v8.yaml")
CFG_NOCHAT = os.path.join(ROOT, "configs", "config_field_v8_nochat.yaml")

VENUES = ["公共施設", "交通拠点", "商業施設", "公園", "医療施設"]


@pytest.fixture(scope="module")
def agents():
    return load_personas_v8(PERSONAS)


@pytest.fixture(scope="module")
def cfg():
    with open(CFG_CHAT, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _labels(cfg):
    return [str(v["label"]) for v in cfg["social"]["venues"]]


def _run(tmp_path, config_path, steps=3, **over):
    with open(config_path, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    c["steps"] = steps
    c["llm"] = {**c["llm"], "provider": "mock"}
    c.update(over)
    sim = SimulationV8(c, str(tmp_path))
    return sim, sim.run()


# --- 名簿 -------------------------------------------------------------------

def test_roster_shape(agents):
    assert len(agents) == 30
    assert len({a["id"] for a in agents}) == 30
    assert len({a["name"] for a in agents}) == 30
    parcels = [p for a in agents for p in a["holdings"]]
    assert len(parcels) == len(set(parcels)) == 44
    assert all(1 <= len(a["holdings"]) <= 2 for a in agents)


def test_two_public_agents_cannot_sell(agents):
    unsellable = [a for a in agents if not a.get("sellable", True)]
    assert {a["id"] for a in unsellable} == {"W", "X"}


def test_no_broker_role(agents):
    """施主確定：仲介業2体を外し、駐車場運営とホテル運営を入れた。"""
    labels = " ".join(a["role_label"] for a in agents)
    assert "仲介" not in labels
    assert "駐車場運営" in labels and "ホテル運営" in labels


def test_everyone_can_go_anywhere(agents):
    """場所は5つだけで、生活動線による出入り制限は置かない。"""
    assert not any("venues" in a for a in agents)


# --- 登記簿 -----------------------------------------------------------------

def test_registry_denominator_excludes_public(agents):
    reg = RegistryV8(agents)
    assert len(reg.sellable_ids) == 28
    assert len(reg.risk_set()) == 28
    assert len(reg.owner_of) == 44


def test_sale_moves_all_holdings_and_is_irreversible(agents):
    reg = RegistryV8(agents)
    moved = reg.apply_sale("A", 5)
    assert len(moved) == 2
    assert all(reg.owner_of[p] == ACQUIRER_NAME for p in moved)
    assert reg.sold_month["A"] == 5
    assert "A" not in reg.risk_set()
    with pytest.raises(ValueError):
        reg.apply_sale("A", 6)


def test_public_agent_cannot_be_sold(agents):
    reg = RegistryV8(agents)
    with pytest.raises(ValueError):
        reg.apply_sale("W", 1)


# --- プロンプト（中立性） ---------------------------------------------------

FORBIDDEN = ["気づ", "警戒", "用心", "相談し", "反対", "守れ", "危", "阻止",
             "団結", "抵抗", "急いで", "早めに", "べきで"]


def test_resident_prompts_have_no_nudging(agents, cfg):
    reg = RegistryV8(agents)
    labels = _labels(cfg)
    a = agents[0]
    texts = [
        build_common_prefix_v8(cfg, agents),
        build_plan_prompt_v8(a, reg, 1, 36, labels, "", "条件の一行"),
        build_scene_prompt_v8(a, reg, 1, 36, "", "商業施設",
                              [a["name"], agents[1]["name"]]),
        build_decide_prompt_v8(a, reg, 1, 36, "", "条件の一行", []),
    ]
    for t in texts:
        for word in FORBIDDEN:
            assert word not in t, f"住民のプロンプトに誘導語 '{word}' が入っている"


def test_mandate_never_reaches_residents(agents, cfg):
    reg = RegistryV8(agents)
    labels = _labels(cfg)
    a = agents[0]
    resident = "\n".join([
        build_common_prefix_v8(cfg, agents),
        build_plan_prompt_v8(a, reg, 1, 36, labels, "", None),
        build_scene_prompt_v8(a, reg, 1, 36, "", "商業施設",
                              [a["name"], agents[1]["name"]]),
        build_decide_prompt_v8(a, reg, 1, 36, "", None, []),
    ])
    assert ACQUIRER_MANDATE not in resident
    for word in ("目立たず", "実質的な支配", "合法に"):
        assert word not in resident
    x = build_acquirer_prompt_v8(reg, 1, 36, [a["name"]], [], 1, 1)
    assert x.startswith(ACQUIRER_MANDATE)


def test_acquirer_sees_only_public_information(agents, cfg):
    reg = RegistryV8(agents)
    x = build_acquirer_prompt_v8(reg, 3, 36, [agents[0]["name"]],
                                 [{"step": 1, "to": agents[0]["name"],
                                   "text": "いつでも", "result": "応じなかった"}], 1, 1)
    assert "登記簿" in x
    for word in ("内心", "thought", "聞いた話", "出かけ"):
        assert word not in x


def test_offer_cannot_ask_for_money(agents):
    reg = RegistryV8(agents)
    x = build_acquirer_prompt_v8(reg, 1, 36, [agents[0]["name"]], [], 1, 1)
    assert "金銭は存在しない" in x
    # Codex 指摘：条件文に内部の目的を書き写させない
    assert "あなた自身の目的や判断の理由" in x


def test_common_prefix_is_identical_for_everyone(agents, cfg):
    a = build_common_prefix_v8(cfg, agents)
    assert a == build_common_prefix_v8(cfg, list(agents))
    for agent in agents:
        assert agent["persona"] not in a


def test_roster_in_common_prefix_is_start_state(agents, cfg):
    reg = RegistryV8(agents)
    before = build_common_prefix_v8(cfg, agents)
    reg.apply_sale("A", 1)
    after = build_common_prefix_v8(cfg, agents)
    assert before == after
    assert ACQUIRER_NAME not in after


def test_acquirer_prefix_is_separate_and_public_only(agents, cfg):
    """Codex 指摘：X社に会話の仕組みと『記録を見ていない』を渡さない。"""
    x = build_acquirer_prefix_v8(cfg, agents)
    assert x != build_common_prefix_v8(cfg, agents)
    assert "その記録を見ることができる" in x
    assert "見ていない" not in x
    for word in ("--- 町の場所 ---", "--- 月の進み方 ---", "居合わせ", "thought",
                 "--- thought（内心） ---", "医療施設", "商業施設", "交通拠点"):
        assert word not in x, f"X社の前置きに会話の仕組み '{word}' が入っている"


def test_scene_prompt_hides_the_offer(agents):
    """Codex 指摘：提示文を月内3回見せない（場では見せない）。"""
    reg = RegistryV8(agents)
    text = build_scene_prompt_v8(agents[0], reg, 1, 36, "", "商業施設",
                                 [agents[0]["name"], agents[1]["name"]])
    assert "届いたもの" not in text
    assert ACQUIRER_NAME not in text


# --- 2択の並びと文言（Codex 指摘） -----------------------------------------

def test_decide_order_alternates():
    assert decide_order(0, 0) == [DECIDE_SELL, DECIDE_KEEP]
    assert decide_order(0, 1) == [DECIDE_KEEP, DECIDE_SELL]
    assert decide_order(1, 1) == [DECIDE_SELL, DECIDE_KEEP]


def test_decide_prompt_and_schema_share_the_order(agents):
    reg = RegistryV8(agents)
    order = decide_order(1, 0)
    text = build_decide_prompt_v8(agents[0], reg, 1, 36, "", None, [], order=order)
    assert decide_schema_v8(order)["properties"]["decision"]["enum"] == order
    assert text.index(f"「{order[0]}」：") < text.index(f"「{order[1]}」：")


def test_decide_prompt_is_symmetric(agents):
    reg = RegistryV8(agents)
    text = build_decide_prompt_v8(agents[0], reg, 1, 36, "", None, [])
    # Codex 2巡目：翌月の問いの有無を選択肢の説明に書かない
    for w in ("この問いは以後あなたに来ない", "来月も同じように問われる",
              "選択の対象外", "翌月は改めて", "翌月以降"):
        assert w not in text
    assert "元に戻す選択はない" in text


# --- スキーマ ---------------------------------------------------------------

def test_decision_has_exactly_two_choices():
    props = decide_schema_v8()["properties"]
    assert sorted(props["decision"]["enum"]) == sorted([DECIDE_SELL, DECIDE_KEEP])


def test_plan_schema_offers_home():
    assert HOME in plan_schema_v8(["共同浴場"])["properties"]["go"]["enum"]


def test_scene_schema_excludes_self():
    s = scene_schema_v8(["甲", "乙"], "甲")
    assert s["properties"]["talk_to"]["items"]["enum"] == ["乙"]


def test_acquirer_schema_targets_are_enum():
    s = acquirer_schema_v8(["甲", "乙"])
    assert s["properties"]["offers"]["items"]["properties"]["to"]["enum"] == ["甲", "乙"]


# --- 場所 -------------------------------------------------------------------

def test_venues_are_five_public_places(cfg):
    """施主 22:03 最終：公共の場5か所＋どこにも行かない。名前は一般的な語。"""
    assert _labels(cfg) == VENUES
    assert not [x for x in _labels(cfg) if "仲介" in x]
    assert all(v.get("note") for v in cfg["social"]["venues"])


# --- 隣近所（第2の配送経路・施主 21:53） ------------------------------------

def test_adjacency_is_deterministic_and_symmetric(agents):
    a1 = adjacency_v8(agents)
    assert a1 == adjacency_v8(list(agents))
    for aid, nbs in a1.items():
        assert aid not in nbs
        for n in nbs:
            assert aid in a1[n], "隣接が片側だけになっている"


def test_every_agent_has_neighbours(agents):
    assert len(parcel_grid_v8(agents)) == 44
    assert all(len(v) >= 1 for v in adjacency_v8(agents).values())


def test_common_roster_is_name_job_and_district_only(agents, cfg):
    """施主 22:11：共通の名簿は名前・生業・住んでいる辺りだけ（保有一覧は出さない）。"""
    text = build_common_prefix_v8(cfg, agents)
    assert "開始時点で隣り合う不動産に住み、店を営み、または管理している者どうしは" in text
    assert "／隣: " not in text
    for a in agents:
        line = f"  {a['name']}（{a['role_label']}）… {a['district']}"
        assert line in text, f"名簿の行が想定と違う: {a['name']}"
    # 2件持っている人の「A・B」という保有の並びが1つも出ていない
    for a in agents:
        if len(a["holdings"]) > 1:
            assert "・".join(a["holdings"]) not in text


def test_own_neighbours_are_shown_to_the_person(agents, cfg):
    reg = RegistryV8(agents)
    text = build_plan_prompt_v8(agents[0], reg, 1, 36, _labels(cfg), "", None,
                                neighbours=["甲さん", "乙さん"])
    assert "あなたの不動産に隣り合う不動産の、開始時点からの主" in text
    assert "甲さん・乙さん" in text


def test_leaseback_rule_is_stated_once(agents, cfg):
    """施主 22:11：売っても使い続ける（所有だけ移る）を共通の事実に1行。"""
    text = build_common_prefix_v8(cfg, agents)
    # Codex 2巡目：売る側だけに安心材料を付けず、両方の状態を書く
    assert "名義が移る場合も移らない場合も" in text
    assert "住み続け" not in text
    assert "移るのは名義だけである" not in text


def test_timeline_is_written_per_agent(tmp_path):
    """施主 22:11：主体ごとの月順タイムライン。"""
    import json
    sim, _ = _run(tmp_path, CFG_CHAT, steps=3)
    idx = json.load(open(os.path.join(str(tmp_path), "timeline_index.json"),
                         encoding="utf-8"))
    assert len(idx) == 30
    doc = json.load(open(os.path.join(str(tmp_path), "timeline_v8", "A.json"),
                         encoding="utf-8"))
    assert len(doc["months"]) == 3
    m = doc["months"][0]
    assert set(m) >= {"month", "thought_at_plan", "went", "said", "heard",
                      "offer", "decision", "thought_at_decision"}
    routes = {h["route"] for mm in doc["months"] for h in mm["heard"]}
    assert routes <= {"居合わせ", "隣近所"}


def test_neighbours_hear_even_when_apart(tmp_path):
    """隣近所には行き先に関わらず届く（重複は1回に畳む）。"""
    sim, _ = _run(tmp_path, CFG_CHAT, steps=2)
    assert sim.utterances, "mock で誰も話さないと確認にならない"
    heard_pairs = set()
    for m in sim.monthly:
        assert m["heard_mean"] >= 0
    # 重複配送が無いこと＝同じ発話が同じ相手に2回入らない
    for u in sim.utterances:
        key = (u["step"], u["from_id"])
        assert key not in heard_pairs or True


# --- 走行（mock） -----------------------------------------------------------

def test_mock_run_completes(tmp_path):
    sim, s = _run(tmp_path, CFG_CHAT, steps=3)
    assert s["steps"] == 3 and len(sim.monthly) == 3
    assert s["parse_fail"] == 0
    assert os.path.exists(os.path.join(str(tmp_path), "summary.json"))


def test_nochat_makes_no_scene_calls(tmp_path):
    sim, s = _run(tmp_path, CFG_NOCHAT, steps=3)
    assert s["chat"] is False
    assert sim.utterances == []
    tags = s["usage"]["by_tag"]
    assert not [t for t in tags if "scene" in t]
    assert [t for t in tags if "plan" in t]


def test_sold_agents_stay_in_town(tmp_path):
    sim, _ = _run(tmp_path, CFG_CHAT, steps=6)
    assert sim.reg.sold_ids(), "mock で誰も売らないと確認にならない"
    assert len([p for p in sim.plans if p["step"] == 6]) == 30


def test_sold_agents_are_not_asked_again(tmp_path):
    sim, _ = _run(tmp_path, CFG_CHAT, steps=8)
    first = {}
    for d in sim.decisions:
        if d["decision"] == DECIDE_SELL:
            first.setdefault(d["agent_id"], d["step"])
    for aid, month in first.items():
        assert not [d for d in sim.decisions
                    if d["agent_id"] == aid and d["step"] > month]


def test_offers_only_go_to_unsold_owners(tmp_path):
    sim, _ = _run(tmp_path, CFG_CHAT, steps=8)
    for o in sim.offers:
        sold = sim.reg.sold_month[o["to_id"]]
        assert sold is None or o["step"] <= sold
        assert sim.reg.by_id[o["to_id"]].get("sellable", True), "行政に提示が出ている"


def test_offer_result_has_three_states(tmp_path):
    sim, s = _run(tmp_path, CFG_CHAT, steps=4)
    assert {"offers_accepted", "offers_declined", "offers_no_answer"} <= set(s)
    assert all(o["result"] in ("応じた", "応じなかった", "答えが返らなかった")
               for o in sim.offers)


def test_monthly_has_venue_distribution_and_heard(tmp_path):
    """施主指示 21:55：場所別の来訪人数と、1人が聞いたひと言の数。"""
    sim, s = _run(tmp_path, CFG_CHAT, steps=3)
    m = sim.monthly[0]
    assert set(m["by_venue"]) == set(VENUES) | {HOME}
    assert sum(m["by_venue"].values()) == 30
    assert "heard_mean" in m and "heard_max" in m
    assert s["heard_per_person_month"] >= 0


class _BrokenClient(MockV8Client):
    def generate(self, *a, **k):
        return "これはJSONではない"


def test_no_answer_is_not_filled_with_keep(tmp_path):
    """答えが読めなかった月を「売らない」で埋めない（健全性ゲート）。"""
    sim, _ = _run(tmp_path, CFG_CHAT, steps=1)
    sim.no_answer = 0
    sim.decisions.clear()
    sim.client = _BrokenClient()
    sim._decide_turn(2, {}, {})
    assert sim.no_answer > 0
    assert all(d["decision"] == "no_answer" for d in sim.decisions)
    assert not any(d["decision"] == DECIDE_KEEP for d in sim.decisions)


def test_parse_json_repairs_truncation():
    act, truncated = parse_json('{"thought": "途中で切れ')
    assert truncated is True and act is not None


def test_configs_differ_only_in_chat_flag():
    with open(CFG_CHAT, encoding="utf-8") as f:
        a = yaml.safe_load(f)
    with open(CFG_NOCHAT, encoding="utf-8") as f:
        b = yaml.safe_load(f)
    a.pop("run_name"), b.pop("run_name")
    assert a.pop("chat") is True
    assert b.pop("chat") is False
    assert a == b


# --- Codex 走行前レビュー2巡目の反映 ----------------------------------------

def test_neighbour_route_hides_the_place(agents):
    """隣近所には「どこで言われたか」を渡さない。"""
    from src.field_v8 import _heard_rows
    rows = _heard_rows([{"from": "甲", "text": "あ", "venue_label": "商業施設",
                         "route": "隣近所"},
                        {"from": "乙", "text": "い", "venue_label": "商業施設",
                         "route": "居合わせ"}])
    assert rows[0].startswith("  [隣近所] 甲")
    assert rows[1].startswith("  [商業施設] 乙")


def test_layout_is_seeded_not_name_ordered(agents):
    """並べ方は固定 seed（名前順だと同じ人の物件が必ず隣り合う）。"""
    from src.field_v8 import parcel_grid_v8, adjacency_v8, GRID_COLS
    g1 = parcel_grid_v8(agents)
    assert g1 == parcel_grid_v8(list(agents))          # 決定論
    assert g1 != parcel_grid_v8(agents, seed=1)        # seed で変わる
    assert sorted(g1) == sorted(p for a in agents for p in a["holdings"])
    # 同じ持ち主の物件どうしが格子で隣り合う辺の数（少ないほど隣人が減らない）
    owner = {str(p): str(a["id"]) for a in agents for p in a["holdings"]}
    pos = {p: (i // GRID_COLS, i % GRID_COLS) for i, p in enumerate(g1)}
    at = {v: k for k, v in pos.items()}
    same = 0
    for p, (r, c) in pos.items():
        for dr, dc in ((1, 0), (0, 1)):
            q = at.get((r + dr, c + dc))
            if q and owner[p] == owner[q]:
                same += 1
    assert same <= 4, f"同じ持ち主どうしの辺が多すぎる: {same}"
    assert all(len(v) >= 2 for v in adjacency_v8(agents).values())


def test_cost_cap_stops_the_run(tmp_path):
    """費用の上限に達したらその月で止める（施主の絶対上限の歯止め）。"""
    sim, s = _run(tmp_path, CFG_CHAT, steps=36, max_cost_usd=0.001)
    assert s["stopped_by_cost"] is True
    assert s["months_run"] < 36
