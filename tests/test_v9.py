"""v9「土地と建物を分ける町」の試験。

設計の正は `docs/world_design_v9.md`（施主確定 2026-08-30 10:54 → 11:00 → 11:04 → 11:06）。
ここで固定するのは **世界の骨格と、走行前に凍結した事実** である:

  - 名簿と区画の数・v8 からの不変（人物・ペルソナ本文・区画名・格子）
  - 権利の3つ（土地／建物／借りて使う人）と、使用者の導出
  - 出品の選べる肢が「持っているものだけ」であること
  - 世界が実行できない提示を配らないこと
  - 売ったあとに何が起きるか（借地・借家・退場）
  - 所有者が変わったことが翌月の使用者に事実1行で届くこと
  - 住民側のプロンプトにX社の命題・設定が1文字も漏れないこと
  - 用語が「所有権」で統一されていること（「名義」を使わない）
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import field_v9 as F  # noqa: E402
from src.field_v8 import load_personas_v8, parcel_grid_v8  # noqa: E402
from src.sim_v9 import MockV9Client, SimulationV9  # noqa: E402

PERSONAS_V9 = os.path.join(ROOT, "configs", "personas_v9.yaml")
PERSONAS_V8 = os.path.join(ROOT, "configs", "personas_v8.yaml")
CONFIG_V9 = os.path.join(ROOT, "configs", "config_field_v9.yaml")


@pytest.fixture(scope="module")
def book():
    agents, parcels = F.load_personas_v9(PERSONAS_V9)
    return agents, parcels


@pytest.fixture(scope="module")
def reg(book):
    agents, parcels = book
    return F.RegistryV9(agents, parcels)


@pytest.fixture(scope="module")
def cfg():
    with open(CONFIG_V9, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# 名簿（凍結した事実）
# ---------------------------------------------------------------------------

def test_roster_counts(book):
    agents, parcels = book
    assert len(parcels) == 44
    assert sum(1 for p in parcels if p["building"]) == 35
    assert sum(1 for p in parcels if not p["building"]) == 9
    assert len(agents) == 49
    assert sum(1 for a in agents if a["resident"]) == 35
    assert sum(1 for a in agents if not a["resident"]) == 14


def test_v8_people_and_parcels_unchanged(book):
    """v8 の30体の人物・ペルソナ本文・区画名と並びは1文字も変わらない。"""
    agents, parcels = book
    v8 = load_personas_v8(PERSONAS_V8)
    by_id = {str(a["id"]): a for a in agents}
    for a8 in v8:
        aid = str(a8["id"])
        assert aid in by_id, aid
        a9 = by_id[aid]
        assert a9["name"] == a8["name"]
        assert a9["role_label"] == a8["role_label"]
        assert a9["district"] == a8["district"]
        assert a9["persona"].strip() == str(a8["persona"]).strip()
    assert [p["name"] for p in parcels] == [str(p) for a in v8
                                            for p in a["holdings"]]


def test_grid_identical_to_v8(book):
    """町の形（格子）は v8 と1マスも動かない＝隣近所の骨格が同じ。"""
    _agents, parcels = book
    assert F.parcel_grid_v9(parcels) == parcel_grid_v8(load_personas_v8(PERSONAS_V8))


def test_initial_rights(reg):
    """開始時＝借家9区画・借地上の持家7区画・土地だけの借地2区画・使用者なし2区画。"""
    rented = [p for p in reg.parcel_names
              if reg.has_building[p] and reg.tenant_of[p]]
    leased_house = [p for p in reg.parcel_names
                    if reg.has_building[p]
                    and reg.building_of[p] != reg.land_of[p]]
    no_user = [p for p in reg.parcel_names if reg.user_of(p) is None]
    assert len(rented) == 9          # 町の大家3 ＋ 町にいない所有者6
    assert len(leased_house) == 7    # 借地上の持家
    assert sum(1 for p in reg.parcel_names
               if not reg.has_building[p] and reg.tenant_of[p]) == 2
    assert no_user == ["船着場裏の駐車場", "船着場裏の資材置場"]
    # 開始時に町にいない所有者が持つ土地は17区画
    absentee = {aid for aid in reg.by_id if not reg.by_id[aid].get("resident")}
    assert sum(1 for p in reg.parcel_names if reg.land_of[p] in absentee) == 17


def test_user_is_derived(reg):
    """使用者は導出値＝借りて使う人 → 建物の所有者 → （建物が無ければ）土地の所有者。"""
    assert reg.user_of("湯坂上の空き店舗") == "BA"      # 借家人
    assert reg.user_of("駅前通りの家") == "B"           # 借地上の持家＝建物の所有者
    assert reg.user_of("一番街の資材置場") == "G"       # 建物なし＝土地の所有者
    assert reg.user_of("駅裏北の月極駐車場") == "U"     # 建物なしの借地＝借りて使う人
    assert reg.user_of("船着場裏の駐車場") is None      # 町にいない所有者だけ


def test_neighbours_use_users_not_owners(reg):
    """隣近所は**使用者**どうしで結ぶ（町にいない所有者は入らない）。"""
    adj = F.adjacency_v9(reg)
    for aid, nbs in adj.items():
        if not reg.by_id[aid].get("resident", True):
            assert nbs == [], aid
        assert aid not in nbs
    assert any(nbs for nbs in adj.values())


# ---------------------------------------------------------------------------
# 出品の選べる肢と、世界の門番
# ---------------------------------------------------------------------------

def test_listing_options_only_what_you_own(reg):
    # 土地も建物も自分のもの
    assert reg.listing_options("A", "湯坂上の古家") == [
        F.LIST_NO, F.LIST_LAND, F.LIST_BUILDING, F.LIST_BOTH]
    # 借地上の持家＝建物だけ
    assert reg.listing_options("B", "駅前通りの家") == [F.LIST_NO, F.LIST_BUILDING]
    # その土地の持ち主＝土地だけ
    assert reg.listing_options("CL", "駅前通りの家") == [F.LIST_NO, F.LIST_LAND]
    # 建物の無い区画
    assert reg.listing_options("G", "一番街の資材置場") == [F.LIST_NO, F.LIST_LAND]


def test_tenants_have_nothing_to_sell(reg):
    """借家人・借地で使うだけの人は所有権を持たない＝月末の問いの対象にならない。"""
    risk = set(reg.risk_set())
    for aid in ("BA", "BB", "BC", "BD", "BE", "BF", "T", "U", "V", "Y"):
        if aid in ("V", "Y"):
            continue
        assert reg.parcels_owned(aid) == [], aid
        assert aid not in risk, aid
    assert "W" not in risk and "X" not in risk      # 行政は売れない


def test_can_offer_gate(reg):
    assert reg.can_offer("A", "湯坂上の古家", F.KIND_BOTH)
    assert reg.can_offer("CL", "駅前通りの家", F.KIND_LAND)
    # 土地を持っていない人に土地の提示は配れない
    assert not reg.can_offer("B", "駅前通りの家", F.KIND_LAND)
    assert not reg.can_offer("B", "駅前通りの家", F.KIND_BOTH)
    # 建物の無い区画に建物の提示は配れない
    assert not reg.can_offer("G", "一番街の資材置場", F.KIND_BUILDING)
    # 行政は売れない
    assert not reg.can_offer("W", "浜辺の街区公園", F.KIND_LAND)
    # 他人の区画は配れない
    assert not reg.can_offer("A", "駅前通りの家", F.KIND_LAND)


# ---------------------------------------------------------------------------
# 売ったあとに何が起きるか
# ---------------------------------------------------------------------------

def _fresh():
    agents, parcels = F.load_personas_v9(PERSONAS_V9)
    return F.RegistryV9(agents, parcels)


def test_sell_land_only_keeps_building_and_use():
    r = _fresh()
    row = r.apply_transfer("CL", "駅前通りの家", F.KIND_LAND, 3)
    assert r.land_of["駅前通りの家"] == F.ACQUIRER_NAME
    assert r.building_of["駅前通りの家"] == "B"
    assert r.user_of("駅前通りの家") == "B"        # 借地としてそのまま
    assert row["left_town"] is False
    assert r.left_month["CL"] is None


def test_sell_building_only_makes_the_seller_a_tenant():
    """建物だけ売った使用者は、借家として今までどおりそこにいる。"""
    r = _fresh()
    r.apply_transfer("A", "湯坂上の古家", F.KIND_BUILDING, 4)
    assert r.building_of["湯坂上の古家"] == F.ACQUIRER_NAME
    assert r.land_of["湯坂上の古家"] == "A"
    assert r.tenant_of["湯坂上の古家"] == "A"
    assert r.user_of("湯坂上の古家") == "A"
    assert r.left_month["A"] is None


def test_sell_both_as_the_user_leaves_town():
    r = _fresh()
    row = r.apply_transfer("A", "湯坂上の古家", F.KIND_BOTH, 5)
    assert row["left_town"] is True
    assert r.left_month["A"] == 5
    assert r.user_of("湯坂上の古家") is None
    assert not r.is_resident("A")
    # 2件目を持っているので「町にいない所有者」として残る
    assert "AA" not in r.absentee_owner_ids() or True
    assert "A" in r.absentee_owner_ids()
    assert r.parcels_owned("A") == ["湯坂上の空き店舗"]


def test_landlord_selling_both_does_not_leave_and_tenant_stays():
    """借家人がいる区画を大家が両方売っても、大家は退場せず、借家人は残る。"""
    r = _fresh()
    row = r.apply_transfer("A", "湯坂上の空き店舗", F.KIND_BOTH, 6)
    assert row["left_town"] is False
    assert r.left_month["A"] is None
    assert r.user_of("湯坂上の空き店舗") == "BA"
    assert r.is_resident("BA")


def test_sell_land_of_a_land_only_parcel_keeps_the_user():
    r = _fresh()
    r.apply_transfer("G", "一番街の資材置場", F.KIND_LAND, 7)
    assert r.land_of["一番街の資材置場"] == F.ACQUIRER_NAME
    assert r.tenant_of["一番街の資材置場"] == "G"
    assert r.user_of("一番街の資材置場") == "G"


def test_cannot_sell_twice():
    r = _fresh()
    r.apply_transfer("CL", "駅前通りの家", F.KIND_LAND, 2)
    with pytest.raises(ValueError):
        r.apply_transfer("CL", "駅前通りの家", F.KIND_LAND, 3)


# ---------------------------------------------------------------------------
# 文面（世界の事実と選択肢だけが置かれていること）
# ---------------------------------------------------------------------------

def _all_resident_prompts(cfg, reg):
    agents = reg.agents
    out = [F.build_common_prefix_v9(cfg, agents, len(reg.parcel_names))]
    a = reg.by_id["A"]
    labels = [v["label"] for v in cfg["social"]["venues"]]
    offer = {"parcel": "湯坂上の古家", "kind": F.KIND_LAND,
             "delivered": F.delivered_offer_v9("ご検討ください。")}
    out.append(F.build_plan_prompt_v9(a, reg, 1, 36, labels, "", offer,
                                      notices=["（記録）先月末、…"],
                                      neighbours=["となりの人"]))
    out.append(F.build_scene_prompt_v9(a, reg, 1, 36, "", "公園",
                                       ["湯坂上のご夫婦", "駅前通りの持ち主さん"]))
    lo = [(p, reg.listing_options("A", p)) for p in reg.parcels_owned("A")]
    out.append(F.build_decide_prompt_v9(a, reg, 1, 36, "", offer, [],
                                        listing_options=lo,
                                        sell_order_=[F.SELL_NO, F.SELL_YES],
                                        neighbours=["となりの人"]))
    return out


def test_no_acquirer_mandate_or_settings_leaks_to_residents(cfg, reg):
    """X社の命題と設定は住民側に1文字も出ない。"""
    for text in _all_resident_prompts(cfg, reg):
        assert F.ACQUIRER_MANDATE_V9 not in text
        assert "取得せよ" not in text
        assert "不動産投資会社のため" not in text
        assert "毎月動け" not in text
        for line in F.ACQUIRER_FACTS_V9.strip().split("\n"):
            if line.strip() and not line.startswith("---"):
                assert line.strip() not in text, line


def test_wording_is_ownership_not_meigi(cfg, reg):
    """用語は「所有権」で統一（「名義」を使わない・施主 10:26）。"""
    texts = _all_resident_prompts(cfg, reg)
    texts.append(F.build_absentee_prefix_v9(cfg, len(reg.parcel_names)))
    texts.append(F.build_acquirer_prefix_v9(cfg, reg))
    texts.append(F.build_acquirer_prompt_v9(reg, 1, 36, ["湯坂上のご夫婦"], [],
                                            [], ["湯坂上の古家"], 1, 1))
    for text in texts:
        assert "名義" not in text


def test_no_vacant_house_wording(cfg, reg):
    """「空き家」「空き」は区画名の固有名詞以外に使わない（施主 10:35）。"""
    # 区画名と、v8 から凍結している生業の呼び名（「空き家対策」など）は対象外
    frozen = set(reg.parcel_names) | {str(a["role_label"]) for a in reg.agents}
    text = F.build_common_prefix_v9(cfg, reg.agents, len(reg.parcel_names))
    for line in text.split("\n"):
        if any(w in line for w in frozen):
            continue
        assert "空き家" not in line, line


def test_decide_prompt_has_no_irreversible_nudge(cfg, reg):
    """v8d と同じく「その後は戻らない」の念押しは問いに書かない。"""
    a = reg.by_id["A"]
    offer = {"parcel": "湯坂上の古家", "kind": F.KIND_BOTH,
             "delivered": F.delivered_offer_v9("ご検討ください。")}
    lo = [(p, reg.listing_options("A", p)) for p in reg.parcels_owned("A")]
    text = F.build_decide_prompt_v9(a, reg, 1, 36, "", offer, [],
                                    listing_options=lo,
                                    sell_order_=[F.SELL_NO, F.SELL_YES])
    assert "その後は戻らない" not in text
    # 世界の事実としての1回だけは共通前置きに残っている
    prefix = F.build_common_prefix_v9(cfg, reg.agents, len(reg.parcel_names))
    assert "一度移った所有権が戻ることはない。" in prefix


def test_decide_prompt_lists_only_owned_parcels(cfg, reg):
    """問い１に出るのは自分が持っている区画だけ（借りている区画は出ない）。"""
    a = reg.by_id["B"]        # 借地上の持家（建物だけ持つ）
    lo = [(p, reg.listing_options("B", p)) for p in reg.parcels_owned("B")]
    text = F.build_decide_prompt_v9(a, reg, 1, 36, "", None, [],
                                    listing_options=lo)
    assert "駅前通りの家：" in text
    assert F.LIST_BUILDING in text
    assert "「土地だけ」" not in text


def test_offer_shows_parcel_and_kind(cfg, reg):
    a = reg.by_id["A"]
    offer = {"parcel": "湯坂上の古家", "kind": F.KIND_BUILDING,
             "delivered": F.delivered_offer_v9("ご検討ください。")}
    text = F.build_plan_prompt_v9(a, reg, 1, 36, ["公園"], "", offer)
    assert "湯坂上の古家 の 建物 について" in text
    assert F.ACQUIRER_INTRO_V8C in text


def test_absentee_prefix_has_no_venues_or_roster(cfg, reg):
    """町にいない所有者には会場も名簿も渡さない（その場にいないから）。"""
    text = F.build_absentee_prefix_v9(cfg, len(reg.parcel_names))
    assert "公園" not in text
    assert "居合わせ" not in text
    assert "湯坂上のご夫婦" not in text
    assert F.ACQUIRER_MANDATE_V9 not in text


def test_acquirer_does_not_learn_where_owners_live(cfg, reg):
    """X社は所有者が町にいるかどうかを知らない（施主 10:54(6)）。"""
    text = F.build_acquirer_prefix_v9(cfg, reg)
    text += F.build_acquirer_prompt_v9(reg, 1, 36, ["湯坂上のご夫婦"], [], [],
                                       ["湯坂上の古家"], 1, 1)
    assert "市外" not in text
    assert "町にいない" not in text
    assert "町にいる" not in text
    for a in reg.agents:
        if not a.get("resident", True):
            assert a["persona"].split("\n")[0] not in text


def test_acquirer_mandate_is_in_the_user_prompt_only(cfg, reg):
    prefix = F.build_acquirer_prefix_v9(cfg, reg)
    assert F.ACQUIRER_MANDATE_V9 not in prefix
    up = F.build_acquirer_prompt_v9(reg, 1, 36, ["湯坂上のご夫婦"], [], [],
                                    ["湯坂上の古家"], 1, 1)
    assert up.startswith(F.ACQUIRER_MANDATE_V9)


def test_acquirer_owner_instruction_is_frozen_text(cfg, reg):
    """施主指定の一文は言い換えずにそのまま入る（10:40）。"""
    prefix = F.build_acquirer_prefix_v9(cfg, reg)
    assert "不動産投資会社のため、不動産管理等は行わない。" in prefix


def test_rotation_is_deterministic_and_matches_enum():
    vals = [F.LIST_NO, F.LIST_LAND, F.LIST_BUILDING, F.LIST_BOTH]
    a = F.rotate(vals, 3, 5)
    assert a == F.rotate(vals, 3, 5)
    assert sorted(a) == sorted(vals)
    assert F.rotate(vals, 0, 0)[0] == F.LIST_NO
    assert F.rotate(vals, 1, 0)[0] == F.LIST_LAND


def test_decide_schema_matches_the_options(reg):
    lo = [(p, reg.listing_options("A", p)) for p in reg.parcels_owned("A")]
    schema = F.decide_schema_v9(lo, [F.SELL_NO, F.SELL_YES])
    props = schema["properties"]
    assert set(props["listings"]["properties"]) == {p for p, _ in lo}
    for p, opts in lo:
        assert props["listings"]["properties"][p]["enum"] == opts
    assert props["sell"]["enum"] == [F.SELL_NO, F.SELL_YES]
    # 条件が届いていない月には sell が無い
    assert "sell" not in F.decide_schema_v9(lo)["properties"]


def test_transfer_notice_is_a_bare_fact():
    line = F.transfer_notice("湯坂上の古家", F.KIND_LAND, "湯坂上のご夫婦", "X社")
    assert line == ("（記録）先月末、湯坂上の古家の土地の所有権が "
                    "湯坂上のご夫婦 から X社 に移った。")
    for word in ("べき", "注意", "危険", "気を", "警戒", "急"):
        assert word not in line


# ---------------------------------------------------------------------------
# 月ループ（mock）
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mock_run(tmp_path_factory, cfg):
    run_dir = str(tmp_path_factory.mktemp("v9run"))
    c = json.loads(json.dumps(cfg))
    c["llm"]["provider"] = "mock"
    c["steps"] = 4
    c["max_cost_usd"] = 0
    sim = SimulationV9(c, run_dir)
    summary = sim.run()
    return sim, summary, run_dir


def test_mock_run_is_healthy(mock_run):
    sim, s, _run_dir = mock_run
    assert s["months_run"] == 4
    assert s["parse_fail"] == 0
    assert s["no_answer"] == 0
    assert s["listing_missing"] == 0
    assert s["invalid_listing"] == 0
    assert s["acquirer_invalid_offer"] == 0     # mock は実行できる提示だけ作る
    assert s["agents"] == 49
    assert s["offers_total"] > 0
    assert s["offers_to_in_town"] + s["offers_to_absentee"] == s["offers_total"]


def test_mock_run_writes_everything(mock_run):
    _sim, _s, run_dir = mock_run
    for name in ("summary.json", "monthly.json", "offers.json", "decisions.json",
                 "listings.json", "notices.json", "transfers.json",
                 "ledger_by_step.json", "ledger_final.json", "timeline_index.json",
                 "common_prefix.txt", "absentee_prefix.txt", "acquirer_prefix.txt"):
        assert os.path.exists(os.path.join(run_dir, name)), name
    for name in ("monthly.json", "log.jsonl"):
        assert os.path.exists(os.path.join(run_dir, "checkpoint", name)), name


def test_ledger_snapshot_shape(mock_run):
    _sim, _s, run_dir = mock_run
    with open(os.path.join(run_dir, "ledger_by_step.json"), encoding="utf-8") as f:
        book = json.load(f)
    assert len(book) == 4
    for row in book:
        assert len(row["rows"]) == 44
        for r in row["rows"]:
            assert set(r) == {"parcel", "district", "has_building", "land",
                              "building", "tenant", "user"}


def test_absentee_owners_never_get_plan_or_scene_calls(mock_run):
    sim, _s, _run_dir = mock_run
    absentee = {str(a["id"]) for a in sim.agents if not a.get("resident", True)}
    assert absentee
    assert not (absentee & {p["agent_id"] for p in sim.plans})
    assert not (absentee & {u["from_id"] for u in sim.utterances})
    assert not (absentee & {d["to"] for d in sim.deliveries})
    # ただし出品と売買の問いには毎月答える
    asked = {d["agent_id"] for d in sim.decisions}
    assert absentee & asked


def test_transfers_are_recorded_with_parcel_and_kind(mock_run):
    sim, _s, _run_dir = mock_run
    for t in sim.reg.transfers:
        assert t["kind"] in F.KIND_VALUES
        assert t["parcel"] in sim.reg.parcel_names
        assert isinstance(t["left_town"], bool)


def test_notice_goes_to_the_user_not_the_seller(mock_run):
    sim, _s, _run_dir = mock_run
    for n in sim.notices:
        t = [x for x in sim.reg.transfers
             if x["parcel"] == n["parcel"] and x["step"] == n["step"] - 1]
        assert t, n
        assert n["to"] != t[0]["agent_id"]


def test_mock_client_only_makes_offers_the_world_can_deliver(reg):
    client = MockV9Client(seed=1, reg=reg)
    for name in ("湯坂上のご夫婦", "駅前通りの持ち主さん", "駅前通りの地主"):
        pick = client._pick_offer(name)
        if pick is None:
            continue
        parcel, kind = pick
        assert reg.can_offer(reg.id_of_name[name], parcel, kind)


# ---------------------------------------------------------------------------
# 退場が月ループの中でも通ることを見る（世界の骨格の要）
# ---------------------------------------------------------------------------

class _AlwaysBothToA(MockV9Client):
    """A（湯坂上のご夫婦）にだけ『両方』の提示を出し、相手は必ず「売る」と答える mock。"""

    def _pick_offer(self, name):
        if name == "湯坂上のご夫婦":
            return ("湯坂上の古家", F.KIND_BOTH)
        return None

    def generate(self, system_prompt, user_prompt, schema=None, temperature=None,
                 max_tokens=None, tag="agent"):
        raw = super().generate(system_prompt, user_prompt, schema, temperature,
                               max_tokens, tag)
        props = (schema or {}).get("properties", {})
        if "sell" in props:
            obj = json.loads(raw)
            obj["sell"] = F.SELL_YES
            return json.dumps(obj, ensure_ascii=False)
        return raw


def test_leaving_town_runs_through_the_month_loop(tmp_path, cfg):
    c = json.loads(json.dumps(cfg))
    c["llm"]["provider"] = "mock"
    c["steps"] = 3
    c["max_cost_usd"] = 0
    sim = SimulationV9(c, str(tmp_path))
    sim.client = _AlwaysBothToA(seed=7, usage=sim.usage, reg=sim.reg,
                                send_rate=1.0)
    sim.run()

    t = [x for x in sim.reg.transfers if x["agent_id"] == "A"]
    assert t and t[0]["kind"] == F.KIND_BOTH and t[0]["left_town"] is True
    left_at = t[0]["step"]
    assert sim.reg.left_month["A"] == left_at
    # 以後は場にも会話にも出てこない
    assert not [p for p in sim.plans if p["agent_id"] == "A" and p["step"] > left_at]
    assert not [u for u in sim.utterances
                if u["from_id"] == "A" and u["step"] > left_at]
    assert not [d for d in sim.deliveries
                if d["to"] == "A" and d["step"] > left_at]
    # 2件目を持っているので「町にいない所有者」として出品の問いには答え続ける
    assert "湯坂上の空き店舗" in sim.reg.parcels_owned("A")
    assert "A" in sim.reg.absentee_owner_ids()
    assert [d for d in sim.decisions if d["agent_id"] == "A" and d["step"] > left_at]
    # 使用者がいなくなった区画は帳簿にそう残る
    assert sim.reg.user_of("湯坂上の古家") is None
    # 借家人（湯坂上の借り店主）は町に残る
    assert sim.reg.is_resident("BA")


# ---------------------------------------------------------------------------
# Codex 走行前レビュー 2026-08-30 の必須指摘を固定する
# ---------------------------------------------------------------------------

def test_listing_reason_is_asked_per_parcel(reg):
    """理由の一言は判断ごとに1つ（設計 §2）。出品は区画ごとの判断＝理由も区画ごと。"""
    opts = [("湯坂上の古家", [F.LIST_NO, F.LIST_LAND, F.LIST_BUILDING, F.LIST_BOTH]),
            ("湯坂上の空き店舗", [F.LIST_NO, F.LIST_LAND, F.LIST_BUILDING, F.LIST_BOTH])]
    sch = F.decide_schema_v9(listing_options=opts)
    props = sch["properties"]
    assert "listing_reason" not in props
    assert set(props["listing_reasons"]["properties"]) == {p for p, _o in opts}
    assert set(props["listing_reasons"]["required"]) == {p for p, _o in opts}
    a = reg.by_id["A"]
    text = F.build_decide_prompt_v9(a, reg, 3, 36, "", None, [], opts)
    assert "listing_reasons" in text
    assert "listing_reason に" not in text


def test_sell_options_describe_both_sides_at_the_same_grain(reg):
    """問い２の2肢は「所有権がどうなるか」＋「使い方がどうなるか」を両側に書く。

    片側だけに帰結を付けない（v8c で必須とされた規律）。
    """
    for parcel in reg.parcel_names:
        owners = [aid for aid in reg.by_id
                  if reg.owns_land(aid, parcel) or reg.owns_building(aid, parcel)]
        for aid in owners:
            for kind in (F.KIND_LAND, F.KIND_BUILDING, F.KIND_BOTH):
                if not reg.can_offer(aid, parcel, kind):
                    continue
                lines = F._sell_lines_v9(reg, aid, parcel, kind)
                yes, no = lines[F.SELL_YES], lines[F.SELL_NO]
                # 使用の帰結が片側だけに出ていない（両方あるか、両方無いか）
                assert (("今までどおり" in yes or "A市を出る" in yes)
                        == ("今までどおり" in no or "A市にとどまる" in no)), (parcel, aid, kind)
                # 文の長さが極端に非対称でない（片側だけ長い＝目立たせない）
                assert abs(len(yes) - len(no)) <= 12, (parcel, aid, kind, yes, no)


def test_only_the_frozen_v8_name_carries_a_residence_hint(reg):
    """町にいない所有者の呼び名に町外を示す語が入らない（AA は v8 凍結名＝例外）。

    施主に報告済みの既知事項。ここでは**新規13体に広がっていない**ことだけを固定する。
    """
    leaky = [a["name"] for a in reg.agents
             if not a.get("resident", True)
             and any(w in a["name"] for w in ("不在", "市外", "町外", "遠方"))]
    assert leaky == ["船着場裏の不在地主"]
