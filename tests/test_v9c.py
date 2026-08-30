"""v9c「借りている人にも出口がある町」の試験。

設計の正は `docs/world_design_v9c.md`（施主確定 2026-08-30 13:23／13:31）。
ここで固定するのは v9b との差分だけ:

  - 名簿＝町にいない所有者14人のペルソナだけが厚くなり、町の人35体は1文字も変わらない
  - 借りて使っている人に「出る／出ない」が届き、出た人は以後どのコールにも現れない
  - 出た区画は使う人がいなくなり、家主に翌月初、事実1行が届く
  - 借りて使っている人と家主の一言が両方向に配られ、**X社には見えない**
  - 毎月のプロンプトの冒頭に経過した時間が置かれる（年齢が無い主体には年齢を書かない）
  - A（実行できない約束の門番）と B（X社の設定2行）が v9b のまま生きている
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import field_v9 as F9  # noqa: E402
from src import field_v9b as F9B  # noqa: E402
from src import field_v9c as F  # noqa: E402
from src.sim_v9b import SimulationV9B  # noqa: E402
from src.sim_v9c import MockV9CClient, RegistryV9C, SimulationV9C  # noqa: E402

PERSONAS_V9 = os.path.join(ROOT, "configs", "personas_v9.yaml")
PERSONAS_V9C = os.path.join(ROOT, "configs", "personas_v9c.yaml")
CONFIG_V9C = os.path.join(ROOT, "configs", "config_field_v9c.yaml")
DRAFT = os.path.join(ROOT, "docs", "personas_absentee_v9c_draft.md")


def load_cfg(steps=3, provider="mock"):
    with open(CONFIG_V9C, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["steps"] = steps
    cfg["llm"] = dict(cfg["llm"], provider=provider)
    return cfg


@pytest.fixture(scope="module")
def reg():
    agents, parcels = F9.load_personas_v9(PERSONAS_V9C)
    return RegistryV9C(agents, parcels)


# ---------------------------------------------------------------------------
# 1. 名簿
# ---------------------------------------------------------------------------

def test_only_absentee_personas_changed():
    a = yaml.safe_load(open(PERSONAS_V9, encoding="utf-8"))
    b = yaml.safe_load(open(PERSONAS_V9C, encoding="utf-8"))
    assert a["parcels"] == b["parcels"]
    ra = {x["id"]: x for x in a["agents"]}
    rb = {x["id"]: x for x in b["agents"]}
    assert list(ra) == list(rb)
    changed = [k for k in ra if ra[k] != rb[k]]
    assert len(changed) == 14
    assert all(not ra[k].get("resident", True) for k in changed)
    for k in ra:
        if ra[k].get("resident", True):
            assert ra[k] == rb[k], k


def test_absentee_personas_keep_the_original_two_lines():
    a = {x["id"]: x for x in yaml.safe_load(
        open(PERSONAS_V9, encoding="utf-8"))["agents"]}
    b = {x["id"]: x for x in yaml.safe_load(
        open(PERSONAS_V9C, encoding="utf-8"))["agents"]}
    for k, v in a.items():
        if not v.get("resident", True):
            assert b[k]["persona"].startswith(v["persona"].strip())
            assert len(b[k]["persona"]) > len(v["persona"])


def test_added_personas_have_no_forbidden_words():
    a = {x["id"]: x for x in yaml.safe_load(
        open(PERSONAS_V9, encoding="utf-8"))["agents"]}
    b = {x["id"]: x for x in yaml.safe_load(
        open(PERSONAS_V9C, encoding="utf-8"))["agents"]}
    # 「取得」は凍結済みの既存2行にもある語（CC「取得して9年」・CM「取得して8年」）
    # なので禁止語に入れない。禁止するのは売買・金銭・X社・希望表現。
    banned = ["売却", "売る", "買収", "売り", "X社", "投資", "円", "価格", "金額",
              "対価", "したい", "すべき", "べきだ"]
    for k, v in a.items():
        if v.get("resident", True):
            continue
        added = b[k]["persona"][len(v["persona"].strip()):]
        for w in banned:
            assert w not in added, (k, w)


# ---------------------------------------------------------------------------
# 2. 経過した時間
# ---------------------------------------------------------------------------

def test_time_header_counts_from_zero(reg):
    a = reg.by_id["A"]          # 74歳の夫婦世帯
    assert F.time_header(1, a) == "（第1月・開始から0か月・あなたは74歳）"
    assert F.time_header(13, a) == "（第13月・開始から12か月・あなたは75歳）"
    assert F.time_header(36, a) == "（第36月・開始から35か月・あなたは76歳）"


def test_no_age_is_invented(reg):
    company = reg.by_id["U"]    # 駐車場の会社（年齢の記載が無い）
    assert F.start_age(company) is None
    assert F.time_header(5, company) == "（第5月・開始から4か月）"
    assert F.time_header(5, None) == "（第5月・開始から4か月）"


def test_every_call_gets_the_header(tmp_path):
    sim = SimulationV9C(load_cfg(steps=1), str(tmp_path))
    seen = {}
    orig = sim.client.generate

    def spy(system_prompt, user_prompt, **kw):
        seen.setdefault(kw.get("tag", ""), []).append(user_prompt)
        return orig(system_prompt, user_prompt, **kw)

    sim.client.generate = spy
    sim.run()
    assert seen
    for tag, prompts in seen.items():
        for up in prompts:
            head = up.split("\n")[1] if "acquirer" in tag else up.split("\n")[0]
            assert head.startswith("（第1月・開始から0か月"), (tag, head)


# ---------------------------------------------------------------------------
# 3. 借りて使っている人
# ---------------------------------------------------------------------------

def test_tenants_are_the_ten_who_own_nothing(reg):
    ids = F.tenant_ids(reg)
    assert len(ids) == 10
    for aid in ids:
        assert reg.parcels_owned(aid) == []
        assert reg.is_resident(aid)
        assert F.tenant_parcels(reg, aid)


def test_landlord_is_the_building_owner_then_the_land_owner(reg):
    for p in reg.parcel_names:
        if reg.tenant_of.get(p) is None:
            continue
        who = F.landlord_of(reg, p)
        if reg.has_building[p]:
            assert who == reg.building_of[p]
        else:
            assert who == reg.land_of[p]


def test_leave_prompt_has_no_nudge(reg):
    a = reg.by_id[F.tenant_ids(reg)[0]]
    ps = F.tenant_parcels(reg, str(a["id"]))
    up = F.build_tenant_prompt_v9c(a, reg, 7, 36, "", ps,
                                   F.leave_order(0, 7))
    assert "leave に「" in up
    assert F.LEAVE_YES in up and F.LEAVE_NO in up
    for bad in ("べきである", "した方がよい", "おすすめ", "検討せよ", "そろそろ",
                "潮時", "危ない", "急いだ"):
        assert bad not in up
    assert "この理由は誰にも伝わらない" in up


def test_leave_order_rotates(reg):
    seen = {tuple(F.leave_order(i, s)) for i in range(4) for s in range(4)}
    assert seen == {(F.LEAVE_YES, F.LEAVE_NO), (F.LEAVE_NO, F.LEAVE_YES)}


@pytest.fixture(scope="module")
def left_run(tmp_path_factory):
    """借りて使っている人が必ず出る mock を1本走らせる（配線の確認用）。"""
    class AlwaysLeave(MockV9CClient):
        def generate(self, system_prompt, user_prompt, schema=None,
                     temperature=None, max_tokens=None, tag="agent"):
            props = (schema or {}).get("properties", {})
            if "leave" in props:
                self.usage.add(tag, input_tokens=10, output_tokens=10)
                return json.dumps({"thought": "", "leave": F.LEAVE_YES,
                                   "leave_reason": "家の事情で",
                                   "to_landlord": "お世話になりました。"},
                                  ensure_ascii=False)
            return super().generate(system_prompt, user_prompt, schema=schema,
                                    temperature=temperature,
                                    max_tokens=max_tokens, tag=tag)

    out = tmp_path_factory.mktemp("v9c_left")
    sim = SimulationV9C(load_cfg(steps=3), str(out))
    sim.client = AlwaysLeave(seed=sim.seed, usage=sim.usage, reg=sim.reg)
    summary = sim.run()
    return sim, summary, str(out)


def test_tenants_leave_and_never_get_called_again(left_run):
    sim, summary, _out = left_run
    assert summary["tenant_left_total"] >= 10
    first = {r["agent_id"]: r["step"] for r in sim.left_tenants}
    for aid, step in first.items():
        later = [r for r in sim.tenant_decisions
                 if r["agent_id"] == aid and r["step"] > step]
        assert later == [], aid
        assert aid not in sim.reg.in_town_ids()
        assert all(p["from"] != sim.reg.name_of[aid]
                   for p in sim.utterances if p["step"] > step)


def test_parcel_has_no_user_after_the_tenant_left(left_run):
    sim, _s, _o = left_run
    for row in sim.left_tenants:
        for p in row["parcels"]:
            assert p in sim.reg.vacated
            assert sim.reg.user_of(p) is None


def test_landlord_gets_the_fact_line_next_month(left_run):
    sim, summary, out = left_run
    assert summary["vacancy_notices_total"] > 0
    with open(os.path.join(out, "vacancy_notices.json"), encoding="utf-8") as f:
        rows = json.load(f)
    for r in rows:
        assert r["text"].startswith("（記録）先月末、")
        assert "この区画を使う人はいない。" in r["text"]
        left = [x for x in sim.left_tenants if x["name"] == r["tenant"]][0]
        assert r["step"] == left["step"] + 1          # 翌月初に届く
        for bad in ("べき", "おすすめ", "残念", "困った"):
            assert bad not in r["text"]


def test_roster_marks_the_people_who_left(left_run):
    sim, _s, _o = left_run
    assert F.LEFT_MARK in sim.common_prefix
    for row in sim.left_tenants:
        assert f"{row['name']}" in sim.common_prefix


def test_roster_has_no_mark_at_the_start(reg):
    cfg = load_cfg()
    pre = F.build_common_prefix_v9c(cfg, reg.agents, 44)
    body = pre.split("--- 町の場所 ---")[0]
    assert body.count(F.LEFT_MARK) == 1     # 凡例の1行だけ


# ---------------------------------------------------------------------------
# 4. 一言（借りて使っている人 ⇔ 家主）
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mock_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("v9c")
    sim = SimulationV9C(load_cfg(steps=4), str(out))
    summary = sim.run()
    return sim, summary, str(out)


def test_messages_go_both_ways(mock_run):
    _sim, summary, _o = mock_run
    assert summary["messages_tenant_to_landlord"] > 0
    assert summary["messages_landlord_to_tenant"] > 0


def test_messages_are_saved_verbatim(mock_run):
    sim, summary, out = mock_run
    with open(os.path.join(out, "messages.json"), encoding="utf-8") as f:
        rows = json.load(f)
    assert len(rows) == summary["messages_total"] == len(sim.messages)
    for r in rows:
        assert r["text"] and len(r["text"]) <= F.MAX_LINE_CHARS
        assert r["direction"] in ("借りて使っている人→家主", "家主→借りて使っている人")


def test_the_acquirer_never_sees_the_lines(mock_run):
    sim, _s, _o = mock_run
    texts = {m["text"] for m in sim.messages}
    blob = json.dumps(sim.acquirer_raw, ensure_ascii=False) + sim.acquirer_prefix
    for t in texts:
        assert t not in blob
    # X社の履歴・登記簿にも入らない
    for o in sim.offers:
        for t in texts:
            assert t not in json.dumps(o, ensure_ascii=False)


def test_landlord_reply_reaches_the_tenant(mock_run):
    sim, _s, _o = mock_run
    replies = [m for m in sim.messages
               if m["direction"] == "家主→借りて使っている人"]
    assert replies
    got = [r for r in sim.tenant_decisions if r["line_in"]]
    assert got, "家主の返事が借りて使っている人に届いていない"


# ---------------------------------------------------------------------------
# 5. v9b の骨格が生きていること
# ---------------------------------------------------------------------------

def test_promise_gate_and_owner_facts_still_apply(mock_run):
    sim, summary, _o = mock_run
    assert "undelivered_total" in summary
    for o in sim.offers:
        assert F9B.undeliverable_promise(o["text"]) is False
    assert "土地だけ、建物だけの取得でもよい。" in sim.acquirer_prefix
    assert "支援・改修・管理・金銭の提供を実行する仕組みはない。" in sim.acquirer_prefix


def test_scenario_version_and_config(mock_run):
    _sim, summary, _o = mock_run
    assert summary["scenario_version"] == "field_v9c"
    cfg = load_cfg()
    assert cfg["personas_file"] == "configs/personas_v9c.yaml"
    assert cfg["seed"] == 85 and cfg["llm"]["temperature"] == 0.75
    assert cfg["llm"]["model"] == "gemini-2.5-flash-lite"


def test_v9b_simulation_rejects_v9c_config(tmp_path):
    with pytest.raises(ValueError):
        SimulationV9B(load_cfg(), str(tmp_path))


def test_health_counters_are_clean(mock_run):
    _sim, summary, _o = mock_run
    for k in ("parse_fail", "invalid_listing", "invalid_sell", "invalid_venue",
              "invalid_leave", "tenant_no_answer", "listing_missing"):
        assert summary[k] == 0, k


def test_tenant_answers_are_not_filled_in_by_default(tmp_path):
    """答えが返らなかったときに「出ない」で埋めない（v9 以来の規律）。"""
    class Silent(MockV9CClient):
        def generate(self, system_prompt, user_prompt, schema=None,
                     temperature=None, max_tokens=None, tag="agent"):
            props = (schema or {}).get("properties", {})
            if "leave" in props:
                self.usage.add(tag, input_tokens=10, output_tokens=1)
                return "こわれた出力"
            return super().generate(system_prompt, user_prompt, schema=schema,
                                    temperature=temperature,
                                    max_tokens=max_tokens, tag=tag)

    sim = SimulationV9C(load_cfg(steps=1), str(tmp_path))
    sim.client = Silent(seed=sim.seed, usage=sim.usage, reg=sim.reg)
    summary = sim.run()
    assert summary["tenant_no_answer"] == 10
    assert summary["tenant_leave_counts"]["出ない"] == 0
    assert summary["tenant_left_total"] == 0
    assert all(r["leave"] == "no_answer" for r in sim.tenant_decisions)


# ---------------------------------------------------------------------------
# 6. 走行前レビュー（2026-08-30）の必須指摘に対する回帰試験
# ---------------------------------------------------------------------------

def test_every_message_has_a_real_recipient(mock_run):
    """一言は書いた時点で宛先を確定する（宛先「—」を作らない）。"""
    sim, _s, out = mock_run
    with open(os.path.join(out, "messages.json"), encoding="utf-8") as f:
        rows = json.load(f)
    names = set(sim.reg.name_of.values())
    for r in rows:
        assert r["to"] in names, r
        assert r["from"] in names, r
        assert r["to"] != "—"


def test_the_acquirer_is_never_a_recipient_of_a_line(mock_run):
    sim, _s, _o = mock_run
    for m in sim.messages:
        assert m["to"] != "X社" and m["from"] != "X社"


def test_every_vacated_parcel_produces_a_fact_line(left_run):
    """使う人がいなくなった区画は、必ず所有者に事実が届く（X社が持ち主なら X社 に）。"""
    sim, summary, _o = left_run
    vacated = {p for row in sim.left_tenants for p in row["parcels"]}
    # 最終月の退場は翌月が無いので通知が出ない＝それ以外は全部届く
    last = sim.n_steps
    expect = {p for row in sim.left_tenants if row["step"] < last
              for p in row["parcels"]}
    got = {r["parcel"] for r in sim.vacancy_notices}
    assert expect <= got, expect - got
    assert got <= vacated


def test_checkpoint_is_written_once_a_month(mock_run):
    sim, summary, out = mock_run
    with open(os.path.join(out, "checkpoint", "log.jsonl"), encoding="utf-8") as f:
        lines = [x for x in f.read().split("\n") if x.strip()]
    assert len(lines) == summary["months_run"]
    assert summary["checkpoints_written"] == summary["months_run"]


def test_no_line_field_when_the_landlord_is_the_acquirer(tmp_path):
    """家主がX社になった区画しか使っていない人には、一言の欄を出さない。"""
    from src.field_v9c import build_tenant_prompt_v9c, leave_order, tenant_ids
    sim = SimulationV9C(load_cfg(steps=1), str(tmp_path))
    reg = sim.reg
    aid = tenant_ids(reg)[0]
    ps = F.tenant_parcels(reg, aid)
    for p in ps:                       # その区画をX社のものにする
        reg.building_of[p] = "X社"
        reg.land_of[p] = "X社"
    assert F.landlord_of(reg, ps[0]) is None
    up = build_tenant_prompt_v9c(reg.by_id[aid], reg, 2, 36, "", ps,
                                 leave_order(0, 2), with_line=False)
    assert "to_landlord" not in up
    assert "leave に「" in up
