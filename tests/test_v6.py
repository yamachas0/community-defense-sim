#!/usr/bin/env python
"""v6（経路ありの世界＝中立な行動の選択肢3段・採点用LLM全廃）の走行前テスト。

    python tests/test_v6.py

外部依存なし・実APIを叩かない（LLMは mock）。ここで固定するのは
「走行前に決めたことが、実際にその通りに凍結されているか」だけである。
設計の正は docs/world_design_v6_two_worlds.md。
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import tempfile

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from src.field_v5 import (build_scene_prompt_v5, inbox_rows_v5)          # noqa: E402
from src.field_v5d import scene_schema_v5d                               # noqa: E402
from src.field_v6 import (MEASURE_NONE, MEASURE_VALUES, PAPER_LABELS,    # noqa: E402
                          PUBLIC_ACT_ASSEMBLY, PUBLIC_ACT_CIRCULAR,
                          PUBLIC_ACT_NONE, PUBLIC_ACT_PETITION,
                          PUBLIC_ACT_VALUES, SELL_INTENT_CLEAR,
                          SELL_INTENT_KEEP, SELL_INTENT_REFUSE,
                          SELL_INTENT_VALUES, action_rows_v6,
                          blocked_acquisitions_v6, is_refused, scene_schema_v6,
                          script_without, set_refusal)
from src.llm_client_factory import MockClient                            # noqa: E402
from src.simulation import Simulation                                    # noqa: E402
from src.world import Ledger, Parcel                                     # noqa: E402
from src.agents import Agent                                             # noqa: E402

PASS = FAIL = 0


def check(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}" + (f"  … {detail}" if detail else ""))


def load(path):
    with io.open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return yaml.safe_load(f)


def jsonl(path):
    if not os.path.exists(path):
        return []
    with io.open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


CFG = load("configs/config_field_v6.yaml")
CFG_V5E2 = load("configs/config_field_v5e2.yaml")
PERSONAS = load(CFG["personas_file"])
DESIGN = io.open(os.path.join(ROOT, "docs", "world_design_v6_two_worlds.md"),
                 encoding="utf-8").read()
STEP_RE = re.compile(r"第(\d+)月")


# ===========================================================================
print("\n[1] 設定：v5e2 との差は4点だけ（世界・台本・月数は同一）")
# ===========================================================================

diff = {k for k in set(CFG) | set(CFG_V5E2)
        if CFG.get(k) != CFG_V5E2.get(k)}
check("config のトップ階層の差は run_name / scenario_version / kpi だけ",
      diff == {"run_name", "scenario_version", "kpi"}, str(sorted(diff)))
check("scenario_version は field_v6", CFG["scenario_version"] == "field_v6")
check("月数は 36 のまま", CFG["steps"] == CFG_V5E2["steps"] == 36)
check("台本は v5e2 と同一（目玉=第15月）",
      CFG["events_file"] == CFG_V5E2["events_file"])
check("seed・ペルソナ・区画・呼び名は同一",
      (CFG["seed"], CFG["personas_file"], CFG["parcels_file"], CFG["names_file"])
      == (CFG_V5E2["seed"], CFG_V5E2["personas_file"], CFG_V5E2["parcels_file"],
          CFG_V5E2["names_file"]))
check("モデル・温度・並列数は同一", CFG["llm"] == CFG_V5E2["llm"])
check("会場・世界の記述は同一",
      CFG["world"] == CFG_V5E2["world"] and CFG["social"] == CFG_V5E2["social"])
check("**採点用LLMは全廃**＝classify_utterances が false",
      CFG["kpi"]["classify_utterances"] is False)
check("占領分類器も off", CFG["kpi"]["classify_occupation"] is False)
kpi_diff = {k for k in set(CFG["kpi"]) | set(CFG_V5E2["kpi"])
            if CFG["kpi"].get(k) != CFG_V5E2["kpi"].get(k)}
check("kpi の差は classify_utterances だけ", kpi_diff == {"classify_utterances"},
      str(sorted(kpi_diff)))


# ===========================================================================
print("\n[2] 選択肢は中立（既定が先頭・促す語が無い）")
# ===========================================================================

check("sell_intent の先頭は「変えない」", SELL_INTENT_VALUES[0] == SELL_INTENT_KEEP)
check("public_act の先頭は「なし」", PUBLIC_ACT_VALUES[0] == PUBLIC_ACT_NONE)
check("measure の先頭は「なし」", MEASURE_VALUES[0] == MEASURE_NONE)
check("sell_intent は3択のみ", SELL_INTENT_VALUES ==
      [SELL_INTENT_KEEP, SELL_INTENT_REFUSE, SELL_INTENT_CLEAR])
check("public_act は4択のみ", PUBLIC_ACT_VALUES ==
      [PUBLIC_ACT_NONE, PUBLIC_ACT_CIRCULAR, PUBLIC_ACT_ASSEMBLY,
       PUBLIC_ACT_PETITION])
check("measure は4択のみ", len(MEASURE_VALUES) == 4)

# 促し・当為・評価の語を1つも含まないこと（誘導の門番）。
BANNED = ["べき", "しましょう", "した方がいい", "したほうがいい", "推奨", "望ましい",
          "重要", "急", "守る", "守り", "自衛", "対抗", "抵抗", "阻止", "団結",
          "危険", "脅威", "被害", "警戒", "注意して", "考えてみ", "検討して",
          "できます", "できる立場", "権利", "手段がある", "有効", "効果"]
for owns in (True, False):
    for muni in (True, False):
        rows = "\n".join(action_rows_v6(owns, muni))
        hit = [w for w in BANNED if w in rows]
        check(f"行動欄の文言に促し・当為の語が無い(owns={owns},muni={muni})",
              not hit, str(hit))
check("行動欄は選択肢の提示と保存の説明だけ（2行×種類）",
      len(action_rows_v6(True, False)) == 4 and len(action_rows_v6(False, True)) == 2,
      str((len(action_rows_v6(True, False)), len(action_rows_v6(False, True)))))
# Codexレビュー（走行前）で落とした語＝行動を意識させる問いかけ・到達力の宣伝。
for word in ("何かするか", "取る措置", "全員が見られる"):
    check(f"行動欄に『{word}』が無い（Codexレビュー反映）",
          all(word not in "|".join(action_rows_v6(o, m))
              for o in (True, False) for m in (True, False)))
check("書かなければ既定として扱う、と明記してある",
      all("書かなければ" in "|".join(action_rows_v6(o, m))
          for o in (True, False) for m in (True, False)))
check("設計文書に『狙った動きを仕組まない』が明記されている",
      "狙った動きを仕組まない" in DESIGN)


# ===========================================================================
print("\n[3] スキーマ：最後のターンでだけ行動欄が出る・既存の項目は不変")
# ===========================================================================

base = scene_schema_v5d("甲", ["甲", "乙"], ["甲", "乙", "丙"], True, True)
same = scene_schema_v6("甲", ["甲", "乙"], ["甲", "乙", "丙"], True, True,
                       ask_actions=False)
check("ask_actions=False のスキーマは v5d と完全一致", same == base,
      str(sorted(set(same["properties"]) ^ set(base["properties"]))))

owner = scene_schema_v6("甲", ["甲", "乙"], ["甲", "乙", "丙"], True, False,
                        ask_actions=True)
check("土地持ちの最後のターンには sell_intent がある",
      owner["properties"]["sell_intent"]["enum"] == SELL_INTENT_VALUES)
check("土地持ちにも public_act がある",
      owner["properties"]["public_act"]["enum"] == PUBLIC_ACT_VALUES)
check("必須項目に行動欄が入っている",
      {"sell_intent", "public_act", "public_act_text"} <= set(owner["required"]))

renter = scene_schema_v6("甲", ["甲", "乙"], ["甲", "乙", "丙"], False, False,
                         ask_actions=True)
check("土地を持たない人に sell_intent は出ない",
      "sell_intent" not in renter["properties"])

muni = scene_schema_v6("甲", ["甲", "乙"], ["甲", "乙", "丙"], True, False,
                       ask_actions=True, is_municipality=True)
check("行政には measure が出る",
      muni["properties"]["measure"]["enum"] == MEASURE_VALUES)
check("行政に public_act は出ない（措置の欄だけ）",
      "public_act" not in muni["properties"])
check("行政以外に measure は出ない", "measure" not in owner["properties"])


# ===========================================================================
print("\n[4] v5 のプロンプトは1文字も変わっていない（action_rows 既定 None）")
# ===========================================================================

_led = Ledger([Parcel("P01", 0, 0, "湾岸", "residential", "HH01", 1000)], {})
_a = Agent(agent_id="HH01", role="household", name="甲", persona="",
           extra={"thought": ""})
kw = dict(names={"HH01": "甲", "HH02": "乙"}, scene_id="S1", scene_label="朝",
          venue_label="床屋", present=["HH01", "HH02"], traces=[], heard=[],
          round_no=2, rounds=2, owns_parcel=True, can_publish=False,
          directs_left=1, articles_left=0, pnames={"P01": "海辺の家"})
old = build_scene_prompt_v5(_a, _led, 3, 36, **kw)
new = build_scene_prompt_v5(_a, _led, 3, 36, action_rows=action_rows_v6(True, False),
                            **kw)
check("action_rows を渡さなければ従来と1バイト一致", old == build_scene_prompt_v5(
    _a, _led, 3, 36, action_rows=None, **kw))
check("action_rows を渡すと行動欄が末尾（JSON指示の直前）に入る",
      new.startswith(old[:old.rindex("説明文を付けず")])
      and new.endswith("説明文を付けずJSONだけ返す。")
      and "public_act" in new)
check("行動欄は stance の説明を壊さない", "これは記録に残るだけで" in new)

check("紙は inbox に [回覧板・第N月] の形で並ぶ",
      inbox_rows_v5(Agent(agent_id="X", role="household", name="x", persona="",
                          inbox=[{"kind": "paper", "from": "HH01", "step": 7,
                                  "label": "回覧板", "text": "本文"}]),
                    {"HH01": "甲"}) == ["  [回覧板・第7月] 甲:「本文」"])
check("紙が無ければ inbox の見え方は従来どおり",
      inbox_rows_v5(Agent(agent_id="X", role="household", name="x", persona="",
                          inbox=[]), {}) == ["  （届いたものはない）"])


# ===========================================================================
print("\n[5] 登記簿の refusal 列と、買い手の台本の従い方（決定論・単体）")
# ===========================================================================

def tiny_ledger():
    return Ledger([Parcel("P01", 0, 0, "A", "residential", "HH01", 100),
                   Parcel("P02", 1, 0, "A", "residential", "HH01", 100),
                   Parcel("P03", 2, 0, "A", "shop", "HH02", 100)], {})


SCRIPT = {"meta": {"months": 6},
          "acquisitions": [
              {"id": "ACQ1", "month": 2, "parcel_id": "P01", "under_name": "A社",
               "kind": "sale", "traces": []},
              {"id": "ACQ2", "month": 2, "parcel_id": "P03", "under_name": "A社",
               "kind": "sale", "traces": []},
              {"id": "ACQ3", "month": 5, "parcel_id": "P02", "under_name": "B社",
               "kind": "sale", "traces": []}]}

SCRIPT_LEASE = {"meta": {"months": 6},
                "acquisitions": [
                    {"id": "ACQ9", "month": 2, "parcel_id": "P01",
                     "under_name": "A社", "kind": "lease", "traces": []}]}

led = tiny_ledger()
check("既定では誰も『当面売らない』ではない",
      all(not is_refused(led, p) for p in ("P01", "P02", "P03")))
changed = set_refusal(led, 1, "HH01", True)
check("宣言は自分の土地にだけ効く", changed == ["P01", "P02"], str(changed))
check("他人の土地は動かない", is_refused(led, "P03") is False)
check("台帳に refusal_set が2行残る",
      len([r for r in led.records if r.get("kind") == "refusal_set"]) == 2)
check("同じ宣言を繰り返しても記録は増えない（変化した分だけ）",
      set_refusal(led, 2, "HH01", True) == [])

blocked = blocked_acquisitions_v6(led, 2, SCRIPT)
check("第2月に買えないのは P01 の1件だけ",
      [b["id"] for b in blocked] == ["ACQ1"], str(blocked))
check("**賃借（lease）は止まらない**（選択肢の文言は「当面売らない」）",
      blocked_acquisitions_v6(led, 2, SCRIPT_LEASE) == [],
      "refusal は売買だけに効く")
rest = script_without(SCRIPT, {"ACQ1"})
check("外した台本から ACQ1 が消える",
      [a["id"] for a in rest["acquisitions"]] == ["ACQ2", "ACQ3"])
check("元の台本は書き換わらない",
      [a["id"] for a in SCRIPT["acquisitions"]] == ["ACQ1", "ACQ2", "ACQ3"])
check("外した台本でも月と順番は動かない",
      [a["month"] for a in rest["acquisitions"]] == [2, 5])

set_refusal(led, 3, "HH01", False)
check("解除すると refusal は消える", not is_refused(led, "P02"))
check("解除後は第5月の予定が買える（買い直しではない・元の予定の月）",
      blocked_acquisitions_v6(led, 5, SCRIPT) == [])
check("台帳に refusal_clear が残る",
      len([r for r in led.records if r.get("kind") == "refusal_clear"]) == 2)


# ===========================================================================
print("\n[6] 走行の配線（mock・実APIを叩かない）")
# ===========================================================================

class ActionMock(MockClient):
    """行動欄が出たターンで、決まった行動を返す配線確認用スタブ。

    **シミュレーションの一部ではない。** 選択肢→登記簿→買い手の台本→紙の配送
    という配線をオフラインで通すためのテストダブルである。
    """

    def __init__(self, refuse_from=2, act_step=2, **kw):
        super().__init__(**kw)
        self.refuse_from = int(refuse_from)
        self.act_step = int(act_step)

    def generate(self, system_prompt, user_prompt, schema=None, temperature=None,
                 max_tokens=None, tag="agent"):
        raw = super().generate(system_prompt, user_prompt, schema=schema,
                               temperature=temperature, max_tokens=max_tokens,
                               tag=tag)
        props = (schema or {}).get("properties", {})
        if not ({"sell_intent", "public_act", "measure"} & set(props)):
            return raw
        m = STEP_RE.search(user_prompt)
        step = int(m.group(1)) if m else 0
        d = json.loads(raw)
        if "sell_intent" in props:
            d["sell_intent"] = (SELL_INTENT_REFUSE if step >= self.refuse_from
                                else SELL_INTENT_KEEP)
        if "public_act" in props:
            d["public_act"] = (PUBLIC_ACT_CIRCULAR if step == self.act_step
                               else PUBLIC_ACT_NONE)
            d["public_act_text"] = ("同じ会社が買っている件" if step == self.act_step
                                    else "")
        if "measure" in props:
            d["measure"] = (MEASURE_VALUES[1] if step == self.act_step
                            else MEASURE_NONE)
            d["measure_text"] = "相談を受け付ける" if step == self.act_step else ""
        return json.dumps(d, ensure_ascii=False)


def run_mock(cfg_src, steps, client_factory=None, prefix="qa_v6_"):
    tmp = tempfile.mkdtemp(prefix=prefix)
    cfg = yaml.safe_load(yaml.safe_dump(cfg_src))
    cfg["llm"]["provider"] = "mock"
    cfg["steps"] = steps
    with io.open(os.path.join(tmp, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    sim = Simulation(cfg, PERSONAS, tmp)
    if client_factory is not None:
        sim.client = client_factory()
        sim.acquirer_client = sim.client
    sim.run()
    return sim, tmp


sim, d = run_mock(CFG, 5, lambda: ActionMock(refuse_from=2, act_step=2))
acts = jsonl(os.path.join(d, "actions_v6.jsonl"))
papers = jsonl(os.path.join(d, "papers_v6.jsonl"))
blocked_rows = jsonl(os.path.join(d, "blocked_v6.jsonl"))
ledger_rows = jsonl(os.path.join(d, "ledger.jsonl"))
deals = jsonl(os.path.join(d, "deals_v5.jsonl"))
traces = jsonl(os.path.join(d, "traces_v5.jsonl"))
summary = json.load(io.open(os.path.join(d, "summary.json"), encoding="utf-8"))

check("行動ログが残る（『変えない/なし』も同じ重みで）", len(acts) > 0)
check("『変えない』も記録されている",
      any(r.get("value") == SELL_INTENT_KEEP for r in acts))
check("『当面売らない』が登記簿に入る",
      len([r for r in ledger_rows if r.get("kind") == "refusal_set"]) > 0)
check("買えなかった取得が出る", len(blocked_rows) > 0, str(len(blocked_rows)))
check("買えなかった取得は台帳に script_blocked で残る",
      len([r for r in ledger_rows if r.get("kind") == "script_blocked"])
      == len(blocked_rows))
check("買えなかった取得は deals にも残る",
      len([r for r in deals if r.get("kind") == "script_blocked"])
      == len(blocked_rows))
blocked_pids = {r["parcel_id"] for r in blocked_rows}
moved = {r.get("parcel_id") for r in ledger_rows
         if r.get("kind") in ("transfer", "lease")}
check("買えなかった土地は名義が動いていない", not (blocked_pids & moved),
      str(sorted(blocked_pids & moved)))
check("買えなかった取得の兆候は配られない",
      not ({r["acq_id"] for r in blocked_rows}
           & {t.get("acq_id") for t in traces}))
check("止まっていない取得は台本どおり成立している",
      summary["v6"]["acquisitions_applied"]
      == len([r for r in ledger_rows if r.get("kind") in ("transfer", "lease")]))

check("紙が作られる", len(papers) > 0)
check("紙の見出しは4種のいずれか",
      all(p["label"] in list(PAPER_LABELS.values()) + ["役場からのお知らせ"]
          for p in papers))
check("回覧板の紙が第2月に出ている",
      any(p["act"] == PUBLIC_ACT_CIRCULAR and p["step"] == 2 for p in papers))
check("行政の措置の紙が出ている",
      any(p["act"] in MEASURE_VALUES[1:] for p in papers))
deliv = jsonl(os.path.join(d, "deliveries.jsonl"))
paper_deliv = [r for r in deliv if r.get("kind") == "paper"]
check("紙は街の全員（26主体）に届く",
      len(paper_deliv) == len(papers) * len(sim.actors),
      f"{len(paper_deliv)} vs {len(papers)}×{len(sim.actors)}")
check("紙は私信ではない（directs には入らない）",
      all(r.get("kind") != "paper" for r in jsonl(os.path.join(d, "directs_v5.jsonl"))))

logs = [p for p in (getattr(sim.client, "prompt_log", []) or [])]
tags = {str(p.get("tag", "")) for p in logs}
check("**採点用のLLMコールが1本も無い**",
      not any(t.startswith("classify") or t.endswith("_stage") for t in tags),
      str(sorted(t for t in tags if "class" in t or "stage" in t)))
check("v5e の停止の記録は作られない（v6 は停止の仕組みを持たない）",
      not os.path.exists(os.path.join(d, "defense_stop_v5e.json")))
check("4色の分類ファイルも作られない",
      not os.path.exists(os.path.join(d, "stage_labels_v5c.jsonl"))
      and not os.path.exists(os.path.join(d, "stage_labels_v5e.jsonl")))

paper_step = min(p["step"] for p in papers)
plans = [p for p in logs if str(p.get("tag", "")).endswith(":plan")]
next_month = [p for p in plans
              if STEP_RE.search(p.get("user", ""))
              and int(STEP_RE.search(p["user"]).group(1)) == paper_step + 1]
check("紙は**翌月**の観測に載る",
      next_month and any("回覧板・第" in p["user"] for p in next_month))
same_month = [p for p in plans
              if STEP_RE.search(p.get("user", ""))
              and int(STEP_RE.search(p["user"]).group(1)) == paper_step]
check("紙は出したその月には載らない",
      not any("回覧板・第" in p.get("user", "") for p in same_month))

agent_prompts = [p.get("user", "") for p in logs
                 if str(p.get("tag", "")).startswith("agent:")]
# S1〜S3 は場面の名前としてプロンプトに出るので、ここでは見ない
# （自衛レベルの S1〜S3 は v6 に存在しない＝採点用LLMごと廃止した）。
for word in ("採点", "自衛", "自衛レベル", "停止", "分類",
             "買えなかった", "refusal", "script_blocked"):
    check(f"主体のプロンプトに『{word}』が出ない",
          not any(word in t for t in agent_prompts))


# ===========================================================================
print("\n[7] 既定（行動しない世界）では v5e2 と同じ結果になる")
# ===========================================================================

sim0, d0 = run_mock(CFG, 5, prefix="qa_v6_plain_")
led0 = jsonl(os.path.join(d0, "ledger.jsonl"))
sim5, d5 = run_mock(CFG_V5E2, 5, prefix="qa_v5e2_ref_")
led5 = jsonl(os.path.join(d5, "ledger.jsonl"))
key = lambda rows: [(r.get("step"), r.get("kind"), r.get("parcel_id"),
                     r.get("under_name")) for r in rows
                    if r.get("kind") in ("transfer", "lease")]
check("誰も行動しなければ登記の動きは v5e2 と同一", key(led0) == key(led5),
      f"{len(key(led0))} vs {len(key(led5))}")
check("誰も行動しなければ買えなかった件は0",
      json.load(io.open(os.path.join(d0, "summary.json"),
                        encoding="utf-8"))["v6"]["acquisitions_blocked"] == 0)


# ===========================================================================
print("\n[8] Codexレビュー（走行前）で凍結した3点")
# ===========================================================================

# (1) config を事故で true に戻しても、v6 では採点用LLMが1本も走らない。
cfg_on = yaml.safe_load(yaml.safe_dump(CFG))
cfg_on["kpi"]["classify_utterances"] = True
cfg_on["kpi"]["classify_occupation"] = True
sim_on, d_on = run_mock(cfg_on, 3, prefix="qa_v6_gate_")
tags_on = {str(p.get("tag", "")) for p in (sim_on.client.prompt_log or [])}
check("config が true でも v6 では採点用LLMが0本（コード側で遮断）",
      not any(t.startswith("classify") or t.endswith("_stage") for t in tags_on),
      str(sorted(t for t in tags_on if "class" in t or "stage" in t)))
check("4色・占領の分類ファイルも作られない",
      not os.path.exists(os.path.join(d_on, "stage_labels_v5c.jsonl"))
      and not os.path.exists(os.path.join(d_on, "occupation_labels.jsonl")))


class AlwaysActMock(MockClient):
    """全ターンで行動のキーを返す（本来は最後のターンにしか出ない）。"""

    def generate(self, system_prompt, user_prompt, schema=None, temperature=None,
                 max_tokens=None, tag="agent"):
        raw = super().generate(system_prompt, user_prompt, schema=schema,
                               temperature=temperature, max_tokens=max_tokens,
                               tag=tag)
        props = (schema or {}).get("properties", {})
        if "talk_to" not in props:
            return raw
        d = json.loads(raw)
        d["sell_intent"] = SELL_INTENT_REFUSE
        d["public_act"] = PUBLIC_ACT_CIRCULAR
        d["public_act_text"] = "混ぜた行"
        return json.dumps(d, ensure_ascii=False)


sim_a, d_a = run_mock(CFG, 3, lambda: AlwaysActMock(), prefix="qa_v6_final_")
acts_a = jsonl(os.path.join(d_a, "actions_v6.jsonl"))
per_month = {}
for r in acts_a:
    key = (r["step"], r["agent_id"], r["field"])
    per_month[key] = per_month.get(key, 0) + 1
check("最後のターン以外の行動は無視する（月・主体・欄ごとに1回だけ）",
      bool(acts_a) and max(per_month.values()) == 1,
      str(max(per_month.values()) if per_month else 0))
papers_a = jsonl(os.path.join(d_a, "papers_v6.jsonl"))
check("紙も月に1人1枚まで",
      len({(p["step"], p["from"]) for p in papers_a}) == len(papers_a))


class EmptyActMock(MockClient):
    """行動欄を空で返す（欠損の扱いを確かめる）。"""

    def generate(self, system_prompt, user_prompt, schema=None, temperature=None,
                 max_tokens=None, tag="agent"):
        raw = super().generate(system_prompt, user_prompt, schema=schema,
                               temperature=temperature, max_tokens=max_tokens,
                               tag=tag)
        props = (schema or {}).get("properties", {})
        if not ({"sell_intent", "public_act", "measure"} & set(props)):
            return raw
        d = json.loads(raw)
        for k in ("sell_intent", "public_act", "measure"):
            if k in props:
                d[k] = ""
        return json.dumps(d, ensure_ascii=False)


sim_e, d_e = run_mock(CFG, 3, lambda: EmptyActMock(), prefix="qa_v6_empty_")
acts_e = jsonl(os.path.join(d_e, "actions_v6.jsonl"))
sum_e = json.load(io.open(os.path.join(d_e, "summary.json"), encoding="utf-8"))
check("空欄は「変えない」として記録される",
      any(r["field"] == "sell_intent" and r["value"] == SELL_INTENT_KEEP
          for r in acts_e))
check("空欄は「なし」として記録される",
      any(r["field"] == "public_act" and r["value"] == PUBLIC_ACT_NONE
          for r in acts_e))
check("空欄では登記簿も紙も動かない",
      sum_e["v6"]["papers_total"] == 0
      and sum_e["v6"]["refusal_first_month"] is None
      and sum_e["v6"]["acquisitions_blocked"] == 0)


print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
