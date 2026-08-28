#!/usr/bin/env python
"""v5c（買い手の戦略・区画の属性・日常の場・4段階の色）の走行前テスト。

    python tests/test_v5c.py

外部依存なし・実APIを叩かない（LLMは mock）。ここで固定するのは
「走行前に決めたことが、実際にその通りに凍結されているか」だけである。
"""

from __future__ import annotations

import collections
import json
import io
import os
import re
import sys
import tempfile

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from src.agents import build_roster                       # noqa: E402
from src.field_v5 import build_system_prompt_v5, validate_script_v5   # noqa: E402
from src.field_v5c import (COMMON_VENUES, build_system_prompt_v5c,    # noqa: E402
                           venue_candidates, venue_candidates_for_all)
from src.simulation import Simulation                     # noqa: E402
from run_metrics import (_v5c_rule_blue, _v5c_rule_green,  # noqa: E402
                         _v5c_rule_red, _v5c_rule_yellow, _v5c_stage,
                         stage_metrics_v5c)

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


CFG = load("configs/config_field_v5c.yaml")
PERSONAS = load(CFG["personas_file"])
ATTRS = load("configs/parcels_v5c.yaml")
SCRIPT = load("configs/events_v5c_seed85.yaml")
ROWS = {r["pid"]: r for r in ATTRS["parcels"]}
TRADABLE = {p: r for p, r in ROWS.items() if r["use"] != "public"}
ACQS = SCRIPT["acquisitions"]
AGENTS = build_roster(PERSONAS, CFG["agents"], CFG["scenario"])
BY_ID = {a.agent_id: a for a in AGENTS}
VENUE_IDS = [v["id"] for v in CFG["social"]["venues"]]

print("\n=== 1. 区画の属性が世界の形とペルソナに整合している ===")

check("非公共46区画すべてに属性が付いている", len(TRADABLE) == 46, str(len(TRADABLE)))
check("zone は中心・中間・郊外の3種だけ",
      {r["zone"] for r in TRADABLE.values()} == {"中心", "中間", "郊外"})
check("frontage は表通り・裏通り・角地の3種だけ",
      {r["frontage"] for r in TRADABLE.values()} == {"表通り", "裏通り", "角地"})
check("frontage は座標から決まる（主要道路 行(2,3)・列(3,4)）",
      all(r["frontage"] == ("角地" if (r["y"] in (2, 3) and r["x"] in (3, 4))
                            else "表通り" if (r["y"] in (2, 3) or r["x"] in (3, 4))
                            else "裏通り")
          for r in TRADABLE.values()))
check("size_class は面積から決まる（小<=110／中<=240／大）",
      all(r["size_class"] == ("小" if r["area_sqm"] <= 110
                              else "中" if r["area_sqm"] <= 240 else "大")
          for r in TRADABLE.values()))

bad_profile = []
for r in TRADABLE.values():
    persona = BY_ID[r["owner_id"]].persona if r["owner_id"] in BY_ID else ""
    age = int(re.search(r"(\d{2})歳", persona).group(1)) if re.search(r"(\d{2})歳", persona) else 0
    p = r["owner_profile"]
    if p == "高齢単身" and not (age >= 65 and "単身" in persona):
        bad_profile.append((r["pid"], p))
    if p == "不在地主" and not any(w in persona for w in
                                   ("県外", "遠方", "転勤", "移住", "支店勤務")):
        bad_profile.append((r["pid"], p))
    if p == "営業中の店" and not (BY_ID[r["owner_id"]].role == "business" or r["tenant_id"]):
        bad_profile.append((r["pid"], p))
    if p == "現役世帯" and age >= 65:
        bad_profile.append((r["pid"], p))
check("owner_profile が所有者ペルソナと矛盾しない", not bad_profile, str(bad_profile[:4]))

lodging = [r for r in TRADABLE.values() if r["use_detail"] == "旅館"]
check("use_detail=旅館 の区画には旅館の使い手が居る",
      all(r["tenant_id"] and "旅館" in BY_ID[r["tenant_id"]].persona for r in lodging),
      str([r["pid"] for r in lodging]))
check("use_detail=空地 は世界の空地と一致",
      {r["pid"] for r in TRADABLE.values() if r["use_detail"] == "空地"}
      == {r["pid"] for r in TRADABLE.values() if r["use"] == "vacant"})
check("古い住宅は住宅系だけから出る",
      all(ROWS[r["pid"]]["use"] == "residential"
          for r in TRADABLE.values() if r["use_detail"] == "古い住宅"))
check("visibility は 0〜10 の整数",
      all(isinstance(r["visibility"], int) and 0 <= r["visibility"] <= 10
          for r in TRADABLE.values()))

print("\n=== 2. 台本が買い手の戦略規則を満たしている ===")

validate_script_v5(SCRIPT)
check("台本が v5 の検査を通る", True)
pids = [a["parcel_id"] for a in ACQS]
check("同じ区画を2度取得していない", len(pids) == len(set(pids)))
check("公共区画は取得していない", all(ROWS[p]["use"] != "public" for p in pids))
in12 = [a for a in ACQS if a["month"] <= 12]
in24 = [a for a in ACQS if a["month"] <= 24]
vis12 = [ROWS[a["parcel_id"]]["visibility"] for a in in12]
vis_late = [ROWS[a["parcel_id"]]["visibility"] for a in ACQS if 13 <= a["month"] <= 24]
check("目立たない所から入っている（前半の目立ちやすさ < 後半）",
      sum(vis12) / len(vis12) < sum(vis_late) / len(vis_late),
      f"{sum(vis12)/len(vis12):.2f} vs {sum(vis_late)/len(vis_late):.2f}")
ease_first6 = [ROWS[a["parcel_id"]] for a in ACQS if a["month"] <= 6]
check("入りやすい所から入っている（最初の6か月は空地・古い住宅・不在/高齢/後継なしが8割以上）",
      sum(1 for r in ease_first6
          if r["use_detail"] in ("空地", "古い住宅")
          or r["owner_profile"] in ("不在地主", "高齢単身", "後継者なし"))
      >= 0.8 * len(ease_first6),
      str([(r["pid"], r["use_detail"], r["owner_profile"]) for r in ease_first6]))

prime = SCRIPT["meta"]["prime_event"]
prime_acq = [a for a in ACQS if a["parcel_id"] == prime["parcel_id"]][0]
check("一等地イベントは第12〜14月に1件だけ固定されている",
      12 <= int(prime["month"]) <= 14 and prime_acq["month"] == prime["month"],
      str(prime))
check("一等地イベントは中心・表通り/角地の店舗または旅館",
      prime["zone"] == "中心" and prime["frontage"] in ("表通り", "角地")
      and prime["use_detail"] in ("店舗", "旅館"), str(prime))
check("一等地イベントは営業中の店（使い手が実際に居る）",
      ROWS[prime["parcel_id"]]["owner_profile"] == "営業中の店"
      and bool(ROWS[prime["parcel_id"]]["tenant_id"]), str(prime))
check("一等地イベントは営業中の店舗・旅館のうち最も目立つ区画",
      prime["visibility"] == max(r["visibility"] for r in TRADABLE.values()
                                 if r["use_detail"] in ("店舗", "旅館")
                                 and r["owner_profile"] == "営業中の店"))
check("一等地イベントは売買（使い手が居るので賃借にしない）",
      prime_acq["kind"] == "sale", prime_acq["kind"])
check("一等地イベントは取得枠を置き換えている（月の件数を増やしていない）",
      "n = max(0, n - 1)" in
      io.open(os.path.join(ROOT, "tools/build_events_v5c.py"), encoding="utf-8").read())
check("台本生成器は「予約」と「取得済み」を分けている（未来の一等地を隣接評価に入れない）",
      "reserved: set" in
      io.open(os.path.join(ROOT, "tools/build_events_v5c.py"), encoding="utf-8").read())
check("一等地イベントの兆候は他と同じ規則（本数1〜3・売買なら登記1本）",
      1 <= len(prime_acq["traces"]) <= 3
      and (prime_acq["kind"] != "sale"
           or len([t for t in prime_acq["traces"] if t["kind"] == "registry"]) == 1),
      str(prime_acq["traces"]))
tr_counts = {len(a["traces"]) for a in ACQS}
check("兆候の本数は v5b と同じ範囲（1〜3本）", tr_counts <= {1, 2, 3}, str(tr_counts))
check("売買には登記の兆候がちょうど1本",
      all(len([t for t in a["traces"] if t["kind"] == "registry"]) == 1
          for a in ACQS if a["kind"] == "sale"))
check("賃借に登記の兆候は無い",
      all(not [t for t in a["traces"] if t["kind"] == "registry"]
          for a in ACQS if a["kind"] == "lease"))

by_month = collections.Counter(a["month"] for a in ACQS)
check("月あたりの件数レートが v5b と同じ（第1〜12月は1〜2件・一等地も枠の中）",
      all(1 <= by_month.get(m, 0) <= 2 for m in range(1, 13)),
      str({m: by_month.get(m, 0) for m in range(1, 13)}))
check("第13〜36月は0〜2件",
      all(0 <= by_month.get(m, 0) <= 2 for m in range(13, 37)))

taken = []
adj = tot = 0
for a in ACQS:
    r = ROWS[a["parcel_id"]]
    if taken:
        tot += 1
        adj += any(abs(ROWS[t]["x"] - r["x"]) + abs(ROWS[t]["y"] - r["y"]) == 1
                   for t in taken)
    taken.append(a["parcel_id"])
check("既取得の隣接を優先して面をつくっている（隣接率 >= 0.8）", adj / tot >= 0.8,
      f"{adj}/{tot}")

late_from = SCRIPT["meta"]["strategy"]["late_tier_from_month"]
exhausted = SCRIPT["meta"]["strategy"]["early_pool_exhausted_at"]
early_late_tier = [a for a in ACQS
                   if a["parcel_id"] != prime["parcel_id"]
                   and a["month"] < min(late_from, exhausted or late_from)
                   and (ROWS[a["parcel_id"]]["frontage"] == "角地"
                        or ROWS[a["parcel_id"]]["size_class"] == "大"
                        or ROWS[a["parcel_id"]]["use_detail"] == "旅館")]
check("一等地・角地・大区画・旅館は後半（早い側の在庫が尽きるまで出ない）",
      not early_late_tier, str([(a["month"], a["parcel_id"]) for a in early_late_tier]))

same_block_seq = []
last = {}
for a in sorted(ACQS, key=lambda x: (x["month"], x["parcel_id"])):
    block = ROWS[a["parcel_id"]]["block"]
    prev = last.get((block, a["under_name"]))
    if prev is not None and a["month"] - prev <= 1:
        same_block_seq.append((a["month"], block, a["under_name"]))
    last[(block, a["under_name"])] = a["month"]
check("名義を散らしている（同じ地区で同じ名義を続けない）", not same_block_seq,
      str(same_block_seq[:4]))
check("名義は4つ（A〜D社）を使う",
      {a["under_name"] for a in ACQS} == set(SCRIPT["meta"]["holders"]))

print("\n=== 3. 日常の場（会場の候補はペルソナの動線から出る・誘導しない） ===")

actors = [a for a in AGENTS if a.role != "acquirer"]
cands = venue_candidates_for_all(actors, VENUE_IDS)
check("会場は15（v5b の8＋日常の場7）", len(VENUE_IDS) == 15, str(len(VENUE_IDS)))
check("月のシーン数は4のまま（S4は月替わりの窓口）",
      CFG["scenario"]["scene_rounds"] == 2 and len(CFG["social"]["venues"]) == 15)
check("誰の候補にも共通の場（スーパー・回覧板・バス停）が入る",
      all(set(COMMON_VENUES) <= set(v) for v in cands.values()))
kid = [a for a in actors if "小学生" in a.persona or "中学生" in a.persona]
check("子どものいる世帯の候補に学校の送り迎え（V12）が入る",
      kid and all("V12" in cands[a.agent_id] for a in kid),
      str([a.agent_id for a in kid]))
elder = [a for a in actors
         if re.search(r"(\d{2})歳", a.persona)
         and int(re.search(r"(\d{2})歳", a.persona).group(1)) >= 65]
check("高齢の主体の候補に診療所（V15）と共同浴場（V03）が入る",
      elder and all({"V15", "V03"} <= set(cands[a.agent_id]) for a in elder),
      str([a.agent_id for a in elder]))
check("仲介の候補に自分の事務所（V07）が入る",
      all("V07" in cands[a.agent_id] for a in actors if a.role == "broker"))
check("行政の候補に市役所の待合（V06）が入る",
      all("V06" in cands[a.agent_id] for a in actors if a.role == "municipality"))
check("温泉・共同浴場に縁の無い主体には共同浴場が入らない",
      "V03" not in cands["HH13"], str(cands["HH13"]))
pool = collections.Counter(v for vs in cands.values() for v in vs)
check("どの会場にも候補者が2人以上いる（会話が成り立ちうる）",
      all(pool[v] >= 2 for v in VENUE_IDS), str(dict(pool)))
check("候補の数は主体ごとに違う（一律に配っていない）",
      len({len(v) for v in cands.values()}) >= 3)

a0 = BY_ID["HH01"]
sysc = build_system_prompt_v5c(a0, CFG, 48, cands["HH01"])
sysb = build_system_prompt_v5(a0, CFG, 48)
diff_lines = [ln for ln in sysb.splitlines() if ln not in sysc.splitlines()]
check("system プロンプトは v5 と同一で、違うのは並ぶ会場の行だけ",
      all(re.match(r"^\s+V\d\d: ", ln) for ln in diff_lines), str(diff_lines[:3]))
check("その人の候補にある会場だけが system プロンプトに並ぶ",
      all((f"  {v}: " in sysc) == (v in cands["HH01"]) for v in VENUE_IDS))
LEAD_WORDS = ("気づ", "注意して", "調べ", "警戒", "疑", "買い占め", "占領", "目立",
              "visibility", "裏通り", "角地", "不在地主", "高齢単身", "戦略")
check("誘導する語・区画属性の語が system プロンプトに無い",
      not [w for w in LEAD_WORDS if w in sysc],
      str([w for w in LEAD_WORDS if w in sysc]))

print("\n=== 4. 区画属性は観測に出ない（実プロンプト全文で確認・mock 1か月） ===")

tmp = tempfile.mkdtemp(prefix="qa_v5c_mock_")
cfg_mock = yaml.safe_load(yaml.safe_dump(CFG))
cfg_mock["llm"]["provider"] = "mock"
cfg_mock["steps"] = 1
cfg_mock["kpi"]["classify_utterances"] = False
sim = Simulation(cfg_mock, PERSONAS, tmp)
sim.run()
prompts = "\n".join(p.get("system", "") + "\n" + p.get("user", "")
                    for p in (getattr(sim.client, "prompt_log", []) or []))
check("mock ランでプロンプトが取れている", len(prompts) > 10000, str(len(prompts)))
LEAK = ("visibility", "目立ちやすさ", "裏通り", "表通り", "角地", "高齢単身",
        "不在地主", "後継者なし", "現役世帯", "営業中の店", "size_class",
        "owner_profile", "郊外", "中心地", "戦略")
leaked = [w for w in LEAK if w in prompts]
check("区画属性・買い手の戦略の語がどのプロンプトにも出ない", not leaked, str(leaked))
check("X社の指示文（acquirer_mandate）はどのプロンプトにも出ない",
      "未使用" not in prompts and "取得は configs" not in prompts)
check("会場は主体ごとに絞られている（全員に15会場を出していない）",
      all(len(sim.venue_choices[a.agent_id]) < 15 for a in sim.actors))
check("行き先の選択肢に HOME が常にある",
      "HOME: 自宅" in prompts)

print("\n=== 5. 4段階の色の分類（走行前に固定した定義の固定ケース） ===")

HOLD = {"A社", "B社", "C社", "D社"}
ACQD = {"P02", "P06", "P30", "P41"}


def row(text, role="household", llm=(True, False, False, False), classified=True):
    d, ar, sb, ad = llm
    return {"classified": classified,
            "rule_blue": _v5c_rule_blue(text, HOLD, ACQD),
            "rule_green": _v5c_rule_green(text, HOLD, ACQD),
            "rule_yellow": _v5c_rule_yellow(text, HOLD, ACQD),
            "rule_red": _v5c_rule_red(role, text),
            "llm_deal": d, "llm_area": ar, "llm_same_buyer": sb, "llm_admin": ad}


check("青＝個別の売買の話題",
      _v5c_stage(row("P02の名義がA社に変わったらしい")) == "blue")
check("緑＝複数の売買・面的な話",
      _v5c_stage(row("P02もP06も名義が変わった。この辺一帯で続いている",
                     llm=(True, True, False, False))) == "green")
check("黄＝名義の違う取引を同じ買い手として結んだ",
      _v5c_stage(row("A社とB社は同じ相手ではないか。裏で繋がっている",
                     llm=(True, True, True, False))) == "yellow")
check("赤＝行政が規制・届出・調査に動いた",
      _v5c_stage(row("市として届出の制度を検討し、実態把握の調査を始める",
                     role="municipality", llm=(True, False, False, True))) == "red")
check("行政以外が行政の話をしても赤にならない",
      _v5c_stage(row("市が条例で規制してくれないかしら",
                     role="household", llm=(True, False, False, True))) != "red")
check("土地と関係ない雑談はどの色にもならない",
      _v5c_stage(row("今日は温泉が混んでいたね", llm=(False, False, False, False)))
      is None)
check("ルールが当たっても LLM が否定すれば色は付かない（∧で読む）",
      _v5c_stage(row("P02の名義がA社に変わったらしい",
                     llm=(False, False, False, False))) is None)
check("LLM が肯定してもルールが当たらなければ色は付かない",
      _v5c_stage(row("なんだか最近きな臭い", llm=(True, True, True, True))) is None)
check("分類に失敗した行は色を付けない（unknown を false にしない）",
      _v5c_stage(row("P02の名義がA社に変わったらしい", classified=False)) is None)
check("色の順は赤>黄>緑>青（到達した最高段階を採る）",
      _v5c_stage(row("A社とB社が同じ動きで、市として調査に入る",
                     role="municipality", llm=(True, True, True, True))) == "red")

print("\n=== 5b. 集計（初出の順・到達した最高段階・きっかけ） ===")


def _mk_run(labels, traces=(), steps=3):
    d = tempfile.mkdtemp(prefix="qa_v5c_run_")
    with io.open(os.path.join(d, "stage_labels_v5c.jsonl"), "w", encoding="utf-8") as f:
        for r in labels:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with io.open(os.path.join(d, "traces_v5.jsonl"), "w", encoding="utf-8") as f:
        for r in traces:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for name in ("articles_v5.jsonl", "venue_choices_v5c.jsonl"):
        io.open(os.path.join(d, name), "w", encoding="utf-8").close()
    return d


L = [
    # 同じ月：S1r1 の発話（青）→ S3r2 の発話（緑）→ 同月の記事（緑）
    {"step": 1, "kind": "article", "from": "MD01", "role": "media", "scene": "S3",
     "round": 2, "venue": "", "text": "P02とP06の名義が変わり、この辺一帯で続いている",
     "deal": True, "area": True, "same_buyer": False, "admin": False,
     "classified": True},
    {"step": 1, "kind": "utterance", "from": "HH01", "role": "household", "scene": "S1",
     "round": 1, "venue": "V03", "text": "P02の名義がA社に変わったらしい",
     "deal": True, "area": False, "same_buyer": False, "admin": False,
     "classified": True},
    {"step": 2, "kind": "utterance", "from": "HH01", "role": "household", "scene": "S2",
     "round": 1, "venue": "V10", "text": "P02もP06も名義が変わった。この辺一帯で続いている",
     "deal": True, "area": True, "same_buyer": False, "admin": False,
     "classified": True},
]
TR = [{"step": 1, "agent_id": "HH01", "scene": "", "venue": "", "kind": "registry",
       "text": "第1月 P02 名義が変わった"},
      {"step": 1, "agent_id": "HH01", "scene": "S4", "venue": "V06", "kind": "survey",
       "text": "P06で測量をしている"}]
M = stage_metrics_v5c(_mk_run(L, TR), {1: {"A社"}, 2: {"A社"}, 3: {"A社"}},
                      {1: {"P02", "P06"}, 2: {"P02", "P06"}, 3: {"P02", "P06"}},
                      {}, 3, 26)
check("初出は月→シーン→ラウンドの順で採る（同じ月の記事に先を越されない）",
      M["C_blue_first"]["agent_id"] == "HH01" and M["C_blue_first"]["month"] == 1,
      str(M["C_blue_first"]))
check("きっかけは発言より前に見ていた観測だけ（後のシーンの兆候を混ぜない）",
      M["C_blue_first"]["available_antecedents"] == ["第1月 P02 名義が変わった"],
      str(M["C_blue_first"]["available_antecedents"]))
check("到達した最高段階は排他的（青から緑へ進んだ人を二重に数えない）",
      M["C_state_public_by_month"][2]["blue"] == 0
      and M["C_state_public_by_month"][2]["green"] == 2,
      str(M["C_state_public_by_month"][2]))
check("排他的な状態の合計は主体数と一致する",
      sum(M["C_state_public_by_month"][3].values()) == 26)
check("『一度でも達した人数』は補助値として別に残す",
      M["C_blue_ever_public_agents_final"] == 1
      and M["C_green_ever_public_agents_final"] == 2,
      str((M["C_blue_ever_public_agents_final"],
           M["C_green_ever_public_agents_final"])))
check("場所別の火種は、その色に初めて達した場で数える",
      M["C_blue_venue_first"] == {"V03": 1}, str(M["C_blue_venue_first"]))

check("色のラベルが無いランでは判定を出さない",
      stage_metrics_v5c(tempfile.mkdtemp(prefix="qa_v5c_empty_"), {}, {}, {}, 1, 26)
      == {"C_available": False})

print("\n=== 6. v5・v5b の成果物は不変 ===")

v5b_cfg = load("configs/config_field_v5b.yaml")
check("v5b の config は 8会場・max_tokens 1800 のまま",
      len(v5b_cfg["social"]["venues"]) == 8 and v5b_cfg["llm"]["max_tokens"] == 1800)
check("v5b の台本は 46件のまま",
      len(load("configs/events_v5b_seed85.yaml")["acquisitions"]) == 46)
check("v5 の台本に kind は無いまま",
      not any("kind" in a for a in load("configs/events_v5_seed85.yaml")["acquisitions"]))
check("v5c の config は別ファイル（v5b を書き換えていない）",
      CFG["scenario_version"] == "field_v5c"
      and CFG["events_file"] == "configs/events_v5c_seed85.yaml")
check("v5c の兆候の規則は v5b から import している（複製していない）",
      "from build_events_v5b import" in
      io.open(os.path.join(ROOT, "tools/build_events_v5c.py"), encoding="utf-8").read())
old_v5b_run = os.path.join(ROOT, "simulations", "2026-08-28_0120_100_field_v5b_runA")
if os.path.exists(os.path.join(old_v5b_run, "summary.json")):
    check("既存 v5b ランに v5c の判定は付かない（出力が変わらない）",
          stage_metrics_v5c(old_v5b_run, {}, {}, {}, 24, 26) == {"C_available": False})

print()
print(f"RESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
