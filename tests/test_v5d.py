#!/usr/bin/env python
"""v5d（窓口の廃止・街の呼び名・兆候の絞り込み）の走行前テスト。

    python tests/test_v5d.py

外部依存なし・実APIを叩かない（LLMは mock）。ここで固定するのは
「走行前に決めたことが、実際にその通りに凍結されているか」だけである。
設計の正は docs/world_design_v5d.md（§4 の節約設定は CEO 決定 2026-08-28 で上書き）。
"""

from __future__ import annotations

import collections
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

from src.agents import build_roster                                  # noqa: E402
from src.field_v5 import TRACE_TEXTS, registry_rows_v5, s4_for_step  # noqa: E402
from src.field_v5 import validate_script_v5                          # noqa: E402
from src.field_v5d import (S4_ROTATION_V5D, TRACE_TEXTS_V5D,         # noqa: E402
                           build_system_prompt_v5d, load_names_v5d,
                           s4_for_step_v5d, scene_schema_v5d)
from src.simulation import Simulation                                # noqa: E402
import run_metrics                                                   # noqa: E402
from build_events_v5d import keep_trace, seller_lived_there          # noqa: E402

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


CFG = load("configs/config_field_v5d.yaml")
CFG_V5C = load("configs/config_field_v5c.yaml")
PERSONAS = load(CFG["personas_file"])
NAMES = load("configs/parcel_names_v5c.yaml")
ATTRS = load("configs/parcels_v5c.yaml")
ROWS = {r["pid"]: r for r in ATTRS["parcels"]}
SRC = load("configs/events_v5c_seed85.yaml")
DST = load("configs/events_v5d_seed85.yaml")
AGENTS = build_roster(PERSONAS, CFG["agents"], CFG["scenario"])

# 内部ID（プロンプト・原文に出てはいけない語）
ID_RE = re.compile(r"P\d\d|R\d\d|HH\d\d|BZ\d\d|BR\d\d|MU\d\d|ME\d\d|V\d\d")

print("\n=== 1. 台本：取得は v5c と1バイトも変わっていない ===")

keys = ("id", "month", "parcel_id", "under_name", "kind", "note")
check("取得は46件", len(DST["acquisitions"]) == 46, str(len(DST["acquisitions"])))
check("取得の件数が v5c と同じ", len(DST["acquisitions"]) == len(SRC["acquisitions"]))
check("順番・月・区画・名義・kind・note が v5c と完全一致",
      [tuple(a[k] for k in keys) for a in DST["acquisitions"]]
      == [tuple(a[k] for k in keys) for a in SRC["acquisitions"]])
check("why（人間向けの注記）も v5c のまま",
      [a.get("why") for a in DST["acquisitions"]]
      == [a.get("why") for a in SRC["acquisitions"]])
check("v5d の台本は validate_script_v5 を通る",
      validate_script_v5(DST) is None)
check("v5c の台本ファイルは書き換えられていない（generated_by が v5c のまま）",
      SRC["meta"]["generated_by"] == "tools/build_events_v5c.py"
      and DST["meta"]["generated_by"] == "tools/build_events_v5d.py")

print("\n=== 2. 台本：兆候の濾過が docs/world_design_v5d.md §3 の表どおり ===")

before = collections.Counter()
after = collections.Counter()
for src, dst in zip(SRC["acquisitions"], DST["acquisitions"]):
    deal = src["kind"]
    row = ROWS[src["parcel_id"]]
    for t in src["traces"]:
        before[t["kind"]] += 1
    for t in dst["traces"]:
        after[t["kind"]] += 1
    kept = [(t["kind"], t["month"], t["audience"]) for t in dst["traces"]]
    want = [(t["kind"], t["month"], t["audience"]) for t in src["traces"]
            if keep_trace(t["kind"], deal, row)]
    check(f"{src['id']}: 残った兆候が規則どおり", kept == want, f"{kept} != {want}")

check("registry は全削除", after["registry"] == 0 and before["registry"] == 33)
check("construction は全削除", after["construction"] == 0 and before["construction"] == 9)
check("tenant_swap は賃借のみ（8本そのまま）", after["tenant_swap"] == 8)
check("survey は売買のみ（8本）", after["survey"] == 8 and before["survey"] == 9)
check("strangers は売買のみ（6本）", after["strangers"] == 6 and before["strangers"] == 7)
check("broker_known は全部残る（7本）", after["broker_known"] == before["broker_known"] == 7)
check("moving_out は売買かつ売主本人が使っていた区画のみ（4本）",
      after["moving_out"] == 4 and before["moving_out"] == 7)
check("sign_change は賃借かつ店舗のみ（seed85 には該当が無い＝0本）",
      after["sign_change"] == 0 and before["sign_change"] == 13)
check("賃借の兆候に registry は無い（v5b からの不変条件）",
      not [t for a in DST["acquisitions"] if a["kind"] == "lease"
           for t in a["traces"] if t["kind"] == "registry"])
check("「使っていた」の定義＝店子なし・空地でない・不在地主でない",
      seller_lived_there({"tenant_id": "", "use": "residential", "owner_profile": "高齢単身"})
      and not seller_lived_there({"tenant_id": "BZ01", "use": "shop",
                                  "owner_profile": "高齢単身"})
      and not seller_lived_there({"tenant_id": "", "use": "vacant",
                                  "owner_profile": "高齢単身"})
      and not seller_lived_there({"tenant_id": "", "use": "residential",
                                  "owner_profile": "不在地主"}))
check("兆候の総数 93→33", before.total() == 93 and after.total() == 33,
      f"{before.total()}->{after.total()}")
check("before/after の件数表がファイルに出ている",
      os.path.exists(os.path.join(ROOT, "docs/events_v5d_trace_diff.md")))

print("\n=== 3. 役場の窓口が世界に存在しない ===")

check("S4 は3か月周期（町内会・仲介の店先・記者の取材）",
      [k for k, _v, _l in S4_ROTATION_V5D] == ["assembly", "broker_front", "press"])
check("S4 に counter は無い",
      not [r for r in S4_ROTATION_V5D if r[0] == "counter" or r[1] == "V06"])
check("24か月のどの月も counter にならない",
      all(s4_for_step_v5d(m)[0] != "counter" for m in range(1, 25)))
check("v5c の S4（4か月周期・第2月が窓口）は変わっていない",
      s4_for_step(2)[0] == "counter" and s4_for_step(1)[0] == "assembly")
check("市役所の待合(V06)は会場として残っている",
      "V06" in [v["id"] for v in CFG["social"]["venues"]])
check("会場は v5c と同じ15か所", [v["id"] for v in CFG["social"]["venues"]]
      == [v["id"] for v in CFG_V5C["social"]["venues"]])

print("\n=== 4. 呼び名の対応表（1対1・重複なし） ===")

book = load_names_v5d(os.path.join(ROOT, "configs/parcel_names_v5c.yaml"))
check("48区画すべてに名前がある", len(book["parcels"]) == 48, str(len(book["parcels"])))
check("土地の名前に重複が無い", len(set(book["parcels"].values())) == 48)
check("locality にも重複が無い",
      len({v["locality"] for v in NAMES["land"].values()}) == 48)
check("主体26体すべてに呼び名がある", len(book["agents"]) == 26, str(len(book["agents"])))
check("主体の呼び名に重複が無い", len(set(book["agents"].values())) == 26)
check("呼び名の付いた内部IDは実在する主体である",
      set(book["agents"]) == {a.agent_id for a in AGENTS if a.role != "acquirer"})
check("16世帯の本拠 locality は互いに重複しない",
      len({NAMES["land"][v["home"]]["locality"]
           for k, v in NAMES["agents"].items() if k.startswith("HH")}) == 16)
check("公有地の登記名義は「A市」", book["registered"].get("G01") == "A市")
check("法人名義は変更しない",
      all(book["registered"].get(h) == h for h in ("X社", "A社", "B社", "C社", "D社")))
check("呼び名にも内部IDの形は出ない",
      not [n for n in list(book["parcels"].values()) + list(book["agents"].values())
           if ID_RE.search(n)])
check("v5d 用の別表は作っていない（対応表は parcel_names_v5c.yaml の1枚）",
      CFG["names_file"] == "configs/parcel_names_v5c.yaml"
      and not os.path.exists(os.path.join(ROOT, "configs/parcel_names_v5d.yaml")))

print("\n=== 5. 兆候の文言（名義が移っただけでは起きないことを直した） ===")

check("看板の文言", TRACE_TEXTS_V5D["sign_change"] == "{parcel}に見慣れない名前の看板が掛かった")
check("使い手の文言", TRACE_TEXTS_V5D["tenant_swap"] == "{parcel}を使う人が変わった")
check("v5 の文言は変えていない",
      TRACE_TEXTS["sign_change"] == "{parcel}の看板が外され、別の名前の看板が掛かった"
      and TRACE_TEXTS["tenant_swap"] == "{parcel}の店子が入れ替わった")
check("それ以外の兆候の文言は v5 と同じ",
      all(TRACE_TEXTS_V5D[k] == TRACE_TEXTS[k] for k in TRACE_TEXTS
          if k not in ("sign_change", "tenant_swap")))

print("\n=== 6. mock 2か月：プロンプト・原文に内部IDが出ない ===")

tmp = tempfile.mkdtemp(prefix="qa_v5d_mock_")
cfg_mock = yaml.safe_load(yaml.safe_dump(CFG))
cfg_mock["llm"]["provider"] = "mock"
cfg_mock["steps"] = 2
sim = Simulation(cfg_mock, PERSONAS, tmp)
sim.run()
prompts = "\n".join(p.get("system", "") + "\n" + p.get("user", "")
                    for p in (getattr(sim.client, "prompt_log", []) or []))
check("mock ランでプロンプトが取れている", len(prompts) > 10000, str(len(prompts)))
found = sorted(set(ID_RE.findall(prompts)))
check("プロンプトに内部ID（P/R/HH/BZ/BR/MU/ME/V番号）が1つも出ない", not found, str(found[:10]))
check("プロンプトに「窓口で閲覧した登記の記録」が出ない",
      "窓口で閲覧した登記の記録" not in prompts)
check("プロンプトに市役所の窓口が開く月の文が出ない",
      "市役所の窓口が開く月" not in prompts)
check("土地の名義の文言が新しいものになっている",
      "ただしあなたはその記録を見ていない" in prompts
      and "土地登記は公開情報である" not in prompts)
check("S4 の説明から窓口が外れている",
      "町内会・仲介の店先・記者の取材が月替わり" in prompts
      and "市役所の窓口・仲介の店先" not in prompts)
check("土地の呼び名が実際にプロンプトに出ている",
      "浜町の旅館" in prompts and "丘の上の集会所" in prompts)
check("持ち主の呼び名が実際にプロンプトに出ている", "の大家さん" in prompts)

traces = jsonl(os.path.join(tmp, "traces_v5.jsonl"))
check("registry_lookup の観測が1件も無い",
      not [t for t in traces if t.get("kind") == "registry_lookup"])
check("registry の兆候が1件も配られていない",
      not [t for t in traces if t.get("kind") == "registry"])
check("配られた兆候の文にも内部IDが出ない",
      not [t for t in traces if ID_RE.search(str(t.get("text", "")))])
raw = "\n".join(str(r.get("text", ""))
                for name in ("utterances_v5.jsonl", "thoughts_all.jsonl",
                             "articles_v5.jsonl", "directs_v5.jsonl")
                for r in jsonl(os.path.join(tmp, name)))
check("原文（発話・内心・記事・私信）に内部IDが出ない",
      not ID_RE.search(raw), str(sorted(set(ID_RE.findall(raw)))[:10]))
# 名簿の記者は MD01/MD02、v5c までの表示名は R/T/B/G/J 番号。どれも出てはいけない。
OLD_RE = re.compile(r"MD\d\d|AQ\d\d|[TBGJ]\d\d")
check("旧表示名・記者の内部IDもプロンプトと原文に出ない",
      not OLD_RE.search(prompts) and not OLD_RE.search(raw),
      str(sorted(set(OLD_RE.findall(prompts)))[:10]))
check("登記名義の表示が呼び名になっている",
      all("の大家さん" in str(p.registered_name) or str(p.registered_name)
          in ("A市", "X社", "A社", "B社", "C社", "D社")
          for p in sim.ledger.parcels.values()))
check("主体の呼び名が表のとおりに載っている",
      sim.names["HH01"] == "浜町の大家さん" and sim.names["MD01"] == "地域紙の記者さん")

print("\n=== 7. 占領分類器 OFF のときの出力形（未計測であって false ではない） ===")

check("config で占領分類器が off", CFG["kpi"]["classify_occupation"] is False)
check("occupation_labels.jsonl を書いていない",
      not os.path.exists(os.path.join(tmp, "occupation_labels.jsonl")))
occ = run_metrics.occupation_metrics(tmp, {}, {}, {}, 2, 26)
check("O1〜O4 は出力に現れない（未計測）",
      occ == {"O_available": False}, str(sorted(occ)[:6]))
check("4色（青緑黄赤）の判定はこれまでどおり出る",
      os.path.exists(os.path.join(tmp, "stage_labels_v5c.jsonl")))
check("既定（v5c）では占領分類器は on のまま",
      CFG_V5C["kpi"].get("classify_occupation", True) is True)

print("\n=== 8. 集計は対応表から機械生成した正規表現で読む ===")

run_metrics._use_parcel_names("field_v5d")
check("土地の呼び名から内部IDに戻して数える",
      run_metrics._parcels_in("浜町の旅館と丘の上の集会所の話") == ["P02", "P48"])
check("v5d では P番号を数えない（世界に存在しないため）",
      run_metrics._parcels_in("P02とP48の話") == [])
check("語彙は表から機械生成している（人手で足していない）",
      set(run_metrics._PARCEL_ALIAS) == set(book["parcels"].values()))
run_metrics._use_parcel_names("field_v5c")
check("v5c 以前は P番号のまま", run_metrics._parcels_in("P02とP48の話") == ["P02", "P48"])
check("S4 の周期は版で切り替わる",
      run_metrics.S4_KINDS_V5D == ["assembly", "broker_front", "press"]
      and run_metrics.S4_KINDS[1] == "counter")

print("\n=== 9. v5c 以前は1バイトも変わっていない（mock で実際に確かめる） ===")

tmp_c = tempfile.mkdtemp(prefix="qa_v5c_ref_")
cfg_c = yaml.safe_load(yaml.safe_dump(CFG_V5C))
cfg_c["llm"]["provider"] = "mock"
cfg_c["steps"] = 2
sim_c = Simulation(cfg_c, PERSONAS, tmp_c)
sim_c.run()
prompts_c = "\n".join(p.get("system", "") + "\n" + p.get("user", "")
                      for p in (getattr(sim_c.client, "prompt_log", []) or []))
check("v5c のプロンプトは従来どおり内部IDを出している",
      "内部ID:HH01" in prompts_c and bool(ID_RE.search(prompts_c)))
check("v5c は第2月に窓口が開き、登記の記録がプロンプトに出る",
      "[窓口で閲覧した登記の記録]" in prompts_c)
check("v5c では窓口で見た観測が残る",
      bool([t for t in jsonl(os.path.join(tmp_c, "traces_v5.jsonl"))
            if t.get("kind") == "registry_lookup"]))
check("v5c では占領分類器が動いて occupation_labels.jsonl が出る",
      os.path.exists(os.path.join(tmp_c, "occupation_labels.jsonl")))
check("v5c の土地登記の文言は従来のまま",
      "土地登記は公開情報である" in prompts_c)
check("registry_rows_v5 は v5c ではこれまでどおり行を返す",
      len(registry_rows_v5(sim_c.ledger, 2)) >= 1)
check("v5c の config は書き換えていない",
      CFG_V5C["scenario_version"] == "field_v5c"
      and CFG_V5C["events_file"] == "configs/events_v5c_seed85.yaml"
      and "names_file" not in CFG_V5C)

print("\n=== 10. config と出力スキーマ ===")

check("scenario_version は field_v5d", CFG["scenario_version"] == "field_v5d")
check("台本は v5d のもの", CFG["events_file"] == "configs/events_v5d_seed85.yaml")
check("区画の事実は v5c のまま", CFG["parcels_file"] == "configs/parcels_v5c.yaml")
check("Batch は使わない（施主決定 2026-08-28 22:27）",
      CFG["llm"]["batch_classify"] is False and CFG["llm"]["batch_agents"] is False)
check("world・agents・scenario は v5c と同一",
      CFG["world"] == CFG_V5C["world"] and CFG["agents"] == CFG_V5C["agents"]
      and {k: v for k, v in CFG["scenario"].items() if k != "acquirer_mandate"}
      == {k: v for k, v in CFG_V5C["scenario"].items() if k != "acquirer_mandate"})
check("seed・月数・ペルソナも v5c と同一",
      CFG["seed"] == CFG_V5C["seed"] and CFG["steps"] == CFG_V5C["steps"]
      and CFG["personas_file"] == CFG_V5C["personas_file"])
schema = scene_schema_v5d("浜町の大家さん", ["浜町の大家さん", "駅裏の大家さん"],
                          ["浜町の大家さん", "駅裏の大家さん", "地域紙の記者さん"],
                          owns_parcel=True, can_publish=True)
check("scene スキーマの構造は v5 と同じ（キーの集合）",
      set(schema["properties"]) == {"thought", "text", "talk_to", "direct_to",
                                    "direct_text", "publish", "stance"})
check("talk_to の enum は呼び名（自分は入らない）",
      schema["properties"]["talk_to"]["items"]["enum"] == ["駅裏の大家さん"])
check("direct_to の enum は NONE ＋ 自分以外の呼び名",
      schema["properties"]["direct_to"]["enum"]
      == ["NONE", "駅裏の大家さん", "地域紙の記者さん"])
sp = build_system_prompt_v5d(
    [a for a in AGENTS if a.agent_id == "MU01"][0], CFG, 48, ["V02", "V06"])
check("行政ロールの文が新しいものになっている",
      "市役所の待合で人の話を聞くことがある" in sp
      and "市役所の窓口が開く月には" not in sp)
check("行ける場所は会場名だけで並ぶ（V番号を出さない）",
      "地区公民館" in sp and "市役所の待合" in sp and not ID_RE.search(sp))

print()
print(f"RESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
