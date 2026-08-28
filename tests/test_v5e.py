#!/usr/bin/env python
"""v5e（赤の再定義・自衛レベル S1〜S3・自衛が出たら台本停止・目玉第15月・36か月）の
走行前テスト。

    python tests/test_v5e.py

外部依存なし・実APIを叩かない（LLMは mock）。ここで固定するのは
「走行前に決めたことが、実際にその通りに凍結されているか」だけである。
設計の正は docs/world_design_v5e.md。
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

from src.kpi import (STAGE_SCHEMA_V5E, STAGE_SYSTEM, STAGE_SYSTEM_V5E,  # noqa: E402
                     classify_stage_v5e)
from src.llm_client_factory import MockClient                          # noqa: E402
from src.simulation import Simulation                                  # noqa: E402
from src.stage_v5e import (V5E_LEVELS, V5E_S1_WORDS, V5E_S2_WORDS,     # noqa: E402
                           V5E_S3_WORDS, defense_level_of,
                           rule_defense_level, rule_red_v5e, stage_v5e)
import run_metrics                                                     # noqa: E402

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


CFG = load("configs/config_field_v5e.yaml")
CFG_V5D = load("configs/config_field_v5d.yaml")
PERSONAS = load(CFG["personas_file"])
DST = load("configs/events_v5e_seed85.yaml")
SRC = load("configs/events_v5d_seed85.yaml")
DESIGN = io.open(os.path.join(ROOT, "docs", "world_design_v5e.md"),
                 encoding="utf-8").read()

STEP_RE = re.compile(r"第(\d+)月")


def run_mock(cfg_src, steps, client_factory=None, prefix="qa_v5e_"):
    """mock で走らせて (sim, run_dir) を返す。実APIは叩かない。"""
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


def prompts_of(sim, skip_classifier=True):
    out = []
    for p in (getattr(sim.client, "prompt_log", []) or []):
        if skip_classifier and str(p.get("tag", "")).startswith("classify"):
            continue
        if skip_classifier and str(p.get("tag", "")).endswith("_stage"):
            continue
        out.append(p.get("system", "") + "\n" + p.get("user", ""))
    return out


class DefenseMock(MockClient):
    """第N月の発話に S1 語を混ぜ、その行だけ defense=true を返す配線確認用スタブ。

    **シミュレーションの一部ではない。** 停止の配線（月次分類→翌月以降の取得停止）を
    オフラインで通すためのテストダブルである。
    """

    # 赤は青（取引・名義・持ち主に触れている）を必要条件にするので、
    # 台本で第1月に名義が動く「A社」を文に入れる（docs/world_design_v5e.md §1-2）。
    S1_TAIL = "A社には売らない。"

    def __init__(self, trigger_step, **kw):
        super().__init__(**kw)
        self.trigger_step = int(trigger_step)

    def generate(self, system_prompt, user_prompt, schema=None, temperature=None,
                 max_tokens=None, tag="agent"):
        raw = super().generate(system_prompt, user_prompt, schema=schema,
                               temperature=temperature, max_tokens=max_tokens, tag=tag)
        props = (schema or {}).get("properties", {})
        item = props.get("results", {}).get("items", {}).get("properties", {})
        if "defense" in item:
            return self._stage(user_prompt)
        if "talk_to" in props:
            m = STEP_RE.search(user_prompt)
            if m and int(m.group(1)) == self.trigger_step:
                d = json.loads(raw)
                d["text"] = str(d.get("text") or "") + self.S1_TAIL
                return json.dumps(d, ensure_ascii=False)
        return raw

    def _stage(self, user_prompt):
        results = []
        for line in user_prompt.split("\n"):
            m = re.match(r"\s*(\d+)\.\s?(.*)$", line, flags=re.S)
            if not m:
                continue
            hit = "売らない" in m.group(2)
            results.append({"id": int(m.group(1)), "deal": False, "area": False,
                            "same_buyer": False, "defense": hit,
                            "defense_level": "S1" if hit else "none"})
        return json.dumps({"results": results}, ensure_ascii=False)


print("\n=== 1. プロンプト不変：v5e の主体への入力は v5d と完全に一致 ===")

sim_e2, dir_e2 = run_mock(CFG, 2, prefix="qa_v5e_p_")
sim_d2, dir_d2 = run_mock(CFG_V5D, 2, prefix="qa_v5d_p_")
pe, pd = prompts_of(sim_e2), prompts_of(sim_d2)
check("mock ランでプロンプトが取れている", len(pe) > 50, str(len(pe)))
check("主体へのプロンプトの本数が v5d と同じ", len(pe) == len(pd),
      f"v5e={len(pe)} v5d={len(pd)}")
# 主体の呼び出しは parallel_workers=8 の並列なので prompt_log の並びは非決定。
# 「同じプロンプトが同じ回数だけ組まれたか」を多重集合で突き合わせる。
check("主体へのプロンプト（system/plan/scene）が v5d と1バイトも違わない",
      collections.Counter(pe) == collections.Counter(pd),
      str(list(collections.Counter(pe) - collections.Counter(pd))[:1]))
check("兆候の配布も v5d と完全一致",
      jsonl(os.path.join(dir_e2, "traces_v5.jsonl"))
      == jsonl(os.path.join(dir_d2, "traces_v5.jsonl")))
check("v5e の system プロンプトは v5d の builder で作られている（呼び名が出る）",
      "浜町の旅館" in "\n".join(pe))

print("\n=== 2. 台本：v5d との差は P04 の month だけ ===")

KEYS = ("id", "parcel_id", "under_name", "kind", "note", "why")
by_id_e = {a["id"]: a for a in DST["acquisitions"]}
by_id_d = {a["id"]: a for a in SRC["acquisitions"]}
check("取得は46件", len(DST["acquisitions"]) == 46, str(len(DST["acquisitions"])))
check("取得のIDの集合が v5d と一致", set(by_id_e) == set(by_id_d))
check("month 以外の全キー・全件が v5d と一致",
      all({k: v for k, v in by_id_e[i].items() if k != "month"}
          == {k: v for k, v in by_id_d[i].items() if k != "month"} for i in by_id_d))
diff_months = sorted(i for i in by_id_d if by_id_e[i]["month"] != by_id_d[i]["month"])
check("month が変わったのは1件だけ", len(diff_months) == 1, str(diff_months))
check("変わったのは P04 の取得（ACQ18）",
      diff_months == ["ACQ18"] and by_id_e["ACQ18"]["parcel_id"] == "P04",
      str(diff_months))
check("P04 は第15月", by_id_e["ACQ18"]["month"] == 15)
check("P04 は v5d では第12月", by_id_d["ACQ18"]["month"] == 12)
months = collections.Counter(int(a["month"]) for a in DST["acquisitions"])
check("第15月の取得は2件", months[15] == 2, str(months[15]))
check("第12月の取得は1件", months[12] == 1, str(months[12]))
check("兆候の総数が v5d と一致",
      sum(len(a.get("traces") or []) for a in DST["acquisitions"])
      == sum(len(a.get("traces") or []) for a in SRC["acquisitions"]))
check("兆候の中身も全件一致（P04 の traces は0本）",
      all(by_id_e[i].get("traces") == by_id_d[i].get("traces") for i in by_id_d))
check("並びは (month, 元の並び順) の安定ソート",
      [int(a["month"]) for a in DST["acquisitions"]]
      == sorted(int(a["month"]) for a in DST["acquisitions"]))
check("id は振り直していない（ACQ18 が ACQ19 より後ろに来る）",
      [a["id"] for a in DST["acquisitions"]].index("ACQ18")
      > [a["id"] for a in DST["acquisitions"]].index("ACQ19"))
check("meta.prime_event.month が 15",
      DST["meta"]["prime_event"]["month"] == 15)
check("meta の出所が書いてある",
      DST["meta"]["source"] == "configs/events_v5d_seed85.yaml"
      and DST["meta"]["generated_by"] == "tools/build_events_v5e.py"
      and "ACQ46" in DST["meta"]["note_v5e"])

print("\n=== 3. 月数：36か月・発火しないのは ACQ46 の1件だけ ===")

check("config の steps は 36", CFG["steps"] == 36, str(CFG["steps"]))
late = [a["id"] for a in DST["acquisitions"] if int(a["month"]) > 36]
check("第36月までに発火しない取得は1件", len(late) == 1, str(late))
check("それは ACQ46（第39月・P28）",
      late == ["ACQ46"] and by_id_e["ACQ46"]["month"] == 39
      and by_id_e["ACQ46"]["parcel_id"] == "P28")
check("台本の meta.months は 60 のまま", DST["meta"]["months"] == 60)
check("config は v5d の写しで差は4点だけ",
      CFG["run_name"] == "field_v5e_a_city"
      and CFG["scenario_version"] == "field_v5e"
      and CFG["events_file"] == "configs/events_v5e_seed85.yaml"
      and {k: v for k, v in CFG.items()
           if k not in ("run_name", "scenario_version", "steps", "events_file")}
      == {k: v for k, v in CFG_V5D.items()
          if k not in ("run_name", "scenario_version", "steps", "events_file")})
check("占領分類器は off・Batch も off・prompt_order は legacy",
      CFG["kpi"]["classify_occupation"] is False
      and CFG["llm"]["batch_classify"] is False
      and CFG["llm"]["batch_agents"] is False
      and CFG["llm"]["prompt_order"] == "legacy")

print("\n=== 4. 赤の定義：語彙が docs §2 の表と一致・役割で絞らない ===")


def doc_words(level):
    for line in DESIGN.split("\n"):
        if line.startswith(f"| **{level}**"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            return tuple(w.strip() for w in cells[-1].split(","))
    return ()


# 走行前に凍結した語彙（スペック v5e Phase1 §2 の表）。テストにベタ書きして
# 「結果を見てから動かす」ことを物理的に検出する。
FROZEN_S3 = ("禁止", "差し止め", "差止", "届出", "義務付け", "義務づけ", "条例",
             "規制", "買い戻し", "買戻", "許可制", "制限区域", "勧告", "命令",
             "要綱で", "指定して", "取引を凍結", "取引の凍結")
FROZEN_S2 = ("回覧", "呼びかけ", "呼び掛け", "周知", "説明会", "注意喚起",
             "知らせよう", "知らせる", "広める", "記事にし", "紙面で",
             "組合として", "町内会として", "みんなに伝え", "声をかけて回")
FROZEN_S1 = ("売らない", "売らん", "売りません", "売却しない", "譲渡しない",
             "貸さない", "貸さん", "貸しません",
             "断ろう", "断る", "応じない", "手放さない", "手放すのはやめ",
             "売るのはやめ", "売却は見送", "控えよう", "見送ろう", "やめておこう")

check("S3 の語彙が凍結した表と一致", V5E_S3_WORDS == FROZEN_S3,
      str(set(FROZEN_S3) ^ set(V5E_S3_WORDS)))
check("S2 の語彙が凍結した表と一致", V5E_S2_WORDS == FROZEN_S2,
      str(set(FROZEN_S2) ^ set(V5E_S2_WORDS)))
check("S1 の語彙が凍結した表と一致", V5E_S1_WORDS == FROZEN_S1,
      str(set(FROZEN_S1) ^ set(V5E_S1_WORDS)))
check("docs §2 の表も機械で読める（表と定数の差分は報告で CTO 判断）",
      bool(doc_words("S3")) and bool(doc_words("S2")) and bool(doc_words("S1")))
check("レベルは S1/S2/S3 の3段", V5E_LEVELS == ("S1", "S2", "S3"))
check("S3 > S2 > S1 の順で当たる",
      rule_defense_level("条例で禁止するよう呼びかけて売らないと決めた") == "S3"
      and rule_defense_level("回覧で知らせよう。うちは売らない") == "S2"
      and rule_defense_level("うちは売らない") == "S1")
check("当たらない文は None", rule_defense_level("空き地の活用とまちづくりの話をした") is None)
check("rule_red_v5e は 自衛語 かつ 青（取引に触れている）",
      rule_red_v5e("うちは売らない", True) is True
      and rule_red_v5e("まちづくりの相談をした", True) is False)
check("【Codex指摘①】自衛語だけで青が偽なら赤にしない＝一般論が漏れない",
      rule_red_v5e("説明会をやることにした", False) is False
      and rule_red_v5e("条例で対応を検討する", False) is False
      and rule_red_v5e("回覧で知らせる", False) is False)
check("青が真なら同じ文が赤になりうる",
      rule_red_v5e("説明会をやることにした", True) is True)
check("役割で絞っていない（住民の文でも赤になりうる）",
      rule_red_v5e("うちは売らない", True) is True)
check("青のルールは v5c のものをそのまま使っている（判定を二重に持たない）",
      run_metrics._v5c_rule_blue("A社に名義が変わった", {"A社"}, set()) is True
      and run_metrics._v5c_rule_blue("説明会をやる", set(), set()) is False)
for role in ("household", "business", "broker", "media", "municipality"):
    row = {"classified": True, "rule_red": rule_red_v5e("うちは売らない", True),
           "rule_yellow": False, "rule_green": False, "rule_blue": False,
           "llm_defense": True, "llm_defense_level": "S1", "text": "うちは売らない",
           "role": role}
    check(f"{role} の文でも赤になる", stage_v5e(row) == "red")
check("v5c の赤（行政限定）は変わっていない",
      run_metrics._v5c_rule_red("municipality", "条例の検討をする") is True
      and run_metrics._v5c_rule_red("household", "条例の検討をする") is False)
check("v5c の行政語彙は変わっていない",
      run_metrics.V5C_ADMIN_WORDS
      == ("届出", "条例", "規制", "調査", "要綱", "指導", "実態把握",
          "実態を把握", "庁内", "議会", "制度", "対応を検討", "照会",
          "手続", "所管", "把握する必要", "報告する"))

print("\n=== 5. 青・緑・黄は v5c から1文字も変わっていない ===")


def para(text, start, end):
    return text[text.index(start):text.index(end)]


check("deal / area / same_buyer の段落が STAGE_SYSTEM と文字列一致",
      para(STAGE_SYSTEM_V5E, "deal:", "defense:")
      == para(STAGE_SYSTEM, "deal:", "admin:"))
check("STAGE_SYSTEM_V5E に admin は無い", "admin" not in STAGE_SYSTEM_V5E)
check("STAGE_SYSTEM（v5c）は admin のまま", "admin:" in STAGE_SYSTEM
      and "defense" not in STAGE_SYSTEM)
check("スキーマは5つの判定＋idが全て required",
      STAGE_SCHEMA_V5E["properties"]["results"]["items"]["required"]
      == ["id", "deal", "area", "same_buyer", "defense", "defense_level"])
check("defense_level は none/S1/S2/S3 の enum",
      (STAGE_SCHEMA_V5E["properties"]["results"]["items"]["properties"]
       ["defense_level"]["enum"]) == ["none", "S1", "S2", "S3"])
check("緑の面の語彙は v5c のまま",
      run_metrics.V5C_AREA_WORDS
      == ("一帯", "この辺", "その辺", "あの通り", "この通り", "界隈",
          "地区ごと", "街ごと", "町ごと", "次々", "相次", "立て続け",
          "あちこち", "周辺の土地", "まとめて", "軒並み", "近隣でも",
          "他にも", "同じ時期に", "続けて"))
check("黄の同じ買い手の語彙は v5c のまま",
      run_metrics.V5C_SAME_BUYER_WORDS
      == ("同じ会社", "同一", "同じ相手", "同じ買い手", "同じところ",
          "一つの会社", "ひとつの会社", "裏で", "背後", "実は同じ",
          "つながって", "繋がって", "関係している", "同じ人",
          "名義を変えて", "別の名前"))
check("_v5c_rule_green は変わっていない",
      run_metrics._v5c_rule_green("この辺一帯の話", set(), set()) is True
      and run_metrics._v5c_rule_green("一件だけの話", set(), set()) is False)
check("_v5c_rule_yellow は変わっていない",
      run_metrics._v5c_rule_yellow("A社とB社が動いた", {"A社", "B社"}, set()) is True)
check("stage_v5e の順序は red > yellow > green > blue",
      stage_v5e({"classified": True, "rule_red": True, "rule_yellow": True,
                 "rule_green": True, "rule_blue": True, "llm_defense": True,
                 "llm_same_buyer": True, "llm_area": True, "llm_deal": True,
                 "llm_defense_level": "S2", "text": "回覧で知らせよう"}) == "red")
check("classified が偽なら None",
      stage_v5e({"classified": False, "rule_red": True, "llm_defense": True,
                 "text": "うちは売らない"}) is None)

print("\n=== 6. 停止の配線（mock で S1 を模擬） ===")

TRIG = 2
sim_s, dir_s = run_mock(CFG, 4, lambda: DefenseMock(TRIG), prefix="qa_v5e_stop_")
stop = json.load(io.open(os.path.join(dir_s, "defense_stop_v5e.json"), encoding="utf-8"))
deals = jsonl(os.path.join(dir_s, "deals_v5.jsonl"))
ledger = jsonl(os.path.join(dir_s, "ledger.jsonl"))
applied = [r for r in ledger if r.get("kind") in ("transfer", "lease")]
stopped_rows = [d for d in deals if d.get("kind") == "script_stopped"]
script_months = collections.Counter(
    int(a["month"]) for a in DST["acquisitions"] if int(a["month"]) <= 4)
check("停止が立った", stop["stopped"] is True, json.dumps(stop, ensure_ascii=False)[:200])
check(f"trigger_month = {TRIG}", stop["trigger_month"] == TRIG)
check(f"stop_from_month = {TRIG + 1}", stop["stop_from_month"] == TRIG + 1)
check("トリガー行に原文全文が入っている",
      bool(stop["triggers"]) and all(DefenseMock.S1_TAIL in t["text"]
                                     for t in stop["triggers"]))
check("トリガー行にレベルと出所が入っている",
      all(t["level"] == "S1" and t["level_source"] == "llm" for t in stop["triggers"]))
check("トリガー行に月・主体・役割・場が入っている",
      all(set(t) >= {"step", "from", "role", "kind", "scene", "venue"}
          for t in stop["triggers"]))
check(f"第{TRIG}月までの取得は台本どおり成立",
      len(applied) == sum(script_months[m] for m in range(1, TRIG + 1)),
      f"applied={len(applied)}")
check(f"第{TRIG + 1}月以降の取得は0件",
      not [r for r in applied if int(r.get("step", 0)) > TRIG])
check("止めた取得が deals_v5.jsonl に script_stopped として残る",
      len(stopped_rows) == sum(script_months[m] for m in range(TRIG + 1, 5)),
      str(len(stopped_rows)))
check("script_stopped の理由は defense_detected",
      all(d["reason"] == "defense_detected" for d in stopped_rows))
check("台帳に script_suspended の記録が残る",
      len([r for r in ledger if r.get("kind") == "script_suspended"])
      == len(stopped_rows))
check("summary の v5e が実測と合う",
      sim_s.summary["v5e"]["stopped"] is True
      and sim_s.summary["v5e"]["trigger_month"] == TRIG
      and sim_s.summary["v5e"]["acquisitions_applied"] == len(applied)
      and sim_s.summary["v5e"]["acquisitions_suspended"] == len(stopped_rows)
      and sim_s.summary["v5e"]["levels_seen"] == ["S1"],
      json.dumps(sim_s.summary["v5e"], ensure_ascii=False))

sim_n, dir_n = run_mock(CFG, 4, prefix="qa_v5e_nostop_")
stop_n = json.load(io.open(os.path.join(dir_n, "defense_stop_v5e.json"),
                           encoding="utf-8"))
ledger_n = jsonl(os.path.join(dir_n, "ledger.jsonl"))
applied_n = [r for r in ledger_n if r.get("kind") in ("transfer", "lease")]
check("赤が出ないランは停止しない", stop_n == {"stopped": False, "months": 4})
check("停止しないランは取得が台本どおり全件成立",
      len(applied_n) == sum(script_months.values()), str(len(applied_n)))
check("停止しないランに script_stopped は無い",
      not [d for d in jsonl(os.path.join(dir_n, "deals_v5.jsonl"))
           if d.get("kind") == "script_stopped"])
check("停止しないランの summary も『出なかった』を残す",
      sim_n.summary["v5e"]["stopped"] is False
      and sim_n.summary["v5e"]["acquisitions_suspended"] == 0)

print("\n=== 7. 停止が主体に漏れていない ===")

# 「S1」〜「S4」はこの世界のシーンID（plan_s1／S1〜S4）としてプロンプトに出るので、
# 漏洩の検出語には使えない。停止・自衛・分類に関する語をこの一覧で見る。
LEAK = ("defense", "停止", "自衛", "script_stopped", "script_suspended",
        "defense_detected", "分類", "取得を止", "買い手が止")
after = []
for p in (getattr(sim_s.client, "prompt_log", []) or []):
    tag = str(p.get("tag", ""))
    if tag.startswith("classify") or tag.endswith("_stage"):
        continue
    body = p.get("system", "") + "\n" + p.get("user", "")
    m = STEP_RE.search(p.get("user", ""))
    if m and int(m.group(1)) > TRIG:
        after.append(body)
check("停止後の月のプロンプトが取れている", len(after) > 10, str(len(after)))
found = sorted({w for body in after for w in LEAK if w in body})
check("停止後のプロンプトに停止・自衛・分類の語が1つも出ない", not found, str(found))
check("停止後の月も会話・内心の観測が続いている",
      bool([u for u in sim_s.ledger.v5_utterances if int(u["step"]) > TRIG])
      and bool([t for t in sim_s.thoughts if int(t["step"]) > TRIG]))
# 起きなかった取得の兆候は配らない（CTO 判断 2026-08-29・docs/world_design_v5e.md §3）。
# 兆候は「その取得が現実に生む痕跡」なので、取得が成立しない以上その痕跡も存在しない。
_traces_stop = jsonl(os.path.join(dir_s, "traces_v5.jsonl"))
_susp_ids = {d.get("acq_id") for d in stopped_rows}
check("止めた取得に由来する兆候は1本も配られていない",
      not [t for t in _traces_stop if t.get("acq_id") in _susp_ids],
      str([t.get("acq_id") for t in _traces_stop if t.get("acq_id") in _susp_ids][:5]))
check("停止前に配られた兆候はそのまま残っている（取りやめになった準備の痕跡）",
      all(int(t.get("step", 0)) <= TRIG for t in _traces_stop
          if t.get("acq_id") not in _susp_ids))
check("止めなかったランの兆候は減っていない（落としたのは停止分だけ）",
      len(jsonl(os.path.join(dir_n, "traces_v5.jsonl"))) >= len(_traces_stop))
d_stop = jsonl(os.path.join(dir_s, "deliveries.jsonl"))
d_none = jsonl(os.path.join(dir_n, "deliveries.jsonl"))
check("deliveries.jsonl に停止由来の行が増えていない",
      not [r for r in d_stop if r.get("kind") not in ("scene", "direct", "article")])
check("traces_v5.jsonl に停止由来の行が増えていない",
      not [r for r in jsonl(os.path.join(dir_s, "traces_v5.jsonl"))
           if "defense" in json.dumps(r, ensure_ascii=False)])
check("停止しても兆候の配布経路は同じ種類のまま",
      {r.get("kind") for r in jsonl(os.path.join(dir_s, "traces_v5.jsonl"))}
      <= {r.get("kind") for r in jsonl(os.path.join(dir_n, "traces_v5.jsonl"))})

print("\n=== 8. 月次の分類＝事後に組み立てた occ_rows と過不足なく一致 ===")

sim6, dir6 = run_mock(CFG, 6, prefix="qa_v5e_rows_")
labels = jsonl(os.path.join(dir6, "stage_labels_v5e.jsonl"))
occ_rows = ([{"kind": "utterance", **u} for u in sim6.ledger.v5_utterances]
            + [{"kind": "thought", **t} for t in sim6.thoughts]
            + [{"kind": "article", **a} for a in sim6.ledger.v5_articles])


def key(r):
    return (int(r.get("step", 0)), r.get("kind"), r.get("from"), str(r.get("text", "")))


check("月次分類の行数と事後の occ_rows の行数が一致",
      len(labels) == len(occ_rows), f"{len(labels)} vs {len(occ_rows)}")
check("行の集合が過不足なく一致（step/kind/from/text）",
      collections.Counter(key(r) for r in labels)
      == collections.Counter(key(r) for r in occ_rows))
check("utterance / thought / article の3種が全部入っている",
      {r["kind"] for r in labels} == {"utterance", "thought", "article"},
      str(sorted({r["kind"] for r in labels})))
check("v5e では事後の stage_labels_v5c.jsonl を書かない",
      not os.path.exists(os.path.join(dir6, "stage_labels_v5c.jsonl")))
check("v5e では占領分類器を走らせない（config で off）",
      not os.path.exists(os.path.join(dir6, "occupation_labels.jsonl")))
check("月次分類のジョブキーが月ごとに分かれている",
      sorted({p.get("tag") for p in (sim6.client.prompt_log or [])
              if str(p.get("tag", "")).startswith("classify_stage")})
      == ["classify_stage_v5e"])

print("\n=== 9. unknown が false に化けない ===")


class BrokenClient:
    """分類の解析に失敗する状況を模擬する（配線確認用）。"""

    model = "broken"

    def generate(self, system_prompt, user_prompt, schema=None, temperature=None,
                 max_tokens=None, tag="agent"):
        return "これはJSONではない"


rows = [{"step": 1, "from": "HH01", "kind": "utterance", "text": "うちは売らない"}]
out = classify_stage_v5e(BrokenClient(), rows, batch=25)
check("解析失敗の行は classified: false", out[0]["classified"] is False)
check("解析失敗の行の defense は None（false ではない）", out[0]["defense"] is None)
check("解析失敗の行の deal / area / same_buyer も None",
      out[0]["deal"] is None and out[0]["area"] is None
      and out[0]["same_buyer"] is None)
check("解析失敗の行の defense_level も None", out[0]["defense_level"] is None)
check("解析失敗の行は赤にならない",
      stage_v5e({"classified": False, "rule_red": True, "rule_yellow": False,
                 "rule_green": False, "rule_blue": False,
                 "llm_defense": out[0]["defense"], "text": rows[0]["text"]}) is None)

class PartialClient:
    model = "partial"

    def generate(self, system, user, schema=None, temperature=0.0,
                 max_tokens=0, tag="", **kw):
        return json.dumps({"results": [{"id": 1, "frame": "our_town",
                                        "about_acquisition": True}]})


class ShortClient:
    model = "short"

    def generate_many(self, items, tag="", kind="", job_key=""):
        return [json.dumps({"results": [{"id": 1, "deal": True, "area": False,
                                         "same_buyer": False, "defense": False,
                                         "defense_level": "none"}]})]


class BadEnumClient:
    model = "bad-enum"

    def generate(self, system, user, schema=None, temperature=0.0,
                 max_tokens=0, tag="", **kw):
        return json.dumps({"results": [{"id": 1, "deal": True, "area": False,
                                        "same_buyer": False, "defense": True,
                                        "defense_level": "STRONG"}]})


_p = classify_stage_v5e(PartialClient(), [{"step": 1, "text": "x"}], batch=25)
check("missing fields stay unknown (never coerced to false)",
      _p[0]["classified"] is False and _p[0]["deal"] is None
      and _p[0]["defense"] is None and _p[0]["defense_level"] is None)

_s = classify_stage_v5e(ShortClient(), [{"step": 1, "text": "a"}] * 30, batch=25)
check("short raw list keeps every row (tail chunk stays unknown)",
      len(_s) == 30 and _s[0]["classified"] is True
      and all(r["classified"] is False for r in _s[25:]))

_b = classify_stage_v5e(BadEnumClient(), [{"step": 1, "text": "y"}], batch=25)
check("out-of-enum defense_level becomes None without dropping the row",
      _b[0]["classified"] is True and _b[0]["defense"] is True
      and _b[0]["defense_level"] is None)
check("that row still gets a level from the rule side",
      (defense_level_of({"classified": True, "rule_red": True, "rule_yellow": False,
                         "rule_green": False, "rule_blue": True,
                         "llm_defense": True, "llm_defense_level": None,
                         "text": "A社には売らない"})
       or {}).get("level_source") == "rule")


print("\n=== 10. 自衛レベルの主値と出所 ===")


def red_row(text, llm_level):
    return {"classified": True, "rule_red": rule_red_v5e(text, True), "rule_yellow": False,
            "rule_green": False, "rule_blue": False, "llm_defense": True,
            "llm_defense_level": llm_level, "text": text}


check("主値は LLM の defense_level",
      defense_level_of(red_row("うちは売らない", "S3"))
      == {"level": "S3", "level_source": "llm", "llm_level": "S3",
          "rule_level": "S1"})
check("LLM が none ならルール側の最上位を使う",
      defense_level_of(red_row("回覧で知らせよう", "none"))
      == {"level": "S2", "level_source": "rule", "llm_level": None,
          "rule_level": "S2"})
check("LLM の値が欠けてもルール側で埋まる",
      defense_level_of(red_row("条例で禁止する", None))["level"] == "S3")
check("赤でない行にはレベルを付けない",
      defense_level_of({"classified": True, "rule_red": False, "rule_yellow": False,
                        "rule_green": False, "rule_blue": False,
                        "llm_defense": False, "text": "まちづくりの話"}) is None)

print("\n=== 11. run_metrics の v5e 集計 ===")

met = run_metrics.metrics_v5(dir_s)
check("version は field_v5e", met["version"] == "field_v5e")
check("red_definition_version は v5e", met.get("red_definition_version") == "v5e")
check("S_counts が3レベル分ある", set(met["S_counts"]) == {"S1", "S2", "S3"})
check("S1 の行が数えられている", met["S_counts"]["S1"] > 0, str(met["S_counts"]))
check("S_first の S1 に原文全文が入っている",
      met["S_first"]["S1"] and DefenseMock.S1_TAIL in met["S_first"]["S1"]["text"])
check("S_first の S1 に月・主体・役割・場が入っている",
      set(met["S_first"]["S1"]) >= {"month", "agent_id", "role", "kind", "scene",
                                    "venue", "venue_label", "text"})
check("S_rows は赤の行の全件", len(met["S_rows"]) == met["C_red_rows"],
      f"{len(met['S_rows'])} vs {met['C_red_rows']}")
check("S_level_agreement の4区分が揃っている",
      set(met["S_level_agreement"]) == {"llm_and_rule_same", "llm_only",
                                        "rule_only", "disagree"})
check("LLM とルールが一致した件数が数えられている",
      met["S_level_agreement"]["llm_and_rule_same"] == met["S_counts"]["S1"],
      json.dumps(met["S_level_agreement"]))
check("defense_stop がそのまま入っている", met["defense_stop"] == stop)
check("赤は行政以外の役割からも出ている（役割で絞っていない）",
      set(met["C_red_by_role"]) - {"municipality"} != set(),
      json.dumps(met["C_red_by_role"], ensure_ascii=False))

met_n = run_metrics.metrics_v5(dir_n)
check("停止しないランでも v5e 集計が出る",
      met_n.get("red_definition_version") == "v5e"
      and met_n["S_counts"] == {"S1": 0, "S2": 0, "S3": 0})
check("停止しないランの defense_stop は『出なかった』",
      met_n["defense_stop"] == {"stopped": False, "months": 4})

print("\n=== 12. v5c / v5d を壊していない ===")

check("v5c の config は書き換えていない",
      load("configs/config_field_v5c.yaml")["scenario_version"] == "field_v5c")
check("v5d の config は書き換えていない",
      CFG_V5D["scenario_version"] == "field_v5d" and CFG_V5D["steps"] == 24
      and CFG_V5D["events_file"] == "configs/events_v5d_seed85.yaml")
check("v5d のランは従来どおり事後に stage_labels_v5c.jsonl を書く",
      os.path.exists(os.path.join(dir_d2, "stage_labels_v5c.jsonl")))
check("v5d のランに v5e の出力は出ない",
      not os.path.exists(os.path.join(dir_d2, "stage_labels_v5e.jsonl"))
      and not os.path.exists(os.path.join(dir_d2, "defense_stop_v5e.json")))
check("v5d の summary に v5e キーは無い", "v5e" not in sim_d2.summary)
check("_v5c_stage は v5c のまま（admin を読む）",
      run_metrics._v5c_stage({"classified": True, "rule_red": True,
                              "llm_admin": True}) == "red")

print()
print(f"RESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
