#!/usr/bin/env python
"""API費の節約（第1便）の走行前テスト。

    python tests/test_cost_saving.py

外部依存なし・実APIを叩かない。ここで固定するのは1点だけである：

  **節約は「呼び方・並び順・出力上限」だけを変え、プロンプトの意味内容・分類ルール・
    世界設計には一切触れていない。**

そのために、
  - 既定（何も指定しない）が従来の文字列・従来の設定を1バイト違わず再現すること
  - stable_first は行を並べ替えただけで、行の集合が legacy と同一であること
  - 分類器を generate_many 経由にしても、渡すプロンプト・スキーマ・温度・上限が
    従来と同一で、結果も同一であること
  - Batch が失敗・欠落したとき同期にフォールバックし、結果が欠けないこと
  - ジョブIDが run_dir に保存され、途中で落ちても取り直せること
を実際に動かして確かめる。
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.agents import Agent                                    # noqa: E402
from src.field_v5 import (PROMPT_ORDER_LEGACY,                   # noqa: E402
                          PROMPT_ORDER_STABLE_FIRST,
                          build_plan_prompt_v5, build_scene_prompt_v5)
from src.kpi import (classify_occupation, classify_stage_v5c,    # noqa: E402
                     classify_utterances)
from src.llm_batch import BatchRunner, request_fingerprint       # noqa: E402
from src.llm_client_factory import (GeminiClient, MockClient,     # noqa: E402
                                    UsageMeter)
from src.world import Ledger, build_town                         # noqa: E402

PASS = FAIL = 0


def check(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}  {detail}")


def _world():
    cfg = yaml.safe_load(io.open(os.path.join(ROOT, "configs", "config_field_v5c.yaml"),
                                 encoding="utf-8"))
    parcels = build_town(cfg["world"], ["HH01", "HH02"], ["BZ01"], "MU01")
    ledger = Ledger(parcels, {"HH01": 0, "HH02": 0, "BZ01": 0, "MU01": 0})
    for p in ledger.parcels.values():
        p.registered_name = "住民A"
    agent = Agent("HH01", "household", "住民A", "70歳。共同浴場に通う。")
    names = {"HH01": "住民A", "HH02": "住民B", "BZ01": "事業者A", "MU01": "市"}
    return cfg, ledger, agent, names


print("--- ① 並び順：既定は従来どおり・stable_first は行の並べ替えだけ ---")

cfg, ledger, agent, names = _world()
plan_kw = dict(names=names, traces=[], s4_label="町内会", s4_venue="V02")
plan_default = build_plan_prompt_v5(agent, ledger, 3, 24, names, [], "町内会", "V02")
plan_legacy = build_plan_prompt_v5(agent, ledger, 3, 24, names, [], "町内会", "V02",
                                   prompt_order=PROMPT_ORDER_LEGACY)
plan_stable = build_plan_prompt_v5(agent, ledger, 3, 24, names, [], "町内会", "V02",
                                   prompt_order=PROMPT_ORDER_STABLE_FIRST)
check("plan: 引数を渡さなければ legacy と完全一致", plan_default == plan_legacy)
check("plan: legacy の末尾2行が従来の指示文のまま",
      plan_legacy.split("\n")[-1] == "説明文を付けずＪＳＯＮだけ返す。"
      and plan_legacy.split("\n")[-2].startswith("まず thought"),
      plan_legacy.split("\n")[-2:])
check("plan: stable_first は同じ行を並べ替えただけ（行の集合が一致）",
      sorted(x for x in plan_stable.split("\n") if x)
      == sorted(x for x in plan_legacy.split("\n") if x))
check("plan: stable_first は指示文が先頭",
      plan_stable.split("\n")[0].startswith("まず thought")
      and plan_stable.split("\n")[1] == "説明文を付けずＪＳＯＮだけ返す。")
check("plan: stable_first でも本文の並びは変えていない",
      "\n".join(plan_stable.split("\n")[3:]) == plan_legacy.rsplit("\n", 2)[0])

scene_args = (agent, ledger, 3, 24, names, "S1", "朝", "共同浴場", ["HH01", "HH02"],
              [], [], 1, 2)
scene_kw = dict(owns_parcel=True, can_publish=False, directs_left=2, articles_left=0)
sc_default = build_scene_prompt_v5(*scene_args, **scene_kw)
sc_legacy = build_scene_prompt_v5(*scene_args, prompt_order=PROMPT_ORDER_LEGACY,
                                  **scene_kw)
sc_stable = build_scene_prompt_v5(*scene_args, prompt_order=PROMPT_ORDER_STABLE_FIRST,
                                  **scene_kw)
check("scene: 引数を渡さなければ legacy と完全一致", sc_default == sc_legacy)
check("scene: legacy の末尾は従来どおり JSON だけ返す指示",
      sc_legacy.split("\n")[-1] == "説明文を付けずJSONだけ返す。")
check("scene: legacy の指示文はラウンド説明の直後（従来の位置）",
      sc_legacy.split("\n")[sc_legacy.split("\n").index(
          "まず thought（内心）を書き、それを踏まえてこの場で話すことを書く。") - 1]
      .startswith("この場面のやりとりは"))
check("scene: stable_first は同じ行を並べ替えただけ（行の集合が一致）",
      sorted(x for x in sc_stable.split("\n") if x)
      == sorted(x for x in sc_legacy.split("\n") if x))
check("scene: stable_first は3行の指示文が先頭",
      sc_stable.split("\n")[:3] == [
          "まず thought（内心）を書き、それを踏まえてこの場で話すことを書く。",
          "話すことがなければ text は空文字でよい。",
          "説明文を付けずJSONだけ返す。"],
      sc_stable.split("\n")[:3])
check("scene: stance の指示（所有者だけに出る）は stable_first でも残る",
      "stance に、今のあなたが自分の土地を手放すことを考えているか" in sc_stable)
sc_stable_np = build_scene_prompt_v5(*scene_args, prompt_order=PROMPT_ORDER_STABLE_FIRST,
                                     owns_parcel=False, can_publish=False,
                                     directs_left=2, articles_left=0)
check("scene: 所有していなければ stance の指示は出ない（従来と同じ条件分岐）",
      "stance に、" not in sc_stable_np)

print("--- ② 設定：既定は従来どおり・節約は別ファイル ---")

base = yaml.safe_load(io.open(os.path.join(ROOT, "configs", "config_field_v5c.yaml"),
                              encoding="utf-8"))
econ = yaml.safe_load(io.open(os.path.join(ROOT, "configs",
                                           "config_field_v5c_econ.yaml"),
                              encoding="utf-8"))
diff = {k for k in set(base) | set(econ) if base.get(k) != econ.get(k)}
check("econ config の差は llm と run_name だけ（世界は同一）",
      diff == {"llm", "run_name"}, sorted(diff))
llm_diff = {k for k in set(base["llm"]) | set(econ["llm"])
            if base["llm"].get(k) != econ["llm"].get(k)}
check("econ の llm 差分は節約キーのみ（model/temperature/max_tokens は不変）",
      llm_diff <= {"enable_cache", "cache_ttl_seconds", "batch_classify",
                   "batch_agents", "batch_poll_interval", "batch_timeout_sec",
                   "prompt_order", "thought_max_tokens"}, sorted(llm_diff))
check("v5c 既定は明示キャッシュ off・Batch なし・並び順 legacy",
      base["llm"].get("enable_cache") is False
      and not base["llm"].get("batch_classify")
      and not base["llm"].get("batch_agents")
      and base["llm"].get("prompt_order") is None)
check("econ の thought_max_tokens は max_tokens と同じ 2200 から始める",
      econ["llm"]["thought_max_tokens"] == econ["llm"]["max_tokens"] == 2200)

print("--- ③ Simulation の既定値 ---")

from src.simulation import Simulation                            # noqa: E402

personas = yaml.safe_load(io.open(os.path.join(ROOT, base["personas_file"]),
                                  encoding="utf-8"))
with tempfile.TemporaryDirectory() as tmp:
    mock_cfg = dict(base)
    mock_cfg["llm"] = {**base["llm"], "provider": "mock"}
    sim = Simulation(mock_cfg, personas, tmp)
    check("既定の prompt_order は legacy", sim.prompt_order == "legacy")
    check("既定の agent_max_tokens は llm.max_tokens と同じ",
          sim.agent_max_tokens == base["llm"]["max_tokens"])
    check("既定で Batch は使わない", not getattr(sim.client, "batch_kinds", set()))
    econ_cfg = dict(econ)
    econ_cfg["llm"] = {**econ["llm"], "provider": "mock"}
    sim2 = Simulation(econ_cfg, personas, tmp)
    # 並び替えは実測で効果が無かったので、econ でも既定は legacy（スイッチは残す）
    check("econ の prompt_order は legacy（並び替えは効果なしと実測）",
          sim2.prompt_order == "legacy")
    check("Batch ジョブ台帳は run_dir の下に置く",
          sim2.jobs_dir == os.path.join(tmp, "batch_jobs"))

print("--- ④ 分類器：generate_many 経由でも渡すものが同一 ---")


class RecordingClient:
    """generate() だけを持つ従来型クライアント（generate_many を持たない）。"""

    def __init__(self):
        self.calls = []
        self.inner = MockClient(seed=1)

    def generate(self, system_prompt, user_prompt, schema=None, temperature=None,
                 max_tokens=None, tag="agent"):
        self.calls.append({"system": system_prompt, "user": user_prompt,
                           "schema": schema, "temperature": temperature,
                           "max_tokens": max_tokens, "tag": tag})
        return self.inner.generate(system_prompt, user_prompt, schema=schema,
                                   temperature=temperature, max_tokens=max_tokens,
                                   tag=tag)


class RecordingManyClient(RecordingClient):
    """generate_many を持つ新型（Batch 設定なし＝同期に落ちる）。"""

    def generate_many(self, items, tag="agent", kind="agents", job_key=None):
        return [self.generate(it["system_prompt"], it["user_prompt"],
                              schema=it.get("schema"),
                              temperature=it.get("temperature"),
                              max_tokens=it.get("max_tokens"),
                              tag=it.get("tag") or tag) for it in items]


rows = [{"step": i % 5 + 1, "role": "household", "name": "住民A",
         "text": f"P0{i % 9 + 1} の名義が変わったらしい。"} for i in range(60)]
for fn, label in ((classify_utterances, "classify"),
                  (classify_occupation, "classify_occupation"),
                  (classify_stage_v5c, "classify_stage_v5c")):
    old_c, new_c = RecordingClient(), RecordingManyClient()
    out_old = fn(old_c, list(rows), batch=25)
    out_new = fn(new_c, list(rows), batch=25)
    keys = ("system", "user", "schema", "temperature", "max_tokens", "tag")
    same = [{k: c[k] for k in keys} for c in old_c.calls] == \
           [{k: c[k] for k in keys} for c in new_c.calls]
    check(f"{label}: 渡す system/user/schema/温度/上限/tag が従来と同一", same)
    check(f"{label}: 結果も従来と同一", out_old == out_new)
    check(f"{label}: コール数が従来と同じ（チャンク分割を変えていない）",
          len(old_c.calls) == len(new_c.calls) == 3,
          (len(old_c.calls), len(new_c.calls)))

print("--- ⑤ MockClient.generate_many ---")

mc = MockClient(seed=7)
items = [{"system_prompt": "S", "user_prompt": f"第{i}月 P01", "schema": None}
         for i in range(1, 4)]
many = mc.generate_many(items, tag="classify", kind="classify")
mc2 = MockClient(seed=7)
one = [mc2.generate("S", it["user_prompt"], schema=None, tag="classify")
       for it in items]
check("MockClient: generate_many は generate の逐次と同じ結果", many == one)
check("MockClient: Batch は決して使わない（配線確認用）", mc.batch_kinds == set())

print("--- ⑥ BatchRunner：ジョブIDの保存・再取得・フォールバック ---")


class _Job:
    def __init__(self, name, state, responses=None, err=None):
        self.name = name
        self.state = type("S", (), {"name": state})()
        self.error = err
        self.dest = type("D", (), {"inlined_responses": responses or []})()


class _Resp:
    def __init__(self, text, err=None):
        self.text = text
        self.error = err
        self.usage_metadata = type("U", (), {"prompt_token_count": 10,
                                             "cached_content_token_count": 0,
                                             "candidates_token_count": 5})()
        self.response = self


class _Item:
    def __init__(self, text, err=None):
        self.response = None if err else _Resp(text)
        self.error = err


class FakeBatches:
    def __init__(self, responses, state="JOB_STATE_SUCCEEDED"):
        self.responses = responses
        self.state = state
        self.created = 0
        self.last_src = None

    def create(self, model, src, config=None):
        self.created += 1
        self.last_src = list(src)
        return _Job("batches/fake-1", self.state, self.responses)

    def get(self, name):
        return _Job(name, self.state, self.responses)


class FakeClient:
    def __init__(self, batches):
        self.batches = batches


class FakeTypes:
    class InlinedRequest:
        def __init__(self, model=None, contents=None, config=None):
            self.model, self.contents, self.config = model, contents, config

    class GenerateContentConfig:
        def __init__(self, **kw):
            self.kw = kw

    class ThinkingConfig:
        def __init__(self, thinking_budget=None):
            self.thinking_budget = thinking_budget


reqs = [{"system_prompt": "S", "user_prompt": f"u{i}", "schema": None,
         "temperature": 0.0, "max_tokens": 100, "thinking_budget": None}
        for i in range(3)]

with tempfile.TemporaryDirectory() as tmp:
    jobs = os.path.join(tmp, "batch_jobs")
    fb = FakeBatches([_Item("a"), _Item("b"), _Item("c")])
    r = BatchRunner(FakeClient(fb), FakeTypes, "m", jobs_dir=jobs, poll_interval=0)
    texts, usages = r.run("classify_x", reqs)
    check("Batch: 全行が返る", texts == ["a", "b", "c"], texts)
    check("Batch: usage も行ごとに取れる",
          usages[0]["input_tokens"] == 10 and usages[0]["output_tokens"] == 5)
    rec_path = os.path.join(jobs, "classify_x.json")
    check("Batch: ジョブIDを run_dir へ保存する", os.path.exists(rec_path))
    rec = json.load(io.open(rec_path, encoding="utf-8"))
    check("Batch: 台帳にジョブ名と件数が入る",
          rec["job_name"] == "batches/fake-1" and rec["requests"] == 3, rec)
    # 同じ tag・同じ件数で再実行すると、新しいジョブを作らず取り直す
    r2 = BatchRunner(FakeClient(fb), FakeTypes, "m", jobs_dir=jobs, poll_interval=0)
    before = fb.created
    texts2, _ = r2.run("classify_x", reqs)
    check("Batch: 途中で落ちても同じジョブを取り直す（作り直さない）",
          fb.created == before and texts2 == ["a", "b", "c"])
    check("Batch: 再取得だと stats に reused が立つ", r2.stats[0]["reused"] is True)

with tempfile.TemporaryDirectory() as tmp:
    fb = FakeBatches([_Item("a"), _Item(None, err="boom"), _Item("c")])
    r = BatchRunner(FakeClient(fb), FakeTypes, "m",
                    jobs_dir=os.path.join(tmp, "batch_jobs"), poll_interval=0)
    texts, _ = r.run("classify_y", reqs)
    check("Batch: エラー行は None で返る（呼び出し側が同期で埋める）",
          texts == ["a", None, "c"], texts)

with tempfile.TemporaryDirectory() as tmp:
    fb = FakeBatches([], state="JOB_STATE_FAILED")
    r = BatchRunner(FakeClient(fb), FakeTypes, "m",
                    jobs_dir=os.path.join(tmp, "batch_jobs"), poll_interval=0)
    texts, _ = r.run("classify_z", reqs)
    check("Batch: ジョブが失敗したら全行 None（同期に落ちる）",
          texts == [None, None, None], texts)

print("--- ⑦ Batch の料金は半額として集計される ---")

sys.path.insert(0, os.path.join(ROOT, "tools"))
from cost_saving_v1 import cost, role_of, slot_of                # noqa: E402

p = {"input": 0.10, "output": 0.40, "cache_read": 0.01}
check("Batch 行のコストは同期の半分",
      abs(cost(1_000_000, 0, 0, p, True) - cost(1_000_000, 0, 0, p) / 2) < 1e-12)
check("tag の |batch を外して役割を引ける",
      role_of("agent:household:S1r1|batch") == "household"
      and role_of("classify_stage_v5c|batch") == "classifier")
check("tag の |batch を外して枠を引ける",
      slot_of("agent:household:S1r1|batch") == "S1r1"
      and slot_of("classify|batch") == "classify")


print("--- ⑧ Codexレビューで直した3点 ---")

# (a) 同じ tag・同じ件数でも中身が違えば別ジョブとして作り直す
with tempfile.TemporaryDirectory() as tmp:
    jobs = os.path.join(tmp, "batch_jobs")
    fb = FakeBatches([_Item("a"), _Item("b"), _Item("c")])
    r = BatchRunner(FakeClient(fb), FakeTypes, "m", jobs_dir=jobs, poll_interval=0)
    r.run("same_tag", reqs)
    created_after_first = fb.created
    other = [dict(x, user_prompt="CHANGED" + x["user_prompt"]) for x in reqs]
    r2 = BatchRunner(FakeClient(fb), FakeTypes, "m", jobs_dir=jobs, poll_interval=0)
    r2.run("same_tag", other)
    check("Batch: 同じtag・同じ件数でも中身が違えば新しいジョブを作る",
          fb.created == created_after_first + 1 and r2.stats[0]["reused"] is False)
    fp_keys = ("system_prompt", "user_prompt", "temperature", "max_tokens",
               "thinking_budget", "schema")
    base_fp = request_fingerprint("m", reqs)
    diffs = []
    for k, v in (("system_prompt", "X"), ("user_prompt", "Y"), ("temperature", 0.9),
                 ("max_tokens", 101), ("thinking_budget", 5),
                 ("schema", {"type": "object"})):
        changed = [dict(reqs[0], **{k: v})] + reqs[1:]
        diffs.append(request_fingerprint("m", changed) != base_fp)
    check("Batch: 指紋は system/user/温度/上限/thinking/schema のどれが変わっても変わる",
          all(diffs), diffs)
    check("Batch: model が変われば指紋も変わる",
          request_fingerprint("other-model", reqs) != base_fp)
    check("Batch: 中身が同じなら指紋は同じ",
          request_fingerprint("m", [dict(x) for x in reqs]) == base_fp)

# (b) 空文字の応答は「取れなかった行」として扱う
with tempfile.TemporaryDirectory() as tmp:
    fb = FakeBatches([_Item("a"), _Item("   "), _Item("c")])
    r = BatchRunner(FakeClient(fb), FakeTypes, "m",
                    jobs_dir=os.path.join(tmp, "batch_jobs"), poll_interval=0)
    texts, _ = r.run("empty", reqs)
    check("Batch: 空文字の応答は None（同期で埋め直す対象）",
          texts == ["a", None, "c"], texts)

# (c) Batch を使わない設定では、分類器は従来どおり1件ずつ順番に呼ぶ
class _FakeGemini(GeminiClient):
    def __init__(self):           # API キー無しで generate_many だけ試す
        self.temperature = 0.0
        self.max_tokens = 100
        self.thinking_budget = None
        self.usage = UsageMeter()
        self.batch_kinds = set()
        self.backend = "genai"
        self.parallel_workers = 8
        self.batch_fallback_calls = 0
        self._batch_runner = None
        self.order = []

    def generate(self, system_prompt, user_prompt, schema=None, temperature=None,
                 max_tokens=None, tag="agent"):
        self.order.append(user_prompt)
        return "{}"


g = _FakeGemini()
out = g.generate_many([{"system_prompt": "S", "user_prompt": f"u{i}", "schema": None}
                       for i in range(6)], tag="classify", kind="classify")
check("Batch off のとき generate_many は逐次・投入順に呼ぶ（従来と同じ）",
      g.order == [f"u{i}" for i in range(6)] and out == ["{}"] * 6, g.order)

print("--- ⑨ 場面プロンプトの全条件で legacy 不変・stable_first は並べ替えのみ ---")

NL = chr(10)
combos = [(o, p_, d, a) for o in (True, False) for p_ in (True, False)
          for d in (0, 2) for a in (0, 1)]
ok_leg = ok_perm = True
for owns, pub, dleft, aleft in combos:
    kw = dict(owns_parcel=owns, can_publish=pub, directs_left=dleft, articles_left=aleft)
    d0 = build_scene_prompt_v5(*scene_args, **kw)
    dl = build_scene_prompt_v5(*scene_args, prompt_order=PROMPT_ORDER_LEGACY, **kw)
    ds = build_scene_prompt_v5(*scene_args, prompt_order=PROMPT_ORDER_STABLE_FIRST, **kw)
    ok_leg = ok_leg and (d0 == dl)
    ok_perm = ok_perm and (sorted(x for x in ds.split(NL) if x)
                           == sorted(x for x in dl.split(NL) if x))
check(f"scene: {len(combos)}通りの条件すべてで 既定＝legacy", ok_leg)
check(f"scene: {len(combos)}通りの条件すべてで stable_first は並べ替えのみ", ok_perm)

print(f"RESULT: {PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
