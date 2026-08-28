#!/usr/bin/env python
"""分類器を「同期」と「Batch API」の両方に流し、結果が一致するかを実測する。

    python tools/verify_batch_classify.py <run_dir> [--rows 50]

既存ランの発話・内心・記事から先頭 N 行を取り、同じ行を
  ① 従来どおりの同期コール
  ② Gemini Batch API（半額）
の両方で分類して、判定の一致率と往復時間と usage を出す。
**プロンプト・スキーマ・温度・出力上限は同一**（src/kpi.py の同じ関数を通す）。

実APIを叩く＝課金が走る。既定 50 行なら 1分類器あたり2チャンク程度。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from run import load_dotenv                                       # noqa: E402
from src.kpi import (classify_occupation, classify_stage_v5c,     # noqa: E402
                     classify_utterances)
from src.llm_client_factory import GeminiClient, UsageMeter       # noqa: E402

FIELDS = {
    "classify": ("frame", "about_acquisition"),
    "classify_occupation": ("links_multiple", "intent"),
    "classify_stage_v5c": ("deal", "area", "same_buyer", "admin"),
}


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with io.open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rows_of(run_dir, limit):
    d = run_dir if os.path.isabs(run_dir) else os.path.join(ROOT, "simulations", run_dir)
    rows = ([{"kind": "utterance", **u} for u in read_jsonl(os.path.join(d, "utterances_v5.jsonl"))]
            + [{"kind": "thought", **t} for t in read_jsonl(os.path.join(d, "thoughts_all.jsonl"))]
            + [{"kind": "article", **a} for a in read_jsonl(os.path.join(d, "articles_v5.jsonl"))])
    rows = [r for r in rows if str(r.get("text", "")).strip()]
    return rows[:limit]


def client(usage, batch, jobs_dir):
    return GeminiClient(model="gemini-2.5-flash-lite", temperature=0.0,
                        max_tokens=1800, usage=usage,
                        batch_kinds=["classify"] if batch else [],
                        batch_poll_interval=10, batch_timeout_sec=7200,
                        jobs_dir=jobs_dir)


def agree(a, b, keys):
    same = 0
    for x, y in zip(a, b):
        if all(x.get(k) == y.get(k) for k in keys):
            same += 1
    return same, len(a)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--rows", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--out", default=os.path.join(ROOT, "_scratch",
                                                  "batch_agreement.json"))
    args = ap.parse_args(argv)
    if sys.stdout is not None:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    load_dotenv()

    rows = rows_of(args.run_dir, args.rows)
    print(f"rows={len(rows)}")
    jobs_dir = os.path.join(ROOT, "_scratch", "batch_verify")
    u_sync, u_batch = UsageMeter(), UsageMeter()
    c_sync = client(u_sync, False, jobs_dir)
    c_batch = client(u_batch, True, jobs_dir)

    report = {"run_dir": args.run_dir, "rows": len(rows), "passes": {}}
    for name, fn in (("classify", classify_utterances),
                     ("classify_occupation", classify_occupation),
                     ("classify_stage_v5c", classify_stage_v5c)):
        t0 = time.time()
        out_sync = fn(c_sync, list(rows), batch=args.batch_size)
        t_sync = time.time() - t0
        t0 = time.time()
        out_batch = fn(c_batch, list(rows), batch=args.batch_size)
        t_batch = time.time() - t0
        same, total = agree(out_sync, out_batch, FIELDS[name])
        report["passes"][name] = {
            "rows": total, "agree": same,
            "agree_pct": round(100 * same / max(1, total), 1),
            "sync_sec": round(t_sync, 1), "batch_sec": round(t_batch, 1),
        }
        print(f"{name}: 一致 {same}/{total} = {100 * same / max(1, total):.1f}%"
              f"  同期 {t_sync:.0f}s / Batch {t_batch:.0f}s")

    report["batch_jobs"] = c_batch.batch_stats
    report["batch_fallback_calls"] = c_batch.batch_fallback_calls
    report["usage_sync"] = u_sync.as_dict()
    report["usage_batch"] = u_batch.as_dict()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with io.open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("written", args.out)
    print("batch jobs:", json.dumps(c_batch.batch_stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
