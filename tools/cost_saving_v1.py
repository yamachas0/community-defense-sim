#!/usr/bin/env python
"""節約設定の効き目を、ランの summary.json から実測して並べる（報告用）。

    python tools/cost_saving_v1.py <run_dir> [<run_dir> ...]

数値の出所は各 run_dir の `summary.json` の `usage.by_tag`（API が実際に返した
トークン数）と `configs/config_field_v5c.yaml` の `cost.price_table` だけ。
手打ちの数字は使わない。

**Batch API の行は tag に `|batch` が付いている**（src/llm_batch.py が付ける）。
Gemini Batch は同期の 50% なので、その行だけ 0.5 を掛けて合算する。
"""

from __future__ import annotations

import collections
import io
import json
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCH_DISCOUNT = 0.5

ROLE_JA = {
    "household": "住民", "business": "事業者", "broker": "仲介",
    "municipality": "行政", "media": "記者", "classifier": "分類器（事後）",
}


def price_table(model: str):
    with open(os.path.join(ROOT, "configs", "config_field_v5c.yaml"),
              encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["cost"]["price_table"][model]


def cost(inp: float, cached: float, out: float, p, batched: bool = False) -> float:
    c = ((inp - cached) / 1e6 * p["input"]
         + cached / 1e6 * p["cache_read"]
         + out / 1e6 * p["output"])
    return c * (BATCH_DISCOUNT if batched else 1.0)


def role_of(tag: str) -> str:
    base = tag.split("|")[0]
    if base.startswith("classify"):
        return "classifier"
    parts = base.split(":")
    return parts[1] if len(parts) > 2 else base


def slot_of(tag: str) -> str:
    base = tag.split("|")[0]
    if base.startswith("classify"):
        return base
    parts = base.split(":")
    return parts[2] if len(parts) > 2 else base


def summarize(run_dir: str):
    path = run_dir
    if not os.path.isabs(path):
        cand = os.path.join(ROOT, "simulations", run_dir)
        path = cand if os.path.isdir(cand) else os.path.join(ROOT, run_dir)
    with open(os.path.join(path, "summary.json"), encoding="utf-8") as f:
        s = json.load(f)
    p = price_table(s["model"])
    u = s["usage"]
    total = 0.0
    batched_calls = 0
    by_role = collections.defaultdict(lambda: [0, 0, 0, 0.0])
    by_slot = collections.defaultdict(lambda: [0, 0, 0, 0.0, 0])
    for tag, v in u["by_tag"].items():
        batched = "|batch" in tag
        c = cost(v["input"], v.get("cached", 0), v["output"], p, batched)
        total += c
        if batched:
            batched_calls += v["calls"]
        a = by_role[role_of(tag)]
        a[0] += v["calls"]; a[1] += v["input"]; a[2] += v.get("cached", 0); a[3] += c
        b = by_slot[slot_of(tag)]
        b[0] += v["calls"]; b[1] += v["input"]; b[2] += v.get("cached", 0)
        b[3] += c; b[4] += v["output"]
    return {"dir": os.path.basename(path.rstrip(os.sep)), "summary": s, "price": p,
            "total": total, "by_role": by_role, "by_slot": by_slot,
            "batched_calls": batched_calls}


def show(r) -> None:
    s, u = r["summary"], r["summary"]["usage"]
    sv = s.get("saving", {})
    steps = s["steps"]
    cached_pct = 100 * u["cached_tokens"] / max(1, u["input_tokens"])
    print(f"=== {r['dir']}  {steps}か月  {s['model']}")
    print(f"    設定: prompt_order={sv.get('prompt_order', 'legacy')} "
          f"cache={sv.get('enable_cache')} batch={sv.get('batch_kinds', [])} "
          f"agent_max_tokens={sv.get('agent_max_tokens')}")
    print(f"    calls {u['calls']}（うちBatch {r['batched_calls']}）"
          f"  input {u['input_tokens']:,}（キャッシュ読み {cached_pct:.1f}%）"
          f"  output {u['output_tokens']:,}")
    print(f"    ${r['total']:.4f}   1か月あたり ${r['total'] / max(1, steps):.4f}")
    print(f"    打切り max_token_finishes={s.get('max_token_finishes')} "
          f"truncated={s.get('truncated_responses')} "
          f"cache_created={sv.get('cache_created')} "
          f"cache_failed={sv.get('cache_failed')} "
          f"batch_fallback={sv.get('batch_fallback_calls')}")
    jobs = sv.get("batch_jobs") or []
    if jobs:
        waits = [j.get("elapsed_sec", 0) for j in jobs]
        print(f"    Batch ジョブ {len(jobs)}本  往復 中央値 {sorted(waits)[len(waits) // 2]:.0f}s"
              f"  最短 {min(waits):.0f}s  最長 {max(waits):.0f}s"
              f"  合計 {sum(waits):.0f}s")
    print("    --- 呼び出し種別ごと ---")
    for k, a in sorted(r["by_role"].items(), key=lambda kv: -kv[1][3]):
        print(f"    {ROLE_JA.get(k, k):<14} calls={a[0]:>5} in={a[1]:>9,}"
              f" cached={100 * a[2] / max(1, a[1]):>4.1f}%"
              f"  ${a[3]:.4f}  ({100 * a[3] / max(1e-9, r['total']):.1f}%)")


def main(argv) -> int:
    if sys.stdout is not None:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    if not argv:
        print(__doc__)
        return 2
    rows = [summarize(d) for d in argv]
    for r in rows:
        show(r)
    if len(rows) >= 2:
        base, econ = rows[0], rows[-1]
        b = base["total"] / max(1, base["summary"]["steps"])
        e = econ["total"] / max(1, econ["summary"]["steps"])
        print(f"=== 比較（1か月あたり）  従来 ${b:.4f} → 節約 ${e:.4f}"
              f"  ＝ {100 * e / b:.0f}%（1/{b / e:.2f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
