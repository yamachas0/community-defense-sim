#!/usr/bin/env python
"""既に走り終わった run を、v5e の新しい赤（自衛の具体的な行動）で読み直す。

    python tools/reclassify_v5e.py <run_dir> --dry-run
    python tools/reclassify_v5e.py <run_dir> --provider google

`utterances_v5.jsonl` / `thoughts_all.jsonl` / `articles_v5.jsonl` から
**事後の occ_rows と同じ組み立て**で行を作り、`classify_stage_v5e` を回して
`stage_labels_v5e.jsonl` を書く（これは観測であって世界には戻らない）。

シミュ本体は動かさない。既存の `stage_labels_v5c.jsonl` は上書きしない。
`--dry-run` は API を一切叩かず、行数・チャンク数・概算費用だけを出す。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from src.kpi import (STAGE_SYSTEM_V5E, build_stage_prompt_v5e,  # noqa: E402
                     classify_stage_v5e)
from src.llm_client_factory import UsageMeter, create_llm_client  # noqa: E402
from estimate_cost import estimate_tokens  # noqa: E402

# 出力の想定トークン（1行あたり）。results の1要素は id と5つのフラグだけなので
# 実測でも 25 前後に収まる。概算にしか使わない。
OUT_TOKENS_PER_ROW = 25


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    with io.open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_rows(run_dir):
    """事後の occ_rows と同じ組み立て（utterance → thought → article の順）。"""
    return ([{"kind": "utterance", **u}
             for u in _read_jsonl(os.path.join(run_dir, "utterances_v5.jsonl"))]
            + [{"kind": "thought", **t}
               for t in _read_jsonl(os.path.join(run_dir, "thoughts_all.jsonl"))]
            + [{"kind": "article", **a}
               for a in _read_jsonl(os.path.join(run_dir, "articles_v5.jsonl"))])


def _price(cfg, model):
    table = (cfg.get("cost") or {}).get("price_table") or {}
    return table.get(model)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--provider", default=None, choices=["google", "mock"])
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--out", default="stage_labels_v5e.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if getattr(sys.stdout, "reconfigure", None):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    run_dir = args.run_dir
    with io.open(os.path.join(run_dir, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    llm = dict(cfg.get("llm") or {})
    if args.provider:
        llm["provider"] = args.provider
    model = llm.get("model", "?")

    rows = build_rows(run_dir)
    chunks = [rows[i:i + args.batch] for i in range(0, len(rows), args.batch)]
    in_tokens = sum(estimate_tokens(STAGE_SYSTEM_V5E)
                    + estimate_tokens(build_stage_prompt_v5e(c)) for c in chunks)
    out_tokens = len(rows) * OUT_TOKENS_PER_ROW
    p = _price(cfg, model)
    est = (None if not p else
           in_tokens / 1e6 * p["input"] + out_tokens / 1e6 * p["output"])

    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"[run] {run_dir}")
    print(f"[model] {model} / provider {llm.get('provider')}")
    print(f"[rows] {len(rows)}  " + " ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    print(f"[chunks] {len(chunks)} (batch {args.batch})")
    print(f"[est tokens] input {in_tokens:.0f} / output {out_tokens}")
    print("[est cost] " + (f"${est:.4f}" if est is not None
                           else f"価格表に {model} が無い"))
    if args.dry_run:
        print("[dry-run] API は叩いていない。ファイルも書いていない。")
        return 0

    out_path = os.path.join(run_dir, args.out)
    if os.path.basename(out_path) == "stage_labels_v5c.jsonl":
        raise SystemExit("stage_labels_v5c.jsonl は上書きしない")

    usage = UsageMeter()
    client = create_llm_client(llm, usage=usage)
    labels = classify_stage_v5e(client, rows, batch=args.batch)
    with io.open(out_path, "w", encoding="utf-8") as f:
        for r in labels:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    u = usage.as_dict()
    actual = (None if not p else
              ((u["input_tokens"] - u["cached_tokens"]) / 1e6 * p["input"]
               + u["cached_tokens"] / 1e6 * p["cache_read"]
               + u["output_tokens"] / 1e6 * p["output"]))
    cost_path = os.path.join(run_dir, "reclassify_v5e_cost.json")
    with io.open(cost_path, "w", encoding="utf-8") as f:
        json.dump({"run_dir": run_dir, "model": model,
                   "provider": llm.get("provider"), "rows": len(rows),
                   "chunks": len(chunks), "batch": args.batch,
                   "usage": u, "cost_usd": actual,
                   "estimated_cost_usd": est,
                   "unknown_rows": len([r for r in labels
                                        if not r.get("classified")])},
                  f, ensure_ascii=False, indent=2)
    print(f"[out] {out_path}  ({len(labels)} 行)")
    print(f"[out] {cost_path}  実費 "
          + (f"${actual:.4f}" if actual is not None else "不明"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
