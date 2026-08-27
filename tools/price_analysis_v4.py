#!/usr/bin/env python
"""v4ランの価格を分析する（買付価格 / 評価額の比の分布、桁外れ価格の月、受諾者の実感）。

    python tools/price_analysis_v4.py --run simulations/<run_dir> [--outlier 100]

ここでは解釈も評価もしない。台帳と記録に実在する数値を並べるだけ。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any, Dict, List


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def analyse(run_dir: str, outlier_ratio: float) -> Dict[str, Any]:
    ledger = _read_jsonl(os.path.join(run_dir, "ledger.jsonl"))
    events = _read_jsonl(os.path.join(run_dir, "events.jsonl"))
    feelings = _read_jsonl(os.path.join(run_dir, "feelings.jsonl"))

    # 区画の評価額は config から決定論的に組み直す
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import yaml  # noqa: E402
    from src.viz import _rebuild_world  # noqa: E402
    with open(os.path.join(run_dir, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(run_dir, "personas.yaml"), encoding="utf-8") as f:
        personas = yaml.safe_load(f)
    _, parcels, _ = _rebuild_world(cfg, personas)
    value = {p.pid: p.assessed_value for p in parcels}

    offers = [r for r in ledger if r.get("kind") == "offer"]
    ratio_of = {}
    rows = []
    for r in offers:
        base = value.get(r.get("parcel_id"), 0) or 1
        ratio = r["price"] / base
        ratio_of[r["offer_id"]] = ratio
        rows.append({"step": r["step"], "offer_id": r["offer_id"],
                     "parcel": r["parcel_id"], "price": r["price"],
                     "assessed": base, "ratio": ratio,
                     "under_name": r.get("under_name", "")})

    accepted_ids = {r.get("offer_id") for r in ledger if r.get("kind") == "transfer"}
    rejected_ids = {r.get("offer_id") for r in ledger if r.get("kind") == "reject"}
    countered_ids = {r.get("offer_id") for r in ledger if r.get("kind") == "counter"}

    def stats(values: List[float]) -> Dict[str, Any]:
        if not values:
            return {"n": 0}
        return {"n": len(values), "min": round(min(values), 2),
                "median": round(statistics.median(values), 2),
                "max": round(max(values), 2)}

    accepted = [row for row in rows if row["offer_id"] in accepted_ids]
    rejected = [row for row in rows if row["offer_id"] in rejected_ids]
    countered = [row for row in rows if row["offer_id"] in countered_ids]

    by_step: Dict[int, List[float]] = {}
    for row in rows:
        by_step.setdefault(row["step"], []).append(row["ratio"])

    outliers = [row for row in rows if row["ratio"] >= outlier_ratio]
    memo_by_step = {e["step"]: e.get("memo", "") for e in events
                    if e.get("role") == "acquirer"}
    accept_feelings = []
    for row in accepted:
        seller = next((r.get("seller") for r in ledger
                       if r.get("kind") == "transfer"
                       and r.get("offer_id") == row["offer_id"]), "")
        text = next((f["text"] for f in feelings
                     if f["step"] == row["step"] and f["from"] == seller), "")
        accept_feelings.append({"step": row["step"], "offer_id": row["offer_id"],
                                "parcel": row["parcel"], "price": row["price"],
                                "ratio": round(row["ratio"], 2), "seller": seller,
                                "feeling": text})

    counter_rows = []
    for r in ledger:
        if r.get("kind") != "counter":
            continue
        base = value.get(r.get("parcel_id"), 0) or 1
        counter_rows.append({"step": r["step"], "offer_id": r["offer_id"],
                             "counter_price": r["price"],
                             "counter_ratio": round(r["price"] / base, 2),
                             "offer_ratio": round(ratio_of.get(r["offer_id"], 0), 2)})

    return {
        "run_dir": os.path.basename(run_dir.rstrip("/\\")),
        "offers": len(rows),
        "ratio_all": stats([r["ratio"] for r in rows]),
        "ratio_accepted": stats([r["ratio"] for r in accepted]),
        "ratio_rejected": stats([r["ratio"] for r in rejected]),
        "ratio_countered": stats([r["ratio"] for r in countered]),
        "ratio_by_step": {s: stats(v) for s, v in sorted(by_step.items())},
        "outlier_threshold_ratio": outlier_ratio,
        "outliers": sorted(outliers, key=lambda r: -r["ratio"]),
        "outlier_steps_memo": {row["step"]: memo_by_step.get(row["step"], "")
                               for row in outliers},
        "accepted_with_seller_feeling": accept_feelings,
        "counters": counter_rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, nargs="+")
    ap.add_argument("--outlier", type=float, default=100.0,
                    help="評価額の何倍以上を桁外れとして抜き出すか")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps([analyse(r, args.outlier) for r in args.run],
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
