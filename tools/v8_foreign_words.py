#!/usr/bin/env python
"""「海外」系の語が、断りの一言・出品の理由・住民の発話にどれだけ出るかを数える。

  python tools/v8_foreign_words.py simulations/<run_dir> [--label v8c] [--samples 5]

**数えるだけ**（判定・分類はしない）。CEO が「海外という自己紹介が効いたか」を
判断するための材料を出す道具で、世界には一切触らない（読み取り専用）。
既存の集計ツール（tools/v8c_*.py）とは独立の新規ファイル。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

# 走行前に凍結した語彙（結果を見てから足さない）
FOREIGN_WORDS = ["海外", "外国", "外資", "国外", "よその国", "他国", "異国",
                 "グローバル", "インバウンド", "外資系"]


def _load(run_dir: str, name: str) -> Any:
    path = os.path.join(run_dir, name)
    if not os.path.exists(path):
        path = os.path.join(run_dir, "checkpoint", name)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _hit(text: str) -> bool:
    t = str(text or "")
    return any(w in t for w in FOREIGN_WORDS)


def count(rows: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
    written = [r for r in rows if str(r.get(field, "") or "").strip()]
    hits = [r for r in written if _hit(r.get(field, ""))]
    return {"written": len(written), "with_foreign_word": len(hits),
            "ratio": round(len(hits) / len(written), 4) if written else 0.0,
            "rows": hits}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--label", default="")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    decisions = _load(args.run_dir, "decisions.json")
    utterances = _load(args.run_dir, "utterances.json")
    offers = _load(args.run_dir, "offers.json")

    sell_no = [d for d in decisions if d.get("sell") == "売らない"]
    out = {
        "run_dir": args.run_dir,
        "label": args.label,
        "words": FOREIGN_WORDS,
        "sell_reason_all": count(decisions, "sell_reason"),
        "sell_reason_declined": count(sell_no, "sell_reason"),
        "listing_reason": count(decisions, "listing_reason"),
        "utterances": count(utterances, "text"),
        "acquirer_offer_text": count(offers, "text"),
    }
    for key in ("sell_reason_all", "sell_reason_declined", "listing_reason",
                "utterances", "acquirer_offer_text"):
        rows = out[key].pop("rows")
        out[key]["samples"] = [
            {"step": r.get("step"), "who": r.get("name") or r.get("from")
             or r.get("to"),
             "text": (r.get("sell_reason") or r.get("listing_reason")
                      or r.get("text"))}
            for r in rows[: args.samples]]
    path = args.out_json or os.path.join(args.run_dir, "foreign_words.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    label = f"[{args.label}] " if args.label else ""
    for key in ("sell_reason_declined", "listing_reason", "utterances"):
        c = out[key]
        print(f"{label}{key}: {c['with_foreign_word']}/{c['written']} "
              f"({c['ratio']*100:.1f}%)")
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
