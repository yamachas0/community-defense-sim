#!/usr/bin/env python
"""ランフォルダの生ログからレポートHTMLを再生成する（再ラン不要・API課金ゼロ）。

    python tools/render_report.py --run simulations/2026-08-14_2249_12_mvp_v1_main
    python tools/render_report.py --all       # simulations/ 配下すべて

生ログ（events.jsonl / ledger.jsonl / kpi.jsonl / owner_frames.json / config.yaml /
personas.yaml / summary.json）は読むだけで、一切書き換えない。
書き換わるのは <run_dir>/<run_dir>.html のみ。
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.viz import render_report_from_dir  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_DIR = os.path.join(ROOT, "simulations")

REQUIRED = ["config.yaml", "personas.yaml", "summary.json", "events.jsonl",
            "ledger.jsonl", "kpi.jsonl", "owner_frames.json"]


def usable(d: str) -> bool:
    return all(os.path.exists(os.path.join(d, f)) for f in REQUIRED)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run_dir のパス")
    ap.add_argument("--all", action="store_true", help="simulations/ 配下すべて")
    args = ap.parse_args()

    if not args.run and not args.all:
        ap.error("--run か --all のどちらかを指定しろ")

    targets = []
    if args.all:
        for name in sorted(os.listdir(SIM_DIR)):
            d = os.path.join(SIM_DIR, name)
            if os.path.isdir(d) and usable(d):
                targets.append(d)
    else:
        d = args.run if os.path.isabs(args.run) else os.path.join(ROOT, args.run)
        if not usable(d):
            missing = [f for f in REQUIRED if not os.path.exists(os.path.join(d, f))]
            print(f"生ログが足りない: {d}\n  不足: {missing}")
            return 1
        targets.append(d)

    for d in targets:
        before = None
        folder = os.path.basename(os.path.normpath(d))
        out = os.path.join(d, f"{folder}.html")
        if os.path.exists(out):
            before = os.path.getsize(out)
        path = render_report_from_dir(d)
        after = os.path.getsize(path)
        delta = f"{before:,} -> {after:,}" if before else f"new {after:,}"
        print(f"[ok] {folder}  ({delta} bytes)")
    print(f"\n{len(targets)} 件のレポートを再生成した（生ログは無変更）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
