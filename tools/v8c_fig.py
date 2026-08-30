#!/usr/bin/env python
"""v8c の取得曲線の図（決定論・API 不使用）。

  python tools/v8c_fig.py --chat simulations/<v8c_chat> --nochat simulations/<v8c_nochat> \
      --out docs/submission/fig_v8c_chat_vs_nochat.png

出すもの＝月ごとの「X社の名義になった区画の割合」を、会話あり／会話なしの2本で描く。
比較用に直前版（v8b）を薄い灰色で重ねられる（`--v8b`）。
数字は summary.json / monthly.json をそのまま読むだけで、加工・平滑化はしない。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402


def _pick_font() -> str:
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Meiryo", "Yu Gothic", "Noto Sans JP", "MS Gothic"):
        if cand in names:
            return cand
    return "DejaVu Sans"


def _series(run_dir: str) -> Dict[str, Any]:
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as f:
        s = json.load(f)
    with open(os.path.join(run_dir, "monthly.json"), encoding="utf-8") as f:
        monthly = json.load(f)
    denom = max(1, int(s["sellable_parcels"]))
    return {
        "months": [0] + [m["step"] for m in monthly],
        "share": [0.0] + [m["parcels_cum"] / denom * 100 for m in monthly],
        "label": s.get("run_name", os.path.basename(run_dir)),
        "summary": s,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat", required=True)
    ap.add_argument("--nochat", default=None)
    ap.add_argument("--v8b", default=None)
    ap.add_argument("--out", default="docs/submission/fig_v8c_chat_vs_nochat.png")
    args = ap.parse_args()

    plt.rcParams["font.family"] = _pick_font()
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.0, 7.0), dpi=160,
                                  gridspec_kw={"height_ratios": [3, 2]},
                                  sharex=True)

    if args.v8b:
        b = _series(args.v8b)
        ax.plot(b["months"], b["share"], color="#bbbbbb", lw=1.6, ls="--",
                label="前の版（比較用・同じ町）", zorder=1)
    chat = _series(args.chat)
    ax.plot(chat["months"], chat["share"], color="#1a1a1a", lw=2.6,
            label="今回の世界", zorder=3)
    ax.scatter([chat["months"][-1]], [chat["share"][-1]], color="#1a1a1a",
               s=28, zorder=4)
    if args.nochat:
        nc = _series(args.nochat)
        ax.plot(nc["months"], nc["share"], color="#0e9f6e", lw=2.6,
                label="会話なし（同じ町・同じ seed）", zorder=2)
        ax.scatter([nc["months"][-1]], [nc["share"][-1]], color="#0e9f6e",
                   s=28, zorder=4)


    ax.set_ylabel("買い手の名義になった不動産の割合（%）")
    ax.set_title("町の不動産が買い手の名義に移っていく速さ（今回は36か月ずっと0件）",
                 fontsize=13, pad=12)
    ax.set_xlim(0, max(chat["months"]))
    ymax = max([chat["share"][-1]] +
               ([_series(args.nochat)["share"][-1]] if args.nochat else []) +
               ([_series(args.v8b)["share"][-1]] if args.v8b else []))
    ax.set_ylim(0, max(10.0, ymax * 1.35))
    ax.grid(True, color="#e6e6e6", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper left")

    # 下段＝月ごとの「買い手が出した提示の件数」と「売りに出した人の数」
    with open(os.path.join(args.chat, "monthly.json"), encoding="utf-8") as f:
        monthly = json.load(f)
    months = [m["step"] for m in monthly]
    ax2.bar(months, [m["offers_sent"] for m in monthly], color="#c8c8c8",
            label="買い手が出した提示の件数")
    ax2.plot(months, [m["listed_this_month"] for m in monthly], color="#0e9f6e",
             lw=2.0, label="売りに出した人の数")
    ax2.set_xlabel("月")
    ax2.set_ylabel("件 / 人")
    ax2.grid(True, axis="y", color="#e6e6e6", lw=0.8)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax2.legend(frameon=False, loc="upper right", ncol=2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
