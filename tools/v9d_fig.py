#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v9d の曲線図（決定論・API 不使用）。v9 単独と同じ形で、前の版は重ねない。

  python tools/v9d_fig.py --run simulations/<v9d_run> \
      --out docs/submission/fig_v9d_curve.png

上の段＝X社の所有権が及んだ区画の数（土地だけ／建物だけ／両方の内訳を積み上げ）と、
（前の版は重ねない＝v9d 単独の図）。
下の段＝町を出た人の数と、町にいない所有者の数の推移。
数字は summary.json / monthly.json をそのまま読むだけで、平滑化も加工もしない。
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402


def _font() -> str:
    names = {f.name for f in font_manager.fontManager.ttflist}
    for c in ("Meiryo", "Yu Gothic", "Noto Sans JP", "MS Gothic"):
        if c in names:
            return c
    return "DejaVu Sans"


def _read(run_dir: str):
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as f:
        s = json.load(f)
    with open(os.path.join(run_dir, "monthly.json"), encoding="utf-8") as f:
        m = json.load(f)
    return s, m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="docs/submission/fig_v9d_curve.png")
    args = ap.parse_args()

    s, m = _read(args.run)
    months = [0] + [x["step"] for x in m]
    land_only = [0] + [x["land_cum"] - x["both_cum"] for x in m]
    bld_only = [0] + [x["building_cum"] - x["both_cum"] for x in m]
    both = [0] + [x["both_cum"] for x in m]
    left = [0] + [x["left_cum"] for x in m]
    absentee = [x["absentee_owners"] for x in m]
    absentee = [absentee[0]] + absentee

    plt.rcParams["font.family"] = _font()
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.2, 7.4), dpi=160,
                                  gridspec_kw={"height_ratios": [3, 2]},
                                  sharex=True)

    ax.stackplot(months, both, land_only, bld_only,
                 colors=["#1a1a1a", "#8c7b5a", "#c9b28a"],
                 labels=["土地と建物の両方がX社のもの",
                         "土地だけがX社のもの（建物は持ち主のまま）",
                         "建物だけがX社のもの（土地は持ち主のまま）"])
    total = int(s.get("parcels_total", 44))
    ax.set_ylabel(f"X社の所有権が及んだ区画の数（全{total}区画）")
    ax.set_title(
        f"A市の36か月：X社の所有権が及んだ区画（{s.get('acquired_parcels')}区画／全{total}区画）\n"
        f"町を出た人 {s.get('left_agents')}人・提示のべ {s.get('offers_total')}件"
        f"（うち成約 {s.get('offers_accepted')}件）",
        fontsize=11)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)

    ax2.plot(months, left, color="#1a1a1a", lw=2.2, label="町を出た人（累計）")
    ax2.plot(months, absentee, color="#8c7b5a", lw=2.0, ls="-.",
             label="町にいない所有者の数")
    ax2.set_xlabel("月")
    ax2.set_ylabel("人")
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax2.grid(alpha=0.25)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
