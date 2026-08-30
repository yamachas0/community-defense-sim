#!/usr/bin/env python
"""v9e の町の地図（平面図と断面図）。決定論・API を一切使わない。

  # 開始時（第1月）＝名簿だけから描ける
  python tools/v9e_map.py --personas configs/personas_v9.yaml --month 1 \
      --out-dir docs/submission

  # 走行のあと（第12・24・36月）
  python tools/v9e_map.py --run simulations/<v9e_run_dir> --months 12,24,36 \
      --out-dir docs/submission

出すもの（施主指示 2026-08-30 11:06）:
  ① 平面 `fig_v9e_map_plan_mNN.png`   … 44区画の配置図（隣近所と同じ格子）。
     土地の色＝土地の持ち主の種類、建物の色＝建物の持ち主の種類、借りている人は印。
  ② 断面 `fig_v9e_map_section_mNN.png` … 権利の重なりを正面から見た帯の図。
     下＝土地、上＝建物、いちばん上＝借りている人の印。44区画を3段に折る。

**数えるだけ・描くだけ**（判定も加工もしない）。凡例は素人語。
「空き家」「空き」の語は使わない（施主 2026-08-30 10:35）。
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
from matplotlib.patches import Rectangle  # noqa: E402

ACQUIRER_NAME = "X社"

# 素人語の凡例（この4つ＋「建物なし」だけ）
T_TOWN = "町にいる人"
T_ABSENT = "町にいない所有者"
T_GOV = "市（行政）"
T_X = "X社"
T_NONE = "建物なし"

COLOR = {
    T_TOWN: "#0e9f6e",
    T_ABSENT: "#e08a1e",
    T_GOV: "#7a8794",
    T_X: "#b3261e",
    T_NONE: "#dfe3e6",
}
HATCH = "///"                 # ブロックに重ねる斜線（借家・借地）
USER_MARK = "#111111"         # ●＝普段町にいる人が使っている（創発に参加）


def is_rented_building(r):
    """借家＝建物があり、その建物を借りて使っている人がいる。"""
    return bool(r.get("has_building") and r.get("tenant"))


def is_leased_land(r):
    """借地＝土地の持ち主と、その上を使っている人が違う。

    ①建物があって土地の持ち主と建物の持ち主が違う（借地上の持家）
    ②建物が無い区画を、持ち主でない人が借りて使っている（駐車場など）
    """
    if r.get("has_building"):
        return bool(r.get("building")) and r.get("building") != r.get("land")
    return bool(r.get("tenant"))


def used_by_townsfolk(r, kinds) -> bool:
    """普段町にいる人が住み、または営んでいる区画か（＝創発に参加している区画）。

    建物を借りている人／建物を持って住んでいる人／土地を借りて使っている会社が対象。
    誰も使っていない区画と、持ち主が町にいない区画には印を付けない。
    """
    in_town = (T_TOWN, T_GOV)      # 行政も町にいて創発に参加する
    t = r.get("tenant")
    if t:
        return kinds.get(t) in in_town
    if r.get("has_building") and r.get("building"):
        return kinds.get(r["building"]) in in_town
    return False


# 区画の記号（地区の頭文字＋番号・施主 2026-08-30 11:21）。走行前に凍結する。
DISTRICT_CODE = {
    "温泉丘陵地区": "Y",
    "中央駅前地区": "E",
    "北部学術・生活地区": "N",
    "湾岸観光地区": "W",
}


def parcel_codes(parcels: List[Dict[str, Any]]) -> Dict[str, str]:
    """区画名 → 記号（Y1, E1, N1, W1 …）。名簿の並び順で番号を振る（決定論）。"""
    n: Dict[str, int] = {}
    out: Dict[str, str] = {}
    for p in parcels:
        code = DISTRICT_CODE.get(str(p["district"]), "X")
        n[code] = n.get(code, 0) + 1
        out[str(p["name"])] = f"{code}{n[code]}"
    return out


def _pick_font() -> str:
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Meiryo", "Yu Gothic", "Noto Sans JP", "MS Gothic"):
        if cand in names:
            return cand
    return "DejaVu Sans"


# ---------------------------------------------------------------------------
# 帳簿の読み込み
# ---------------------------------------------------------------------------

def load_personas(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def kind_map(book: Dict[str, Any]) -> Dict[str, str]:
    """呼び名 → 種類（町にいる人／町にいない所有者／市（行政）／X社）。"""
    out: Dict[str, str] = {ACQUIRER_NAME: T_X}
    for a in book["agents"]:
        if a.get("sellable", True) is False:
            out[str(a["name"])] = T_GOV
        elif a.get("resident", True):
            out[str(a["name"])] = T_TOWN
        else:
            out[str(a["name"])] = T_ABSENT
    return out


def initial_rows(book: Dict[str, Any]) -> List[Dict[str, Any]]:
    """名簿から開始時（第1月のはじめ）の帳簿を作る。"""
    name = {str(a["id"]): str(a["name"]) for a in book["agents"]}
    rows = []
    for p in book["parcels"]:
        rows.append({
            "parcel": str(p["name"]),
            "district": str(p["district"]),
            "has_building": bool(p["building"]),
            "land": name[str(p["land"])],
            "building": name[str(p["bld_owner"])] if p["bld_owner"] else None,
            "tenant": name[str(p["tenant"])] if p["tenant"] else None,
        })
    return rows


def rows_from_run(run_dir: str, month: int) -> List[Dict[str, Any]]:
    path = os.path.join(run_dir, "ledger_by_step.json")
    if not os.path.exists(path):
        path = os.path.join(run_dir, "checkpoint", "ledger_by_step.json")
    with open(path, encoding="utf-8") as f:
        book = json.load(f)
    for row in book:
        if int(row["step"]) == int(month):
            return row["rows"]
    raise SystemExit(f"第{month}月の帳簿が run_dir に無い")


# ---------------------------------------------------------------------------
# 図
# ---------------------------------------------------------------------------

def _legend(ax, y: float, width: float) -> None:
    """凡例（素人語）。色の帯を上段に、斜線と●の説明を下段に置く。"""
    items = [(T_TOWN, COLOR[T_TOWN]), (T_ABSENT, COLOR[T_ABSENT]),
             (T_GOV, COLOR[T_GOV]), (T_X, COLOR[T_X]), (T_NONE, COLOR[T_NONE])]
    step = width / 5.0
    sw = min(0.55, step * 0.16)
    for i, (label, c) in enumerate(items):
        x = i * step
        ax.add_patch(Rectangle((x, y + 0.62), sw, 0.5, facecolor=c,
                               edgecolor="#333333", lw=0.6))
        ax.text(x + sw * 1.4, y + 0.87, label, va="center", ha="left",
                fontsize=10.5)
    ax.add_patch(Rectangle((0, y), sw, 0.5, facecolor=COLOR[T_ABSENT],
                           edgecolor="#333333", lw=0.6, hatch=HATCH))
    ax.text(sw * 1.4, y + 0.25,
            "斜線の建物＝借家がある／斜線の土地＝借地",
            va="center", ha="left", fontsize=10.5)
    x = width * 0.52
    ax.plot([x + sw * 0.5], [y + 0.25], marker="o", markersize=9,
            color=USER_MARK)
    ax.text(x + sw * 1.4, y + 0.25,
            "●＝普段町にいる人が使っている（創発に参加）",
            va="center", ha="left", fontsize=10.5)


def draw_section(rows: List[Dict[str, Any]], kinds: Dict[str, str], month: int,
                 out: str, codes: Dict[str, str], n_town: int, n_absent: int,
                 bands: int = 3) -> str:
    """断面図＝土地の上に建物、その上に借りている人の印（帯を3段に折る）。"""
    plt.rcParams["font.family"] = _pick_font()
    plt.rcParams["axes.unicode_minus"] = False
    n = len(rows)
    per = (n + bands - 1) // bands
    fig_h = 2.6 * bands + 2.3
    fig, ax = plt.subplots(figsize=(17.0, fig_h), dpi=150)
    ax.set_xlim(-0.6, per + 0.6)
    ax.set_ylim(-0.8, bands * 3.6 + 2.3)
    ax.axis("off")

    for i, r in enumerate(rows):
        band = i // per
        col = i % per
        y0 = (bands - 1 - band) * 3.6
        land_kind = kinds.get(r["land"], T_TOWN)
        # 土地（下のブロック）。借地の区画はブロックそのものに斜線を重ねる。
        ax.add_patch(Rectangle(
            (col + 0.05, y0), 0.9, 1.0, facecolor=COLOR[land_kind],
            edgecolor="#333333", lw=0.7,
            hatch=(HATCH if is_leased_land(r) else None)))
        # 建物（上のブロック）。借家がある区画はブロックそのものに斜線を重ねる。
        if r["has_building"]:
            bk = kinds.get(r["building"], T_TOWN)
            ax.add_patch(Rectangle(
                (col + 0.15, y0 + 1.05), 0.7, 1.0, facecolor=COLOR[bk],
                edgecolor="#333333", lw=0.7,
                hatch=(HATCH if is_rented_building(r) else None)))
        else:
            ax.add_patch(Rectangle((col + 0.15, y0 + 1.05), 0.7, 1.0,
                                   facecolor="white", edgecolor="#c8ced3",
                                   lw=0.7, ls=":"))
        # ●＝普段町にいる人が使っている（創発に参加）
        if used_by_townsfolk(r, kinds):
            ax.plot([col + 0.5], [y0 + 2.38], marker="o", markersize=9,
                    color=USER_MARK)
        # ラベルは記号（地区の頭文字＋番号）を横書きで大きく（施主 11:21）
        ax.text(col + 0.5, y0 - 0.22, codes.get(r["parcel"], ""), ha="center",
                va="top", fontsize=15)
    ax.text(-0.5, bands * 3.6 + 1.75,
            f"A市の不動産の権利（第{month}月）　下＝土地の持ち主／上＝建物の持ち主　"
            f"創発に参加＝{n_town}人／町にいない所有者＝{n_absent}人",
            fontsize=14, ha="left", va="center")
    _legend(ax, bands * 3.6 + 0.15, per)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def draw_plan(rows: List[Dict[str, Any]], kinds: Dict[str, str], month: int,
              grid: List[str], out: str, codes: Dict[str, str],
              n_town: int, n_absent: int, cols: int = 8) -> str:
    """平面図＝町の形（隣近所と同じ格子）に、土地の色と建物の色を置く。"""
    plt.rcParams["font.family"] = _pick_font()
    plt.rcParams["axes.unicode_minus"] = False
    by_parcel = {r["parcel"]: r for r in rows}
    n = len(grid)
    nrows = (n + cols - 1) // cols
    fig, ax = plt.subplots(figsize=(14.0, 2.2 * nrows + 2.2), dpi=150)
    ax.set_xlim(-0.3, cols + 0.3)
    ax.set_ylim(-0.5, nrows * 1.6 + 2.3)
    ax.axis("off")
    for i, parcel in enumerate(grid):
        r = by_parcel[parcel]
        c, rw = i % cols, i // cols
        x, y = c, (nrows - 1 - rw) * 1.6
        land_kind = kinds.get(r["land"], T_TOWN)
        ax.add_patch(Rectangle(
            (x + 0.03, y), 0.94, 1.15, facecolor=COLOR[land_kind],
            edgecolor="#333333", lw=0.8,
            hatch=(HATCH if is_leased_land(r) else None)))
        if r["has_building"]:
            bk = kinds.get(r["building"], T_TOWN)
            ax.add_patch(Rectangle(
                (x + 0.30, y + 0.22), 0.40, 0.55, facecolor=COLOR[bk],
                edgecolor="#1a1a1a", lw=1.0,
                hatch=(HATCH if is_rented_building(r) else None)))
        if used_by_townsfolk(r, kinds):
            ax.plot([x + 0.85], [y + 0.95], marker="o", markersize=8,
                    color=USER_MARK)
        ax.text(x + 0.10, y + 0.95, codes.get(parcel, ""), ha="left", va="top",
                fontsize=13, color="#111111")
        ax.text(x + 0.5, y - 0.06, parcel, ha="center", va="top", fontsize=7.2)
    ax.text(-0.25, nrows * 1.6 + 1.95,
            f"A市の不動産の持ち主（第{month}月）　"
            "四角＝土地／中の小さな四角＝建物　"
            f"創発に参加＝{n_town}人／町にいない所有者＝{n_absent}人",
            fontsize=13, ha="left", va="center")
    _legend(ax, nrows * 1.6 + 0.35, cols)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", default="configs/personas_v9.yaml")
    ap.add_argument("--run", default=None, help="走行の run_dir（無ければ開始時を描く）")
    ap.add_argument("--months", default="1")
    ap.add_argument("--out-dir", default="docs/submission")
    ap.add_argument("--which", default="both", choices=["both", "plan", "section"])
    args = ap.parse_args()

    book = load_personas(args.personas)
    kinds = kind_map(book)
    # 格子は v8 と同じ（世界の形を変えない）
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.field_v9 import parcel_grid_v9  # noqa: E402
    grid = parcel_grid_v9(book["parcels"])
    codes = parcel_codes(book["parcels"])

    n_town0 = sum(1 for a in book["agents"] if a.get("resident", True))
    n_absent0 = sum(1 for a in book["agents"] if not a.get("resident", True))
    monthly = None
    if args.run:
        path = os.path.join(args.run, "monthly.json")
        if not os.path.exists(path):
            path = os.path.join(args.run, "checkpoint", "monthly.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                monthly = {int(r["step"]): r for r in json.load(f)}

    os.makedirs(args.out_dir, exist_ok=True)
    for m in [int(x) for x in args.months.split(",") if x.strip()]:
        from_run = not (args.run is None or m <= 0)
        rows = (rows_from_run(args.run, m) if from_run else initial_rows(book))
        n_town, n_absent = n_town0, n_absent0
        if monthly and m in monthly:
            n_town = int(monthly[m].get("in_town", n_town0))
            n_absent = int(monthly[m].get("absentee_owners", n_absent0))
        if args.which in ("section", "both"):
            out = os.path.join(args.out_dir, f"fig_v9e_map_section_m{m:02d}.png")
            print("wrote", draw_section(rows, kinds, m, out, codes,
                                        n_town, n_absent))
        if args.which in ("plan", "both"):
            out = os.path.join(args.out_dir, f"fig_v9e_map_plan_m{m:02d}.png")
            print("wrote", draw_plan(rows, kinds, m, grid, out, codes,
                                     n_town, n_absent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
