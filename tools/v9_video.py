#!/usr/bin/env python
"""走行（1シミュ）を動画にする／複数の動画を1本に連結する。
mp4・1920x1080・H.264・30fps。

  # ① 1シミュぶんの本編（約20秒・冒頭に版ラベルのカード2秒）
  python tools/v9_video.py --run simulations/<run_dir> \
      --label "v9 土地と建物を分ける町" --out docs/submission/body_v9.mp4

  # ② 冒頭タイトル演出（約10秒）＋本編…＋末尾の比較の一枚 を1本に連結
  python tools/v9_video.py --concat docs/submission/body_v9.mp4 --title \
      --out docs/submission/video_v9.mp4

数えるだけ・描くだけ。判定も加工もしない。数字は全部 run_dir の JSON から取る。
色と記号の規則は tools/v9_map.py をそのまま読み込んで使う（スライドの図と一致）。

画面（本編）:
  最上部  ＝「第 N 月」特大＋「3年のうち X年Yか月」
  その下  ＝36か月のプログレスバー（現在月を強調・所有権が動いた月に印）
  中段左  ＝平面図（町の形）／中段右＝断面図（権利の重なり）
  凡例    ＝中段の下に1行
  最下段  ＝その月の出来事（1〜3行）と、X社の所有権が及んだ区画の累計
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)
import v9_map as M  # noqa: E402

# スライド（quiet-acquisition-pages/slides.html）の配色
BG = "#0f1115"
PANEL = "#161a21"
PANEL_EDGE = "#262c36"
CARD = "#f7f8fa"
FG = "#e9ecf1"
MUTED = "#98a1ad"
DIM = "#5b6673"
ACCENT = "#34d399"

TITLE_MAIN_1 = "外的不動産投資への"
TITLE_MAIN_2 = "コミュニティ自衛"
TITLE_Q_1 = "コミュニティの創発は、"
TITLE_Q_2 = "外的な不動産投資を抑制できるのか？"

W, H = 1920, 1080
DPI = 120
FPS = 30
SEC_MONTH = 0.45          # ふつうの月
SEC_MONTH_EVENT = 1.2     # 所有権が動いた／町を出た月
SEC_LABEL = 2.0           # 版ラベルのカード
SEC_COMPARE = 5.5         # 末尾の比較の一枚


def px(v: float) -> float:
    """ピクセル指定を matplotlib の pt に直す（dpi=120）。"""
    return v * 72.0 / DPI


def box(x: float, y: float, w: float, h: float) -> List[float]:
    """左上原点のピクセル矩形 → matplotlib の figure 座標。"""
    return [x / W, (H - y - h) / H, w / W, h / H]


# ---------------------------------------------------------------------------
# 文字（日本語の折り返し。全角=1・半角=0.5 で数える）
# ---------------------------------------------------------------------------

def _w(ch: str) -> float:
    return 1.0 if unicodedata.east_asian_width(ch) in "WFA" else 0.5


def wrap_cjk(text: str, limit: float) -> List[str]:
    lines: List[str] = []
    cur, acc = "", 0.0
    for ch in str(text).replace("\n", " ").strip():
        if acc + _w(ch) > limit and cur:
            lines.append(cur)
            cur, acc = "", 0.0
        cur += ch
        acc += _w(ch)
    if cur:
        lines.append(cur)
    return lines or [""]


def clip_cjk(text: str, limit: float) -> str:
    out, acc = "", 0.0
    for ch in str(text).replace("\n", " ").strip():
        if acc + _w(ch) > limit:
            return out + "…"
        out += ch
        acc += _w(ch)
    return out


# ---------------------------------------------------------------------------
# 走行の読み込み
# ---------------------------------------------------------------------------

def _load(run_dir: str, name: str, default: Any = None) -> Any:
    for p in (os.path.join(run_dir, name),
              os.path.join(run_dir, "checkpoint", name)):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    if default is None:
        raise SystemExit(f"{name} が run_dir に無い: {run_dir}")
    return default


def acquirer_mandate(version: str) -> Optional[str]:
    """X社への命令文を、その版のソース src/<version>.py から原文のまま読む。"""
    for mod in (f"{version}.py", "field_v9.py"):
        p = os.path.join(ROOT, "src", mod)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            src = f.read()
        m = re.search(r'"(あなたはX社である。[^"]*)"', src)
        if m:
            return m.group(1)
    return None


class Run:
    def __init__(self, run_dir: str, label: Optional[str] = None) -> None:
        self.dir = os.path.abspath(run_dir)
        self.summary = _load(run_dir, "summary.json")
        self.label = label or str(self.summary.get("run_name", ""))
        self.version = str(self.summary.get("scenario_version", "field_v9"))
        self.book = M.load_personas(os.path.join(run_dir, "personas.yaml"))
        self.kinds = M.kind_map(self.book)
        self.codes = M.parcel_codes(self.book["parcels"])
        try:
            from src.field_v9 import parcel_grid_v9
            self.grid = parcel_grid_v9(self.book["parcels"])
        except Exception:
            self.grid = [str(p["name"]) for p in self.book["parcels"]]
        self.ledger = {int(r["step"]): r["rows"]
                       for r in _load(run_dir, "ledger_by_step.json")}
        self.months = sorted(self.ledger)
        self.offers = _load(run_dir, "offers.json", [])
        self.transfers = _load(run_dir, "transfers.json", [])
        self.left = _load(run_dir, "left_agents.json", [])
        self.monthly = {int(r["step"]): r
                        for r in _load(run_dir, "monthly.json", [])}

        def bucket(rows, key="step"):
            out: Dict[int, List[dict]] = {}
            for r in rows:
                out.setdefault(int(r[key]), []).append(r)
            return out

        self.offers_by_month = bucket(self.offers)
        self.transfers_by_month = bucket(self.transfers)
        self.left_by_month = bucket(self.left)

        self.x_parcels: Dict[int, List[str]] = {}
        self.offers_cum: Dict[int, int] = {}
        cum = 0
        for m in self.months:
            cum += len(self.offers_by_month.get(m, []))
            self.offers_cum[m] = cum
            self.x_parcels[m] = [
                r["parcel"] for r in self.ledger[m]
                if r.get("land") == M.ACQUIRER_NAME
                or r.get("building") == M.ACQUIRER_NAME]

    def event_months(self) -> List[int]:
        return sorted(set(self.transfers_by_month) | set(self.left_by_month))

    def n_town(self, m: int) -> int:
        r = self.monthly.get(m)
        if r and "in_town" in r:
            return int(r["in_town"])
        return sum(1 for a in self.book["agents"] if a.get("resident", True))

    def n_absent(self, m: int) -> int:
        r = self.monthly.get(m)
        if r and "absentee_owners" in r:
            return int(r["absentee_owners"])
        return sum(1 for a in self.book["agents"]
                   if not a.get("resident", True))

    def rented_offer_stat(self) -> Optional[Tuple[int, int]]:
        """「借家がある区画への提示」の件数と成約。

        この走行について公開済みの集計 docs/submission/emergence_*.json が
        あり run_dir が一致するときだけ使う（別の走行では出さない）。
        """
        import glob
        for p in sorted(glob.glob(os.path.join(
                ROOT, "docs", "submission", "emergence_*.json"))):
            try:
                with open(p, encoding="utf-8") as f:
                    e = json.load(f)
            except Exception:
                continue
            if str(e.get("run_dir")) != os.path.basename(self.dir):
                continue
            row = (e.get("by_rights_shape") or {}).get("借家がある区画")
            if not row:
                continue
            return int(row["offers"]), int(row["sold"])
        return None

    def meta(self) -> Dict[str, Any]:
        s = self.summary
        last = self.months[-1]
        rs = self.rented_offer_stat()
        return {
            "label": self.label,
            "run_dir": os.path.basename(self.dir),
            "months": int(s.get("months_run", last)),
            "agents": int(s.get("agents", len(self.book["agents"]))),
            "parcels": int(s.get("parcels_total", len(self.book["parcels"]))),
            "offers_total": int(s.get("offers_total", len(self.offers))),
            "offers_declined": int(s.get("offers_declined", 0)),
            "transfers_total": int(s.get("transfers_total",
                                         len(self.transfers))),
            "sold_agents": int(s.get("sold_agents", 0)),
            "left_agents": int(s.get("left_agents", len(self.left))),
            "in_town_end": int(s.get("in_town_end", 0)),
            "x_parcels_end": len(self.x_parcels[last]),
            "rented_offers": rs[0] if rs else None,
            "rented_sold": rs[1] if rs else None,
            "mandate": acquirer_mandate(self.version),
        }


# ---------------------------------------------------------------------------
# その月の出来事（すべて JSON の値そのまま）
# ---------------------------------------------------------------------------

def transfer_line(run: Run, t: dict) -> str:
    code = run.codes.get(t["parcel"], "")
    kind = str(t.get("kind", ""))
    if t.get("left_town"):
        tail = f"{kind}を売って町を出た"
    elif kind == "建物" and t.get("user_after") == t.get("name"):
        tail = "建物だけ売って、そのまま住み続けた"
    elif t.get("user_after") == t.get("name"):
        tail = f"{kind}を売って、そのまま使い続けている"
    elif not t.get("was_user"):
        tail = f"誰も使っていない{kind}を手放した"
    else:
        tail = f"{kind}を売って、使う人がいなくなった"
    return f"{t['name']}　{code} {t['parcel']}　→　{tail}"


def one_decline(run: Run, m: int) -> Optional[Tuple[str, str]]:
    for o in run.offers_by_month.get(m, []):
        if o.get("accepted"):
            continue
        txt = str(o.get("decline_reason") or "").strip()
        if txt:
            return str(o.get("to", "")), clip_cjk(txt, 40)
    return None


# ---------------------------------------------------------------------------
# 図（本編のコマ）
# ---------------------------------------------------------------------------

def new_fig():
    return plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)


def plain_ax(fig, rect, facecolor=None):
    ax = fig.add_axes(rect)
    if facecolor:
        ax.set_facecolor(facecolor)
    else:
        ax.patch.set_alpha(0.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return ax


def draw_section_ax(ax, rows, kinds, codes, highlight, bands: int = 3) -> None:
    """v9_map.draw_section と同じ規則で、渡した軸に断面図を描く。"""
    n = len(rows)
    per = (n + bands - 1) // bands
    ax.set_xlim(-0.55, per + 0.35)
    ax.set_ylim(-0.95, bands * 3.6 - 0.55)
    ax.axis("off")
    for i, r in enumerate(rows):
        band, col = i // per, i % per
        y0 = (bands - 1 - band) * 3.6
        land_kind = kinds.get(r["land"], M.T_TOWN)
        ax.add_patch(Rectangle(
            (col + 0.05, y0), 0.9, 1.0, facecolor=M.COLOR[land_kind],
            edgecolor="#333333", lw=0.7,
            hatch=(M.HATCH if M.is_leased_land(r) else None)))
        if r["has_building"]:
            bk = kinds.get(r["building"], M.T_TOWN)
            ax.add_patch(Rectangle(
                (col + 0.15, y0 + 1.05), 0.7, 1.0, facecolor=M.COLOR[bk],
                edgecolor="#333333", lw=0.7,
                hatch=(M.HATCH if M.is_rented_building(r) else None)))
        else:
            ax.add_patch(Rectangle((col + 0.15, y0 + 1.05), 0.7, 1.0,
                                   facecolor="white", edgecolor="#c8ced3",
                                   lw=0.7, ls=":"))
        if M.used_by_townsfolk(r, kinds):
            ax.plot([col + 0.5], [y0 + 2.36], marker="o",
                    markersize=px(11), color=M.USER_MARK)
        ax.text(col + 0.5, y0 - 0.16, codes.get(r["parcel"], ""), ha="center",
                va="top", fontsize=px(21))
        if r["parcel"] in highlight:
            ax.add_patch(Rectangle(
                (col - 0.02, y0 - 0.10), 1.04, 2.78, fill=False,
                edgecolor=M.COLOR[M.T_X], lw=px(3.6), ls="--", zorder=6))
            ax.text(col + 0.5, y0 + 2.72, "▼", ha="center", va="bottom",
                    fontsize=px(24), color=M.COLOR[M.T_X], zorder=7)


def draw_plan_ax(ax, rows, kinds, codes, grid, highlight, cols: int = 8):
    """v9_map.draw_plan と同じ規則で、渡した軸に平面図を描く（記号のみ）。"""
    by_parcel = {r["parcel"]: r for r in rows}
    n = len(grid)
    nrows = (n + cols - 1) // cols
    ax.set_xlim(-0.15, cols + 0.15)
    ax.set_ylim(-0.35, nrows * 1.6 - 0.30)
    ax.axis("off")
    for i, parcel in enumerate(grid):
        r = by_parcel[parcel]
        c, rw = i % cols, i // cols
        x, y = c, (nrows - 1 - rw) * 1.6
        land_kind = kinds.get(r["land"], M.T_TOWN)
        ax.add_patch(Rectangle(
            (x + 0.03, y), 0.94, 1.15, facecolor=M.COLOR[land_kind],
            edgecolor="#333333", lw=0.8,
            hatch=(M.HATCH if M.is_leased_land(r) else None)))
        if r["has_building"]:
            bk = kinds.get(r["building"], M.T_TOWN)
            ax.add_patch(Rectangle(
                (x + 0.30, y + 0.22), 0.40, 0.55, facecolor=M.COLOR[bk],
                edgecolor="#1a1a1a", lw=1.0,
                hatch=(M.HATCH if M.is_rented_building(r) else None)))
        if M.used_by_townsfolk(r, kinds):
            ax.plot([x + 0.86], [y + 0.96], marker="o", markersize=px(10),
                    color=M.USER_MARK)
        ax.text(x + 0.09, y + 0.99, codes.get(parcel, ""), ha="left", va="top",
                fontsize=px(20), color="#111111")
        if parcel in highlight:
            ax.add_patch(Rectangle(
                (x - 0.01, y - 0.04), 1.02, 1.23, fill=False,
                edgecolor=M.COLOR[M.T_X], lw=px(3.6), ls="--", zorder=6))


def legend_row(fig, y: float) -> None:
    """凡例（v9_map と同じ色・同じ言葉）を1行で。"""
    ax = plain_ax(fig, box(0, y, W, 56))
    items = [M.T_TOWN, M.T_ABSENT, M.T_GOV, M.T_X, M.T_NONE]
    x = 22.0
    sw, size = 30.0, 26.0
    for lab in items:
        ax.add_patch(Rectangle((x / W, 0.30), sw / W, 0.42,
                               facecolor=M.COLOR[lab], edgecolor=DIM, lw=1.0))
        ax.text((x + sw + 9) / W, 0.51, lab, va="center", ha="left",
                fontsize=px(size), color=FG)
        x += sw + 9 + sum(_w(c) for c in lab) * size + 26
    ax.add_patch(Rectangle((x / W, 0.30), sw / W, 0.42, facecolor="#ffffff",
                           edgecolor=DIM, lw=1.0, hatch=M.HATCH))
    t = "斜線＝借家／借地"
    ax.text((x + sw + 9) / W, 0.51, t, va="center", ha="left",
            fontsize=px(size), color=FG)
    x += sw + 9 + sum(_w(c) for c in t) * size + 26
    ax.text(x / W, 0.51, "●＝普段町にいる人が使っている（創発に参加）",
            va="center", ha="left", fontsize=px(size), color=FG)


def draw_progress(fig, run: Run, m: int) -> None:
    ax = plain_ax(fig, box(22, 162, W - 44, 54))
    months = run.months
    n = len(months)
    ev = set(run.event_months())
    gap = 0.0016
    seg = (1.0 - gap * (n - 1)) / n
    for i, k in enumerate(months):
        x = i * (seg + gap)
        if k < m:
            c = ACCENT
        elif k == m:
            c = "#ffffff"
        else:
            c = "#2b323d"
        ax.add_patch(Rectangle((x, 0.46), seg, 0.30, facecolor=c, lw=0))
        if k in ev:
            ax.add_patch(Rectangle((x, 0.80), seg, 0.18,
                                   facecolor=M.COLOR[M.T_X], lw=0))
        if k == m:
            ax.add_patch(Rectangle((x - gap, 0.40), seg + gap * 2, 0.42,
                                   fill=False, edgecolor="#ffffff",
                                   lw=px(2.0)))
    ax.text(0.0, 0.30, f"第{months[0]}月", fontsize=px(21), color=DIM,
            va="top", ha="left")
    ax.text(1.0, 0.30, f"第{months[-1]}月", fontsize=px(21), color=DIM,
            va="top", ha="right")
    ax.text(0.5, 0.30, "赤い印＝所有権が動いた月", fontsize=px(21),
            color=DIM, va="top", ha="center")


def draw_month(run: Run, m: int, out: str) -> str:
    plt.rcParams["font.family"] = M._pick_font()
    plt.rcParams["axes.unicode_minus"] = False
    fig = new_fig()
    rows = run.ledger[m]
    ts = run.transfers_by_month.get(m, [])
    lv = run.left_by_month.get(m, [])
    highlight = {t["parcel"] for t in ts}
    last = run.months[-1]

    # ── 最上部：第N月（特大）
    hd = plain_ax(fig, box(0, 0, W, 168))
    hd.text(0.5, 0.52, f"第 {m} 月", fontsize=px(112), color=FG,
            va="center", ha="center", fontweight="bold")
    y_, mo_ = divmod(m - 1, 12)
    hd.text(0.5, 0.10, f"開始から {m} か月目　／　"
            f"{last // 12}年のうち {y_}年{mo_ + 1}か月",
            fontsize=px(27), color=MUTED, va="center", ha="center")
    hd.text(22 / W, 0.68, TITLE_MAIN_1 + TITLE_MAIN_2, fontsize=px(28),
            color=ACCENT, va="center", ha="left")
    hd.text(22 / W, 0.40, clip_cjk(run.label, 26), fontsize=px(25),
            color=MUTED, va="center", ha="left")
    hd.text(1 - 22 / W, 0.68,
            f"A市・{run.summary.get('agents')}人・"
            f"{run.summary.get('parcels_total')}区画",
            fontsize=px(28), color=MUTED, va="center", ha="right")
    hd.text(1 - 22 / W, 0.40, "X社＝町の外の不動産投資会社",
            fontsize=px(25), color=M.COLOR[M.T_X], va="center", ha="right")

    draw_progress(fig, run, m)

    # ── 中段：平面図（左）と断面図（右）。白いカードの上に描く。
    for x0, w0, cap in ((22, 606, "平面図｜町の形"),
                        (640, W - 662, "断面図｜下＝土地の持ち主／上＝建物の持ち主")):
        b = box(x0, 238, w0, 650)
        fig.patches.append(Rectangle(
            (b[0], b[1]), b[2], b[3], transform=fig.transFigure,
            facecolor=CARD, edgecolor=PANEL_EDGE, lw=1.2, zorder=-5))
        cax = plain_ax(fig, box(x0 + 14, 246, w0 - 28, 34))
        cax.text(0.0, 0.5, cap, fontsize=px(25), color="#3a434f",
                 va="center", ha="left")
    pax = fig.add_axes(box(34, 288, 582, 592))
    pax.set_facecolor(CARD)
    draw_plan_ax(pax, rows, run.kinds, run.codes, run.grid, highlight)
    sax = fig.add_axes(box(652, 288, W - 686, 592))
    sax.set_facecolor(CARD)
    draw_section_ax(sax, rows, run.kinds, run.codes, highlight)

    legend_row(fig, 892)

    # ── 最下段：出来事（左）と累計（右）
    ev = plain_ax(fig, box(22, 950, 1468, 122))
    offs = run.offers_by_month.get(m, [])
    by_kind: Dict[str, int] = {}
    for o in offs:
        by_kind[str(o.get("kind"))] = by_kind.get(str(o.get("kind")), 0) + 1
    kt = "・".join(f"{k} {v}" for k, v in sorted(by_kind.items(),
                                                key=lambda x: -x[1]))
    ly = 0.86
    ev.text(0.0, ly, f"X社の提示 {len(offs)} 件"
            + (f"（{kt}）" if kt else ""), fontsize=px(30), color=FG,
            va="center", ha="left")
    ly -= 0.32
    for t in ts:
        ev.text(0.0, ly, "所有権が動いた　"
                + clip_cjk(transfer_line(run, t), 46),
                fontsize=px(26), color=M.COLOR[M.T_X], va="center", ha="left")
        ly -= 0.30
    for a in lv:
        ev.text(0.0, ly, f"町を出た　{a.get('name', '')}", fontsize=px(28),
                color=M.COLOR[M.T_ABSENT], va="center", ha="left")
        ly -= 0.30
    if ts or lv:
        d = one_decline(run, m)
        if d and ly > -0.05:
            ev.text(0.0, ly, f"断りの一言「{clip_cjk(d[1], 26)}」"
                    f"— {clip_cjk(d[0], 12)}",
                    fontsize=px(26), color=MUTED, va="center", ha="left")

    cu = plain_ax(fig, box(1510, 950, W - 1532, 122))
    n_x = len(run.x_parcels[m])
    cu.text(1.0, 0.87, "X社の所有権が及んだ区画", fontsize=px(25), color=MUTED,
            va="center", ha="right")
    cu.text(1.0, 0.48, f"{n_x} / {run.summary.get('parcels_total')}",
            fontsize=px(46), color=ACCENT if n_x else MUTED, va="center",
            ha="right", fontweight="bold")
    cu.text(1.0, 0.09, f"X社の提示 累計 {run.offers_cum[m]} 件",
            fontsize=px(25), color=MUTED, va="center", ha="right")
    sp = fig.add_axes(box(1512, 986, 176, 52))
    sp.set_facecolor(BG)
    sp.axis("off")
    ymax = max(1, len(run.x_parcels[run.months[-1]]))
    sp.set_xlim(run.months[0] - 0.5, run.months[-1] + 0.5)
    sp.set_ylim(-0.15, ymax + 0.35)
    sp.plot(run.months, [len(run.x_parcels[k]) for k in run.months],
            color="#2b323d", lw=px(2.0))
    upto = [k for k in run.months if k <= m]
    sp.plot(upto, [len(run.x_parcels[k]) for k in upto], color=ACCENT,
            lw=px(3.4))
    sp.plot([m], [n_x], marker="o", markersize=px(8), color=ACCENT)

    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    return out


def draw_label_card(run: Run, out: str) -> str:
    plt.rcParams["font.family"] = M._pick_font()
    fig = new_fig()
    ax = plain_ax(fig, [0, 0, 1, 1])
    s = run.summary
    ax.text(0.5, 0.615, clip_cjk(run.label, 30), fontsize=px(78), color=FG,
            va="center", ha="center", fontweight="bold")
    ax.plot([0.42, 0.58], [0.545, 0.545], color=ACCENT, lw=px(4))
    ax.text(0.5, 0.475,
            f"A市・{s.get('agents')}人・{s.get('parcels_total')}区画・"
            f"{s.get('months_run')}か月",
            fontsize=px(40), color=ACCENT, va="center", ha="center")
    ax.text(0.5, 0.395, f"（{s.get('scenario_version')}）", fontsize=px(28),
            color=MUTED, va="center", ha="center")
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    return out


def draw_compare(metas: List[Dict[str, Any]], out: str) -> str:
    plt.rcParams["font.family"] = M._pick_font()
    fig = new_fig()
    ax = plain_ax(fig, [0, 0, 1, 1])
    mo = int(metas[0].get("months", 0))
    yr = f"{mo // 12}年（{mo}か月）" if mo % 12 == 0 else f"{mo}か月"
    ax.text(0.5, 0.925, f"{yr}のあと", fontsize=px(62), color=FG,
            va="center", ha="center", fontweight="bold")
    ax.plot([0.44, 0.56], [0.875, 0.875], color=ACCENT, lw=px(4))

    cols = [("X社の提示", "offers_total", "件"),
            ("成約", "transfers_total", "件"),
            ("売った人", "sold_agents", "人"),
            ("町を出た人", "left_agents", "人"),
            ("X社の区画", "x_parcels_end", None)]
    xs = [0.355, 0.485, 0.605, 0.725, 0.865]
    ax.text(0.035, 0.795, "版", fontsize=px(28), color=MUTED, va="center",
            ha="left")
    for x, (lab, _, _) in zip(xs, cols):
        ax.text(x, 0.795, lab, fontsize=px(28), color=MUTED, va="center",
                ha="center")
    ax.plot([0.03, 0.97], [0.762, 0.762], color=PANEL_EDGE, lw=px(2))

    # 行間＝版の本数で自動調整（4本までは従来どおり0.115。5本目以降は
    # 最終行の位置が4本のときと揃うよう詰める＝下の区切り線とはみ出さない）。
    n_rows = max(1, len(metas))
    row_step = 0.115 if n_rows <= 4 else (0.685 - 0.340) / (n_rows - 1)

    y = 0.685
    for meta in metas:
        ax.text(0.035, y, clip_cjk(meta.get("label", ""), 20),
                fontsize=px(34), color=FG, va="center", ha="left")
        for x, (_, key, unit) in zip(xs, cols):
            v = meta.get(key)
            if key == "x_parcels_end":
                txt = f"{v} / {meta.get('parcels')}"
            else:
                txt = f"{v}{unit or ''}"
            ax.text(x, y, txt, fontsize=px(40), color=ACCENT, va="center",
                    ha="center", fontweight="bold")
        y -= row_step

    m0 = metas[0]
    rule = max(0.30, y + 0.045)
    ax.plot([0.03, 0.97], [rule, rule], color=PANEL_EDGE, lw=px(2))
    col_a: List[str] = []
    first = True
    for meta in metas:
        if meta.get("rented_offers") is None:
            continue
        head = str(meta.get("label", "")).split()[0] if meta.get("label") else ""
        what = "他人が借りて使う区画への提示" if first else "同"
        col_a.append(f"{head}：{what} {meta['rented_offers']}件 → "
                     f"成約 {meta['rented_sold']}件")
        first = False
    col_b: List[str] = []
    first = True
    for meta in metas:
        head = str(meta.get("label", "")).split()[0] if meta.get("label") else ""
        what = ("断られた提示" if first else "同")
        tail = ("／3年後に町にいる人" if first else "／")
        col_b.append(f"{head}：{what} {meta.get('offers_declined')}件"
                     f"{tail}{meta.get('in_town_end')}人")
        first = False
    n_note = max(len(col_a), len(col_b))
    top = rule - 0.075
    step = min(0.062, max(0.036, (top - 0.075) / max(1, n_note)))
    size = 32.0 if step >= 0.058 else 28.0
    for cx, col in ((0.035, col_a), (0.520, col_b)):
        yy = top
        for t in col:
            ax.text(cx, yy, t, fontsize=px(size), color=FG, va="center",
                    ha="left")
            yy -= step
    yy = top - step * n_note
    srcs = ["summary.json"] + [
        f"emergence_{str(m.get('label', '')).split()[0]}.json"
        for m in metas
        if m.get("rented_offers") is not None and m.get("label")]
    src_text = "数字はすべて走行の記録（" + "・".join(srcs) + "）から。"
    src_obj = ax.text(0.035, max(0.05, yy - 0.02), src_text,
                      fontsize=px(25.0), color=MUTED, va="center", ha="left")
    # 版が増えて出典の列挙が長くなったら、右端からはみ出さないよう縮める
    # （実際の描画幅を測って合わせる＝文字幅の見積りに頼らない）。
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = src_obj.get_window_extent(renderer=renderer)
    max_x = 0.965 * W
    if bbox.x1 > max_x and bbox.x1 > bbox.x0:
        scale = max(0.5, (max_x - bbox.x0) / (bbox.x1 - bbox.x0))
        src_obj.set_fontsize(px(25.0) * scale)
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 冒頭のタイトル演出（PIL でコマを作り、ffmpeg に直接流す）
# ---------------------------------------------------------------------------

def _pil_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for name in (("meiryob.ttc", "YuGothB.ttc") if bold
                 else ("meiryo.ttc", "YuGothR.ttc", "msgothic.ttc")):
        p = os.path.join(os.environ.get("WINDIR", r"C:\Windows"),
                         "Fonts", name)
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _hex(c: str) -> Tuple[int, int, int]:
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore


def _mix(c: Tuple[int, int, int], a: float) -> Tuple[int, int, int]:
    a = max(0.0, min(1.0, a))
    return (int(c[0] * a), int(c[1] * a), int(c[2] * a))


def _ease(t0: float, t1: float, t: float) -> float:
    if t <= t0:
        return 0.0
    if t >= t1:
        return 1.0
    x = (t - t0) / (t1 - t0)
    return 1 - (1 - x) ** 3


def intro_frame(t: float, meta: Dict[str, Any], total: float) -> Image.Image:
    img = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(img)
    out = 1.0 - _ease(total - 1.0, total, t)   # 末尾で暗転

    def center(text, y, font, color, a, dy=0.0):
        if a <= 0.01:
            return
        bb = d.textbbox((0, 0), text, font=font)
        d.text(((W - (bb[2] - bb[0])) / 2 - bb[0], y + dy * (1 - a)),
               text, font=font, fill=_mix(color, a * out))

    a1 = _ease(0.7, 2.1, t)
    center(TITLE_MAIN_1, 168, _pil_font(92, True), _hex(FG), a1, 34)
    center(TITLE_MAIN_2, 278, _pil_font(92, True), _hex(FG), a1, 34)

    a2 = _ease(2.1, 2.8, t)
    if a2 > 0.01:
        half = int(210 * a2)
        d.rectangle([W // 2 - half, 424, W // 2 + half, 429],
                    fill=_mix(_hex(ACCENT), out))

    a3 = _ease(2.8, 4.4, t)
    center(TITLE_Q_1, 484, _pil_font(44), _hex(MUTED), a3, 20)
    center(TITLE_Q_2, 546, _pil_font(46, True), _hex(ACCENT), a3, 20)

    a4 = _ease(4.9, 6.5, t)
    if a4 > 0.01 and meta.get("mandate"):
        lines = wrap_cjk(str(meta["mandate"]), 26)
        h = 42 + 50 * len(lines)
        d.rectangle([322, 660, 328, 660 + h], fill=_mix(_hex(M.COLOR[M.T_X]),
                                                        a4 * out))
        d.text((356, 658), "X社への命令", font=_pil_font(28),
               fill=_mix(_hex(M.COLOR[M.T_X]), a4 * out))
        for i, ln in enumerate(lines):
            d.text((356, 700 + i * 50), ln, font=_pil_font(38, True),
                   fill=_mix(_hex(FG), a4 * out))

    a5 = _ease(6.9, 8.3, t)
    center(f"A市・{meta.get('agents')}人・{meta.get('parcels')}区画・"
           f"{meta.get('months')}か月", 858, _pil_font(40, True),
           _hex(FG), a5, 18)
    center("会社は毎月、誰かに声を掛ける。", 928, _pil_font(32),
           _hex(MUTED), _ease(7.6, 9.0, t), 18)
    return img


# ---------------------------------------------------------------------------
# ffmpeg
# ---------------------------------------------------------------------------

def _bin(name: str = "ffmpeg") -> str:
    p = shutil.which(name)
    if p:
        return p
    cand = (r"D:\ユーザー\ffmpeg-master-latest-win64-gpl-shared\bin"
            "\\" + name + ".exe")
    if os.path.exists(cand):
        return cand
    raise SystemExit(f"{name} が見つからない")


def _noshow() -> Dict[str, Any]:
    kw: Dict[str, Any] = {}
    if os.name == "nt":
        kw["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0
        kw["startupinfo"] = si
    return kw


def _run_quiet(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", **_noshow())


X264 = ["-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart"]


def encode_stills(frames: List[Tuple[str, float]], out: str,
                  crf: int = 20) -> None:
    listfile = os.path.join(os.path.dirname(frames[0][0]), "concat_frames.txt")
    with open(listfile, "w", encoding="utf-8") as f:
        for path, dur in frames:
            f.write(f"file '{os.path.basename(path)}'\nduration {dur:.3f}\n")
        f.write(f"file '{os.path.basename(frames[-1][0])}'\n")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    cmd = [_bin(), "-y", "-f", "concat", "-safe", "0", "-i", listfile,
           "-r", str(FPS), "-crf", str(crf)] + X264 + [os.path.abspath(out)]
    r = _run_quiet(cmd)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-4000:] + "\n")
        raise SystemExit("ffmpeg（静止画→動画）が失敗した")


def encode_intro(meta: Dict[str, Any], out: str, seconds: float = 10.5,
                 crf: int = 20, frame_png: Optional[str] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    cmd = [_bin(), "-y", "-loglevel", "error", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-crf", str(crf)] + X264 + [os.path.abspath(out)]
    # stderr はパイプにしない（書き込み中に埋まると相互待ちで止まる）
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         **_noshow())
    n = int(seconds * FPS)
    assert p.stdin is not None
    for i in range(n):
        t = i / FPS
        img = intro_frame(t, meta, seconds)
        if frame_png and abs(t - (seconds - 1.6)) < 0.5 / FPS:
            img.save(frame_png)
        p.stdin.write(img.tobytes())
    p.stdin.close()
    err = p.stderr.read().decode("utf-8", "replace") if p.stderr else ""
    if p.wait() != 0:
        sys.stderr.write(err[-4000:] + "\n")
        raise SystemExit("ffmpeg（冒頭演出）が失敗した")


def concat_mp4(parts: List[str], out: str, crf: int = 20) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    d = os.path.dirname(os.path.abspath(parts[0]))
    listfile = os.path.join(d, "_concat_parts.txt")
    with open(listfile, "w", encoding="utf-8") as f:
        for p in parts:
            f.write("file '" + os.path.abspath(p).replace("\\", "/") + "'\n")
    cmd = [_bin(), "-y", "-f", "concat", "-safe", "0", "-i", listfile,
           "-r", str(FPS), "-crf", str(crf)] + X264 + [os.path.abspath(out)]
    r = _run_quiet(cmd)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-4000:] + "\n")
        raise SystemExit("ffmpeg（連結）が失敗した")


def probe(path: str) -> Dict[str, Any]:
    r = _run_quiet([_bin("ffprobe"), "-v", "error",
                    "-show_entries", "format=duration,size",
                    "-show_entries",
                    "stream=codec_name,width,height,r_frame_rate,nb_frames",
                    "-of", "json", os.path.abspath(path)])
    return json.loads(r.stdout or "{}")


def report(path: str) -> str:
    j = probe(path)
    fmt = j.get("format", {})
    st = (j.get("streams") or [{}])[0]
    return (f"{os.path.abspath(path)}  "
            f"{float(fmt.get('duration', 0)):.1f}秒  "
            f"{int(fmt.get('size', 0)) / 1e6:.1f}MB  "
            f"{st.get('width')}x{st.get('height')}  {st.get('codec_name')}  "
            f"{st.get('r_frame_rate')}fps  {st.get('nb_frames')}フレーム")


# ---------------------------------------------------------------------------

def build_body(run: Run, out: str, crf: int, shots: Optional[str]) -> str:
    fdir = os.path.join(run.dir, "video_frames")
    os.makedirs(fdir, exist_ok=True)
    frames: List[Tuple[str, float]] = []

    card = draw_label_card(run, os.path.join(fdir, "f000_label.png"))
    base = Image.open(card).convert("RGB")
    black = Image.new("RGB", base.size, (0, 0, 0))
    for i in range(10):                      # 黒からのフェードイン
        p = os.path.join(fdir, f"e{i:02d}_fadein.png")
        Image.blend(black, base, (i + 1) / 11.0).save(p)
        frames.append((p, 1.0 / FPS))
    frames.append((card, SEC_LABEL))

    ev = set(run.event_months())
    for i, m in enumerate(run.months, start=1):
        p = os.path.join(fdir, f"f{i:03d}_m{m:02d}.png")
        draw_month(run, m, p)
        frames.append((p, SEC_MONTH_EVENT if m in ev else SEC_MONTH))
        print("frame", m, flush=True)

    encode_stills(frames, out, crf)
    with open(out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(run.meta(), f, ensure_ascii=False, indent=2)

    if shots:
        os.makedirs(shots, exist_ok=True)
        tag = re.sub(r"[^0-9a-zA-Z_]+", "_", run.version)
        picks = [("label", card), ("m01", frames[11][0])]
        evs = sorted(ev)
        if evs:
            picks.append((f"m{evs[-1]:02d}",
                          frames[11 + run.months.index(evs[-1])][0]))
        for name, src in picks:
            dst = os.path.join(shots, f"video_{tag}_frame_{name}.png")
            shutil.copyfile(src, dst)
            print("wrote", dst)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="走行の run_dir（本編を作る）")
    ap.add_argument("--label", default=None, help="版ラベル（無ければ run_name）")
    ap.add_argument("--concat", nargs="*", default=None,
                    help="連結する本編 mp4（--title で冒頭演出を付ける）")
    ap.add_argument("--title", action="store_true", help="冒頭タイトル演出を付ける")
    ap.add_argument("--no-compare", action="store_true",
                    help="末尾の比較の一枚を付けない")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--max-mb", type=float, default=25.0)
    ap.add_argument("--shots", default="docs/submission",
                    help="確認用フレーム PNG の置き場（none で出さない）")
    args = ap.parse_args()
    shots = None if args.shots in (None, "none", "") else args.shots

    if args.run:
        run = Run(args.run, args.label)
        build_body(run, args.out, args.crf, shots)
        print(report(args.out))
        return 0

    if not args.concat:
        raise SystemExit("--run か --concat のどちらかが要る")

    metas: List[Dict[str, Any]] = []
    for p in args.concat:
        mp = p + ".meta.json"
        if os.path.exists(mp):
            with open(mp, encoding="utf-8") as f:
                metas.append(json.load(f))
    if not metas:
        raise SystemExit("本編の .meta.json が無い（--run で作り直す）")

    parts_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                             "_video_parts")
    os.makedirs(parts_dir, exist_ok=True)
    parts: List[str] = []
    title_png = os.path.join(shots, "video_frame_title.png") if shots else None
    if args.title:
        intro = os.path.join(parts_dir, "intro.mp4")
        encode_intro(metas[0], intro, crf=args.crf, frame_png=title_png)
        parts.append(intro)
        print("wrote", intro, flush=True)
    parts += [os.path.abspath(p) for p in args.concat]
    if not args.no_compare:
        cmp_png = os.path.join(parts_dir, "compare.png")
        draw_compare(metas, cmp_png)
        cmp_mp4 = os.path.join(parts_dir, "compare.mp4")
        encode_stills([(cmp_png, SEC_COMPARE)], cmp_mp4, args.crf)
        parts.append(cmp_mp4)
        if shots:
            shutil.copyfile(cmp_png,
                            os.path.join(shots, "video_frame_compare.png"))
            print("wrote", os.path.join(shots, "video_frame_compare.png"))

    crf = args.crf
    for _ in range(4):
        concat_mp4(parts, args.out, crf)
        mb = os.path.getsize(args.out) / 1e6
        print(f"crf={crf} size={mb:.1f}MB", flush=True)
        if mb <= args.max_mb:
            break
        crf += 4
    print(report(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
