# -*- coding: utf-8 -*-
"""最小の町の走行から、創発の効き目 (2)(3)(4) の1枚図を作る。

  python make_figures.py <走行フォルダ か emergence_v8c.json>

決まった入力からは必ず同じ絵が出る（乱数の種は固定・LLM は使わない）。
"""

from __future__ import annotations

import os
import sys
import collections

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.lines import Line2D

import emergence_data as E

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SOURCE = os.path.join(
    HERE, "..", "..", "..", "simulations", "2026-08-29_2358_102_v8b_listing_chat_A"
)

INK = "#1b1b1b"
GREY = "#b9b9b9"
BLUE = "#2f6fb5"      # X社の話
ORANGE = "#e08a2e"    # 誰かが売った話
RED = "#c0392b"       # 売った
AMBER = "#e8a33d"     # 売りに出しただけ
SLATE = "#9aa3ab"     # 動かなかった

for fam in ("Noto Sans JP", "Yu Gothic", "Meiryo", "MS Gothic"):
    try:
        matplotlib.font_manager.findfont(fam, fallback_to_default=False)
        plt.rcParams["font.family"] = fam
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"


def _save(fig, path):
    fig.savefig(path, dpi=125, bbox_inches="tight")
    plt.close(fig)
    mb = os.path.getsize(path) / 1024.0 / 1024.0
    print("%s  (%.2f MB)" % (path, mb))
    return path


# ==========================================================================
# (2) 聞いた量と判断
# ==========================================================================

def fig_heard_vs_action(town, out_path):
    rows = []
    for p in town.people:
        if not p.sellable:
            continue
        acq, sale, win, averaged = E.heard_before_decision(town, p)
        rows.append({
            "name": p.name, "acq": acq, "sale": sale, "total": acq + sale,
            "outcome": p.outcome, "sold": p.sold_month,
            "listed": p.first_listed_month, "averaged": averaged,
        })
    rows.sort(key=lambda r: r["total"])

    fig, ax = plt.subplots(figsize=(12.4, 10.4))
    y = np.arange(len(rows))
    acq = [r["acq"] for r in rows]
    sale = [r["sale"] for r in rows]
    ax.barh(y, acq, color=BLUE, height=0.62, label="X社の話を聞いた回数")
    ax.barh(y, sale, left=acq, color=ORANGE, height=0.62, label="誰かが売った話を聞いた回数")

    colour = {"sold": RED, "listed": AMBER, "stayed": SLATE}
    labels = []
    for r in rows:
        if r["outcome"] == "sold":
            labels.append("第%d月に売った" % r["sold"])
        elif r["outcome"] == "listed":
            labels.append("第%d月に売りに出した（売らず）" % r["listed"])
        else:
            labels.append("最後まで動かなかった")

    xmax = max(r["total"] for r in rows) or 1
    for i, (r, lab) in enumerate(zip(rows, labels)):
        ax.scatter([r["total"] + xmax * 0.02], [i], s=70,
                   color=colour[r["outcome"]], zorder=5, clip_on=False)
        ax.text(r["total"] + xmax * 0.045, i, lab, va="center", ha="left",
                fontsize=9.5, color=colour[r["outcome"]])

    ax.set_yticks(y)
    ax.set_yticklabels([
        r["name"] + ("  *" if r["averaged"] else "") for r in rows
    ], fontsize=10)
    ax.set_xlim(0, xmax * 1.42)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xlabel("判断する直前の3か月に、その人の耳に入った話の件数", fontsize=11.5)
    ax.set_title(
        "(2) 「うわさを浴びた人ほど、動いたのか」\n"
        "町の28人を、判断の前3か月に聞いた話の量で並べた",
        fontsize=15, pad=16, loc="left",
    )
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e8e8e8", zorder=0)
    ax.set_axisbelow(True)

    means = {}
    for k in ("sold", "listed", "stayed"):
        got = [r["total"] for r in rows if r["outcome"] == k]
        means[k] = (np.mean(got) if got else 0.0, len(got))
    note = (
        "平均（前3か月に聞いた話の件数）\n"
        "  売った人      %.1f 件 （%d人）\n"
        "  出しただけの人 %.1f 件 （%d人）\n"
        "  動かなかった人 %.1f 件 （%d人）\n"
        "  * 動かなかった人は判断の月が無いので、\n"
        "     全期間を3か月ずつに直した平均で置いた"
        % (means["sold"][0], means["sold"][1],
           means["listed"][0], means["listed"][1],
           means["stayed"][0], means["stayed"][1])
    )
    ax.text(0.985, 0.02, note, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, family="monospace" if False else None,
            bbox=dict(boxstyle="round,pad=0.6", fc="#f6f6f4", ec="#d8d8d4"))

    handles = [
        Patch(color=BLUE, label="X社の話を聞いた回数"),
        Patch(color=ORANGE, label="誰かが売った話を聞いた回数"),
        Line2D([], [], marker="o", ls="", color=RED, label="売った"),
        Line2D([], [], marker="o", ls="", color=AMBER, label="売りに出したが売らなかった"),
        Line2D([], [], marker="o", ls="", color=SLATE, label="最後まで動かなかった"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.075),
              ncol=3, frameon=False, fontsize=10)
    return _save(fig, out_path)


# ==========================================================================
# (3) 伝染の形
# ==========================================================================

def _layout(town):
    """隣近所のつながりから町の並びを作る。地区ごとにまとめた位置から
    始めて、同じ種で必ず同じ絵になるようにする（実際の地理ではない）。"""
    import networkx as nx

    g = nx.Graph()
    for p in town.people:
        g.add_node(p.pid)
    for p in town.people:
        for q in p.neighbours:
            if q in g:
                g.add_edge(p.pid, q)

    districts = sorted({p.district for p in town.people})
    anchors = {}
    for i, d in enumerate(districts):
        ang = 2 * np.pi * i / max(1, len(districts))
        anchors[d] = np.array([np.cos(ang), np.sin(ang)])
    init = {}
    for i, p in enumerate(town.people):
        off = np.array([np.cos(i * 2.39996), np.sin(i * 2.39996)]) * 0.18
        init[p.pid] = anchors.get(p.district, np.zeros(2)) + off
    pos = nx.spring_layout(g, pos=init, iterations=250, seed=7, k=0.55)
    return g, pos


def _box(cx, cy, w, h, ha, va):
    x0 = cx - w / 2 if ha == "center" else (cx if ha == "left" else cx - w)
    y0 = cy - h / 2 if va == "center" else (cy if va == "bottom" else cy - h)
    return (x0, y0, x0 + w, y0 + h)


def _overlap(a, b):
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def _place_labels(ax, pos, items, fontsize=8.2, node_pt=12.5):
    """名前が重ならないように、上下左右の候補からぶつからない置き場所を選ぶ。
    大きさは実際の画面上の点で測るので、図の縦横比が変わってもずれない。
    候補の順番は固定なので、同じ入力なら必ず同じ絵になる。"""
    ax.figure.canvas.draw()
    inv = ax.transData.inverted()
    o = inv.transform((0.0, 0.0))
    ux = abs(inv.transform((1.0, 0.0))[0] - o[0])   # 1画素あたりの横の目盛り
    uy = abs(inv.transform((0.0, 1.0))[1] - o[1])   # 1画素あたりの縦の目盛り
    pt = ax.figure.dpi / 72.0         # 1ポイントあたりの画素数
    char_w = fontsize * pt * 1.05 * ux    # 日本語1文字はほぼ正方形
    line_h = fontsize * pt * 1.55 * uy
    node_rx, node_ry = node_pt * pt * ux, node_pt * pt * uy

    placed = [_box(x, y, node_rx * 2, node_ry * 2, "center", "center")
              for x, y in pos.values()]
    offsets = []
    for ring in (1.0, 1.8, 2.6, 3.5, 4.5, 5.8, 7.2):
        dx_, dy_ = node_rx * ring, node_ry * ring
        offsets += [
            (0, -dy_, "center", "top"), (0, dy_, "center", "bottom"),
            (dx_, 0, "left", "center"), (-dx_, 0, "right", "center"),
            (dx_ * .8, -dy_ * .8, "left", "top"), (-dx_ * .8, -dy_ * .8, "right", "top"),
            (dx_ * .8, dy_ * .8, "left", "bottom"), (-dx_ * .8, dy_ * .8, "right", "bottom"),
            (dx_ * 1.6, -dy_ * .5, "left", "top"), (-dx_ * 1.6, -dy_ * .5, "right", "top"),
            (dx_ * 1.6, dy_ * .5, "left", "bottom"), (-dx_ * 1.6, dy_ * .5, "right", "bottom"),
            (dx_ * .4, -dy_ * 1.6, "left", "top"), (-dx_ * .4, -dy_ * 1.6, "right", "top"),
            (dx_ * .4, dy_ * 1.6, "left", "bottom"), (-dx_ * .4, dy_ * 1.6, "right", "bottom"),
        ]
    order = sorted(items, key=lambda it: (-pos[it[0]][1], pos[it[0]][0]))
    for pid, text, colour in order:
        x, y = pos[pid]
        lines = text.split("\n")
        w = max(len(s) for s in lines) * char_w
        h = len(lines) * line_h
        best = None
        for dx, dy, ha, va in offsets:
            box = _box(x + dx, y + dy, w, h, ha, va)
            if not any(_overlap(box, b) for b in placed):
                best = (x + dx, y + dy, ha, va, box)
                break
        if best is None:
            best = (x, y - node_ry * 5.5, "center", "top",
                    _box(x, y - node_ry * 5.5, w, h, "center", "top"))
        cx, cy, ha, va, box = best
        placed.append(box)
        ax.text(cx, cy, text, ha=ha, va=va, fontsize=fontsize, color=colour,
                linespacing=1.35, zorder=7,
                bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none",
                          alpha=0.8))


def fig_contagion(town, out_path):
    g, pos = _layout(town)
    by_id = {p.pid: p for p in town.people}

    fig = plt.figure(figsize=(13.2, 11.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.05, 1.0], hspace=0.20)
    ax = fig.add_subplot(gs[0])
    axb = fig.add_subplot(gs[1])

    for a, b in g.edges():
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                color="#dcdcdc", lw=1.0, zorder=1)

    cmap = plt.get_cmap("plasma")
    movers = [p for p in town.people if p.first_move_month]

    # 「先に動いた人から、話を聞いたあとで動いた」矢印
    arrows = []
    for p in movers:
        e = p.first_move_month
        lo, hi = max(1, e - E.WINDOW), max(1, e - 1)
        tally = collections.Counter()
        routes = collections.defaultdict(set)
        for d in town.deliveries:
            if d["to_pid"] != p.pid or not (lo <= d["month"] <= hi):
                continue
            src = by_id.get(d["from_pid"])
            if src is None or src.pid == p.pid:
                continue
            if not src.first_move_month or src.first_move_month >= e:
                continue
            if not (d["about_acquirer"] or d["about_sale"]):
                continue
            tally[src.pid] += 1
            routes[src.pid].add(d["route"])
        for src_pid, n in tally.most_common(2):
            arrows.append((src_pid, p.pid, n, "隣近所" in routes[src_pid]))

    for a, b, n, is_neighbour in arrows:
        ax.add_patch(FancyArrowPatch(
            pos[a], pos[b], arrowstyle="-|>", mutation_scale=15,
            lw=1.1 + min(n, 12) * 0.12,
            linestyle="-" if is_neighbour else (0, (4, 3)),
            color="#8e44ad" if is_neighbour else "#5dade2",
            connectionstyle="arc3,rad=0.16",
            shrinkA=13, shrinkB=15, alpha=0.85, zorder=3,
        ))

    months = max(town.months, 1)
    labels = []
    for p in town.people:
        x, y = pos[p.pid]
        m = p.first_move_month
        fc = cmap(0.10 + 0.85 * (m - 1) / months) if m else "#eceff1"
        ax.scatter([x], [y], s=520 if p.sold_month else 380,
                   color=fc, marker="o" if p.sellable else "s",
                   edgecolor=RED if p.sold_month else "#8d99a6",
                   linewidth=2.6 if p.sold_month else 1.0, zorder=4)
        if m:
            ax.text(x, y, str(m), ha="center", va="center", fontsize=9.5,
                    color="white", fontweight="bold", zorder=6)
        if p.sold_month and p.first_listed_month and p.first_listed_month < p.sold_month:
            tag = "%s\n第%d月に出す→第%d月に売却" % (p.name, p.first_listed_month, p.sold_month)
        elif p.sold_month:
            tag = "%s\n第%d月に売却" % (p.name, p.sold_month)
        elif m:
            tag = "%s\n第%d月に売りに出す" % (p.name, m)
        else:
            tag = p.name
        labels.append((p.pid, tag, RED if p.sold_month else
                       (INK if m else "#8a8a8a")))

    ax.margins(0.16)
    ax.set_axis_off()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, months))
    cb = fig.colorbar(sm, ax=ax, fraction=0.028, pad=0.01)
    cb.set_label("はじめて動いた月", fontsize=10)
    _place_labels(ax, pos, labels)
    ax.set_title(
        "(3) 「売る話は、どこから隣へ移ったか」\n"
        "丸＝町の28人（四角＝売る立場にない行政2人）／うすい線＝隣近所どうし\n"
        "丸の中の数字と色＝はじめて動いた月／赤い縁＝実際に名義が移った人\n"
        "矢印＝先に動いた人から売る話を聞いたあとで動いた（紫の実線＝隣近所・水色の破線＝場に居合わせて）",
        fontsize=13, pad=14, loc="left",
    )

    # 下段：どの場で売る話が続いたか
    venues = town.venues
    mat = np.zeros((len(venues), months))
    for d in town.deliveries:
        if not (d["about_acquirer"] or d["about_sale"]):
            continue
        if d["venue"] in venues and 1 <= d["month"] <= months:
            mat[venues.index(d["venue"]), d["month"] - 1] += 1
    im = axb.imshow(mat, aspect="auto", cmap="YlOrRd", interpolation="nearest",
                    extent=[0.5, months + 0.5, len(venues) - 0.5, -0.5])
    axb.set_yticks(range(len(venues)))
    axb.set_yticklabels(venues, fontsize=10)
    axb.set_xticks(range(2, months + 1, 2))
    axb.set_xlabel("月", fontsize=11)
    axb.set_title("場所ごとの連鎖：その月・その場所で「売る話」が耳に入った件数",
                  fontsize=12, loc="left", pad=8)
    cb2 = fig.colorbar(im, ax=axb, fraction=0.02, pad=0.01)
    cb2.set_label("件数", fontsize=10)
    return _save(fig, out_path)


# ==========================================================================
# (4) X社の適応
# ==========================================================================

def fig_acquirer(town, out_path):
    months = town.months
    xs = np.arange(1, months + 1)

    kinds = collections.defaultdict(set)
    seen = set()
    cum = []
    for m in xs:
        for o in town.offers:
            if o["month"] == m:
                kinds[m].add(o["template"])
                seen.add(o["template"])
        cum.append(len(seen))
    per_month = [len(kinds[m]) for m in xs]

    sent = {m["month"]: m["offers_sent"] for m in town.monthly}
    listed = {m["month"]: m["listed"] for m in town.monthly}
    sold = {m["month"]: m["sold"] for m in town.monthly}

    reasons = collections.Counter(
        (o.get("reply_reason") or "").strip() for o in town.offers
    )
    reasons.pop("", None)

    fig, axes = plt.subplots(3, 1, figsize=(13.0, 12.2),
                             gridspec_kw={"hspace": 0.42})
    fig.suptitle(
        "(4) 「買い手は、断られながら言い方を変えたか」",
        fontsize=15.5, x=0.007, ha="left", y=0.965,
    )

    ax = axes[0]
    ax.bar(xs, per_month, color=BLUE, width=0.68, label="その月に使った言い方の種類数")
    ax.plot(xs, cum, color=RED, lw=2.0, marker="o", ms=3.4,
            label="はじめから数えた種類数（累計）")
    ax.set_title("① X社が持ちかけた条件文の種類数（相手の物件名を外して数えた）",
                 fontsize=12, loc="left")
    ax.set_xlabel("月", fontsize=10.5)
    ax.set_ylabel("種類", fontsize=10.5)
    ax.set_xlim(0.3, months + 0.7)
    top = max(4, max(cum) + 2)
    ax.set_ylim(0, top)
    ax.set_yticks(range(0, top + 1))
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.grid(axis="y", color="#ededed")
    ax.set_axisbelow(True)

    tally = collections.Counter(o["template"] for o in town.offers)
    lines = ["使われた言い方は全部で %d 通り" % len(tally)]
    for i, (t, n) in enumerate(tally.most_common(4), 1):
        t = t.replace("\n", " ")
        lines.append("%d. %s（%d件）" % (i, t if len(t) <= 46 else t[:46] + "…", n))
    ax.text(0.995, 0.965, "\n".join(lines), transform=ax.transAxes,
            ha="right", va="top", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.5", fc="#f6f6f4", ec="#d8d8d4"))

    ax = axes[1]
    ax.set_title("② 断りの一言（住民が返した理由）", fontsize=12, loc="left")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#cfcfcf"); s.set_linestyle((0, (5, 4)))
    if reasons:
        names = [k for k, _ in reasons.most_common(8)][::-1]
        vals = [reasons[k] for k in names]
        ax.clear()
        ax.barh(np.arange(len(names)), vals, color=ORANGE, height=0.6)
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel("件数", fontsize=10.5)
        ax.set_title("② 断りの一言（多い順）", fontsize=12, loc="left")
    else:
        ax.text(0.5, 0.5,
                "この走行には断りの一言が残っていない。\n"
                "住民の返事は「応じた／応じなかった」の二択だけで、\n"
                "断った理由は記録されていない（%d件が「応じなかった」）。\n"
                "理由を一言で残す走行になれば、ここが埋まる。"
                % sum(1 for o in town.offers if not o["accepted"]),
                ha="center", va="center", fontsize=12, color="#7a7a7a")

    ax = axes[2]
    ax.bar(xs - 0.2, [sent.get(m, 0) for m in xs], width=0.38,
           color=BLUE, label="X社が条件を持ちかけた件数")
    ax.bar(xs + 0.2, [listed.get(m, 0) for m in xs], width=0.38,
           color="#9ccfe0", label="住民が売りに出した件数")
    ax.plot(xs, [sold.get(m, 0) for m in xs], color=RED, lw=2.0,
            marker="o", ms=5, label="名義が移った件数")
    ax.set_title("③ 持ちかけた件数・売りに出した件数・実際に名義が移った件数",
                 fontsize=12, loc="left")
    ax.set_xlabel("月", fontsize=10.5)
    ax.set_ylabel("件数", fontsize=10.5)
    ax.set_xlim(0.3, months + 0.7)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(axis="y", color="#ededed")
    ax.set_axisbelow(True)
    return _save(fig, out_path)


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE
    town = E.load(os.path.abspath(source))
    print("読み込み:", town.label, "／", town.months, "か月／", len(town.people), "人")
    fig_heard_vs_action(town, os.path.join(HERE, "fig2_heard_vs_action.png"))
    fig_contagion(town, os.path.join(HERE, "fig3_contagion.png"))
    fig_acquirer(town, os.path.join(HERE, "fig4_acquirer.png"))


if __name__ == "__main__":
    main()
