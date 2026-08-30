#!/usr/bin/env python
"""提出動画（1本）を作る。1920x1080・H.264・30fps・音なし。

  python tools/final_video.py

構成（施主 2026-08-30 20:58）:
  ① タイトル（問いを前面に大きく）      6.0秒
  ② 動機（ニセコ・対馬＋法の空白3点）    8.0秒
  ③ 世界設計（背景＝温泉観光都市）      10.0秒
  ④ X社の命題（背景＝高層ビル群）        6.0秒
  ⑤ シミュの流れ（フロー図）             8.0秒
  ⑥ 本編の所有変遷 36か月               10.08秒
  ⑦ 結果の一枚                           3.0秒
  ⑧ 締め「あなたの町は大丈夫？」         5.0秒

コマは PIL で描き、ffmpeg に生のまま流す（v9_video.py の道具を借りる）。
第N月のコマは v9_video.draw_month が描いたものをそのまま使う。
"""

from __future__ import annotations

import functools
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)

import v9_video as V   # noqa: E402
import final_bg as BG  # noqa: E402

W, H, FPS = V.W, V.H, V.FPS

RUN_DIR = os.path.join(
    ROOT, "simulations",
    "2026-08-30_1711_132_field_v9f_pay_more_if_needed_chat")
OUT_MP4 = os.path.join(ROOT, "docs", "submission",
                       "0105_KentaYamakawa_CommunityDefense_Demo.mp4")
FRAMES_DIR = os.path.join(ROOT, "docs", "submission", "frames_final")

FG = V._hex(V.FG)
MUTED = V._hex(V.MUTED)
DIM = V._hex(V.DIM)
ACCENT = V._hex(V.ACCENT)
RED = V._hex("#e0574a")
WHITE = (255, 255, 255)

SEC = {
    "title": 6.0, "why": 8.0, "world": 10.0, "xco": 6.0,
    "flow": 8.0, "body": 10.08, "result": 3.0, "end": 5.0,
}
MONTH_SEC = 0.28


# ---------------------------------------------------------------------------
# 文字を描く道具
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=64)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return V._pil_font(size, bold)


def ease(t0: float, t1: float, t: float) -> float:
    return V._ease(t0, t1, t)


class Canvas:
    """1コマぶんの下地。文字は左／中央／右で置ける。"""

    def __init__(self, bg: Optional[Image.Image] = None) -> None:
        self.img = bg.copy() if bg is not None else Image.new(
            "RGB", (W, H), (0, 0, 0))
        self.d = ImageDraw.Draw(self.img)

    def _fill(self, color, a: float):
        return V._mix(color, max(0.0, min(1.0, a)))

    def text(self, x: float, y: float, s: str, size: int, color, a: float = 1.0,
             bold: bool = False, anchor: str = "l", dy: float = 0.0) -> None:
        if a <= 0.01 or not s:
            return
        f = font(size, bold)
        bb = self.d.textbbox((0, 0), s, font=f)
        w = bb[2] - bb[0]
        if anchor == "c":
            x = x - w / 2 - bb[0]
        elif anchor == "r":
            x = x - w - bb[0]
        else:
            x = x - bb[0]
        self.d.text((x, y + dy * (1 - a)), s, font=f, fill=self._fill(color, a))

    def block(self, x: float, y: float, lines: List[str], size: int, color,
              a: float = 1.0, bold: bool = False, lh: float = 1.45,
              anchor: str = "l", dy: float = 0.0) -> float:
        step = size * lh
        for i, ln in enumerate(lines):
            self.text(x, y + i * step, ln, size, color, a, bold, anchor, dy)
        return y + len(lines) * step

    def rule(self, x0: float, y: float, x1: float, color, a: float = 1.0,
             h: int = 5) -> None:
        if a <= 0.01:
            return
        self.d.rectangle([x0, y, x1, y + h], fill=self._fill(color, a))

    def card(self, x0, y0, x1, y1, a: float = 1.0, edge=None) -> None:
        if a <= 0.01:
            return
        self.d.rectangle([x0, y0, x1, y1], fill=V._mix((22, 26, 33), a),
                         outline=self._fill(edge or (38, 44, 54), a), width=2)

    def dot(self, x, y, r, color, a: float = 1.0) -> None:
        if a <= 0.01:
            return
        self.d.ellipse([x - r, y - r, x + r, y + r], fill=self._fill(color, a))


def fade_out(c: Canvas, t: float, total: float, tail: float = 0.45) -> None:
    """最後を黒に沈める。"""
    k = ease(total - tail, total, t)
    if k > 0.001:
        c.img.paste(Image.blend(c.img, Image.new("RGB", (W, H), (0, 0, 0)), k),
                    (0, 0))


# ---------------------------------------------------------------------------
# ① タイトル
# ---------------------------------------------------------------------------

TITLE_1 = "外的不動産投資への"
TITLE_2 = "コミュニティ自衛"
Q_1 = "コミュニティの創発は、"
Q_2 = "外的な不動産投資を"
Q_3 = "抑制できるのか？"


def scene_title(t: float) -> Canvas:
    c = Canvas()
    total = SEC["title"]
    a0 = ease(0.15, 0.8, t)
    c.text(W / 2, 66, "エージェントシミュレーション", 30, DIM, a0, anchor="c")
    a1 = ease(0.3, 1.1, t)
    c.text(W / 2, 116, TITLE_1 + "　" + TITLE_2, 54, MUTED, a1, True,
           anchor="c", dy=18)
    a2 = ease(1.0, 1.5, t)
    half = int(150 * a2)
    c.rule(W / 2 - half, 205, W / 2 + half, ACCENT, a2)

    a3 = ease(1.2, 2.4, t)
    c.text(W / 2, 262, Q_1, 76, FG, a3, True, anchor="c", dy=26)
    a4 = ease(1.7, 2.9, t)
    c.text(W / 2, 396, Q_2, 92, ACCENT, a4, True, anchor="c", dy=26)
    a5 = ease(2.1, 3.3, t)
    c.text(W / 2, 522, Q_3, 92, ACCENT, a5, True, anchor="c", dy=26)

    a6 = ease(3.2, 4.2, t)
    c.text(W / 2, 700, "制度でも規制でもなく、人が人と話すことだけで。",
           38, MUTED, a6, anchor="c", dy=16)
    a7 = ease(4.1, 5.1, t)
    c.rule(W / 2 - 60, 830, W / 2 + 60, DIM, a7, h=3)
    c.text(W / 2, 862, "やまちゃそ", 40, FG, a7, True, anchor="c")
    fade_out(c, t, total)
    return c


# ---------------------------------------------------------------------------
# ② 動機
# ---------------------------------------------------------------------------

def scene_why(t: float) -> Canvas:
    c = Canvas()
    total = SEC["why"]
    a0 = ease(0.1, 0.7, t)
    c.text(96, 60, "動機", 30, ACCENT, a0)
    c.text(96, 104, "すでに、そうなった町がある", 62, FG, a0, True, dy=16)

    a1 = ease(0.6, 1.5, t)
    a2 = ease(1.0, 1.9, t)
    for a, x0, x1, head, sub, lines in (
        (a1, 96, 936, "ニセコ（北海道）", "令和6年",
         ["居住地が海外の法人・個人が取得した森林は",
          "全国48件。うち32件がニセコ町・倶知安町",
          "など後志のスキー地域。目的の多くは資産保有。"]),
        (a2, 984, 1824, "対馬（長崎県）", "平成19年",
         ["海上自衛隊対馬防備隊の隣接地 約3,000坪が",
          "島民名義で韓国資本に買収された。防衛省が",
          "登記簿を調べたのは、その報道のあと。"]),
    ):
        c.card(x0, 216, x1, 470, a)
        c.text(x0 + 32, 244, head, 40, FG, a, True)
        c.text(x1 - 32, 252, sub, 30, ACCENT, a, anchor="r")
        c.block(x0 + 32, 314, lines, 30, MUTED, a)

    a3 = ease(1.9, 2.7, t)
    c.text(96, 500, "どちらも、買われてから分かった。", 46, RED, a3, True,
           dy=14)

    a4 = ease(2.9, 3.7, t)
    c.rule(96, 596, 1824, (40, 46, 56), a4, h=2)
    c.text(96, 620, "普通の町の土地は、法律で守られていない", 54, FG, a4,
           True, dy=14)

    rows = [
        ("重要土地等調査法（2021年）が見るのは",
         "防衛関係施設の周囲おおむね1000mと国境離島等だけ。普通の町は対象外。"),
        ("外国人の土地取得そのものに",
         "一般的な制限は無い。古い法律は残るが、発動させる命令が置かれていない。"),
        ("先に動いたのは自治体だった。",
         "水源地の売買に事前届出を課す条例が道県にある。それが無い町は無防備。"),
    ]
    y = 720
    for i, (h1, h2) in enumerate(rows):
        a = ease(3.9 + i * 0.35, 4.7 + i * 0.35, t)
        c.dot(112, y + 20, 9, ACCENT, a)
        c.text(146, y, h1, 33, FG, a, True)
        c.text(146, y + 44, h2, 30, MUTED, a)
        y += 100
    fade_out(c, t, total)
    return c


# ---------------------------------------------------------------------------
# ③ 世界設計（背景＝温泉観光都市）
# ---------------------------------------------------------------------------

WORLD_ROWS = [
    ("A市・49人・44区画",
     "町にいる人 35人＋町にいない所有者 14人。建物あり35区画・土地だけ9区画。"),
    ("帳簿に3つの欄がある",
     "土地の所有者／建物の所有者／借りて使う人。今そこを使う人はこの3欄で決まる。"),
    ("評価額は町に公開されている",
     "X社の資金は、A市の不動産の評価額の合計の51%。金額を含む条件を出せる。"),
    ("売った後は、世界が保証する",
     "土地だけ→借地で使い続ける／建物だけ→借家で住み続ける／両方→町を出る。"),
    ("X社＝町の外の海外の不動産投資会社",
     "命題は町の人には見えない。町に与えたのは、会う場所5つと隣近所だけ。"),
]


def scene_world(t: float, bg: Image.Image) -> Canvas:
    c = Canvas(bg)
    total = SEC["world"]
    a0 = ease(0.1, 0.8, t)
    c.text(96, 58, "世界設計", 30, ACCENT, a0)
    c.text(96, 102, "町ひとつを、まるごと言葉で動かす", 64, WHITE, a0, True,
           dy=16)
    c.rule(96, 200, 96 + int(260 * ease(0.5, 1.2, t)), ACCENT,
           ease(0.5, 1.2, t))

    y = 268
    for i, (h1, h2) in enumerate(WORLD_ROWS):
        a = ease(1.0 + i * 0.7, 2.0 + i * 0.7, t)
        c.rule(100, y + 6, 106, (0, 0, 0), 0)     # 位置合わせ用（描かない）
        c.d.rectangle([100, y + 4, 106, y + 92],
                      fill=V._mix(ACCENT, max(0.0, min(1.0, a))))
        c.text(140, y, h1, 42, WHITE, a, True, dy=12)
        c.text(140, y + 56, h2, 31, (200, 208, 218), a, dy=12)
        y += 150
    fade_out(c, t, total)
    return c


# ---------------------------------------------------------------------------
# ④ X社の命題（背景＝高層ビル群）
# ---------------------------------------------------------------------------

CLOSERS = "、。」』）,."


def kinsoku(lines: List[str]) -> List[str]:
    """行頭の句読点・閉じ括弧を前の行の末尾へ送る（行頭禁則）。"""
    out = list(lines)
    for i in range(1, len(out)):
        while out[i] and out[i][0] in CLOSERS:
            out[i - 1] += out[i][0]
            out[i] = out[i][1:]
    return [ln for ln in out if ln]


def mandate_text() -> str:
    p = os.path.join(ROOT, "src", "field_v9f.py")
    with open(p, encoding="utf-8") as f:
        src = f.read()
    import re
    m = re.search(r"ACQUIRER_MANDATE_V9F = \((.*?)\n\)", src, re.S)
    if not m:
        raise SystemExit("命題が src/field_v9f.py から読めない")
    return "".join(re.findall(r'"([^"]*)"', m.group(1)))


def scene_xco(t: float, bg: Image.Image, mandate: str) -> Canvas:
    c = Canvas(bg)
    total = SEC["xco"]
    a0 = ease(0.1, 0.7, t)
    c.text(96, 62, "X社", 30, RED, a0)
    c.text(96, 106, "町の外から来る、海外の不動産投資会社", 58, WHITE, a0,
           True, dy=14)
    a1 = ease(0.6, 1.2, t)
    c.text(96, 214, "X社への命題（原文・町の人には見えない）", 32, RED, a1)

    lines = kinsoku(V.wrap_cjk(mandate, 25))
    c.d.rectangle([96, 282, 104, 282 + len(lines) * 74],
                  fill=V._mix(RED, max(0.0, min(1.0, ease(0.9, 1.5, t)))))
    y = 276
    for i, ln in enumerate(lines):
        a = ease(1.1 + i * 0.28, 1.9 + i * 0.28, t)
        c.text(140, y, ln, 46, WHITE, a, True, dy=10)
        y += 74
    a2 = ease(4.0, 4.8, t)
    c.text(96, 986, "X社の提示には、世界が必ず1行を添える —"
           "「私どもは海外の不動産投資会社です。」国名なし・警戒をあおる語なし。",
           28, (198, 206, 216), a2)
    fade_out(c, t, total)
    return c


# ---------------------------------------------------------------------------
# ⑤ シミュの流れ（フロー図）
# ---------------------------------------------------------------------------

def icon_letter(d, x, y, s, col, a):
    col = V._mix(col, a)
    d.rectangle([x - s, y - s * 0.66, x + s, y + s * 0.66], outline=col,
                width=5)
    d.line([(x - s, y - s * 0.66), (x, y + s * 0.10)], fill=col, width=5)
    d.line([(x + s, y - s * 0.66), (x, y + s * 0.10)], fill=col, width=5)


def icon_talk(d, x, y, s, col, a):
    col = V._mix(col, a)
    d.rounded_rectangle([x - s, y - s * 0.85, x + s * 0.25, y + s * 0.10],
                        radius=12, outline=col, width=5)
    d.polygon([(x - s * 0.62, y + s * 0.10), (x - s * 0.30, y + s * 0.10),
               (x - s * 0.56, y + s * 0.48)], fill=col)
    d.rounded_rectangle([x - s * 0.20, y - s * 0.10, x + s, y + s * 0.72],
                        radius=12, outline=col, width=5)


def icon_list(d, x, y, s, col, a):
    col = V._mix(col, a)
    d.rectangle([x - s * 0.80, y - s * 0.85, x + s * 0.80, y + s * 0.85],
                outline=col, width=5)
    for i in range(4):
        yy = y - s * 0.52 + i * s * 0.40
        d.rectangle([x - s * 0.56, yy - 7, x - s * 0.30, yy + 7], outline=col,
                    width=4)
        d.line([(x - s * 0.16, yy), (x + s * 0.58, yy)], fill=col, width=4)


def icon_yesno(d, x, y, s, col, a):
    col = V._mix(col, a)
    d.ellipse([x - s * 0.92, y - s * 0.52, x - s * 0.08, y + s * 0.32],
              outline=col, width=6)
    d.line([(x + s * 0.20, y - s * 0.48), (x + s * 0.90, y + s * 0.30)],
           fill=col, width=6)
    d.line([(x + s * 0.90, y - s * 0.48), (x + s * 0.20, y + s * 0.30)],
           fill=col, width=6)


def icon_book(d, x, y, s, col, a):
    col = V._mix(col, a)
    d.rectangle([x - s * 0.90, y - s * 0.70, x + s * 0.90, y + s * 0.70],
                outline=col, width=5)
    d.line([(x, y - s * 0.70), (x, y + s * 0.70)], fill=col, width=4)
    for i in range(3):
        yy = y - s * 0.34 + i * s * 0.34
        d.line([(x - s * 0.70, yy), (x - s * 0.16, yy)], fill=col, width=3)
        d.line([(x + s * 0.16, yy), (x + s * 0.70, yy)], fill=col, width=3)


FLOW = [
    (icon_letter, "①　X社が提示する", ["持ち主ごとに、", "金額を含む条件の手紙"]),
    (icon_talk, "②　場での会話", ["5つの場＋隣近所で", "ひと言を交わす"]),
    (icon_list, "③　出品の4択", ["出さない／土地だけ／", "建物だけ／両方"]),
    (icon_yesno, "④　売る／売らない", ["提示が来た人だけが答える", "理由を一言（40字）"]),
    (icon_book, "⑤　登記の更新", ["売ると所有権が移る", "売った人も町に残れる"]),
]


def scene_flow(t: float) -> Canvas:
    c = Canvas()
    total = SEC["flow"]
    d = c.d
    a0 = ease(0.1, 0.7, t)
    c.text(96, 58, "シミュレーションの流れ", 30, ACCENT, a0)
    c.text(96, 102, "毎月、この順に一巡する", 62, FG, a0, True, dy=14)
    a1 = ease(0.5, 1.2, t)
    c.text(1824, 118, "1単位＝1か月　×　36か月＝3年", 38, ACCENT, a1,
           True, anchor="r")

    x0, gap, bw = 78, 20, 328
    top, bh = 250, 470
    for i, (icon, head, body) in enumerate(FLOW):
        a = ease(0.9 + i * 0.55, 1.7 + i * 0.55, t)
        x = x0 + i * (bw + gap)
        c.card(x, top, x + bw, top + bh, a, edge=(46, 54, 66))
        icon(d, x + bw / 2, top + 118, 62, ACCENT, max(0.0, min(1.0, a)))
        c.text(x + bw / 2, top + 214, head, 33, FG, a, True, anchor="c")
        c.block(x + bw / 2, top + 286, body, 26, MUTED, a, anchor="c")
        if i < len(FLOW) - 1:
            aa = ease(1.25 + i * 0.55, 1.85 + i * 0.55, t)
            if aa > 0.01:
                ax = x + bw + 3
                ay = top + 118
                col = V._mix(ACCENT, aa)
                d.line([(ax, ay), (ax + gap - 4, ay)], fill=col, width=5)
                d.polygon([(ax + gap + 8, ay), (ax + gap - 8, ay - 11),
                           (ax + gap - 8, ay + 11)], fill=col)

    a2 = ease(4.6, 5.5, t)
    c.d.rectangle([78, 790, 1842, 796], fill=V._mix((40, 46, 56),
                                                    max(0.0, min(1.0, a2))))
    c.text(78, 826, "この一巡を36回くり返す。", 40, FG, a2, True)
    c.text(78, 892, "「〜なら売る」という条件分岐・確率・閾値は1行も置いていない。"
           "X社も町の人も、言葉で考えて言葉で動く。", 32, MUTED,
           ease(5.0, 5.9, t))
    fade_out(c, t, total)
    return c


# ---------------------------------------------------------------------------
# ⑦ 結果の一枚
# ---------------------------------------------------------------------------

RESULT_COLS = [
    ("成約", "11", "件"),
    ("売った人", "9", "人"),
    ("土地の面積", "13.7", "%"),
    ("評価額", "8.3", "%"),
    ("X社が使った資金", "28", "%"),
    ("町を出た人", "8", "人"),
]


def scene_result(t: float) -> Canvas:
    c = Canvas()
    total = SEC["result"]
    a0 = ease(0.05, 0.5, t)
    c.text(W / 2, 118, "3年（36か月）のあと", 66, FG, a0, True, anchor="c",
           dy=14)
    c.rule(W / 2 - 130, 226, W / 2 + 130, ACCENT, ease(0.3, 0.7, t))

    cols = 3
    cw, ch = 540, 246
    x0 = (W - (cols * cw + (cols - 1) * 40)) / 2
    for i, (lab, num, unit) in enumerate(RESULT_COLS):
        a = ease(0.5 + i * 0.13, 1.1 + i * 0.13, t)
        cx = x0 + (i % cols) * (cw + 40)
        cy = 300 + (i // cols) * (ch + 34)
        c.card(cx, cy, cx + cw, cy + ch, a, edge=(46, 54, 66))
        c.text(cx + cw / 2, cy + 28, lab, 32, MUTED, a, anchor="c")
        f = font(96, True)
        bb = c.d.textbbox((0, 0), num, font=f)
        fu = font(40, True)
        bu = c.d.textbbox((0, 0), unit, font=fu)
        tw = (bb[2] - bb[0]) + 12 + (bu[2] - bu[0])
        nx = cx + cw / 2 - tw / 2
        c.text(nx, cy + 92, num, 96, ACCENT, a, True)
        c.text(nx + (bb[2] - bb[0]) + 12, cy + 152, unit, 40, ACCENT, a, True)

    a1 = ease(1.7, 2.3, t)
    c.text(W / 2, 1002, "44区画のうち、X社の所有権が及んだのは11。"
           "誰にも「警戒しろ」とは言っていない。", 34, MUTED, a1, anchor="c")
    fade_out(c, t, total)
    return c


# ---------------------------------------------------------------------------
# ⑧ 締め
# ---------------------------------------------------------------------------

def scene_end(t: float) -> Canvas:
    c = Canvas()
    a0 = ease(0.4, 2.2, t)
    c.text(W / 2, 396, "あなたの町は大丈夫？", 110, FG, a0, True, anchor="c",
           dy=24)
    a1 = ease(2.4, 3.2, t)
    c.rule(W / 2 - 110, 596, W / 2 + 110, ACCENT, a1, h=4)
    a2 = ease(2.8, 3.6, t)
    c.text(W / 2, 668, "4パターンの全記録", 30, MUTED, a2, anchor="c")
    c.text(W / 2, 716,
           "https://yamachas0.github.io/community-defense-sim-report/"
           "reports.html", 30, ACCENT, a2, anchor="c")
    fade_out(c, t, SEC["end"], tail=0.5)
    return c


# ---------------------------------------------------------------------------
# ⑥ 本編（36か月）
# ---------------------------------------------------------------------------

def month_frames() -> Tuple[List[str], List[int]]:
    fdir = os.path.join(RUN_DIR, "video_frames")
    paths = []
    for m in range(1, 37):
        p = os.path.join(fdir, f"f{m:03d}_m{m:02d}.png")
        if not os.path.exists(p):
            raise SystemExit(f"第{m}月のコマが無い: {p}")
        paths.append(p)
    with open(os.path.join(RUN_DIR, "transfers.json"), encoding="utf-8") as f:
        ev = sorted({int(r["step"]) for r in json.load(f)})
    return paths, ev


def month_image(path: str, flash: bool) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if img.size != (W, H):
        img = img.resize((W, H), Image.LANCZOS)
    if flash:
        # 赤い枠だけで強調する（「第N月」の特大文字を隠さない）。
        d = ImageDraw.Draw(img)
        for k in range(14):
            d.rectangle([k, k, W - 1 - k, H - 1 - k], outline=RED)
    return img


# ---------------------------------------------------------------------------
# 組み立て
# ---------------------------------------------------------------------------

def build_frames(bg_town: Image.Image, bg_xco: Image.Image, mandate: str,
                 save: Optional[Dict[str, str]] = None):
    """全コマを順に返す generator（1コマ＝PIL Image）。"""
    paths, ev = month_frames()
    saved = set()

    def maybe_save(key: str, img: Image.Image) -> None:
        if save and key in save and key not in saved:
            img.save(save[key])
            saved.add(key)

    plan = [
        ("title", SEC["title"], lambda t: scene_title(t).img, 4.6),
        ("why", SEC["why"], lambda t: scene_why(t).img, 7.0),
        ("world", SEC["world"], lambda t: scene_world(t, bg_town).img, 8.8),
        ("xco", SEC["xco"], lambda t: scene_xco(t, bg_xco, mandate).img, 5.0),
        ("flow", SEC["flow"], lambda t: scene_flow(t).img, 6.8),
        ("result", SEC["result"], lambda t: scene_result(t).img, 2.2),
        ("end", SEC["end"], lambda t: scene_end(t).img, 3.8),
    ]
    order = ["title", "why", "world", "xco", "flow", "__body__", "result",
             "end"]
    by_key = {p[0]: p for p in plan}

    for key in order:
        if key == "__body__":
            per = max(1, round(MONTH_SEC * FPS))
            for m, p in enumerate(paths, start=1):
                plain = month_image(p, False)
                hot = month_image(p, True) if m in ev else None
                for i in range(per):
                    img = hot if (hot is not None and i < 3) else plain
                    yield img
                if m == 36:
                    maybe_save("m36", plain)
                if m == ev[0]:
                    maybe_save("m_event", hot or plain)
            continue
        _, sec, fn, shot_t = by_key[key]
        n = int(round(sec * FPS))
        for i in range(n):
            t = i / FPS
            img = fn(t)
            if abs(t - shot_t) < 0.5 / FPS:
                maybe_save(key, img)
            yield img


def encode(frames, out: str, crf: int) -> None:
    import subprocess
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    cmd = [V._bin(), "-y", "-loglevel", "error", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-an", "-crf", str(crf)] + V.X264 + [os.path.abspath(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         **V._noshow())
    assert p.stdin is not None
    n = 0
    for img in frames:
        p.stdin.write(img.tobytes())
        n += 1
        if n % 300 == 0:
            print(f"  {n} コマ", flush=True)
    p.stdin.close()
    if p.wait() != 0:
        raise SystemExit("ffmpeg が失敗した")
    print(f"  {n} コマ（{n / FPS:.2f}秒）", flush=True)


def main() -> int:
    os.makedirs(FRAMES_DIR, exist_ok=True)
    town, xco = BG.build()
    bg_town = Image.open(BG.to_grey_dim(
        town, os.path.join(BG.ASSETS, "bg_town_dim.png"), 0.35)).convert("RGB")
    bg_xco = Image.open(BG.to_grey_dim(
        xco, os.path.join(BG.ASSETS, "bg_xco_dim.png"), 0.32)).convert("RGB")
    mandate = mandate_text()
    print("命題:", mandate)

    save = {k: os.path.join(FRAMES_DIR, f"final_{k}.png")
            for k in ("title", "why", "world", "xco", "flow", "m36",
                      "m_event", "result", "end")}

    crf = 21
    for _ in range(4):
        encode(build_frames(bg_town, bg_xco, mandate, save), OUT_MP4, crf)
        mb = os.path.getsize(OUT_MP4) / 1e6
        print(f"crf={crf} size={mb:.1f}MB", flush=True)
        if mb <= 25.0:
            break
        crf += 4
    print(V.report(OUT_MP4))
    for k, v in sorted(save.items()):
        print("frame", v, os.path.exists(v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
