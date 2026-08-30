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

K = 1.4                      # 0.7倍速（読める速さ）にする係数
SEC = {
    "title": 6.0 * K, "why": 8.0 * K, "world": 8.0 * K, "xco": 6.0 * K,
    "flow": 6.5 * K, "m0": 2.5, "black": 1.5, "end": 4.4 * K,
}
MONTH_SEC = 0.28


# ---------------------------------------------------------------------------
# 文字を描く道具
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=64)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return V._pil_font(size, bold)


def ease(t0: float, t1: float, t: float) -> float:
    """演出の時刻は K 倍に引き伸ばす（全体を 0.7 倍速にするため）。"""
    return V._ease(t0 * K, t1 * K, t)


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

TITLE_1 = "外的不動産買収への"
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

def scene_why(t: float, bg: Image.Image) -> Canvas:
    c = Canvas(bg)
    total = SEC["why"]
    a0 = ease(0.1, 0.7, t)
    c.text(96, 84, "課題感", 34, ACCENT, a0)
    c.text(96, 140, "すでに、そうなった町がある", 82, WHITE, a0, True, dy=18)
    c.rule(96, 264, 96 + int(300 * ease(0.5, 1.2, t)), ACCENT,
           ease(0.5, 1.2, t))

    rows = [
        ("ニセコ（北海道）",
         "令和6年、居住地が海外の法人・個人が取得した森林は全国48件。"
         "うち32件が後志のスキー地域。"),
        ("対馬（長崎県）",
         "平成19年、海上自衛隊の隣接地 約3,000坪が島民名義で"
         "韓国資本に買収された。"),
        ("どちらも、買われてから分かった",
         "普通の町の土地は、法律で守られていない。"),
    ]
    y = 380
    for i, (h1, h2) in enumerate(rows):
        a = ease(1.4 + i * 1.5, 2.6 + i * 1.5, t)
        col = RED if i == 2 else ACCENT
        c.d.rectangle([100, y + 6, 108, y + 116],
                      fill=V._mix(col, max(0.0, min(1.0, a))))
        c.text(150, y, h1, 54, WHITE, a, True, dy=14)
        c.text(150, y + 76, h2, 34, (198, 206, 216), a, dy=14)
        y += 190
    fade_out(c, t, total)
    return c


# ---------------------------------------------------------------------------
# ③ 世界設計（背景＝温泉観光都市）
# ---------------------------------------------------------------------------

WORLD_ROWS = [
    ("A市・49人・44区画",
     "町にいる人 35人＋町にいない所有者 14人。建物あり35区画・土地だけ9区画。"),
    ("評価額は公開されている",
     "44区画すべてに評価額を置いた。町の合計 22億9,770万円。"),
    ("X社＝海外の不動産投資会社",
     "資金は町の全評価額の51%＝11億7,180万円。命題は町の人には見えない。"),
]


def scene_world(t: float, bg: Image.Image) -> Canvas:
    c = Canvas(bg)
    total = SEC["world"]
    a0 = ease(0.1, 0.8, t)
    c.text(96, 84, "世界設計", 34, ACCENT, a0)
    c.text(96, 140, "町ひとつを、まるごと言葉で動かす", 82, WHITE, a0, True,
           dy=18)
    c.rule(96, 264, 96 + int(300 * ease(0.5, 1.2, t)), ACCENT,
           ease(0.5, 1.2, t))

    y = 400
    for i, (h1, h2) in enumerate(WORLD_ROWS):
        a = ease(1.4 + i * 1.5, 2.6 + i * 1.5, t)
        c.d.rectangle([100, y + 6, 108, y + 116],
                      fill=V._mix(ACCENT, max(0.0, min(1.0, a))))
        c.text(150, y, h1, 54, WHITE, a, True, dy=14)
        c.text(150, y + 76, h2, 34, (200, 208, 218), a, dy=14)
        y += 190
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


MANDATE_LINES = [
    "合法な手段で、A市の不動産の所有権を取得せよ。",
    "土地の面積では、A市の過半を最後まで目指すこと。",
    "買えるまで、金額を含む条件を変えて働きかけ続けること。毎月動け。",
]


def scene_xco(t: float, bg: Image.Image) -> Canvas:
    c = Canvas(bg)
    total = SEC["xco"]
    a0 = ease(0.1, 0.7, t)
    c.text(96, 84, "X社", 34, RED, a0)
    c.text(96, 140, "海外の不動産投資会社", 76, WHITE, a0,
           True, dy=16)
    a1 = ease(0.6, 1.2, t)
    c.text(96, 292, "X社への命題（抜粋・町の人には見えない）", 34, RED, a1)

    c.d.rectangle([96, 380, 106, 380 + len(MANDATE_LINES) * 130],
                  fill=V._mix(RED, max(0.0, min(1.0, ease(0.9, 1.5, t)))))
    y = 380
    for i, ln in enumerate(MANDATE_LINES):
        a = ease(1.2 + i * 0.9, 2.2 + i * 0.9, t)
        c.text(150, y, ln, 52, WHITE, a, True, dy=12)
        y += 130
    a2 = ease(4.3, 5.1, t)
    c.text(96, 880, "X社の提示には、世界が必ず1行を添える —"
           "「私どもは海外の不動産投資会社です。」", 34, (198, 206, 216), a2)
    c.text(96, 940, "国名なし・警戒をあおる語なし。", 34, (198, 206, 216),
           ease(4.7, 5.5, t))
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
    (icon_letter, "①", "金額付きの手紙"),
    (icon_talk, "②", "場での会話"),
    (icon_list, "③", "出品の4択"),
    (icon_yesno, "④", "売る／売らない"),
    (icon_book, "⑤", "登記の更新"),
]


def scene_flow(t: float) -> Canvas:
    c = Canvas()
    total = SEC["flow"]
    d = c.d
    a0 = ease(0.1, 0.7, t)
    c.text(96, 84, "毎月の手順", 34, ACCENT, a0)
    c.text(96, 140, "命題を1つ渡し、同じ5手順を36回まわす", 76, FG, a0, True,
           dy=16)

    x0, gap, bw = 78, 20, 328
    top, bh = 320, 420
    for i, (icon, num, head) in enumerate(FLOW):
        a = ease(0.9 + i * 0.7, 1.7 + i * 0.7, t)
        x = x0 + i * (bw + gap)
        c.card(x, top, x + bw, top + bh, a, edge=(46, 54, 66))
        c.text(x + bw / 2, top + 40, num, 46, ACCENT, a, True, anchor="c")
        icon(d, x + bw / 2, top + 208, 74, ACCENT, max(0.0, min(1.0, a)))
        c.text(x + bw / 2, top + 316, head, 40, FG, a, True, anchor="c")
        if i < len(FLOW) - 1:
            aa = ease(1.25 + i * 0.7, 1.85 + i * 0.7, t)
            if aa > 0.01:
                ax = x + bw + 3
                ay = top + 208
                col = V._mix(ACCENT, aa)
                d.line([(ax, ay), (ax + gap - 4, ay)], fill=col, width=5)
                d.polygon([(ax + gap + 8, ay), (ax + gap - 8, ay - 11),
                           (ax + gap - 8, ay + 11)], fill=col)

    a2 = ease(4.4, 5.2, t)
    c.d.rectangle([78, 828, 1842, 834], fill=V._mix((40, 46, 56),
                                                    max(0.0, min(1.0, a2))))
    c.text(78, 872, "1単位＝1か月　×　36か月＝3年", 46, FG, a2, True)
    c.text(78, 946, "全員が LLM。行動を決めるコードはゼロ。", 34, MUTED,
           ease(4.8, 5.6, t))
    fade_out(c, t, total)
    return c


# ---------------------------------------------------------------------------
# ⑧ 締め
# ---------------------------------------------------------------------------

def scene_black(t: float) -> Canvas:
    """締めの前の溜め（黒・無文字）。"""
    return Canvas()


def scene_end(t: float, bg: Image.Image) -> Canvas:
    # 黒の溜めから背景ごと明けていく。
    k = max(0.0, min(1.0, V._ease(0.0, 1.2, t)))
    c = Canvas(Image.blend(Image.new("RGB", (W, H), (0, 0, 0)), bg, k))
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

LABEL = "買い手が意図を明かさず、町に会話がある場合"
KEY = {"土地": "land", "建物": "building"}


def month0_rows(run) -> List[dict]:
    """第0月（開始時点）の帳簿＝第1月の帳簿から第1月の移転を戻したもの。"""
    import copy
    rows = copy.deepcopy(run.ledger[1])
    by_parcel = {r["parcel"]: r for r in rows}
    for t in run.transfers_by_month.get(1, []):
        r = by_parcel.get(t["parcel"])
        if r is None:
            continue
        for kind in t.get("moved", []):
            k = KEY.get(str(kind))
            if k and str(kind) in (t.get("before") or {}):
                r[k] = t["before"][str(kind)]
        if t.get("was_user"):
            r["user"] = t.get("name")
    return rows


@functools.lru_cache(maxsize=1)
def render_months() -> Tuple[List[str], List[int], str]:
    """本編36か月＋第0月のコマを描き直す（左上の版ラベルを出さない）。"""
    run = V.Run(RUN_DIR, LABEL)
    fdir = os.path.join(RUN_DIR, "video_frames")
    os.makedirs(fdir, exist_ok=True)
    paths = []
    for m in range(1, 37):
        p = os.path.join(fdir, f"f{m:03d}_m{m:02d}.png")
        V.draw_month(run, m, p)
        paths.append(p)

    run.ledger[0] = month0_rows(run)
    run.x_parcels[0] = []
    run.offers_cum[0] = 0
    p0 = os.path.join(fdir, "f000_m00.png")
    V.draw_month(run, 0, p0, subtitle="開始時点")
    return tuple(paths), tuple(sorted(run.transfers_by_month)), p0


LEGEND = "赤に変わった区画＝X社の所有になったところ"


def month_hold(path: str, t: float) -> Image.Image:
    """本編に入る前の待機コマ（第0月＝開始時点の地図＋読み方の一言）。"""
    img = month_image(path, False)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 930, W, H], fill=(0, 0, 0))
    a = max(0.0, min(1.0, V._ease(0.3, 1.0, t)))
    d.rectangle([96, 986, 140, 1030], fill=V._mix(RED, a),
                outline=V._mix((255, 255, 255), a), width=2)
    f = V._pil_font(44, True)
    d.text((166, 982), LEGEND, font=f, fill=V._mix((255, 255, 255), a))
    f2 = V._pil_font(30, False)
    a2 = max(0.0, min(1.0, V._ease(0.9, 1.6, t)))
    d.text((166, 1040 - 6), "これから36か月を早回しで見る。", font=f2,
           fill=V._mix((198, 206, 216), a2))
    return img


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

def build_frames(bg_town: Image.Image, bg_xco: Image.Image,
                 bg_why: Image.Image, bg_end: Image.Image,
                 save: Optional[Dict[str, str]] = None):
    """全コマを順に返す generator（1コマ＝PIL Image）。"""
    paths, ev, p0 = render_months()
    saved = set()

    def maybe_save(key: str, img: Image.Image) -> None:
        if save and key in save and key not in saved:
            img.save(save[key])
            saved.add(key)

    plan = [
        ("title", SEC["title"], lambda t: scene_title(t).img, 6.4),
        ("why", SEC["why"], lambda t: scene_why(t, bg_why).img, 9.8),
        ("world", SEC["world"], lambda t: scene_world(t, bg_town).img, 9.8),
        ("xco", SEC["xco"], lambda t: scene_xco(t, bg_xco).img, 8.0),
        ("flow", SEC["flow"], lambda t: scene_flow(t).img, 8.6),
        ("m0", SEC["m0"], lambda t: month_hold(p0, t), 2.0),
        ("black", SEC["black"], lambda t: scene_black(t).img, 0.5),
        ("end", SEC["end"], lambda t: scene_end(t, bg_end).img, 4.6),
    ]
    order = ["title", "why", "world", "xco", "flow", "m0", "__body__",
             "black", "end"]
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
                maybe_save("m00" if key == "m0" else key, img)
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
    def owner(name: str, alpha: float, mode: str = "cover",
              bias: float = 0.5, box=None, anchor: float = 0.5,
              sat=None) -> Image.Image:
        src = os.path.join(BG.OWNER, name + ".png")
        out = os.path.join(BG.OWNER, name + "_dim.png")
        return Image.open(BG.owner_bg(src, out, alpha, mode, bias, box,
                                      anchor, sat)).convert("RGB")

    bg_town = owner("town_photo", 0.35, "cover", 0.45)
    bg_xco = owner("xco_building", 0.15, "cover", 0.18)
    # 地図は文字入りなので、注記の文字を外して列島だけを右側に敷く
    # （画面の文字と地図の文字が重なって読めなくなるため）。
    bg_why = owner("japan_map", 0.60, "contain", box=(318, 18, 1182, 1012),
                   anchor=0.86)
    bg_end = owner("ending_bg", 0.40, "cover", 0.5, sat=0.5)
    save = {k: os.path.join(FRAMES_DIR, f"final_{k}.png")
            for k in ("title", "why", "world", "xco", "flow", "m00", "m36",
                      "m_event", "end")}

    crf = 21
    for _ in range(4):
        encode(build_frames(bg_town, bg_xco, bg_why, bg_end, save),
               OUT_MP4, crf)
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
