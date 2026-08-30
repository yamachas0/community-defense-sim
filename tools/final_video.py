#!/usr/bin/env python
"""提出動画（1本）を作る。1920x1080・H.264・30fps・音なし。

  python tools/final_video.py

構成（施主 2026-08-30 22:11・スライド最終版に合わせる）:
  ① タイトル（正式タイトル＋問い）
  ② 外的不動産買収の事例（日本地図＝ニセコ・対馬）
  ③ 外的不動産買収シミュレーション（X社 → A市）
  ④ シミュレーションフロー（5手順 ▶ ×36か月）
  ⑤ 所有の変遷（第0月で待機 → 36か月 → 第36月で静止）
  ⑥ 締め（黒 1.5秒 →「あなたの町は大丈夫？」）

コマは PIL で描き、ffmpeg に生のまま流す（v9_video.py の道具を借りる）。
第N月のコマは v9_video.draw_month が描いたものをそのまま使う。
"""

from __future__ import annotations

import functools
import os
import sys
from typing import Dict, List, Optional, Tuple

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
BODY = (200, 208, 218)

K = 1.4                      # 0.7倍速（読める速さ）にする係数
AUDIO_DIR = os.path.join(ROOT, "docs", "submission", "video_assets", "audio")
BGM = os.path.join(AUDIO_DIR, "bgm_pad.wav")
SILENT_MP4 = os.path.join(ROOT, "docs", "submission", "video_assets",
                          "_silent_v7.mp4")

# ナレーション（v7・全部 Kore で一度に撮ったもの）
NAR = {k: os.path.join(AUDIO_DIR, f"nar_v7_{k}.wav")
       for k in ("01", "02", "03", "04", "05", "06", "07", "08")}
LEAD = 0.4                   # シーン頭からナレーションが始まるまでの間


def wav_sec(path: str) -> float:
    import wave
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


NAR_SEC = {k: wav_sec(v) for k, v in NAR.items()}

SEC = {
    "title": 10.0, "why": 11.0, "world": 9.9, "flow": 13.2,
    "m0": 2.0, "m36": 4.8, "summary": 10.0, "black": 1.5, "end": 5.6,
}
MONTH_SEC = 0.34
# ナレーションが収まる長さへ伸ばす（声が切れない）。
for _key, _num, _tail in (("title", "01", 0.8), ("why", "02", 0.8),
                          ("world", "03", 0.8), ("flow", "04", 0.8),
                          ("m36", "06", 0.8), ("summary", "07", 0.6)):
    SEC[_key] = max(SEC[_key], LEAD + NAR_SEC[_num] + _tail)


# ---------------------------------------------------------------------------
# 文字を描く道具
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=64)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return V._pil_font(size, bold)


def ease(t0: float, t1: float, t: float) -> float:
    """演出の時刻は K 倍に引き伸ばす（全体を 0.7 倍速にするため）。"""
    return V._ease(t0 * K, t1 * K, t)


def cover(src: Image.Image, w: int, h: int) -> Image.Image:
    """縦横比を保って箱いっぱいに切り抜く。"""
    sw, sh = src.size
    k = max(w / sw, h / sh)
    im = src.resize((max(1, int(sw * k)), max(1, int(sh * k))),
                    Image.LANCZOS)
    x = (im.size[0] - w) // 2
    y = (im.size[1] - h) // 2
    return im.crop((x, y, x + w, y + h))


def contain(src: Image.Image, w: int, h: int) -> Image.Image:
    """縦横比を保って箱に収める（欠けない）。余白は黒。"""
    sw, sh = src.size
    k = min(w / sw, h / sh)
    im = src.resize((max(1, int(sw * k)), max(1, int(sh * k))),
                    Image.LANCZOS)
    out = Image.new("RGB", (w, h), (0, 0, 0))
    out.paste(im, ((w - im.size[0]) // 2, (h - im.size[1]) // 2))
    return out


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

    def photo(self, im: Image.Image, x: int, y: int, a: float = 1.0) -> None:
        if a <= 0.01:
            return
        if a < 0.999:
            im = Image.blend(Image.new("RGB", im.size, (0, 0, 0)), im, a)
        self.img.paste(im, (x, y))


def fade_out(c: Canvas, t: float, total: float, tail: float = 0.45) -> None:
    """最後を黒に沈める。"""
    k = ease(total - tail, total, t)
    if k > 0.001:
        c.img.paste(Image.blend(c.img, Image.new("RGB", (W, H), (0, 0, 0)), k),
                    (0, 0))


# ---------------------------------------------------------------------------
# ① タイトル
# ---------------------------------------------------------------------------

TITLE_SLIDE = os.path.join(ROOT, "docs", "submission", "video_assets",
                           "title_slide_p1.png")


def scene_title(t: float, slide: Image.Image) -> Canvas:
    """スライド P1（最新デザイン）をそのまま冒頭に置く。"""
    k = max(0.0, min(1.0, V._ease(0.0, 1.0, t)))
    c = Canvas(Image.blend(Image.new("RGB", (W, H), (0, 0, 0)), slide, k))
    total = SEC["title"]
    kk = max(0.0, min(1.0, V._ease(total - 0.6, total, t)))
    if kk > 0.001:
        c.img.paste(Image.blend(c.img, Image.new("RGB", (W, H), (0, 0, 0)),
                                kk), (0, 0))
    return c


# ---------------------------------------------------------------------------
# ② 外的不動産買収の事例（日本地図）
# ---------------------------------------------------------------------------

WHY_ROWS = [
    ("ニセコ（北海道）",
     "令和6年、海外資本の森林取得48件のうち32件が後志のスキー地域。",
     ACCENT),
    ("対馬（長崎県）",
     "平成19年、海上自衛隊の隣接地 約3,000坪が韓国資本に。",
     ACCENT),
    ("どちらも、買われてから分かった。", "", RED),
]


def scene_why(t: float, map_img: Image.Image) -> Canvas:
    c = Canvas()
    total = SEC["why"]
    a0 = ease(0.1, 0.7, t)
    c.text(96, 92, "外的不動産買収の事例", 72, WHITE, a0, True, dy=18)
    aw = ease(0.4, 1.3, t)
    c.rule(96, 200, 96 + int(300 * aw), ACCENT, aw)
    # 地図はコンテンツ。暗くせず、そのまま右に置く。
    c.photo(map_img, 1050, 90, ease(0.5, 1.6, t))

    y = 380
    for i, (h1, h2, col) in enumerate(WHY_ROWS):
        a = ease(1.6 + i * 1.3, 2.7 + i * 1.3, t)
        c.d.rectangle([100, y + 8, 108, y + (104 if h2 else 62)],
                      fill=V._mix(col, max(0.0, min(1.0, a))))
        c.text(148, y, h1, 50, WHITE, a, True, dy=14)
        if h2:
            c.text(148, y + 74, h2, 30, BODY, a, dy=14)
        y += 170 if h2 else 130
    fade_out(c, t, total)
    return c


# ---------------------------------------------------------------------------
# ③ 外的不動産買収シミュレーション（X社 → A市）
# ---------------------------------------------------------------------------

WORLD_1 = "海外不動産投資会社X社が、国内地方都市A市の"
WORLD_2 = "不動産の過半買収を試みる。"


def scene_world(t: float, xco: Image.Image, town: Image.Image) -> Canvas:
    c = Canvas()
    total = SEC["world"]
    a0 = ease(0.1, 0.7, t)
    c.text(96, 74, "外的不動産買収シミュレーション", 40, ACCENT, a0)
    a1 = ease(0.5, 1.5, t)
    c.text(W / 2, 168, WORLD_1, 62, WHITE, a1, True, anchor="c", dy=20)
    a2 = ease(0.9, 1.9, t)
    c.text(W / 2, 262, WORLD_2, 62, WHITE, a2, True, anchor="c", dy=20)

    px, py, pw, ph = 150, 470, 560, 380
    qx = W - px - pw
    a3 = ease(2.0, 2.9, t)
    c.photo(xco, px, py, a3)
    c.d.rectangle([px, py, px + pw, py + ph], outline=V._mix(
        (70, 78, 92), max(0.0, min(1.0, a3))), width=3)
    c.text(px + pw / 2, py + ph + 26, "X社", 46, WHITE, a3, True, anchor="c")
    c.text(px + pw / 2, py + ph + 88, "X社は海外不動産投資会社",
           30, BODY, a3, anchor="c")

    a4 = ease(2.4, 3.3, t)
    c.photo(town, qx, py, a4)
    c.d.rectangle([qx, py, qx + pw, py + ph], outline=V._mix(
        (70, 78, 92), max(0.0, min(1.0, a4))), width=3)
    c.text(qx + pw / 2, py + ph + 26, "A市", 46, WHITE, a4, True, anchor="c")
    c.text(qx + pw / 2, py + ph + 88, "国内の地方都市", 30, BODY, a4,
           anchor="c")

    a5 = ease(3.4, 4.3, t)
    if a5 > 0.01:
        ay = py + ph / 2
        x0, x1 = px + pw + 40, qx - 40
        col = V._mix(ACCENT, max(0.0, min(1.0, a5)))
        c.d.line([(x0, ay), (x1 - 26, ay)], fill=col, width=7)
        c.d.polygon([(x1, ay), (x1 - 30, ay - 20), (x1 - 30, ay + 20)],
                    fill=col)
        c.text((x0 + x1) / 2, ay - 90, "買収提案", 44, ACCENT, a5, True,
               anchor="c")
    fade_out(c, t, total)
    return c


# ---------------------------------------------------------------------------
# ④ シミュレーションフロー
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
    (icon_letter, "1", "X社 → 町の人", "金額付きの手紙"),
    (icon_talk, "2", "町の人 ⇄ 町の人", "場で会話"),
    (icon_list, "3", "町の人 → 町", "売りに出す"),
    (icon_yesno, "4", "町の人 → X社", "売る／売らない"),
    (icon_book, "5", "世界", "登記の更新"),
]


def scene_flow(t: float, xco_small: Image.Image) -> Canvas:
    c = Canvas()
    total = SEC["flow"]
    d = c.d
    a0 = ease(0.1, 0.7, t)
    c.text(96, 84, "シミュレーションフロー", 40, ACCENT, a0)
    c.text(96, 150, "毎月、同じ5手順をくり返す", 72, FG, a0, True, dy=16)
    c.photo(xco_small, W - 96 - xco_small.size[0], 84, a0)
    d.rectangle([W - 96 - xco_small.size[0], 84, W - 96, 84 +
                 xco_small.size[1]],
                outline=V._mix((70, 78, 92), max(0.0, min(1.0, a0))), width=3)
    c.text(W - 96, 84 + xco_small.size[1] + 14, "X社", 30, BODY, a0,
           anchor="r")

    x0, gap, bw = 78, 20, 328
    top, bh = 356, 400
    for i, (icon, num, who, head) in enumerate(FLOW):
        a = V._ease(0.9 + i * 2.2, 1.7 + i * 2.2, t)
        x = x0 + i * (bw + gap)
        c.card(x, top, x + bw, top + bh, a, edge=(46, 54, 66))
        c.text(x + bw / 2, top + 30, num, 40, ACCENT, a, True, anchor="c")
        c.text(x + bw / 2, top + 86, who, 30, BODY, a, anchor="c")
        icon(d, x + bw / 2, top + 216, 68, ACCENT, max(0.0, min(1.0, a)))
        c.text(x + bw / 2, top + 308, head, 38, FG, a, True, anchor="c")
        if i < len(FLOW) - 1:
            aa = V._ease(1.9 + i * 2.2, 2.5 + i * 2.2, t)
            c.text(x + bw + gap / 2, top + 186, "▶", 40, ACCENT, aa, True,
                   anchor="c")

    a2 = V._ease(10.4, 11.2, t)
    d.rectangle([78, 832, 1842, 838], fill=V._mix((40, 46, 56),
                                                  max(0.0, min(1.0, a2))))
    c.text(W / 2, 880, "×36か月（3年）", 58, FG, a2, True, anchor="c")
    fade_out(c, t, total)
    return c


# ---------------------------------------------------------------------------
# ⑥ 検証まとめ（スライド P10）
# ---------------------------------------------------------------------------

SUMMARY_SLIDE = os.path.join(ROOT, "docs", "submission", "video_assets",
                             "summary_slide_p10.png")
# P10（1280x745 に切ったもの）の中の区画。順に出していく。
SUM_BOXES = [
    ("head", (0, 0, 1280, 96), 0.2, 1.0),
    ("p1", (0, 96, 344, 420), 1.4, 2.3),
    ("p2", (344, 96, 638, 420), 3.4, 4.3),
    ("p3", (638, 96, 932, 420), 5.4, 6.3),
    ("p4", (932, 96, 1280, 420), 7.4, 8.3),
    ("box", (0, 420, 1280, 552), 10.0, 11.4),
    ("more", (0, 552, 1280, 745), 16.0, 17.4),
]


def summary_layout(slide: Image.Image) -> Tuple[Image.Image, float, int, int]:
    """P10 を画面に収め、元画像座標→画面座標の倍率とずれを返す。"""
    sw, sh = slide.size
    k = min(W / sw, H / sh)
    im = slide.resize((int(sw * k), int(sh * k)), Image.LANCZOS)
    full = Image.new("RGB", (W, H), (0, 0, 0))
    ox, oy = (W - im.size[0]) // 2, (H - im.size[1]) // 2
    full.paste(im, (ox, oy))
    return full, k, ox, oy


def scene_summary(t: float, lay) -> Canvas:
    full, k, ox, oy = lay
    total = SEC["summary"]
    c = Canvas()
    # ナレーションの長さに合わせて、出す間隔を引き伸ばす。
    ks = max(1.0, (total - 4.0) / SUM_BOXES[-1][3])
    for _name, (x0, y0, x1, y1), t0, t1 in SUM_BOXES:
        a = max(0.0, min(1.0, V._ease(t0 * ks, t1 * ks, t)))
        if a <= 0.01:
            continue
        box = (int(x0 * k) + ox, int(y0 * k) + oy,
               int(x1 * k) + ox, int(y1 * k) + oy)
        part = full.crop(box)
        if a < 0.999:
            part = Image.blend(Image.new("RGB", part.size, (0, 0, 0)), part, a)
        c.img.paste(part, (box[0], box[1]))
    c.d = ImageDraw.Draw(c.img)
    fade_out(c, t, total)
    return c


# ---------------------------------------------------------------------------
# ⑦ 締め
# ---------------------------------------------------------------------------

def scene_black(t: float) -> Canvas:
    """締めの前の溜め（黒・無文字）。"""
    return Canvas()


def scene_end(t: float, bg: Image.Image) -> Canvas:
    # 黒の溜めから背景ごと明けていく。
    k = max(0.0, min(1.0, V._ease(0.0, 1.2, t)))
    c = Canvas(Image.blend(Image.new("RGB", (W, H), (0, 0, 0)), bg, k))
    a0 = ease(0.4, 2.2, t)
    c.text(W / 2, 470, "あなたの町は大丈夫？", 110, FG, a0, True, anchor="c",
           dy=24)
    a1 = ease(2.4, 3.2, t)
    c.rule(W / 2 - 110, 660, W / 2 + 110, ACCENT, a1, h=4)
    fade_out(c, t, SEC["end"], tail=0.5)
    return c


# ---------------------------------------------------------------------------
# ⑤ 所有の変遷（36か月）
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
def render_months() -> Tuple[Tuple[str, ...], str]:
    """本編36か月＋第0月のコマを描き直す（移転の枠ハイライトは出さない）。"""
    run = V.Run(RUN_DIR, LABEL)
    fdir = os.path.join(RUN_DIR, "video_frames")
    os.makedirs(fdir, exist_ok=True)
    paths = []
    for m in range(1, 37):
        p = os.path.join(fdir, f"f{m:03d}_m{m:02d}.png")
        V.draw_month(run, m, p, mark_transfers=False)
        paths.append(p)

    run.ledger[0] = month0_rows(run)
    run.x_parcels[0] = []
    run.offers_cum[0] = 0
    p0 = os.path.join(fdir, "f000_m00.png")
    V.draw_month(run, 0, p0, subtitle="開始時点", mark_transfers=False)
    return tuple(paths), p0


LEGEND = "赤に変わった区画＝X社の所有になったところ"
END_LINE = "買収は進んだが、過半の取得には至らなかった。"


def month_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if img.size != (W, H):
        img = img.resize((W, H), Image.LANCZOS)
    return img


def banner(img: Image.Image, t: float, head: str, sub: str,
           swatch: bool) -> Image.Image:
    """地図のコマの下に、読み方の一言を敷く。"""
    img = img.copy()
    d = ImageDraw.Draw(img)
    d.rectangle([0, 930, W, H], fill=(0, 0, 0))
    a = max(0.0, min(1.0, V._ease(0.3, 1.0, t)))
    x = 96
    if swatch:
        d.rectangle([96, 986, 140, 1030], fill=V._mix(RED, a),
                    outline=V._mix(WHITE, a), width=2)
        x = 166
    d.text((x, 982), head, font=V._pil_font(44, True), fill=V._mix(WHITE, a))
    if sub:
        a2 = max(0.0, min(1.0, V._ease(0.9, 1.6, t)))
        d.text((x, 1034), sub, font=V._pil_font(30, False),
               fill=V._mix(BODY, a2))
    return img


# ---------------------------------------------------------------------------
# 組み立て
# ---------------------------------------------------------------------------

def build_frames(assets: Dict[str, Image.Image],
                 save: Optional[Dict[str, str]] = None):
    """全コマを順に返す generator（1コマ＝PIL Image）。"""
    paths, p0 = render_months()
    saved = set()

    def maybe_save(key: str, img: Image.Image) -> None:
        if save and key in save and key not in saved:
            img.save(save[key])
            saved.add(key)

    m0_img = month_image(p0)
    m36_img = month_image(paths[-1])

    plan = [
        ("title", SEC["title"], lambda t: scene_title(t, assets["slide"]).img, 4.0),
        ("why", SEC["why"], lambda t: scene_why(t, assets["map"]).img, 9.0),
        ("world", SEC["world"],
         lambda t: scene_world(t, assets["xco"], assets["town"]).img, 7.0),
        ("flow", SEC["flow"],
         lambda t: scene_flow(t, assets["xco_small"]).img, 11.6),
        ("m0", SEC["m0"],
         lambda t: banner(m0_img, t, LEGEND,
                          "これから36か月を早回しで見る。", True), 1.6),
        ("m36", SEC["m36"], lambda t: banner(m36_img, t, END_LINE, "", False),
         2.4),
        ("summary", SEC["summary"],
         lambda t: scene_summary(t, assets["summary"]).img, 12.0),
        ("black", SEC["black"], lambda t: scene_black(t).img, 0.5),
        ("end", SEC["end"], lambda t: scene_end(t, assets["end"]).img, 4.6),
    ]
    order = ["title", "why", "world", "flow", "m0", "__body__", "m36",
             "summary", "black", "end"]
    by_key = {p[0]: p for p in plan}

    for key in order:
        if key == "__body__":
            per = max(1, round(MONTH_SEC * FPS))
            for p in paths:
                img = month_image(p)
                for _ in range(per):
                    yield img
            continue
        _, sec, fn, shot_t = by_key[key]
        n = int(round(sec * FPS))
        for i in range(n):
            t = i / FPS
            img = fn(t)
            if abs(t - shot_t) < 0.5 / FPS:
                maybe_save({"m0": "m00"}.get(key, key), img)
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


def narration_plan() -> List[Tuple[float, str]]:
    """各ナレーションの開始秒（シーンの並びから決める）。"""
    order = ["title", "why", "world", "flow", "m0", "__body__", "m36",
             "summary", "black", "end"]
    at = {}
    t = 0.0
    for key in order:
        at[key] = t
        t += (MONTH_SEC * 36) if key == "__body__" else SEC[key]
    plan = [
        (at["title"] + LEAD, NAR["01"]),
        (at["why"] + LEAD, NAR["02"]),
        (at["world"] + LEAD, NAR["03"]),
        (at["flow"] + LEAD, NAR["04"]),
        (at["m0"] + LEAD, NAR["05"]),
        (at["m36"] + LEAD, NAR["06"]),
        (at["summary"] + LEAD, NAR["07"]),
        (at["end"] + 1.0, NAR["08"]),
    ]
    print("ナレーション配置（秒）", flush=True)
    for start, wav in plan:
        print(f"  {start:7.2f}  +{wav_sec(wav):5.2f}  "
              f"{os.path.basename(wav)}", flush=True)
    print(f"  全体 {t:.2f}秒", flush=True)
    return plan


def mix_narration(silent: str, out: str) -> None:
    import mix_audio
    mix_audio.mix(silent, narration_plan(), BGM, out)


def raw(name: str) -> Image.Image:
    return Image.open(os.path.join(BG.OWNER, name + ".png")).convert("RGB")


def main() -> int:
    os.makedirs(FRAMES_DIR, exist_ok=True)
    # 締めだけは背景として沈める（赤が強烈にならない程度に）。
    end_src = os.path.join(BG.OWNER, "ending_bg2.png")
    end_out = os.path.join(BG.OWNER, "ending_bg2_dim.png")
    assets = {
        # 地図はコンテンツ＝暗くしない。正方形版なら見切れずに大きく置ける。
        "map": contain(raw("japan_map_square"), 790, 900),
        "xco": cover(raw("xco_building"), 560, 380),
        "xco_small": cover(raw("xco_building"), 300, 200),
        "town": cover(raw("town_photo"), 560, 380),
        "slide": Image.open(TITLE_SLIDE).convert("RGB").resize(
            (W, H), Image.LANCZOS),
        "end": Image.open(BG.owner_bg(end_src, end_out, 0.42, "cover", 0.5,
                                      None, 0.5, 0.45)).convert("RGB"),
        "summary": summary_layout(
            Image.open(SUMMARY_SLIDE).convert("RGB")),
    }
    save = {k: os.path.join(FRAMES_DIR, f"final_{k}.png")
            for k in ("title", "why", "world", "flow", "m00", "m36",
                      "summary", "end")}

    crf = 21
    for _ in range(4):
        encode(build_frames(assets, save), SILENT_MP4, crf)
        mix_narration(SILENT_MP4, OUT_MP4)
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
