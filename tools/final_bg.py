#!/usr/bin/env python
"""提出動画の背景画像を用意する（AI 生成・失敗したら PIL で描く）。

  python tools/final_bg.py

出すもの:
  docs/submission/video_assets/bg_town.png … 温泉観光都市の街並み
  docs/submission/video_assets/bg_xco.png  … 東アジアの大都市の高層ビル群

生成できない（モデルが無い・権限が無い）ときは落ちずにシルエットを描く。
どちらだったかは標準出力に出す（報告に使う）。
"""

from __future__ import annotations

import os
import random
import sys
from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "docs", "submission", "video_assets")

W, H = 1920, 1080

PROMPT_TOWN = (
    "A photograph of a Japanese seaside hot-spring resort town: steam rising "
    "from bathhouses, a small harbour with fishing boats, low-rise wooden inns "
    "and shopfronts along narrow streets, wooded hills behind, overcast sky, "
    "documentary photo style, no text, no letters, no signage, no people."
)
PROMPT_XCO = (
    "A photograph of an East Asian metropolis skyline: dense cluster of glass "
    "and steel high-rise office towers seen from a low angle, overcast grey "
    "sky, cold light, documentary photo style, no text, no letters, no "
    "signage, no flags, no logos, no people."
)

# 指示された候補（2.5-preview / imagen 3・4）はこの API キーでは 404 だった。
# 実際に使えるものを ListModels で確かめて並べる（先頭から順に試す）。
IMAGE_MODELS = (
    "gemini-3.1-flash-image",
    "gemini-2.5-flash-image",
    "gemini-2.5-flash-image-preview",
    "imagen-4.0-generate-001",
    "imagen-3.0-generate-002",
)


def load_env() -> Optional[str]:
    key = os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    p = os.path.join(ROOT, ".env")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("GOOGLE_API_KEY"):
                    _, _, v = line.partition("=")
                    v = v.strip().strip('"').strip("'")
                    if v:
                        os.environ["GOOGLE_API_KEY"] = v
                        return v
    return None


def try_generate(prompt: str, out: str) -> Optional[str]:
    """AI 画像生成を試す。成功したらモデル名、駄目なら None。"""
    key = load_env()
    if not key:
        print("GOOGLE_API_KEY が無い → フォールバック", file=sys.stderr)
        return None
    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        print(f"google-genai が使えない: {e}", file=sys.stderr)
        return None
    client = genai.Client(api_key=key)
    for model in IMAGE_MODELS:
        try:
            if model.startswith("imagen"):
                r = client.models.generate_images(
                    model=model, prompt=prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1, aspect_ratio="16:9"))
                img = r.generated_images[0].image
                data = getattr(img, "image_bytes", None)
                if data is None:
                    continue
            else:
                r = client.models.generate_content(
                    model=model, contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        image_config=types.ImageConfig(aspect_ratio="16:9")))
                data = None
                for part in r.candidates[0].content.parts:
                    inline = getattr(part, "inline_data", None)
                    if inline is not None and inline.data:
                        data = inline.data
                        break
                if data is None:
                    continue
            with open(out, "wb") as f:
                f.write(data)
            Image.open(out).convert("RGB").resize(
                (W, H), Image.LANCZOS).save(out)
            print(f"AI生成 {model} → {out}")
            return model
        except Exception as e:
            print(f"{model} 失敗: {str(e)[:200]}", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# フォールバック（PIL で描くシルエット）
# ---------------------------------------------------------------------------

def fallback_town(out: str) -> str:
    rnd = random.Random(7)
    img = Image.new("RGB", (W, H), (74, 80, 88))
    d = ImageDraw.Draw(img)
    for y in range(H):                       # 曇り空のグラデーション
        v = int(96 - 34 * y / H)
        d.line([(0, y), (W, y)], fill=(v, v + 4, v + 8))
    # 背後の山
    for k, (base, col) in enumerate(((640, (56, 62, 66)), (700, (44, 50, 54)))):
        pts = [(0, H)]
        x = 0
        while x <= W:
            pts.append((x, base + int(90 * (0.5 - abs(((x / W) * 3 % 1) - 0.5))
                                      * (1 + 0.4 * rnd.random())) - k * 20))
            x += 60
        pts.append((W, H))
        d.polygon(pts, fill=col)
    # 湯けむり
    steam = Image.new("RGB", (W, H), (0, 0, 0))
    sd = ImageDraw.Draw(steam)
    for _ in range(40):
        x, y = rnd.randint(60, W - 60), rnd.randint(560, 860)
        r = rnd.randint(40, 130)
        sd.ellipse([x - r, y - r // 2, x + r, y + r // 2], fill=(70, 74, 78))
    steam = steam.filter(ImageFilter.GaussianBlur(38))
    img = Image.blend(img, Image.blend(img, steam, 0.0), 0.0)
    img.paste(Image.blend(img.crop((0, 0, W, H)), steam, 0.35), (0, 0))
    d = ImageDraw.Draw(img)
    # 低層の旅館・商店（屋根の連なり）
    x = -40
    while x < W + 40:
        w = rnd.randint(90, 190)
        h = rnd.randint(90, 190)
        top = 820 - h
        c = rnd.randint(28, 52)
        d.polygon([(x - 14, top), (x + w // 2, top - 42), (x + w + 14, top)],
                  fill=(c - 6, c - 4, c - 2))
        d.rectangle([x, top, x + w, 900], fill=(c, c + 3, c + 5))
        for wy in range(top + 26, 890, 46):
            for wx in range(x + 16, x + w - 20, 40):
                if rnd.random() < 0.5:
                    d.rectangle([wx, wy, wx + 20, wy + 22],
                                fill=(c + 26, c + 26, c + 24))
        x += w + rnd.randint(8, 28)
    # 手前の港（水面）
    d.rectangle([0, 900, W, H], fill=(38, 44, 50))
    for _ in range(140):
        y = rnd.randint(905, H - 6)
        x0 = rnd.randint(0, W - 120)
        d.line([(x0, y), (x0 + rnd.randint(30, 120), y)], fill=(56, 62, 70))
    for bx in (240, 700, 1300, 1660):        # 漁船
        d.polygon([(bx - 70, 960), (bx + 70, 960), (bx + 46, 1000),
                   (bx - 46, 1000)], fill=(30, 34, 38))
        d.line([(bx, 960), (bx, 878)], fill=(30, 34, 38), width=5)
    img.filter(ImageFilter.GaussianBlur(0.6)).save(out)
    print(f"フォールバック（PIL 描画）→ {out}")
    return out


def fallback_xco(out: str) -> str:
    rnd = random.Random(11)
    img = Image.new("RGB", (W, H), (70, 76, 84))
    d = ImageDraw.Draw(img)
    for y in range(H):
        v = int(104 - 44 * y / H)
        d.line([(0, y), (W, y)], fill=(v, v + 3, v + 9))
    layers = ((0.55, (52, 58, 66)), (0.75, (38, 44, 52)), (1.0, (24, 29, 36)))
    for depth, col in layers:
        x = -60
        while x < W + 60:
            w = int(rnd.randint(80, 190) * (0.7 + depth * 0.6))
            h = int(rnd.randint(330, 900) * (0.5 + depth * 0.7))
            top = H - h
            d.rectangle([x, top, x + w, H], fill=col)
            if rnd.random() < 0.35:          # 頂部の塔屋
                d.rectangle([x + w // 3, top - rnd.randint(30, 90),
                             x + w - w // 3, top], fill=col)
            lit = tuple(min(255, c + int(26 + 30 * depth)) for c in col)
            for wy in range(top + 22, H - 20, 40):
                for wx in range(x + 12, x + w - 16, 30):
                    if rnd.random() < 0.42:
                        d.rectangle([wx, wy, wx + 16, wy + 24], fill=lit)
            x += w + rnd.randint(10, 34)
    img.filter(ImageFilter.GaussianBlur(0.7)).save(out)
    print(f"フォールバック（PIL 描画）→ {out}")
    return out


def to_grey_dim(path: str, out: str, alpha: float = 0.35) -> str:
    """グレースケール化して黒に沈める（文字が読める暗さに）。"""
    img = Image.open(path).convert("L").convert("RGB").resize((W, H),
                                                              Image.LANCZOS)
    black = Image.new("RGB", (W, H), (0, 0, 0))
    Image.blend(black, img, alpha).save(out)
    return out


def build(force: bool = False) -> Tuple[str, str]:
    """背景2枚を用意する。既にあれば作り直さない（生成は課金される）。"""
    os.makedirs(ASSETS, exist_ok=True)
    town = os.path.join(ASSETS, "bg_town.png")
    xco = os.path.join(ASSETS, "bg_xco.png")
    if force or not os.path.exists(town):
        if not try_generate(PROMPT_TOWN, town):
            fallback_town(town)
    if force or not os.path.exists(xco):
        if not try_generate(PROMPT_XCO, xco):
            fallback_xco(xco)
    return town, xco


OWNER = os.path.join(ASSETS, "owner")


def owner_bg(src: str, out: str, alpha: float = 0.35, mode: str = "cover",
             bias: float = 0.5, box=None, anchor: float = 0.5,
             sat: Optional[float] = None) -> str:
    """施主支給の画像を背景に仕立てる（グレー化＋黒に沈める）。

    mode="cover" … 16:9 に切り抜いて全面に敷く（bias=切り抜きの上下位置）
    mode="contain" … 縦横比を保ったまま収める（余白は黒）。地図はこちら。
    """
    if sat is None:
        img = Image.open(src).convert("L").convert("RGB")
    else:
        # 色は残したまま彩度だけ落とす（締めの赤を「見えるが強烈でない」に）。
        from PIL import ImageEnhance
        img = ImageEnhance.Color(
            Image.open(src).convert("RGB")).enhance(sat)
    if box is not None:
        img = img.crop(box)
    sw, sh = img.size
    if mode == "cover":
        th = int(round(sw * H / W))
        if th <= sh:
            top = int(round((sh - th) * bias))
            img = img.crop((0, top, sw, top + th))
        else:
            tw = int(round(sh * W / H))
            left = int(round((sw - tw) * 0.5))
            img = img.crop((left, 0, left + tw, sh))
        img = img.resize((W, H), Image.LANCZOS)
    else:
        k = min(W / sw, H / sh)
        img = img.resize((max(1, int(sw * k)), max(1, int(sh * k))),
                         Image.LANCZOS)
        canvas = Image.new("RGB", (W, H), (0, 0, 0))
        canvas.paste(img, (int(round((W - img.size[0]) * anchor)),
                           (H - img.size[1]) // 2))
        img = canvas
    black = Image.new("RGB", (W, H), (0, 0, 0))
    Image.blend(black, img, max(0.0, min(1.0, alpha))).save(out)
    return out


if __name__ == "__main__":
    build()
