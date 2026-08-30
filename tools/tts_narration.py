"""ナレーション wav を Gemini TTS（声 Kore）で合成する。

    python tools/tts_narration.py            # 全部
    python tools/tts_narration.py 04 08      # 指定番号だけ
"""

from __future__ import annotations

import os
import sys
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "docs", "submission", "video_assets", "audio")
MODEL = "gemini-2.5-flash-preview-tts"
VOICE = "Kore"

# 全ナレーションを一度に同じ設定・同じ読み方指示で撮る（声色が途中で変わらない）。
STYLE = ("落ち着いた低めの声で、ドキュメンタリーの解説のように、"
         "はっきりと、ややゆっくり読み上げてください：")
# 締めだけは同じ声・同じ基本指示のまま、読み方だけ重くする。
STYLE_END = STYLE[:-1] + "。深刻に、低く、ゆっくり、間を取って読んでください："

LINES = {
    "01": "がいてき不動産買収へのコミュニティ自衛。コミュニティの創発は、がいてきな不動産買収を抑制できるのか、検証しました。",
    "02": "がいてき不動産買収の事例です。ニセコや対馬では、実際に不動産を買われてから町は知りました。安全保障上の危機事例です。",
    "03": "今回のシミュレーションについて説明します。海外の不動産投資会社X社が、地方都市A市の不動産の過半買収を試みるシミュレーションです。",
    "04": "X社が、不動産所有者に買収を提案します。町の人は迷い、相談し、売るか売らないかを決めます。これを36か月・3年間シミュレーションしました。",
    "05": "シミュレーションの結果です。みどりとオレンジが、スタート時点の土地建物所有状況です。赤に変わったところが、X社のものになった不動産です。",
    "06": "徐々にX社による買収が進みますが、最終的に過半の買収には至りませんでした。",
    "07": "買収の進み具合と町の人の創発の関係性を探るため、X社の買収目的を明示した場合のシミュレーション、町の人同士の相談がない場合のシミュレーションを行いました。結果、買収目的明示の場合、明らかに買収抑制がかかり、町の人同士の創発がないと売買が進みにくい可能性が見えました。がいてき不動産買収の抑制には、実効支配を意図しているのでは？という疑いや気づきができるか、それをコミュニティ内で共有できるかがカギを握っている可能性があります。",
    "08": "あなたの町は、大丈夫？",
}


def load_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    with open(os.path.join(ROOT, ".env"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("GOOGLE_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("GOOGLE_API_KEY が無い")


def main() -> int:
    from google import genai
    from google.genai import types as t

    want = sys.argv[1:] or sorted(LINES)
    client = genai.Client(api_key=load_key())
    os.makedirs(OUT_DIR, exist_ok=True)
    for num in want:
        cfg = t.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=t.SpeechConfig(
                voice_config=t.VoiceConfig(
                    prebuilt_voice_config=t.PrebuiltVoiceConfig(
                        voice_name=VOICE))),
        )
        style = STYLE_END if num == "08" else STYLE
        res = client.models.generate_content(
            model=MODEL, contents=style + LINES[num], config=cfg)
        pcm = res.candidates[0].content.parts[0].inline_data.data
        path = os.path.join(OUT_DIR, f"nar_v7_{num}.wav")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(pcm)
        print(f"nar_v7_{num}.wav {len(pcm) / 2 / 24000:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
