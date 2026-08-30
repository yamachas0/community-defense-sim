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

LINES = {
    "01": "がいてき不動産買収へのコミュニティ自衛。コミュニティの創発は、がいてきな不動産買収を抑制できるのか、検証しました。",
    "02": "がいてき不動産買収の事例です。ニセコや対馬では、実際に不動産を買われてから町は知りました。安全保障上の危機事例です。",
    "03": "今回のシミュレーションについて説明します。海外の不動産投資会社X社が、地方都市A市の不動産の過半買収を試みるシミュレーションです。",
    "04": "X社が、不動産所有者に買収を提案します。町の人は迷い、相談し、売るか売らないかを決めます。これを36か月・3年間シミュレーションしました。",
    "05": "シミュレーションの結果です。みどりとオレンジが、スタート時点の土地建物所有状況です。赤に変わったところが、X社のものになった不動産です。",
    "06": "徐々にX社による買収が進みますが、最終的に過半の買収には至りませんでした。",
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
        res = client.models.generate_content(model=MODEL, contents=LINES[num],
                                             config=cfg)
        pcm = res.candidates[0].content.parts[0].inline_data.data
        path = os.path.join(OUT_DIR, f"nar_{num}.wav")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(pcm)
        print(f"nar_{num}.wav {len(pcm) / 2 / 24000:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
