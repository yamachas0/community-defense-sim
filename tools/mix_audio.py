"""映像(音声なし mp4) + ナレーション(複数、開始秒指定) + BGM を1本に混ぜる。

事前検証用ツール（本番の mp4 にはまだ触らない）。CLI 例:

    python tools/mix_audio.py --video in.mp4 \
        --narration 2.0:docs/submission/video_assets/audio/tts_test_Kore.wav \
        --bgm docs/submission/video_assets/audio/bgm_pad.wav \
        --out out/mix_test/mixed_10s.mp4

--narration は "開始秒:wavパス" を複数回指定できる。BGM は自動で映像の長さに
トリム/フェードなしでそのまま重ねる（音量は既定 -14dB）。

ffmpeg は必ず窓なし（CREATE_NO_WINDOW）で起動し、stderr はパイプせずログ
ファイルへ吐かせる（長時間 stdin 書き込みとの相互待ちを避けるため）。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Tuple

FFMPEG_FALLBACK_DIR = r"D:\ユーザー\ffmpeg-master-latest-win64-gpl-shared\bin"


def _bin(name: str = "ffmpeg") -> str:
    p = shutil.which(name)
    if p:
        return p
    cand = os.path.join(FFMPEG_FALLBACK_DIR, name + ".exe")
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


def _run_quiet_to_log(cmd: List[str], log_path: str) -> int:
    """stderr/stdout はパイプせずログファイルへ直接書く（デッドロック回避）。"""
    with open(log_path, "wb") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                              **_noshow())
    return proc.returncode


def probe_duration(path: str) -> float:
    log_path = os.path.join(tempfile.gettempdir(), "mix_audio_probe.log")
    cmd = [_bin("ffprobe"), "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", path]
    with open(log_path, "wb") as logf:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=logf,
                              **_noshow())
    out = proc.stdout.decode("utf-8", "replace").strip()
    if proc.returncode != 0 or not out:
        raise SystemExit(f"ffprobe に失敗（{path}）: ログ={log_path}")
    return float(out)


def parse_narration_arg(raw: str) -> Tuple[float, str]:
    start_str, _, wav_path = raw.partition(":")
    if not wav_path:
        raise SystemExit(f"--narration の形式が不正: {raw!r}（'開始秒:パス' 形式で）")
    return float(start_str), wav_path


def mix(video_path: str, narrations: List[Tuple[float, str]], bgm_path: str,
        out_path: str, bgm_db: float = -14.0, log_path: str = None) -> None:
    if not os.path.exists(video_path):
        raise SystemExit(f"映像が見つからない: {video_path}")
    for _, wav in narrations:
        if not os.path.exists(wav):
            raise SystemExit(f"ナレーションが見つからない: {wav}")
    if not os.path.exists(bgm_path):
        raise SystemExit(f"BGM が見つからない: {bgm_path}")

    video_dur = probe_duration(video_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    log_path = log_path or os.path.join(
        os.path.dirname(os.path.abspath(out_path)), "mix_audio_ffmpeg.log")

    cmd = [_bin(), "-y", "-loglevel", "error", "-i", video_path]
    for _, wav in narrations:
        cmd += ["-i", wav]
    cmd += ["-i", bgm_path]

    bgm_input_idx = 1 + len(narrations)
    filters = []
    narration_labels = []
    for i, (start_sec, _) in enumerate(narrations):
        ms = max(0, int(round(start_sec * 1000)))
        in_idx = i + 1
        label = f"an{i}"
        filters.append(f"[{in_idx}:a]adelay={ms}:all=1,apad[{label}]")
        narration_labels.append(f"[{label}]")

    filters.append(
        f"[{bgm_input_idx}:a]volume={bgm_db}dB,"
        f"atrim=0:{video_dur:.3f},apad[bgm]"
    )

    mix_inputs = "[bgm]" + "".join(narration_labels)
    n_inputs = 1 + len(narrations)
    filters.append(
        f"{mix_inputs}amix=inputs={n_inputs}:duration=first:"
        f"dropout_transition=0:normalize=0[aout]"
    )

    filter_complex = ";".join(filters)
    cmd += [
        "-filter_complex", filter_complex,
        "-map", "0:v:0", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-t", f"{video_dur:.3f}",
        os.path.abspath(out_path),
    ]

    rc = _run_quiet_to_log(cmd, log_path)
    if rc != 0:
        tail = ""
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                tail = f.read()[-4000:]
        except OSError:
            pass
        sys.stderr.write(tail + "\n")
        raise SystemExit(f"ffmpeg（ミックス）が失敗した。ログ={log_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, help="音声なし mp4")
    ap.add_argument("--narration", action="append", default=[],
                    help="開始秒:wavパス（複数指定可）")
    ap.add_argument("--bgm", required=True, help="BGM wav")
    ap.add_argument("--bgm-db", type=float, default=-14.0,
                    help="BGM の減衰量(dB、既定 -14)")
    ap.add_argument("--out", required=True, help="出力 mp4")
    args = ap.parse_args()

    narrations = [parse_narration_arg(n) for n in args.narration]
    mix(args.video, narrations, args.bgm, args.out, bgm_db=args.bgm_db)
    print("WROTE", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
