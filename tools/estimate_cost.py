#!/usr/bin/env python
"""1ランの API コストを、**実際に組み立てられたプロンプトの実測トークン数**から見積もる。

使い方:
    # 1) mock で 1ラン通す（API課金ゼロ）と、その run_dir に prompt_log.jsonl が出る
    python run.py --config configs/config_mvp_v1.yaml --provider mock
    # 2) その実プロンプトからコストを見積もる
    python tools/estimate_cost.py --run simulations/2026-08-14_2200_01_mvp_v1
    # 3) Gemini の countTokens で校正する（countTokens は課金対象外。要 GOOGLE_API_KEY）
    python tools/estimate_cost.py --run <run_dir> --calibrate 24

トークン数の出し方:
  - 既定はローカル推定（日本語=文字数ベース／ASCII=4文字1トークン）。API を一切叩かない。
  - --calibrate N を付けると実プロンプト N 本を countTokens API に投げ、
    推定値との比率で全体を補正する（countTokens は生成を伴わず課金されない）。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402


def estimate_tokens(text: str) -> float:
    """CJK は概ね 1文字≈1トークン、ASCII は概ね 4文字≈1トークン として数える。"""
    cjk = 0
    ascii_chars = 0
    other = 0
    for ch in text:
        o = ord(ch)
        if o < 128:
            ascii_chars += 1
        elif unicodedata.east_asian_width(ch) in ("W", "F"):
            cjk += 1
        else:
            other += 1
    return cjk * 1.0 + ascii_chars / 4.0 + other / 2.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="mock ランの run_dir")
    ap.add_argument("--calibrate", type=int, default=0,
                    help="countTokens API で校正するサンプル数（0=校正しない）")
    ap.add_argument("--out-tokens", type=int, default=None,
                    help="1回あたりの出力トークン想定（既定 = config の max_tokens の 0.62）")
    args = ap.parse_args()

    run_dir = args.run
    with open(os.path.join(run_dir, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    plog_path = os.path.join(run_dir, "prompt_log.jsonl")
    if not os.path.exists(plog_path):
        print(f"prompt_log.jsonl が無い: {plog_path}\n"
              f"--provider mock で走らせたランを指定しろ。")
        return 1
    prompts = [json.loads(l) for l in open(plog_path, encoding="utf-8")]

    model = cfg["llm"]["model"]
    price = cfg["cost"]["price_table"].get(model)
    if price is None:
        print(f"config の cost.price_table に {model} の単価が無い")
        return 1
    safety = float(cfg["cost"].get("safety_factor", 2.0))

    ratio = 1.0
    calib_note = "ローカル推定のみ（countTokens 未使用）"
    if args.calibrate > 0:
        from src.llm_client_factory import GeminiClient
        from run import load_dotenv
        load_dotenv()
        client = GeminiClient(model=model)
        rng = random.Random(7)
        sample = rng.sample(prompts, min(args.calibrate, len(prompts)))
        est_sum = real_sum = 0.0
        for p in sample:
            text = p["system"] + "\n" + p["user"]
            est_sum += estimate_tokens(text)
            real_sum += client.count_tokens(text)
        ratio = real_sum / est_sum if est_sum else 1.0
        calib_note = (f"countTokens {len(sample)}本で校正: 実測 {real_sum:.0f} tok / "
                      f"推定 {est_sum:.0f} tok → 係数 {ratio:.3f}")

    total_in = 0.0
    by_tag = {}
    for p in prompts:
        t = (estimate_tokens(p["system"]) + estimate_tokens(p["user"])) * ratio
        total_in += t
        slot = by_tag.setdefault(p["tag"], {"calls": 0, "in": 0.0})
        slot["calls"] += 1
        slot["in"] += t

    max_tokens = int(cfg["llm"].get("max_tokens", 420))
    out_per_call = args.out_tokens if args.out_tokens else int(max_tokens * 0.62)
    n_calls = len(prompts)
    # 分類呼び出しは出力が長い（バッチ分の結果を返す）
    n_classify = sum(1 for p in prompts if p["tag"] == "classify")
    total_out = (n_calls - n_classify) * out_per_call + n_classify * 900

    cost_in = total_in / 1_000_000 * price["input"]
    cost_out = total_out / 1_000_000 * price["output"]
    total = cost_in + cost_out

    print("=" * 66)
    print(f"run           : {os.path.basename(run_dir)}")
    print(f"model         : {model}  (in ${price['input']}/1M, out ${price['output']}/1M)")
    print(f"steps         : {cfg['steps']}  agents: {sum(cfg['agents'].values())}")
    print(f"tokenizer     : {calib_note}")
    print("-" * 66)
    print(f"API 呼び出し数 : {n_calls}  (うち発話分類 {n_classify})")
    print(f"入力トークン   : {total_in:,.0f}   (平均 {total_in/max(1,n_calls):,.0f}/call)")
    print(f"出力トークン   : {total_out:,.0f}   (1call {out_per_call} tok 想定 / cache割引なし)")
    print("-" * 66)
    for tag, s in sorted(by_tag.items()):
        print(f"  {tag:<22} calls={s['calls']:>5}  in={s['in']:>12,.0f}")
    print("-" * 66)
    print(f"入力コスト     : ${cost_in:.4f}")
    print(f"出力コスト     : ${cost_out:.4f}")
    print(f"■ 1ラン見積り  : ${total:.3f}")
    print(f"■ 安全係数x{safety:g} : ${total*safety:.3f}  ← 上限として扱う値")
    print("=" * 66)
    print("前提: explicit context cache は使わない（system prompt が最小トークン数に届かず")
    print("      作成失敗する見込み）。implicit cache が効けば入力コストはこれより下がる。")

    out = {
        "run": os.path.basename(run_dir), "model": model, "price": price,
        "calls": n_calls, "classify_calls": n_classify,
        "input_tokens": round(total_in), "output_tokens": total_out,
        "cost_input_usd": round(cost_in, 5), "cost_output_usd": round(cost_out, 5),
        "cost_total_usd": round(total, 4), "safety_factor": safety,
        "cost_upper_usd": round(total * safety, 4), "calibration": calib_note,
    }
    with open(os.path.join(run_dir, "cost_estimate.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[out] {os.path.join(run_dir, 'cost_estimate.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
