#!/usr/bin/env python
"""静かな占領 — エントリポイント。

  python run.py --config configs/config_mvp_v1.yaml                 # config通り
  python run.py --config configs/config_mvp_v1.yaml --provider mock # API課金なしのドライラン
  python run.py --config configs/config_mvp_v1.yaml --steps 3       # 短縮

出力は simulations/YYYY-MM-DD_HHMM_NN_<name>/ に自己完結で置かれる（config写し・
生ログ・KPI・レポートHTML）。この命名規則は前回ハッカソンのリポと同じ。
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
import shutil
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.simulation import Simulation   # noqa: E402
from src.viz import render_report        # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.join(ROOT, "simulations")


def load_dotenv(path: str = None) -> None:
    path = path or os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def resolve_run_dir(run_name: str) -> str:
    """simulations/YYYY-MM-DD_HHMM_NN_name/ を作って返す。NN は既存の最大+1。"""
    os.makedirs(SIM_DIR, exist_ok=True)
    nn = 0
    for d in os.listdir(SIM_DIR):
        m = re.match(r"^\d{4}-\d{2}-\d{2}_\d{4}_(\d{2})_", d)
        if m:
            nn = max(nn, int(m.group(1)))
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    folder = f"{stamp}_{nn + 1:02d}_{run_name}"
    path = os.path.join(SIM_DIR, folder)
    os.makedirs(path, exist_ok=True)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_mvp_v1.yaml")
    ap.add_argument("--provider", default=None, help="google | mock | openai (configを上書き)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--no-classify", action="store_true", help="発話のLLM分類をしない")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    load_dotenv()

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.provider:
        cfg["llm"]["provider"] = args.provider
    if args.model:
        cfg["llm"]["model"] = args.model
    if args.steps:
        cfg["steps"] = args.steps
    if args.run_name:
        cfg["run_name"] = args.run_name
    if args.no_classify:
        cfg.setdefault("kpi", {})["classify_utterances"] = False

    persona_path = os.path.join(ROOT, cfg["personas_file"])
    with open(persona_path, encoding="utf-8") as f:
        personas = yaml.safe_load(f)

    run_dir = resolve_run_dir(cfg["run_name"])
    folder = os.path.basename(run_dir)
    print(f"[run] {folder}")
    print(f"[run] provider={cfg['llm']['provider']} model={cfg['llm'].get('model')} "
          f"steps={cfg['steps']}")

    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    shutil.copy(persona_path, os.path.join(run_dir, "personas.yaml"))

    sim = Simulation(cfg, personas, run_dir)
    summary = sim.run()

    # mock ランでは実プロンプト全文を残す（tools/estimate_cost.py の入力になる）
    plog = getattr(sim.client, "prompt_log", None)
    if plog:
        with open(os.path.join(run_dir, "prompt_log.jsonl"), "w", encoding="utf-8") as f:
            for p in plog:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"[out] prompt_log.jsonl ({len(plog)} 本)")

    html_path = os.path.join(run_dir, f"{folder}.html")
    render_report(sim, html_path, folder)

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[out] {run_dir}")
    print(f"[out] report: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
