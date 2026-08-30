#!/usr/bin/env python
"""v8「最小の町」— エントリポイント。

  python run_v8.py --config configs/config_field_v8.yaml --provider mock
  python run_v8.py --config configs/config_field_v8.yaml --steps 1     # 実APIスモーク
  python run_v8.py --config configs/config_field_v8.yaml               # 本走（36か月）

既存の `run.py` には触らない（v1〜v6b の実行経路はそのまま）。
出力は simulations/YYYY-MM-DD_HHMM_NN_<run_name>/ に自己完結で置く。
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

from src.sim_v8 import SimulationV8   # noqa: E402

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
    os.makedirs(SIM_DIR, exist_ok=True)
    nn = 0
    for d in os.listdir(SIM_DIR):
        m = re.match(r"^\d{4}-\d{2}-\d{2}_\d{4}_(\d{2})_", d)
        if m:
            nn = max(nn, int(m.group(1)))
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    path = os.path.join(SIM_DIR, f"{stamp}_{nn + 1:02d}_{run_name}")
    os.makedirs(path, exist_ok=True)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_field_v8.yaml")
    ap.add_argument("--provider", default=None, help="google | mock")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--no-cache", action="store_true",
                    help="明示キャッシュを使わない（before/after の実測用）")
    ap.add_argument("--max-cost", type=float, default=None,
                    help="手元集計の費用がこれを超えたらその月で止める")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_dotenv()

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.provider:
        cfg["llm"]["provider"] = args.provider
    if args.steps:
        cfg["steps"] = int(args.steps)
    if args.no_cache:
        cfg["llm"]["enable_cache"] = False
    if args.max_cost is not None:
        cfg["max_cost_usd"] = float(args.max_cost)
    if args.run_name:
        cfg["run_name"] = args.run_name

    run_dir = resolve_run_dir(str(cfg["run_name"]))
    shutil.copy(cfg_path, os.path.join(run_dir, "config.yaml"))
    sim = SimulationV8(cfg, run_dir)
    summary = sim.run()
    print(json.dumps({k: v for k, v in summary.items() if k != "usage"},
                     ensure_ascii=False, indent=2))
    print(f"run_dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
