#!/usr/bin/env python
"""v9g — エントリポイント。

  python run_v9g.py --config configs/config_field_v9g.yaml --provider mock
  python run_v9g.py --config configs/config_field_v9g.yaml --steps 1     # 実APIスモーク
  python run_v9g.py --config configs/config_field_v9g.yaml --workers 4   # 本走（36か月）

既存の `run.py`・`run_v8*.py`・`run_v9.py`・`run_v9b.py`・`run_v9c.py`・`run_v9d.py`・`run_v9e.py`・`run_v9f.py` には触らない。
出力は simulations/YYYY-MM-DD_HHMM_NN_<run_name>/ に自己完結で置く
（月ごとのチェックポイントは その中の checkpoint/）。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_v8c import ROOT, load_dotenv, resolve_run_dir   # noqa: E402
from src.sim_v9g import SimulationV9G                    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_field_v9g.yaml")
    ap.add_argument("--provider", default=None, help="google | mock")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--max-cost", type=float, default=None,
                    help="手元集計の費用がこれを超えたらその月で止める")
    ap.add_argument("--out-dir", default=None,
                    help="出力先の親フォルダ（既定 simulations/）")
    ap.add_argument("--workers", type=int, default=None,
                    help="同時に投げるコール数（世界は変わらない・機械の都合だけ）")
    ap.add_argument("--timeout", type=float, default=None,
                    help="1コールの締切（秒・既定 60）")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_dotenv()

    cfg_path = (args.config if os.path.isabs(args.config)
                else os.path.join(ROOT, args.config))
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
    if args.workers:
        cfg["llm"]["parallel_workers"] = int(args.workers)
    if args.timeout is not None:
        cfg["llm"]["request_timeout_sec"] = float(args.timeout)
    if args.run_name:
        cfg["run_name"] = args.run_name

    run_dir = resolve_run_dir(str(cfg["run_name"]), args.out_dir)
    # 走った設定は**コマンドラインの上書きを反映した実効値**を残す
    # （元ファイルをそのまま写すと --steps / --max-cost / --workers が証拠に残らない。
    #  走行前レビューの必須指摘 2026-08-30）。元ファイルも別名で並べて残す。
    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    shutil.copy(cfg_path, os.path.join(run_dir, "config_source.yaml"))
    personas = str(cfg["personas_file"])
    shutil.copy(personas if os.path.isabs(personas)
                else os.path.join(ROOT, personas),
                os.path.join(run_dir, "personas.yaml"))
    print(f"run_dir: {run_dir}", flush=True)
    sim = SimulationV9G(cfg, run_dir)
    summary = sim.run()
    print(json.dumps({k: v for k, v in summary.items() if k != "usage"},
                     ensure_ascii=False, indent=2))
    print(f"run_dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
