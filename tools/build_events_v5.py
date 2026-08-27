#!/usr/bin/env python
"""v5 の台本（取得スケジュールと兆候）を1回だけ生成して YAML に凍結する。

  python tools/build_events_v5.py --config configs/config_field_v5.yaml \
      --out configs/events_v5_seed85.yaml

生成規則は docs/world_design_v5_impl.md 3.3 に固定してある。
**結果を見て調整しない**。走行前に1回だけ回し、出力を commit して以後は固定する。
（この台本は「世界で起きた事実」であり、主体の判断には一切触れない。）
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.agents import build_roster            # noqa: E402
from src.world import assign_tenancies, build_town   # noqa: E402

# 兆候の候補（種類, audience の型）。registry は必ず1本付くので候補には入れない。
TRACE_POOL = [
    ("moving_out", "neighbors", 1),
    ("sign_change", "venue", 0),
    ("construction", "neighbors", 2),
    ("survey", "venue", 0),
    ("strangers", "venue", 0),
    ("broker_known", "broker", 0),
]
GOSSIP_VENUES = ["V01", "V03", "V04", "V05"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_field_v5.yaml")
    ap.add_argument("--out", default="configs/events_v5_seed85.yaml")
    ap.add_argument("--months", type=int, default=60)
    ap.add_argument("--total", type=int, default=28)
    args = ap.parse_args()

    with open(os.path.join(ROOT, args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(ROOT, cfg["personas_file"]), encoding="utf-8") as f:
        personas = yaml.safe_load(f)

    agents = build_roster(personas, cfg["agents"], cfg["scenario"])
    hh = [a.agent_id for a in agents if a.role == "household"]
    bz = [a.agent_id for a in agents if a.role == "business"]
    brokers = [a.agent_id for a in agents if a.role == "broker"]
    muni = next(a.agent_id for a in agents if a.role == "municipality")
    acquirer = next(a for a in agents if a.role == "acquirer")
    holders = list(acquirer.extra.get("aliases", [acquirer.name]))
    if len(holders) < 4:
        holders = (holders + ["A社", "B社", "C社", "D社"])[:4]

    parcels = build_town(cfg["world"], hh, bz, muni)
    assign_tenancies(parcels, bz, 0)
    blocks = list(cfg["world"]["block_names"])
    by_block = {b: [p.pid for p in parcels if p.block == b and p.use != "public"]
                for b in blocks}

    rng = random.Random(int(cfg.get("seed", 85)))

    # --- 取得の順番 ---------------------------------------------------------
    # 第1〜12月は湾岸観光地区、第13〜36月は中央駅前地区へ拡大、第37月以降は残りの地区。
    pool_a = list(by_block[blocks[0]]); rng.shuffle(pool_a)
    pool_b = list(by_block[blocks[1]]); rng.shuffle(pool_b)
    pool_c = list(by_block[blocks[2]]) + list(by_block[blocks[3]]); rng.shuffle(pool_c)

    schedule = []          # (month, parcel_id, under_name)
    used = set()

    def take(pools):
        for pool in pools:
            while pool:
                pid = pool.pop(0)
                if pid not in used:
                    used.add(pid)
                    return pid
        return None

    for month in range(1, args.months + 1):
        if len(schedule) >= args.total:
            break
        if month <= 12:
            n = rng.choice([1, 1, 2])
            pools, names = [pool_a, pool_b], holders[:2]
        elif month <= 36:
            n = rng.choice([0, 1, 1, 2])
            pools, names = [pool_b, pool_a, pool_c], holders
        else:
            n = rng.choice([0, 0, 1])
            pools, names = [pool_c, pool_b, pool_a], holders
        for _ in range(n):
            if len(schedule) >= args.total:
                break
            pid = take(pools)
            if pid is None:
                break
            schedule.append((month, pid, rng.choice(names)))

    # --- 兆候 ---------------------------------------------------------------
    acquisitions = []
    for i, (month, pid, name) in enumerate(schedule, start=1):
        traces = [{"kind": "registry", "month": month, "audience": "registry"}]
        for kind, kind_aud, delay in rng.sample(TRACE_POOL, k=rng.choice([0, 1, 1, 2])):
            if kind_aud == "neighbors":
                aud = "neighbors"
            elif kind_aud == "broker":
                aud = f"agents:[{rng.choice(brokers)}]"
            else:
                aud = f"venue:{rng.choice(GOSSIP_VENUES)}"
            m = min(month + delay, args.months)
            traces.append({"kind": kind, "month": m, "audience": aud})
        acquisitions.append({
            "id": f"ACQ{i:02d}",
            "month": month,
            "parcel_id": pid,
            "under_name": name,
            "traces": traces,
        })

    # 第37月以降：既に取得済みの区画に「共通支配が見えやすい動き」が現れる
    acquired = [a for a in acquisitions if a["month"] <= 30]
    late_months = [m for m in range(37, args.months + 1)]
    for m in late_months[::6]:
        if not acquired:
            break
        target = rng.choice(acquired)
        target["traces"].append({"kind": "tenant_swap", "month": m,
                                 "audience": f"venue:{rng.choice(GOSSIP_VENUES)}"})
        target2 = rng.choice(acquired)
        target2["traces"].append({"kind": "renovation_sweep", "month": min(m + 1, args.months),
                                  "audience": "neighbors"})

    doc = {
        "meta": {
            "seed": int(cfg.get("seed", 85)),
            "months": args.months,
            "holders": holders,
            "generated_by": "tools/build_events_v5.py",
            "note": ("取得と兆候の台本。走行前に凍結し、結果を見て調整しない。"
                     "主体の判断には一切触れない（世界で起きた事実のみ）。"),
        },
        "acquisitions": acquisitions,
    }
    out = os.path.join(ROOT, args.out)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False, width=100)
    print(f"[out] {out}  取得 {len(acquisitions)}件 / "
          f"第1〜12月 {sum(1 for a in acquisitions if a['month'] <= 12)}件 / "
          f"兆候 {sum(len(a['traces']) for a in acquisitions)}本")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
