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

# 手放した事情の筋書き（評価語なし・v4.1 の非金銭ペルソナと矛盾しない範囲）。
# seed で選ぶだけで、主体の判断には一切触れない。
# 事情は区画の用途と矛盾しないものだけを使う（空地に「店を閉めた」を割り当てない）。
# 走行前に固定した定型で、seed で選ぶだけ。主体の判断には触れない。
SALE_NOTES = {
    "residential": [
        "相続した家で、遠方に住んでいて管理の手が回らなくなった",
        "転居することになり、住まいを引き払った",
        "築古で改修の手間が重く、手放すことにした",
    ],
    "shop": [
        "後継ぎがおらず、続けていく見通しが立たなくなった",
        "店を閉めることにした",
        "建物の傷みが進み、続けるのが難しくなった",
    ],
    "vacant": [
        "空き地のまま草刈りだけが続いていた",
        "相続した土地で、使い道が決まらないままだった",
        "遠方に住んでいて手入れに通えなくなった",
    ],
}
LEASE_NOTES = {
    "residential": [
        "自分では使わなくなった家を貸すことにした",
        "空いたままの家に住み手を入れることにした",
    ],
    "shop": [
        "店を閉めたあと、建物を貸すことにした",
        "建物はそのままに、使い手を入れることにした",
    ],
    "vacant": [
        "空き地のまま置いておくより、借り手に使ってもらうことにした",
        "駐車場として使いたいという話があり、貸すことにした",
    ],
}


def _notes_for(deal, use):
    table = LEASE_NOTES if deal == "lease" else SALE_NOTES
    return table.get(use, table["residential"])
# 賃借は登記が動かない＝窓口では見えない。代わりに看板や店子の入れ替わりで見える。
LEASE_VISIBLE = ["sign_change", "tenant_swap"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_field_v5.yaml")
    ap.add_argument("--out", default="configs/events_v5b_seed85.yaml")
    ap.add_argument("--months", type=int, default=60)
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
    tenanted = {p.pid: bool(p.tenant_id) for p in parcels}
    use_of = {p.pid: p.use for p in parcels}
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

    # v5 は総数28件で打ち切っていたため第24月で台本が尽き、第25月以降が0件だった。
    # 上限を外し、設計の正 §2 のレート規則のまま回す（非公共区画が尽きたら止まる）。
    # 総数は結果として決まる値であり、狙って決めた数ではない。
    for month in range(1, args.months + 1):
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
            pid = take(pools)
            if pid is None:
                break
            schedule.append((month, pid, rng.choice(names)))

    # --- 兆候 ---------------------------------------------------------------
    acquisitions = []
    for i, (month, pid, name) in enumerate(schedule, start=1):
        # 売買か賃借か。賃借は登記が動かない（施主追記 2026-08-27 22:27）。
        deal = "lease" if rng.random() < 0.25 else "sale"
        # 既に使い手が居る区画を賃借にすると、その店子を無通知で追い出すことになる
        # （Codexレビュー 2026-08-28）。立ち退きという機構は世界に無いので、
        # 使い手が居る区画は賃借にしない。
        if deal == "lease" and tenanted.get(pid):
            deal = "sale"
        note = rng.choice(_notes_for(deal, use_of.get(pid, "residential")))
        if deal == "sale":
            traces = [{"kind": "registry", "month": month, "audience": "registry"}]
        else:
            traces = [{"kind": rng.choice(LEASE_VISIBLE), "month": month,
                       "audience": f"venue:{rng.choice(GOSSIP_VENUES)}"}]
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
            "kind": deal,
            "note": note,
            "traces": traces,
        })

    # v5 にあった「第37月以降に共通支配が見えやすい動きを足す」処理は削除した。
    # 結果の方向を持つ処置であり、施主指示（2026-08-28 00:53「兆候出したら意味ない。
    # 初心に帰れ」）に反する。名義が同じ主体のものだと分かる材料は、登記の反復・
    # 売主の発言・仲介の見聞という既存の物理だけに限る。

    doc = {
        "meta": {
            "seed": int(cfg.get("seed", 85)),
            "months": args.months,
            "holders": holders,
            "generated_by": "tools/build_events_v5.py",
            "note": ("取得と兆候の台本。走行前に凍結し、結果を見て調整しない。"
                     "主体の判断には一切触れない（世界で起きた事実のみ）。"
                     "kind=sale は登記が移る／kind=lease は登記は動かず使い手が変わる。"),
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
