#!/usr/bin/env python
"""v5c の台本（取得スケジュールと兆候）を1回だけ生成して YAML に凍結する。

    python tools/build_events_v5c.py --config configs/config_field_v5c.yaml \
        --out configs/events_v5c_seed85.yaml

v5b との違いは **買い手が区画を選ぶ順番だけ**である（施主FB 2026-08-28 09:15
「機械的に買ってってる。占領を進める台本がちょっと適当すぎる」）。
兆候（trace）の種類・本数・可視範囲・月あたりの件数レート・事情（note）の文面は
v5b と同一（`tools/build_events_v5b.py` から直接 import している）。
＝**街に気づかせるための兆候は1本も足していない**（feedback_quiet_no_engineered_dynamics）。

戦略規則（docs/world_design_v5c_buyer_strategy.md §3・走行前に固定）：
  1. 入りやすい所から  … 空地・古い住宅・不在地主・高齢単身・後継者なし を加点
  2. 目立たない所から  … visibility（表通り・角地・大・店舗・中心ほど高い）を減点
  3. 面をつくる        … 既取得の隣接区画を加点
  4. 一等地・角地・大区画・旅館は後半 … 第31月以降に解禁（早い側の在庫が尽きた時だけ前倒し）
  5. 名義を散らす      … 同じ地区で同じ名義を続けない
  6. 月あたりの件数レートは v5b と同じ。総数は非公共区画が尽きるまで。
これは**買い手の合理性**であって、街に気づかせるための調整ではない
（visibility の低い所から入る＝むしろ気づかれにくい方向に働く）。
生成後に結果を見て台本を触らない。

一等地イベント（施主指示②）：台本に1件だけ、**この街で最も目立つ売買**を第12〜14月に
固定する。定義は「店舗・旅館のうち visibility 最大（同点なら面積最大・なお同点なら pid 順）」。
※設計 §3 の「size 大」は、この世界では大区画（320/450㎡）が周縁の x=6,7 にしか無く
中心 zone に存在しないため、満たせる区画が1つも無い。よって**走行前に**上の定義へ固定した
（結果を見て選び直してはいない）。兆候の規則は他の取得と同じ＝目立つ取引だから兆候を増やす、はしない。
"""

from __future__ import annotations

import argparse
import collections
import os
import random
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from src.agents import build_roster                    # noqa: E402
from src.world import assign_tenancies, build_town, neighbors   # noqa: E402
# 兆候・事情の規則は v5b をそのまま使う（複製せず import して同一性を担保する）。
from build_events_v5b import (GOSSIP_VENUES, LEASE_VISIBLE,     # noqa: E402
                              TRACE_POOL, _notes_for)

# --- 戦略の重み（走行前に固定・結果を見て調整しない） ----------------------
EASE_USE = {"空地": 3, "古い住宅": 3, "住宅": 0, "事業所": -1, "店舗": -1, "旅館": -1}
EASE_OWNER = {"不在地主": 3, "高齢単身": 2, "後継者なし": 2, "現役世帯": 0,
              "営業中の店": -2}
ADJACENCY_BONUS = 4.0        # 既取得区画に隣接していれば加点（面をつくる）
JITTER = 0.5                 # 同点を seed で崩すだけの微小な揺らぎ
LATE_FROM_MONTH = 31         # 一等地・角地・大区画・旅館は後半（第31月以降）
LEASE_RATE = 0.25            # v5b と同じ
LEASE_RATE_LODGING = 0.6     # 旅館・宿は賃借で入ることが多い（設計 §3-4）
PRIME_MONTH_CHOICES = [12, 13, 14]


def is_late_tier(row) -> bool:
    """後半に回す区画＝一等地の要素を持つもの（角地・大区画・旅館）。"""
    return (row["frontage"] == "角地" or row["size_class"] == "大"
            or row["use_detail"] == "旅館")


def ease_score(row) -> float:
    return (EASE_USE.get(row["use_detail"], 0)
            + EASE_OWNER.get(row["owner_profile"], 0))


def pick_prime(rows):
    """一等地イベントの対象＝**営業中の**店舗・旅館のうち最も目立つ区画。

    Codexレビュー 2026-08-28（走行前）：use_detail だけで選ぶと、店舗用途でも
    使い手の居ない区画（＝営業していない）が当たり、設計 §3 の「営業中の店舗 or 旅館」
    に反する。owner_profile が「営業中の店」であることを条件に加える。
    """
    cands = [r for r in rows if r["use_detail"] in ("店舗", "旅館")
             and r["owner_profile"] == "営業中の店"]
    return sorted(cands, key=lambda r: (-r["visibility"], -r["area_sqm"], r["pid"]))[0]


def choose_holder(holders, block, month, used_by_block, rng):
    """同じ地区で同じ名義を続けない（同月・前月に使った名義を避ける）。"""
    recent = {h for h, m in used_by_block.get(block, []) if month - m <= 1}
    pool = [h for h in holders if h not in recent] or \
           [h for h in holders if h not in {h2 for h2, m in used_by_block.get(block, [])
                                            if m == month}] or list(holders)
    return rng.choice(pool)


def build_traces(month, deal, months, brokers, rng):
    """兆候の規則は v5b と完全に同一。"""
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
        traces.append({"kind": kind, "month": min(month + delay, months),
                       "audience": aud})
    return traces


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_field_v5c.yaml")
    ap.add_argument("--parcels", default="configs/parcels_v5c.yaml")
    ap.add_argument("--out", default="configs/events_v5c_seed85.yaml")
    ap.add_argument("--months", type=int, default=60)
    args = ap.parse_args()

    with open(os.path.join(ROOT, args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(ROOT, cfg["personas_file"]), encoding="utf-8") as f:
        personas = yaml.safe_load(f)
    with open(os.path.join(ROOT, args.parcels), encoding="utf-8") as f:
        attrs = yaml.safe_load(f)

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
    pmap = {p.pid: p for p in parcels}
    tenanted = {p.pid: bool(p.tenant_id) for p in parcels}
    use_of = {p.pid: p.use for p in parcels}
    nb = {p.pid: neighbors(pmap, p.pid) for p in parcels}

    rows = {r["pid"]: r for r in attrs["parcels"] if r["use"] != "public"}
    prime = pick_prime(list(rows.values()))

    rng = random.Random(int(cfg.get("seed", 85)))
    prime_month = rng.choice(PRIME_MONTH_CHOICES)

    # 一等地は通常の選択から外すが、**まだ買っていない**（Codexレビュー 2026-08-28）。
    # 隣接ボーナスは実際に取得済みの区画だけを見る＝前半の順番が未来を知らない。
    reserved: set = {prime["pid"]}
    taken: set = set()                   # 実際に取得した区画
    schedule = []                        # (month, pid, holder)
    used_by_block = collections.defaultdict(list)
    early_exhausted_at = None

    def pick(month):
        """その月に買う1区画を戦略規則で選ぶ。"""
        nonlocal early_exhausted_at
        pool = [r for r in rows.values()
                if r["pid"] not in taken and r["pid"] not in reserved]
        if not pool:
            return None
        eligible = [r for r in pool
                    if not is_late_tier(r) or month >= LATE_FROM_MONTH]
        if not eligible:                 # 早い側の在庫が尽きたら後半枠を前倒しする
            if early_exhausted_at is None:
                early_exhausted_at = month
            eligible = pool

        def score(r):
            adj = ADJACENCY_BONUS if any(n in taken for n in nb[r["pid"]]) else 0.0
            return (ease_score(r) - float(r["visibility"]) + adj
                    + rng.random() * JITTER)

        return max(eligible, key=score)

    for month in range(1, args.months + 1):
        if month <= 12:
            n = rng.choice([1, 1, 2])
        elif month <= 36:
            n = rng.choice([0, 1, 1, 2])
        else:
            n = rng.choice([0, 0, 1])
        if month == prime_month:
            # 一等地はその月の**取得枠の1件を置き換える**（足さない）。月あたりの件数
            # レートは v5b と同一のままにする（Codexレビュー 2026-08-28）。
            holder = choose_holder(holders, prime["block"], month, used_by_block, rng)
            schedule.append((month, prime["pid"], holder))
            used_by_block[prime["block"]].append((holder, month))
            taken.add(prime["pid"])
            n = max(0, n - 1)
        for _ in range(n):
            row = pick(month)
            if row is None:
                break
            taken.add(row["pid"])
            holder = choose_holder(holders, row["block"], month, used_by_block, rng)
            schedule.append((month, row["pid"], holder))
            used_by_block[row["block"]].append((holder, month))

    schedule.sort(key=lambda s: (s[0], s[1]))

    acquisitions = []
    for i, (month, pid, name) in enumerate(schedule, start=1):
        row = rows[pid]
        rate = LEASE_RATE_LODGING if row["use_detail"] == "旅館" else LEASE_RATE
        deal = "lease" if rng.random() < rate else "sale"
        # 使い手が居る区画を賃借にすると店子を無通知で追い出すことになる（v5b と同じ判断）。
        if deal == "lease" and tenanted.get(pid):
            deal = "sale"
        note = rng.choice(_notes_for(deal, use_of.get(pid, "residential")))
        acquisitions.append({
            "id": f"ACQ{i:02d}",
            "month": month,
            "parcel_id": pid,
            "under_name": name,
            "kind": deal,
            "note": note,
            "traces": build_traces(month, deal, args.months, brokers, rng),
            # 台本の読み手（人間）のための注記。世界にもプロンプトにも渡らない。
            "why": (f"{row['zone']}/{row['frontage']}/{row['size_class']}/"
                    f"{row['use_detail']}/{row['owner_profile']}"
                    f"（目立ちやすさ{row['visibility']}）"
                    + ("・一等地イベント" if pid == prime["pid"] else "")),
        })

    doc = {
        "meta": {
            "seed": int(cfg.get("seed", 85)),
            "months": args.months,
            "holders": holders,
            "generated_by": "tools/build_events_v5c.py",
            "strategy": {
                "ease_use": EASE_USE, "ease_owner": EASE_OWNER,
                "adjacency_bonus": ADJACENCY_BONUS, "jitter": JITTER,
                "late_tier_from_month": LATE_FROM_MONTH,
                "late_tier": "角地 or 大区画 or 旅館",
                "early_pool_exhausted_at": early_exhausted_at,
                "lease_rate": LEASE_RATE, "lease_rate_lodging": LEASE_RATE_LODGING,
            },
            "prime_event": {"parcel_id": prime["pid"], "month": prime_month,
                            "zone": prime["zone"], "frontage": prime["frontage"],
                            "size_class": prime["size_class"],
                            "area_sqm": prime["area_sqm"],
                            "use_detail": prime["use_detail"],
                            "visibility": prime["visibility"]},
            "note": ("取得と兆候の台本。走行前に凍結し、結果を見て調整しない。"
                     "主体の判断には一切触れない（世界で起きた事実のみ）。"
                     "kind=sale は登記が移る／kind=lease は登記は動かず使い手が変わる。"
                     "兆候の規則は v5b と同一（build_events_v5b から import）。"),
        },
        "acquisitions": acquisitions,
    }
    out = os.path.join(ROOT, args.out)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False, width=100)

    in24 = [a for a in acquisitions if a["month"] <= 24]
    print(f"[out] {out}  取得 {len(acquisitions)}件 / 第1〜24月 {len(in24)}件 / "
          f"兆候 {sum(len(a['traces']) for a in acquisitions)}本")
    print(f"      一等地イベント {prime['pid']} 第{prime_month}月 "
          f"({prime['zone']}/{prime['frontage']}/{prime['area_sqm']}㎡/"
          f"{prime['use_detail']}/目立ちやすさ{prime['visibility']})")
    print("      第1〜24月の目立ちやすさ",
          dict(sorted(collections.Counter(rows[a["parcel_id"]]["visibility"]
                                          for a in in24).items())))
    print("      第1〜24月の用途",
          dict(collections.Counter(rows[a["parcel_id"]]["use_detail"]
                                   for a in in24).most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
