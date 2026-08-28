#!/usr/bin/env python
"""v5d の台本 = v5c の台本の **兆候だけ** を濾したもの。

    python tools/build_events_v5d.py

設計の正は `docs/world_design_v5d.md` §3。ここでするのは1つだけ：
`configs/events_v5c_seed85.yaml` を読み、**取得の順番・月・区画・名義・kind・note を
1バイトも変えずに** traces を規則で濾して `configs/events_v5d_seed85.yaml` に書く。

RNG は回し直さない（回し直すと取得そのものが変わる）。取得の不変は
「v5c の acquisitions をそのまま写す」ことで構造的に保証される。

濾す規則（docs/world_design_v5d.md §3 の表・根拠つき）:
  registry      … 全削除。窓口を廃止したので配送先が無い（§2）
  construction  … 全削除。買い手は当面 解体も新築もしない＝工事は起きない
  sign_change   … 賃借かつ対象区画が店舗のときだけ残す
  tenant_swap   … 賃借のみ残す
  moving_out    … 売買かつ「売主本人がその区画を使っていた」ときだけ残す
                  「使っていた」＝ parcels_v5c.yaml で tenant_id が空 かつ
                  use != "vacant" かつ owner_profile が「不在地主」でない
  survey        … 売買のみ残す（引渡し前の境界確定測量）
  strangers     … 売買のみ残す（買う前の現地確認・仲介の案内）
  broker_known  … 全部残す（仲介は職業上 名義の移転を知る）
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SRC = "configs/events_v5c_seed85.yaml"
PARCELS = "configs/parcels_v5c.yaml"
OUT = "configs/events_v5d_seed85.yaml"
DIFF = "docs/events_v5d_trace_diff.md"

TRACE_KINDS = ["registry", "construction", "sign_change", "tenant_swap",
               "moving_out", "survey", "strangers", "broker_known"]


def seller_lived_there(row: dict) -> bool:
    """売主本人がその区画を使っていたか（docs/world_design_v5d.md §3 の定義）。

    店子がいない（＝人に貸していない）／空地でない（＝使っている建物がある）／
    所有者が不在地主でない（＝そこに居る）、の3つを満たすときだけ true。
    """
    return (not str(row.get("tenant_id", "") or "").strip()
            and row.get("use") != "vacant"
            and row.get("owner_profile") != "不在地主")


def keep_trace(kind: str, deal: str, row: dict) -> bool:
    if kind in ("registry", "construction"):
        return False
    if kind == "sign_change":
        return deal == "lease" and row.get("use") == "shop"
    if kind == "tenant_swap":
        return deal == "lease"
    if kind == "moving_out":
        return deal == "sale" and seller_lived_there(row)
    if kind in ("survey", "strangers"):
        return deal == "sale"
    if kind == "broker_known":
        return True
    raise ValueError(f"未知の兆候: {kind}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--parcels", default=PARCELS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--diff", default=DIFF)
    args = ap.parse_args()

    with open(os.path.join(ROOT, args.src), encoding="utf-8") as f:
        src = yaml.safe_load(f)
    with open(os.path.join(ROOT, args.parcels), encoding="utf-8") as f:
        rows = {r["pid"]: r for r in yaml.safe_load(f)["parcels"]}

    before = collections.Counter()
    after = collections.Counter()
    acquisitions = []
    for acq in src["acquisitions"]:
        deal = str(acq.get("kind", "sale"))
        row = rows[acq["parcel_id"]]
        traces = []
        for tr in acq.get("traces", []):
            before[(tr["kind"], deal)] += 1
            if keep_trace(tr["kind"], deal, row):
                after[(tr["kind"], deal)] += 1
                traces.append(dict(tr))
        # 取得そのものは1バイトも触らない（traces のキーだけ差し替える）。
        out_acq = {k: v for k, v in acq.items()}
        out_acq["traces"] = traces
        acquisitions.append(out_acq)

    meta = dict(src.get("meta", {}))
    meta["generated_by"] = "tools/build_events_v5d.py"
    meta["source"] = args.src
    meta["note"] = (
        "取得と兆候の台本（v5d）。取得（順番・月・区画・名義・kind・note）は "
        "configs/events_v5c_seed85.yaml と完全に同一で、兆候だけを "
        "docs/world_design_v5d.md §3 の規則で濾したもの。走行前に凍結し、"
        "結果を見て調整しない。registry と construction は全削除"
        "（窓口を廃止したので配送先が無い／買い手は当面 解体も新築もしない）。")
    doc = {"meta": meta, "acquisitions": acquisitions}

    out = os.path.join(ROOT, args.out)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False, width=100)

    lines = ["# v5d 台本 — 兆候の before/after（種類×取引種別）", "",
             "`tools/build_events_v5d.py` が機械生成する。取得46件（順番・月・区画・"
             "名義・kind・note）は v5c と完全に同一で、変わったのは traces だけである。",
             "", "| 種類 | 取引種別 | v5c | v5d | 差 |", "|---|---|---|---|---|"]
    for kind in TRACE_KINDS:
        for deal in ("sale", "lease"):
            b, a = before[(kind, deal)], after[(kind, deal)]
            if b or a:
                lines.append(f"| {kind} | {deal} | {b} | {a} | {a - b:+d} |")
    tb, ta = sum(before.values()), sum(after.values())
    lines += [f"| **合計** | — | **{tb}** | **{ta}** | **{ta - tb:+d}** |", "",
              f"取得件数 {len(acquisitions)}（v5c と同一）／"
              f"兆候ゼロの取得 {len([a for a in acquisitions if not a['traces']])}件"]
    with open(os.path.join(ROOT, args.diff), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    if getattr(sys.stdout, "reconfigure", None):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n".join(lines))
    print(f"\n[out] {out}")
    print(f"[out] {os.path.join(ROOT, args.diff)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
