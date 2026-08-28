#!/usr/bin/env python
"""v5e の台本 = v5d の台本の **一等地イベントの月だけ** を第12月→第15月へ移したもの。

    python tools/build_events_v5e.py

設計の正は `docs/world_design_v5e.md` §4。ここでするのは1つだけ：
`configs/events_v5d_seed85.yaml` を読み、`meta.prime_event.parcel_id`（P04）の取得1件の
`month` を `meta.prime_event.month`（12）から 15 へ変える。他の取得
（順番・区画・名義・kind・note・兆候）は v5d と1バイトも変えない。

**乱数を一切使わない**（回し直すと取得そのものが変わる）。取得の不変は
「v5d の acquisitions をそのまま写す」ことで構造的に保証される。
`id` は振り直さない（ACQ18 が第15月に来る＝id は月に対して単調でなくなる。これは意図どおり）。
出力の並びは (month, 元の並び順) の安定ソート。
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.field_v5 import validate_script_v5  # noqa: E402

SRC = "configs/events_v5d_seed85.yaml"
OUT = "configs/events_v5e_seed85.yaml"
NEW_MONTH = 15

NOTE_V5E = (
    "一等地イベント ACQ18/P04 を第12月→第15月へ移動。他の取得"
    "（順番・区画・名義・kind・note・兆候）は v5d と完全に同一。"
    "steps:36 で走らせるため ACQ46（第39月・P28）の1件だけが発火しない")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    with open(os.path.join(ROOT, args.src), encoding="utf-8") as f:
        src = yaml.safe_load(f)

    meta = dict(src.get("meta", {}))
    prime = dict(meta.get("prime_event") or {})
    pid = str(prime.get("parcel_id", ""))
    old_month = int(prime.get("month", 0))
    delta = NEW_MONTH - old_month
    if not pid or not old_month:
        raise ValueError("meta.prime_event に parcel_id / month が無い")

    moved = 0
    acquisitions = []
    for order, acq in enumerate(src["acquisitions"]):
        out_acq = {k: v for k, v in acq.items()}
        if out_acq["parcel_id"] == pid:
            if int(out_acq["month"]) != old_month:
                raise ValueError(
                    f"一等地の取得の月が meta と食い違う: {out_acq['id']} "
                    f"month={out_acq['month']} meta={old_month}")
            out_acq["month"] = NEW_MONTH
            # 兆候は取得に対する相対の位置を保つ（既定は取得の月そのもの）。
            traces = []
            for tr in (out_acq.get("traces") or []):
                tr2 = dict(tr)
                tr2["month"] = int(tr.get("month", old_month)) + delta
                traces.append(tr2)
            if "traces" in out_acq:
                out_acq["traces"] = traces
            moved += 1
        acquisitions.append((int(out_acq["month"]), order, out_acq))
    if moved != 1:
        raise ValueError(f"一等地の取得が1件でない: {moved}")

    acquisitions.sort(key=lambda t: (t[0], t[1]))
    acquisitions = [a for _, _, a in acquisitions]

    prime["month"] = NEW_MONTH
    meta["prime_event"] = prime
    meta["generated_by"] = "tools/build_events_v5e.py"
    meta["source"] = args.src
    meta["note_v5e"] = NOTE_V5E

    doc = {"meta": meta, "acquisitions": acquisitions}
    validate_script_v5(doc)

    out = os.path.join(ROOT, args.out)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False, width=100)

    if getattr(sys.stdout, "reconfigure", None):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    by_month = collections.Counter(int(a["month"]) for a in acquisitions)
    traces = sum(len(a.get("traces") or []) for a in acquisitions)
    print(f"[src] {args.src}")
    print(f"[out] {args.out}")
    print(f"acquisitions {len(acquisitions)} / traces {traces}")
    print(f"prime {pid}: 第{old_month}月 -> 第{NEW_MONTH}月 (delta {delta:+d})")
    print(f"第12月 {by_month.get(12, 0)}件 / 第15月 {by_month.get(15, 0)}件")
    print("36か月で発火しない取得: "
          + ", ".join(f"{a['id']}(第{a['month']}月/{a['parcel_id']})"
                      for a in acquisitions if int(a["month"]) > 36))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
