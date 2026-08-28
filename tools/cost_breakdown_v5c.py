#!/usr/bin/env python
"""v5c の実ランから API 費用の内訳を実測して出す（報告用・本体には影響しない）。

    python tools/cost_breakdown_v5c.py

数値の出所は各 run_dir の `summary.json` の `usage.by_tag`（実際に API が返した
トークン数）と `configs/config_field_v5c.yaml` の `cost.price_table` だけ。
手打ちの数字は使わない。`docs/cost_breakdown_v5c.md` はこの出力から書いている。
"""

from __future__ import annotations

import collections
import io
import json
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUNS = [
    ("runA", "2026-08-28_1016_100_field_v5c_runA"),
    ("runB", "2026-08-28_1110_100_field_v5c_runB"),
    ("runC", "2026-08-28_1110_100_field_v5c_runC"),
]

ROLE_JA = {
    "household": "住民（16人）",
    "business": "事業者（4）",
    "broker": "仲介（2）",
    "municipality": "行政（2）",
    "media": "記者（2）",
    "classifier": "分類器（事後）",
}


def price(model: str):
    with open(os.path.join(ROOT, "configs", "config_field_v5c.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["cost"]["price_table"][model]


def cost(inp, cached, out, p):
    return ((inp - cached) / 1e6 * p["input"]
            + cached / 1e6 * p["cache_read"]
            + out / 1e6 * p["output"])


def load(run_dir):
    with open(os.path.join(ROOT, "simulations", run_dir, "summary.json"),
              encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    if sys.stdout is not None:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    p = None
    grand = 0.0
    for label, run_dir in RUNS:
        s = load(run_dir)
        p = p or price(s["model"])
        u = s["usage"]
        by_role = collections.defaultdict(lambda: [0, 0, 0, 0])
        by_slot = collections.defaultdict(lambda: [0, 0, 0, 0])
        for tag, v in u["by_tag"].items():
            role = "classifier" if tag.startswith("classify") else tag.split(":")[1]
            slot = tag if tag.startswith("classify") else tag.split(":")[2]
            for key, d in ((role, by_role), (slot, by_slot)):
                a = d[key]
                a[0] += v["calls"]
                a[1] += v["input"]
                a[2] += v.get("cached", 0)
                a[3] += v["output"]
        total = cost(u["input_tokens"], u["cached_tokens"], u["output_tokens"], p)
        grand += total
        print(f"=== {label} / {run_dir}  {s['steps']}か月 {s['model']}")
        print(f"    calls {u['calls']}  input {u['input_tokens']:,}"
              f"（うちキャッシュ読み {u['cached_tokens']:,}"
              f"＝{100 * u['cached_tokens'] / u['input_tokens']:.1f}%）"
              f"  output {u['output_tokens']:,}  ${total:.4f}"
              f"  1か月あたり ${total / s['steps']:.4f}")
        print("    --- 呼び出し種別ごと ---")
        for k, a in sorted(by_role.items(), key=lambda kv: -cost(kv[1][1], kv[1][2], kv[1][3], p)):
            c = cost(a[1], a[2], a[3], p)
            print(f"    {ROLE_JA.get(k, k):<16} calls={a[0]:>5} in={a[1]:>9,}"
                  f" cached={100 * a[2] / max(1, a[1]):>4.1f}% out={a[3]:>8,}"
                  f"  ${c:.4f}  ({100 * c / total:.1f}%)")
        print("    --- 呼び出し枠ごと（1コールあたり） ---")
        for k in sorted(by_slot):
            a = by_slot[k]
            print(f"    {k:<22} calls={a[0]:>5} in/call={a[1] / a[0]:>6.0f}"
                  f" cached={100 * a[2] / max(1, a[1]):>4.1f}% out/call={a[3] / a[0]:>5.0f}"
                  f"  ${cost(a[1], a[2], a[3], p):.4f}")
    print(f"=== 3本合計 ${grand:.4f}")

    # --- 節約案の試算（採用済みの 1〜3 と、その先） ---------------------------
    s = load(RUNS[2][1])
    bt = s["usage"]["by_tag"]
    sim = [0, 0, 0]
    cls = [0, 0, 0]
    for tag, v in bt.items():
        d = cls if tag.startswith("classify") else sim
        d[0] += v["input"]
        d[1] += v.get("cached", 0)
        d[2] += v["output"]
    base = cost(*sim, p) + cost(*cls, p)

    def scen(name, batch_sim, batch_cls, cache_share, out_sim, out_cls,
             sim_scale=1.0, cls_scale=1.0):
        si, so = sim[0] * sim_scale, sim[2] * out_sim * sim_scale
        ci, co = cls[0] * cls_scale, cls[2] * out_cls * cls_scale
        c = (cost(si, si * cache_share, so, p) * (0.5 if batch_sim else 1.0)
             + cost(ci, 0, co, p) * (0.5 if batch_cls else 1.0))
        print(f"    {name:<44} ${c:.4f}  基準の {100 * c / base:.0f}%  （1/{base / c:.1f}）")
        return c

    print(f"=== 節約案の試算（基準＝runC ${base:.4f}）")
    scen("現状", False, False, sim[1] / sim[0], 1.0, 1.0)
    scen("A案 分類のみBatch＋キャッシュ35%＋出力-25%", False, True, 0.35, 0.75, 0.70)
    scen("B案 全Batch＋キャッシュ55%＋出力-50%", True, True, 0.55, 0.50, 0.50)
    scen("B案＋同席8人上限（コール-35%）", True, True, 0.55, 0.50, 0.50, 0.65, 0.65)
    scen("B案＋同席8人上限＋2巡→1巡", True, True, 0.55, 0.50, 0.50, 0.455, 0.455)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
