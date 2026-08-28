#!/usr/bin/env python
"""既に走り終わった run を、v5e の新しい赤（自衛の具体的な行動）で読み直す。

    python tools/reclassify_v5e.py <run_dir> --dry-run
    python tools/reclassify_v5e.py <run_dir> --provider google

`utterances_v5.jsonl` / `thoughts_all.jsonl` / `articles_v5.jsonl` から
**事後の occ_rows と同じ組み立て**で行を作り、`classify_stage_v5e` を回して
`stage_labels_v5e.jsonl` を書く（これは観測であって世界には戻らない）。

これは**監査用の別成果物**であり、run の正本には一切触らない。出力先は
`run_dir/reclass_v5e/`（labels.jsonl と summary.json）で、run 直下の
`stage_labels_v5e.jsonl` / `stage_labels_v5c.jsonl` は書かない
（Codexレビュー 2026-08-29：正本を上書きすると、実際に取得を止めた分類器の出力と
報告値が食い違う。run 直下に v5e ラベルを置くと v5c/v5d の集計が v5e の意味に化ける）。
**v5e の run は受け付けない**（正本が既にある＝「結果を見てから読み直す」道を塞ぐ）。
シミュ本体は動かさない。
`--dry-run` は API を一切叩かず、行数・チャンク数・概算費用だけを出す。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from src.kpi import (STAGE_SYSTEM_V5E, build_stage_prompt_v5e,  # noqa: E402
                     classify_stage_v5e)
from src.llm_client_factory import UsageMeter, create_llm_client  # noqa: E402
from estimate_cost import estimate_tokens  # noqa: E402
import run_metrics  # noqa: E402
from src.stage_v5e import defense_level_of, rule_red_v5e, stage_v5e  # noqa: E402

# 出力の想定トークン（1行あたり）。results の1要素は id と5つのフラグだけなので
# 実測でも 25 前後に収まる。概算にしか使わない。
OUT_TOKENS_PER_ROW = 25


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    with io.open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_rows(run_dir):
    """事後の occ_rows と同じ組み立て（utterance → thought → article の順）。"""
    return ([{"kind": "utterance", **u}
             for u in _read_jsonl(os.path.join(run_dir, "utterances_v5.jsonl"))]
            + [{"kind": "thought", **t}
               for t in _read_jsonl(os.path.join(run_dir, "thoughts_all.jsonl"))]
            + [{"kind": "article", **a}
               for a in _read_jsonl(os.path.join(run_dir, "articles_v5.jsonl"))])


def _price(cfg, model):
    table = (cfg.get("cost") or {}).get("price_table") or {}
    return table.get(model)



def _holders_acquired(run_dir, n_steps):
    """`tools/run_metrics.py` と同じ作り方（台帳の transfer / lease から）。"""
    recs = _read_jsonl(os.path.join(run_dir, "ledger.jsonl"))
    hs, ps = set(), set()
    holders, acquired = {}, {}
    for step in range(1, n_steps + 1):
        for r in recs:
            if r.get("kind") in ("transfer", "lease") and int(r.get("step", 0)) <= step:
                hs.add(str(r.get("under_name", "")))
                ps.add(r.get("parcel_id"))
        holders[step], acquired[step] = set(hs), set(ps)
    return holders, acquired


def _key(r):
    return (int(r.get("step", 0) or 0), r.get("kind", ""), r.get("from", ""),
            str(r.get("text", "")))


def _write_audit(run_dir, out_dir, labels):
    """旧赤（v5c）と新赤（v5e）を並べた監査用の summary を書く。

    数えるだけで、判定のコードは `tools/run_metrics.py` と `src/stage_v5e.py` の
    ものをそのまま使う（このツール専用の判定を作らない）。
    """
    with io.open(os.path.join(run_dir, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_metrics._use_parcel_names(str(cfg.get("scenario_version", "")))
    n_steps = int(cfg.get("steps", 24))
    holders, acquired = _holders_acquired(run_dir, n_steps)
    old_by = {_key(r): r for r in _read_jsonl(
        os.path.join(run_dir, "stage_labels_v5c.jsonl"))}

    rows, s_counts, s_first, reds = [], {"S1": 0, "S2": 0, "S3": 0}, {}, []
    old_red = new_red = 0
    for r in labels:
        step = int(r.get("step", 0) or 0)
        text = str(r.get("text", ""))
        hs, ps = holders.get(step, set()), acquired.get(step, set())
        blue = run_metrics._v5c_rule_blue(text, hs, ps)
        base = {"classified": bool(r.get("classified")), "text": text,
                "rule_blue": blue,
                "rule_green": run_metrics._v5c_rule_green(text, hs, ps),
                "rule_yellow": run_metrics._v5c_rule_yellow(text, hs, ps),
                "rule_red": rule_red_v5e(text, blue),
                "llm_deal": bool(r.get("deal")) if r.get("classified") else None,
                "llm_area": bool(r.get("area")) if r.get("classified") else None,
                "llm_same_buyer": (bool(r.get("same_buyer"))
                                   if r.get("classified") else None),
                "llm_defense": bool(r.get("defense")) if r.get("classified") else None,
                "llm_defense_level": r.get("defense_level")}
        st = stage_v5e(base)
        rows.append(st)
        old = old_by.get(_key(r))
        if old is not None:
            o = {"classified": bool(old.get("classified", True))
                 and old.get("deal") is not None,
                 "rule_blue": blue,
                 "rule_green": base["rule_green"], "rule_yellow": base["rule_yellow"],
                 "rule_red": run_metrics._v5c_rule_red(r.get("role", ""), text),
                 "llm_deal": bool(old.get("deal")) if old.get("deal") is not None else None,
                 "llm_area": bool(old.get("area")) if old.get("deal") is not None else None,
                 "llm_same_buyer": (bool(old.get("same_buyer"))
                                    if old.get("deal") is not None else None),
                 "llm_admin": (bool(old.get("admin"))
                               if old.get("deal") is not None else None)}
            if run_metrics._v5c_stage(o) == "red":
                old_red += 1
        if st != "red":
            continue
        new_red += 1
        lv = defense_level_of(base) or {}
        level = lv.get("level")
        if level in s_counts:
            s_counts[level] += 1
            s_first.setdefault(level, {
                "month": step, "agent_id": r.get("from", ""), "role": r.get("role", ""),
                "kind": r.get("kind", ""), "scene": r.get("scene", ""),
                "venue": r.get("venue", ""), "text": text,
                "level_source": lv.get("level_source")})
        reds.append({"month": step, "agent_id": r.get("from", ""),
                     "role": r.get("role", ""), "kind": r.get("kind", ""),
                     "venue": r.get("venue", ""), "level": level,
                     "level_source": lv.get("level_source"), "text": text})

    counts = {c: rows.count(c) for c in ("blue", "green", "yellow", "red")}
    summary = {"run_dir": run_dir, "rows": len(labels),
               "unknown": len([r for r in labels if not r.get("classified")]),
               "stage_counts_v5e": counts,
               "red_old_v5c": old_red, "red_new_v5e": new_red,
               "S_counts": s_counts, "S_first": s_first, "S_rows": reds,
               "note": ("旧赤=v5c の定義（行政の発話に限る）／"
                        "新赤=v5e の定義（立場を問わず、静かな占領への具体的アクション）。"
                        "この summary は監査用で、run の正本には影響しない。")}
    with io.open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[audit] 旧赤 {old_red} 行 -> 新赤 {new_red} 行 / S {s_counts}")
    return summary

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--provider", default=None, choices=["google", "mock"])
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--out-dir", default="reclass_v5e",
                    help="run_dir 配下の出力先（run 直下には書かない）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if getattr(sys.stdout, "reconfigure", None):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    run_dir = args.run_dir
    with io.open(os.path.join(run_dir, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    version = str(cfg.get("scenario_version", ""))
    if version == "field_v5e":
        raise SystemExit("この run は v5e＝stage_labels_v5e.jsonl が正本。"
                         "再分類はしない（取得を止めた出力と食い違わせないため）。")
    if version not in ("field_v5c", "field_v5d"):
        raise SystemExit(f"対象は v5c / v5d の run だけ（この run は {version}）")
    llm = dict(cfg.get("llm") or {})
    if args.provider:
        llm["provider"] = args.provider
    model = llm.get("model", "?")

    rows = build_rows(run_dir)
    chunks = [rows[i:i + args.batch] for i in range(0, len(rows), args.batch)]
    in_tokens = sum(estimate_tokens(STAGE_SYSTEM_V5E)
                    + estimate_tokens(build_stage_prompt_v5e(c)) for c in chunks)
    out_tokens = len(rows) * OUT_TOKENS_PER_ROW
    p = _price(cfg, model)
    est = (None if not p else
           in_tokens / 1e6 * p["input"] + out_tokens / 1e6 * p["output"])

    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"[run] {run_dir}")
    print(f"[model] {model} / provider {llm.get('provider')}")
    print(f"[rows] {len(rows)}  " + " ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    print(f"[chunks] {len(chunks)} (batch {args.batch})")
    print(f"[est tokens] input {in_tokens:.0f} / output {out_tokens}")
    print("[est cost] " + (f"${est:.4f}" if est is not None
                           else f"価格表に {model} が無い"))
    if args.dry_run:
        print("[dry-run] API は叩いていない。ファイルも書いていない。")
        return 0

    out_dir = os.path.join(run_dir, args.out_dir)
    if os.path.abspath(out_dir) == os.path.abspath(run_dir):
        raise SystemExit("run 直下には書かない（正本と混ざる）")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "labels.jsonl")

    usage = UsageMeter()
    client = create_llm_client(llm, usage=usage)
    labels = classify_stage_v5e(client, rows, batch=args.batch)
    with io.open(out_path, "w", encoding="utf-8") as f:
        for r in labels:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    u = usage.as_dict()
    actual = (None if not p else
              ((u["input_tokens"] - u["cached_tokens"]) / 1e6 * p["input"]
               + u["cached_tokens"] / 1e6 * p["cache_read"]
               + u["output_tokens"] / 1e6 * p["output"]))
    cost_path = os.path.join(out_dir, "cost.json")
    with io.open(cost_path, "w", encoding="utf-8") as f:
        json.dump({"run_dir": run_dir, "model": model,
                   "provider": llm.get("provider"), "rows": len(rows),
                   "chunks": len(chunks), "batch": args.batch,
                   "usage": u, "cost_usd": actual,
                   "estimated_cost_usd": est,
                   "unknown_rows": len([r for r in labels
                                        if not r.get("classified")])},
                  f, ensure_ascii=False, indent=2)
    print(f"[out] {out_path}  ({len(labels)} 行)")
    print(f"[out] {cost_path}  実費 "
          + (f"${actual:.4f}" if actual is not None else "不明"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
