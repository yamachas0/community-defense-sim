#!/usr/bin/env python
"""人手監査：機械判定の取りこぼし（偽陰性）と行き過ぎ（偽陽性）を人が読んで確かめる。

    python tools/build_audit_v5c.py --run simulations/<run_dir>
    → docs/audit_v5c_sample.md（人が読んで判定する用紙）

    python tools/build_audit_v5c.py --run simulations/<run_dir> \
        --judgments docs/audit_v5c_judgments.json
    → 判定を突き合わせ、偽陰性率・偽陽性率を末尾に追記

標本は3層（Codexレビュー 2026-08-28）：
  A ルール1次抽出に1つも掛からなかった行 100（＝主値が構造的に取りこぼす層）
  B ルールは当たったが LLM が否定した行 50（＝∧で落ちた層）
  C 主値が色を付けた行 50（＝偽陽性の確認）
seed 固定の無作為抽出で、結果を見て選び直さない。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import yaml  # noqa: E402

from run_metrics import (_read_jsonl, _v5c_rule_blue, _v5c_rule_green,   # noqa: E402
                         _v5c_rule_red, _v5c_rule_yellow, _v5c_stage)

COLORS = ["blue", "green", "yellow", "red", "none"]
COLOR_JA = {"blue": "青（個別の売買）", "green": "緑（複数・面）",
            "yellow": "黄（同じ買い手＝X社に届いた）", "red": "赤（行政が動いた）",
            "none": "色なし"}
STRATA = [("A", "ルール1次抽出に1つも掛からなかった行", 100),
          ("B", "ルールは当たったが LLM が否定した行", 50),
          ("C", "主値が色を付けた行（偽陽性の確認）", 50)]


def collect(run_dir):
    utts = _read_jsonl(os.path.join(run_dir, "utterances_v5.jsonl"))
    thoughts = _read_jsonl(os.path.join(run_dir, "thoughts_all.jsonl"))
    articles = _read_jsonl(os.path.join(run_dir, "articles_v5.jsonl"))
    ledger = _read_jsonl(os.path.join(run_dir, "ledger.jsonl"))
    labels = _read_jsonl(os.path.join(run_dir, "stage_labels_v5c.jsonl"))
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    n_steps = int(summary.get("steps") or 0)
    role_of = {}
    for r in utts + thoughts:
        if r.get("from") and r.get("role"):
            role_of[r["from"]] = r["role"]

    holders_by_step, acquired_by_step = {}, {}
    hs, ps = set(), set()
    for step in range(1, n_steps + 1):
        for r in ledger:
            if r.get("kind") in ("transfer", "lease") and int(r.get("step", 0)) <= step:
                hs.add(str(r.get("under_name", "")))
                ps.add(r.get("parcel_id"))
        holders_by_step[step] = set(hs)
        acquired_by_step[step] = set(ps)

    label_of = {}
    for r in labels:
        label_of[(int(r.get("step", 0)), r.get("kind", ""), r.get("from", ""),
                  str(r.get("text", ""))[:60])] = r

    rows = ([{"kind": "utterance", **u} for u in utts]
            + [{"kind": "thought", **t} for t in thoughts]
            + [{"kind": "article", **a} for a in articles])
    out = []
    for r in rows:
        text = str(r.get("text", ""))
        step = int(r.get("step", 0))
        hsx = holders_by_step.get(step, set())
        psx = acquired_by_step.get(step, set())
        role = r.get("role") or role_of.get(r.get("from"), "")
        rule = {"rule_blue": _v5c_rule_blue(text, hsx, psx),
                "rule_green": _v5c_rule_green(text, hsx, psx),
                "rule_yellow": _v5c_rule_yellow(text, hsx, psx),
                "rule_red": _v5c_rule_red(role, text)}
        lab = label_of.get((step, r["kind"], r.get("from", ""), text[:60]))
        classified = bool(lab) and lab.get("deal") is not None
        row = {"step": step, "from": r.get("from"), "role": role, "kind": r["kind"],
               "scene": r.get("scene", ""), "venue": r.get("venue", ""), "text": text,
               "classified": classified,
               "llm_deal": bool(lab.get("deal")) if classified else None,
               "llm_area": bool(lab.get("area")) if classified else None,
               "llm_same_buyer": bool(lab.get("same_buyer")) if classified else None,
               "llm_admin": bool(lab.get("admin")) if classified else None,
               **rule}
        row["rule_hit"] = any(rule.values())
        row["stage"] = _v5c_stage(row) if classified else None
        out.append(row)
    return out, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=85)
    ap.add_argument("--judgments", default="")
    ap.add_argument("--out", default="docs/audit_v5c_sample.md")
    args = ap.parse_args()

    rows, summary = collect(args.run)
    pools = {
        "A": [r for r in rows if not r["rule_hit"]],
        "B": [r for r in rows if r["rule_hit"] and r["stage"] is None],
        "C": [r for r in rows if r["stage"] is not None],
    }
    rng = random.Random(args.seed)
    samples = {}
    for key, _label, n in STRATA:
        pool = pools[key]
        sel = rng.sample(pool, min(n, len(pool)))
        sel.sort(key=lambda r: (r["step"], r["kind"], r["from"] or ""))
        samples[key] = sel

    venue_label = {}
    cfgp = os.path.join(args.run, "config.yaml")
    if os.path.exists(cfgp):
        with open(cfgp, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        venue_label = {v["id"]: v["label"]
                       for v in (cfg.get("social", {}).get("venues") or [])}

    lines = [
        "# 人手監査（v5c・4段階の色）",
        "",
        f"- 対象ラン：`{os.path.basename(args.run.rstrip('/'))}`"
        f"（{summary.get('steps')}か月・{summary.get('model')}）",
        f"- 母集団：発話・内心・記事 **{len(rows)} 行**",
        f"  - A ルール1次抽出に1つも掛からなかった行：{len(pools['A'])}",
        f"  - B ルールは当たったが LLM が否定した行：{len(pools['B'])}",
        f"  - C 主値が色を付けた行：{len(pools['C'])}",
        f"- 標本：seed {args.seed} の無作為抽出（A {len(samples['A'])}／"
        f"B {len(samples['B'])}／C {len(samples['C'])}）。結果を見て選び直していない。",
        "- 判定の基準（走行前に固定）：青＝個別の売買の話題／緑＝複数の売買の関係性・"
        "面的な話／黄＝名義の違う取引を同じ買い手として結んだ／赤＝行政の主体が"
        "届出・条例・調査等の対応を口にした・決めた。",
        "",
    ]
    for key, label, _n in STRATA:
        lines += [f"## {key}. {label}（{len(samples[key])}行）", "",
                  "| # | 月 | 主体 | 立場 | 種別 | 場 | 機械の判定 | 原文 |",
                  "|---:|---:|---|---|---|---|---|---|"]
        for i, r in enumerate(samples[key], start=1):
            venue = venue_label.get(r["venue"], r["venue"]
                                    or ("記事" if r["kind"] == "article" else "自宅・計画"))
            text = r["text"].replace("|", "｜").replace("\n", " ")
            machine = (COLOR_JA.get(r["stage"], "—") if r["stage"]
                       else ("ルール○/LLM×" if r["rule_hit"] else "ルール×"))
            lines.append(f"| {key}{i} | {r['step']} | {r['from']} | {r['role']} | "
                         f"{r['kind']} | {venue} | {machine} | {text} |")
        lines.append("")

    if args.judgments:
        with open(os.path.join(ROOT, args.judgments), encoding="utf-8") as f:
            judgments = json.load(f)
        lines += ["## 判定結果（CTO が全行を読んで付けた）", ""]
        summary_rows = []
        for key, label, _n in STRATA:
            sel = samples[key]
            hits = []
            for i, r in enumerate(sel, start=1):
                human = str(judgments.get(f"{key}{i}", "none"))
                machine = r["stage"] or "none"
                if human != machine:
                    hits.append((f"{key}{i}", machine, human, r))
            summary_rows.append((key, label, len(sel), hits))
        rank = {"none": -1, "blue": 0, "green": 1, "yellow": 2, "red": 3}

        def split(hits):
            fp = [h for h in hits if h[2] == "none"]          # 色を付けるべきでなかった
            fn = [h for h in hits if h[1] == "none"]          # 色を付け落とした
            over = [h for h in hits if h[1] != "none" and h[2] != "none"
                    and rank[h[1]] > rank[h[2]]]             # 色が過大
            under = [h for h in hits if h[1] != "none" and h[2] != "none"
                     and rank[h[1]] < rank[h[2]]]            # 色が過少
            return fp, fn, over, under

        lines += ["| 層 | 標本 | 食い違い | 色を付け落とし | 色を付けすぎ | 段階が過大 | 段階が過少 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for key, label, n, hits in summary_rows:
            fp, fn, over, under = split(hits)
            lines.append(f"| {key} {label} | {n} | {len(hits)} | {len(fn)} | "
                         f"{len(fp)} | {len(over)} | {len(under)} |")
        a = [x for x in summary_rows if x[0] == "A"][0]
        c = [x for x in summary_rows if x[0] == "C"][0]
        a_fp, a_fn, a_over, a_under = split(a[3])
        c_fp, c_fn, c_over, c_under = split(c[3])
        lines += [
            "",
            f"- **偽陰性率（A層・機械が色を付け落とした割合）＝{len(a_fn)} / {a[2]} ＝ "
            f"{len(a_fn) / max(1, a[2]) * 100:.1f}%**"
            f"（母集団 {len(pools['A'])} 行への外挿でおよそ "
            f"{round(len(pools['A']) * len(a_fn) / max(1, a[2]))} 行）",
            f"- **偽陽性率（C層・色を付けるべきでなかった割合）＝{len(c_fp)} / {c[2]} ＝ "
            f"{len(c_fp) / max(1, c[2]) * 100:.1f}%**",
            f"- **段階が過少だった割合（C層・機械の色より人の方が上）＝{len(c_under)} / {c[2]} ＝ "
            f"{len(c_under) / max(1, c[2]) * 100:.1f}%**"
            f"／段階が過大＝{len(c_over)} / {c[2]}",
            "",
        ]
        for key, label, n, hits in summary_rows:
            if not hits:
                continue
            lines += [f"### {key} 層で食い違った行", ""]
            for tag, machine, human, r in hits:
                lines.append(f"- **{tag}・第{r['step']}月・{r['from']}**："
                             f"機械={COLOR_JA.get(machine, machine)} / "
                             f"人={COLOR_JA.get(human, human)}　{r['text']}")
            lines.append("")

    out = os.path.join(ROOT, args.out)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[out] {out}  母集団 {len(rows)} / A {len(pools['A'])} B {len(pools['B'])} "
          f"C {len(pools['C'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
