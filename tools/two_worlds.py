#!/usr/bin/env python
"""2つの世界（経路なし／経路あり）を、まったく同じ4つの観測で読む。

    python tools/two_worlds.py --out docs/two_worlds.json

**LLM を1本も呼ばない。** 全部決定論である（採点用の LLM は v6 で全廃した）。

読むのは4つだけ（docs/world_design_v6_two_worlds.md §5）：

  1. 登記の線     … 買い手が押さえた土地の累積（台帳の transfer / lease）
  2. 気づいた月   … 公の場（発話・記事）で「同じ買い手」と口にした最初の月
                    ＝ `_v5c_rule_yellow`（名義2つ以上、または名義1つ＋「同じ会社」等の語）
                    **候補は全件を人が読んで確認する**（このツールは候補を出すだけ）
  3. 行動した月   … 「当面売らない」／回覧・議題・申入れ／行政の措置が初めて出た月
                    （経路なしの世界は構造上ゼロ＝空欄）
  4. 買えなかった件数 … 「当面売らない」で成立しなかった取得（経路なしは構造上0）
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from run_metrics import _v5c_rule_yellow, _v5c_rule_blue    # noqa: E402

PUBLIC_KINDS = ("utterance", "article")

# 「同じ買い手だ」と言っている行の判定（人手確認の基準を機械にしたもの）。
# 緩い網（`_v5c_rule_yellow`）は「2つの名前が同じ文に出てきた」だけで拾うので、
# 別々の相手として話しているだけの行も入る（読んで確認したら実際そうだった）。
# そこで**採用する1件**は次の基準で選び、原文を必ず載せる：
#   2つ以上の名義（A社〜D社）が出ていて、かつ**結びつけの語**があること
#   （関係／関連／繋がって／連携／同一／同じ／裏で／買い集め／買い占め／別の名前）
# 判定は決定論であり、選ばれた行は人が読んで確認する（docs/audit_v6.md）。
LINK_WORDS = ("関係があ", "関係し", "関連", "繋がって", "つながって", "連携",
              "同一", "同じ会社", "同じ買い手", "同じ相手", "同じところ",
              "裏で", "買い集め", "買い占め", "別の名前", "名義を変えて",
              "一つの会社", "ひとつの会社")
HOLDER_NAMES = ("A社", "B社", "C社", "D社")


def _linked(text: str) -> bool:
    names = [n for n in HOLDER_NAMES if n in text]
    return len(names) >= 2 and any(w in text for w in LINK_WORDS)



def _jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _summary(run_dir):
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as f:
        return json.load(f)


def read_run(run_dir: str, label: str) -> dict:
    """1本のランを4つの観測で読む（世界の版に関わらず同じ読み方）。"""
    summary = _summary(run_dir)
    n_steps = int(summary.get("steps") or 36)
    ledger = _jsonl(os.path.join(run_dir, "ledger.jsonl"))
    utts = _jsonl(os.path.join(run_dir, "utterances_v5.jsonl"))
    articles = _jsonl(os.path.join(run_dir, "articles_v5.jsonl"))
    acqs = [r for r in ledger if r.get("kind") == "transfer"]
    leases = [r for r in ledger if r.get("kind") == "lease"]

    # --- 1) 登記の線（その月までに買い手が押さえた土地の累積）---------------
    line, holders_by, acquired_by = [], {}, {}
    hs, ps = set(), set()
    for step in range(1, n_steps + 1):
        for r in acqs + leases:
            if int(r.get("step", 0)) <= step:
                hs.add(str(r.get("under_name", "")))
                ps.add(r.get("parcel_id"))
        holders_by[step] = set(hs)
        acquired_by[step] = set(ps)
        line.append({"step": step, "parcels": len(ps), "names": len(hs)})

    # --- 2) 気づいた月（公の場の呼び名一致・決定論）--------------------------
    rows = ([{"kind": "utterance", **u} for u in utts]
            + [{"kind": "article", "role": "media", **a} for a in articles])
    rows.sort(key=lambda r: (int(r.get("step", 0)), str(r.get("scene", "")),
                            int(r.get("round", 0) or 0)))
    hits = []
    for r in rows:
        if r.get("kind") not in PUBLIC_KINDS:
            continue
        step = int(r.get("step", 0))
        text = str(r.get("text", ""))
        if _v5c_rule_yellow(text, holders_by.get(step, set()),
                            acquired_by.get(step, set())):
            hits.append({"step": step, "from": r.get("from", ""),
                         "name": r.get("name", ""), "role": r.get("role", ""),
                         "kind": r.get("kind"), "venue": r.get("venue", ""),
                         "text": text})
    noticed = min([h["step"] for h in hits]) if hits else None
    linked = [h for h in hits if _linked(h["text"])]
    noticed_linked = min([h["step"] for h in linked]) if linked else None

    # --- 3) 行動した月（v6 だけ・経路なしの世界は構造上ゼロ）-----------------
    acts = _jsonl(os.path.join(run_dir, "actions_v6.jsonl"))
    papers = _jsonl(os.path.join(run_dir, "papers_v6.jsonl"))
    blocked = _jsonl(os.path.join(run_dir, "blocked_v6.jsonl"))
    has_path = bool(summary.get("v6"))

    def _first(rows_, pred):
        ms = [int(r.get("step", 0)) for r in rows_ if pred(r)]
        return min(ms) if ms else None

    refusals = [r for r in acts if r.get("value") == "当面売らない"]
    acted = {
        "refusal_first": _first(refusals, lambda r: True),
        "refusal_people": sorted({r.get("name") or r.get("agent_id")
                                  for r in refusals}),
        "refusal_parcels": sorted({p for r in refusals
                                   for p in (r.get("parcel_names") or [])}),
        "paper_first": _first(papers, lambda r: True),
        "papers": len(papers),
        "by_act": {},
        "measure_first": _first(papers, lambda r: r.get("role") == "municipality"),
        # 第36月に出た紙は誰にも読まれない（第37月が無い）＝別に数える
        "papers_read": len([p for p in papers if int(p["step"]) < n_steps]),
    }
    for p in papers:
        d = acted["by_act"].setdefault(p["act"], {"count": 0, "first": None,
                                                  "label": p["label"]})
        d["count"] += 1
        d["first"] = p["step"] if d["first"] is None else min(d["first"], p["step"])

    # --- 4) 買えなかった件数 ------------------------------------------------
    blocked_by_month = collections.Counter(int(b["step"]) for b in blocked)

    share = summary.get("kpi", {}).get("final_acquirer_share")
    stopped = None
    stop_path = os.path.join(run_dir, "defense_stop_v5e.json")
    if os.path.exists(stop_path):
        with open(stop_path, encoding="utf-8") as f:
            st = json.load(f)
        stopped = st.get("trigger_month") if st.get("stopped") else None

    return {
        "label": label,
        "run_dir": os.path.basename(run_dir),
        "world": "経路あり" if has_path else "経路なし",
        "steps": n_steps,
        "line": line,
        "parcels_final": line[-1]["parcels"] if line else 0,
        "final_share": share,
        "noticed_month": noticed,
        "noticed_hits": hits,
        "noticed_hit_count": len(hits),
        # 採用する「気づいた月」＝結びつけの語つき（原文を必ず載せる）
        "noticed_linked_month": noticed_linked,
        "noticed_linked_first": linked[0] if linked else None,
        "noticed_linked_count": len(linked),
        "acted": acted,
        "blocked": len(blocked),
        "blocked_by_month": dict(sorted(blocked_by_month.items())),
        "blocked_rows": blocked,
        "acquisitions_applied": (summary.get("v6", {}) or {}).get(
            "acquisitions_applied",
            len(acqs) + len(leases)),
        # 経路なしの世界で買い手が途中で手を引いた本（v5e2 runF）は右打ち切り
        "buyer_stopped_month": stopped,
        "utterances": len(utts),
        "articles": len(articles),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/two_worlds.json")
    ap.add_argument("--no-path", nargs="*", default=None,
                    help="経路なしの世界のラン（run_dir）")
    ap.add_argument("--path", nargs="*", default=None,
                    help="経路ありの世界のラン（run_dir）")
    args = ap.parse_args()

    sims = os.path.join(ROOT, "simulations")

    def _find(pattern):
        return sorted(os.path.join(sims, d) for d in os.listdir(sims)
                      if pattern in d)

    no_path = (args.no_path if args.no_path is not None
               else _find("field_v5e2_run"))
    path = args.path if args.path is not None else _find("field_v6_run")

    out = {"no_path": [], "path": []}
    for d in no_path:
        out["no_path"].append(read_run(d, os.path.basename(d).split("_run")[-1]))
    for d in path:
        out["path"].append(read_run(d, os.path.basename(d).split("_run")[-1]))

    dst = os.path.join(ROOT, args.out)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    for key, title in (("no_path", "経路なしの世界"), ("path", "経路ありの世界")):
        print(f"\n== {title} ==")
        for r in out[key]:
            print(f"  {r['label']}: 気づいた月 {r['noticed_linked_month']}"
                  f"（緩い網の初出 {r['noticed_month']}）"
                  f" / 売らない {r['acted']['refusal_first']}"
                  f" / 紙 {r['acted']['paper_first']}"
                  f" / 買えなかった {r['blocked']}件"
                  f" / 最終 {r['parcels_final']}区画"
                  f" シェア {r['final_share']}")
    print(f"\n[out] {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
