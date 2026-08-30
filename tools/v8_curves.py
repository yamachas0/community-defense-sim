#!/usr/bin/env python
"""v8 の集計（決定論・API を一切使わない）。

  python tools/v8_curves.py simulations/<run_dir> [--out docs/v8_run.md]

出すもの（設計 §6）:
  - 月別の取得区画数・売却人数・未売却割合の曲線
  - 提示件数と成約率
  - 売った人の「売る」直前の内心（原文）と、その人に届いた X社 の提示文（原文）
  - 買えなかった（提示したが売らなかった）件数
  - 費用（手元集計）と、請求見込み（安全率2.0）

**判定・分類はしない**（採点用 LLM は v8 に存在しない）。ここでやるのは数えることだけ。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List


def _load(run_dir: str, name: str) -> Any:
    path = os.path.join(run_dir, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def curves(run_dir: str) -> Dict[str, Any]:
    summary = _load(run_dir, "summary.json")
    monthly = _load(run_dir, "monthly.json") or []
    offers = _load(run_dir, "offers.json") or []
    decisions = _load(run_dir, "decisions.json") or []
    utterances = _load(run_dir, "utterances.json") or []
    n = int(summary["sellable_agents"])
    rows = []
    for m in monthly:
        rows.append({
            "month": m["step"],
            "offers": m["offers_sent"],
            "sold": m["sold_this_month"],
            "sold_cum": m["sold_cum"],
            "parcels_cum": m["parcels_cum"],
            "unsold_ratio": round(m["risk_set"] / n, 4) if n else 0.0,
            "attended": m["attended"],
            "utterances": m["utterances"],
        })
    # 提示 → その月の「売る」で成約したか
    accepted = sum(1 for o in offers if o["accepted"])
    # 売った人の直前の内心と、その月に届いていた提示
    sold_rows = []
    for d in decisions:
        if d["decision"] != "売る":
            continue
        sold_rows.append({"month": d["step"], "name": d["name"],
                          "thought": d["thought"], "offer": d.get("offer", "")})
    sold_rows.sort(key=lambda r: (r["month"], r["name"]))
    return {"summary": summary, "rows": rows, "offers": offers,
            "offers_accepted": accepted,
            "offers_declined": len(offers) - accepted,
            "sold_rows": sold_rows, "utterances": utterances,
            "decisions": decisions}


def to_markdown(c: Dict[str, Any], run_dir: str) -> str:
    s = c["summary"]
    n = int(s["sellable_agents"])
    safety = 2.0
    out: List[str] = []
    out.append(f"### 走行 `{s['run_name']}`（{os.path.basename(run_dir)}）")
    out.append("")
    out.append(f"- 会話: {'あり' if s['chat'] else 'なし'}／集まり1回×{s['scene_rounds']}巡"
               f"／{s['steps']}か月／seed {s['seed']}")
    out.append(f"- 主体 {s['agents']}体（うち売れる {n}体）／不動産 {s['parcels_total']}件"
               f"（うち売れる {s['sellable_parcels']}件）")
    out.append(f"- **X社の取得＝{s['acquired_parcels']}区画 / {s['sold_agents']}人**"
               f"（未売却の割合 {s['final_unsold_ratio']*100:.1f}%）")
    out.append(f"- 提示 {s['offers_total']}件（応じた {c['offers_accepted']}／"
               f"応じなかった {c['offers_declined']}）")
    out.append(f"- 健全性：答えが返らなかった月 {s['no_answer']}／読めなかった応答 "
               f"{s['parse_fail']}／打切り {s['truncated']}／行き先の無効 {s['invalid_venue']}")
    out.append(f"- 費用（手元集計）**${s['cost_usd']:.4f}**／請求見込み（安全率{safety}）"
               f"**${s['cost_usd']*safety:.4f}**／{s['elapsed_sec']/60:.1f}分／"
               f"{s['usage']['calls']}コール")
    out.append(f"- 共通前置き {s.get('common_prefix_tokens')}トークン／"
               f"キャッシュ比率 {s.get('cached_ratio', 0)*100:.1f}%"
               f"（作成 {s.get('cache_created', 0)}／失敗 {s.get('cache_failed', 0)}）")
    out.append("")
    out.append("#### 月別（取得曲線）")
    out.append("")
    out.append("```")
    out.append("月  提示  売却  累計人  累計区画  未売却割合  出席  発話")
    for r in c["rows"]:
        bar = "#" * int(round(r["parcels_cum"] / max(1, s["sellable_parcels"]) * 30))
        out.append(f"{r['month']:>2}  {r['offers']:>4}  {r['sold']:>4}  "
                   f"{r['sold_cum']:>6}  {r['parcels_cum']:>8}  "
                   f"{r['unsold_ratio']*100:>9.1f}%  {r['attended']:>4}  "
                   f"{r['utterances']:>4}  {bar}")
    out.append("```")
    out.append("")
    out.append("#### 売った人（売る直前の内心と、その月に届いていたX社の提示・原文）")
    out.append("")
    if not c["sold_rows"]:
        out.append("（36か月で誰も売らなかった）")
    for r in c["sold_rows"]:
        out.append(f"- **第{r['month']}月 {r['name']}**")
        out.append(f"  - 内心:「{r['thought']}」")
        out.append(f"  - X社の提示:「{r['offer']}」" if r["offer"]
                   else "  - X社の提示: （その月は届いていない）")
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    c = curves(args.run_dir)
    md = to_markdown(c, args.run_dir)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"wrote {args.out}")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
