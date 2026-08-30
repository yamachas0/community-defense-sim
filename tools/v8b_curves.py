#!/usr/bin/env python
"""v8b の集計（決定論・API を一切使わない）。

  python tools/v8b_curves.py simulations/<v8b_run_dir> \
      --compare simulations/<v8_run_dir> [--out docs/v8b_run.md]

出すもの（設計 `docs/world_design_v8b.md` §1-5）:
  - 月別の 提示／出品／応諾／取得区画数／未売却割合 の曲線
  - 提示件数と成約率（提示した同じ月の月末に「応じる」と答えた割合）
  - 出品したが名義が動かなかった件数＝「売れない家」の内訳
  - 応じた人の「応じる」直前の内心（原文）と、その人に届いた条件文（原文）
  - **内心と選択の食い違いの候補**（決定論の語句照合・判定はしない）
  - v8 1本目との対比表
  - 費用（手元集計）と、請求見込み（安全率2.0）

**判定・分類はしない**（採点用 LLM は v8b に存在しない）。ここでやるのは数えることだけ。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

# 走行前に凍結した照合語（設計 §1-5。結果を見てから足さない）
NEG_WORDS = ["売らない", "売却は考え", "手放すつもりはない", "現状維持",
             "売るつもりはない", "売却しない", "守り続け"]
POS_WORDS = ["売る", "売却する", "名義を移す", "手放す", "応じる"]
# 打ち消しの語（照合語そのものは凍結したまま、**否定文を肯定に数えない**ための後置き検査）。
# 「売却するつもりはない」「売る気はない」を『内心は肯定』に数えてしまう素朴な部分一致を潰す。
NEGATORS = ["ない", "ません", "ぬ。"]
LOOKAHEAD = 14


def _load(run_dir: str, name: str) -> Any:
    path = os.path.join(run_dir, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pos_hits(text: str) -> List[str]:
    """肯定語の出現のうち、**直後が打ち消しでないもの**だけを返す。

    走行後に語彙を足したのではなく、部分一致の誤検出を落とすためのフィルタである
    （「売却するつもりはない」は肯定ではない）。
    """
    hits = []
    for w in POS_WORDS:
        start = 0
        while True:
            i = text.find(w, start)
            if i < 0:
                break
            tail = text[i + len(w): i + len(w) + LOOKAHEAD]
            if not any(ng in tail for ng in NEGATORS):
                hits.append(w)
                break
            start = i + len(w)
    return hits


def mismatch_candidates(decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """内心と選択の食い違いの**候補**を並べる（判定はしない・人が原文を読む）。

    - 内心に否定語があるのに「出す」または「応じる」を選んだ
    - 内心に肯定語があるのに「出さない」かつ「応じない」を選んだ
    """
    out = []
    for d in decisions:
        th = str(d.get("thought", "") or "")
        if not th:
            continue
        neg = [w for w in NEG_WORDS if w in th]
        pos = _pos_hits(th)
        acted = (d.get("listing") == "出す") or (d.get("respond") == "応じる")
        if neg and acted:
            out.append({**d, "kind": "内心は否定・選択は行動", "hit": neg})
        elif pos and not acted and d.get("listing") == "出さない":
            out.append({**d, "kind": "内心は肯定・選択は不作為", "hit": pos})
    return out


def curves(run_dir: str) -> Dict[str, Any]:
    summary = _load(run_dir, "summary.json")
    monthly = _load(run_dir, "monthly.json") or []
    offers = _load(run_dir, "offers.json") or []
    decisions = _load(run_dir, "decisions.json") or []
    listings = _load(run_dir, "listings.json") or []
    utterances = _load(run_dir, "utterances.json") or []
    n = int(summary["sellable_agents"])
    rows = []
    for m in monthly:
        rows.append({
            "month": m["step"],
            "offers": m["offers_sent"],
            "listed": m.get("listed_this_month", 0),
            "accepted": m.get("accepted_this_month", m.get("sold_this_month", 0)),
            "sold_cum": m["sold_cum"],
            "parcels_cum": m["parcels_cum"],
            "unsold_ratio": round(m["risk_set"] / n, 4) if n else 0.0,
            "attended": m["attended"],
            "utterances": m["utterances"],
        })
    accepted_rows = []
    for d in decisions:
        if d.get("respond") == "応じる":
            accepted_rows.append({"month": d["step"], "name": d["name"],
                                  "thought": d.get("thought", ""),
                                  "offer": d.get("offer", "")})
    accepted_rows.sort(key=lambda r: (r["month"], r["name"]))
    return {"summary": summary, "rows": rows, "offers": offers,
            "listings": listings, "decisions": decisions,
            "utterances": utterances, "accepted_rows": accepted_rows,
            "mismatch": mismatch_candidates(decisions)}


def _compare_table(b: Dict[str, Any], a: Optional[Dict[str, Any]]) -> List[str]:
    """v8 1本目との対比（a=v8 の summary）。"""
    if not a:
        return []
    out = ["#### v8 1本目との対比", "", "```",
           f"{'':<26}{'v8 (1本目)':>14}{'v8b':>14}"]

    def line(label: str, va: Any, vb: Any) -> str:
        return f"{label:<26}{str(va):>14}{str(vb):>14}"

    out.append(line("X社の取得（区画）", a["acquired_parcels"], b["acquired_parcels"]))
    out.append(line("名義が移った人", a["sold_agents"], b["sold_agents"]))
    out.append(line("未売却の割合", f"{a['final_unsold_ratio']*100:.1f}%",
                    f"{b['final_unsold_ratio']*100:.1f}%"))
    out.append(line("X社の提示（のべ）", a["offers_total"], b["offers_total"]))
    out.append(line("提示の相手（実人数）", a.get("offer_targets", "—"),
                    b.get("offer_targets", "—")))
    out.append(line("応じた（成約）", a["offers_accepted"], b["offers_accepted"]))
    out.append(line("出品（のべ）", "—（制度が無い）", b.get("listings_total", 0)))
    out.append(line("売れない家", "—", b.get("unsold_listings", 0)))
    out.append(line("コール数", a["usage"]["calls"], b["usage"]["calls"]))
    out.append(line("手元費用($)", f"{a['cost_usd']:.4f}", f"{b['cost_usd']:.4f}"))
    out.append(line("キャッシュ比率", f"{a.get('cached_ratio',0)*100:.1f}%",
                    f"{b.get('cached_ratio',0)*100:.1f}%"))
    out.append("```")
    out.append("")
    return out


def to_markdown(c: Dict[str, Any], run_dir: str,
                compare: Optional[Dict[str, Any]] = None) -> str:
    s = c["summary"]
    n = int(s["sellable_agents"])
    safety = 2.0
    targets = len({o["to_id"] for o in c["offers"]})
    s = {**s, "offer_targets": targets}
    out: List[str] = []
    out.append(f"### 走行 `{s['run_name']}`（{os.path.basename(run_dir)}）")
    out.append("")
    out.append(f"- 会話: {'あり' if s['chat'] else 'なし'}／集まり1回×{s['scene_rounds']}巡"
               f"／走った月 {s['months_run']}／全{s['steps']}か月／seed {s['seed']}")
    out.append(f"- 主体 {s['agents']}体（うち売れる {n}体）／不動産 {s['parcels_total']}件"
               f"（うち売れる {s['sellable_parcels']}件）")
    out.append(f"- **X社の取得＝{s['acquired_parcels']}区画 / {s['sold_agents']}人**"
               f"（未売却の割合 {s['final_unsold_ratio']*100:.1f}%）")
    out.append(f"- 提示 {s['offers_total']}件・相手は実人数 {targets}人"
               f"（応じた {s['offers_accepted']}／応じなかった {s['offers_declined']}／"
               f"答えが返らなかった {s['offers_no_answer']}）")
    out.append(f"  - 提示のうち 出品していた人へ {s.get('offers_to_listed', 0)}件／"
               f"出品していない人へ {s.get('offers_to_unlisted', 0)}件")
    out.append(f"- **出品 のべ{s.get('listings_total', 0)}件"
               f"（{s.get('listed_agents_ever', 0)}人）**／"
               f"**売れない家 {s.get('unsold_listings', 0)}件**")
    out.append(f"  - 内訳：翌月に提示が来なかった {s.get('listing_no_offer_next', 0)}／"
               f"提示は来たが応じなかった {s.get('listing_offer_not_accepted_next', 0)}／"
               f"提示が来て応じた {s.get('listing_accepted_next', 0)}／"
               f"同じ月に応じて名義が移った {s.get('listing_sold_same_month', 0)}／"
               f"翌月なし（最終月） {s.get('listing_no_next_month', 0)}")
    out.append(f"- 健全性：答えが返らなかった月 {s['no_answer']}"
               f"（うち応諾の欄 {s.get('respond_no_answer', 0)}）／"
               f"読めなかった応答 {s['parse_fail']}／打切り {s['truncated']}／"
               f"行き先の無効 {s['invalid_venue']}")
    out.append(f"  - X社の応答：判断が返らなかった相手 "
               f"{s.get('acquirer_missing_targets', 0)}／重複 "
               f"{s.get('acquirer_dup_rows', 0)}／対象外 "
               f"{s.get('acquirer_off_range', 0)}／塊まるごと欠損 "
               f"{s.get('acquirer_chunk_fail', 0)}")
    out.append(f"- 費用（手元集計）**${s['cost_usd']:.4f}**／請求見込み（安全率{safety}）"
               f"**${s['cost_usd']*safety:.4f}**／{s['elapsed_sec']/60:.1f}分／"
               f"{s['usage']['calls']}コール／費用で停止 {s['stopped_by_cost']}")
    out.append(f"- 共通前置き {s.get('common_prefix_tokens')}トークン／"
               f"キャッシュ比率 {s.get('cached_ratio', 0)*100:.1f}%")
    out.append("")
    out += _compare_table(s, compare)
    out.append("#### 月別（取得曲線）")
    out.append("")
    out.append("```")
    out.append("月  提示  出品  応諾  累計人  累計区画  未売却割合  出席  発話")
    for r in c["rows"]:
        bar = "#" * int(round(r["parcels_cum"] / max(1, s["sellable_parcels"]) * 30))
        out.append(f"{r['month']:>2}  {r['offers']:>4}  {r['listed']:>4}  "
                   f"{r['accepted']:>4}  {r['sold_cum']:>6}  {r['parcels_cum']:>8}  "
                   f"{r['unsold_ratio']*100:>9.1f}%  {r['attended']:>4}  "
                   f"{r['utterances']:>4}  {bar}")
    out.append("```")
    out.append("")
    out.append("#### 応じた人（応じる直前の内心と、届いていた条件文・原文）")
    out.append("")
    if not c["accepted_rows"]:
        out.append("（誰も条件に応じなかった）")
    for r in c["accepted_rows"]:
        out.append(f"- **第{r['month']}月 {r['name']}**")
        out.append(f"  - 内心:「{r['thought']}」")
        out.append(f"  - X社の条件:「{r['offer']}」")
    out.append("")
    out.append("#### 内心と選択の食い違いの候補（決定論の語句照合・判定ではない）")
    out.append("")
    neg = [d for d in c["mismatch"] if d["kind"] == "内心は否定・選択は行動"]
    pos = [d for d in c["mismatch"] if d["kind"] != "内心は否定・選択は行動"]
    out.append(f"候補 {len(c['mismatch'])}件（照合語は走行前に凍結）＝"
               f"内心は否定・選択は行動 {len(neg)}件／内心は肯定・選択は不作為 "
               f"{len(pos)}件。**これは判定ではない**。"
               "原文を人が読んで数えたものだけを本文に書く。"
               "全件は decisions.json にある。")
    for d in neg + pos[:6]:
        out.append(f"- 第{d['step']}月 {d['name']}｜{d['kind']}"
                   f"（{'・'.join(d['hit'])}）"
                   f"｜出品={d.get('listing')}／応諾={d.get('respond')}")
        out.append(f"  - 内心:「{d.get('thought','')}」")
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--compare", default=None, help="v8 1本目の run_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    c = curves(args.run_dir)
    comp = _load(args.compare, "summary.json") if args.compare else None
    if comp:
        offers = _load(args.compare, "offers.json") or []
        comp["offer_targets"] = len({o["to_id"] for o in offers})
    md = to_markdown(c, args.run_dir, comp)
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
