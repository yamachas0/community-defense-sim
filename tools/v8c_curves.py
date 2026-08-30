#!/usr/bin/env python
"""v8c の集計（決定論・API を一切使わない）。

  python tools/v8c_curves.py simulations/<v8c_chat_dir> \
      --nochat simulations/<v8c_nochat_dir> --compare simulations/<v8b_dir> \
      [--out docs/_v8c_body.md]

出すもの:
  - 走行の数字（取得・提示・出品・売れない家・理由の記入率・断りの一言・健全性・費用）
  - 月別の曲線（提示／出品／売る／累計区画／未売却割合／出席／発話）
  - 会話あり／なし／直前版(v8b) の対比表
  - 売った人の原文（内心・理由・届いていた条件文）
  - 断りの一言の原文と、X社の条件文の型
  - 内心と選択・**理由と選択**の食い違いの**候補**（決定論の語句照合・判定はしない）

**判定・分類はしない**（採点用 LLM は v8c に存在しない）。ここでやるのは数えることだけ。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

# 走行前に凍結した照合語（v8b から引き継ぎ、「応じる」→「売る」だけ置換）
NEG_WORDS = ["売らない", "売却は考え", "手放すつもりはない", "現状維持",
             "売るつもりはない", "売却しない", "守り続け"]
POS_WORDS = ["売る", "売却する", "名義を移す", "手放す"]
NEGATORS = ["ない", "ません", "ぬ。"]
LOOKAHEAD = 14


def _load(run_dir: Optional[str], name: str) -> Any:
    if not run_dir:
        return None
    path = os.path.join(run_dir, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pos_hits(text: str) -> List[str]:
    """肯定語の出現のうち、**直後が打ち消しでないもの**だけを返す。"""
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
    """食い違いの**候補**（判定はしない・人が原文を読む）。

    - 内心に否定語があるのに「出す」または「売る」を選んだ
    - 内心に肯定語があるのに「出さない」かつ「売る」でない
    - **理由の一言**と選択が逆を向いている（v8c で新設）
    """
    out = []
    for d in decisions:
        th = str(d.get("thought", "") or "")
        acted = (d.get("listing") == "出す") or (d.get("sell") == "売る")
        if th:
            neg = [w for w in NEG_WORDS if w in th]
            pos = _pos_hits(th)
            if neg and acted:
                out.append({**d, "kind": "内心は否定・選択は行動", "hit": neg})
            elif pos and not acted and d.get("listing") == "出さない":
                out.append({**d, "kind": "内心は肯定・選択は不作為", "hit": pos})
        for field, choice, label in (("listing_reason", d.get("listing"), "出品"),
                                     ("sell_reason", d.get("sell"), "売買")):
            r = str(d.get(field, "") or "")
            if not r:
                continue
            neg = [w for w in NEG_WORDS if w in r]
            pos = _pos_hits(r)
            if neg and choice in ("出す", "売る"):
                out.append({**d, "kind": f"理由は否定・選択は行動（{label}）",
                            "hit": neg, "reason_text": r})
            elif pos and choice in ("出さない", "売らない"):
                out.append({**d, "kind": f"理由は肯定・選択は不作為（{label}）",
                            "hit": pos, "reason_text": r})
    return out


def curves(run_dir: str) -> Dict[str, Any]:
    summary = _load(run_dir, "summary.json")
    monthly = _load(run_dir, "monthly.json") or []
    offers = _load(run_dir, "offers.json") or []
    decisions = _load(run_dir, "decisions.json") or []
    listings = _load(run_dir, "listings.json") or []
    utterances = _load(run_dir, "utterances.json") or []
    n = int(summary["sellable_agents"])
    # 費用上限で月の途中で止まった場合、その月にX社が出した提示は
    # **誰にも問われないまま** offers.json に result="売らなかった" で残る。
    # 走った月（months_run）より後の月の記録は集計から外す（読み取り監査 2026-08-30 の指摘）。
    months_run = int(summary.get("months_run", 0))
    offers = [o for o in offers if int(o["step"]) <= months_run]
    decisions = [d for d in decisions if int(d["step"]) <= months_run]
    listings = [r for r in listings if int(r["step"]) <= months_run]
    utterances = [u for u in utterances if int(u["step"]) <= months_run]
    rows = []
    for m in monthly:
        rows.append({
            "month": m["step"], "offers": m["offers_sent"],
            "listed": m.get("listed_this_month", 0),
            "sold": m.get("accepted_this_month", 0),
            "sold_cum": m["sold_cum"], "parcels_cum": m["parcels_cum"],
            "unsold_ratio": round(m["risk_set"] / n, 4) if n else 0.0,
            "attended": m["attended"], "utterances": m["utterances"],
            "declines_with_reason": m.get("declines_with_reason", 0),
        })
    sold_rows = [{"month": d["step"], "name": d["name"],
                  "thought": d.get("thought", ""),
                  "reason": d.get("sell_reason", ""),
                  "offer": d.get("offer", "")}
                 for d in decisions if d.get("sell") == "売る"]
    sold_rows.sort(key=lambda r: (r["month"], r["name"]))
    # 途中で止まった場合、summary の件数には「誰にも問われていない提示」が入っている。
    # 走った月までの記録から数え直す（完走時は元の値と一致する）。
    summary = dict(summary)
    summary["offers_total"] = len(offers)
    summary["offers_accepted"] = sum(1 for o in offers if o.get("accepted"))
    summary["offers_declined"] = sum(1 for o in offers
                                     if o.get("result") == "売らなかった"
                                     and not o.get("accepted"))
    summary["offers_no_answer"] = sum(1 for o in offers
                                      if o.get("result") == "答えが返らなかった")
    summary["listings_total"] = len(listings)
    summary["listed_agents_ever"] = len({r["agent_id"] for r in listings})
    summary["declines_delivered"] = sum(
        1 for o in offers if str(o.get("decline_reason", "") or "").strip())
    return {"summary": summary, "rows": rows, "offers": offers,
            "listings": listings, "decisions": decisions,
            "utterances": utterances, "sold_rows": sold_rows,
            "mismatch": mismatch_candidates(decisions)}


def _x_mentions(utterances: List[Dict[str, Any]]) -> Dict[str, Any]:
    # 「X社」または「海外」を含む発話（emergence 側の数え方と揃える）
    hits = [u for u in utterances
            if "X社" in u["text"] or "海外" in u["text"]]
    return {"total": len(utterances), "mentions": len(hits),
            "first_month": min([u["step"] for u in hits], default=None),
            "examples": hits[:3]}


def numbers_block(c: Dict[str, Any]) -> List[str]:
    s = c["summary"]
    n = int(s["sellable_agents"])
    targets = len({o["to_id"] for o in c["offers"]})
    x = _x_mentions(c["utterances"])
    out = []
    out.append(f"- **X社の取得＝{s['acquired_parcels']}区画 / {s['sold_agents']}人**"
               f"（売れる {s['sellable_parcels']}区画・{n}人のうち）。"
               f"未売却の割合 **{s['final_unsold_ratio']*100:.1f}%**")
    out.append(f"- **提示 {s['offers_total']}件・相手は実人数 {targets}人**"
               f"（売った {s['offers_accepted']}／売らなかった {s['offers_declined']}／"
               f"答えが返らなかった {s['offers_no_answer']}）")
    out.append(f"  - 提示のうち 出品していた人へ {s.get('offers_to_listed', 0)}件／"
               f"出品していない人へ {s.get('offers_to_unlisted', 0)}件")
    out.append(f"- **出品 のべ{s.get('listings_total', 0)}件"
               f"（{s.get('listed_agents_ever', 0)}人）／"
               f"売れない家 {s.get('unsold_listings', 0)}件**")
    out.append(f"  - 内訳：翌月に提示が来なかった {s.get('listing_no_offer_next', 0)}／"
               f"提示は来たが売らなかった {s.get('listing_offer_not_accepted_next', 0)}／"
               f"提示が来て売った {s.get('listing_accepted_next', 0)}／"
               f"同じ月に売って名義が移った {s.get('listing_sold_same_month', 0)}／"
               f"翌月なし（最終月） {s.get('listing_no_next_month', 0)}")
    rc = s.get("reason_counts", {})
    out.append(f"- **理由の一言の記入率**：行き先 {s.get('reason_rate_plan', 0)*100:.1f}%"
               f"（{s.get('reason_written_plan', 0)}/{rc.get('plan_total', 0)}）／"
               f"出す・出さない {s.get('reason_rate_listing', 0)*100:.1f}%"
               f"（{s.get('reason_written_listing', 0)}/{rc.get('listing_total', 0)}）／"
               f"売る・売らない {s.get('reason_rate_sell', 0)*100:.1f}%"
               f"（{s.get('reason_written_sell', 0)}/{rc.get('sell_total', 0)}）／"
               f"X社の判断 {s.get('reason_rate_acquirer', 0)*100:.1f}%"
               f"（{s.get('reason_written_acquirer', 0)}/{rc.get('acquirer_total', 0)}）")
    out.append(f"- **X社に届いた断りの一言 {s.get('declines_delivered', 0)}件**")
    if x["mentions"]:
        out.append(f"- 発話 {x['total']}件のうち **{x['mentions']}件"
                   f"（{x['mentions']/x['total']*100:.0f}%）がX社に言及**・"
                   f"初出は **第{x['first_month']}月**")
    else:
        out.append(f"- 発話 {x['total']}件（X社への言及なし）")
    out.append(f"- 健全性：答えが返らなかった月 {s['no_answer']}"
               f"（うち売買の欄 {s.get('respond_no_answer', 0)}）／"
               f"読めなかった応答 {s['parse_fail']}／打切り {s['truncated']}／"
               f"行き先の無効 {s['invalid_venue']}／"
               f"X社の応答の取りこぼし {s.get('acquirer_missing_targets', 0)}・"
               f"重複 {s.get('acquirer_dup_rows', 0)}・"
               f"対象外 {s.get('acquirer_off_range', 0)}・"
               f"塊欠損 {s.get('acquirer_chunk_fail', 0)}")
    out.append(f"- 費用（手元集計）**${s['cost_usd']:.4f}**／請求見込み（安全率2.0）"
               f"**${s['cost_usd']*2:.4f}**／{s['elapsed_sec']/60:.1f}分／"
               f"{s['usage']['calls']}コール／走った月 {s['months_run']}／"
               f"費用で停止 {s['stopped_by_cost']}"
               + (f"（第{s.get('partial_month')}月の途中で打ち切り）"
                  if s.get("partial_month") else ""))
    out.append(f"- キャッシュ比率 {s.get('cached_ratio', 0)*100:.1f}%"
               f"（暗黙キャッシュ・共通前置き {s.get('common_prefix_tokens')}トークン）")
    return out


def compare_table(chat: Dict[str, Any], nochat: Optional[Dict[str, Any]],
                  v8b: Optional[Dict[str, Any]]) -> List[str]:
    def col(c):
        if not c:
            return None
        s = c["summary"]
        u = _x_mentions(c["utterances"])
        return {
            "取得（区画）": s["acquired_parcels"],
            "名義が移った人": s["sold_agents"],
            "未売却の割合": f"{s['final_unsold_ratio']*100:.1f}%",
            "X社の提示（のべ）": s["offers_total"],
            "提示の相手（実人数）": len({o["to_id"] for o in c["offers"]}),
            "売った（成約）": s["offers_accepted"],
            "出品（のべ）": s.get("listings_total", 0),
            "売れない家": s.get("unsold_listings", 0),
            "発話": u["total"],
            "X社に言及した発話": u["mentions"],
            "X社の初出（月）": u["first_month"] if u["first_month"] else "—",
            "走った月": s["months_run"],
            "コール数": s["usage"]["calls"],
            "手元費用($)": f"{s['cost_usd']:.4f}",
        }
    cols = [("v8c 会話あり", col(chat)), ("v8c 会話なし", col(nochat)),
            ("v8b（直前版）", col(v8b))]
    cols = [(h, c) for h, c in cols if c]
    keys = list(cols[0][1].keys())
    out = ["```", f"{'':<24}" + "".join(f"{h:>16}" for h, _ in cols)]
    for k in keys:
        out.append(f"{k:<24}" + "".join(f"{str(c[k]):>16}" for _, c in cols))
    out.append("```")
    return out


def monthly_block(c: Dict[str, Any]) -> List[str]:
    s = c["summary"]
    out = ["```", "月  提示  出品  売った  累計人  累計区画  未売却割合  出席  発話  断りの一言"]
    for r in c["rows"]:
        bar = "#" * int(round(r["parcels_cum"] / max(1, s["sellable_parcels"]) * 30))
        out.append(f"{r['month']:>2}  {r['offers']:>4}  {r['listed']:>4}  "
                   f"{r['sold']:>6}  {r['sold_cum']:>6}  {r['parcels_cum']:>8}  "
                   f"{r['unsold_ratio']*100:>9.1f}%  {r['attended']:>4}  "
                   f"{r['utterances']:>4}  {r['declines_with_reason']:>10}  {bar}")
    out.append("```")
    return out


def texts_block(c: Dict[str, Any]) -> List[str]:
    out = []
    out.append("#### 売った人（その月の内心・理由の一言・届いていた提示文）")
    out.append("")
    if not c["sold_rows"]:
        out.append("（誰も売らなかった）")
    for r in c["sold_rows"]:
        out.append(f"- **第{r['month']}月 {r['name']}**")
        out.append(f"  - 届いた提示:「{r['offer']}」")
        out.append(f"  - 理由の一言:「{r['reason']}」")
        out.append(f"  - 内心:「{r['thought']}」")
    out.append("")
    notes = [o for o in c["offers"] if str(o.get("decline_reason", "") or "").strip()]
    out.append(f"#### 断りの一言（X社にだけ届いた・{len(notes)}件のうち先頭20件）")
    out.append("")
    for o in notes[:20]:
        out.append(f"- 第{o['step']}月 {o['to']}：「{o['decline_reason']}」")
    out.append("")
    texts = Counter(o["text"] for o in c["offers"])
    out.append(f"#### X社の条件文（{len(texts)}通り・多い順に5つ）")
    out.append("")
    for t, k in texts.most_common(5):
        out.append(f"- {k}回：「{t}」")
    reasons = [o for o in c["offers"] if str(o.get("reason", "") or "").strip()]
    if reasons:
        out.append("")
        out.append("#### X社が提示に付けた理由（自分の判断の理由・相手には届かない・先頭8件）")
        out.append("")
        for o in reasons[:8]:
            out.append(f"- 第{o['step']}月 {o['to']}：「{o['reason']}」")
    return out


def mismatch_block(c: Dict[str, Any]) -> List[str]:
    ms = c["mismatch"]
    kinds = Counter(m["kind"] for m in ms)
    out = [f"候補 {len(ms)}件（照合語は走行前に凍結）＝"
           + "／".join(f"{k} {v}件" for k, v in kinds.items())
           + "。**これは判定ではない**。原文を人が読んで数えたものだけを本文に書く。"
             "全件は `decisions.json` にある。", ""]
    for m in ms[:12]:
        out.append(f"- 第{m['step']}月 {m['name']}｜{m['kind']}"
                   f"（{'・'.join(m['hit'])}）"
                   f"｜出品={m.get('listing')}／売買={m.get('sell')}")
        if m.get("reason_text"):
            out.append(f"  - 理由:「{m['reason_text']}」")
        else:
            out.append(f"  - 内心:「{m.get('thought','')}」")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--nochat", default=None)
    ap.add_argument("--compare", default=None, help="v8b の run_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    chat = curves(args.run_dir)
    nochat = curves(args.nochat) if args.nochat else None
    v8b = curves(args.compare) if args.compare else None

    out: List[str] = []
    out.append("### 数字（会話あり）")
    out.append("")
    out += numbers_block(chat)
    out.append("")
    if nochat:
        out.append("### 数字（会話なし・同じ町・同じ seed）")
        out.append("")
        out += numbers_block(nochat)
        out.append("")
    out.append("### 対比")
    out.append("")
    out += compare_table(chat, nochat, v8b)
    out.append("")
    out.append("### 月別（会話あり）")
    out.append("")
    out += monthly_block(chat)
    out.append("")
    if nochat:
        out.append("### 月別（会話なし）")
        out.append("")
        out += monthly_block(nochat)
        out.append("")
    out += texts_block(chat)
    out.append("")
    out.append("### 内心・理由と選択の食い違いの候補（決定論の語句照合・判定ではない）")
    out.append("")
    out += mismatch_block(chat)
    md = "\n".join(out)
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
