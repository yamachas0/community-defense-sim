#!/usr/bin/env python
"""v8c「創発がどれだけ売買判断に効いたか」の数表（決定論・API を一切使わない）。

  python tools/v8c_emergence.py simulations/<v8c_run_dir> [--out-json ...] [--out-md ...]

設計 `docs/world_design_v8c.md` §4 の1〜4を機械で数える。**判定・分類はしない**：
理由の分類は**候補の抽出まで**（語彙による当たり付け）で、確定は人が原文を読む（別発注）。
採点用 LLM は v8c に存在しない。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

# 走行前に凍結した語彙（結果を見てから足さない）
SELF_WORDS = ["家族", "子", "進学", "高齢", "年", "体力", "後継", "相続", "生活",
              "経営", "商売", "仕事", "資金", "維持", "管理", "修繕", "住み",
              "自分", "将来", "健康", "引退", "愛着"]
OFFER_WORDS = ["条件", "申し出", "提示", "内容", "具体", "相手", "X社", "会社",
               "海外", "投資", "名義"]
HEARD_WORDS = ["聞い", "噂", "話", "みな", "皆", "周り", "他の", "近所", "隣",
               "町の", "街の", "様子", "動向", "情報"]


def _load(run_dir: str, name: str) -> Any:
    path = os.path.join(run_dir, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _bucket(text: str, names: List[str]) -> List[str]:
    """理由一言の**候補**分類（複数該当あり・確定ではない）。"""
    t = str(text or "")
    if not t:
        return []
    hits = []
    if any(w in t for w in SELF_WORDS):
        hits.append("自分の事情")
    if any(w in t for w in OFFER_WORDS):
        hits.append("X社の条件")
    if any(w in t for w in HEARD_WORDS) or any(n in t for n in names):
        hits.append("聞いた話")
    return hits


def reason_candidates(decisions: List[Dict[str, Any]],
                      names: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for field, label in (("listing_reason", "listing"), ("sell_reason", "sell")):
        rows = [d for d in decisions if str(d.get(field, "") or "").strip()]
        counts = {"自分の事情": 0, "X社の条件": 0, "聞いた話": 0,
                  "どれにも当たらない": 0}
        samples: Dict[str, List[Dict[str, Any]]] = {k: [] for k in counts}
        for d in rows:
            hits = _bucket(d[field], names) or ["どれにも当たらない"]
            choice = d.get("listing") if label == "listing" else d.get("sell")
            for h in hits:
                counts[h] += 1
                if len(samples[h]) < 8:
                    samples[h].append({"month": d["step"], "name": d["name"],
                                       "choice": choice, "reason": d[field]})
        out[label] = {"written": len(rows), "candidate_counts": counts,
                      "samples": samples,
                      "note": ("語彙による候補抽出。確定は人が原文を読む"
                               "（採点LLMは使わない）")}
    return out


def heard_before_decision(timelines: List[Dict[str, Any]],
                          window: int = 3) -> Dict[str, Any]:
    """判断の前 `window` か月に聞いた話の量（X社に触れたもの／すべて）。

    群は3つ：売った月の本人／出した月の本人／どちらもしなかった月の本人。
    """
    groups: Dict[str, List[Any]] = {"売った": [], "出した": [], "どちらもしない": []}
    for doc in timelines:
        months = doc["months"]
        by_month = {m["month"]: m for m in months}
        for m in months:
            if m["listing"] == "問われていない":
                continue
            heard_all = 0
            heard_x = 0
            for back in range(m["month"] - window, m["month"] + 1):
                mm = by_month.get(back)
                if not mm:
                    continue
                for h in mm["heard"]:
                    heard_all += 1
                    if "X社" in h["text"] or "海外" in h["text"]:
                        heard_x += 1
            if m["sell"] == "売る":
                key = "売った"
            elif m["listing"] == "出す":
                key = "出した"
            else:
                key = "どちらもしない"
            groups[key].append((heard_all, heard_x))
    out: Dict[str, Any] = {}
    for k, rows in groups.items():
        n = len(rows)
        out[k] = {
            "n": n,
            "heard_mean": round(sum(r[0] for r in rows) / n, 2) if n else 0.0,
            "heard_about_x_mean": (round(sum(r[1] for r in rows) / n, 2)
                                   if n else 0.0),
            "share_with_any_x_talk": (round(sum(1 for r in rows if r[1] > 0) / n, 4)
                                      if n else 0.0),
        }
    out["window_months"] = window
    return out


def contagion(timelines: List[Dict[str, Any]], transfers: List[Dict[str, Any]],
              deliveries: List[Dict[str, Any]], window: int = 3) -> Dict[str, Any]:
    """売った人の隣人・同席者が、そのあと出す／売る割合（対照＝それ以外の人）。"""
    docs = {d["agent_id"]: d for d in timelines}
    name_to_id = {d["name"]: d["agent_id"] for d in timelines}
    acted_after: Dict[str, List[int]] = {}
    for aid, doc in docs.items():
        acted_after[aid] = [m["month"] for m in doc["months"]
                            if m["listing"] == "出す" or m["sell"] == "売る"]

    exposed: set = set()
    events = []
    for t in transfers:
        seller = t["agent_id"]
        m0 = int(t["step"])
        if seller not in docs:
            continue
        nb = [name_to_id[n] for n in docs[seller]["neighbours"] if n in name_to_id]
        co = {d["to"] for d in deliveries
              if d["from_id"] == seller and int(d["step"]) == m0
              and d.get("route") == "居合わせ"}
        touched = (set(nb) | set(co)) - {seller}
        exposed |= touched
        after = [aid for aid in touched
                 if any(m0 < x <= m0 + window for x in acted_after.get(aid, []))]
        events.append({"month": m0, "seller": docs[seller]["name"],
                       "neighbours": len(nb), "co_present": len(co),
                       "touched": len(touched), "acted_within_window": len(after),
                       "who_acted": [docs[a]["name"] for a in after]})
    others = [aid for aid, doc in docs.items()
              if doc["sellable"] and aid not in exposed]
    base = (sum(1 for aid in others if acted_after.get(aid)) / len(others)
            if others else 0.0)
    return {"window_months": window, "events": events,
            "exposed_agents": len(exposed),
            "exposed_acted_share": (
                round(sum(1 for a in exposed if acted_after.get(a)) / len(exposed), 4)
                if exposed else 0.0),
            "not_exposed_agents": len(others),
            "not_exposed_acted_share": round(base, 4),
            "note": ("『触れた人』＝売った人の隣人 or 売った月に同席した人。"
                     "相関であって因果ではない")}


def acquirer_adaptation(offers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """断りの一言を受けたあと、X社の条件文が変わったか（原文の対つき）。"""
    by_to: Dict[str, List[Dict[str, Any]]] = {}
    for o in offers:
        by_to.setdefault(o["to"], []).append(o)
    changed_after_note = []
    changed_any = 0
    for to, rows in by_to.items():
        rows.sort(key=lambda r: int(r["step"]))
        for prev, nxt in zip(rows, rows[1:]):
            if prev["text"] != nxt["text"]:
                changed_any += 1
                if str(prev.get("decline_reason", "") or "").strip():
                    changed_after_note.append({
                        "to": to, "month_before": prev["step"],
                        "month_after": nxt["step"],
                        "decline_reason": prev["decline_reason"],
                        "text_before": prev["text"], "text_after": nxt["text"]})
    with_note = [o for o in offers if str(o.get("decline_reason", "") or "").strip()]
    followups = 0
    for o in with_note:
        nxt = [x for x in by_to[o["to"]] if int(x["step"]) > int(o["step"])]
        if nxt:
            followups += 1
    return {
        "distinct_texts": len({o["text"] for o in offers}),
        "offers_total": len(offers),
        "offers_with_decline_note": len(with_note),
        "notes_followed_by_another_offer": followups,
        "text_changes_between_consecutive_offers": changed_any,
        "changes_right_after_a_note": len(changed_after_note),
        "examples": changed_after_note[:10],
    }


def build(run_dir: str) -> Dict[str, Any]:
    summary = _load(run_dir, "summary.json")
    decisions = _load(run_dir, "decisions.json") or []
    offers = _load(run_dir, "offers.json") or []
    deliveries = _load(run_dir, "deliveries.json") or []
    transfers = _load(run_dir, "transfers.json") or []
    monthly = _load(run_dir, "monthly.json") or []
    index = _load(run_dir, "timeline_index.json") or []
    timelines = []
    for row in index:
        with open(os.path.join(run_dir, row["file"]), encoding="utf-8") as f:
            timelines.append(json.load(f))
    names = [t["name"] for t in timelines]
    # 走った月より後の記録（費用上限で月の途中に止まった場合に残る、誰にも問われていない
    # 提示など）は集計から外す（読み取り監査 2026-08-30 の指摘）。
    months_run = int(summary.get("months_run", 0))
    decisions = [d for d in decisions if int(d["step"]) <= months_run]
    offers = [o for o in offers if int(o["step"]) <= months_run]
    deliveries = [d for d in deliveries if int(d["step"]) <= months_run]
    transfers = [t for t in transfers if int(t["step"]) <= months_run]
    return {
        "run": {"run_dir": os.path.basename(run_dir),
                "run_name": summary.get("run_name"),
                "chat": summary.get("chat"), "seed": summary.get("seed"),
                "months_run": summary.get("months_run"),
                "acquired_parcels": summary.get("acquired_parcels"),
                "sold_agents": summary.get("sold_agents"),
                "cost_usd": summary.get("cost_usd")},
        "reason_candidates": reason_candidates(decisions, names),
        "heard_before_decision": heard_before_decision(timelines),
        "contagion": contagion(timelines, transfers, deliveries),
        "acquirer_adaptation": acquirer_adaptation(offers),
        "monthly": [{"month": m["step"], "offers": m["offers_sent"],
                     "listed": m["listed_this_month"],
                     "sold": m["accepted_this_month"],
                     "parcels_cum": m["parcels_cum"],
                     "declines_with_reason": m.get("declines_with_reason", 0),
                     "utterances": m["utterances"], "attended": m["attended"]}
                    for m in monthly],
    }


def to_markdown(e: Dict[str, Any]) -> str:
    out: List[str] = []
    rc = e["reason_candidates"]
    out.append("#### 1. 理由の一言は何を根拠にしているか（**候補抽出まで**・確定は人手）")
    out.append("")
    out.append("```")
    out.append("問い            書かれた数  自分の事情  X社の条件  聞いた話  どれにも当たらない")
    for label, jp in (("listing", "出す/出さない"), ("sell", "売る/売らない")):
        c = rc[label]["candidate_counts"]
        out.append(f"{jp:<14}{rc[label]['written']:>10}{c['自分の事情']:>11}"
                   f"{c['X社の条件']:>10}{c['聞いた話']:>10}"
                   f"{c['どれにも当たらない']:>18}")
    out.append("```")
    out.append("")
    out.append("※ 1つの理由が複数に当たることがある（合計は書かれた数と一致しない）。"
               "**これは機械の当たり付けであって分類の確定ではない**（確定は人手・別発注）。")
    out.append("")
    h = e["heard_before_decision"]
    out.append(f"#### 2. 判断の前{h['window_months']}か月に聞いた話の量")
    out.append("")
    out.append("```")
    out.append("その月の選択        件数   聞いた話(平均)   うちX社の話(平均)   X社の話を1件でも聞いた割合")
    for k in ("売った", "出した", "どちらもしない"):
        r = h[k]
        out.append(f"{k:<16}{r['n']:>8}{r['heard_mean']:>14.2f}"
                   f"{r['heard_about_x_mean']:>18.2f}"
                   f"{r['share_with_any_x_talk']*100:>24.1f}%")
    out.append("```")
    out.append("")
    c = e["contagion"]
    out.append(f"#### 3. 伝染の形（売った人の隣人・同席者が、その後{c['window_months']}か月で"
               "出す／売るか）")
    out.append("")
    out.append(f"- 触れた人 {c['exposed_agents']}人のうち、その後に出す／売るを選んだ割合 "
               f"**{c['exposed_acted_share']*100:.1f}%**")
    out.append(f"- 触れていない人 {c['not_exposed_agents']}人では "
               f"**{c['not_exposed_acted_share']*100:.1f}%**")
    out.append(f"- {c['note']}")
    out.append("")
    a = e["acquirer_adaptation"]
    out.append("#### 4. X社は断りの一言に反応したか")
    out.append("")
    out.append(f"- 提示 {a['offers_total']}件・条件文の種類 {a['distinct_texts']}通り")
    out.append(f"- 断りの一言が付いた提示 {a['offers_with_decline_note']}件／"
               f"そのうち同じ相手に次の提示が続いたもの "
               f"{a['notes_followed_by_another_offer']}件")
    out.append(f"- 同じ相手への連続する提示で条件文が変わった回数 "
               f"{a['text_changes_between_consecutive_offers']}回／"
               f"そのうち直前に断りの一言があったもの {a['changes_right_after_a_note']}回")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()
    e = build(args.run_dir)
    path = args.out_json or os.path.join(args.run_dir, "emergence_v8c.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(e, f, ensure_ascii=False, indent=2)
    md = to_markdown(e)
    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(md)
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"wrote {path}")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
