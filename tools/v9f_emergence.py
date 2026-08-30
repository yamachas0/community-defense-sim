#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v9f の観測をぜんぶ機械で数える（決定論・API を一切使わない・採点 LLM なし）。

  python tools/v9_emergence.py simulations/<v9_run_dir> \
      --v8d simulations/<v8d_run_dir> --out docs/submission/emergence_v9.json

設計 `docs/world_design_v9.md` §7 の 1〜9 を数える。**分類も判定もしない**
（理由は原文のまま全件保存し、走行前に凍結した語彙の件数だけ数える）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

# 走行前に凍結した語彙（v8d と同じ表を作るため・結果を見てから足さない）
FOREIGN_WORDS = ["海外", "外国", "国外", "外資", "グローバル", "インバウンド"]
MONEY_WORDS = ["お金", "費用", "負担", "対価", "価格", "金額", "値段", "報酬",
               "補償", "金銭", "資金", "代金"]
# 断りの一言で数える語（v9 の報告と同じ表を作るため・走行前に凍結）
DECLINE_WORDS = ["不明", "条件", "生活", "地域", "必要", "時期", "継続", "海外",
                 "支援", "詳細", "具体"]

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.field_v9b import (EXECUTED_FACTS, NEGATIONS,  # noqa: E402
                           PROMISE_STRONG, PROMISE_WEAK)


def load(run_dir: str, name: str, default=None):
    p = os.path.join(run_dir, name)
    if not os.path.exists(p):
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def word_counts(texts: List[str], words: List[str]) -> Dict[str, Any]:
    hit = {w: 0 for w in words}
    rows = 0
    for t in texts:
        t = str(t or "")
        if not t:
            continue
        if any(w in t for w in words):
            rows += 1
        for w in words:
            hit[w] += t.count(w)
    return {"texts_checked": len([t for t in texts if str(t or "").strip()]),
            "texts_with_any": rows,
            "by_word": {w: c for w, c in hit.items() if c},
            "total_hits": sum(hit.values())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default="docs/submission/emergence_v9f.json")
    args = ap.parse_args()
    r = args.run_dir

    s = load(r, "summary.json", {})
    monthly = load(r, "monthly.json", [])
    offers = load(r, "offers.json", [])
    decisions = load(r, "decisions.json", [])
    utter = load(r, "utterances.json", [])
    notices = load(r, "notices.json", [])
    left = load(r, "left_agents.json", [])
    transfers = load(r, "transfers.json", [])
    ledger = load(r, "ledger_by_step.json", [])

    out: Dict[str, Any] = {"run_dir": os.path.basename(r),
                           "run_name": s.get("run_name"),
                           "months_run": s.get("months_run"),
                           "cost_usd_local": s.get("cost_usd"),
                           "stopped_by_cost": s.get("stopped_by_cost")}

    out["acquisition"] = {
        "parcels_total": s.get("parcels_total"),
        "sellable_parcels": s.get("sellable_parcels"),
        "acquired_parcels": s.get("acquired_parcels"),
        "acquired_land": s.get("acquired_land"),
        "acquired_buildings": s.get("acquired_buildings"),
        "acquired_both": s.get("acquired_both"),
        "curve": [{"month": m["step"], "parcels_cum": m["parcels_cum"],
                   "land_cum": m["land_cum"], "building_cum": m["building_cum"],
                   "both_cum": m["both_cum"], "left_cum": m["left_cum"]}
                  for m in monthly],
    }

    out["offers_by_kind"] = {
        "total": s.get("offers_by_kind"),
        "sold": s.get("sold_by_kind"),
        "monthly": [dict({"month": m["step"]}, **m["offers_by_kind"])
                    for m in monthly],
    }

    out["listings"] = {
        "choice_counts": s.get("listing_choice_counts"),
        "by_kind": s.get("listings_by_kind"),
        "monthly": [dict({"month": m["step"], "listed": m["listed_this_month"]},
                         **m["listed_by_kind"]) for m in monthly],
    }

    out["by_where_owner_lives"] = {
        "offers_to_in_town": s.get("offers_to_in_town"),
        "offers_to_absentee": s.get("offers_to_absentee"),
        "sold_from_in_town": s.get("sold_from_in_town"),
        "sold_from_absentee": s.get("sold_from_absentee"),
        "accept_rate_in_town": s.get("accept_rate_in_town"),
        "accept_rate_absentee": s.get("accept_rate_absentee"),
    }
    # 呼び名から町外が読める唯一の主体（v8 凍結名）を除いた感度も出す
    leaky = "船着場裏の不在地主"
    ex_ab = [o for o in offers if o["to"] != leaky and not o["to_in_town"]]
    out["by_where_owner_lives"]["absentee_excluding_frozen_name"] = {
        "excluded": leaky,
        "offers": len(ex_ab),
        "sold": len([o for o in ex_ab if o["accepted"]]),
        "accept_rate": (round(len([o for o in ex_ab if o["accepted"]])
                              / len(ex_ab), 4) if ex_ab else None),
    }

    out["leaving"] = {
        "left_total": s.get("left_agents"),
        "who": [{"month": x["step"], "name": x["name"], "parcel": x["parcel"],
                 "still_owns": x["still_owns"]} for x in left],
        "in_town_end": s.get("in_town_end"),
        "absentee_owners_end": s.get("absentee_owners_end"),
        "monthly": [{"month": m["step"], "in_town": m["in_town"],
                     "absentee_owners": m["absentee_owners"],
                     "left_cum": m["left_cum"],
                     "no_user_parcels": m["no_user_parcels"]} for m in monthly],
    }

    # 借家人・借地人の声（自分の使う区画の所有者が X社 になった後の発言）
    became_x: Dict[str, int] = {}
    for t in transfers:
        became_x.setdefault(t["parcel"], t["step"])
    user_at: Dict[int, Dict[str, Any]] = {}
    for row in ledger:
        user_at[row["step"]] = {x["parcel"]: x.get("user") for x in row["rows"]}
    after_rows = []
    for u in utter:
        st = u["step"]
        for parcel, m0 in became_x.items():
            if st <= m0:
                continue
            if user_at.get(st, {}).get(parcel) == u["from"]:
                after_rows.append({"month": st, "who": u["from"],
                                   "parcel": parcel, "at": u["venue_label"],
                                   "text": u["text"]})
                break
    out["voices_after_the_owner_changed"] = {"count": len(after_rows),
                                             "rows": after_rows}

    # 権利の形ごとの売買
    start = {x["parcel"]: x for x in (ledger[0]["rows"] if ledger else [])}

    def shape(p: str) -> str:
        x = start.get(p)
        if not x:
            return "不明"
        if x.get("tenant") and x.get("building"):
            return "借家がある区画"
        if x.get("tenant") and not x.get("building"):
            return "借地の区画"
        if x.get("building") and x["building"] != x["land"]:
            return "借地の区画"
        return "所有者が使っている区画"

    groups: Dict[str, Dict[str, Any]] = {}
    for o in offers:
        g = groups.setdefault(shape(o["parcel"]), {"offers": 0, "sold": 0})
        g["offers"] += 1
        g["sold"] += 1 if o["accepted"] else 0
    for g in groups.values():
        g["accept_rate"] = (round(g["sold"] / g["offers"], 4)
                            if g["offers"] else None)
    out["by_rights_shape"] = groups

    # 理由は全件そのまま
    decline = [{"month": o["step"], "to": o["to"], "in_town": o["to_in_town"],
                "parcel": o["parcel"], "kind": o["kind"],
                "text": o.get("decline_reason", "")}
               for o in offers if not o["accepted"]]
    listing_reasons = []
    for d in decisions:
        for parcel, txt in (d.get("listing_reasons") or {}).items():
            listing_reasons.append({"month": d["step"], "who": d["name"],
                                    "in_town": d["in_town"], "parcel": parcel,
                                    "choice": (d.get("listings") or {}).get(parcel),
                                    "text": txt})
    sell_reasons = [{"month": d["step"], "who": d["name"], "in_town": d["in_town"],
                     "parcel": d.get("offer_parcel"), "kind": d.get("offer_kind"),
                     "answer": d.get("sell"), "text": d.get("sell_reason", "")}
                    for d in decisions if d.get("offer")]
    acquirer_reasons = [{"month": o["step"], "to": o["to"], "parcel": o["parcel"],
                         "kind": o["kind"], "text": o.get("reason", "")}
                        for o in offers]
    out["reasons_verbatim"] = {"decline": decline, "listing": listing_reasons,
                               "sell": sell_reasons, "acquirer": acquirer_reasons}

    texts = {
        "utterances": [u["text"] for u in utter],
        "thoughts": [u.get("thought", "") for u in utter],
        "decline_reason": [d["text"] for d in decline],
        "listing_reason": [x["text"] for x in listing_reasons],
        "sell_reason": [x["text"] for x in sell_reasons],
        "acquirer_offer_text": [o["text"] for o in offers],
    }
    out["words"] = {
        "foreign": {k: word_counts(v, FOREIGN_WORDS) for k, v in texts.items()},
        "money": {k: word_counts(v, MONEY_WORDS) for k, v in texts.items()},
        "vocab_frozen_before_the_run": {"foreign": FOREIGN_WORDS,
                                        "money": MONEY_WORDS},
    }

    out["notices"] = {"total": len(notices),
                      "rows": [{"month": n["step"], "to": n["to_name"],
                                "parcel": n["parcel"], "kind": n["kind"]}
                               for n in notices]}

    out["health"] = {k: s.get(k) for k in
                     ("no_answer", "plan_no_answer", "sell_no_answer",
                      "invalid_listing", "listing_missing", "invalid_sell",
                      "invalid_venue", "truncated", "parse_fail",
                      "acquirer_missing_targets", "acquirer_dup_rows",
                      "acquirer_off_range", "acquirer_chunk_fail",
                      "acquirer_empty_text", "acquirer_invalid_offer",
                      "acquirer_missing_parcel", "timeout_retries",
                      "timeout_giveups", "cached_ratio")}


    # --- A の観測（v9b で足したもの） -------------------------------------
    und = load(r, "undelivered.json", [])
    by_reason: Dict[str, int] = {}
    by_month: Dict[int, Dict[str, int]] = {}
    by_to: Dict[str, int] = {}
    for u in und:
        by_reason[u["why"]] = by_reason.get(u["why"], 0) + 1
        mrow = by_month.setdefault(int(u["step"]), {})
        mrow[u["why"]] = mrow.get(u["why"], 0) + 1
        by_to[u["to"]] = by_to.get(u["to"], 0) + 1
    out["undelivered"] = {
        "total": len(und),
        "by_reason": by_reason,
        "monthly": [{"month": m, **by_month[m]} for m in sorted(by_month)],
        "by_target": dict(sorted(by_to.items(), key=lambda kv: -kv[1])),
        "rows_verbatim": und,
        "vocab_frozen_before_the_run": {
            "strong": list(PROMISE_STRONG), "weak": list(PROMISE_WEAK),
            "negations": list(NEGATIONS), "executed_facts": list(EXECUTED_FACTS)},
    }

    # X社の条件文の語彙（届いたもの／届かなかったもの・月別の推移も）
    def vocab_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        words = list(PROMISE_STRONG) + list(PROMISE_WEAK)
        return word_counts([x.get("text", "") for x in rows], words)

    months = sorted({int(x["step"]) for x in offers} | {int(u["step"]) for u in und})
    out["acquirer_vocabulary"] = {
        "delivered": vocab_rows(offers),
        "undelivered": vocab_rows(und),
        "all": vocab_rows(list(offers) + list(und)),
        "monthly_shien": [
            {"month": m,
             "delivered_texts": len([o for o in offers if o["step"] == m]),
             "undelivered_texts": len([u for u in und if u["step"] == m]),
             "shien_hits": sum(str(x.get("text", "")).count("支援")
                               for x in list(offers) + list(und)
                               if x["step"] == m)}
            for m in months],
    }

    # 断りの一言に多い語（v9 の表と同じ語で数える＝走行前に凍結）
    decline_texts = [d["text"] for d in decline]
    out["decline_words"] = word_counts(decline_texts, DECLINE_WORDS)
    freq: Dict[str, int] = {}
    for t in decline_texts:
        t = str(t or "").strip()
        if t:
            freq[t] = freq.get(t, 0) + 1
    out["decline_top_phrases"] = [
        {"text": k, "count": v}
        for k, v in sorted(freq.items(), key=lambda kv: -kv[1])[:20]]


    # --- v9c の観測（借りて使っている人・一言・通知） ----------------------
    tdec = load(r, "tenant_decisions.json", [])
    left_t = load(r, "left_tenants.json", [])
    msgs = load(r, "messages.json", [])
    vnot = load(r, "vacancy_notices.json", [])

    out["tenants"] = {
        "calls_total": len(tdec),
        "leave_counts": s.get("tenant_leave_counts"),
        "no_answer": s.get("tenant_no_answer"),
        "left_total": len(left_t),
        "left_rows": [{"month": x["step"], "who": x["name"],
                       "parcels": x["parcels"], "reason": x.get("reason", "")}
                      for x in left_t],
        "leave_reasons_verbatim": [
            {"month": x["step"], "who": x["name"], "answer": x["leave"],
             "text": x.get("leave_reason", "")} for x in tdec],
        "monthly": [{"month": m["step"],
                     "calls": m.get("tenant_calls"),
                     "left": m.get("tenant_left_this_month"),
                     "left_cum": m.get("tenant_left_cum"),
                     "no_user_parcels": m.get("no_user_parcels")}
                    for m in monthly],
    }

    out["vacancy_notices"] = {
        "total": len(vnot),
        "rows": [{"month": x["step"], "to": x["to_name"], "parcel": x["parcel"],
                  "tenant": x["tenant"]} for x in vnot],
    }

    t2l = [m for m in msgs if m["direction"].startswith("借りて")]
    l2t = [m for m in msgs if m["direction"].startswith("家主")]
    out["messages"] = {
        "total": len(msgs),
        "tenant_to_landlord": len(t2l),
        "landlord_to_tenant": len(l2t),
        "reply_rate": (round(len(l2t) / len(t2l), 4) if t2l else None),
        "monthly": [{"month": m["step"], "messages": m.get("messages_this_month")}
                    for m in monthly],
        "rows_verbatim": msgs,
    }

    # 家主の応諾率＝一言が届いていた家主とそうでない家主で分ける
    got_line: Dict[str, set] = {}
    for m in t2l:
        got_line.setdefault(str(m["to"]), set()).add(int(m["step"]))
    with_line = {"offers": 0, "sold": 0}
    without = {"offers": 0, "sold": 0}
    for o in offers:
        months = got_line.get(str(o["to"]), set())
        # その月までに一言が届いたことのある家主か
        bucket = with_line if any(x < int(o["step"]) for x in months) else without
        bucket["offers"] += 1
        bucket["sold"] += 1 if o["accepted"] else 0
    for b in (with_line, without):
        b["accept_rate"] = (round(b["sold"] / b["offers"], 4)
                            if b["offers"] else None)
    out["landlord_accept_rate"] = {
        "landlords_who_had_received_a_line": with_line,
        "everyone_else": without,
    }


    # --- v9d の観測（お金） --------------------------------------------------
    # 「得なのに断った」「損なのに売った」は**観測の名前**であって、
    # 世界の文には1文字も出していない（設計 §4）。
    pays = load(r, "payments.json", [])
    wallets = load(r, "wallets.json", {})

    X_WORDS = ["X社", "海外", "投資", "買収", "申し出", "提示"]
    touched = {}
    for u in utter:
        t = str(u.get("text", "") or "")
        if any(w in t for w in X_WORDS):
            who = u["from"]
            touched[who] = min(touched.get(who, 10 ** 9), int(u["step"]))

    rows = []
    for o in offers:
        ratio = o.get("ratio")
        if ratio is None:
            continue
        spoke = touched.get(o["to"])
        rows.append({
            "month": o["step"], "to": o["to"], "in_town": o["to_in_town"],
            "parcel": o["parcel"], "kind": o["kind"], "amount": o["amount"],
            "valuation": o["valuation"], "ratio": ratio,
            "sold": bool(o["accepted"]),
            "at_or_above_valuation": ratio >= 1.0,
            "spoke_about_the_buyer_before": bool(spoke is not None
                                                 and spoke <= int(o["step"])),
        })

    def bucket(pred):
        sel = [x for x in rows if pred(x)]
        return {"count": len(sel), "who": sorted({x["to"] for x in sel}),
                "rows": sel}

    out["money"] = {
        "valuation_total": s.get("valuation_total"),
        "budget_total": s.get("budget_total"),
        "budget_share": s.get("budget_share"),
        "spent_total": s.get("spent_total"),
        "budget_left": s.get("budget_left"),
        "over_budget_offers": s.get("acquirer_over_budget"),
        "bad_amount_offers": s.get("acquirer_bad_amount"),
        "payments": pays,
        "wallets_nonzero": {k: v for k, v in (wallets or {}).items() if v},
        "monthly": [{"month": m["step"], "spent_cum": m.get("spent_cum"),
                     "budget_left": m.get("budget_left"),
                     "paid_this_month": m.get("paid_this_month")}
                    for m in monthly],
    }

    ratios = [x["ratio"] for x in rows]
    ratios_sorted = sorted(ratios)

    def pct(p):
        if not ratios_sorted:
            return None
        i = min(len(ratios_sorted) - 1, int(p * (len(ratios_sorted) - 1)))
        return ratios_sorted[i]

    out["offer_ratio"] = {
        "offers_with_amount": len(rows),
        "mean": (round(sum(ratios) / len(ratios), 4) if ratios else None),
        "min": (min(ratios) if ratios else None),
        "p25": pct(0.25), "median": pct(0.5), "p75": pct(0.75),
        "max": (max(ratios) if ratios else None),
        "at_or_above_valuation": len([x for x in rows if x["ratio"] >= 1.0]),
        "below_valuation": len([x for x in rows if x["ratio"] < 1.0]),
        "monthly_mean": [
            {"month": mm,
             "offers": len([x for x in rows if x["month"] == mm]),
             "mean": (round(sum(x["ratio"] for x in rows if x["month"] == mm)
                            / len([x for x in rows if x["month"] == mm]), 4)
                      if [x for x in rows if x["month"] == mm] else None)}
            for mm in sorted({x["month"] for x in rows})],
    }

    out["money_decisions"] = {
        "note": "「得なのに断った／損なのに売った」は観測の名前であり、"
                "世界の文には一切出していない。",
        "refused_at_or_above_valuation": bucket(
            lambda x: x["at_or_above_valuation"] and not x["sold"]),
        "sold_below_valuation": bucket(
            lambda x: (not x["at_or_above_valuation"]) and x["sold"]),
        "sold_at_or_above_valuation": bucket(
            lambda x: x["at_or_above_valuation"] and x["sold"]),
        "refused_below_valuation_count": len(
            [x for x in rows if (not x["at_or_above_valuation"])
             and not x["sold"]]),
        "by_whether_they_spoke_about_the_buyer": {
            "spoke": {
                "offers": len([x for x in rows
                               if x["spoke_about_the_buyer_before"]]),
                "refused_at_or_above": len(
                    [x for x in rows if x["spoke_about_the_buyer_before"]
                     and x["at_or_above_valuation"] and not x["sold"]]),
                "sold_below": len(
                    [x for x in rows if x["spoke_about_the_buyer_before"]
                     and not x["at_or_above_valuation"] and x["sold"]]),
                "sold": len([x for x in rows
                             if x["spoke_about_the_buyer_before"] and x["sold"]]),
            },
            "did_not_speak": {
                "offers": len([x for x in rows
                               if not x["spoke_about_the_buyer_before"]]),
                "refused_at_or_above": len(
                    [x for x in rows if not x["spoke_about_the_buyer_before"]
                     and x["at_or_above_valuation"] and not x["sold"]]),
                "sold_below": len(
                    [x for x in rows if not x["spoke_about_the_buyer_before"]
                     and not x["at_or_above_valuation"] and x["sold"]]),
                "sold": len([x for x in rows
                             if not x["spoke_about_the_buyer_before"]
                             and x["sold"]]),
            },
            "vocab_frozen_before_the_run": X_WORDS,
        },
        "absentee_owners_sold": [
            {"month": x["month"], "who": x["to"], "parcel": x["parcel"],
             "kind": x["kind"], "ratio": x["ratio"]}
            for x in rows if x["sold"] and not x["in_town"]],
        "rows_all": rows,
    }

    MONEY_WORDS2 = ["円", "金額", "価格", "評価額", "値段", "資金", "支払",
                    "対価", "お金"]
    out["money_words"] = {
        "decline": word_counts([d["text"] for d in decline], MONEY_WORDS2),
        "listing": word_counts([x["text"] for x in listing_reasons], MONEY_WORDS2),
        "sell": word_counts([x["text"] for x in sell_reasons], MONEY_WORDS2),
        "utterances": word_counts([u["text"] for u in utter], MONEY_WORDS2),
        "vocab_frozen_before_the_run": MONEY_WORDS2,
    }


    # --- v9e の観測（過半への進み具合） --------------------------------------
    prog = load(r, "progress.json", [])
    out["majority"] = {
        "mandate": s.get("mandate"),
        "total_value": s.get("valuation_total"),
        "total_area_m2": s.get("total_area_m2"),
        "acquired_value_end": s.get("acquired_value_end"),
        "acquired_area_m2_end": s.get("acquired_area_m2_end"),
        "value_share_end": s.get("value_share_end"),
        "area_share_end": s.get("area_share_end"),
        "reached_majority_value": s.get("reached_majority_value"),
        "reached_majority_area": s.get("reached_majority_area"),
        "budget_used_share": s.get("budget_used_share"),
        "monthly": prog,
    }


    # --- v9f の観測（積んだ提示・再提示のエスカレーション） -------------------
    reoff = load(r, "reoffers.json", [])
    piled = [x for x in offers if (x.get("ratio") or 0) >= 1.2]
    out["piling"] = {
        "threshold_ratio": 1.2,
        "piled_offers": len(piled),
        "piled_accepted": len([x for x in piled if x["accepted"]]),
        "piled_accept_rate": (round(len([x for x in piled if x["accepted"]])
                                    / len(piled), 4) if piled else None),
        "other_offers": len(offers) - len(piled),
        "other_accepted": len([x for x in offers
                               if (x.get("ratio") or 0) < 1.2 and x["accepted"]]),
        "note": "この成約率は相手・物件の選び方と再提示が内生的なので、"
                "高く積んだことの因果効果とは読めない（走行前レビューの指摘）。",
        "rows": [{"month": x["step"], "to": x["to"], "parcel": x["parcel"],
                  "kind": x["kind"], "amount": x["amount"],
                  "valuation": x["valuation"], "ratio": x["ratio"],
                  "sold": x["accepted"]} for x in piled],
    }
    ups = [x for x in reoff if x["delta"] > 0]
    downs = [x for x in reoff if x["delta"] < 0]
    ds = sorted(x["delta"] for x in ups)
    out["reoffers"] = {
        "total": len(reoff),
        "amount_up": len(ups), "amount_down": len(downs),
        "amount_same": len(reoff) - len(ups) - len(downs),
        "up_median_yen": (ds[len(ds) // 2] if ds else None),
        "up_max_yen": (ds[-1] if ds else None),
        "accepted_after_reoffer": len([x for x in reoff if x["accepted"]]),
        "accepted_after_raise": len([x for x in ups if x["accepted"]]),
        "rows": reoff,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({"out": args.out,
                      "undelivered": out["undelivered"]["total"],
                      "tenants_left": out["tenants"]["left_total"],
                      "spent": out["money"]["spent_total"],
                      "ratio_mean": out["offer_ratio"]["mean"],
                      "value_share": out["majority"]["value_share_end"],
                      "area_share": out["majority"]["area_share_end"],
                      "piled": out["piling"]["piled_offers"],
                      "raises": out["reoffers"]["amount_up"],
                      "messages": out["messages"]["total"],
                      "acquired": out["acquisition"]["acquired_parcels"],
                      "left": out["leaving"]["left_total"],
                      "offers": len(offers),
                      "voices_after": len(after_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
