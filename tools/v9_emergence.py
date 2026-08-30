#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v9 の観測をぜんぶ機械で数える（決定論・API を一切使わない・採点 LLM なし）。

  python tools/v9_emergence.py simulations/<v9_run_dir> \
      --v8d simulations/<v8d_run_dir> --out docs/submission/emergence_v9.json

設計 `docs/world_design_v9.md` §7 の 1〜9 を数える。**分類も判定もしない**
（理由は原文のまま全件保存し、走行前に凍結した語彙の件数だけ数える）。
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

# 走行前に凍結した語彙（v8d と同じ表を作るため・結果を見てから足さない）
FOREIGN_WORDS = ["海外", "外国", "国外", "外資", "グローバル", "インバウンド"]
MONEY_WORDS = ["お金", "費用", "負担", "対価", "価格", "金額", "値段", "報酬",
               "補償", "金銭", "資金", "代金"]


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
    ap.add_argument("--v8d", default=None, help="比較する v8d の run_dir")
    ap.add_argument("--out", default="docs/submission/emergence_v9.json")
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
        # 借家＝**建物**を借りて使っている人がいる区画（9区画）。
        # 建物の無い区画を借りて使っている場合（駐車場2区画）は借地に数える。
        # 定義の正＝docs/world_design_v9.md §1「借家がある区画は9（②3＋③6）、
        # 借地の区画は9（①7＋④の借地2）」。2026-08-30 に食い違いを修正。
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

    if args.v8d:
        v = load(args.v8d, "summary.json", {})
        vm = load(args.v8d, "monthly.json", [])
        out["v8d_for_comparison"] = {
            "run_name": v.get("run_name"),
            "agents": v.get("agents"),
            "parcels_total": v.get("parcels_total"),
            "acquired_parcels": v.get("parcels_acquired",
                                      v.get("acquired_parcels")),
            "sold_agents": v.get("sold_agents"),
            "offers_total": v.get("offers_total"),
            "cost_usd_local": v.get("cost_usd"),
            "curve": [{"month": m["step"], "parcels_cum": m.get("parcels_cum")}
                      for m in vm],
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({"out": args.out,
                      "acquired": out["acquisition"]["acquired_parcels"],
                      "left": out["leaving"]["left_total"],
                      "offers": len(offers),
                      "voices_after": len(after_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
