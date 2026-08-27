#!/usr/bin/env python
"""1ランの判定指標を、events.jsonl / ledger.jsonl / summary.json から機械的に集計する。

    python tools/run_metrics.py --run simulations/<run_dir> [--json]

run36 / run38 の診断で使った指標と同じ定義で出す（版をまたいで比較できるように）。
ここでは解釈も評価もしない。数える対象は帳簿と行動記録に実在する事実だけ。
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
from typing import Any, Dict, List

MEETING_WORDS = ("来週", "日程", "ヒアリング", "訪問", "打ち合わせ", "アポ",
                 "面談", "お伺い", "伺いま", "調整")
MONEY_RE = re.compile(r"\d+\s*万")


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def metrics_v41(run_dir: str) -> Dict[str, Any]:
    """v4.1（金額のない世界）の集計。判定基準は docs/world_design_v4_impl.md と同じ定義。"""
    events = _read_jsonl(os.path.join(run_dir, "events.jsonl"))
    ledger = _read_jsonl(os.path.join(run_dir, "ledger.jsonl"))
    thoughts = _read_jsonl(os.path.join(run_dir, "thoughts.jsonl"))
    deliveries = _read_jsonl(os.path.join(run_dir, "deliveries.jsonl"))
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)

    kinds = collections.Counter(r.get("kind", "") for r in ledger)
    offer_records = [r for r in ledger if r.get("kind") == "offer"]
    transfers = [r for r in ledger if r.get("kind") == "transfer"]
    acquirer_ids = {e["agent_id"] for e in events if e.get("role") == "acquirer"}
    valid_transfers = [r for r in transfers if r.get("buyer") in acquirer_ids
                       and r.get("buyer") != r.get("seller")]

    offer_ops = [op for e in events if e.get("role") == "acquirer"
                 for op in e.get("operations", []) or []
                 if op.get("action_type") == "make_offer"]
    offer_fail = collections.Counter(
        op.get("outcome", {}).get("reason", "")
        for op in offer_ops
        if str(op.get("outcome", {}).get("kind", "")).endswith(
            ("_rejected", "invalid_action")))

    response_ops = [op for e in events if e.get("action_type") == "responses"
                    for op in e.get("operations", []) or []]
    decisions = collections.Counter(op.get("action_type", "") for op in response_ops)
    sell_decisions = decisions.get("sell", 0)
    sell_transfer = sum(1 for op in response_ops
                        if op.get("outcome", {}).get("kind") == "transfer")
    sell_filing = sum(1 for op in response_ops
                      if op.get("outcome", {}).get("kind") == "filing_required")
    sell_failed = sum(1 for op in response_ops
                      if op.get("outcome", {}).get("kind") == "response_rejected"
                      and op.get("action_type") == "sell")
    filed_ids = {r.get("offer_id") for r in ledger if r.get("kind") == "filing_required"}
    filed_completed = [r for r in transfers if r.get("offer_id") in filed_ids]

    parse_fail = sum(1 for e in events if e.get("action_type") == "PARSE_FAIL")
    offer_steps = sorted({r["step"] for r in offer_records})
    called = {(e["step"], e["agent_id"]) for e in events
              if e.get("action_type") == "responses"}
    about = [t for t in thoughts if t.get("about_acquisition")]
    about_people = sorted({t["from"] for t in about})
    about_without_own_offer = [t for t in about
                               if (t["step"], t["from"]) not in called]
    money_rows = [r for r in ledger
                  if any(k in r for k in ("price", "amount", "rent", "fee"))]

    # --- 初回反応と owner-month 反応を分ける（長期holdが分母を膨らませないように）---
    first_response: Dict[str, str] = {}
    for e in sorted(events, key=lambda x: (x.get("step", 0), x.get("agent_id", ""))):
        if e.get("action_type") != "responses":
            continue
        for op in e.get("operations", []) or []:
            oid = str(op.get("target", "")).strip().upper().strip("[]")
            if oid and oid not in first_response:
                first_response[oid] = op.get("action_type", "")
    first_decisions = collections.Counter(first_response.values())
    implicit_holds = sum(1 for e in events if e.get("action_type") == "responses"
                         for op in e.get("operations", []) or []
                         if op.get("implicit"))

    # --- 届出（条例）による実現遅延を offer 単位で照合する ---
    filing_rows = [r for r in ledger if r.get("kind") == "filing_required"]
    transfer_by_offer = {r.get("offer_id"): r for r in transfers}
    filing_delays = []
    for row in filing_rows:
        done = transfer_by_offer.get(row.get("offer_id"))
        filing_delays.append({
            "offer_id": row.get("offer_id"), "parcel_id": row.get("parcel_id"),
            "sell_step": row.get("step"), "due_step": row.get("due_step"),
            "transfer_step": done.get("step") if done else None,
            "realized_delay": ((done.get("step") - row.get("step")) if done else None),
        })
    pending_at_end = [r["offer_id"] for r in filing_delays
                      if r["transfer_step"] is None]

    # --- 認知の転相を主体ごとに見る（総件数ではなく個人内の初回転換）---
    first_transition = {}
    for t in sorted(thoughts, key=lambda x: (x.get("step", 0), x.get("from", ""))):
        if t.get("about_acquisition") and t.get("from") not in first_transition:
            first_transition[t["from"]] = t.get("step")
    transition_steps = sorted(first_transition.values())

    # --- 噂（取得を扱う発話）の初出月 ---
    rumor_steps = sorted({d.get("step") for d in deliveries
                          if d.get("kind") in ("ambient", "direct")
                          and any(w in str(d.get("text", ""))
                                  for w in ("名義", "買い", "取得", "外資", "X社",
                                            "まとめて", "手放"))})

    # --- 名義の分散（面積シェアのハーフィンダール指数）---
    name_area = collections.Counter()
    for r in valid_transfers:
        name_area[r.get("under_name", "")] += 1
    total_named = sum(name_area.values()) or 1
    name_hhi = round(sum((v / total_named) ** 2 for v in name_area.values()), 4)

    return {
        "run_dir": os.path.basename(run_dir.rstrip("/\\")),
        "scenario": "field_v4_1",
        "steps": summary.get("steps"),
        "model": summary.get("model"),
        # --- 配線 ---
        "calls": summary.get("usage", {}).get("calls"),
        "api_errors": summary.get("usage", {}).get("errors"),
        "input_tokens": summary.get("usage", {}).get("input_tokens"),
        "output_tokens": summary.get("usage", {}).get("output_tokens"),
        "parse_fail": parse_fail,
        "truncated": summary.get("truncated_responses"),
        "invalid_actions": summary.get("invalid_actions"),
        "offers_returned_by_acquirer": len(offer_ops),
        "offers_recorded": len(offer_records),
        "offer_rejection_reasons": dict(offer_fail),
        "responders_called": sum(1 for e in events if e.get("action_type") == "responses"),
        "responses": dict(decisions),
        "sell_decisions": sell_decisions,
        "sell_immediate_transfer": sell_transfer,
        "sell_filing_required": sell_filing,
        "sell_rejected": sell_failed,
        "sell_reconciles": sell_decisions == (sell_transfer + sell_filing + sell_failed),
        "filed_transfer_completed": len(filed_completed),
        "filing_void": kinds.get("filing_void", 0),
        "offer_void": kinds.get("offer_void", 0),
        "response_rejected": kinds.get("response_rejected", 0),
        "ledger_rows_with_money_fields": len(money_rows),
        "deliveries_by_kind": dict(collections.Counter(d.get("kind", "")
                                                       for d in deliveries)),
        # --- 世界で起きたこと ---
        "months_with_offers": len(offer_steps),
        "first_offer_step": offer_steps[0] if offer_steps else None,
        "months_with_transfers": len({r["step"] for r in valid_transfers}),
        "first_transfer_step": (min(r["step"] for r in valid_transfers)
                                if valid_transfers else None),
        "transfers": len(valid_transfers),
        "withdrawn": kinds.get("withdraw", 0),
        "kept": kinds.get("keep", 0),
        "held": kinds.get("hold", 0),
        "ownership_share": summary.get("kpi", {}).get("final_acquirer_share"),
        "area_share": summary.get("kpi", {}).get("final_acquirer_area_share"),
        "under_names_used": dict(collections.Counter(r.get("under_name", "")
                                                     for r in valid_transfers)),
        "thoughts": len(thoughts),
        "thought_frames": dict(collections.Counter(t.get("frame", "") for t in thoughts)),
        "thoughts_about_acquisition": len(about),
        "residents_aware": len(about_people),
        "aware_without_own_offer": len(about_without_own_offer),
        "first_aware_step": min((t["step"] for t in about), default=None),
        "ordinances": [{"step": r["step"], "by": r.get("by"), "title": r.get("title"),
                        "threshold_sqm": r.get("threshold_sqm"),
                        "delay_months": r.get("delay_months")}
                       for r in ledger if r.get("kind") == "ordinance"],
        "publications": kinds.get("publication", 0),
        "publications_about_acquisition": sum(
            1 for r in ledger if r.get("kind") == "publication"
            and r.get("about_acquisition")),
        "investigations": kinds.get("investigate", 0),
        # --- 判定（事前固定） ---
        "verdict_acquisition_progresses": bool(
            len({r["step"] for r in valid_transfers}) >= 2
            and (summary.get("kpi", {}).get("final_acquirer_area_share") or 0) > 0),
        "verdict_residents_notice": bool(len(about_people) >= 2
                                         and len(about_without_own_offer) >= 1),
        # --- 追加集計（Codexレビュー 2026-08-27 の指摘反映）---
        "offers_first_month": sum(1 for r in offer_records if r.get("step") == 1),
        "first_response_by_offer": dict(first_decisions),
        "owner_month_responses": dict(decisions),
        "implicit_holds": implicit_holds,
        "response_not_recorded": kinds.get("response_not_recorded", 0),
        "ordinance_same_step_conflict": kinds.get("ordinance_same_step_conflict", 0),
        "filing_delays": filing_delays,
        "filings_pending_at_end": pending_at_end,
        "first_transition_step_by_agent": first_transition,
        "median_first_transition_step": (
            transition_steps[len(transition_steps) // 2] if transition_steps else None),
        "first_rumor_step": rumor_steps[0] if rumor_steps else None,
        "under_name_area_hhi": name_hhi,
    }


def metrics_v4(run_dir: str) -> Dict[str, Any]:
    """v4（同期3フェーズ）の集計。配線の指標と、世界で起きたことの指標を分けて出す。"""
    events = _read_jsonl(os.path.join(run_dir, "events.jsonl"))
    ledger = _read_jsonl(os.path.join(run_dir, "ledger.jsonl"))
    feelings = _read_jsonl(os.path.join(run_dir, "feelings.jsonl"))
    deliveries = _read_jsonl(os.path.join(run_dir, "deliveries.jsonl"))
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)

    kinds = collections.Counter(r.get("kind", "") for r in ledger)
    offer_records = [r for r in ledger if r.get("kind") == "offer"]
    transfers = [r for r in ledger if r.get("kind") == "transfer"]
    fees = [r for r in ledger if r.get("kind") == "broker_fee"]

    offer_ops = [op for e in events if e.get("role") == "acquirer"
                 for op in e.get("operations", []) or []
                 if op.get("action_type") == "make_offer"]
    offer_fail = collections.Counter(
        op.get("outcome", {}).get("reason", "")
        for op in offer_ops if str(op.get("outcome", {}).get("kind", "")).endswith(
            ("_rejected", "invalid_action")))
    via_counter = collections.Counter(op.get("via", "") for op in offer_ops)

    response_ops = [op for e in events if e.get("action_type") == "responses"
                    for op in e.get("operations", []) or []]
    decisions = collections.Counter(op.get("action_type", "") for op in response_ops)
    decision_results = collections.Counter(
        op.get("outcome", {}).get("kind", "") for op in response_ops)

    parse_fail = sum(1 for e in events if e.get("action_type") == "PARSE_FAIL")

    acquirer_ids = {e["agent_id"] for e in events if e.get("role") == "acquirer"}
    valid_transfers = [r for r in transfers if r.get("buyer") in acquirer_ids
                       and r.get("buyer") != r.get("seller")]
    filed_ids = {r.get("offer_id") for r in ledger if r.get("kind") == "filing_required"}
    filed_completed = [r for r in transfers if r.get("offer_id") in filed_ids]
    accept_decisions = sum(1 for op in response_ops if op.get("action_type") == "accept")
    accept_immediate = sum(1 for op in response_ops
                           if op.get("outcome", {}).get("kind") == "transfer")
    accept_filing = sum(1 for op in response_ops
                        if op.get("outcome", {}).get("kind") == "filing_required")
    accept_failed = sum(1 for op in response_ops
                        if op.get("outcome", {}).get("kind") == "accept_rejected")
    # 仲介経由の有効成立1件につき手数料が1件で、金額が契約料率と一致するか
    fee_by_offer = collections.Counter(r.get("offer_id") for r in fees)
    fee_duplicates = sum(1 for n in fee_by_offer.values() if n > 1)
    fee_expected = 0
    for record in valid_transfers + filed_completed:
        channel_fee = [r for r in fees if r.get("offer_id") == record.get("offer_id")]
        if channel_fee:
            fee_expected += int(round(record.get("price", 0) * channel_fee[0].get("rate", 0)))
    # 「その月に買付が届いていた所有者」＝応答フェーズに呼ばれた所有者
    called = {(e["step"], e["agent_id"]) for e in events
              if e.get("action_type") == "responses"}
    offer_steps = sorted({r["step"] for r in offer_records})
    transfer_steps = sorted({r["step"] for r in transfers})

    about = [f for f in feelings if f.get("about_acquisition")]
    about_people = sorted({f["from"] for f in about})
    about_without_own_offer = [f for f in about
                               if (f["step"], f["from"]) not in called]
    frames = collections.Counter(f.get("frame", "") for f in feelings)

    ordinances = [{"step": r["step"], "by": r.get("by"), "title": r.get("title"),
                   "threshold_sqm": r.get("threshold_sqm"),
                   "delay_months": r.get("delay_months")}
                  for r in ledger if r.get("kind") == "ordinance"]

    acquirer_steps = sorted({e["step"] for e in events if e.get("role") == "acquirer"})
    no_offer_months = sum(1 for e in events if e.get("role") == "acquirer"
                          for op in e.get("operations", []) or []
                          if op.get("action_type") == "no_offer")

    return {
        "run_dir": os.path.basename(run_dir.rstrip("/\\")),
        "scenario": "field_v4",
        "steps": summary.get("steps"),
        "model": summary.get("model"),
        # --- 配線（ここが崩れていたら結果は解釈できない） ---
        "calls": summary.get("usage", {}).get("calls"),
        "api_errors": summary.get("usage", {}).get("errors"),
        "input_tokens": summary.get("usage", {}).get("input_tokens"),
        "output_tokens": summary.get("usage", {}).get("output_tokens"),
        "parse_fail": parse_fail,
        "truncated": summary.get("truncated_responses"),
        "invalid_actions": summary.get("invalid_actions"),
        "offers_returned_by_acquirer": len(offer_ops),
        "offers_recorded": len(offer_records),
        "offer_rejection_reasons": dict(offer_fail),
        "offers_via": dict(via_counter),
        "responders_called": sum(1 for e in events if e.get("action_type") == "responses"),
        "responses": dict(decisions),
        "response_outcomes": dict(decision_results),
        "accept_decisions": accept_decisions,
        "accept_immediate_transfer": accept_immediate,
        "accept_filing_required": accept_filing,
        "accept_rejected_outcome": accept_failed,
        "accept_reconciles": accept_decisions == (accept_immediate + accept_filing
                                                  + accept_failed),
        "filed_transfer_completed": len(filed_completed),
        "filing_void": kinds.get("filing_void", 0),
        "offer_void": kinds.get("offer_void", 0),
        "no_response_rejected": kinds.get("no_response_rejected", 0),
        "transfers_total": kinds.get("transfer", 0),
        "transfers_valid_acquirer": len(valid_transfers),
        "fee_duplicates": fee_duplicates,
        "fee_total_matches_rate": fee_expected == sum(r.get("amount", 0) for r in fees),
        "broker_fee_count": len(fees),
        "broker_fee_total": sum(r.get("amount", 0) for r in fees),
        "deliveries_by_kind": dict(collections.Counter(d.get("kind", "")
                                                       for d in deliveries)),
        # --- 世界で起きたこと（設計の指標） ---
        "months_with_offers": len(offer_steps),
        "first_offer_step": offer_steps[0] if offer_steps else None,
        "months_with_transfers": len({r["step"] for r in valid_transfers}),
        "first_transfer_step": (min(r["step"] for r in valid_transfers)
                                if valid_transfers else None),
        "transfers": len(valid_transfers),
        "no_offer_months": no_offer_months,
        "acquirer_active_months": len(acquirer_steps),
        "ownership_share": summary.get("kpi", {}).get("final_acquirer_share"),
        "area_share": summary.get("kpi", {}).get("final_acquirer_area_share"),
        "under_names_used": dict(collections.Counter(r.get("under_name", "")
                                                     for r in transfers)),
        "financing_raised": sum(r.get("amount", 0) for r in ledger
                                if r.get("kind") == "financing_raised"),
        "feelings": len(feelings),
        "feeling_frames": dict(frames),
        "feelings_about_acquisition": len(about),
        "residents_aware": len(about_people),
        "aware_without_own_offer": len(about_without_own_offer),
        "first_aware_step": min((f["step"] for f in about), default=None),
        "ordinances": ordinances,
        "publications": kinds.get("publication", 0),
        "publications_about_acquisition": sum(
            1 for r in ledger if r.get("kind") == "publication"
            and r.get("about_acquisition")),
        "investigations": kinds.get("investigate", 0),
        # --- 判定（docs/world_design_v4_impl.md の事前定義） ---
        "verdict_acquisition_progresses": bool(
            len({r["step"] for r in valid_transfers}) >= 2
                                               and (summary.get("kpi", {})
                                                    .get("final_acquirer_area_share") or 0) > 0),
        "verdict_residents_notice": bool(len(about_people) >= 2
                                         and len(about_without_own_offer) >= 1),
    }


def metrics(run_dir: str) -> Dict[str, Any]:
    events = _read_jsonl(os.path.join(run_dir, "events.jsonl"))
    ledger = _read_jsonl(os.path.join(run_dir, "ledger.jsonl"))
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)

    acquirer_ids = {e["agent_id"] for e in events if e.get("role") == "acquirer"}
    operations: List[Dict[str, Any]] = []
    for event in events:
        if event.get("role") != "acquirer":
            continue
        for op in event.get("operations", []) or []:
            operations.append({**op, "step": event["step"]})

    verbs = collections.Counter(op.get("action_type", "") for op in operations)
    registry_targets = collections.Counter(
        op.get("target", "") for op in operations
        if op.get("action_type") == "check_land_registry")
    duplicate_registry = sum(max(0, n - 1) for n in registry_targets.values())
    preparation = sum(verbs[v] for v in (
        "internal_review", "market_research", "financing_review",
        "existing_asset_management", "check_land_registry", "due_diligence",
        "contact_broker", "public_statement", "wait"))

    direct = [e for e in events
              if e.get("utterance") and e.get("utterance_channel") == "direct"]
    meeting = [e for e in direct if any(w in e["utterance"] for w in MEETING_WORDS)]
    with_money = [e for e in direct if MONEY_RE.search(e["utterance"])]

    no_such_offer = sum(1 for r in ledger
                        if r.get("kind") == "accept_rejected"
                        and r.get("reason") == "no_such_offer")
    kinds = collections.Counter(r.get("kind", "") for r in ledger)
    first_month = collections.Counter(
        op.get("action_type", "") for op in operations if op["step"] == 1)

    plans = [e.get("plan") for e in events if e.get("plan")]
    repeated = {}
    if plans:
        for key in ("strategy", "next_milestone", "goal_assessment",
                    "expected_goal_effect", "revision_reason"):
            same = sum(1 for a, b in zip(plans, plans[1:])
                       if a.get(key, "") and a.get(key, "") == b.get(key, ""))
            repeated[key] = f"{same}/{max(0, len(plans) - 1)}"

    return {
        "run_dir": os.path.basename(run_dir.rstrip("/\\")),
        "steps": summary.get("steps"),
        "acquirer_model": summary.get("acquirer_model"),
        "calls": summary.get("usage", {}).get("calls"),
        "input_tokens": summary.get("usage", {}).get("input_tokens"),
        "output_tokens": summary.get("usage", {}).get("output_tokens"),
        "api_errors": summary.get("usage", {}).get("errors"),
        "invalid_actions": summary.get("invalid_actions"),
        "truncated": summary.get("truncated_responses"),
        "ownership_share": summary.get("kpi", {}).get("final_acquirer_share"),
        "effective_control_area_share": summary.get("kpi", {}).get(
            "final_effective_control_area_share"),
        "transfers": kinds.get("transfer", 0),
        "lease_controls": kinds.get("lease_control", 0),
        "acquirer_operations": len(operations),
        "operation_verbs": dict(verbs.most_common()),
        "preparation_ratio": (round(preparation / len(operations), 3)
                              if operations else None),
        "first_month_operations": dict(first_month),
        "registry_checks": sum(registry_targets.values()),
        "registry_distinct_parcels": len(registry_targets),
        "registry_duplicate_checks": duplicate_registry,
        "make_offer": verbs.get("make_offer", 0),
        "make_lease_offer": verbs.get("make_lease_offer", 0),
        "direct_messages": len(direct),
        "direct_with_meeting_words": len(meeting),
        "direct_with_money": len(with_money),
        "accept_no_such_offer": no_such_offer,
        "inquiry_request": kinds.get("inquiry_request", 0),
        "inquiry_asked": kinds.get("inquiry_asked", 0),
        "inquiry_answer": kinds.get("inquiry_answer", 0),
        "inquiry_report": kinds.get("inquiry_report", 0),
        "land_registry_check_records": kinds.get("land_registry_check", 0),
        "plan_repeat_vs_previous_month": repeated,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, nargs="+")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    rows = []
    for run in args.run:
        cfg_path = os.path.join(run, "config.yaml")
        version = ""
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                for line in f:
                    if line.startswith("scenario_version:"):
                        version = line.split(":", 1)[1].strip()
                        break
        if version == "field_v4_1":
            rows.append(metrics_v41(run))
        elif version == "field_v4":
            rows.append(metrics_v4(run))
        else:
            rows.append(metrics(run))
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    for row in rows:
        print("=" * 66)
        for key, value in row.items():
            print(f"{key:34s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
