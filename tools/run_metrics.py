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
    rows = [metrics(r) for r in args.run]
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
