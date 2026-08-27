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


# --- v5: 出来事はピン留め・観測するのは街の会話 ---------------------------
# 判定と抽出の定義は docs/world_design_v5_impl.md 6章で**走行前に固定**したもの。
# ここでは解釈も評価もしない。数えるのは記録に実在する事実だけ。

S4_KINDS = ["assembly", "counter", "broker_front", "press"]
V5_HOLDER_RE = re.compile(r"[A-Z]社")
V5_PARCEL_RE = re.compile(r"P\d{2}")
V5_WORDS = ("名義", "所有者が変わ", "売られ", "手放し", "買い取", "買われ",
            "地上げ", "底地", "よそ者", "外の会社", "東京の会社")


def _v5_s4_kind(step: int) -> str:
    return S4_KINDS[(int(step) - 1) % len(S4_KINDS)]


def _v5_mentions(text: str, holders) -> Dict[str, Any]:
    """走行前に固定したルールベース1次抽出（絞り込みであって判定ではない）。"""
    parcels = sorted(set(V5_PARCEL_RE.findall(text)))
    named = sorted({h for h in holders if h and h in text})
    words = [w for w in V5_WORDS if w in text]
    return {"parcels": parcels, "holders": named, "words": words,
            "hit": bool(parcels or named or words)}


def metrics_v5(run_dir: str) -> Dict[str, Any]:
    events = _read_jsonl(os.path.join(run_dir, "events.jsonl"))
    ledger = _read_jsonl(os.path.join(run_dir, "ledger.jsonl"))
    utts = _read_jsonl(os.path.join(run_dir, "utterances_v5.jsonl"))
    traces = _read_jsonl(os.path.join(run_dir, "traces_v5.jsonl"))
    stances = _read_jsonl(os.path.join(run_dir, "stances_v5.jsonl"))
    articles = _read_jsonl(os.path.join(run_dir, "articles_v5.jsonl"))
    directs = _read_jsonl(os.path.join(run_dir, "directs_v5.jsonl"))
    plans = _read_jsonl(os.path.join(run_dir, "plans_v5.jsonl"))
    classified = _read_jsonl(os.path.join(run_dir, "utterances.jsonl"))
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)

    # 取得（台本の実現）＝台帳の名義移転
    acqs = [r for r in ledger if r.get("kind") == "transfer"]
    holders = sorted({str(a.get("under_name", "")) for a in acqs if a.get("under_name")})
    by_parcel = {a["parcel_id"]: a for a in acqs}
    sellers = {a["parcel_id"]: a.get("seller", "") for a in acqs}

    # LLM の事後分類（走行中の主体には一切見せていない）
    about = {(int(r.get("step", 0)), r.get("from", ""), str(r.get("text", "")))
             for r in classified if r.get("about_acquisition")}
    llm_ran = any("about_acquisition" in r for r in classified)

    first_mention: Dict[str, Any] = {}
    edges = 0
    mention_rows = []
    knowers_by_month: Dict[int, set] = collections.defaultdict(set)
    form = collections.Counter()
    for u in sorted(utts, key=lambda r: (r["step"], r.get("utt_id", ""))):
        text = str(u.get("text", ""))
        hit = _v5_mentions(text, holders)
        if not hit["hit"]:
            continue
        key = (int(u["step"]), u.get("from", ""), text)
        if llm_ran and key not in about:
            continue          # 1次抽出に掛かっても LLM が「取得の話ではない」としたものは数えない
        mention_rows.append(u)
        edges += len(u.get("heard_by", []))
        knowers_by_month[int(u["step"])].add(u.get("from", ""))
        # 噂の形（台帳と突き合わせた事後判定・LLMの主観を入れない）
        step = int(u["step"])
        if hit["parcels"]:
            for pid in hit["parcels"]:
                rec = by_parcel.get(pid)
                if rec and int(rec.get("step", 0)) <= step:
                    ok = (not hit["holders"]
                          or rec.get("under_name") in hit["holders"])
                    form["fact" if ok else "error"] += 1
                else:
                    form["error"] += 1
        else:
            form["unspecific"] += 1
        # 初認知（当事者＝売主を除く）
        for pid in (hit["parcels"] or [a["parcel_id"] for a in acqs
                                       if hit["holders"]
                                       and a.get("under_name") in hit["holders"]
                                       and int(a["step"]) <= step]):
            rec = by_parcel.get(pid)
            if rec is None or int(rec.get("step", 0)) > step:
                continue
            acq_id = rec.get("acq_id") or pid
            if acq_id in first_mention or u.get("from") == sellers.get(pid):
                continue
            seen = [t for t in traces
                    if t.get("agent_id") == u.get("from")
                    and t.get("parcel_id") == pid and int(t.get("step", 0)) <= step]
            first_mention[acq_id] = {
                "parcel_id": pid, "transfer_month": int(rec.get("step", 0)),
                "month": step, "agent_id": u.get("from"), "role": u.get("role"),
                "scene": u.get("scene"), "venue": u.get("venue"),
                "lag_months": step - int(rec.get("step", 0)),
                "trace_seen_first": (seen[0].get("kind") if seen else ""),
            }

    noticed = set(first_mention)
    all_ids = [a.get("acq_id") or a["parcel_id"] for a in acqs]
    unnoticed = [a for a in all_ids if a not in noticed]

    # 明るみに出たか
    article_months = sorted({int(a["step"]) for a in articles
                             if _v5_mentions(str(a.get("text", "")), holders)["hit"]})
    assembly, counter = collections.defaultdict(set), []
    for u in mention_rows:
        if u.get("scene") != "S4":
            continue
        kind = _v5_s4_kind(u["step"])
        if kind == "assembly":
            assembly[int(u["step"])].add(u.get("from"))
        elif kind == "counter":
            counter.append(int(u["step"]))
    assembly_months = sorted(m for m, who in assembly.items() if len(who) >= 3)
    counter_months = sorted(set(counter))

    # 元所有者の沈黙・発言
    seller_spoke = {}
    for pid, seller in sellers.items():
        rows = [u for u in mention_rows
                if u.get("from") == seller
                and (pid in str(u.get("text", ""))
                     or by_parcel[pid].get("under_name") in str(u.get("text", "")))]
        seller_spoke[f"{seller}/{pid}"] = (min(int(r["step"]) for r in rows)
                                           if rows else None)

    # D1: 当事者以外の3主体以上が言及した取得
    by_acq_speakers: Dict[str, set] = collections.defaultdict(set)
    for u in mention_rows:
        text = str(u.get("text", ""))
        for pid in V5_PARCEL_RE.findall(text):
            rec = by_parcel.get(pid)
            if rec is None or int(rec.get("step", 0)) > int(u["step"]):
                continue
            if u.get("from") == sellers.get(pid):
                continue
            by_acq_speakers[rec.get("acq_id") or pid].add(u.get("from"))
    d1 = sorted(k for k, who in by_acq_speakers.items() if len(who) >= 3)
    d2 = bool(article_months or assembly_months or counter_months)

    scene_sizes = collections.Counter()
    for u in utts:
        scene_sizes[(u["step"], u["scene"], u["venue"])] = len(u.get("heard_by", [])) + 1
    parse_fail = len([e for e in events if e.get("action_type") == "PARSE_FAIL"])
    talk_events = [e for e in events if e.get("action_type") == "utterance"]
    usage = summary.get("usage", {})

    return {
        "run": os.path.basename(run_dir),
        "version": "field_v5",
        "steps": summary.get("steps"),
        "model": summary.get("model"),
        "calls": usage.get("calls", 0),
        "errors": usage.get("errors", 0),
        "truncated": summary.get("truncated_responses", 0),
        "parse_fail": parse_fail,
        "invalid_actions": summary.get("invalid_actions", 0),
        "llm_classified": llm_ran,
        # 出来事（台本）
        "acquisitions": len(acqs),
        "acquisition_months": sorted(int(a["step"]) for a in acqs),
        "holders": holders,
        "final_acquirer_share": summary.get("kpi", {}).get("final_acquirer_share"),
        # 会話
        "utterance_turns": len(talk_events),
        "utterances_spoken": len(utts),
        "silence_rate": (round(1 - len(utts) / len(talk_events), 3)
                         if talk_events else None),
        "mean_scene_size": (round(sum(scene_sizes.values()) / len(scene_sizes), 2)
                            if scene_sizes else 0),
        "directs": len(directs),
        "articles": len(articles),
        # 気づき
        "mention_utterances": len(mention_rows),
        "propagation_edges": edges,
        "knowers_by_month": {m: sorted(v) for m, v in sorted(knowers_by_month.items())},
        "first_mention": first_mention,
        "noticed_acquisitions": len(noticed),
        "unnoticed_acquisitions": unnoticed,
        "unnoticed_ratio": (round(len(unnoticed) / len(all_ids), 3) if all_ids else None),
        "mean_detection_lag_months": (
            round(sum(v["lag_months"] for v in first_mention.values())
                  / len(first_mention), 2) if first_mention else None),
        "rumor_form": dict(form),
        # 明るみ
        "article_months": article_months,
        "assembly_months": assembly_months,
        "counter_months": counter_months,
        "registry_lookups": len([t for t in traces if t.get("kind") == "registry_lookup"]),
        "traces_by_kind": dict(collections.Counter(t.get("kind") for t in traces)),
        "seller_first_mention_of_own_sale": seller_spoke,
        "sellers_silent": len([v for v in seller_spoke.values() if v is None]),
        # 姿勢（台帳を動かさない観測値）
        "stance_sell": len([s for s in stances if s.get("stance") == "sell"]),
        "stance_keep": len([s for s in stances if s.get("stance") == "keep"]),
        "plans_home_rate": (round(sum(1 for p in plans for s in ("S1", "S2", "S3", "S4")
                                      if p.get(s) == "HOME")
                                  / max(1, len(plans) * 4), 3) if plans else None),
        # 判定（走行前に固定した定義）
        "D1_exposed_acquisitions": d1,
        "D2_town_exposed": d2,
        "D3_wiring_ok": (parse_fail == 0 and summary.get("truncated_responses", 0) == 0
                         and bool(utts)),
    }


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

    # --- 既に世界にあった相談経路（direct 私信）の量 ---
    #     v4.1b の「専用の相談プロトコル」と比べるための基準値。
    roles = {e["agent_id"]: e.get("role", "") for e in events}
    owner_to_broker = [d for d in deliveries if d.get("kind") == "direct"
                       and roles.get(d.get("from", "")) in ("household", "business")
                       and roles.get(d.get("to", "")) == "broker"]
    broker_to_owner = [d for d in deliveries if d.get("kind") == "direct"
                       and roles.get(d.get("from", "")) == "broker"
                       and roles.get(d.get("to", "")) in ("household", "business")]

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
        "owner_to_broker_direct": len(owner_to_broker),
        "broker_to_owner_direct": len(broker_to_owner),
        "owners_using_direct_to_broker": len({d.get("from") for d in owner_to_broker}),
    }


def metrics_v41b(run_dir: str) -> Dict[str, Any]:
    """v4.1b（相談経路を戻した版）の集計。v4.1 の全指標に相談・回答・条例実績を足す。"""
    base = metrics_v41(run_dir)
    base["scenario"] = "field_v4_1b"
    events = _read_jsonl(os.path.join(run_dir, "events.jsonl"))
    ledger = _read_jsonl(os.path.join(run_dir, "ledger.jsonl"))
    deliveries = _read_jsonl(os.path.join(run_dir, "deliveries.jsonl"))

    consults = [r for r in ledger if r.get("kind") == "consult"]
    advices = [r for r in ledger if r.get("kind") == "advice"]
    consult_rejects = collections.Counter(
        r.get("reason", "") for r in ledger if r.get("kind") == "consult_rejected")
    advice_rejects = collections.Counter(
        r.get("reason", "") for r in ledger if r.get("kind") == "advice_rejected")
    # 台帳に載らない不成立（相手が空・非object・月次上限超過など）は events 側にある。
    for e in events:
        for op in e.get("operations", []) or []:
            outcome = op.get("outcome", {}) or {}
            if str(outcome.get("kind", "")) != "invalid_action":
                continue
            if op.get("action_type") == "consult":
                consult_rejects[str(outcome.get("reason", ""))] += 1
            elif op.get("action_type") == "advise":
                advice_rejects[str(outcome.get("reason", ""))] += 1
    consult_step = {r.get("consult_id"): r.get("step") for r in consults}
    answered_ids = [r.get("consult_id") for r in advices]
    latency = [int(r.get("step", 0)) - int(consult_step.get(r.get("consult_id"), 0))
               for r in advices if r.get("consult_id") in consult_step]

    # 回答が実際に届いた所有者と、その月
    advice_delivered: Dict[str, int] = {}
    for d in deliveries:
        if d.get("kind") != "advice":
            continue
        who = d.get("to", "")
        step = int(d.get("step", 0))
        if who and (who not in advice_delivered or step < advice_delivered[who]):
            advice_delivered[who] = step

    # 助言を読めるのは届いた翌月から。読了前後で owner-month の判断を分けて数える。
    advice_read_step = {who: step + 1 for who, step in advice_delivered.items()}
    decided_after_advice: Dict[str, List[str]] = {}
    before = collections.Counter()
    after = collections.Counter()
    for e in sorted(events, key=lambda x: (x.get("step", 0), x.get("agent_id", ""))):
        if e.get("action_type") != "responses":
            continue
        who = e.get("agent_id", "")
        read = advice_read_step.get(who)
        readable = read is not None and int(e.get("step", 0)) >= read
        for op in e.get("operations", []) or []:
            decision = op.get("action_type", "")
            if decision not in ("sell", "keep", "hold"):
                continue
            (after if readable else before)[decision] += 1
            if readable and decision in ("sell", "keep"):
                decided_after_advice.setdefault(who, []).append(
                    f"M{e.get('step')}:{decision}")

    # 相談の語が所有者の内心に出た回数（v4.1 の 58/70/60 と直接比べるため）
    thoughts_all = _read_jsonl(os.path.join(run_dir, "thoughts_all.jsonl"))
    consult_word = sum(1 for t in thoughts_all
                       if t.get("role") in ("household", "business")
                       and "仲介" in str(t.get("text", "")))

    # 条例が空振りしたか（閾値 > 区画の最大面積）
    areas: List[int] = []
    cfg_path = os.path.join(run_dir, "config.yaml")
    if os.path.exists(cfg_path):
        import yaml
        with open(cfg_path, encoding="utf-8") as f:
            run_cfg = yaml.safe_load(f) or {}
        areas = [int(x) for x in
                 (run_cfg.get("world", {}) or {}).get("area_pattern_sqm", []) or []]
    max_area = max(areas) if areas else None
    ordinances = base.get("ordinances", [])
    base.update({
        "consults_recorded": len(consults),
        "consult_rejection_reasons": dict(consult_rejects),
        "owners_who_consulted": len({r.get("by") for r in consults}),
        "consult_steps": sorted({int(r.get("step", 0)) for r in consults}),
        "advices_recorded": len(advices),
        "advice_rejection_reasons": dict(advice_rejects),
        "consults_answered": len(set(answered_ids)),
        "consult_answer_rate": (round(len(set(answered_ids)) / len(consults), 3)
                                if consults else None),
        "answer_latency_steps": sorted(latency),
        "advices_delivered": sum(1 for d in deliveries if d.get("kind") == "advice"),
        "owners_who_received_advice": len(advice_delivered),
        "terminal_decisions_after_first_advice_read": decided_after_advice,
        "owners_deciding_after_advice_read": len(decided_after_advice),
        "owner_month_decisions_before_advice_read": dict(before),
        "owner_month_decisions_after_advice_read": dict(after),
        "unanswered_consults_at_end": len(consults) - len(set(answered_ids)),
        "advices_unread_at_end": sum(
            1 for d in deliveries if d.get("kind") == "advice"
            and int(d.get("step", 0)) >= int(base.get("steps") or 0)),
        "broker_mentions_in_owner_thoughts": consult_word,
        "max_parcel_area_sqm": max_area,
        "ordinance_thresholds_vs_max_parcel": [
            {"step": o.get("step"), "threshold_sqm": o.get("threshold_sqm"),
             "exceeds_every_parcel": (None if max_area is None
                                      else bool((o.get("threshold_sqm") or 0) >= max_area))}
            for o in ordinances],
        "filings_triggered": sum(1 for r in ledger
                                 if r.get("kind") == "filing_required"),
    })
    return base

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
        if version == "field_v5":
            rows.append(metrics_v5(run))
        elif version == "field_v4_1b":
            rows.append(metrics_v41b(run))
        elif version == "field_v4_1":
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
