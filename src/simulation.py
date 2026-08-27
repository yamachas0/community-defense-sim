"""シミュレーション本体。

このファイルの責務は3つだけ:
  1. 各エージェントに「今月あなたに見えているもの」を配る
  2. LLM の返した行動を **そのまま** 帳簿に記帳する（解釈・補正・代行をしない）
  3. 記録を残す

やっていないこと（意図的に）:
  - 誰かの行動を条件分岐で決める
  - 「〜なら売る」「〜%の確率で拡散」といったパラメータ
  無効な行動（存在しない区画IDなど）は補正せず invalid として記録し、その月は何も起きない。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from .agents import Agent, build_roster, index_by_id, name_map
from .field_v3 import (action_schema_v3, build_acquirer_decision_prompt_v3,
                       build_system_prompt_v3, build_user_prompt_v3, control_share,
                       effective_control_area_share, ensure_v3_state, list_for_lease,
                       make_lease_offer, normalize_acquirer_plan_v3,
                       resolve_lease_offer, seed_acquirer_intelligence_v3, settle_v3_control, verbs_for_v3)
from .kpi import (classify_publications, classify_utterances, cognition_series,
                  detection_lag, late_index, step_metrics)
from .llm_client_factory import UsageMeter, create_llm_client
from .prompts import build_system_prompt, build_user_prompt
from .schemas import action_schema, verbs_for
from .world import Ledger, assign_tenancies, build_town

logger = logging.getLogger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.S)
_FIELD_RE = re.compile(r'"(\w+)"\s*:\s*(?:"((?:[^"\\]|\\.)*)"|(-?\d+))')


def _repair_truncated_json(raw: str) -> Optional[Dict[str, Any]]:
    """max_tokens で途中で切れた JSON から、拾える限りのフィールドを回収する。

    切り捨ては LLM の落ち度でも我々の解釈でもなく単なる通信の欠落なので、
    復元してよい（行動の中身を推測して補うことは一切しない）。
    """
    out: Dict[str, Any] = {}
    for m in _FIELD_RE.finditer(raw):
        key, sval, ival = m.group(1), m.group(2), m.group(3)
        if ival is not None:
            out[key] = int(ival)
        else:
            try:
                out[key] = json.loads(f'"{sval}"')
            except Exception:
                out[key] = sval
    return out if "action_type" in out else None


def _parse_action(raw: str) -> Optional[Dict[str, Any]]:
    """LLM 応答を dict にする。dict にならないものは全て None（＝PARSE_FAIL）。"""
    if not raw:
        return None
    for candidate in (raw, (_JSON_RE.search(raw).group(0) if _JSON_RE.search(raw) else None)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
        return None  # [] や "文字列" や 数値 が返ってきた場合は行動として成立しない
    repaired = _repair_truncated_json(raw)
    if repaired is not None:
        repaired["_truncated"] = True
    return repaired


# 金額を伴う動詞。amount が欠けたまま 0 で成立させると「賃料0」等の値の捏造になるため、
# これらの動詞では amount フィールドが実際に返ってきていることを必須にする。
_AMOUNT_REQUIRED = {
    "list_for_sale", "counter_offer", "set_rent", "negotiate_rent",
    "make_offer", "redevelop", "list_for_lease", "make_lease_offer",
}
# 対象を伴う動詞。target が空なら成立しない。
_TARGET_REQUIRED = {
    "list_for_sale", "unlist", "accept_offer", "counter_offer", "reject_offer", "set_rent",
    "list_for_lease", "accept_lease_offer", "reject_lease_offer",
    "make_lease_offer", "consult_broker", "contact_broker", "client_followup",
    "property_assessment", "due_diligence", "interdepartmental_contact",
    "interview", "cultivate_source", "request_comment",
    "relocate", "negotiate_rent", "circulate_listing", "approach_owner",
    "make_offer", "withdraw_offer", "redevelop", "request_report",
    "check_land_registry",
}


def _has_amount(act: Dict[str, Any]) -> bool:
    """amount が実際に返ってきているか（欠損を 0 で代行しないための判定）。"""
    if "amount" not in act:
        return False
    v = act.get("amount")
    if v is None or v == "":
        return False
    try:
        return int(float(v)) > 0
    except (TypeError, ValueError):
        return False


def _is_rejected(outcome: Any) -> bool:
    """帳簿上「成立しなかった」結果か。"""
    kind = outcome.get("kind", "") if isinstance(outcome, dict) else str(outcome or "")
    return str(kind).endswith(("_rejected", "invalid_action", "parse_fail"))


def _as_int(v: Any) -> int:
    try:
        return int(float(v))
    except Exception:
        return 0


class Simulation:
    def __init__(self, cfg: Dict[str, Any], personas: Dict[str, Any], run_dir: str):
        self.cfg = cfg
        self.field_v3 = cfg.get("scenario_version") == "field_v3"
        self.personas = personas
        self.run_dir = run_dir
        self.n_steps = int(cfg["steps"])
        self.usage = UsageMeter()
        self.client = create_llm_client({**cfg["llm"], "seed": cfg.get("seed", 42)}, self.usage)
        acquirer_cfg = dict(cfg["llm"])
        acquirer_model = str(acquirer_cfg.get("acquirer_model", "")).strip()
        if self.field_v3 and acquirer_model:
            acquirer_cfg["model"] = acquirer_model
            acquirer_cfg["temperature"] = float(
                acquirer_cfg.get("acquirer_temperature", acquirer_cfg.get("temperature", 0.7)))
            acquirer_cfg["max_tokens"] = int(
                acquirer_cfg.get("acquirer_max_tokens", acquirer_cfg.get("max_tokens", 720)))
            if "acquirer_thinking_budget" in acquirer_cfg:
                acquirer_cfg["thinking_budget"] = int(
                    acquirer_cfg["acquirer_thinking_budget"])
            self.acquirer_client = create_llm_client(
                {**acquirer_cfg, "seed": cfg.get("seed", 42)}, self.usage)
        else:
            self.acquirer_client = self.client
        self.agents: List[Agent] = build_roster(personas, cfg["agents"], cfg["scenario"])
        self.by_id = index_by_id(self.agents)
        self.names = name_map(self.agents)
        self.acquirer_ids = [a.agent_id for a in self.agents if a.role == "acquirer"]
        self.household_ids = [a.agent_id for a in self.agents if a.role == "household"]
        self.business_ids = [a.agent_id for a in self.agents if a.role == "business"]
        self.municipality_id = next(a.agent_id for a in self.agents if a.role == "municipality")

        parcels = build_town(cfg["world"], self.household_ids, self.business_ids,
                             self.municipality_id)
        assign_tenancies(parcels, self.business_ids, int(cfg["world"]["initial_shop_rent"]))
        for p in parcels:
            p.registered_name = self.names.get(p.owner_id, p.owner_id)
        cash: Dict[str, int] = {}
        for a in self.agents:
            cash[a.agent_id] = int(cfg["scenario"]["initial_cash"].get(a.role, 0))
        for a in self.agents:
            if a.role == "acquirer" and "budget" in a.extra:
                cash[a.agent_id] = int(a.extra["budget"])
        self.ledger = Ledger(parcels, cash)
        self.ledger.enable_on_demand_financing([
            a.agent_id for a in self.agents
            if a.role == "acquirer" and a.extra.get("financing") == "on_demand"
        ])
        ensure_v3_state(self.ledger)
        if self.field_v3:
            for agent in (a for a in self.agents if a.role == "acquirer"):
                seed_acquirer_intelligence_v3(agent, self.ledger, cfg["scenario"])

        if self.field_v3:
            self.system_prompts = {a.agent_id: build_system_prompt_v3(a, cfg, len(parcels))
                                   for a in self.agents}
        else:
            self.system_prompts = {
                a.agent_id: build_system_prompt(a, cfg["world"], self.n_steps, len(parcels))
                for a in self.agents
            }
        self.timeline: List[Dict[str, Any]] = []       # 直前stepの公開発話
        self.all_utterances: List[Dict[str, Any]] = []  # 全公開発話（分類対象）
        self.events: List[Dict[str, Any]] = []
        self.kpi_rows: List[Dict[str, Any]] = []
        self.owner_frames: List[Dict[str, Any]] = []
        self.invalid_count = 0
        self.truncated_count = 0
        self.business_margins = {a.agent_id: int(a.extra.get("monthly_margin", 0))
                                 for a in self.agents if a.role == "business"}

    # -- main loop ---------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        t0 = time.time()
        self._snapshot_owners(0)
        for step in range(1, self.n_steps + 1):
            self._step(step)
            row = step_metrics(self.ledger, step, self.acquirer_ids, self.household_ids,
                               self.business_ids,
                               int(self.cfg.get("kpi", {}).get("cascade_window", 3)))
            if self.field_v3:
                row["acquirer_control_share"] = control_share(self.ledger, self.acquirer_ids)
            self.kpi_rows.append(row)
            self._snapshot_owners(step)
            logger.info("step %d/%d share=%.1f%% hhi=%.3f transfers=%d",
                        step, self.n_steps, row["acquirer_share"] * 100, row["hhi"],
                        row["transfers_cum"])
        elapsed = time.time() - t0
        return self._finalize(elapsed)

    def _step(self, step: int) -> None:
        if self.field_v3:
            self._step_v3(step)
            return
        prompts: List[Tuple[Agent, str, str]] = []
        for a in self.agents:
            up = build_user_prompt(a, self.ledger, step, self.n_steps, self.names,
                                   self.timeline, self.acquirer_ids, self.business_margins)
            prompts.append((a, self.system_prompts[a.agent_id], up))

        workers = int(self.cfg["llm"].get("parallel_workers", 8))
        results: Dict[str, Dict[str, Any]] = {}

        def call(item):
            a, sp, up = item
            t = time.time()
            raw = self.client.generate(sp, up, schema=action_schema(a.role),
                                       tag=f"agent:{a.role}")
            return a.agent_id, raw, up, time.time() - t

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for aid, raw, up, dt in ex.map(call, prompts):
                results[aid] = {"raw": raw, "user_prompt": up, "latency": dt}

        # 受信箱は「観測を作り終えた直後」に空にする。以降 _apply や私信で積まれたものは
        # 翌月の観測に載る（ここより後ろでクリアすると _apply が入れた連絡が消える）。
        for a in self.agents:
            a.inbox = []

        # 記帳は agent_id 順（決定論）。先着順の競合は帳簿側で処理される。
        new_timeline: List[Dict[str, Any]] = []
        messages: List[Tuple[str, str, str]] = []   # (from, to, text)
        for a in sorted(self.agents, key=lambda x: x.agent_id):
            r = results.get(a.agent_id, {})
            act = _parse_action(r.get("raw", ""))
            ev: Dict[str, Any] = {
                "step": step, "agent_id": a.agent_id, "role": a.role, "name": a.name,
                "latency_sec": round(r.get("latency", 0.0), 2),
            }
            if act is None:
                self.invalid_count += 1
                ev.update({"action_type": "PARSE_FAIL",
                           "outcome": {"kind": "parse_fail", "reason": "unparseable_response"},
                           "raw": (r.get("raw") or "")[:400]})
                self.events.append(ev)
                continue

            action_type = str(act.get("action_type", "")).strip()
            target = str(act.get("target", "")).strip()
            amount = _as_int(act.get("amount", 0))
            utter = str(act.get("utterance", "")).strip()
            channel = str(act.get("utterance_channel", "none")).strip()
            utter_to = str(act.get("utterance_to", "")).strip()
            memory = str(act.get("memory", "")).strip()
            reasoning = str(act.get("reasoning", "")).strip()
            raw_evidence = act.get("evidence", [])
            evidence = ([str(x).strip() for x in raw_evidence if str(x).strip()][:12]
                        if isinstance(raw_evidence, list) else [])
            under_name = str(act.get("under_name", "")).strip()

            a.memory = memory

            if action_type not in verbs_for(a.role):
                outcome = {"kind": "invalid_action", "reason": "unknown_verb",
                           "given": action_type}
            elif action_type in _AMOUNT_REQUIRED and not _has_amount(act):
                # 途中で切れた等で金額が返ってきていない。0 で代行しない＝不成立。
                outcome = {"kind": "invalid_action", "reason": "missing_amount",
                           "given": action_type}
            elif action_type in _TARGET_REQUIRED and not target:
                outcome = {"kind": "invalid_action", "reason": "missing_target",
                           "given": action_type}
            else:
                outcome = self._apply(step, a, action_type, target, amount, under_name,
                                      utter, messages)
            if _is_rejected(outcome):
                self.invalid_count += 1

            if act.get("_truncated"):
                self.truncated_count += 1
            ev.update({"action_type": action_type, "target": target, "amount": amount,
                       "utterance": utter, "utterance_channel": channel,
                       "utterance_to": utter_to, "memory": memory, "reasoning": reasoning,
                       "evidence": evidence,
                       "under_name": under_name, "truncated": bool(act.get("_truncated")),
                       "outcome": outcome})
            self.events.append(ev)

            if utter and channel == "public":
                row = {"step": step, "from": a.agent_id, "role": a.role, "name": a.name,
                       "text": utter}
                new_timeline.append(row)
                self.all_utterances.append(row)
            elif utter and channel == "private" and utter_to in self.by_id:
                messages.append((a.agent_id, utter_to, utter))

        for src, dst, text in messages:
            self.by_id[dst].inbox.append({"from": src, "text": text, "step": step})
        self.timeline = new_timeline

        # 月次清算（契約の履行＝会計処理。誰も「払うか」を選ばない）
        self.ledger.settle_month(step, self.business_margins)

    # -- 記帳 (行動 -> 帳簿) ------------------------------------------------

    def _step_v3(self, step: int) -> None:
        prompts: List[Tuple[Agent, str, str]] = []
        for a in self.agents:
            up = build_user_prompt_v3(a, self.ledger, step, self.n_steps,
                                      self.names, self.cfg)
            if a.role == "acquirer":
                up = build_acquirer_decision_prompt_v3(a, up)
            prompts.append((a, self.system_prompts[a.agent_id], up))

        workers = int(self.cfg["llm"].get("parallel_workers", 8))
        results: Dict[str, Dict[str, Any]] = {}

        def call(item):
            a, sp, up = item
            client = self.acquirer_client if a.role == "acquirer" else self.client
            started = time.time()
            raw_action = client.generate(sp, up, schema=action_schema_v3(a),
                                         tag=f"agent:{a.role}")
            return a.agent_id, {
                "raw": raw_action,
                "user_prompt": up,
                "latency": time.time() - started,
            }

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for aid, result in ex.map(call, prompts):
                results[aid] = result
        # 今月の観測は作成済み。ここから届く情報は翌月にだけ見える。
        for a in self.agents:
            a.inbox = []

        messages: List[Dict[str, Any]] = []
        presences: Dict[str, str] = {}
        ambient_rows: List[Dict[str, Any]] = []
        venue_ids = {v["id"] for v in self.cfg.get("social", {}).get("venues", [])}

        for a in sorted(self.agents, key=lambda x: x.agent_id):
            result = results.get(a.agent_id, {})
            act = _parse_action(result.get("raw", ""))
            event: Dict[str, Any] = {
                "step": step, "agent_id": a.agent_id, "role": a.role, "name": a.name,
                "latency_sec": round(result.get("latency", 0.0), 2),
            }
            if act is None:
                self.invalid_count += 1
                event.update({
                    "action_type": "PARSE_FAIL",
                    "outcome": {"kind": "parse_fail", "reason": "unparseable_response"},
                    "raw": (result.get("raw") or "")[:400],
                })
                self.events.append(event)
                continue

            if a.role == "acquirer":
                event = self._process_acquirer_portfolio(
                    step, a, act, result, venue_ids, messages, presences, ambient_rows)
                self.events.append(event)
                continue

            action = str(act.get("action_type", "")).strip()
            target = str(act.get("target", "")).strip()
            amount = _as_int(act.get("amount", 0))
            location = str(act.get("location", "")).strip()
            utterance = str(act.get("utterance", "")).strip()
            channel = str(act.get("utterance_channel", "none")).strip()
            utterance_to = str(act.get("utterance_to", "")).strip()
            memory = str(act.get("memory", "")).strip()
            reasoning = str(act.get("reasoning", "")).strip()
            raw_evidence = act.get("evidence", [])
            evidence = ([str(x).strip() for x in raw_evidence if str(x).strip()][:12]
                        if isinstance(raw_evidence, list) else [])
            under_name = str(act.get("under_name", "")).strip()
            plan = (normalize_acquirer_plan_v3(act) if a.role == "acquirer" else {})
            if a.role == "acquirer":
                a.extra["strategy_state"] = plan
            a.memory = memory

            valid_location = location in venue_ids or location in ("HOME", "OFFICE")
            place = f"{location}:{a.agent_id}" if location in ("HOME", "OFFICE") else location
            if action not in verbs_for_v3(a.role):
                outcome = {"kind": "invalid_action", "reason": "unknown_verb",
                           "given": action}
            elif not valid_location:
                outcome = {"kind": "invalid_action", "reason": "unknown_location",
                           "given": location}
            elif action in _AMOUNT_REQUIRED and not _has_amount(act):
                outcome = {"kind": "invalid_action", "reason": "missing_amount",
                           "given": action}
            elif action in _TARGET_REQUIRED and not target:
                outcome = {"kind": "invalid_action", "reason": "missing_target",
                           "given": action}
            else:
                outcome = self._apply_v3(step, a, action, target, amount,
                                         under_name, utterance)
            if _is_rejected(outcome):
                self.invalid_count += 1
            if act.get("_truncated"):
                self.truncated_count += 1

            event.update({
                "action_type": action, "target": target, "amount": amount,
                "location": location, "utterance": utterance,
                "utterance_channel": channel, "utterance_to": utterance_to,
                "memory": memory, "reasoning": reasoning, "evidence": evidence,
                "under_name": under_name, "truncated": bool(act.get("_truncated")),
                "outcome": outcome, "plan": plan,
            })
            self.events.append(event)

            if valid_location:
                presences[a.agent_id] = place
            if utterance and channel == "ambient" and valid_location:
                row = {"step": step, "from": a.agent_id, "role": a.role,
                       "name": a.name, "location": place, "text": utterance}
                ambient_rows.append(row)
                self.all_utterances.append(row)
            elif utterance and channel == "direct":
                destination = self._agent_id(utterance_to)
                if destination:
                    messages.append({
                        "from": a.agent_id, "to": destination, "text": utterance,
                        "step": step,
                        "obs_id": f"MSG-M{step:02d}-{a.agent_id}-{destination}",
                    })

        # 発話内容ではなく、当月の実在する同席だけで配送する。
        for row in ambient_rows:
            for destination, place in presences.items():
                if destination != row["from"] and place == row["location"]:
                    messages.append({
                        "from": row["from"], "to": destination, "text": row["text"],
                        "step": step, "location": row["location"],
                        "obs_id": f"TALK-{row['location']}-M{step:02d}-{row['from']}",
                    })
        for message in messages:
            destination = message["to"]
            self.by_id[destination].inbox.append({k: v for k, v in message.items()
                                                  if k != "to"})

        self.timeline = []
        self.ledger.settle_month(step, self.business_margins)
        settle_v3_control(self.ledger, step)

        for a in self.agents:
            if a.role != "acquirer":
                continue
            event = next((e for e in reversed(self.events)
                          if e.get("step") == step and e.get("agent_id") == a.agent_id), None)
            if event is None:
                continue
            effective_area = sum(
                p.area_sqm for p in self.ledger.parcels.values()
                if p.use != "public" and (
                    p.owner_id == a.agent_id
                    or getattr(p, "controller_id", None) == a.agent_id
                )
            )
            outcome = event.get("outcome", {})
            outcome_kind = outcome.get("kind", "") if isinstance(outcome, dict) else str(outcome)
            operations = event.get("operations", [])
            operation_summary = " ; ".join(
                f"{op.get('action_type')}:{op.get('target') or '-'}=>"
                f"{op.get('outcome', {}).get('kind', '-')}"
                for op in operations)
            history = a.extra.setdefault("execution_history", [])
            previous_area = history[-1]["effective_area"] if history else 0
            history.append({
                "step": step,
                "action_type": event.get("action_type", ""),
                "target": event.get("target", ""),
                "amount": event.get("amount", 0),
                "outcome_kind": outcome_kind,
                "operations": operation_summary,
                "effective_area": effective_area,
                "control_delta": effective_area - previous_area,
                "cash": self.ledger.cash.get(a.agent_id, 0),
                "financing_raised": self.ledger.financing_raised.get(a.agent_id, 0),
            })

    def _process_acquirer_portfolio(
            self, step: int, a: Agent, act: Dict[str, Any], result: Dict[str, Any],
            venue_ids: set, messages: List[Dict[str, Any]],
            presences: Dict[str, str], ambient_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        plan = normalize_acquirer_plan_v3(act)
        a.extra["strategy_state"] = plan
        memory = str(act.get("memory", "")).strip()
        reasoning = str(act.get("reasoning", "")).strip()
        location = str(act.get("location", "")).strip()
        utterance = str(act.get("utterance", "")).strip()
        channel = str(act.get("utterance_channel", "none")).strip()
        utterance_to = str(act.get("utterance_to", "")).strip()
        raw_evidence = act.get("evidence", [])
        evidence = ([str(x).strip() for x in raw_evidence if str(x).strip()][:12]
                    if isinstance(raw_evidence, list) else [])
        a.memory = memory

        valid_location = location in venue_ids or location in ("HOME", "OFFICE")
        place = f"{location}:{a.agent_id}" if location in ("HOME", "OFFICE") else location
        capacity = int(a.extra.get("monthly_operation_capacity", 6))
        raw_operations = act.get("operations", [])
        if not isinstance(raw_operations, list):
            raw_operations = []
        operation_rows: List[Dict[str, Any]] = []
        contacts = {
            "consult_broker", "contact_broker", "client_followup",
            "circulate_listing", "approach_owner", "interdepartmental_contact",
            "interview", "cultivate_source", "request_comment", "request_report",
        }

        if not raw_operations:
            self.invalid_count += 1
            operation_rows.append({
                "action_type": "",
                "target": "",
                "amount": 0,
                "under_name": "",
                "note": "",
                "evidence": [],
                "outcome": {"kind": "invalid_action", "reason": "missing_operations"},
            })

        for index, raw_operation in enumerate(raw_operations):
            if not isinstance(raw_operation, dict):
                row = {
                    "action_type": "", "target": "", "amount": 0,
                    "under_name": "", "note": "", "evidence": [],
                    "outcome": {"kind": "invalid_action", "reason": "operation_not_object"},
                }
                self.invalid_count += 1
                operation_rows.append(row)
                continue
            action = str(raw_operation.get("action_type", "")).strip()
            target = str(raw_operation.get("target", "")).strip()
            amount = _as_int(raw_operation.get("amount", 0))
            under_name = str(raw_operation.get("under_name", "")).strip()
            note = str(raw_operation.get("note", "")).strip()
            raw_op_evidence = raw_operation.get("evidence", [])
            op_evidence = (
                [str(x).strip() for x in raw_op_evidence if str(x).strip()][:12]
                if isinstance(raw_op_evidence, list) else [])

            if index >= capacity:
                outcome = {
                    "kind": "invalid_action", "reason": "monthly_capacity_exceeded",
                    "capacity": capacity,
                }
            elif action not in verbs_for_v3(a.role):
                outcome = {"kind": "invalid_action", "reason": "unknown_verb",
                           "given": action}
            elif not valid_location:
                outcome = {"kind": "invalid_action", "reason": "unknown_location",
                           "given": location}
            elif action in _AMOUNT_REQUIRED and not _has_amount(raw_operation):
                outcome = {"kind": "invalid_action", "reason": "missing_amount",
                           "given": action}
            elif action in _TARGET_REQUIRED and not target:
                outcome = {"kind": "invalid_action", "reason": "missing_target",
                           "given": action}
            else:
                outcome = self._apply_v3(
                    step, a, action, target, amount, under_name, note)
            if _is_rejected(outcome):
                self.invalid_count += 1
            elif action in contacts and note:
                destination = self._agent_id(target)
                if destination:
                    messages.append({
                        "from": a.agent_id, "to": destination, "text": note,
                        "step": step,
                        "obs_id": f"MSG-M{step:02d}-{a.agent_id}-{destination}-O{index + 1}",
                    })
            operation_rows.append({
                "action_type": action,
                "target": target,
                "amount": amount,
                "under_name": under_name,
                "note": note,
                "evidence": op_evidence,
                "outcome": outcome,
            })

        if act.get("_truncated"):
            self.truncated_count += 1
        if valid_location:
            presences[a.agent_id] = place
        if utterance and channel == "ambient" and valid_location:
            row = {
                "step": step, "from": a.agent_id, "role": a.role,
                "name": a.name, "location": place, "text": utterance,
            }
            ambient_rows.append(row)
            self.all_utterances.append(row)
        elif utterance and channel == "direct":
            destination = self._agent_id(utterance_to)
            if destination:
                messages.append({
                    "from": a.agent_id, "to": destination, "text": utterance,
                    "step": step,
                    "obs_id": f"MSG-M{step:02d}-{a.agent_id}-{destination}",
                })

        rejected = sum(1 for op in operation_rows if _is_rejected(op["outcome"]))
        return {
            "step": step,
            "agent_id": a.agent_id,
            "role": a.role,
            "name": a.name,
            "latency_sec": round(result.get("latency", 0.0), 2),
            "action_type": "operation_portfolio",
            "target": "",
            "amount": 0,
            "location": location,
            "utterance": utterance,
            "utterance_channel": channel,
            "utterance_to": utterance_to,
            "memory": memory,
            "reasoning": reasoning,
            "evidence": evidence,
            "under_name": "",
            "truncated": bool(act.get("_truncated")),
            "operations": operation_rows,
            "outcome": {
                "kind": "operation_portfolio",
                "selected": len(raw_operations),
                "completed": len(operation_rows) - rejected,
                "rejected": rejected,
                "capacity": capacity,
            },
            "plan": plan,
        }

    def _agent_id(self, value: str) -> Optional[str]:
        if value in self.by_id:
            return value
        return next((a.agent_id for a in self.agents if a.name == value), None)

    def _apply_v3(self, step: int, a: Agent, action: str, target: str, amount: int,
                  under_name: str, utterance: str) -> Dict[str, Any]:
        ledger = self.ledger
        if action == "list_for_lease":
            return list_for_lease(ledger, step, target, a.agent_id, amount)
        if action == "make_lease_offer":
            return make_lease_offer(ledger, step, target, a, amount, under_name, utterance)
        if action in ("accept_lease_offer", "reject_lease_offer"):
            return resolve_lease_offer(ledger, step, target, a.agent_id,
                                       action == "accept_lease_offer")
        if action == "make_offer":
            allowed = [a.name] + list(a.extra.get("aliases", []))
            if under_name not in allowed:
                return {"kind": "invalid_action", "reason": "unknown_legal_name",
                        "given": under_name}
            return ledger.record_offer(step, target, a.agent_id, amount,
                                       under_name=under_name, note=utterance[:120])
        if action == "check_land_registry":
            if target not in ledger.parcels:
                return {"kind": "invalid_action", "reason": "no_such_parcel",
                        "target": target}
            targets = a.extra.setdefault("land_registry_targets", [])
            if target not in targets:
                targets.append(target)
            return ledger.record_note(step, a.agent_id, action, f"parcel={target}")
        if action == "check_corporate_registry":
            a.extra["corporate_registry_seen"] = True
            return ledger.record_note(step, a.agent_id, action, "公開法人記録を閲覧")
        if action == "market_research":
            a.extra["market_research_seen"] = True
            return ledger.record_note(step, a.agent_id, action, "地区別公開市場情報を閲覧")
        if action == "publish":
            headline = (utterance.splitlines()[0] if utterance else "（無題）")[:80]
            outcome = ledger.record_publication(step, a.agent_id, headline,
                                                utterance[:400],
                                                self._is_about_acquisition(utterance))
            article_id = f"NEWS-{a.agent_id}-M{step:02d}"
            for recipient in self.agents:
                subscriptions = recipient.extra.get("subscriptions", [])
                if a.name in subscriptions or a.agent_id in subscriptions:
                    recipient.inbox.append({"from": a.agent_id, "text": utterance,
                                            "step": step, "obs_id": article_id})
            return outcome

        if action in ("property_assessment", "due_diligence"):
            if target not in ledger.parcels:
                return {"kind": "invalid_action", "reason": "no_such_parcel",
                        "target": target}
            return ledger.record_note(step, a.agent_id, action, f"parcel={target}")

        contacts = {
            "consult_broker", "contact_broker", "client_followup",
            "circulate_listing", "approach_owner", "interdepartmental_contact",
            "interview", "cultivate_source", "request_comment", "request_report",
        }
        if action in contacts:
            destination = self._agent_id(target)
            if not destination:
                return {"kind": "invalid_action", "reason": "no_such_agent",
                        "target": target}
            return ledger.record_note(step, a.agent_id, action, f"to={destination}")

        if action == "list_for_sale":
            return ledger.record_listing(step, target, a.agent_id, amount)
        if action == "unlist":
            return ledger.record_unlist(step, target, a.agent_id)
        if action == "accept_offer":
            return ledger.record_accept(step, target, a.agent_id)
        if action == "reject_offer":
            return ledger.record_reject(step, target, a.agent_id)
        if action == "counter_offer":
            return ledger.record_counter(step, target, a.agent_id, amount)
        if action == "withdraw_offer":
            return ledger.record_withdraw(step, target, a.agent_id)
        if action == "move_out":
            return ledger.record_move_out(step, a.agent_id, utterance[:120])
        if action == "close_shop":
            return ledger.record_close(step, a.agent_id, utterance[:120])
        if action == "relocate":
            return ledger.record_relocate(step, a.agent_id, target)
        if action == "negotiate_rent":
            return ledger.record_note(step, a.agent_id, action,
                                      f"to={target} ask={amount}")
        if action == "study_ordinance":
            return ledger.record_study(step, a.agent_id, utterance[:160])
        if action == "enact_ordinance":
            return ledger.record_ordinance(step, a.agent_id, target, utterance[:400])
        # 残りは日常・調査・待機。選ばれた事実だけを記録する。
        return ledger.record_note(step, a.agent_id, action, utterance[:140])


    def _apply(self, step: int, a: Agent, action: str, target: str, amount: int,
               under_name: str, utter: str,
               messages: List[Tuple[str, str, str]]) -> Dict[str, Any]:
        L = self.ledger
        # household -------------------------------------------------------
        if action == "list_for_sale":
            return L.record_listing(step, target, a.agent_id, amount)
        if action == "unlist":
            return L.record_unlist(step, target, a.agent_id)
        if action == "accept_offer":
            return L.record_accept(step, target, a.agent_id)
        if action == "reject_offer":
            return L.record_reject(step, target, a.agent_id)
        if action == "counter_offer":
            return L.record_counter(step, target, a.agent_id, amount)
        if action == "set_rent":
            return L.record_rent_change(step, target, a.agent_id, amount)
        if action == "move_out":
            return L.record_move_out(step, a.agent_id, utter[:120])
        # business --------------------------------------------------------
        if action == "close_shop":
            return L.record_close(step, a.agent_id, utter[:120])
        if action == "relocate":
            return L.record_relocate(step, a.agent_id, target)
        if action == "negotiate_rent":
            if target not in self.by_id:
                return {"kind": "invalid_action", "reason": "no_such_agent", "target": target}
            if not L._valid_money(amount):
                return {"kind": "invalid_action", "reason": "invalid_amount", "given": amount}
            landlord_of = {p.owner_id for p in L.parcels.values()
                           if p.tenant_id == a.agent_id and p.use == "shop"}
            if target not in landlord_of:
                return {"kind": "invalid_action", "reason": "not_my_landlord",
                        "target": target}
            messages.append((a.agent_id, target,
                             f"（賃料交渉）希望月額 {amount}万円。{utter[:100]}"))
            return L.record_note(step, a.agent_id, "rent_negotiation",
                                 f"to={target} ask={amount}")
        if action in ("continue", "hold", "wait", "monitor", "silent"):
            return L.record_note(step, a.agent_id, "no_ledger_change", action)
        # broker ----------------------------------------------------------
        if action in ("circulate_listing", "approach_owner"):
            if target in self.by_id:
                return L.record_note(step, a.agent_id, action, f"to={target}")
            return {"kind": "invalid_action", "reason": "no_such_agent", "target": target}
        # acquirer --------------------------------------------------------
        if action == "make_offer":
            return L.record_offer(step, target, a.agent_id, amount,
                                  under_name=under_name or a.name, note=utter[:120])
        if action == "withdraw_offer":
            return L.record_withdraw(step, target, a.agent_id)
        if action == "redevelop":
            return L.record_redevelop(step, target, a.agent_id, amount)
        if action == "public_statement":
            return L.record_note(step, a.agent_id, "public_statement", utter[:160])
        # municipality ----------------------------------------------------
        if action == "study_ordinance":
            return L.record_study(step, a.agent_id, utter[:160])
        if action == "enact_ordinance":
            return L.record_ordinance(step, a.agent_id, target or "土地取引規制", utter[:400])
        if action == "request_report":
            if target not in self.by_id:
                return {"kind": "invalid_action", "reason": "no_such_agent", "target": target}
            messages.append((a.agent_id, target, f"（自治体からの照会）{utter[:140]}"))
            return L.record_note(step, a.agent_id, "request_report", f"to={target}")
        # media -----------------------------------------------------------
        if action == "investigate":
            a.extra["investigated"] = True
            return L.record_note(step, a.agent_id, "investigate", "登記の取材に着手")
        if action == "publish":
            head = (utter.splitlines()[0] if utter else "（無題）")[:80]
            about = self._is_about_acquisition(utter)
            return L.record_publication(step, a.agent_id, head, utter[:400], about)
        return {"kind": "invalid_action", "reason": "unhandled_verb", "given": action}

    @staticmethod
    def _is_about_acquisition(text: str) -> bool:
        """報道が土地取得を扱っているかの判定。

        NOTE: これはエージェントの行動を決めるものではなく、記事の分類（観測）である。
        取りこぼしは終了後の LLM 分類 (kpi.classify_utterances) 側で補正できる。
        """
        keys = ("取得", "買収", "所有", "登記", "地権", "名義", "土地", "売買", "地価", "賃料")
        return any(k in text for k in keys)

    # -- 記録 --------------------------------------------------------------

    def _snapshot_owners(self, step: int) -> None:
        self.owner_frames.append({
            "step": step,
            "owner": {pid: p.owner_id for pid, p in self.ledger.parcels.items()},
            "registered": {pid: p.registered_name for pid, p in self.ledger.parcels.items()},
            "use": {pid: p.use for pid, p in self.ledger.parcels.items()},
            "rent": {pid: p.rent for pid, p in self.ledger.parcels.items()},
            "tenant": {pid: (p.tenant_id or "") for pid, p in self.ledger.parcels.items()},
            "controller": {pid: (getattr(p, "controller_id", None) or "") for pid, p in self.ledger.parcels.items()},
            "controller_name": {pid: (getattr(p, "controller_name", "") or "") for pid, p in self.ledger.parcels.items()},
            "control_rent": {pid: int(getattr(p, "control_rent", 0) or 0) for pid, p in self.ledger.parcels.items()},
        })

    def _finalize(self, elapsed: float) -> Dict[str, Any]:
        share_by_step = {0: 0.0}
        share_by_step.update({r["step"]: r["acquirer_share"] for r in self.kpi_rows})

        classified: List[Dict[str, Any]] = []
        kcfg = self.cfg.get("kpi", {})
        targets = [u for u in self.all_utterances
                   if u["role"] in ("household", "business")]
        pub_steps = None
        if kcfg.get("classify_utterances", True) and targets:
            classified = classify_utterances(
                self.client, targets, batch=int(kcfg.get("classify_batch", 25)))
        if kcfg.get("classify_utterances", True) and self.ledger.publications:
            pub_steps = classify_publications(
                self.client, self.ledger.publications,
                batch=int(kcfg.get("classify_batch", 25)))
        cog = cognition_series(classified, self.n_steps)

        summary = {
            "run_name": self.cfg.get("run_name"),
            "steps": self.n_steps,
            "agents": {r: sum(1 for a in self.agents if a.role == r)
                       for r in sorted({a.role for a in self.agents})},
            "parcels": len(self.ledger.parcels),
            "model": getattr(self.client, "model", "?"),
            "provider": self.cfg["llm"].get("provider"),
            "elapsed_sec": round(elapsed, 1),
            "acquirer_model": getattr(self.acquirer_client, "model", "?"),
            "invalid_actions": self.invalid_count,
            "truncated_responses": self.truncated_count,
            "usage": self.usage.as_dict(),
            "kpi": {
                "final_acquirer_share": self.kpi_rows[-1]["acquirer_share"] if self.kpi_rows else 0,
                "final_hhi": self.kpi_rows[-1]["hhi"] if self.kpi_rows else 0,
                "final_acquirer_control_share": control_share(self.ledger, self.acquirer_ids)
                if self.field_v3 else None,
                "cognition_shift_final": cog[-1]["shift_rate_cum"] if cog else None,
                "cascade": {"induced": self.kpi_rows[-1]["cascade_induced"],
                            "max_chain": self.kpi_rows[-1]["cascade_max_chain"]}
                if self.kpi_rows else {},
                "final_effective_control_area_share": effective_control_area_share(
                    self.ledger, self.acquirer_ids) if self.field_v3 else None,
                "late_index": late_index(self.ledger, self.acquirer_ids, share_by_step),
                "detection_lag": detection_lag(self.ledger, share_by_step, pub_steps),
                "business_survival": self.kpi_rows[-1]["business_survival"] if self.kpi_rows else None,
                "resident_outflow": self.kpi_rows[-1]["resident_outflow"] if self.kpi_rows else None,
            },
        }

        d = self.run_dir
        _write_jsonl(os.path.join(d, "events.jsonl"), self.events)
        _write_jsonl(os.path.join(d, "kpi.jsonl"), self.kpi_rows)
        _write_jsonl(os.path.join(d, "cognition.jsonl"), cog)
        _write_jsonl(os.path.join(d, "utterances.jsonl"), classified or self.all_utterances)
        self.ledger.dump_records(os.path.join(d, "ledger.jsonl"))
        with open(os.path.join(d, "owner_frames.json"), "w", encoding="utf-8") as f:
            json.dump(self.owner_frames, f, ensure_ascii=False)
        with open(os.path.join(d, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        self.summary = summary
        self.cognition = cog
        self.classified = classified
        return summary


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
