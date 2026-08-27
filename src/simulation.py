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
from .field_v3 import (action_schema_v3, answer_owner_inquiry,
                       build_acquirer_decision_prompt_v3,
                       build_system_prompt_v3, build_user_prompt_v3,
                       check_land_registry_v3, control_share,
                       effective_control_area_share, ensure_v3_state,
                       inquire_owner_intent, list_for_lease,
                       make_lease_offer, normalize_acquirer_plan_v3, own_result_row,
                       report_owner_intent, request_owner_inquiry,
                       resolve_lease_offer, seed_acquirer_intelligence_v3,
                       settle_v3_control, verbs_for_v3)
from .field_v4 import (own_results_text_v4,  # noqa: F401
                       build_phase1_prompt_v4, build_phase2_prompt_v4,
                       build_system_prompt_v4, enact_ordinance_v4,
                       ensure_v4_state, execute_pending_transfers_v4,
                       owners_with_offers, phase1_schema_v4, phase2_schema_v4,
                       record_offer_v4, respond_to_offer_v4)
from .field_v4_1 import MAX_TEXT_CHARS as MAX_TEXT_CHARS_V41
from .field_v4_1 import (build_phase1_prompt_v41, build_phase2_prompt_v41,
                         build_system_prompt_v41, enact_ordinance_v41,
                         ensure_v41_state, execute_pending_v41,
                         own_result_row_v41, owners_with_offers_v41,
                         phase1_schema_v41, phase2_schema_v41, record_offer_v41,
                         respond_to_offer_v41, withdraw_offer_v41)
from .field_v4_1b import (CONSULT_NONE, DEFAULT_ADVICE_CAPACITY,
                          answer_consult_v41b,
                          build_phase1_prompt_v41b, build_phase2_prompt_v41b,
                          build_system_prompt_v41b, ensure_v41b_state,
                          phase1_schema_v41b, record_consult_v41b)
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
    "check_land_registry", "request_owner_inquiry", "inquire_owner_intent",
    "answer_broker_inquiry", "report_owner_intent",
}


def _structured_fields(source: Dict[str, Any]) -> Dict[str, Any]:
    """区画・照会ID・意向・希望額という、行為が運ぶ構造化された事実を取り出す。"""
    return {
        "parcel_id": str(source.get("parcel_id", "") or "").strip(),
        "inquiry_id": str(source.get("inquiry_id", "") or "").strip(),
        "owner_intent": str(source.get("owner_intent", "") or "").strip(),
        "asking_price": source.get("asking_price", ""),
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
        self.field_v4 = cfg.get("scenario_version") == "field_v4"
        self.field_v41b = cfg.get("scenario_version") == "field_v4_1b"
        # v4.1b は v4.1 の世界に相談経路と行政の面積観測を足しただけ＝土台は v4.1 と同じ。
        self.field_v41 = cfg.get("scenario_version") in ("field_v4_1", "field_v4_1b")
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
        if self.field_v41:
            ensure_v41_state(self.ledger)
            for agent in (a for a in self.agents if a.role == "acquirer"):
                agent.extra["monthly_offer_capacity"] = int(
                    cfg["scenario"].get("acquirer_monthly_offer_capacity", 6))
        if self.field_v41b:
            ensure_v41b_state(self.ledger)
            for agent in (a for a in self.agents if a.role == "broker"):
                agent.extra["monthly_advice_capacity"] = int(
                    cfg["scenario"].get("broker_monthly_advice_capacity",
                                        DEFAULT_ADVICE_CAPACITY))
        if self.field_v4:
            ensure_v4_state(self.ledger)
            self.ledger.v4_fee_rate = float(cfg["scenario"].get("broker_fee_rate", 0.0))
            for agent in (a for a in self.agents if a.role == "acquirer"):
                agent.extra["monthly_offer_capacity"] = int(
                    cfg["scenario"].get("acquirer_monthly_offer_capacity", 8))
        if self.field_v3:
            for agent in (a for a in self.agents if a.role == "acquirer"):
                seed_acquirer_intelligence_v3(agent, self.ledger, cfg["scenario"])

        self.broker_ids = [a.agent_id for a in self.agents if a.role == "broker"]
        if self.field_v41b:
            self.system_prompts = {
                a.agent_id: build_system_prompt_v41b(a, cfg, len(parcels),
                                                     self.broker_ids)
                for a in self.agents}
        elif self.field_v41:
            self.system_prompts = {a.agent_id: build_system_prompt_v41(a, cfg, len(parcels))
                                   for a in self.agents}
        elif self.field_v4:
            self.system_prompts = {a.agent_id: build_system_prompt_v4(a, cfg, len(parcels))
                                   for a in self.agents}
        elif self.field_v3:
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
        self.broker_ids = [a.agent_id for a in self.agents if a.role == "broker"]
        self.feelings: List[Dict[str, Any]] = []      # 住民・事業者の月次の実感（認知の観測値）
        self.thoughts: List[Dict[str, Any]] = []      # v4.1: 月次の内心（認知の観測値）
        self.deliveries: List[Dict[str, Any]] = []    # 誰に何が実際に届いたか（経路の追跡用）

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
        if self.field_v41:
            self._step_v41(step)
            return
        if self.field_v4:
            self._step_v4(step)
            return
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
            fields = _structured_fields(act)
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
                                         under_name, utterance, fields)
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
                "outcome": outcome, "plan": plan, "structured": fields,
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
        self._record_own_results(step)

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

    # -- v4: 同期3フェーズ（提示 → 応答 → 清算） ---------------------------

    def _call_batch(self, items, tag: str):
        """(agent, system_prompt, user_prompt, schema) をまとめて呼ぶ。"""
        workers = int(self.cfg["llm"].get("parallel_workers", 8))
        results: Dict[str, Dict[str, Any]] = {}

        def call(item):
            agent, system_prompt, user_prompt, schema = item
            started = time.time()
            raw = self.client.generate(system_prompt, user_prompt, schema=schema,
                                       tag=f"agent:{agent.role}:{tag}")
            return agent.agent_id, {"raw": raw, "user_prompt": user_prompt,
                                    "latency": time.time() - started}

        if not items:
            return results
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for aid, result in ex.map(call, items):
                results[aid] = result
        return results

    def _deliver_v4(self, messages: List[Dict[str, Any]], step: int) -> None:
        for message in messages:
            destination = message["to"]
            agent = self.by_id.get(destination)
            if agent is None:
                continue
            agent.inbox.append({k: v for k, v in message.items() if k != "to"})
            self.deliveries.append({
                "step": step, "to": destination, "from": message.get("from", ""),
                "obs_id": message.get("obs_id", ""), "kind": message.get("kind", "talk"),
                "location": message.get("location", ""),
                "text": str(message.get("text", ""))[:200],
            })

    def _step_v4(self, step: int) -> None:
        ensure_v4_state(self.ledger)
        venue_ids = {v["id"] for v in self.cfg.get("social", {}).get("venues", [])}
        names = self.names

        # --- フェーズ1: 提示 -------------------------------------------------
        items = [(a, self.system_prompts[a.agent_id],
                  build_phase1_prompt_v4(a, self.ledger, step, self.n_steps, names, self.cfg),
                  phase1_schema_v4(a, self.cfg, self.broker_ids))
                 for a in self.agents]
        results = self._call_batch(items, "p1")
        # 観測は作成済み。ここから届く情報は翌月にだけ見える。
        # 同じ月の応答フェーズでも「すでに自分へ届いていた情報」は見えている必要があるので、
        # 消す前に写しを取る。
        inbox_snapshot = {a.agent_id: list(a.inbox) for a in self.agents}
        for a in self.agents:
            a.inbox = []

        messages: List[Dict[str, Any]] = []
        presences: Dict[str, str] = {}
        ambient_rows: List[Dict[str, Any]] = []
        month_feelings: Dict[str, Dict[str, Any]] = {}

        for a in sorted(self.agents, key=lambda x: x.agent_id):
            result = results.get(a.agent_id, {})
            act = _parse_action(result.get("raw", ""))
            event: Dict[str, Any] = {
                "step": step, "phase": 1, "agent_id": a.agent_id, "role": a.role,
                "name": a.name, "latency_sec": round(result.get("latency", 0.0), 2),
            }
            if act is None:
                self.invalid_count += 1
                event.update({"action_type": "PARSE_FAIL",
                              "outcome": {"kind": "parse_fail",
                                          "reason": "unparseable_response"},
                              "raw": (result.get("raw") or "")[:400]})
                self.events.append(event)
                continue

            location = str(act.get("location", "")).strip()
            utterance = str(act.get("utterance", "")).strip()
            channel = str(act.get("utterance_channel", "none")).strip()
            utterance_to = str(act.get("utterance_to", "")).strip()
            memory = str(act.get("memory", "")).strip()
            feeling = str(act.get("feeling", "")).strip()
            a.memory = memory
            valid_location = location in venue_ids or location in ("HOME", "OFFICE")
            place = f"{location}:{a.agent_id}" if location in ("HOME", "OFFICE") else location
            operations: List[Dict[str, Any]] = []

            if a.role == "acquirer":
                capacity = int(a.extra.get("monthly_offer_capacity", 8))
                raw_offers = act.get("offers", [])
                raw_offers = raw_offers if isinstance(raw_offers, list) else []
                for index, row in enumerate(raw_offers):
                    if not isinstance(row, dict):
                        outcome = {"kind": "invalid_action", "reason": "offer_not_object"}
                        operations.append({"action_type": "make_offer", "target": "",
                                           "amount": 0, "outcome": outcome})
                        self.invalid_count += 1
                        continue
                    parcel_id = str(row.get("parcel_id", "")).strip()
                    price = row.get("price", 0)
                    under_name = str(row.get("under_name", "")).strip()
                    via = str(row.get("via", "")).strip()
                    broker_id = str(row.get("broker_id", "")).strip()
                    note = str(row.get("note", "")).strip()
                    if index >= capacity:
                        outcome = {"kind": "invalid_action",
                                   "reason": "monthly_offer_capacity_exceeded",
                                   "capacity": capacity}
                    elif not parcel_id:
                        outcome = {"kind": "invalid_action", "reason": "missing_target"}
                    elif not _has_amount({"amount": price}):
                        outcome = {"kind": "invalid_action", "reason": "missing_amount"}
                    else:
                        outcome = record_offer_v4(self.ledger, step, a, parcel_id, price,
                                                  under_name, via, broker_id,
                                                  self.broker_ids, note)
                    if _is_rejected(outcome):
                        self.invalid_count += 1
                    operations.append({"action_type": "make_offer", "target": parcel_id,
                                       "amount": _as_int(price), "under_name": under_name,
                                       "via": via, "broker_id": broker_id, "note": note,
                                       "outcome": outcome})
                raw_withdraw = act.get("withdraw", [])
                raw_withdraw = raw_withdraw if isinstance(raw_withdraw, list) else []
                for value in raw_withdraw:
                    offer_id = str(value).strip()
                    if not offer_id:
                        continue
                    outcome = self.ledger.record_withdraw(step, offer_id, a.agent_id)
                    if _is_rejected(outcome):
                        self.invalid_count += 1
                    operations.append({"action_type": "withdraw_offer", "target": offer_id,
                                       "amount": 0, "outcome": outcome})
                if not operations:
                    # 買付を出さない月も、選ばれた事実として記録する。
                    operations.append({"action_type": "no_offer", "target": "", "amount": 0,
                                       "outcome": {"kind": "no_offer"}})
                event["memo"] = str(act.get("memo", "")).strip()[:200]

            elif a.role in ("municipality", "media"):
                investigate = str(act.get("investigate", "none")).strip()
                if investigate == "land_registry":
                    a.extra["registry_stats_seen"] = True
                    outcome = self.ledger.record_note(step, a.agent_id, "investigate",
                                                      "land_registry")
                    operations.append({"action_type": "investigate",
                                       "target": "land_registry", "amount": 0,
                                       "outcome": outcome})
                elif investigate == "corporate_records":
                    a.extra["corporate_records_seen"] = True
                    outcome = self.ledger.record_note(step, a.agent_id, "investigate",
                                                      "corporate_records")
                    operations.append({"action_type": "investigate",
                                       "target": "corporate_records", "amount": 0,
                                       "outcome": outcome})
                publish = str(act.get("publish", "")).strip()
                if publish:
                    if a.role == "media":
                        headline = (publish.splitlines()[0] if publish else "（無題）")[:80]
                        outcome = self.ledger.record_publication(
                            step, a.agent_id, headline, publish[:400],
                            self._is_about_acquisition(publish))
                        article_id = f"NEWS-{a.agent_id}-M{step:02d}"
                        for recipient in self.agents:
                            subscriptions = recipient.extra.get("subscriptions", [])
                            if a.name in subscriptions or a.agent_id in subscriptions:
                                messages.append({"from": a.agent_id, "to": recipient.agent_id,
                                                 "text": publish[:400], "step": step,
                                                 "kind": "article", "obs_id": article_id})
                    else:
                        outcome = self.ledger.record_note(step, a.agent_id,
                                                          "public_statement", publish[:400])
                        notice_id = f"GOV-{a.agent_id}-M{step:02d}"
                        for recipient in self.agents:
                            if recipient.agent_id == a.agent_id:
                                continue
                            messages.append({"from": a.agent_id, "to": recipient.agent_id,
                                             "text": publish[:400], "step": step,
                                             "kind": "public_statement", "obs_id": notice_id})
                    operations.append({"action_type": "publish", "target": "", "amount": 0,
                                       "outcome": outcome})
                if a.role == "municipality":
                    title = str(act.get("ordinance_title", "")).strip()
                    body = str(act.get("ordinance_text", "")).strip()
                    if title and body:
                        outcome = enact_ordinance_v4(
                            self.ledger, step, a.agent_id, title, body,
                            act.get("ordinance_threshold_sqm", 0),
                            act.get("ordinance_delay_months", 0))
                        if _is_rejected(outcome):
                            self.invalid_count += 1
                        else:
                            notice_id = f"ORD-{a.agent_id}-M{step:02d}"
                            text = (f"【条例施行の告示】{title}：{body[:200]}"
                                    f"（対象:1件{outcome.get('threshold_sqm')}㎡超の取得 / "
                                    f"届出による成立の遅延:{outcome.get('delay_months')}か月 / "
                                    f"施行:第{outcome.get('effective_step')}月）")
                            for recipient in self.agents:
                                if recipient.agent_id == a.agent_id:
                                    continue
                                messages.append({"from": a.agent_id,
                                                 "to": recipient.agent_id, "text": text,
                                                 "step": step, "kind": "ordinance",
                                                 "obs_id": notice_id})
                        operations.append({"action_type": "enact_ordinance",
                                           "target": title, "amount": 0,
                                           "outcome": outcome})
                if not operations:
                    operations.append({"action_type": "routine", "target": "", "amount": 0,
                                       "outcome": {"kind": "no_ledger_change"}})
            else:
                operations.append({"action_type": "day", "target": "", "amount": 0,
                                   "outcome": {"kind": "day", "location": location}})

            if valid_location:
                presences[a.agent_id] = place
            if utterance and channel == "ambient" and valid_location:
                row = {"step": step, "from": a.agent_id, "role": a.role, "name": a.name,
                       "location": place, "text": utterance}
                ambient_rows.append(row)
                self.all_utterances.append(row)
            elif utterance and channel == "direct":
                destination = self._agent_id(utterance_to)
                if destination:
                    messages.append({"from": a.agent_id, "to": destination,
                                     "text": utterance, "step": step, "kind": "direct",
                                     "obs_id": f"MSG-M{step:02d}-{a.agent_id}-{destination}"})
            if feeling and a.role in ("household", "business"):
                month_feelings[a.agent_id] = {
                    "step": step, "from": a.agent_id, "role": a.role,
                    "name": a.name, "phase": "day", "text": feeling}
            if act.get("_truncated"):
                self.truncated_count += 1

            event.update({
                "action_type": ("offers" if a.role == "acquirer" else "day"),
                "target": "", "amount": 0, "location": location, "utterance": utterance,
                "utterance_channel": channel, "utterance_to": utterance_to,
                "memory": memory, "feeling": feeling, "reasoning": "",
                "evidence": [], "under_name": "",
                "truncated": bool(act.get("_truncated")),
                "operations": operations,
                "outcome": {"kind": "phase1", "operations": len(operations)},
            })
            self.events.append(event)

        # --- フェーズ2: 応答（買付が届いている所有者だけ） ---------------------
        offers_by_owner = owners_with_offers(self.ledger, step)
        responders = [self.by_id[oid] for oid in sorted(offers_by_owner)
                      if oid in self.by_id]
        items2 = [(a, self.system_prompts[a.agent_id],
                   build_phase2_prompt_v4(a, self.ledger, step, self.n_steps, names,
                                          offers_by_owner[a.agent_id],
                                          inbox_snapshot.get(a.agent_id, [])),
                   phase2_schema_v4())
                  for a in responders]
        results2 = self._call_batch(items2, "p2")

        for a in sorted(responders, key=lambda x: x.agent_id):
            result = results2.get(a.agent_id, {})
            act = _parse_action(result.get("raw", ""))
            event = {"step": step, "phase": 2, "agent_id": a.agent_id, "role": a.role,
                     "name": a.name, "latency_sec": round(result.get("latency", 0.0), 2)}
            if act is None:
                self.invalid_count += 1
                event.update({"action_type": "PARSE_FAIL",
                              "outcome": {"kind": "parse_fail",
                                          "reason": "unparseable_response"},
                              "raw": (result.get("raw") or "")[:400]})
                self.events.append(event)
                continue
            raw_responses = act.get("responses", [])
            raw_responses = raw_responses if isinstance(raw_responses, list) else []
            operations = []
            for row in raw_responses:
                if not isinstance(row, dict):
                    self.invalid_count += 1
                    operations.append({"action_type": "response", "target": "", "amount": 0,
                                       "outcome": {"kind": "invalid_action",
                                                   "reason": "response_not_object"}})
                    continue
                offer_id = str(row.get("offer_id", "")).strip()
                decision = str(row.get("decision", "")).strip()
                counter_price = row.get("counter_price", 0)
                if not offer_id:
                    outcome = {"kind": "invalid_action", "reason": "missing_target"}
                elif decision == "counter" and not _has_amount({"amount": counter_price}):
                    outcome = {"kind": "invalid_action", "reason": "missing_amount"}
                else:
                    outcome = respond_to_offer_v4(self.ledger, step, offer_id,
                                                  a.agent_id, decision, counter_price)
                if _is_rejected(outcome):
                    self.invalid_count += 1
                operations.append({"action_type": decision or "response",
                                   "target": offer_id,
                                   "amount": _as_int(counter_price), "outcome": outcome})
            feeling = str(act.get("feeling", "")).strip()
            memory = str(act.get("memory", "")).strip()
            if memory:
                a.memory = memory
            if feeling:
                # 同じ月の実感は1件に統合する（応答フェーズの実感がその月の最後の実感）。
                month_feelings[a.agent_id] = {
                    "step": step, "from": a.agent_id, "role": a.role,
                    "name": a.name, "phase": "response", "text": feeling}
            if act.get("_truncated"):
                self.truncated_count += 1
            answered = {str(op.get("target", "")).strip().upper().strip("[]")
                        for op in operations}
            for offer in offers_by_owner[a.agent_id]:
                if offer.offer_id in answered or offer.status != "open":
                    continue
                # 触れられなかった買付は、買付ごとに「今月は答えなかった」と記録する。
                outcome = respond_to_offer_v4(self.ledger, step, offer.offer_id,
                                              a.agent_id, "no_response", 0)
                operations.append({"action_type": "no_response",
                                   "target": offer.offer_id, "amount": 0,
                                   "outcome": outcome})
            event.update({"action_type": "responses", "target": "", "amount": 0,
                          "location": "", "utterance": "", "utterance_channel": "none",
                          "utterance_to": "", "memory": memory, "feeling": feeling,
                          "reasoning": "", "evidence": [], "under_name": "",
                          "truncated": bool(act.get("_truncated")),
                          "operations": operations,
                          "outcome": {"kind": "phase2", "responses": len(operations)}})
            self.events.append(event)

        # --- フェーズ3: 清算・配送 -------------------------------------------
        for row in ambient_rows:
            for destination, place in presences.items():
                if destination != row["from"] and place == row["location"]:
                    messages.append({"from": row["from"], "to": destination,
                                     "text": row["text"], "step": step,
                                     "location": row["location"], "kind": "ambient",
                                     "obs_id": f"TALK-{row['location']}-M{step:02d}-{row['from']}"})
        self._deliver_v4(messages, step)
        self.timeline = []
        # 届出を終えた成立も、即時の成立と同じく賃料清算の前に記帳する（帰属を揃える）。
        filed_outcomes = execute_pending_transfers_v4(self.ledger, step)
        self.ledger.settle_month(step, self.business_margins)
        for row in sorted(month_feelings.values(), key=lambda r: r["from"]):
            self.feelings.append(row)
        self._record_own_results_v4(step, filed_outcomes)

    def _record_own_results_v4(self, step: int,
                               filed_outcomes: Optional[List[Dict[str, Any]]] = None) -> None:
        """今月の自分の行為が帳簿でどうなったかを、本人の翌月観測へ渡す（事実だけ）。"""
        for a in self.agents:
            rows: List[Dict[str, Any]] = []
            for event in self.events:
                if event.get("step") != step or event.get("agent_id") != a.agent_id:
                    continue
                operations = event.get("operations")
                if operations:
                    rows.extend(own_result_row(step, op.get("action_type", ""),
                                               op.get("target", ""), op.get("outcome", {}))
                                for op in operations)
                else:
                    rows.append(own_result_row(step, event.get("action_type", ""),
                                               event.get("target", ""),
                                               event.get("outcome", {})))
            for outcome in (filed_outcomes or []):
                # 届出を終えた成立は行為の記録ではなく帳簿の記録なので、
                # 当事者（買主・売主）にだけ結果として返す。
                if a.agent_id not in (outcome.get("buyer"), outcome.get("seller"),
                                      outcome.get("by")):
                    continue
                rows.append(own_result_row(step, "filed_acquisition",
                                           outcome.get("parcel_id", ""), outcome))
            a.extra["last_month_results"] = rows

    # -- v4.1: 金額のない同期3フェーズ（内心 → 行動） -----------------------

    def _step_v41(self, step: int) -> None:
        ensure_v41_state(self.ledger)
        venue_ids = {v["id"] for v in self.cfg.get("social", {}).get("venues", [])}
        names = self.names

        if self.field_v41b:
            ensure_v41b_state(self.ledger)
            items = [(a, self.system_prompts[a.agent_id],
                      build_phase1_prompt_v41b(a, self.ledger, step, self.n_steps,
                                               names, self.cfg),
                      phase1_schema_v41b(a, self.broker_ids))
                     for a in self.agents]
        else:
            items = [(a, self.system_prompts[a.agent_id],
                      build_phase1_prompt_v41(a, self.ledger, step, self.n_steps, names,
                                              self.cfg),
                      phase1_schema_v41(a))
                     for a in self.agents]
        results = self._call_batch(items, "p1")
        inbox_snapshot = {a.agent_id: list(a.inbox) for a in self.agents}
        for a in self.agents:
            a.inbox = []

        messages: List[Dict[str, Any]] = []
        presences: Dict[str, str] = {}
        ambient_rows: List[Dict[str, Any]] = []
        month_thoughts: Dict[str, Dict[str, Any]] = {}

        for a in sorted(self.agents, key=lambda x: x.agent_id):
            result = results.get(a.agent_id, {})
            act = _parse_action(result.get("raw", ""))
            event: Dict[str, Any] = {
                "step": step, "phase": 1, "agent_id": a.agent_id, "role": a.role,
                "name": a.name, "latency_sec": round(result.get("latency", 0.0), 2),
            }
            if act is None:
                self.invalid_count += 1
                event.update({"action_type": "PARSE_FAIL",
                              "outcome": {"kind": "parse_fail",
                                          "reason": "unparseable_response"},
                              "raw": (result.get("raw") or "")[:400]})
                self.events.append(event)
                continue

            thought = str(act.get("thought", "")).strip()
            location = str(act.get("location", "")).strip()
            utterance = str(act.get("utterance", "")).strip()
            channel = str(act.get("utterance_channel", "none")).strip()
            utterance_to = str(act.get("utterance_to", "")).strip()
            if thought:
                # 内心はそのまま翌月へ持ち越す（世界は要約も編集もしない）。
                a.extra["thought"] = thought
                a.memory = thought
                month_thoughts[a.agent_id] = {
                    "step": step, "from": a.agent_id, "role": a.role,
                    "name": a.name, "phase": "day", "text": thought}
            valid_location = location in venue_ids or location in ("HOME", "OFFICE")
            place = f"{location}:{a.agent_id}" if location in ("HOME", "OFFICE") else location
            operations: List[Dict[str, Any]] = []

            if a.role == "acquirer":
                capacity = int(a.extra.get("monthly_offer_capacity", 6))
                raw_offers = act.get("offers", [])
                raw_offers = raw_offers if isinstance(raw_offers, list) else []
                for index, row in enumerate(raw_offers):
                    if not isinstance(row, dict):
                        self.invalid_count += 1
                        operations.append({"action_type": "make_offer", "target": "",
                                           "outcome": {"kind": "invalid_action",
                                                       "reason": "offer_not_object"}})
                        continue
                    parcel_id = str(row.get("parcel_id", "")).strip()
                    under_name = str(row.get("under_name", "")).strip()
                    if index >= capacity:
                        outcome = {"kind": "invalid_action",
                                   "reason": "monthly_offer_capacity_exceeded",
                                   "capacity": capacity}
                    elif not parcel_id:
                        outcome = {"kind": "invalid_action", "reason": "missing_target"}
                    else:
                        outcome = record_offer_v41(self.ledger, step, a, parcel_id,
                                                   under_name)
                    if _is_rejected(outcome):
                        self.invalid_count += 1
                    operations.append({"action_type": "make_offer", "target": parcel_id,
                                       "under_name": under_name, "outcome": outcome})
                raw_withdraw = act.get("withdraw", [])
                raw_withdraw = raw_withdraw if isinstance(raw_withdraw, list) else []
                for value in raw_withdraw:
                    offer_id = str(value).strip()
                    if not offer_id:
                        continue
                    outcome = withdraw_offer_v41(self.ledger, step, offer_id, a.agent_id)
                    if _is_rejected(outcome):
                        self.invalid_count += 1
                    operations.append({"action_type": "withdraw_offer",
                                       "target": offer_id, "outcome": outcome})
                if not operations:
                    operations.append({"action_type": "no_offer", "target": "",
                                       "outcome": {"kind": "no_offer"}})

            elif a.role in ("municipality", "media"):
                investigate = str(act.get("investigate", "none")).strip()
                if investigate in ("land_registry", "corporate_records"):
                    a.extra["registry_stats_seen" if investigate == "land_registry"
                            else "corporate_records_seen"] = True
                    outcome = self.ledger.record_note(step, a.agent_id, "investigate",
                                                      investigate)
                    operations.append({"action_type": "investigate",
                                       "target": investigate, "outcome": outcome})
                publish = str(act.get("publish", "")).strip()
                if publish:
                    if a.role == "media":
                        headline = (publish.splitlines()[0] if publish else "（無題）")[:80]
                        outcome = self.ledger.record_publication(
                            step, a.agent_id, headline, publish[:400],
                            self._is_about_acquisition(publish))
                        article_id = f"NEWS-{a.agent_id}-M{step:02d}"
                        for recipient in self.agents:
                            subscriptions = recipient.extra.get("subscriptions", [])
                            if a.name in subscriptions or a.agent_id in subscriptions:
                                messages.append({"from": a.agent_id,
                                                 "to": recipient.agent_id,
                                                 "text": publish[:400], "step": step,
                                                 "kind": "article", "obs_id": article_id})
                    else:
                        outcome = self.ledger.record_note(step, a.agent_id,
                                                          "public_statement",
                                                          publish[:400])
                        notice_id = f"GOV-{a.agent_id}-M{step:02d}"
                        for recipient in self.agents:
                            if recipient.agent_id != a.agent_id:
                                messages.append({"from": a.agent_id,
                                                 "to": recipient.agent_id,
                                                 "text": publish[:400], "step": step,
                                                 "kind": "public_statement",
                                                 "obs_id": notice_id})
                    operations.append({"action_type": "publish", "target": "",
                                       "outcome": outcome})
                if a.role == "municipality":
                    title = str(act.get("ordinance_title", "")).strip()
                    body = str(act.get("ordinance_text", "")).strip()
                    threshold_given = act.get("ordinance_threshold_sqm", None)
                    delay_given = act.get("ordinance_delay_months", None)
                    if title or body:
                        # どれか1つでも書かれていたら必ず検証へ渡す（欠損は不成立にする）。
                        outcome = enact_ordinance_v41(
                            self.ledger, step, a.agent_id, title, body,
                            threshold_given, delay_given)
                        if _is_rejected(outcome):
                            self.invalid_count += 1
                        else:
                            notice_id = f"ORD-{a.agent_id}-M{step:02d}"
                            text = (f"【条例施行の告示】{title}：{body[:200]}"
                                    f"（対象:1件{outcome.get('threshold_sqm')}㎡超の取得 / "
                                    f"届出による成立の遅延:{outcome.get('delay_months')}か月 / "
                                    f"施行:第{outcome.get('effective_step')}月）")
                            for recipient in self.agents:
                                if recipient.agent_id != a.agent_id:
                                    messages.append({"from": a.agent_id,
                                                     "to": recipient.agent_id,
                                                     "text": text, "step": step,
                                                     "kind": "ordinance",
                                                     "obs_id": notice_id})
                        operations.append({"action_type": "enact_ordinance",
                                           "target": title, "outcome": outcome})
                if not operations:
                    operations.append({"action_type": "routine", "target": "",
                                       "outcome": {"kind": "no_ledger_change"}})
            elif self.field_v41b and a.role in ("household", "business"):
                # 相談（月1件）。相談するかどうか・何を相談するかは本人が決める。
                broker_id = str(act.get("consult_broker_id", "")).strip()
                question = str(act.get("consult_question", "")).strip()
                if broker_id and broker_id != CONSULT_NONE:
                    outcome = record_consult_v41b(self.ledger, step, a, broker_id,
                                                  question, self.broker_ids)
                    if _is_rejected(outcome):
                        self.invalid_count += 1
                    elif outcome.get("kind") == "consult":
                        messages.append({
                            "from": a.agent_id, "to": broker_id,
                            "text": outcome.get("question", ""), "step": step,
                            "kind": "consult",
                            "obs_id": str(outcome.get("consult_id"))})
                    operations.append({"action_type": "consult", "target": broker_id,
                                       "outcome": outcome})
                elif question:
                    # 相手の指定が無い相談は届け先が決まらない＝不成立（補完しない）。
                    self.invalid_count += 1
                    operations.append({"action_type": "consult", "target": "",
                                       "outcome": {"kind": "invalid_action",
                                                   "reason": "missing_broker_id"}})
                if not operations:
                    operations.append({"action_type": "day", "target": "",
                                       "outcome": {"kind": "day", "location": location}})
            elif self.field_v41b and a.role == "broker":
                # 自分宛の相談への回答。答えるかどうか・何を答えるかは本人が決める。
                capacity = int(a.extra.get("monthly_advice_capacity",
                                           DEFAULT_ADVICE_CAPACITY))
                raw_advices = act.get("advices", [])
                raw_advices = raw_advices if isinstance(raw_advices, list) else []
                for index, row in enumerate(raw_advices):
                    if not isinstance(row, dict):
                        self.invalid_count += 1
                        operations.append({"action_type": "advise", "target": "",
                                           "outcome": {"kind": "invalid_action",
                                                       "reason": "advice_not_object"}})
                        continue
                    consult_id = str(row.get("consult_id", "")).strip()
                    reply = str(row.get("reply", "")).strip()
                    if index >= capacity:
                        outcome = {"kind": "invalid_action",
                                   "reason": "monthly_advice_capacity_exceeded",
                                   "capacity": capacity}
                    elif not consult_id:
                        outcome = {"kind": "invalid_action", "reason": "missing_target"}
                    else:
                        outcome = answer_consult_v41b(self.ledger, step, a.agent_id,
                                                      consult_id, reply,
                                                      capacity=capacity)
                    if _is_rejected(outcome):
                        self.invalid_count += 1
                    elif outcome.get("kind") == "advice":
                        messages.append({
                            "from": a.agent_id, "to": outcome.get("to"),
                            "text": outcome.get("reply", ""), "step": step,
                            "kind": "advice",
                            "obs_id": str(outcome.get("consult_id"))})
                    operations.append({"action_type": "advise", "target": consult_id,
                                       "outcome": outcome})
                if not operations:
                    operations.append({"action_type": "day", "target": "",
                                       "outcome": {"kind": "day", "location": location}})
            else:
                operations.append({"action_type": "day", "target": "",
                                   "outcome": {"kind": "day", "location": location}})

            if valid_location:
                presences[a.agent_id] = place
            elif location:
                operations.append({"action_type": "visit", "target": location,
                                   "outcome": {"kind": "presence_rejected",
                                               "reason": "unknown_location"}})
            if utterance and channel == "ambient" and valid_location:
                row = {"step": step, "from": a.agent_id, "role": a.role, "name": a.name,
                       "location": place, "text": utterance[:MAX_TEXT_CHARS_V41]}
                ambient_rows.append(row)
                self.all_utterances.append(row)
            elif utterance and channel == "ambient" and not valid_location:
                operations.append({"action_type": "talk", "target": location,
                                   "outcome": {"kind": "delivery_rejected",
                                               "reason": "unknown_location"}})
            elif utterance and channel == "direct":
                destination = self._agent_id(utterance_to)
                if destination:
                    messages.append({"from": a.agent_id, "to": destination,
                                     "text": utterance[:MAX_TEXT_CHARS_V41], "step": step,
                                     "kind": "direct",
                                     "obs_id": f"MSG-M{step:02d}-{a.agent_id}-{destination}"})
                else:
                    operations.append({"action_type": "talk", "target": utterance_to,
                                       "outcome": {"kind": "delivery_rejected",
                                                   "reason": "unknown_recipient"}})
            if act.get("_truncated"):
                self.truncated_count += 1

            event.update({
                "action_type": ("offers" if a.role == "acquirer" else "day"),
                "target": "", "location": location,
                "utterance": utterance, "utterance_channel": channel,
                "utterance_to": utterance_to, "thought": thought,
                "evidence": [], "under_name": "",
                "truncated": bool(act.get("_truncated")),
                "operations": operations,
                "outcome": {"kind": "phase1", "operations": len(operations)},
            })
            self.events.append(event)

        # --- フェーズ2: 応答（打診が届いている所有者だけ） ---------------------
        offers_by_owner = owners_with_offers_v41(self.ledger)
        responders = [self.by_id[oid] for oid in sorted(offers_by_owner)
                      if oid in self.by_id]
        phase2_builder = (build_phase2_prompt_v41b if self.field_v41b
                          else build_phase2_prompt_v41)
        items2 = [(a, self.system_prompts[a.agent_id],
                   phase2_builder(a, self.ledger, step, self.n_steps, names,
                                  offers_by_owner[a.agent_id],
                                  inbox_snapshot.get(a.agent_id, [])),
                   phase2_schema_v41())
                  for a in responders]
        results2 = self._call_batch(items2, "p2")

        for a in sorted(responders, key=lambda x: x.agent_id):
            result = results2.get(a.agent_id, {})
            act = _parse_action(result.get("raw", ""))
            event = {"step": step, "phase": 2, "agent_id": a.agent_id, "role": a.role,
                     "name": a.name, "latency_sec": round(result.get("latency", 0.0), 2)}
            if act is None:
                self.invalid_count += 1
                # 本人の判断を捏造しない。open のまま残った打診を事実として記録する。
                pending_ids = [o["id"] for o in offers_by_owner[a.agent_id]
                               if o["status"] == "open"]
                for offer_id in pending_ids:
                    self.ledger._rec(step, "response_not_recorded", offer_id=offer_id,
                                     by=a.agent_id, reason="unparseable_response")
                event.update({"action_type": "PARSE_FAIL",
                              "outcome": {"kind": "parse_fail",
                                          "reason": "unparseable_response",
                                          "offers_left_open": pending_ids},
                              "raw": (result.get("raw") or "")[:400]})
                self.events.append(event)
                continue
            thought = str(act.get("thought", "")).strip()
            if thought:
                a.extra["thought"] = thought
                a.memory = thought
                month_thoughts[a.agent_id] = {
                    "step": step, "from": a.agent_id, "role": a.role,
                    "name": a.name, "phase": "response", "text": thought}
            raw_responses = act.get("responses", [])
            raw_responses = raw_responses if isinstance(raw_responses, list) else []
            operations = []
            for row in raw_responses:
                if not isinstance(row, dict):
                    self.invalid_count += 1
                    operations.append({"action_type": "response", "target": "",
                                       "outcome": {"kind": "invalid_action",
                                                   "reason": "response_not_object"}})
                    continue
                offer_id = str(row.get("offer_id", "")).strip()
                decision = str(row.get("decision", "")).strip()
                if not offer_id:
                    outcome = {"kind": "invalid_action", "reason": "missing_target"}
                else:
                    outcome = respond_to_offer_v41(self.ledger, step, offer_id,
                                                   a.agent_id, decision)
                if _is_rejected(outcome):
                    self.invalid_count += 1
                operations.append({"action_type": decision or "response",
                                   "target": offer_id, "outcome": outcome})
            answered = set()
            for op in operations:
                raw_target = str(op.get("target", "")).strip()
                if raw_target:
                    answered.add(self.ledger._normalize_id(raw_target, "O"))
            for offer in offers_by_owner[a.agent_id]:
                if offer["id"] in answered or offer["status"] != "open":
                    continue
                outcome = respond_to_offer_v41(self.ledger, step, offer["id"],
                                               a.agent_id, "hold")
                operations.append({"action_type": "hold", "target": offer["id"],
                                   "implicit": True, "outcome": outcome})
            if act.get("_truncated"):
                self.truncated_count += 1
            event.update({"action_type": "responses", "target": "",
                          "location": "", "utterance": "", "utterance_channel": "none",
                          "utterance_to": "", "thought": thought,
                          "evidence": [], "under_name": "",
                          "truncated": bool(act.get("_truncated")),
                          "operations": operations,
                          "outcome": {"kind": "phase2", "responses": len(operations)}})
            self.events.append(event)

        # --- フェーズ3: 清算・配送（金銭の清算は無い） -------------------------
        for row in ambient_rows:
            for destination, place in presences.items():
                if destination != row["from"] and place == row["location"]:
                    messages.append({"from": row["from"], "to": destination,
                                     "text": row["text"], "step": step,
                                     "location": row["location"], "kind": "ambient",
                                     "obs_id": f"TALK-{row['location']}-M{step:02d}-{row['from']}"})
        self._deliver_v4(messages, step)
        self.timeline = []
        filed = execute_pending_v41(self.ledger, step)
        for row in sorted(month_thoughts.values(), key=lambda r: r["from"]):
            self.thoughts.append(row)
        self._record_own_results_v41(step, filed)

    def _record_own_results_v41(self, step: int,
                                filed: Optional[List[Dict[str, Any]]] = None) -> None:
        for a in self.agents:
            rows: List[Dict[str, Any]] = []
            for event in self.events:
                if event.get("step") != step or event.get("agent_id") != a.agent_id:
                    continue
                operations = event.get("operations")
                if operations:
                    rows.extend(own_result_row_v41(step, op.get("action_type", ""),
                                                   op.get("target", ""),
                                                   op.get("outcome", {}))
                                for op in operations)
                else:
                    rows.append(own_result_row_v41(step, event.get("action_type", ""),
                                                   event.get("target", ""),
                                                   event.get("outcome", {})))
            for outcome in (filed or []):
                if a.agent_id not in (outcome.get("buyer"), outcome.get("seller"),
                                      outcome.get("by")):
                    continue
                rows.append(own_result_row_v41(step, "filed_acquisition",
                                               outcome.get("parcel_id", ""), outcome))
            a.extra["last_month_results"] = rows

    def _record_own_results(self, step: int) -> None:
        """今月自分が選んだ行為が帳簿でどうなったかを、本人の翌月観測へ渡す。

        成立も不成立も同じ形で返す。返すのは結果の種別と理由コードという事実だけで、
        次にどうすべきかは書かない。
        """
        for a in self.agents:
            event = next((e for e in reversed(self.events)
                          if e.get("step") == step and e.get("agent_id") == a.agent_id),
                         None)
            if event is None:
                a.extra["last_month_results"] = []
                continue
            operations = event.get("operations")
            if operations:
                rows = [own_result_row(step, op.get("action_type", ""),
                                       op.get("target", ""), op.get("outcome", {}))
                        for op in operations]
            else:
                rows = [own_result_row(step, event.get("action_type", ""),
                                       event.get("target", ""), event.get("outcome", {}))]
            a.extra["last_month_results"] = rows

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
            op_fields = _structured_fields(raw_operation)
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
                    step, a, action, target, amount, under_name, note, op_fields)
            outcome_kind = (outcome.get("kind", "") if isinstance(outcome, dict)
                            else str(outcome))
            engaged_parcel = next(
                (pid for pid in (target, op_fields.get("parcel_id", ""),
                                 outcome.get("parcel_id", "")
                                 if isinstance(outcome, dict) else "")
                 if pid in self.ledger.parcels), "")
            if engaged_parcel:
                a.extra.setdefault("parcel_last_action", {})[engaged_parcel] = {
                    "step": step, "action": action, "outcome": outcome_kind,
                    "amount": amount,
                }
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
                "structured": op_fields,
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
                  under_name: str, utterance: str,
                  fields: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ledger = self.ledger
        fields = fields or {}
        parcel_id = str(fields.get("parcel_id", "") or "").strip()
        inquiry_id = str(fields.get("inquiry_id", "") or "").strip()
        owner_intent = str(fields.get("owner_intent", "") or "").strip()
        asking_price = fields.get("asking_price", "")
        if action == "request_owner_inquiry":
            destination = self._agent_id(target)
            if not destination:
                return {"kind": "invalid_action", "reason": "no_such_agent",
                        "target": target}
            if self.by_id[destination].role != "broker":
                # 仲介でない相手はこの照会を実行できない＝依頼として成立しない。
                return {"kind": "invalid_action", "reason": "not_a_broker",
                        "target": target}
            return request_owner_inquiry(ledger, step, a.agent_id, destination,
                                         parcel_id, utterance)
        if action == "inquire_owner_intent":
            return inquire_owner_intent(ledger, step, a.agent_id,
                                        target or parcel_id, inquiry_id, utterance)
        if action == "answer_broker_inquiry":
            return answer_owner_inquiry(ledger, step, target, a.agent_id,
                                        owner_intent, asking_price, utterance)
        if action == "report_owner_intent":
            destination = self._agent_id(target)
            if not destination:
                return {"kind": "invalid_action", "reason": "no_such_agent",
                        "target": target}
            return report_owner_intent(ledger, step, a.agent_id, destination,
                                       inquiry_id, owner_intent, asking_price,
                                       utterance)
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
            return check_land_registry_v3(ledger, step, a, target)
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
        classified_thoughts: List[Dict[str, Any]] = []
        if self.field_v41 and kcfg.get("classify_utterances", True) and self.thoughts:
            targets_thoughts = [t for t in self.thoughts
                                if t["role"] in ("household", "business")]
            classified_thoughts = classify_utterances(
                self.client, targets_thoughts,
                batch=int(kcfg.get("classify_batch", 25)))
        classified_feelings: List[Dict[str, Any]] = []
        if self.field_v4 and kcfg.get("classify_utterances", True) and self.feelings:
            classified_feelings = classify_utterances(
                self.client, self.feelings, batch=int(kcfg.get("classify_batch", 25)))
        # v4 の認知は「毎月の実感（feeling）」の事後分類から測る。
        cog = cognition_series(
            classified_thoughts or classified_feelings or classified, self.n_steps)

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
                "final_acquirer_area_share": (
                    round(sum(p.area_sqm for p in self.ledger.parcels.values()
                              if p.use != "public" and p.owner_id in self.acquirer_ids)
                          / max(1, sum(p.area_sqm for p in self.ledger.parcels.values()
                                       if p.use != "public")), 4)
                    if (self.field_v4 or self.field_v41) else None),
                "business_survival": self.kpi_rows[-1]["business_survival"] if self.kpi_rows else None,
                "resident_outflow": self.kpi_rows[-1]["resident_outflow"] if self.kpi_rows else None,
            },
        }

        d = self.run_dir
        _write_jsonl(os.path.join(d, "events.jsonl"), self.events)
        _write_jsonl(os.path.join(d, "kpi.jsonl"), self.kpi_rows)
        _write_jsonl(os.path.join(d, "cognition.jsonl"), cog)
        _write_jsonl(os.path.join(d, "utterances.jsonl"), classified or self.all_utterances)
        if self.field_v41:
            _write_jsonl(os.path.join(d, "thoughts.jsonl"),
                         classified_thoughts or self.thoughts)
            # 分類対象は住民・事業者だけだが、内心そのものは全主体ぶん残す
            # （X社・行政・記者・仲介の内心が記録から落ちないように）。
            _write_jsonl(os.path.join(d, "thoughts_all.jsonl"), self.thoughts)
            _write_jsonl(os.path.join(d, "deliveries.jsonl"), self.deliveries)
        if self.field_v4:
            _write_jsonl(os.path.join(d, "feelings.jsonl"),
                         classified_feelings or self.feelings)
            _write_jsonl(os.path.join(d, "deliveries.jsonl"), self.deliveries)
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
