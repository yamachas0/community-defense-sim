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
from .field_v5 import (DIRECT_NONE as DIRECT_NONE_V5, HOME as HOME_V5,
                       SCENE_IDS, SCENE_LABELS,
                       acquisitions_at,
                       ambient_traces_v5, apply_script_v5, build_plan_prompt_v5,
                       build_scene_prompt_v5, build_system_prompt_v5,
                       ensure_v5_state, load_script_v5, plan_schema_v5,
                       registry_rows_v5, s4_for_step, scene_schema_v5,
                       venue_traces_v5)
from .field_v5c import build_system_prompt_v5c, venue_candidates_for_all
from .field_v5d import (TRACE_TEXTS_V5D, build_system_prompt_v5d, load_names_v5d,
                        s4_for_step_v5d, scene_schema_v5d, venue_labels_v5d)
from .field_v6 import (MAX_ACT_TEXT_CHARS, MEASURE_NONE, MEASURE_PAPER_LABEL,
                       MEASURE_VALUES, PAPER_LABELS, PUBLIC_ACT_NONE,
                       PUBLIC_ACT_VALUES, SELL_INTENT_CLEAR, SELL_INTENT_KEEP,
                       SELL_INTENT_REFUSE,
                       SELL_INTENT_VALUES, action_rows_v6,
                       blocked_acquisitions_v6, paper_row, scene_schema_v6,
                       script_without, set_refusal)
from .kpi import (classify_occupation, classify_publications, classify_utterances,
                  classify_stage_v5c, classify_stage_v5e,
                  cognition_series,
                  detection_lag, late_index, step_metrics)
from .stage_v5e import defense_level_of, rule_red_v5e


def _v5e_blue_rule():
    """青のルール1次抽出は `tools/run_metrics.py` の `_v5c_rule_blue` が正。

    走行中の停止判定と事後集計で判定が二本に割れないよう、**同じ関数を** 使う
    （docs/world_design_v5e.md §1-2・§3）。tools はパッケージではないので
    パスを足して読み込む（tests も同じやり方をしている）。
    """
    import sys as _sys
    tools = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "tools")
    if tools not in _sys.path:
        _sys.path.insert(0, tools)
    from run_metrics import _use_parcel_names, _v5c_rule_blue
    _use_parcel_names("field_v5e")      # 土地の言い方を呼び名に合わせる
    return _v5c_rule_blue
from .llm_client_factory import UsageMeter, create_llm_client
from .prompts import build_system_prompt, build_user_prompt
from .schemas import action_schema, verbs_for
from .world import Ledger, assign_tenancies, build_town

logger = logging.getLogger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.S)
_FIELD_RE = re.compile(r'"(\w+)"\s*:\s*(?:"((?:[^"\\]|\\.)*)"|(-?\d+))')


def _latency_sec(result: Dict[str, Any]) -> Optional[float]:
    """1コールの所要秒。Batch 経路では個別に測れないので None を返す。"""
    value = result.get("latency", 0.0)
    return None if value is None else round(value, 2)


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
        self.field_v5 = cfg.get("scenario_version") in ("field_v5", "field_v5b",
                                                        "field_v5c", "field_v5d",
                                                        "field_v5e", "field_v6")
        self.field_v5b = cfg.get("scenario_version") == "field_v5b"
        # v5c は v5b の世界に「買い手の戦略で組んだ台本」と「日常の場」を足しただけ。
        # 観測の作り方・兆候・プロンプトの文面は v5/v5b と同一である。
        self.field_v5c = cfg.get("scenario_version") == "field_v5c"
        # v5d は v5c の世界から「役場の窓口」を外し、土地と主体を街の呼び名で呼ぶ層。
        # 区画・日常の場・4段階の色は v5c と同じものを使う（docs/world_design_v5d.md）。
        # v5e の世界（主体への入力）は v5d と完全に同一なので、世界の層は v5d を使う。
        # v5e で変わるのは観測側（赤の定義・自衛レベル）・買い手の台本（目玉の月・停止）・
        # 月数だけである（docs/world_design_v5e.md §0）。
        # v6 の世界も v5d の層をそのまま使う（呼び名・会場・兆候・台本・月数は同一）。
        # v6 で足すのは「町の人が選べる中立な行動」だけで、v5e の分類・停止は一切通らない
        # （docs/world_design_v6_two_worlds.md §2・§3）。
        self.field_v5d = cfg.get("scenario_version") in ("field_v5d", "field_v5e",
                                                         "field_v6")
        self.field_v5e = cfg.get("scenario_version") == "field_v5e"
        self.field_v6 = cfg.get("scenario_version") == "field_v6"
        # v6: 行動ログ（enum の選択そのもの）・紙・買えなかった取得。全部決定論。
        self.v6_actions: List[Dict[str, Any]] = []
        self.v6_papers: List[Dict[str, Any]] = []
        self.v6_blocked: List[Dict[str, Any]] = []
        self.v6_blocked_acq_ids: set = set()
        self.v6_applied = 0
        self.v5c_like = self.field_v5c or self.field_v5d
        # v5e: 月末に回す分類の結果と、自衛が出て台本を止めた記録（主体には一切見せない）。
        self.stage_labels_v5e: List[Dict[str, Any]] = []
        self.defense_stop: Optional[Dict[str, Any]] = None
        self.v5e_applied = 0
        self.v5e_suspended = 0
        self.v5e_suspended_ids: List[str] = []
        self._v5e_acq_month: Dict[str, int] = {}
        # v4.1b は v4.1 の世界に相談経路と行政の面積観測を足しただけ＝土台は v4.1 と同じ。
        self.field_v41 = cfg.get("scenario_version") in ("field_v4_1", "field_v4_1b")
        self.personas = personas
        self.run_dir = run_dir
        self.n_steps = int(cfg["steps"])
        self.usage = UsageMeter()
        # Batch ジョブ台帳の置き場（途中で落ちても同じジョブを取り直せる）
        self.jobs_dir = os.path.join(run_dir, "batch_jobs")
        self.client = create_llm_client({**cfg["llm"], "seed": cfg.get("seed", 42)},
                                        self.usage, jobs_dir=self.jobs_dir)
        # 主体コールの出力上限。既定は llm.max_tokens（＝従来どおり）。
        # llm.thought_max_tokens を置いたときだけ、そちらを使う（節約設定用）。
        self.agent_max_tokens = int(cfg["llm"].get("thought_max_tokens",
                                                   cfg["llm"].get("max_tokens", 420)))
        # プロンプトの並び順。legacy＝従来／stable_first＝毎回同じ指示文を先頭に置き、
        # 変わる部分を後ろにする（**文言は1文字も変えない**。キャッシュのための並べ替え）。
        self.prompt_order = str(cfg["llm"].get("prompt_order", "legacy"))
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
        # v5d: 土地と持ち主の呼び名（対応表は configs/parcel_names_v5c.yaml の1枚だけ）。
        self.pnames: Dict[str, str] = {}
        self.registered_display: Dict[str, str] = {}
        self._display_before_v5d: Dict[str, str] = dict(self.names)
        if self.field_v5d:
            _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            _nf = str(cfg.get("names_file", "configs/parcel_names_v5c.yaml"))
            if not os.path.isabs(_nf):
                _nf = os.path.join(_root, _nf)
            _book = load_names_v5d(_nf)
            self.pnames = _book["parcels"]
            self.registered_display = _book["registered"]
            for a in self.agents:
                if a.agent_id in _book["agents"]:
                    a.name = _book["agents"][a.agent_id]
            self.names = name_map(self.agents)
        self.acquirer_ids = [a.agent_id for a in self.agents if a.role == "acquirer"]
        self.household_ids = [a.agent_id for a in self.agents if a.role == "household"]
        self.business_ids = [a.agent_id for a in self.agents if a.role == "business"]
        self.municipality_id = next(a.agent_id for a in self.agents if a.role == "municipality")

        parcels = build_town(cfg["world"], self.household_ids, self.business_ids,
                             self.municipality_id)
        assign_tenancies(parcels, self.business_ids, int(cfg["world"]["initial_shop_rent"]))
        for p in parcels:
            if self.field_v5d:
                # 登記名義の表示も同じ対応表から引く（公有地は個人名ではなく「A市」）。
                old = self._display_before_v5d.get(p.owner_id, "")
                p.registered_name = self.registered_display.get(
                    old, self.names.get(p.owner_id, p.owner_id))
            else:
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
        if self.field_v5:
            ensure_v5_state(self.ledger)
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            events_file = str(cfg.get("events_file", ""))
            if not os.path.isabs(events_file):
                events_file = os.path.join(root, events_file)
            self.script = load_script_v5(events_file)
            # 兆候をどの取得が生んだかを引くための対応表（v5e の停止で使う）。
            self._v5e_acq_month = {str(a["id"]): int(a["month"])
                                   for a in self.script.get("acquisitions", [])}
            # X社は v5 では主体ではない（一度も LLM を呼ばれない・誰の同席者にも出ない）
            self.actors = [a for a in self.agents if a.role != "acquirer"]
            self.actor_ids = [a.agent_id for a in self.actors]
            venues = cfg.get("social", {}).get("venues", [])
            self.venue_ids = [v["id"] for v in venues]
            self.venue_labels = {v["id"]: f"{v['id']} {v['label']}" for v in venues}
            self.venue_by_label: Dict[str, str] = {}
            self.agent_by_name: Dict[str, str] = {}
            if self.field_v5d:
                # 会場も呼び名で呼ぶ（V番号は出さない）。返ってきた呼び名は内部IDへ戻す。
                self.venue_labels = venue_labels_v5d(cfg)
                self.venue_by_label = {lb: vid for vid, lb in self.venue_labels.items()}
                self.agent_by_name = {a.name: a.agent_id for a in self.actors}
            scen = cfg.get("scenario", {})
            self.scene_rounds = int(scen.get("scene_rounds", 2))
            self.direct_quota = int(scen.get("direct_quota_per_month", 2))
            self.article_quota = int(scen.get("article_quota_per_month", 1))
            if self.v5c_like:
                # 行ける場所は主体ごとに違う（世界の事実＝生活動線）。毎月どこへ行くかは
                # 主体が選ぶ。system プロンプトの文面は v5 と同一で、並ぶ会場だけが違う。
                self.venue_choices = venue_candidates_for_all(self.actors, self.venue_ids)
                builder = (build_system_prompt_v5d if self.field_v5d
                           else build_system_prompt_v5c)
                self.system_prompts = {
                    a.agent_id: builder(a, cfg, len(parcels),
                                        self.venue_choices[a.agent_id])
                    for a in self.actors}
            else:
                self.venue_choices = {a.agent_id: list(self.venue_ids)
                                      for a in self.actors}
                self.system_prompts = {a.agent_id: build_system_prompt_v5(a, cfg,
                                                                          len(parcels))
                                       for a in self.actors}
        elif self.field_v41b:
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
        self.parse_fail_raw: List[Dict[str, Any]] = []  # 解釈できなかった応答の全文

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
        if self.field_v5:
            self._step_v5(step)
            return
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
                "latency_sec": _latency_sec(r),
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
                "latency_sec": _latency_sec(result),
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

    def _call_batch(self, items, tag: str, job_key: Optional[str] = None):
        """(agent, system_prompt, user_prompt, schema) をまとめて呼ぶ。

        `llm.batch_agents` が真なら Batch API（半額・返却時刻の保証なし）へ回す。
        既定は従来どおり同期の並列。プロンプト・スキーマは一切変えない。
        """
        workers = int(self.cfg["llm"].get("parallel_workers", 8))
        results: Dict[str, Dict[str, Any]] = {}
        if not items:
            return results

        use_batch = ("agents" in getattr(self.client, "batch_kinds", set())
                     and hasattr(self.client, "generate_many"))
        if use_batch:
            started = time.time()
            raws = self.client.generate_many(
                [{"system_prompt": sp, "user_prompt": up, "schema": sc,
                  "max_tokens": self.agent_max_tokens,
                  "tag": f"agent:{ag.role}:{tag}"} for ag, sp, up, sc in items],
                tag=f"agent:{tag}", kind="agents", job_key=job_key or tag)
            elapsed = time.time() - started
            for (agent, _sp, user_prompt, _sc), raw in zip(items, raws):
                # Batch では1件ごとの所要時間は測れない。ジョブ全体の時間を主体の数だけ
                # 複製すると latency_sec の合計・平均が壊れるので、個別の値は持たせない
                # （ジョブ全体の時間は summary.saving.batch_jobs に残る。
                #   Codexレビュー 2026-08-28）。
                results[agent.agent_id] = {"raw": raw, "user_prompt": user_prompt,
                                           "latency": None,
                                           "batch_job_sec": elapsed}
            return results

        def call(item):
            agent, system_prompt, user_prompt, schema = item
            started = time.time()
            raw = self.client.generate(system_prompt, user_prompt, schema=schema,
                                       max_tokens=self.agent_max_tokens,
                                       tag=f"agent:{agent.role}:{tag}")
            return agent.agent_id, {"raw": raw, "user_prompt": user_prompt,
                                    "latency": time.time() - started}

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
                "name": a.name, "latency_sec": _latency_sec(result),
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
                     "name": a.name, "latency_sec": _latency_sec(result)}
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
                "name": a.name, "latency_sec": _latency_sec(result),
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
                     "name": a.name, "latency_sec": _latency_sec(result)}
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


    # -- v5: 出来事はピン留め・観測するのは街の会話 -------------------------

    def _v5_owns(self, agent: Agent) -> bool:
        return any(p.owner_id == agent.agent_id for p in self.ledger.parcels.values())

    def _v5_thought(self, agent: Agent, act: Dict[str, Any], step: int,
                    scene: str, venue: str, round_no: int = 0) -> None:
        thought = str(act.get("thought", "") or "").strip()
        if not thought:
            return
        agent.extra["thought"] = thought
        # round は同じ月の中の前後関係を後から復元するために残す
        # （Codexレビュー 2026-08-28：初出の時系列が月単位でしか分からなかった）。
        self.thoughts.append({"step": step, "from": agent.agent_id, "role": agent.role,
                              "name": agent.name, "text": thought,
                              "scene": scene, "venue": venue, "round": round_no})

    def _v5_stance(self, agent: Agent, act: Dict[str, Any], step: int,
                   scene: str) -> None:
        if "stance" not in act:
            return
        stance = str(act.get("stance", "") or "").strip()
        if stance not in ("sell", "keep"):
            return
        # その月の「最後のコール」の値を採る（同月の既存行を差し替える）。
        rows = self.ledger.v5_stances
        for i in range(len(rows) - 1, -1, -1):
            if rows[i]["step"] == step and rows[i]["agent_id"] == agent.agent_id:
                rows[i] = {"step": step, "agent_id": agent.agent_id, "role": agent.role,
                           "stance": stance, "scene": scene}
                return
        rows.append({"step": step, "agent_id": agent.agent_id, "role": agent.role,
                     "stance": stance, "scene": scene})

    def _v6_actions(self, agent: Agent, act: Dict[str, Any], step: int,
                    scene: str, venue: str = "") -> None:
        """v6: 主体が自分で選んだ行動を、そのまま世界に記帳する（解釈しない）。

        ここでやるのは3つだけである（docs/world_design_v6_two_worlds.md §3）。

          1. `sell_intent` を登記簿の `refusal` 列へ反映する（自分の土地にだけ効く）
          2. `public_act` / `measure` を紙にして**翌月**街の全員へ届ける
             （配送は記事とまったく同じ型＝私信ではない・公の記録）
          3. 選ばれたものを全部ログに残す（「変えない／なし」も同じ重みで残す）

        促さない・採点しない・条件分岐で行動を決めない。空欄は空欄のまま。
        """
        if not self.field_v6:
            return
        base = {"step": step, "agent_id": agent.agent_id, "role": agent.role,
                "name": agent.name, "scene": scene, "venue": venue}

        # 欠けていたら既定（「変えない」「なし」）として扱う＝**本当の既定**を実装する
        # （Codexレビュー 2026-08-29 走行前指摘）。選ばれなかった月も同じ重みで残す。
        intent = str(act.get("sell_intent", "") or "").strip()
        if not intent and "sell_intent" in act:
            intent = SELL_INTENT_KEEP
        if intent in SELL_INTENT_VALUES:
            changed: List[str] = []
            if intent in (SELL_INTENT_REFUSE, SELL_INTENT_CLEAR):
                changed = set_refusal(self.ledger, step, agent.agent_id,
                                      intent == SELL_INTENT_REFUSE)
            self.v6_actions.append({**base, "field": "sell_intent", "value": intent,
                                    "parcels": changed,
                                    "parcel_names": [self.pnames.get(p, p)
                                                     for p in changed]})
        elif intent:
            self.invalid_count += 1
            self.ledger._rec(step, "action_rejected", by=agent.agent_id,
                             field="sell_intent", given=intent, reason="not_in_enum")

        if agent.role == "municipality":
            value = str(act.get("measure", "") or "").strip()
            text = str(act.get("measure_text", "") or "").strip()[:MAX_ACT_TEXT_CHARS]
            self._v6_public(agent, step, scene, venue, base, "measure", value,
                            MEASURE_VALUES, MEASURE_NONE, text,
                            MEASURE_PAPER_LABEL)
        else:
            value = str(act.get("public_act", "") or "").strip()
            text = str(act.get("public_act_text", "") or "").strip()[:MAX_ACT_TEXT_CHARS]
            self._v6_public(agent, step, scene, venue, base, "public_act", value,
                            PUBLIC_ACT_VALUES, PUBLIC_ACT_NONE, text,
                            PAPER_LABELS.get(value, "紙"))

    def _v6_public(self, agent: Agent, step: int, scene: str, venue: str,
                   base: Dict[str, Any], field_name: str, value: str,
                   allowed: List[str], none_value: str, text: str,
                   label: str) -> None:
        """公の行動（回覧板・議題・申入れ・行政の措置）を紙にして配る。"""
        if not value:
            if field_name not in ("public_act", "measure"):
                return
            value = none_value    # 欠けていたら既定（なし）として扱う
        if value not in allowed:
            self.invalid_count += 1
            self.ledger._rec(step, "action_rejected", by=agent.agent_id,
                             field=field_name, given=value, reason="not_in_enum")
            return
        row = {**base, "field": field_name, "value": value, "text": text}
        if value == none_value:
            self.v6_actions.append(row)
            return
        if not text:
            # 中身の無い紙は世界に残らない（配らない）。選んだ事実は記録する。
            self.ledger._rec(step, "paper_rejected", by=agent.agent_id,
                             field=field_name, given=value, reason="empty_text")
            self.v6_actions.append({**row, "delivered": False})
            return
        paper = paper_row(step, agent.agent_id, agent.role, agent.name,
                          value, label, text)
        self.v6_papers.append(paper)
        self.v6_actions.append({**row, "delivered": True, "label": label})
        for other in self.actors:
            other.inbox.append({"kind": "paper", "from": agent.agent_id,
                                "text": paper["text"], "step": step,
                                "label": label})
            self.deliveries.append({"step": step, "to": other.agent_id,
                                    "from": agent.agent_id, "kind": "paper",
                                    "location": scene, "text": paper["text"][:200]})

    def _v5_direct(self, agent: Agent, act: Dict[str, Any], step: int,
                   scene: str) -> None:
        to = str(act.get("direct_to", "") or "").strip()
        text = str(act.get("direct_text", "") or "").strip()
        if to in ("", DIRECT_NONE_V5):
            return
        if to not in self.by_id or to == agent.agent_id or to in self.acquirer_ids:
            self.ledger._rec(step, "direct_rejected", from_id=agent.agent_id,
                             to=to, reason="no_such_recipient")
            self.invalid_count += 1
            return
        if not text:
            self.ledger._rec(step, "direct_rejected", from_id=agent.agent_id,
                             to=to, reason="empty_text")
            return
        if agent.extra.get("directs_used", 0) >= self.direct_quota:
            self.ledger._rec(step, "direct_rejected", from_id=agent.agent_id,
                             to=to, reason="quota_exhausted")
            return
        agent.extra["directs_used"] = agent.extra.get("directs_used", 0) + 1
        self.by_id[to].inbox.append({"kind": "direct", "from": agent.agent_id,
                                     "text": text, "step": step})
        self.ledger.v5_directs.append({"step": step, "from": agent.agent_id, "to": to,
                                       "scene": scene, "text": text})
        self.deliveries.append({"step": step, "to": to, "from": agent.agent_id,
                                "kind": "direct", "location": scene, "text": text[:200]})

    def _v5_publish(self, agent: Agent, act: Dict[str, Any], step: int,
                    scene: str, round_no: int = 0) -> None:
        text = str(act.get("publish", "") or "").strip()
        if not text:
            return
        if agent.extra.get("articles_used", 0) >= self.article_quota:
            self.ledger._rec(step, "article_rejected", from_id=agent.agent_id,
                             reason="quota_exhausted")
            return
        agent.extra["articles_used"] = agent.extra.get("articles_used", 0) + 1
        self.ledger.record_publication(step, agent.agent_id, text[:40], text, False)
        self.ledger.v5_articles.append({"step": step, "from": agent.agent_id,
                                        "scene": scene, "round": round_no,
                                        "text": text})
        for other in self.actors:
            other.inbox.append({"kind": "article", "from": agent.agent_id,
                                "text": text, "step": step})
            self.deliveries.append({"step": step, "to": other.agent_id,
                                    "from": agent.agent_id, "kind": "article",
                                    "location": scene, "text": text[:200]})

    def _v5_parse(self, agent: Agent, results: Dict[str, Any], step: int,
                  tag: str) -> Optional[Dict[str, Any]]:
        raw = results.get(agent.agent_id, {}).get("raw", "")
        act = _parse_action(raw)
        if act is None:
            self.invalid_count += 1
            self.events.append({"step": step, "agent_id": agent.agent_id,
                                "role": agent.role, "name": agent.name,
                                "action_type": "PARSE_FAIL", "tag": tag,
                                "outcome": {"kind": "parse_fail",
                                            "reason": "unparseable_response"},
                                "raw": (raw or "")[:400]})
            # 400字で切ると打切りの原因が追えないので、全文を別ログに残す。
            self.parse_fail_raw.append({"step": step, "agent_id": agent.agent_id,
                                        "tag": tag, "raw": raw or ""})
            return None
        if act.get("_truncated"):
            self.truncated_count += 1
        return act

    def _v5d_decode(self, act: Dict[str, Any]) -> None:
        """呼び名で返ってきた相手を内部IDへ戻す。

        スキーマの構造は v5 と同一で、enum に並ぶ値が呼び名になっただけである
        （docs/world_design_v5d.md §1-3）。
        """
        to = str(act.get("direct_to", "") or "").strip()
        if to and to != DIRECT_NONE_V5:
            act["direct_to"] = self.agent_by_name.get(to, to)
        if act.get("talk_to"):
            act["talk_to"] = [self.agent_by_name.get(str(t), str(t))
                              for t in act["talk_to"]]

    def _step_v5(self, step: int) -> None:
        ensure_v5_state(self.ledger)
        acquirer_id = self.acquirer_ids[0] if self.acquirer_ids else ""

        # --- 0) 台本どおりに名義が移る（LLMは関与しない） --------------------
        old_names = {p.pid: (p.registered_name or self.names.get(p.owner_id, p.owner_id))
                     for p in self.ledger.parcels.values()}
        # v5e: 自衛の具体的な行動が観測された翌月から、買い手は台本の取得を止める
        # （買い手側の反応であって、主体には一切知らされない＝docs/world_design_v5e.md §3）。
        if (self.field_v5e and self.defense_stop
                and step >= int(self.defense_stop["stop_from_month"])):
            for acq in acquisitions_at(self.script, step):
                self.v5e_suspended += 1
                self.v5e_suspended_ids.append(str(acq["id"]))
                self.ledger._rec(step, "script_suspended", acq_id=acq["id"],
                                 parcel_id=acq["parcel_id"],
                                 reason="defense_detected")
                self.ledger.v5_deals.append(
                    {"step": step, "kind": "script_stopped", "acq_id": acq["id"],
                     "parcel_id": acq["parcel_id"], "reason": "defense_detected"})
            done_list = []
        elif self.field_v6:
            # v6: 予定の土地が「当面売らない」なら、その月は買えず次の予定へ移る。
            # 順番も月も台本のまま＝**買えなかった取得は後で買い直さない**
            # （買い手に新しい行動を足さない＝docs/world_design_v6_two_worlds.md §3-2）。
            blocked = blocked_acquisitions_v6(self.ledger, step, self.script)
            for acq in blocked:
                self.v6_blocked_acq_ids.add(str(acq["id"]))
                row = {"step": step, "acq_id": str(acq["id"]),
                       "parcel_id": str(acq["parcel_id"]),
                       "parcel": self.pnames.get(str(acq["parcel_id"]),
                                                 str(acq["parcel_id"])),
                       "kind": str(acq.get("kind", "sale")),
                       "under_name": str(acq.get("under_name", "")),
                       "reason": "refusal"}
                self.v6_blocked.append(row)
                self.ledger._rec(step, "script_blocked", acq_id=acq["id"],
                                 parcel_id=acq["parcel_id"], reason="refusal")
                self.ledger.v5_deals.append(
                    {"step": step, "kind": "script_blocked", "acq_id": acq["id"],
                     "parcel_id": acq["parcel_id"], "reason": "refusal"})
            done_list = apply_script_v5(
                self.ledger, step,
                script_without(self.script, self.v6_blocked_acq_ids), acquirer_id)
            self.v6_applied += len(done_list)
        else:
            done_list = apply_script_v5(self.ledger, step, self.script, acquirer_id)
            self.v5e_applied += len(done_list)
        for done in done_list:
            self.events.append({"step": step, "agent_id": "SCRIPT", "role": "script",
                                "name": "台本",
                                "action_type": ("scripted_lease"
                                                if done.get("kind") == "lease"
                                                else "scripted_transfer"),
                                "outcome": {"kind": done.get("kind", "transfer"),
                                            "acq_id": done.get("acq_id"),
                                            "parcel_id": done.get("parcel_id"),
                                            "seller": done.get("seller"),
                                            "lessor": done.get("lessor"),
                                            "under_name": done.get("under_name")}})

        # --- 1) 会場に依らない兆候を配る ------------------------------------
        ambient = ambient_traces_v5(self.ledger, step, self.script, old_names, acquirer_id,
                                    self.pnames or None,
                                    TRACE_TEXTS_V5D if self.field_v5d else None)
        for a in self.actors:
            a.extra["traces"] = [tr for tr in ambient.get(a.agent_id, [])
                                 if self._v5e_trace_alive(tr)]
            a.extra["heard"] = []
            a.extra["directs_used"] = 0
            a.extra["articles_used"] = 0
            for tr in a.extra["traces"]:
                self.ledger.v5_traces_seen.append(
                    {"step": step, "agent_id": a.agent_id, "scene": "", "venue": "", **tr})

        kind4, venue4, label4 = (s4_for_step_v5d(step) if self.field_v5d
                                 else s4_for_step(step))

        # --- 2) 計画コール（月1回・全主体） ---------------------------------
        items = []
        for a in self.actors:
            prompt = build_plan_prompt_v5(a, self.ledger, step, self.n_steps, self.names,
                                          a.extra["traces"], label4, venue4,
                                          prompt_order=self.prompt_order,
                                          pnames=self.pnames or None)
            choices = self.venue_choices[a.agent_id]
            if self.field_v5d:
                schema = plan_schema_v5([self.venue_labels[v] for v in choices],
                                        self.venue_labels[venue4])
            else:
                schema = plan_schema_v5(choices, venue4)
            items.append((a, self.system_prompts[a.agent_id], prompt, schema))
        results = self._call_batch(items, "plan", job_key=f"m{step:02d}_plan")
        for a in self.actors:
            a.inbox = []      # 観測を作り終えた直後に空にする（以降は翌月ぶん）

        plans: Dict[str, Dict[str, str]] = {}
        for a in sorted(self.actors, key=lambda x: x.agent_id):
            act = self._v5_parse(a, results, step, "plan")
            plan = {sid: HOME_V5 for sid in SCENE_IDS}
            if act is not None:
                self._v5_thought(a, act, step, "plan", "")
                for sid in SCENE_IDS:
                    value = str(act.get(f"plan_{sid.lower()}", "") or "").strip()
                    if self.field_v5d:
                        value = self.venue_by_label.get(value, value)
                    allowed = ([venue4] if sid == "S4"
                               else self.venue_choices[a.agent_id])
                    if value in allowed:
                        plan[sid] = value
                    elif value and value != HOME_V5:
                        self.invalid_count += 1
                        self.ledger._rec(step, "invalid_location", by=a.agent_id,
                                         scene=sid, given=value)
            plans[a.agent_id] = plan
            self.ledger.v5_plans.append({"step": step, "agent_id": a.agent_id, **plan})

        # 月内の出席をここで確定させる（計画は決まっている）。
        # 「その主体の今月最後のターン」が分かると、姿勢と記事をそこ1回に絞れる。
        attend: Dict[str, List[str]] = {}
        for sid in SCENE_IDS:
            venues: Dict[str, List[str]] = {}
            for a in self.actors:
                venue = plans[a.agent_id][sid]
                if venue != HOME_V5:
                    venues.setdefault(venue, []).append(a.agent_id)
            for venue, members in venues.items():
                if len(members) < 2:
                    continue
                for aid in members:
                    attend.setdefault(aid, []).append(sid)
        last_scene = {aid: scenes[-1] for aid, scenes in attend.items() if scenes}

        # --- 3) シーン（同席者だけの会話・各2ラウンド） ----------------------
        for sid in SCENE_IDS:
            scene_label = SCENE_LABELS.get(sid, label4)
            groups: Dict[str, List[Agent]] = {}
            for a in self.actors:
                venue = plans[a.agent_id][sid]
                if venue == HOME_V5:
                    continue
                groups.setdefault(venue, []).append(a)
            # その場に居れば見えるもの（兆候・登記の閲覧）は、話し相手が居なくても見える。
            # 会話が成立するかどうかとは別（Codexレビュー 2026-08-27）。
            attendance = dict(groups)
            for venue_id, members in sorted(attendance.items()):
                for tr in [t for t in venue_traces_v5(
                        step, self.script, venue_id, old_names, self.pnames or None,
                        TRACE_TEXTS_V5D if self.field_v5d else None)
                        if self._v5e_trace_alive(t)]:
                    for a in members:
                        a.extra["traces"].append(tr)
                        self.ledger.v5_traces_seen.append(
                            {"step": step, "agent_id": a.agent_id, "scene": sid,
                             "venue": venue_id, **tr})
            # v5d には窓口が無い（S4 に counter が来ない）＝registry_rows_v5 は呼ばれない。
            registry = (registry_rows_v5(self.ledger, step)
                        if (sid == "S4" and kind4 == "counter" and not self.field_v5d)
                        else None)
            if registry:
                # 誰がどの名義変更を窓口で見たかを、取得ごとに観測へ残す。
                viewed = [r for r in self.ledger.records
                          if r.get("kind") == "transfer" and r.get("step", 0) <= step]
                for venue_id, members in sorted(attendance.items()):
                    for a in members:
                        a.extra["registry_seen"] = True
                        for rec in viewed:
                            self.ledger.v5_traces_seen.append(
                                {"step": step, "agent_id": a.agent_id, "scene": sid,
                                 "venue": venue_id, "kind": "registry_lookup",
                                 "acq_id": rec.get("acq_id", ""),
                                 "parcel_id": rec.get("parcel_id", ""),
                                 "audience": "registry",
                                 "text": (f"窓口で見た: 第{rec['step']}月 "
                                          f"{rec['parcel_id']} の名義が "
                                          f"{rec.get('under_name', '')} に変わっていた")})
            groups = {v: m for v, m in attendance.items() if len(m) >= 2}
            if not groups:
                continue

            for rnd in range(1, self.scene_rounds + 1):
                items = []
                ctx: Dict[str, Tuple[str, List[str]]] = {}
                for venue_id, members in sorted(groups.items()):
                    present = sorted(m.agent_id for m in members)
                    for a in members:
                        final_turn = (last_scene.get(a.agent_id) == sid
                                      and rnd == self.scene_rounds)
                        owns = self._v5_owns(a) and final_turn
                        can_publish = a.role == "media" and final_turn
                        # v6: 行動欄は「その月の最後のターン」でだけ尋ねる（stance と同じ）。
                        is_muni = a.role == "municipality"
                        act_rows = (action_rows_v6(owns, is_muni)
                                    if (self.field_v6 and final_turn) else None)
                        prompt = build_scene_prompt_v5(
                            a, self.ledger, step, self.n_steps, self.names, sid,
                            scene_label, self.venue_labels.get(venue_id, venue_id),
                            present, a.extra["traces"], a.extra["heard"], rnd,
                            self.scene_rounds, registry, owns, can_publish,
                            self.direct_quota - a.extra.get("directs_used", 0),
                            self.article_quota - a.extra.get("articles_used", 0),
                            prompt_order=self.prompt_order,
                            pnames=self.pnames or None,
                            action_rows=act_rows)
                        if self.field_v6:
                            schema = scene_schema_v6(
                                a.name, [self.names[p] for p in present],
                                [self.names[i] for i in self.actor_ids],
                                owns, can_publish, ask_actions=final_turn,
                                is_municipality=is_muni)
                        elif self.field_v5d:
                            schema = scene_schema_v5d(
                                a.name, [self.names[p] for p in present],
                                [self.names[i] for i in self.actor_ids],
                                owns, can_publish)
                        else:
                            schema = scene_schema_v5(a, present, self.actor_ids,
                                                     owns, can_publish)
                        items.append((a, self.system_prompts[a.agent_id], prompt, schema))
                        ctx[a.agent_id] = (venue_id, present)
                results = self._call_batch(items, f"{sid}r{rnd}",
                                           job_key=f"m{step:02d}_{sid}r{rnd}")

                spoken: List[Dict[str, Any]] = []
                for aid in sorted(ctx):
                    a = self.by_id[aid]
                    venue_id, present = ctx[aid]
                    act = self._v5_parse(a, results, step, f"{sid}r{rnd}")
                    if act is None:
                        continue
                    if self.field_v5d:
                        self._v5d_decode(act)
                    self._v5_thought(a, act, step, sid, venue_id, rnd)
                    self._v5_stance(a, act, step, sid)
                    if last_scene.get(aid) == sid and rnd == self.scene_rounds:
                        # v6: 行動欄はその月の最後のターンでだけ受け取る。
                        # 他のターンに行動のキーが混ざって返っても無視する
                        # （Codexレビュー 2026-08-29 走行前指摘）。
                        self._v6_actions(a, act, step, sid, venue_id)
                    self._v5_direct(a, act, step, sid)
                    if a.role == "media":
                        self._v5_publish(a, act, step, sid, rnd)
                    text = str(act.get("text", "") or "").strip()
                    raw_to = [str(t) for t in (act.get("talk_to") or [])]
                    talk_to = [t for t in raw_to if t in present and t != aid]
                    for t in raw_to:
                        if t not in talk_to:
                            self.invalid_count += 1
                            self.ledger._rec(step, "talk_to_rejected", by=aid,
                                             given=t, scene=sid, venue=venue_id,
                                             reason=("self" if t == aid
                                                     else "not_present"))
                    self.events.append({"step": step, "agent_id": aid, "role": a.role,
                                        "name": a.name, "action_type": "utterance",
                                        "scene": sid, "venue": venue_id, "round": rnd,
                                        "text": text, "talk_to": talk_to,
                                        "truncated": bool(act.get("_truncated")),
                                        "outcome": {"kind": "spoke" if text else "silent",
                                                    "heard_by": len(present) - 1}})
                    if not text:
                        continue
                    self.ledger.v5_utt_seq += 1
                    row = {"utt_id": f"U{self.ledger.v5_utt_seq:05d}", "step": step,
                           "scene": sid, "venue": venue_id, "round": rnd,
                           "from": aid, "role": a.role, "name": a.name,
                           "text": text, "talk_to": talk_to,
                           "heard_by": [p for p in present if p != aid]}
                    self.ledger.v5_utterances.append(row)
                    self.all_utterances.append({"step": step, "from": aid, "role": a.role,
                                                "name": a.name, "text": text})
                    spoken.append(row)

                # ラウンド末に、その場に居た全員へ全文を配送する
                for row in spoken:
                    for pid_ in ctx[row["from"]][1]:
                        self.by_id[pid_].extra.setdefault("heard", []).append(
                            {"from": row["from"], "text": row["text"],
                             "scene": sid, "venue": row["venue"],
                             "venue_label": self.venue_labels.get(row["venue"],
                                                                  row["venue"]),
                             "talk_to": row["talk_to"], "step": step})
                        if pid_ != row["from"]:
                            self.deliveries.append(
                                {"step": step, "to": pid_, "from": row["from"],
                                 "kind": "scene", "location": row["venue"],
                                 "obs_id": row["utt_id"], "text": row["text"][:200]})

        if self.field_v5e:
            self._v5e_month_end(step)

    # 買い手が現実に観測できる行（人の目に触れたもの）。内心は入らない。
    # **私信（direct）は入れない**（Codexレビュー 2026-08-29 指摘）。私信は今も分類対象外だが、
    # ここに残しておくと、将来分類対象に足しただけで停止条件まで黙って広がる。
    # 私信を停止に入れるかは、その時に別途 施主決定を取る。
    V5E_OBSERVABLE_KINDS = ("utterance", "article")

    # 台本の取得を止めるレベル（施主決定 2026-08-29 07:50）。
    # 「個人的な売買の拒否は検出しちゃダメ」＝S1（自分は売らない・貸さない）は
    # **どれだけ出ても買い手は止まらない**。止まるのは S2（特定の買い手についての
    # 広範囲な呼びかけ・周知）と S3（行政の禁止／差し止め措置）だけである。
    # S1 は観測・集計には従来どおり残す（落とすのは停止の判定だけ）。
    V5E_STOP_LEVELS = ("S2", "S3")

    def _v5e_role_of(self, agent_id: Any) -> str:
        """主体IDから役割を引く（居なければ空）。記事の行の役割を補うために使う。"""
        a = self.by_id.get(agent_id)
        return str(getattr(a, "role", "") or "") if a is not None else ""

    def _v5e_red_level(self, r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """分類器の1行が赤なら自衛レベルを返す（赤でなければ None）。

        **停止判定と最終集計はこの1か所だけを使う**（同じ組み立てを2度書くと
        いつか片方だけ直されてズレる＝Codexレビュー 2026-08-29 指摘）。
        """
        text = str(r.get("text", ""))
        if not (bool(r.get("classified")) and bool(r.get("defense"))):
            return None
        hs, ps = self._v5e_holders_acquired(int(r.get("step", 0) or 0))
        blue = bool(self._v5e_blue(text, hs, ps))
        role = r.get("role") or ""
        if not rule_red_v5e(text, blue, role):
            return None
        return defense_level_of(
            {"classified": True, "rule_red": True, "rule_yellow": False,
             "rule_green": False, "rule_blue": blue, "llm_defense": True,
             "llm_defense_level": r.get("defense_level"), "text": text,
             "role": role})

    def _v5e_stop_trigger(self, step: int,
                          r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """その1行が「台本の取得を止める行」なら記録用の dict を返す。

        止まる条件は2つだけで、**どちらも観測側の分類器の出力から決まる**
        （別の仕掛け・別の閾値を作らない＝docs/world_design_v5e.md §3）：

        1. **買い手が現実に観測できる行**であること（発話・記事・私信）。
           内心では止まらない＝他人の頭の中は誰にも見えない。
        2. その行の自衛レベルが **S2 または S3** であること（施主決定 2026-08-29 07:50
           「個人的な売買の拒否は検出しちゃダメ」）。S1 は何件出ても止まらない。

        **「観測できる」の意味**（走行前に確定・Codexレビュー 2026-08-29）：ここで見ているのは
        「その買い手がその行を実際に聞いたか」ではなく、**街に人の目に触れる形で表出したか**
        である（買い手には受信の経路が無い＝台本だから）。したがってこれは
        「街に S2/S3 が表出したら買い手が手を引く」という**撤退の方針**であって、
        買い手の観測の再現ではない。報告にもそう書く。

        **レベルが割れたとき**：主値は LLM の `defense_level`（`defense_level_of`）なので、
        ルール側が S1・LLM 側が S2 なら**止まる**。両方の値をトリガー行に残して監査する。

        停止判定と集計が同じ組み立てを使うための唯一の持ち場である。
        """
        if r.get("kind") not in self.V5E_OBSERVABLE_KINDS:
            return None
        lv = self._v5e_red_level(r)
        if lv is None or lv.get("level") not in self.V5E_STOP_LEVELS:
            return None
        return {"step": step, "from": r.get("from", ""),
                "role": r.get("role", ""), "kind": r.get("kind", ""),
                "scene": r.get("scene", ""), "venue": r.get("venue", ""),
                "text": str(r.get("text", "")), "level": lv.get("level"),
                "level_source": lv.get("level_source"),
                # 監査用に LLM 側とルール側のレベルを両方残す（Codexレビュー 2026-08-29）。
                # 主値は LLM 側なので、rule=S1 / llm=S2 の行でも止まる。
                # 走行前に決めた仕様であり、結果を見てから変えない。
                "llm_level": lv.get("llm_level"), "rule_level": lv.get("rule_level")}

    def _v5e_trace_alive(self, tr: Dict[str, Any]) -> bool:
        """起きなかった取得の兆候を配らない（v5e で買い手が止まったあと）。

        兆候は「その取得が現実に生む痕跡」なので、取得そのものが成立しない以上、
        痕跡も存在しない。これを残すと、**起きていない売買の証拠**を住民が
        見せられることになる（世界が住民に嘘をつく）。CTO 判断 2026-08-29。

        止まる前の月に既に配られた準備の痕跡（測量など）はそのままで良い＝
        その時点では買い手は現実に動いていた。落とすのは停止月以降だけである。

        v6 も同じ理屈で、**買えなかった取得の兆候をその月以降配らない**
        （起きなかった売買の痕跡を街に見せない＝世界が住民に嘘をつかない）。
        買えなかった月より前に配られた準備の痕跡はそのままで良い。
        """
        if self.field_v6:
            return str(tr.get("acq_id", "")) not in self.v6_blocked_acq_ids
        if not (self.field_v5e and self.defense_stop):
            return True
        month = self._v5e_acq_month.get(tr.get("acq_id"))
        if month is None:
            return True
        return int(month) < int(self.defense_stop["stop_from_month"])

    def _v5e_blue(self, text, hs, ps) -> bool:
        """青のルール1次抽出（`tools/run_metrics.py` の実装を1回だけ読み込む）。"""
        if getattr(self, "_v5e_blue_fn", None) is None:
            self._v5e_blue_fn = _v5e_blue_rule()
        return bool(self._v5e_blue_fn(text, hs, ps))

    def _v5e_holders_acquired(self, step: int):
        """その月までに成立した名義と区画（判定に未来の取得を混ぜない）。

        `tools/run_metrics.py` の `holders_by_step` / `acquired_by_step` と
        **同じ作り方**（台帳の transfer / lease の記録から採る）。
        """
        hs, ps = set(), set()
        for r in self.ledger.records:
            if r.get("kind") not in ("transfer", "lease"):
                continue
            if int(r.get("step", 0) or 0) > step:
                continue
            hs.add(str(r.get("under_name", "")))
            ps.add(r.get("parcel_id"))
        return hs, ps

    def _v5e_month_end(self, step: int) -> None:
        """その月の発話・内心・記事を観測側と同じ分類器で読む（v5e だけ）。

        結果は最終の `stage_labels_v5e.jsonl` にそのまま使う（事後に分類し直さない＝
        同じ入力に二重課金しない・判定が二本に割れない）。赤が1行でも立ったら
        翌月以降の台本の取得を止める。**主体には一切見せない**（inbox・traces・
        プロンプトのどこにも入らない）。
        """
        kcfg = self.cfg.get("kpi", {})
        if not kcfg.get("classify_utterances", True):
            return
        rows = ([{"kind": "utterance", **u} for u in self.ledger.v5_utterances
                 if int(u.get("step", 0)) == step]
                + [{"kind": "thought", **t} for t in self.thoughts
                   if int(t.get("step", 0)) == step]
                # 記事の行には役割が入っていない（`v5_articles` は from だけ）。
                # 事後集計（tools/run_metrics.py）は `venue_choices` から書き手の
                # 役割を補って判定しているので、走行中も同じように補わないと
                # **同じ行が走行中と事後で違う判定になる**（S3 は行政の主体に限る）。
                + [{**a, "kind": "article",
                    "role": a.get("role") or self._v5e_role_of(a.get("from"))}
                   for a in self.ledger.v5_articles
                   if int(a.get("step", 0)) == step])
        if not rows:
            return
        labels = classify_stage_v5e(self.client, rows,
                                    batch=int(kcfg.get("classify_batch", 25)),
                                    job_key=f"m{step:02d}_stage")
        self.stage_labels_v5e.extend(labels)

        # **停止のトリガーは「観測できる行の S2/S3」だけ**（`_v5e_stop_trigger`）。
        # 内心では止まらない・S1（個人の売買拒否）では止まらない。
        triggers = [t for t in (self._v5e_stop_trigger(step, r) for r in labels)
                    if t is not None]
        if triggers and self.defense_stop is None:
            self.defense_stop = {"stop_from_month": step + 1, "trigger_month": step,
                                 "triggers": triggers}

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
            "latency_sec": _latency_sec(result),
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
        # v6 は**採点用の LLM を持たない世界**である（施主の絶対原則）。
        # config の設定に関わらずコード側で遮断する＝設定の事故で分類器が
        # 復活しない（Codexレビュー 2026-08-29 走行前指摘）。
        # v1〜v5e2 では従来どおり kpi.classify_utterances がそのまま効く。
        classify_on = (not self.field_v6) and bool(kcfg.get("classify_utterances", True))
        targets = [u for u in self.all_utterances
                   if u["role"] in ("household", "business")]
        if self.field_v5:
            # v5 は仲介・行政・記者も同じ場で話す（会話が観測の本体）ので全主体を分類する。
            targets = list(self.all_utterances)
        pub_steps = None
        if classify_on and targets:
            classified = classify_utterances(
                self.client, targets, batch=int(kcfg.get("classify_batch", 25)),
                job_key="classify_utterances")
        if classify_on and self.ledger.publications:
            pub_steps = classify_publications(
                self.client, self.ledger.publications,
                batch=int(kcfg.get("classify_batch", 25)))
        classified_thoughts: List[Dict[str, Any]] = []
        if ((self.field_v41 or self.field_v5)
                and classify_on and self.thoughts):
            targets_thoughts = [t for t in self.thoughts
                                if t["role"] in ("household", "business")]
            classified_thoughts = classify_utterances(
                self.client, targets_thoughts,
                batch=int(kcfg.get("classify_batch", 25)),
                job_key="classify_thoughts")
        classified_feelings: List[Dict[str, Any]] = []
        if self.field_v4 and classify_on and self.feelings:
            classified_feelings = classify_utterances(
                self.client, self.feelings, batch=int(kcfg.get("classify_batch", 25)),
                job_key="classify_feelings")
        # v4 の認知は「毎月の実感（feeling）」の事後分類から測る。
        cog = cognition_series(
            classified_thoughts or classified_feelings or classified, self.n_steps)

        # 「占領の認知」と「4段階の色」の事後分類は **summary より前に** 済ませる。
        # 分類器の消費と打切りを usage / max_token_finishes に含めるため
        # （Codexレビュー 2026-08-28：分類前に summary を確定していて費用が過少計上だった）。
        # どちらも走行中の主体には一切見せず、世界には戻らない。
        occ_rows: List[Dict[str, Any]] = []
        occ_labels: List[Dict[str, Any]] = []
        stage_labels: List[Dict[str, Any]] = []
        if self.field_v5:
            occ_rows = ([{"kind": "utterance", **u} for u in self.ledger.v5_utterances]
                        + [{"kind": "thought", **t} for t in self.thoughts]
                        + [{"kind": "article", **a} for a in self.ledger.v5_articles])
            # v5b 比較用の占領分類器。config で切れる（既定は従来どおり on）。
            # off のときは occupation_labels.jsonl を書かない＝O1〜O4 は
            # 「未計測」として集計から落ちる（欠損を false に化けさせない）。
            if ((self.field_v5b or self.v5c_like)
                    and classify_on
                    and kcfg.get("classify_occupation", True) and occ_rows):
                occ_labels = classify_occupation(
                    self.client, occ_rows, batch=int(kcfg.get("classify_batch", 25)))
            if (self.v5c_like and not self.field_v5e
                    and classify_on and occ_rows):
                # 定義は走行前に固定（docs/world_design_v5c_buyer_strategy.md §1）。
                stage_labels = classify_stage_v5c(
                    self.client, occ_rows, batch=int(kcfg.get("classify_batch", 25)))

        summary = {
            "run_name": self.cfg.get("run_name"),
            "steps": self.n_steps,
            "agents": {r: sum(1 for a in self.agents if a.role == r)
                       for r in sorted({a.role for a in self.agents})},
            "parcels": len(self.ledger.parcels),
            "parcels_tradable": len([p for p in self.ledger.parcels.values()
                                     if p.use != "public"]),
            "model": getattr(self.client, "model", "?"),
            "provider": self.cfg["llm"].get("provider"),
            "elapsed_sec": round(elapsed, 1),
            "acquirer_model": getattr(self.acquirer_client, "model", "?"),
            "invalid_actions": self.invalid_count,
            "truncated_responses": self.truncated_count,
            "max_token_finishes": getattr(self.client, "max_token_finishes", 0),
            "usage": self.usage.as_dict(),
            # 節約設定の実測（既定では全部 従来どおりの値になる）
            "saving": {
                "prompt_order": self.prompt_order,
                "agent_max_tokens": self.agent_max_tokens,
                "enable_cache": bool(self.cfg["llm"].get("enable_cache", False)),
                "batch_kinds": sorted(getattr(self.client, "batch_kinds", set())),
                "cache_created": getattr(self.client, "cache_created", 0),
                "cache_failed": getattr(self.client, "cache_failed", 0),
                "batch_fallback_calls": getattr(self.client, "batch_fallback_calls", 0),
                "batch_jobs": list(getattr(self.client, "batch_stats", [])),
            },
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

        if self.field_v5e:
            # v5e: 自衛の観測と、それに対する買い手の反応（台本の停止）の記録。
            # 「出なかった」も同じ重みで残す（docs/world_design_v5e.md §3）。
            levels_seen = sorted({lv["level"]
                                  for lv in map(self._v5e_red_level,
                                                self.stage_labels_v5e)
                                  if lv and lv.get("level")})
            stop = self.defense_stop
            summary["v5e"] = {
                "stopped": bool(stop),
                "trigger_month": stop["trigger_month"] if stop else None,
                "stop_from_month": stop["stop_from_month"] if stop else None,
                "trigger_count": len(stop["triggers"]) if stop else 0,
                "levels_seen": levels_seen,
                "acquisitions_applied": self.v5e_applied,
                "acquisitions_suspended": self.v5e_suspended,
            }

        if self.field_v6:
            # v6: 行動の時計と、買い手が買えなかった件数。全部決定論（採点は無い）。
            # 「出なかった」も同じ重みで残す（None は「一度も無かった」の意味）。
            def _first(rows, pred):
                ms = [int(r.get("step", 0)) for r in rows if pred(r)]
                return min(ms) if ms else None

            acts = self.v6_actions
            papers = self.v6_papers
            by_act = {}
            for value in (PUBLIC_ACT_VALUES[1:] + MEASURE_VALUES[1:]):
                rows = [p for p in papers if p.get("act") == value]
                by_act[value] = {"count": len(rows),
                                 "first_month": _first(rows, lambda r: True)}
            summary["v6"] = {
                "acquisitions_applied": self.v6_applied,
                "acquisitions_blocked": len(self.v6_blocked),
                "blocked": self.v6_blocked,
                "refusal_first_month": _first(
                    acts, lambda r: r.get("value") == SELL_INTENT_REFUSE),
                "refusal_agents": sorted({r["agent_id"] for r in acts
                                          if r.get("value") == SELL_INTENT_REFUSE}),
                "refusal_parcels_final": sorted(
                    p.pid for p in self.ledger.parcels.values()
                    if getattr(p, "refusal", False)),
                "clear_count": len([r for r in acts
                                    if r.get("value") == SELL_INTENT_CLEAR]),
                "paper_first_month": _first(papers, lambda r: True),
                "papers_total": len(papers),
                "by_act": by_act,
                "measure_first_month": _first(
                    papers, lambda r: r.get("act") in MEASURE_VALUES[1:]),
                "action_rows": len(acts),
            }

        # 自分が作った明示キャッシュを片付ける（保存料は保持時間で課金されるため）。
        try:
            closed = self.client.close_caches()
            if closed:
                summary["saving"]["cache_closed"] = closed
        except Exception:  # 片付けの失敗で走行結果を落とさない（TTLで消える）
            pass

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
        if self.field_v5:
            # 「占領の認知」の事後分類。発話と内心の両方を対象にする（走行中の主体には
            # 一切見せない・世界には戻らない）。判定の定義は実装仕様6章で走行前に固定。
            if occ_labels:
                _write_jsonl(os.path.join(d, "occupation_labels.jsonl"), occ_labels)
            if self.field_v5e:
                _write_jsonl(os.path.join(d, "stage_labels_v5e.jsonl"),
                             self.stage_labels_v5e)
                stop = self.defense_stop or {"stopped": False, "months": self.n_steps}
                if self.defense_stop:
                    stop = {"stopped": True, "months": self.n_steps,
                            **self.defense_stop}
                with open(os.path.join(d, "defense_stop_v5e.json"), "w",
                          encoding="utf-8") as f:
                    json.dump(stop, f, ensure_ascii=False, indent=2)
            if self.v5c_like:
                if stage_labels:
                    _write_jsonl(os.path.join(d, "stage_labels_v5c.jsonl"), stage_labels)
                _write_jsonl(os.path.join(d, "venue_choices_v5c.jsonl"),
                             [{"agent_id": a.agent_id, "role": a.role, "name": a.name,
                               "venues": self.venue_choices.get(a.agent_id, [])}
                              for a in self.actors])
            if self.field_v6:
                _write_jsonl(os.path.join(d, "actions_v6.jsonl"), self.v6_actions)
                _write_jsonl(os.path.join(d, "papers_v6.jsonl"), self.v6_papers)
                _write_jsonl(os.path.join(d, "blocked_v6.jsonl"), self.v6_blocked)
            _write_jsonl(os.path.join(d, "deals_v5.jsonl"), self.ledger.v5_deals)
            _write_jsonl(os.path.join(d, "thoughts.jsonl"),
                         classified_thoughts or self.thoughts)
            _write_jsonl(os.path.join(d, "thoughts_all.jsonl"), self.thoughts)
            _write_jsonl(os.path.join(d, "utterances_v5.jsonl"), self.ledger.v5_utterances)
            _write_jsonl(os.path.join(d, "traces_v5.jsonl"), self.ledger.v5_traces_seen)
            _write_jsonl(os.path.join(d, "plans_v5.jsonl"), self.ledger.v5_plans)
            _write_jsonl(os.path.join(d, "stances_v5.jsonl"), self.ledger.v5_stances)
            _write_jsonl(os.path.join(d, "articles_v5.jsonl"), self.ledger.v5_articles)
            _write_jsonl(os.path.join(d, "directs_v5.jsonl"), self.ledger.v5_directs)
            _write_jsonl(os.path.join(d, "parse_fail_raw.jsonl"), self.parse_fail_raw)
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
