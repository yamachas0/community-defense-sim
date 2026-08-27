"""A市フィールドv4.1b: 金額は無いまま「仲介に相談する経路」だけを世界へ戻した版。

設計の正は `docs/world_design_v4_1b_broker_consult.md`。
**v1〜v4.1 のファイル（`src/field_v4_1.py` を含む）は変更しない。** ここは追加だけを持つ。

v4.1 との差は2つだけ：
  1. 所有者（住民・事業者）が仲介へ**相談できる**（`consult`・月1件）。
     仲介はその相談に**答えられる**（`advise`・月 `broker_monthly_advice_capacity` 件）。
     相談も回答も世界の配送を通り、相手が読むのは翌月。双方の観測に機械記録として状態が出る。
     取引経路（取次ぎ・手数料）と打診の文面は戻していない＝金額のない世界のまま。
  2. 行政の観測に、A市の区画面積の分布と、施行中の条例の適用実績（＝空振りの事実）を機械記録で足す。

ここに置いてよいのは可能な行為・有限資源・配送・記録だけである。
売却確率・閾値・強制イベント・台本・当為は置かない。
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from .agents import Agent
from .field_v4_1 import (MAX_TEXT_CHARS, active_ordinance_v41,
                         build_phase1_prompt_v41, build_phase2_prompt_v41,
                         build_system_prompt_v41, ensure_v41_state,
                         phase1_schema_v41, phase2_schema_v41)
from .world import Ledger

CONSULT_NONE = "none"
DEFAULT_ADVICE_CAPACITY = 6

_P1_TAIL = "まず thought（内心）を書き、それを踏まえて今月の行動をJSONで1つ返す。"
_P2_TAIL = "まず thought（内心）を書き、それを踏まえて responses を返す。"


# ---------------------------------------------------------------------------
# 世界の状態
# ---------------------------------------------------------------------------

def ensure_v41b_state(ledger: Ledger) -> None:
    ensure_v41_state(ledger)
    if not hasattr(ledger, "v41b_consults"):
        ledger.v41b_consults = {}       # consult_id -> dict
        ledger.v41b_consult_seq = 0


def _normalize_consult_id(ledger: Ledger, value: Any) -> str:
    return ledger._normalize_id(value, "Q")


# ---------------------------------------------------------------------------
# 相談（所有者 → 仲介）と回答（仲介 → 所有者）
# ---------------------------------------------------------------------------

def record_consult_v41b(ledger: Ledger, step: int, agent: Agent, broker_id: str,
                        question: str,
                        broker_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """仲介に相談する。金銭の授受は無い＝相談は依頼でも契約でもない。"""
    ensure_v41b_state(ledger)
    text = str(question or "").strip()
    target = str(broker_id or "").strip()
    if agent.role not in ("household", "business"):
        return ledger._rec(step, "consult_rejected", by=agent.agent_id, to=target,
                           reason="role_cannot_consult")
    if target == agent.agent_id:
        return ledger._rec(step, "consult_rejected", by=agent.agent_id, to=target,
                           reason="self_consult")
    if not target or (broker_ids is not None and target not in broker_ids):
        return ledger._rec(step, "consult_rejected", by=agent.agent_id, to=target,
                           reason="unknown_broker")
    if not text:
        return ledger._rec(step, "consult_rejected", by=agent.agent_id, to=target,
                           reason="missing_question")
    ledger.v41b_consult_seq += 1
    consult_id = f"Q{ledger.v41b_consult_seq:04d}"
    ledger.v41b_consults[consult_id] = {
        "id": consult_id, "step": step, "from": agent.agent_id, "to": target,
        "question": text[:MAX_TEXT_CHARS], "status": "open",
        "answered_step": None, "reply": "",
    }
    return ledger._rec(step, "consult", consult_id=consult_id, by=agent.agent_id,
                       to=target, question=text[:MAX_TEXT_CHARS])


def answer_consult_v41b(ledger: Ledger, step: int, broker_id: str, consult_id: str,
                        reply: str) -> Dict[str, Any]:
    """相談に答える。答えるかどうか・何を答えるかは仲介が決める。"""
    ensure_v41b_state(ledger)
    cid = _normalize_consult_id(ledger, consult_id)
    consult = ledger.v41b_consults.get(cid)
    text = str(reply or "").strip()
    if consult is None:
        return ledger._rec(step, "advice_rejected", consult_id=cid, by=broker_id,
                           reason="no_such_consultation")
    if consult["to"] != broker_id:
        return ledger._rec(step, "advice_rejected", consult_id=cid, by=broker_id,
                           reason="not_your_consultation")
    if consult["status"] != "open":
        return ledger._rec(step, "advice_rejected", consult_id=cid, by=broker_id,
                           reason="already_answered")
    if not text:
        return ledger._rec(step, "advice_rejected", consult_id=cid, by=broker_id,
                           reason="missing_reply")
    consult["status"] = "answered"
    consult["answered_step"] = step
    consult["reply"] = text[:MAX_TEXT_CHARS]
    return ledger._rec(step, "advice", consult_id=cid, by=broker_id,
                       to=consult["from"], reply=text[:MAX_TEXT_CHARS])


# ---------------------------------------------------------------------------
# 観測（機械記録・金額なし・評価語なし）
# ---------------------------------------------------------------------------

def _consult_status_ja(consult: Dict[str, Any]) -> str:
    if consult["status"] == "answered":
        return f"回答済み(第{consult['answered_step']}月)"
    return "未回答"


def incoming_consults_rows_v41b(ledger: Ledger, broker_id: str,
                                names: Dict[str, str]) -> List[str]:
    ensure_v41b_state(ledger)
    rows = []
    for consult in sorted(ledger.v41b_consults.values(), key=lambda c: c["id"]):
        if consult["to"] != broker_id:
            continue
        rows.append(f"  [{consult['id']}] 相談者:{names.get(consult['from'], consult['from'])} "
                    f"第{consult['step']}月 状態:{_consult_status_ja(consult)} "
                    f"内容:「{consult['question']}」")
    return rows or ["  （相談は来ていない）"]


def own_consults_rows_v41b(ledger: Ledger, owner_id: str,
                           names: Dict[str, str]) -> List[str]:
    ensure_v41b_state(ledger)
    rows = []
    for consult in sorted(ledger.v41b_consults.values(), key=lambda c: c["id"]):
        if consult["from"] != owner_id:
            continue
        rows.append(f"  [{consult['id']}] 相談先:{names.get(consult['to'], consult['to'])} "
                    f"第{consult['step']}月 状態:{_consult_status_ja(consult)}")
    return rows or ["  （まだ相談していない）"]


def parcel_area_rows_v41b(ledger: Ledger, names: Dict[str, str],
                          with_names: bool = False) -> List[str]:
    """A市の区画1件あたりの面積の分布（機械記録）。誰が持っているかは含めない。"""
    areas = sorted(p.area_sqm for p in ledger.parcels.values() if p.use != "public")
    if not areas:
        return ["  （区画がない）"]
    bins: Dict[int, int] = {}
    for area in areas:
        bins[(area // 100) * 100] = bins.get((area // 100) * 100, 0) + 1
    rows = [f"  非公共区画 {len(areas)}件 最小{areas[0]}㎡ "
            f"中央値{int(statistics.median(areas))}㎡ 最大{areas[-1]}㎡ "
            f"合計{sum(areas)}㎡",
            "  1件あたり面積の分布: "
            + " / ".join(f"{low}〜{low + 99}㎡ {count}件"
                         for low, count in sorted(bins.items()))]
    if with_names:
        per_name: Dict[str, List[int]] = {}
        for parcel in ledger.parcels.values():
            if parcel.use == "public":
                continue
            key = parcel.registered_name or names.get(parcel.owner_id, parcel.owner_id)
            per_name.setdefault(key, []).append(parcel.area_sqm)
        rows.append("  名義別の1件あたり面積: "
                    + " / ".join(f"{key} {len(v)}件 最大{max(v)}㎡ 合計{sum(v)}㎡"
                                 for key, v in sorted(per_name.items(),
                                                      key=lambda kv: (-sum(kv[1]), kv[0]))))
    return rows


def ordinance_effect_rows_v41b(ledger: Ledger, step: int) -> List[str]:
    """施行中の条例が実際に何件に掛かったか（該当0件ならその事実を返す）。"""
    ordinance = active_ordinance_v41(ledger, step)
    if not ordinance:
        return ["  （施行中の届出制度はない）"]
    threshold = int(ordinance["threshold_sqm"])
    effective = int(ordinance["effective_step"])
    over = [p.pid for p in ledger.parcels.values()
            if p.use != "public" and p.area_sqm > threshold]
    total = sum(1 for p in ledger.parcels.values() if p.use != "public")
    transfers = [r for r in ledger.records
                 if r.get("kind") == "transfer" and int(r.get("step", 0)) >= effective]
    filings = [r for r in ledger.records
               if r.get("kind") == "filing_required" and int(r.get("step", 0)) >= effective]
    last = step - 1
    return [
        f"  第{effective}月施行「{ordinance['title']}」 対象:1件{threshold}㎡超の取得",
        f"  現在の登記でこの対象面積を超える区画: {len(over)}件 / 非公共{total}件",
        f"  施行後の名義移転: {len(transfers)}件 うち届出の対象になったもの: {len(filings)}件",
        f"  先月（第{last}月）の名義移転: "
        f"{sum(1 for r in transfers if int(r.get('step', 0)) == last)}件 "
        f"うち届出の対象になったもの: "
        f"{sum(1 for r in filings if int(r.get('step', 0)) == last)}件",
    ]


# ---------------------------------------------------------------------------
# 出力スキーマ（v4.1 に追加するのは相談と回答だけ）
# ---------------------------------------------------------------------------

def phase1_schema_v41b(agent: Agent, broker_ids: List[str]) -> Dict[str, Any]:
    schema = phase1_schema_v41(agent)
    props = schema["properties"]
    if agent.role in ("household", "business"):
        choices = [b for b in broker_ids if b != agent.agent_id] + [CONSULT_NONE]
        props["consult_broker_id"] = {"type": "string", "enum": choices}
        props["consult_question"] = {"type": "string"}
    elif agent.role == "broker":
        capacity = int(agent.extra.get("monthly_advice_capacity", DEFAULT_ADVICE_CAPACITY))
        item_props = {"consult_id": {"type": "string"}, "reply": {"type": "string"}}
        props["advices"] = {"type": "array", "maxItems": capacity,
                            "items": {"type": "object", "properties": item_props,
                                      "required": list(item_props)}}
    schema["required"] = list(props)
    return schema


def phase2_schema_v41b() -> Dict[str, Any]:
    return phase2_schema_v41()


# ---------------------------------------------------------------------------
# プロンプト（v4.1 の本文に、可能な行為とJSONの形だけを足す）
# ---------------------------------------------------------------------------

def build_system_prompt_v41b(agent: Agent, cfg: Dict[str, Any], n_parcels: int,
                             broker_ids: List[str]) -> str:
    text = build_system_prompt_v41(agent, cfg, n_parcels)
    if agent.role in ("household", "business"):
        choices = " / ".join(b for b in broker_ids if b != agent.agent_id)
        text += f"""
--- 今月できること（追加） ---
不動産仲介に相談する。相談先の仲介ID（{choices}）と相談内容を書く。
相談は今月のうちに相手へ渡り、相手が読むのは翌月である。答えが返るかどうかは相手が決める。
1か月に相談できるのは1件。相談しない月は consult_broker_id を "{CONSULT_NONE}"、
consult_question を空文字にする。

--- JSON出力（v4.1bの最終形） ---
thought, consult_broker_id, consult_question, location, utterance,
utterance_channel, utterance_to を必ず含める。説明文を付けずJSONだけ返す。
"""
    elif agent.role == "broker":
        capacity = int(agent.extra.get("monthly_advice_capacity", DEFAULT_ADVICE_CAPACITY))
        text += f"""
--- 今月できること（追加） ---
自分に来ている相談に答える。答えるのは自分宛の相談だけで、1件の相談に答えられるのは1回、
1か月に答えられるのは最大{capacity}件である。答えは今月のうちに相談者へ渡り、相談者が読むのは翌月である。
土地の売買を取り次ぐ経路はこの世界に無い。相談への答え以外に伝えたいことは発言を使う。

--- JSON出力（v4.1bの最終形） ---
thought, advices, location, utterance, utterance_channel, utterance_to を必ず含める。
advices の各要素は consult_id（[Q0001]の形）と reply（答えの本文）。
答えない月は advices を空配列にする。説明文を付けずJSONだけ返す。
"""
    return text


def _insert_before(text: str, tail: str, rows: List[str]) -> str:
    block = "\n".join(rows)
    index = text.rfind(tail)
    if index < 0:
        return text + "\n" + block
    return text[:index] + block + "\n" + text[index:]


def build_phase1_prompt_v41b(agent: Agent, ledger: Ledger, step: int, n_steps: int,
                             names: Dict[str, str], cfg: Dict[str, Any]) -> str:
    ensure_v41b_state(ledger)
    text = build_phase1_prompt_v41(agent, ledger, step, n_steps, names, cfg)
    rows: List[str] = []
    if agent.role in ("household", "business"):
        rows += ["[自分がした相談（機械記録）]"]
        rows += own_consults_rows_v41b(ledger, agent.agent_id, names)
    elif agent.role == "broker":
        rows += ["[自分に来ている相談（機械記録）]"]
        rows += incoming_consults_rows_v41b(ledger, agent.agent_id, names)
    elif agent.role == "municipality":
        rows += ["[A市の区画1件あたりの面積の分布（機械記録）]"]
        rows += parcel_area_rows_v41b(ledger, names,
                                      with_names=bool(agent.extra.get("registry_stats_seen")))
        rows += ["[施行中の条例の適用実績（機械記録）]"]
        rows += ordinance_effect_rows_v41b(ledger, step)
    if not rows:
        return text
    return _insert_before(text, _P1_TAIL, rows + [""])


def build_phase2_prompt_v41b(agent: Agent, ledger: Ledger, step: int, n_steps: int,
                             names: Dict[str, str], offers: List[Dict[str, Any]],
                             inbox: Optional[List[Dict[str, Any]]] = None) -> str:
    ensure_v41b_state(ledger)
    text = build_phase2_prompt_v41(agent, ledger, step, n_steps, names, offers, inbox)
    rows = ["[自分がした相談（機械記録）]"]
    rows += own_consults_rows_v41b(ledger, agent.agent_id, names)
    return _insert_before(text, _P2_TAIL, rows + [""])


__all__ = [
    "CONSULT_NONE", "DEFAULT_ADVICE_CAPACITY", "ensure_v41b_state",
    "record_consult_v41b", "answer_consult_v41b", "incoming_consults_rows_v41b",
    "own_consults_rows_v41b", "parcel_area_rows_v41b", "ordinance_effect_rows_v41b",
    "phase1_schema_v41b", "phase2_schema_v41b", "build_system_prompt_v41b",
    "build_phase1_prompt_v41b", "build_phase2_prompt_v41b",
]
