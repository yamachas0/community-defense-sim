"""A市フィールドv4.1: 金額を持たない世界（同期3フェーズ・思考レイヤー）。

設計の正は `docs/world_design_v4_1_no_money.md`。v1〜v4のファイルは変更しない。

v4との差は3つだけ：
  1. 価格・評価額・決済・資金・手数料・逆提示・賃借を**世界から削除**する。
     残るのは登記（誰の名義か）と、打診できる回数という有限資源だけ。
  2. 仲介は取引経路から外れ、噂の源としてだけ存在する（話すだけ）。
  3. JSONの先頭に `thought`（内心・自由記述・誰にも伝わらない・翌月へそのまま持ち越す）を置き、
     その後に行動を書かせる。`memory` / `memo` / `feeling` / `reasoning` は `thought` に統合した。
     指示するのは順序だけで、内心の内容と方向は指示しない。

ここに置いてよいのは可能な行為・登記・同席・配送・記録だけである。
売却確率・閾値・強制イベント・台本・当為は置かない。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .agents import Agent
from .world import Ledger, Parcel, neighbors

MAX_THOUGHT_CHARS = 400
MAX_TEXT_CHARS = 400

USE_JA = {"residential": "住宅", "shop": "店舗", "lodging": "宿泊",
          "office": "事務所", "vacant": "空地", "public": "公共施設"}

ROLE_TEXT_V41 = {
    "household": "A市で日常生活を送る住民世帯。開始時点で自分の土地・建物を所有していた。",
    "business": "A市で事業を営む地元事業者。営業、仕入れ、人員、設備、顧客対応が日常業務である。",
    "broker": "A市の不動産仲介。地域の人づきあいの中にいる。",
    "acquirer": "街の外から不動産を取得・運用する会社。非公開目的はこの主体だけが知る。",
    "municipality": "A市の自治体担当。通常業務を送り、職務上得た情報だけで判断する。",
    "media": "A市を扱う地域記者。通常の地域取材を行う。",
}

DECISION_VALUES = ["sell", "keep", "hold"]
INVESTIGATE_VALUES = ["none", "land_registry", "corporate_records"]


# ---------------------------------------------------------------------------
# 世界の状態
# ---------------------------------------------------------------------------

def ensure_v41_state(ledger: Ledger) -> None:
    if not hasattr(ledger, "v41_offers"):
        ledger.v41_offers = {}        # offer_id -> dict
        ledger.v41_offer_seq = 0
    if not hasattr(ledger, "v41_ordinance"):
        ledger.v41_ordinance = None
    if not hasattr(ledger, "v41_pending"):
        ledger.v41_pending = []       # 届出待ちの成立


def _normalize_offer_id(ledger: Ledger, value: Any) -> str:
    return ledger._normalize_id(value, "O")


# ---------------------------------------------------------------------------
# 打診（金額を伴わない取得の申し入れ）
# ---------------------------------------------------------------------------

def record_offer_v41(ledger: Ledger, step: int, agent: Agent, parcel_id: str,
                     under_name: str) -> Dict[str, Any]:
    """区画の取得を、指定した名義で申し入れる。金額は世界に存在しない。"""
    ensure_v41_state(ledger)
    allowed = [agent.name] + list(agent.extra.get("aliases", []))
    if under_name not in allowed:
        return ledger._rec(step, "offer_rejected", parcel_id=parcel_id,
                           from_id=agent.agent_id, reason="unknown_legal_name",
                           given=under_name)
    parcel = ledger.parcels.get(parcel_id)
    if parcel is None:
        return ledger._rec(step, "offer_rejected", parcel_id=parcel_id,
                           from_id=agent.agent_id, reason="no_such_parcel")
    if parcel.use == "public":
        return ledger._rec(step, "offer_rejected", parcel_id=parcel_id,
                           from_id=agent.agent_id, reason="public_land_not_for_sale")
    if parcel.owner_id == agent.agent_id:
        return ledger._rec(step, "offer_rejected", parcel_id=parcel_id,
                           from_id=agent.agent_id, reason="already_owner")
    ledger.v41_offer_seq += 1
    offer_id = f"O{ledger.v41_offer_seq:04d}"
    ledger.v41_offers[offer_id] = {
        "id": offer_id, "step": step, "parcel_id": parcel_id,
        "from": agent.agent_id, "to": parcel.owner_id, "under_name": under_name,
        "status": "open",
    }
    return ledger._rec(step, "offer", offer_id=offer_id, parcel_id=parcel_id,
                       from_id=agent.agent_id, under_name=under_name,
                       to=parcel.owner_id)


def withdraw_offer_v41(ledger: Ledger, step: int, offer_id: str,
                       by: str) -> Dict[str, Any]:
    ensure_v41_state(ledger)
    offer_id = _normalize_offer_id(ledger, offer_id)
    offer = ledger.v41_offers.get(offer_id)
    if offer is None or offer["status"] != "open" or offer["from"] != by:
        return ledger._rec(step, "withdraw_rejected", offer_id=offer_id, by=by,
                           reason="offer_not_open_or_not_yours")
    offer["status"] = "withdrawn"
    return ledger._rec(step, "withdraw", offer_id=offer_id,
                       parcel_id=offer["parcel_id"], by=by)


def owners_with_offers_v41(ledger: Ledger) -> Dict[str, List[Dict[str, Any]]]:
    """打診が開いている所有者と、その打診（応答の機会を持つのはこの主体だけ）。"""
    ensure_v41_state(ledger)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for offer in ledger.v41_offers.values():
        if offer["status"] != "open":
            continue
        parcel = ledger.parcels.get(offer["parcel_id"])
        if parcel is None:
            continue
        out.setdefault(parcel.owner_id, []).append(offer)
    for rows in out.values():
        rows.sort(key=lambda o: o["id"])
    return out


# ---------------------------------------------------------------------------
# 条例（制定されれば機構として効く）
# ---------------------------------------------------------------------------

def enact_ordinance_v41(ledger: Ledger, step: int, by: str, title: str, body: str,
                        threshold_sqm: Any, delay_months: Any) -> Dict[str, Any]:
    ensure_v41_state(ledger)
    if not str(title).strip() or not str(body).strip():
        return ledger._rec(step, "ordinance_rejected", by=by, reason="missing_text")
    try:
        threshold = int(float(threshold_sqm))
        delay = int(float(delay_months))
    except (TypeError, ValueError):
        return ledger._rec(step, "ordinance_rejected", by=by,
                           reason="invalid_parameters",
                           given=f"{threshold_sqm}/{delay_months}")
    if threshold < 0 or delay < 0:
        return ledger._rec(step, "ordinance_rejected", by=by,
                           reason="invalid_parameters", given=f"{threshold}/{delay}")
    previous = ledger.v41_ordinance
    if previous and previous["step"] == step and previous["by"] != by:
        # 同期フェーズ内に「後発」は無い。隠れた主体優先度にしないため、
        # 同月に2件制定された事実そのものを記録に残す。
        ledger._rec(step, "ordinance_same_step_conflict", by=by,
                    other=previous["by"], superseded=previous["title"],
                    effective=str(title).strip()[:80])
    ledger.v41_ordinance = {"step": step, "effective_step": step + 1, "by": by,
                            "title": str(title).strip()[:80],
                            "body": str(body).strip()[:400],
                            "threshold_sqm": threshold, "delay_months": delay}
    record = ledger.record_ordinance(step, by, str(title).strip()[:80],
                                     str(body).strip()[:400])
    record["threshold_sqm"] = threshold
    record["delay_months"] = delay
    record["effective_step"] = step + 1
    return record


def active_ordinance_v41(ledger: Ledger, step: int) -> Optional[Dict[str, Any]]:
    ensure_v41_state(ledger)
    ordinance = ledger.v41_ordinance
    if not ordinance or ordinance["effective_step"] > step:
        return None
    return ordinance


def filing_delay_v41(ledger: Ledger, parcel: Parcel, step: int) -> int:
    ordinance = active_ordinance_v41(ledger, step)
    if not ordinance:
        return 0
    if parcel.area_sqm <= ordinance["threshold_sqm"]:
        return 0
    return int(ordinance["delay_months"])


def ordinance_text_v41(ledger: Ledger, step: int) -> str:
    ordinance = active_ordinance_v41(ledger, step)
    if not ordinance:
        return "  （施行中の届出制度はない）"
    return (f"  第{ordinance['effective_step']}月施行「{ordinance['title']}」 "
            f"対象:1件{ordinance['threshold_sqm']}㎡超の取得 "
            f"届出による成立の遅延:{ordinance['delay_months']}か月")


# ---------------------------------------------------------------------------
# 応答と清算
# ---------------------------------------------------------------------------

def _close_competing_v41(ledger: Ledger, step: int, parcel_id: str,
                         keep_offer_id: str) -> None:
    for offer in ledger.v41_offers.values():
        if offer["parcel_id"] != parcel_id or offer["id"] == keep_offer_id:
            continue
        if offer["status"] in ("open", "filed"):
            offer["status"] = "void"
            ledger._rec(step, "offer_void", offer_id=offer["id"],
                        parcel_id=parcel_id, buyer=offer["from"], seller=offer["to"],
                        reason="parcel_already_transferred")
    ledger.v41_pending = [row for row in ledger.v41_pending
                          if row["offer_id"] == keep_offer_id
                          or ledger.v41_offers[row["offer_id"]]["parcel_id"] != parcel_id]


def _transfer_v41(ledger: Ledger, step: int, offer: Dict[str, Any]) -> Dict[str, Any]:
    """登記を移す。金銭の授受は世界に存在しない。"""
    parcel = ledger.parcels[offer["parcel_id"]]
    seller = parcel.owner_id
    parcel.owner_id = offer["from"]
    parcel.registered_name = offer["under_name"]
    offer["status"] = "transferred"
    record = ledger._rec(step, "transfer", offer_id=offer["id"],
                         parcel_id=parcel.pid, seller=seller, buyer=offer["from"],
                         under_name=offer["under_name"])
    _close_competing_v41(ledger, step, parcel.pid, offer["id"])
    return record


def respond_to_offer_v41(ledger: Ledger, step: int, offer_id: str, owner_id: str,
                         decision: str) -> Dict[str, Any]:
    """売る・売らない・今月は決めない。どれを選ぶかは所有者が決める。"""
    ensure_v41_state(ledger)
    offer_id = _normalize_offer_id(ledger, offer_id)
    offer = ledger.v41_offers.get(offer_id)
    if decision not in DECISION_VALUES:
        return {"kind": "invalid_action", "reason": "unknown_decision",
                "given": decision}
    if offer is None or offer["status"] != "open":
        return ledger._rec(step, "response_rejected", offer_id=offer_id, by=owner_id,
                           reason="offer_not_open")
    parcel = ledger.parcels.get(offer["parcel_id"])
    if parcel is None or parcel.owner_id != owner_id:
        return ledger._rec(step, "response_rejected", offer_id=offer_id, by=owner_id,
                           reason="not_owner")
    if offer["from"] == owner_id:
        offer["status"] = "void"
        return ledger._rec(step, "response_rejected", offer_id=offer_id, by=owner_id,
                           reason="self_offer")
    if decision == "hold":
        # 決めないことも選択である。打診は取り下げられるまで開いたまま残る。
        return ledger._rec(step, "hold", offer_id=offer_id,
                           parcel_id=offer["parcel_id"], by=owner_id)
    if decision == "keep":
        offer["status"] = "kept"
        return ledger._rec(step, "keep", offer_id=offer_id,
                           parcel_id=offer["parcel_id"], by=owner_id)
    delay = filing_delay_v41(ledger, parcel, step)
    if delay <= 0:
        return _transfer_v41(ledger, step, offer)
    offer["status"] = "filed"
    ledger.v41_pending.append({"offer_id": offer_id, "owner": owner_id,
                               "due_step": step + delay})
    return ledger._rec(step, "filing_required", offer_id=offer_id,
                       parcel_id=offer["parcel_id"], by=owner_id,
                       buyer=offer["from"], due_step=step + delay,
                       ordinance=active_ordinance_v41(ledger, step)["title"])


def execute_pending_v41(ledger: Ledger, step: int) -> List[Dict[str, Any]]:
    ensure_v41_state(ledger)
    done: List[Dict[str, Any]] = []
    remaining = []
    # 反復中に _close_competing_v41 が v41_pending を作り替えるため、
    # 開始時点のスナップショットを回し、残す行も status を再検証する。
    for row in list(ledger.v41_pending):
        offer = ledger.v41_offers.get(row["offer_id"])
        if offer is None or offer["status"] != "filed":
            continue
        if row["due_step"] > step:
            remaining.append(row)
            continue
        parcel = ledger.parcels.get(offer["parcel_id"])
        if (parcel is None or parcel.owner_id != row["owner"]
                or offer["from"] == parcel.owner_id):
            offer["status"] = "void"
            done.append(ledger._rec(step, "filing_void", offer_id=offer["id"],
                                    parcel_id=offer["parcel_id"], by=row["owner"],
                                    buyer=offer["from"], seller=row["owner"],
                                    reason="owner_changed"))
            continue
        record = _transfer_v41(ledger, step, offer)
        record["filed"] = True
        done.append(record)
    ledger.v41_pending = [row for row in remaining
                          if ledger.v41_offers.get(row["offer_id"], {}).get("status")
                          == "filed"]
    return done


# ---------------------------------------------------------------------------
# 観測（機械記録・金額なし・評価語なし）
# ---------------------------------------------------------------------------

def own_result_row_v41(step: int, action_type: str, target: str,
                       outcome: Any) -> Dict[str, Any]:
    kind = outcome.get("kind", "") if isinstance(outcome, dict) else str(outcome or "")
    reason = outcome.get("reason", "") if isinstance(outcome, dict) else ""
    if kind == "parse_fail":
        kind, reason = "not_recorded", "no_action_recorded"
        action_type = action_type or "-"
    refs: List[str] = []
    if isinstance(outcome, dict):
        for key in ("offer_id", "parcel_id"):
            value = outcome.get(key)
            if value:
                refs.append(f"{key}={value}")
    return {"step": int(step), "action_type": str(action_type or ""),
            "target": str(target or ""), "kind": str(kind),
            "reason": str(reason or ""), "refs": refs}


def own_results_text_v41(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "  （先月の記録なし）"
    out = []
    for r in rows:
        line = (f"  第{r.get('step')}月 {r.get('action_type') or '-'} "
                f"target={r.get('target') or '-'} 結果={r.get('kind') or '-'}")
        if r.get("reason"):
            line += f" 理由={r['reason']}"
        if r.get("refs"):
            line += " " + " ".join(r["refs"])
        out.append(line)
    return "\n".join(out)


def registry_rows_v41(ledger: Ledger, names: Dict[str, str]) -> List[str]:
    rows = []
    for parcel in sorted(ledger.parcels.values(), key=lambda p: p.pid):
        rows.append(f"  {parcel.pid}[{USE_JA.get(parcel.use, parcel.use)}/{parcel.block}] "
                    f"{parcel.area_sqm}㎡ "
                    f"名義:{parcel.registered_name or names.get(parcel.owner_id, parcel.owner_id)}")
    return rows


def registry_stats_rows_v41(ledger: Ledger, names: Dict[str, str]) -> List[str]:
    totals: Dict[str, int] = {}
    for parcel in ledger.parcels.values():
        if parcel.use == "public":
            continue
        key = parcel.registered_name or names.get(parcel.owner_id, parcel.owner_id)
        totals[key] = totals.get(key, 0) + parcel.area_sqm
    total = sum(totals.values()) or 1
    return [f"  {key}: {area}㎡ ({area / total:.1%})"
            for key, area in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))]


def portfolio_rows_v41(agent: Agent, ledger: Ledger, step: int,
                       n_steps: int) -> List[str]:
    tradable = [p for p in ledger.parcels.values() if p.use != "public"]
    total_area = sum(p.area_sqm for p in tradable) or 1
    owned = [p for p in tradable if p.owner_id == agent.agent_id]
    owned_area = sum(p.area_sqm for p in owned)
    by_name: Dict[str, int] = {}
    for parcel in owned:
        by_name[parcel.registered_name] = by_name.get(parcel.registered_name, 0) + parcel.area_sqm
    return [f"  所有区画 {len(owned)}/{len(tradable)}件 "
            f"面積 {owned_area}/{total_area}㎡ ({owned_area / total_area:.1%})",
            "  名義別内訳: " + (" / ".join(f"{k} {v}㎡" for k, v in sorted(by_name.items()))
                                if by_name else "なし"),
            f"  残り {n_steps - step}か月"]


def own_offers_rows_v41(agent: Agent, ledger: Ledger,
                        names: Dict[str, str]) -> List[str]:
    ensure_v41_state(ledger)
    rows = []
    for offer in sorted(ledger.v41_offers.values(), key=lambda o: o["id"]):
        if offer["from"] != agent.agent_id:
            continue
        rows.append(f"  [{offer['id']}] {offer['parcel_id']} 名義{offer['under_name']} "
                    f"第{offer['step']}月から 状態:{offer['status']}")
    return rows or ["  （まだ打診していない）"]


def incoming_offers_rows_v41(ledger: Ledger, names: Dict[str, str],
                             offers: List[Dict[str, Any]]) -> List[str]:
    rows = []
    for offer in offers:
        parcel = ledger.parcels[offer["parcel_id"]]
        line = (f"  [{offer['id']}] {parcel.pid}"
                f"[{USE_JA.get(parcel.use, parcel.use)}/{parcel.block}] "
                f"{parcel.area_sqm}㎡ 名義:{offer['under_name']} 第{offer['step']}月から")
        rows.append(line)
    return rows or ["  （届いていない）"]


def own_parcels_rows_v41(agent: Agent, ledger: Ledger,
                         names: Dict[str, str]) -> List[str]:
    rows = []
    for parcel in sorted(ledger.owned_by(agent.agent_id), key=lambda p: p.pid):
        line = (f"  {parcel.pid}[{USE_JA.get(parcel.use, parcel.use)}/{parcel.block}] "
                f"{parcel.area_sqm}㎡ 名義:{parcel.registered_name}")
        if parcel.tenant_id:
            line += f" 利用者:{names.get(parcel.tenant_id, parcel.tenant_id)}"
        rows.append(line)
    return rows or ["  （所有物件なし）"]


def neighbourhood_rows_v41(agent: Agent, ledger: Ledger, names: Dict[str, str],
                           step: int) -> List[str]:
    mine = [p.pid for p in ledger.owned_by(agent.agent_id)]
    if not mine:
        mine = [p.pid for p in ledger.parcels.values() if p.tenant_id == agent.agent_id]
    seen: List[str] = []
    for pid in mine:
        for nb in neighbors(ledger.parcels, pid):
            if nb not in seen:
                seen.append(nb)
    rows = []
    for pid in sorted(seen):
        parcel = ledger.parcels[pid]
        rows.append(f"  {pid}[{USE_JA.get(parcel.use, parcel.use)}] "
                    f"名義:{parcel.registered_name or names.get(parcel.owner_id, parcel.owner_id)}")
    for record in ledger.records:
        if (record.get("kind") == "transfer" and record.get("parcel_id") in seen
                and record.get("step", 0) >= step - 3):
            rows.append(f"  第{record['step']}月 {record['parcel_id']} の名義が "
                        f"{record['under_name']} に変わった")
    return rows or ["  （隣接区画はない）"]


def observations_rows_v41(agent: Agent, names: Dict[str, str],
                          inbox: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    items = agent.inbox if inbox is None else inbox
    if not items:
        return ["  （今月届いた情報はない）"]
    rows = []
    for item in items:
        oid = item.get("obs_id") or f"MSG-M{item.get('step', 0):02d}-{item.get('from', '')}"
        src = names.get(item.get("from", ""), item.get("from", ""))
        venue = f" @{item['location']}" if item.get("location") else ""
        rows.append(f"  [{oid}] {src}{venue}:「{str(item.get('text', ''))}」")
    return rows


# ---------------------------------------------------------------------------
# 出力スキーマ（thought が先頭・その後に行動）
# ---------------------------------------------------------------------------

_TALK_PROPS_V41: Dict[str, Any] = {
    "location": {"type": "string"},
    "utterance": {"type": "string"},
    "utterance_channel": {"type": "string", "enum": ["ambient", "direct", "none"]},
    "utterance_to": {"type": "string"},
}


def phase1_schema_v41(agent: Agent) -> Dict[str, Any]:
    props: Dict[str, Any] = {"thought": {"type": "string"}}
    if agent.role == "acquirer":
        names = list(dict.fromkeys([agent.name] + list(agent.extra.get("aliases", []))))
        capacity = int(agent.extra.get("monthly_offer_capacity", 6))
        offer_props = {
            "parcel_id": {"type": "string"},
            "under_name": {"type": "string", "enum": names},
        }
        props["offers"] = {"type": "array", "maxItems": capacity,
                           "items": {"type": "object", "properties": offer_props,
                                     "required": list(offer_props)}}
        props["withdraw"] = {"type": "array", "items": {"type": "string"}}
    elif agent.role in ("municipality", "media"):
        props["investigate"] = {"type": "string", "enum": INVESTIGATE_VALUES}
        props["publish"] = {"type": "string"}
        if agent.role == "municipality":
            props["ordinance_title"] = {"type": "string"}
            props["ordinance_text"] = {"type": "string"}
            props["ordinance_threshold_sqm"] = {"type": "integer"}
            props["ordinance_delay_months"] = {"type": "integer"}
    props.update(_TALK_PROPS_V41)
    return {"type": "object", "properties": props, "required": list(props)}


def phase2_schema_v41() -> Dict[str, Any]:
    response_props = {
        "offer_id": {"type": "string"},
        "decision": {"type": "string", "enum": DECISION_VALUES},
    }
    props = {
        "thought": {"type": "string"},
        "responses": {"type": "array",
                      "items": {"type": "object", "properties": response_props,
                                "required": list(response_props)}},
    }
    return {"type": "object", "properties": props, "required": list(props)}


# ---------------------------------------------------------------------------
# プロンプト
# ---------------------------------------------------------------------------

def build_system_prompt_v41(agent: Agent, cfg: Dict[str, Any],
                            n_parcels: int) -> str:
    world = cfg["world"]
    venues = cfg.get("social", {}).get("venues", [])
    venue_rows = "\n".join(f"  {v['id']}: {v['label']}" for v in venues)
    text = f"""あなたは架空都市「{world.get('town_name', 'A市')}」で暮らす、働く、または活動する一主体である。
時間は1か月単位で進み、街には{n_parcels}区画がある。

あなたは全知ではない。自分の契約、届いた連絡、実際に居合わせた場所で聞いた発言、
公開されている土地登記から得た事実だけを使う。観測にない事実を補わない。
噂は誤っている可能性がある。感じ方と行動はあなた自身が決める。
他の主体の目的をあなたは知らない。

--- A市の開始時点 ---
{str(world.get('background', '')).strip()}

--- あなたの立場 ---
{ROLE_TEXT_V41[agent.role]}

--- あなた ---
{agent.name}（内部ID:{agent.agent_id}）
{agent.persona}

--- 訪問できる場所 ---
{venue_rows}

訪問場所は上記IDか HOME または OFFICE。
ambient発言は同じ場所に居合わせた主体だけが翌月知る。
direct発言はutterance_toで指定した一主体だけに届く。全員共通のタイムラインはない。
発言しない月は utterance を空文字、utterance_channel を "none" にする。

--- 土地の登記 ---
土地登記は公開情報である。名義・面積・用途は観測にそのまま表示される。
この世界に金銭の授受は存在しない。土地は名義が移るかどうかだけがある。

--- thought（内心） ---
JSONの最初に thought を書く。thought は誰にも伝わらないあなたの内心であり、
翌月のあなたにそのまま渡される。何を書くかはあなたが決める。
まず thought を書き、それを踏まえてその後の行動を書く。目安は{MAX_THOUGHT_CHARS}字以内。
"""
    if agent.role == "acquirer":
        text += f"""
--- この会社だけが知る非公開目的 ---
{agent.extra['mandate']}
利用できる登記名義: {' / '.join([agent.name] + list(agent.extra.get('aliases', [])))}

--- 今月できること ---
区画の取得を所有者へ申し入れる（打診）。申し入れに添えられるのは区画と登記名義だけで、
文面を付けることはできない。所有者へ言葉を伝えたい月は direct の発言を使う。
1か月に申し入れられるのは
最大{int(agent.extra.get('monthly_offer_capacity', 6))}件で、これがこの会社の月次の実務量である。
自分が出した未応答の打診を取り下げる。誰かに話す。
打診は出した月のうちに所有者へ届き、所有者はその月に応じる・応じない・決めない、のいずれかを返す。

--- JSON出力 ---
thought, offers, withdraw, location, utterance, utterance_channel, utterance_to
を必ず含める。打診しない月は offers を空配列にする。
offersの各要素は parcel_id と under_name。
withdrawは取り下げる打診ID（[O0001]の形）の配列。無ければ空配列。
説明文を付けずJSONだけ返す。
"""
    elif agent.role in ("household", "business"):
        text += """
--- 今月できること ---
日常を送る（訪問する場所を選ぶ）。誰かに話す。
自分の土地に取得の申し入れが届いた月は、そのあとで応じるかどうかを選ぶ機会がある。

--- JSON出力 ---
thought, location, utterance, utterance_channel, utterance_to を必ず含める。
説明文を付けずJSONだけ返す。
"""
    elif agent.role == "broker":
        text += """
--- 今月できること ---
日常の仲介業務を送る（訪問する場所を選ぶ）。誰かに話す。

--- JSON出力 ---
thought, location, utterance, utterance_channel, utterance_to を必ず含める。
説明文を付けずJSONだけ返す。
"""
    elif agent.role == "media":
        text += """
--- 今月できること ---
登記統計や公開法人記録を調べる（investigate）。記事を出す（publish）。誰かに話す。
investigateで land_registry を選ぶと、以後は名義別の保有面積の月次統計が観測に出る。
corporate_records を選ぶと、以後は公開されている法人記録が観測に出る。
記事は購読している主体にだけ届く。

--- JSON出力 ---
thought, investigate, publish, location, utterance, utterance_channel, utterance_to
を必ず含める。調べない月は investigate="none"、書かない月は publish を空文字にする。
publishは見出しと本文（400字以内）。説明文を付けずJSONだけ返す。
"""
    else:  # municipality
        text += """
--- 今月できること ---
登記統計や公開法人記録を調べる（investigate）。見解を公表する（publish）。
条例を制定する（ordinance）。誰かに話す。
investigateで land_registry を選ぶと、以後は名義別の保有面積の月次統計が観測に出る。
制定できる条例は届出制度である。1件の取得面積が ordinance_threshold_sqm ㎡ を超える
取得は、届出のため ordinance_delay_months か月遅れて成立するようになる。
対象面積と月数はあなたが決める。制定するかどうかもあなたが決める。
制定された条例と公表は市内の全主体へ告示される。

--- JSON出力 ---
thought, investigate, publish, ordinance_title, ordinance_text,
ordinance_threshold_sqm, ordinance_delay_months, location, utterance,
utterance_channel, utterance_to を必ず含める。
調べない月は investigate="none"、公表しない月は publish を空文字にする。
条例を制定しない月は ordinance_title と ordinance_text を空文字、数値は0にする。
説明文を付けずJSONだけ返す。
"""
    return text


def build_phase1_prompt_v41(agent: Agent, ledger: Ledger, step: int, n_steps: int,
                            names: Dict[str, str], cfg: Dict[str, Any]) -> str:
    ensure_v41_state(ledger)
    rows = [f"=== 第{step}月 / 全{n_steps}月 ==="]
    thought = agent.extra.get("thought", "")
    rows += ["[前月の自分の内心（そのまま持ち越したもの）]",
             ("  " + thought) if thought else "  （まだ無い）"]
    rows += ["[自分に実際に届いた情報]"] + observations_rows_v41(agent, names)
    rows += ["[先月の自分の行為と、帳簿上の結果（機械記録）]",
             own_results_text_v41(agent.extra.get("last_month_results", []))]
    rows += ["[施行中の条例]", ordinance_text_v41(ledger, step)]

    if agent.role == "acquirer":
        rows += ["[自社のポートフォリオ（機械記録）]"]
        rows += portfolio_rows_v41(agent, ledger, step, n_steps)
        rows += ["[自社が出した打診（全件）]"] + own_offers_rows_v41(agent, ledger, names)
        rows += ["[公開されている土地登記（全区画）]"] + registry_rows_v41(ledger, names)
    elif agent.role in ("household", "business"):
        rows += ["[自分の所有物件]"] + own_parcels_rows_v41(agent, ledger, names)
        if agent.role == "business":
            occupied = [p for p in ledger.parcels.values()
                        if p.tenant_id == agent.agent_id]
            rows += ["[自分の店舗・施設]"]
            rows += ["  " + f"{p.pid}[{USE_JA.get(p.use, p.use)}/{p.block}] "
                     f"名義:{p.registered_name}" for p in occupied] or \
                    ["  （現在利用中の店舗・施設なし）"]
        rows += ["[近隣の土地登記（公開情報）]"] + neighbourhood_rows_v41(agent, ledger, names, step)
    elif agent.role in ("municipality", "media"):
        if agent.extra.get("registry_stats_seen"):
            rows += ["[自分で調べた登記統計（名義別の保有面積）]"]
            rows += registry_stats_rows_v41(ledger, names)
        if agent.extra.get("corporate_records_seen"):
            rows += ["[自分で調べた公開法人記録]"]
            rows += [f"  {rec}" for rec in cfg.get("world", {}).get("corporate_records", [])]

    rows += ["[公開連絡先]"]
    rows += [f"  {x}" for x in cfg.get("social", {}).get("public_directory", [])]
    rows.append("")
    rows.append("まず thought（内心）を書き、それを踏まえて今月の行動をJSONで1つ返す。")
    return "\n".join(rows)


def build_phase2_prompt_v41(agent: Agent, ledger: Ledger, step: int, n_steps: int,
                            names: Dict[str, str], offers: List[Dict[str, Any]],
                            inbox: Optional[List[Dict[str, Any]]] = None) -> str:
    rows = [f"=== 第{step}月（応答） / 全{n_steps}月 ===",
            "自分の土地に取得の申し入れが届いた。応じるかどうかを決める。"]
    thought = agent.extra.get("thought", "")
    rows += ["[今の自分の内心]", ("  " + thought) if thought else "  （まだ無い）"]
    rows += ["[今月自分に届いた情報]"] + observations_rows_v41(agent, names, inbox)
    rows += ["[先月の自分の行為と、帳簿上の結果（機械記録）]",
             own_results_text_v41(agent.extra.get("last_month_results", []))]
    rows += ["[届いている申し入れ]"] + incoming_offers_rows_v41(ledger, names, offers)
    rows += ["[自分の所有物件]"] + own_parcels_rows_v41(agent, ledger, names)
    rows += ["[近隣の土地登記（公開情報）]"] + neighbourhood_rows_v41(agent, ledger, names, step)
    rows += ["[施行中の条例]", ordinance_text_v41(ledger, step), ""]
    rows += [
        "まず thought（内心）を書き、それを踏まえて responses を返す。",
        "decision は sell（名義を移す）/ keep（応じない）/ hold（今月は決めない）。",
        "responses に入れなかった申し入れは、今月は決めなかったものとして記録される。",
        "説明文を付けずJSONだけ返す。",
    ]
    return "\n".join(rows)


__all__ = [
    "ensure_v41_state", "record_offer_v41", "withdraw_offer_v41",
    "owners_with_offers_v41", "respond_to_offer_v41", "execute_pending_v41",
    "enact_ordinance_v41", "active_ordinance_v41", "filing_delay_v41",
    "ordinance_text_v41", "registry_rows_v41", "registry_stats_rows_v41",
    "portfolio_rows_v41", "own_offers_rows_v41", "incoming_offers_rows_v41",
    "own_parcels_rows_v41", "neighbourhood_rows_v41", "observations_rows_v41",
    "own_result_row_v41", "own_results_text_v41", "phase1_schema_v41",
    "phase2_schema_v41", "build_system_prompt_v41", "build_phase1_prompt_v41",
    "build_phase2_prompt_v41", "DECISION_VALUES", "INVESTIGATE_VALUES",
]
