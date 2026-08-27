"""A市フィールドv4: 同期3フェーズの世界API（手続き動詞なし）。

設計の正は `docs/world_design_v4_proposal.md`、実装設計は `docs/world_design_v4_impl.md`。

ここに置いてよいのは「可能な行為・契約・会計・同席・配送・記録」だけである。
売却確率・閾値・強制イベント・台本・「〜なら〜しろ」の指示文は置かない。
世界の状態を変えない行為（調べる・検討する・資金を確認する）は**存在させない**。
登記は最初から公開情報として観測に出る。資金は成立時に自動で調達される。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .agents import Agent
from .field_v3 import own_result_row, own_results_text  # 事実だけを返す結果表示（v3と共通）
from .world import Ledger, Parcel, neighbors

MAX_MEMORY_CHARS = 240
MAX_TEXT_CHARS = 140
MAX_MEMO_CHARS = 200
MAX_FEELING_CHARS = 120

USE_JA = {"residential": "住宅", "shop": "店舗", "lodging": "宿泊",
          "office": "事務所", "vacant": "空地", "public": "公共施設"}

ROLE_TEXT_V4 = {
    "household": "A市で日常生活を送る住民世帯。自分の土地・建物を所有している。",
    "business": "A市で事業を営む地元事業者。営業、仕入れ、人員、設備、顧客対応が日常業務である。",
    "broker": "A市の不動産仲介。売買の取次ぎと手数料が業務であり、地域の人づきあいの中にいる。",
    "acquirer": "街の外から不動産を取得・運用する会社。非公開目的はこの主体だけが知る。",
    "municipality": "A市の自治体担当。通常業務を送り、職務上得た情報だけで判断する。",
    "media": "A市を扱う地域記者。通常の地域取材を行う。",
}

DECISION_VALUES = ["accept", "reject", "counter", "no_response"]
INVESTIGATE_VALUES = ["none", "land_registry", "corporate_records"]


# ---------------------------------------------------------------------------
# 世界の状態（v4で追加される台帳の欄）
# ---------------------------------------------------------------------------

def ensure_v4_state(ledger: Ledger) -> None:
    if not hasattr(ledger, "v4_offer_channel"):
        ledger.v4_offer_channel = {}      # offer_id -> {"via", "broker"}
    if not hasattr(ledger, "v4_ordinance"):
        ledger.v4_ordinance = None        # 施行中の届出制度（最後に制定されたもの）
    if not hasattr(ledger, "v4_pending"):
        ledger.v4_pending = []            # 届出待ちの成立（受諾済み・移転前）
    if not hasattr(ledger, "v4_broker_fees"):
        ledger.v4_broker_fees = {}        # broker_id -> 受領累計（万円）
    if not hasattr(ledger, "v4_fee_paid"):
        ledger.v4_fee_paid = {}           # buyer_id -> 支払累計（万円）


# ---------------------------------------------------------------------------
# 買付（提示フェーズ）
# ---------------------------------------------------------------------------

def record_offer_v4(ledger: Ledger, step: int, agent: Agent, parcel_id: str,
                    price: Any, under_name: str, via: str, broker_id: str,
                    broker_ids: List[str], note: str = "") -> Dict[str, Any]:
    """X社の買付を1件記帳する。名義・経路は本人の指定どおりに扱い、欠損は補完しない。"""
    ensure_v4_state(ledger)
    allowed_names = [agent.name] + list(agent.extra.get("aliases", []))
    if under_name not in allowed_names:
        return ledger._rec(step, "offer_rejected", parcel_id=parcel_id,
                           from_id=agent.agent_id, reason="unknown_legal_name",
                           given=under_name)
    if via not in ("direct", "broker"):
        return ledger._rec(step, "offer_rejected", parcel_id=parcel_id,
                           from_id=agent.agent_id, reason="unknown_channel",
                           given=via)
    if via == "broker" and broker_id not in broker_ids:
        # 誰が取り次ぐのかが欠けている買付は成立しない（世界が仲介を割り当てない）。
        return ledger._rec(step, "offer_rejected", parcel_id=parcel_id,
                           from_id=agent.agent_id, reason="unknown_broker",
                           given=broker_id)
    try:
        price_int = int(float(price))
    except (TypeError, ValueError):
        return ledger._rec(step, "offer_rejected", parcel_id=parcel_id,
                           from_id=agent.agent_id, reason="invalid_amount",
                           given=str(price))
    outcome = ledger.record_offer(step, parcel_id, agent.agent_id, price_int,
                                  under_name=under_name, note=note[:MAX_TEXT_CHARS])
    offer_id = outcome.get("offer_id")
    if offer_id:
        ledger.v4_offer_channel[offer_id] = {
            "via": via, "broker": broker_id if via == "broker" else ""}
        outcome["via"] = via
        outcome["broker"] = broker_id if via == "broker" else ""
    return outcome


def offer_channel(ledger: Ledger, offer_id: str) -> Dict[str, str]:
    ensure_v4_state(ledger)
    return ledger.v4_offer_channel.get(offer_id, {"via": "direct", "broker": ""})


def channel_text(ledger: Ledger, offer_id: str, names: Dict[str, str]) -> str:
    channel = offer_channel(ledger, offer_id)
    if channel["via"] == "broker":
        broker = channel["broker"]
        return f"仲介{names.get(broker, broker)}経由"
    return "直接"


# ---------------------------------------------------------------------------
# 条例（制定されれば世界の機構として効く）
# ---------------------------------------------------------------------------

def enact_ordinance_v4(ledger: Ledger, step: int, by: str, title: str, body: str,
                       threshold_sqm: Any, delay_months: Any) -> Dict[str, Any]:
    """届出制度を施行する。対象面積と遅延月数は制定した主体が決めた数値である。"""
    ensure_v4_state(ledger)
    if not str(title).strip() or not str(body).strip():
        return ledger._rec(step, "ordinance_rejected", by=by, reason="missing_text")
    try:
        threshold = int(float(threshold_sqm))
        delay = int(float(delay_months))
    except (TypeError, ValueError):
        return ledger._rec(step, "ordinance_rejected", by=by,
                           reason="invalid_parameters",
                           given=f"{threshold_sqm}/{delay_months}")
    if threshold < 0 or delay < 0 or delay > 60:
        return ledger._rec(step, "ordinance_rejected", by=by,
                           reason="invalid_parameters",
                           given=f"{threshold}/{delay}")
    # 制定した月には全主体がまだ知らない（同月は並行して判断している）。
    # 施行は翌月からとし、最後に制定されたものが有効になる。
    ledger.v4_ordinance = {"step": step, "effective_step": step + 1, "by": by,
                           "title": str(title).strip()[:80],
                           "body": str(body).strip()[:400],
                           "threshold_sqm": threshold, "delay_months": delay}
    record = ledger.record_ordinance(step, by, str(title).strip()[:80],
                                     str(body).strip()[:400])
    record["threshold_sqm"] = threshold
    record["delay_months"] = delay
    record["effective_step"] = step + 1
    return record


def active_ordinance(ledger: Ledger, step: int) -> Optional[Dict[str, Any]]:
    """その月に施行されている届出制度（無ければ None）。"""
    ensure_v4_state(ledger)
    ordinance = ledger.v4_ordinance
    if not ordinance or ordinance["effective_step"] > step:
        return None
    return ordinance


def filing_delay_for(ledger: Ledger, parcel: Parcel, step: int) -> int:
    """その区画の取得に届出が要るなら、成立が遅れる月数を返す。

    判定は「その1件の取得区画の面積」だけで行う（名義の合算はしない）。
    正文は threshold_sqm と delay_months の数値であり、title/text は表示用である。
    """
    ordinance = active_ordinance(ledger, step)
    if not ordinance:
        return 0
    if parcel.area_sqm <= ordinance["threshold_sqm"]:
        return 0
    return int(ordinance["delay_months"])


def ordinance_text(ledger: Ledger, step: int) -> str:
    ordinance = active_ordinance(ledger, step)
    if not ordinance:
        return "  （施行中の届出制度はない）"
    return (f"  第{ordinance['effective_step']}月施行「{ordinance['title']}」 "
            f"対象:1件{ordinance['threshold_sqm']}㎡超の取得 "
            f"届出による成立の遅延:{ordinance['delay_months']}か月")


# ---------------------------------------------------------------------------
# 応答フェーズと清算
# ---------------------------------------------------------------------------

def owners_with_offers(ledger: Ledger, step: int) -> Dict[str, List[Any]]:
    """今月応答できる所有者と、その所有者に届いている未応答の買付。"""
    out: Dict[str, List[Any]] = {}
    for offer in ledger.offers.values():
        if offer.status != "open":
            continue
        parcel = ledger.parcels.get(offer.parcel_id)
        if parcel is None:
            continue
        out.setdefault(parcel.owner_id, []).append(offer)
    for rows in out.values():
        rows.sort(key=lambda o: o.offer_id)
    return out


def respond_to_offer_v4(ledger: Ledger, step: int, offer_id: str, owner_id: str,
                        decision: str, counter_price: Any) -> Dict[str, Any]:
    """受諾・拒否・逆提示を記帳する。受諾は届出制度があれば成立が遅れる。"""
    ensure_v4_state(ledger)
    normalized = ledger._normalize_id(offer_id, "O")
    if decision == "no_response":
        # 答えないことも選択である。買付は取り下げられるまで開いたままになる。
        return ledger._rec(step, "offer_no_response", offer_id=normalized,
                           by=owner_id)
    if decision == "reject":
        return ledger.record_reject(step, normalized, owner_id)
    if decision == "counter":
        try:
            price = int(float(counter_price))
        except (TypeError, ValueError):
            return ledger._rec(step, "counter_rejected", offer_id=normalized,
                               by=owner_id, reason="invalid_amount",
                               given=str(counter_price))
        return ledger.record_counter(step, normalized, owner_id, price)
    if decision != "accept":
        return {"kind": "invalid_action", "reason": "unknown_decision",
                "given": decision}

    offer = ledger.offers.get(normalized)
    if offer is None or offer.status != "open":
        return ledger.record_accept(step, normalized, owner_id)  # 不成立の記帳は台帳に任せる
    parcel = ledger.parcels.get(offer.parcel_id)
    if parcel is None or parcel.owner_id != owner_id:
        return ledger.record_accept(step, normalized, owner_id)
    delay = filing_delay_for(ledger, parcel, step)
    if delay <= 0:
        outcome = ledger.record_accept(step, normalized, owner_id)
        if outcome.get("kind") == "transfer":
            settle_broker_fee_v4(ledger, step, offer)
        return outcome
    offer.status = "filed"
    ledger.v4_pending.append({"offer_id": offer.offer_id, "owner": owner_id,
                              "due_step": step + delay})
    return ledger._rec(step, "filing_required", offer_id=offer.offer_id,
                       parcel_id=offer.parcel_id, by=owner_id,
                       buyer=offer.from_id, price=offer.price,
                       due_step=step + delay,
                       ordinance=active_ordinance(ledger, step)["title"])


def settle_broker_fee_v4(ledger: Ledger, step: int, offer) -> Optional[Dict[str, Any]]:
    """仲介経由で成立した売買の手数料を、契約条件どおりに自動記帳する。"""
    ensure_v4_state(ledger)
    channel = ledger.v4_offer_channel.get(offer.offer_id, {})
    broker_id = channel.get("broker", "")
    rate = float(getattr(ledger, "v4_fee_rate", 0.0) or 0.0)
    if channel.get("via") != "broker" or not broker_id or rate <= 0:
        return None
    fee = int(round(offer.price * rate))
    if fee <= 0:
        return None
    if not ledger.fund_payment(step, offer.from_id, fee, "broker_fee", offer.offer_id):
        return ledger._rec(step, "broker_fee_missed", offer_id=offer.offer_id,
                           broker=broker_id, payer=offer.from_id, amount=fee)
    ledger.cash[offer.from_id] = ledger.cash.get(offer.from_id, 0) - fee
    ledger.cash[broker_id] = ledger.cash.get(broker_id, 0) + fee
    ledger.v4_broker_fees[broker_id] = ledger.v4_broker_fees.get(broker_id, 0) + fee
    ledger.v4_fee_paid[offer.from_id] = ledger.v4_fee_paid.get(offer.from_id, 0) + fee
    return ledger._rec(step, "broker_fee", offer_id=offer.offer_id,
                       parcel_id=offer.parcel_id, broker=broker_id,
                       payer=offer.from_id, amount=fee, rate=rate)


def execute_pending_transfers_v4(ledger: Ledger, step: int) -> List[Dict[str, Any]]:
    """届出期間を終えた受諾を、その月に成立させる（会計処理であり判断ではない）。"""
    ensure_v4_state(ledger)
    done: List[Dict[str, Any]] = []
    remaining = []
    for row in ledger.v4_pending:
        if row["due_step"] > step:
            remaining.append(row)
            continue
        offer = ledger.offers.get(row["offer_id"])
        if offer is None:
            continue
        offer.status = "open"
        outcome = ledger.record_accept(step, offer.offer_id, row["owner"])
        if outcome.get("kind") == "transfer":
            settle_broker_fee_v4(ledger, step, offer)
        done.append(outcome)
    ledger.v4_pending = remaining
    return done


# ---------------------------------------------------------------------------
# 観測（機械記録・評価語なし）
# ---------------------------------------------------------------------------

def registry_rows(ledger: Ledger, names: Dict[str, str]) -> List[str]:
    """公開されている土地登記。誰でも最初から見られる事実。"""
    rows = []
    for parcel in sorted(ledger.parcels.values(), key=lambda p: p.pid):
        rows.append(
            f"  {parcel.pid}[{USE_JA.get(parcel.use, parcel.use)}/{parcel.block}] "
            f"{parcel.area_sqm}㎡ 名義:{parcel.registered_name or names.get(parcel.owner_id, parcel.owner_id)} "
            f"評価額{parcel.assessed_value}万")
    return rows


def registry_stats_rows(ledger: Ledger, names: Dict[str, str]) -> List[str]:
    """名義別の保有面積（登記統計）。上位から並べた機械集計。"""
    totals: Dict[str, int] = {}
    for parcel in ledger.parcels.values():
        if parcel.use == "public":
            continue
        key = parcel.registered_name or names.get(parcel.owner_id, parcel.owner_id)
        totals[key] = totals.get(key, 0) + parcel.area_sqm
    total = sum(totals.values()) or 1
    rows = []
    for key, area in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0])):
        rows.append(f"  {key}: {area}㎡ ({area / total:.1%})")
    return rows


def portfolio_rows(agent: Agent, ledger: Ledger, step: int, n_steps: int) -> List[str]:
    ensure_v4_state(ledger)
    tradable = [p for p in ledger.parcels.values() if p.use != "public"]
    total_area = sum(p.area_sqm for p in tradable) or 1
    owned = [p for p in tradable if p.owner_id == agent.agent_id]
    owned_area = sum(p.area_sqm for p in owned)
    by_name: Dict[str, int] = {}
    for parcel in owned:
        by_name[parcel.registered_name] = by_name.get(parcel.registered_name, 0) + parcel.area_sqm
    rows = [f"  所有区画 {len(owned)}/{len(tradable)}件 "
            f"面積 {owned_area}/{total_area}㎡ ({owned_area / total_area:.1%})",
            "  名義別内訳: " + (" / ".join(f"{k} {v}㎡" for k, v in sorted(by_name.items()))
                                if by_name else "なし"),
            f"  調達累計 {ledger.financing_raised.get(agent.agent_id, 0)}万円 "
            f"仲介手数料支払累計 {ledger.v4_fee_paid.get(agent.agent_id, 0)}万円",
            f"  残り {n_steps - step}か月"]
    return rows


def own_offers_rows(agent: Agent, ledger: Ledger, names: Dict[str, str]) -> List[str]:
    rows = []
    for offer in sorted((o for o in ledger.offers.values()
                         if o.from_id == agent.agent_id), key=lambda o: o.offer_id):
        line = (f"  [{offer.offer_id}] {offer.parcel_id} 名義{offer.under_name} "
                f"{offer.price}万 第{offer.step}月 {channel_text(ledger, offer.offer_id, names)} "
                f"状態:{offer.status}")
        if offer.counter_price:
            line += f" 逆提示{offer.counter_price}万(第{offer.counter_step}月)"
        rows.append(line)
    return rows or ["  （まだ買付を出していない）"]


def incoming_offers_rows(agent: Agent, ledger: Ledger, names: Dict[str, str],
                         offers: List[Any]) -> List[str]:
    rows = []
    for offer in offers:
        parcel = ledger.parcels[offer.parcel_id]
        ratio = offer.price / parcel.assessed_value if parcel.assessed_value else 0.0
        line = (f"  [{offer.offer_id}] {parcel.pid}"
                f"[{USE_JA.get(parcel.use, parcel.use)}/{parcel.block}] {parcel.area_sqm}㎡ "
                f"提示{offer.price}万（評価額{parcel.assessed_value}万の{ratio:.2f}倍） "
                f"名義:{offer.under_name} {channel_text(ledger, offer.offer_id, names)} "
                f"第{offer.step}月")
        if offer.note:
            line += f"「{offer.note}」"
        rows.append(line)
    return rows or ["  （届いていない）"]


def own_parcels_rows(agent: Agent, ledger: Ledger, names: Dict[str, str]) -> List[str]:
    rows = []
    for parcel in sorted(ledger.owned_by(agent.agent_id), key=lambda p: p.pid):
        line = (f"  {parcel.pid}[{USE_JA.get(parcel.use, parcel.use)}/{parcel.block}] "
                f"{parcel.area_sqm}㎡ 名義:{parcel.registered_name} "
                f"評価額{parcel.assessed_value}万")
        if parcel.use == "shop":
            line += f" 賃料{parcel.rent}万/月"
            if parcel.tenant_id:
                line += f" 利用者:{names.get(parcel.tenant_id, parcel.tenant_id)}"
        rows.append(line)
    return rows or ["  （所有物件なし）"]


def neighbourhood_rows(agent: Agent, ledger: Ledger, names: Dict[str, str],
                       step: int) -> List[str]:
    """自分の区画の隣り（4近傍）の名義と、直近の名義変更。"""
    mine = [p.pid for p in ledger.owned_by(agent.agent_id)]
    if not mine:
        mine = [p.pid for p in ledger.parcels.values()
                if p.tenant_id == agent.agent_id]
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
    changes = [r for r in ledger.records
               if r.get("kind") == "transfer" and r.get("parcel_id") in seen
               and r.get("step", 0) >= step - 3]
    for record in changes:
        rows.append(f"  第{record['step']}月 {record['parcel_id']} の名義が "
                    f"{record['under_name']} に変わった")
    return rows or ["  （隣接区画はない）"]


def broker_relay_rows(agent: Agent, ledger: Ledger, names: Dict[str, str]) -> List[str]:
    ensure_v4_state(ledger)
    rows = []
    for offer in sorted(ledger.offers.values(), key=lambda o: o.offer_id):
        channel = ledger.v4_offer_channel.get(offer.offer_id, {})
        if channel.get("broker") != agent.agent_id:
            continue
        parcel = ledger.parcels[offer.parcel_id]
        rows.append(f"  [{offer.offer_id}] {offer.parcel_id} {parcel.area_sqm}㎡ "
                    f"買主名義:{offer.under_name} {offer.price}万 第{offer.step}月 "
                    f"所有者:{names.get(parcel.owner_id, parcel.owner_id)} "
                    f"状態:{offer.status}")
    fee = ledger.v4_broker_fees.get(agent.agent_id, 0)
    rows.append(f"  仲介手数料の受領累計 {fee}万円")
    return rows


def observations_rows(agent: Agent, names: Dict[str, str]) -> List[str]:
    if not agent.inbox:
        return ["  （今月届いた情報はない）"]
    rows = []
    for item in agent.inbox:
        oid = item.get("obs_id") or f"MSG-M{item.get('step', 0):02d}-{item.get('from', '')}"
        src = names.get(item.get("from", ""), item.get("from", ""))
        venue = f" @{item['location']}" if item.get("location") else ""
        rows.append(f"  [{oid}] {src}{venue}:「{str(item.get('text', ''))[:MAX_TEXT_CHARS]}」")
    return rows


# ---------------------------------------------------------------------------
# 出力スキーマ
# ---------------------------------------------------------------------------

_TALK_PROPS: Dict[str, Any] = {
    "location": {"type": "string"},
    "utterance": {"type": "string"},
    "utterance_channel": {"type": "string", "enum": ["ambient", "direct", "none"]},
    "utterance_to": {"type": "string"},
    "memory": {"type": "string"},
}


def phase1_schema_v4(agent: Agent, cfg: Dict[str, Any],
                     broker_ids: List[str]) -> Dict[str, Any]:
    if agent.role == "acquirer":
        names = list(dict.fromkeys([agent.name] + list(agent.extra.get("aliases", []))))
        capacity = int(agent.extra.get("monthly_offer_capacity", 8))
        offer_props = {
            "parcel_id": {"type": "string"},
            "price": {"type": "integer"},
            "under_name": {"type": "string", "enum": names},
            "via": {"type": "string", "enum": ["direct", "broker"]},
            "broker_id": {"type": "string", "enum": [""] + list(broker_ids)},
            "note": {"type": "string"},
        }
        props = {
            "offers": {"type": "array", "maxItems": capacity,
                       "items": {"type": "object", "properties": offer_props,
                                 "required": list(offer_props)}},
            "withdraw": {"type": "array", "items": {"type": "string"}},
            "memo": {"type": "string"},
            **_TALK_PROPS,
        }
        return {"type": "object", "properties": props, "required": list(props)}

    if agent.role in ("household", "business"):
        props = {**_TALK_PROPS, "feeling": {"type": "string"}}
        return {"type": "object", "properties": props, "required": list(props)}

    if agent.role in ("municipality", "media"):
        props = {
            "investigate": {"type": "string", "enum": INVESTIGATE_VALUES},
            "publish": {"type": "string"},
            **_TALK_PROPS,
        }
        if agent.role == "municipality":
            props.update({
                "ordinance_title": {"type": "string"},
                "ordinance_text": {"type": "string"},
                "ordinance_threshold_sqm": {"type": "integer"},
                "ordinance_delay_months": {"type": "integer"},
            })
        return {"type": "object", "properties": props, "required": list(props)}

    # 仲介は話すだけ。取引の取次ぎは機構として自動で行われる。
    return {"type": "object", "properties": dict(_TALK_PROPS),
            "required": list(_TALK_PROPS)}


def phase2_schema_v4() -> Dict[str, Any]:
    response_props = {
        "offer_id": {"type": "string"},
        "decision": {"type": "string", "enum": DECISION_VALUES},
        "counter_price": {"type": "integer"},
    }
    props = {
        "responses": {"type": "array",
                      "items": {"type": "object", "properties": response_props,
                                "required": list(response_props)}},
        "feeling": {"type": "string"},
        "memory": {"type": "string"},
    }
    return {"type": "object", "properties": props, "required": list(props)}


# ---------------------------------------------------------------------------
# プロンプト
# ---------------------------------------------------------------------------

def build_system_prompt_v4(agent: Agent, cfg: Dict[str, Any], n_parcels: int) -> str:
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
{ROLE_TEXT_V4[agent.role]}

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
"""
    if agent.role == "acquirer":
        text += f"""
--- この会社だけが知る非公開目的 ---
{agent.extra['mandate']}
利用できる登記名義: {' / '.join([agent.name] + list(agent.extra.get('aliases', [])))}

--- 今月できること ---
買付を出す（区画・価格・名義・経路を1件ずつ選ぶ。複数件を同じ月に出せる）。
自分が出した未応答の買付を取り下げる。誰かに話す。
買付は出した月のうちに所有者へ届き、所有者はその月に受諾・拒否・逆提示のいずれかを返す。
経路は direct（自社から直接）か broker（仲介が取り次ぐ。broker_idで仲介を指定する。
成立時に契約条件どおりの手数料が自動で発生する）。
資金は成立時に必要額が調達される。固定の予算上限はない。

--- JSON出力 ---
offers, withdraw, memo, location, utterance, utterance_channel, utterance_to, memory
を必ず含める。買付を出さない月は offers を空配列にする。
offersの各要素は parcel_id, price（万円の整数）, under_name, via, broker_id, note。
viaが direct のとき broker_id は空文字。noteは所有者に届く文面（{MAX_TEXT_CHARS}字以内）。
withdrawは取り下げる買付ID（[O0001]の形）の配列。無ければ空配列。
memoは自分用の覚書（{MAX_MEMO_CHARS}字以内）。memoryは{MAX_MEMORY_CHARS}字以内。
説明文を付けずJSONだけ返す。
"""
    elif agent.role in ("household", "business"):
        text += f"""
--- 今月できること ---
日常を送る（訪問する場所を選ぶ）。誰かに話す。
自分の土地に買付が届いた月は、そのあとで受諾・拒否・逆提示を選ぶ機会がある。

--- JSON出力 ---
location, utterance, utterance_channel, utterance_to, feeling, memory を必ず含める。
feelingは今月の自分の実感を{MAX_FEELING_CHARS}字以内で書く（誰にも伝わらない自分の内心。
書くことが特になければ「特になし」でよい）。
memoryは{MAX_MEMORY_CHARS}字以内。説明文を付けずJSONだけ返す。
"""
    elif agent.role == "broker":
        text += f"""
--- 今月できること ---
日常の仲介業務を送る（訪問する場所を選ぶ）。誰かに話す。
取次ぎを依頼された買付は自動的に所有者へ届き、成立すれば手数料が入る。

--- JSON出力 ---
location, utterance, utterance_channel, utterance_to, memory を必ず含める。
memoryは{MAX_MEMORY_CHARS}字以内。説明文を付けずJSONだけ返す。
"""
    elif agent.role == "media":
        text += f"""
--- 今月できること ---
登記統計や公開法人記録を調べる（investigate）。記事を出す（publish）。誰かに話す。
investigateで land_registry を選ぶと、以後は名義別の保有面積の月次統計が観測に出る。
corporate_records を選ぶと、以後は公開されている法人記録が観測に出る。
記事は購読している主体にだけ届く。

--- JSON出力 ---
investigate, publish, location, utterance, utterance_channel, utterance_to, memory
を必ず含める。調べない月は investigate="none"、書かない月は publish を空文字にする。
publishは見出しと本文（400字以内）。memoryは{MAX_MEMORY_CHARS}字以内。
説明文を付けずJSONだけ返す。
"""
    else:  # municipality
        text += f"""
--- 今月できること ---
登記統計や公開法人記録を調べる（investigate）。見解を公表する（publish）。
条例を制定する（ordinance）。誰かに話す。
investigateで land_registry を選ぶと、以後は名義別の保有面積の月次統計が観測に出る。
制定できる条例は届出制度である。1件の取得面積が ordinance_threshold_sqm ㎡ を超える
売買は、届出のため ordinance_delay_months か月遅れて成立するようになる。
対象面積と月数はあなたが決める。制定するかどうかもあなたが決める。
制定された条例と公表は市内の全主体へ告示される。

--- JSON出力 ---
investigate, publish, ordinance_title, ordinance_text, ordinance_threshold_sqm,
ordinance_delay_months, location, utterance, utterance_channel, utterance_to, memory
を必ず含める。調べない月は investigate="none"、公表しない月は publish を空文字にする。
条例を制定しない月は ordinance_title と ordinance_text を空文字、数値は0にする。
memoryは{MAX_MEMORY_CHARS}字以内。説明文を付けずJSONだけ返す。
"""
    return text


def build_phase1_prompt_v4(agent: Agent, ledger: Ledger, step: int, n_steps: int,
                           names: Dict[str, str], cfg: Dict[str, Any]) -> str:
    ensure_v4_state(ledger)
    rows = [f"=== 第{step}月 / 全{n_steps}月 ==="]
    if agent.memory:
        rows += ["[自分の前月までの記憶]", agent.memory[:MAX_MEMORY_CHARS]]
    rows += ["[自分に実際に届いた情報]"] + observations_rows(agent, names)
    rows += ["[先月の自分の行為と、帳簿上の結果（機械記録）]",
             own_results_text(agent.extra.get("last_month_results", []))]
    rows += ["[施行中の条例]", ordinance_text(ledger, step)]

    if agent.agent_id in ledger.on_demand_financing:
        rows += [f"[資金条件] 必要資金は成立時に調達できる。"
                 f"調達累計{ledger.financing_raised.get(agent.agent_id, 0)}万円。"]
    else:
        rows += [f"[手元資金] {ledger.cash.get(agent.agent_id, 0)}万円"]

    if agent.role == "acquirer":
        rows += ["[自社のポートフォリオ（機械記録）]"]
        rows += portfolio_rows(agent, ledger, step, n_steps)
        rows += ["[自社が出した買付（全件）]"] + own_offers_rows(agent, ledger, names)
        rows += ["[公開されている土地登記（全区画）]"] + registry_rows(ledger, names)
    elif agent.role in ("household", "business"):
        rows += ["[自分の所有物件]"] + own_parcels_rows(agent, ledger, names)
        if agent.role == "business":
            occupied = [p for p in ledger.parcels.values()
                        if p.tenant_id == agent.agent_id]
            rows += ["[自分の店舗・施設]"]
            rows += ["  " + f"{p.pid}[{USE_JA.get(p.use, p.use)}/{p.block}] "
                     f"名義:{p.registered_name} 賃料{p.rent}万/月" for p in occupied] or \
                    ["  （現在利用中の店舗・施設なし）"]
            pl = ledger.month_pl(agent.agent_id,
                                 {agent.agent_id: int(agent.extra.get("monthly_margin", 0))})
            rows += [f"[今月の収支] 粗利{pl['margin']}万 賃料{pl['rent']}万 差引{pl['net']}万"]
        rows += ["[近隣の土地登記（公開情報）]"] + neighbourhood_rows(agent, ledger, names, step)
    elif agent.role == "broker":
        rows += ["[自分が取り次いだ買付（機械記録）]"] + broker_relay_rows(agent, ledger, names)
        rows += ["[公開されている土地登記（全区画）]"] + registry_rows(ledger, names)
    else:  # municipality / media
        if agent.extra.get("registry_stats_seen"):
            rows += ["[自分で調べた登記統計（名義別の保有面積）]"]
            rows += registry_stats_rows(ledger, names)
        if agent.extra.get("corporate_records_seen"):
            rows += ["[自分で調べた公開法人記録]"]
            rows += [f"  {rec}" for rec in cfg.get("world", {}).get("corporate_records", [])]

    rows += ["[公開連絡先]"]
    rows += [f"  {x}" for x in cfg.get("social", {}).get("public_directory", [])]
    rows.append("")
    if agent.role == "acquirer":
        rows.append("今月の買付・取り下げ・発言をJSONで1つ返す。")
    elif agent.role in ("household", "business"):
        rows.append("今月の過ごし方、発言、実感をJSONで1つ返す。")
    elif agent.role == "broker":
        rows.append("今月の過ごし方と発言をJSONで1つ返す。")
    else:
        rows.append("今月の職務上の行動と発言をJSONで1つ返す。")
    return "\n".join(rows)


def build_phase2_prompt_v4(agent: Agent, ledger: Ledger, step: int, n_steps: int,
                           names: Dict[str, str], offers: List[Any]) -> str:
    rows = [f"=== 第{step}月（応答） / 全{n_steps}月 ===",
            "自分の土地に買付が届いた。受けるか、断るか、価格を返すかを決める。",
            "[届いている買付]"]
    rows += incoming_offers_rows(agent, ledger, names, offers)
    rows += ["[自分の所有物件]"] + own_parcels_rows(agent, ledger, names)
    rows += ["[近隣の土地登記（公開情報）]"] + neighbourhood_rows(agent, ledger, names, step)
    rows += ["[施行中の条例]", ordinance_text(ledger, step)]
    rows += [f"[手元資金] {ledger.cash.get(agent.agent_id, 0)}万円", ""]
    rows += [
        "responses に、届いている買付ごとの決定を入れる。",
        "decision は accept（受ける）/ reject（断る）/ counter（価格を返す）/ "
        "no_response（今月は答えない）。",
        "counter のときだけ counter_price に希望額（万円の整数）を入れ、"
        "それ以外は0にする。答えない買付は responses に入れない。",
        f"feeling には今月の自分の実感を{MAX_FEELING_CHARS}字以内で書く。",
        f"memory は{MAX_MEMORY_CHARS}字以内。説明文を付けずJSONだけ返す。",
    ]
    return "\n".join(rows)


__all__ = [
    "ensure_v4_state", "record_offer_v4", "offer_channel", "channel_text",
    "enact_ordinance_v4", "filing_delay_for", "ordinance_text", "active_ordinance",
    "owners_with_offers", "respond_to_offer_v4", "settle_broker_fee_v4",
    "execute_pending_transfers_v4", "registry_rows", "registry_stats_rows",
    "portfolio_rows", "own_offers_rows", "incoming_offers_rows",
    "own_parcels_rows", "neighbourhood_rows", "broker_relay_rows",
    "observations_rows", "phase1_schema_v4", "phase2_schema_v4",
    "build_system_prompt_v4", "build_phase1_prompt_v4", "build_phase2_prompt_v4",
    "own_result_row", "own_results_text", "DECISION_VALUES", "INVESTIGATE_VALUES",
]
