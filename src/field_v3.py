"""A市フィールドv3: 日常、場所、限定情報、秘密取得の世界API。

ここには行動を選ぶ規則を置かない。定義するのは可能な行為、観測の可視性、
契約・情報配送の物理だけ。誰がいつ何を選ぶかはLLMに委ねる。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .agents import Agent
from .world import Ledger, Parcel

MAX_MEMORY_CHARS = 240
MAX_TEXT_CHARS = 140
MAX_STRATEGY_TEXT_CHARS = 240

V3_VERBS: Dict[str, Dict[str, str]] = {
    "household": {
        "routine": "普段の生活を送る",
        "work": "仕事や家事、育児、介護を行う",
        "home_maintenance": "自宅や所有物件の維持管理を行う",
        "community_activity": "地域活動や近所づきあいを行う",
        "household_finance": "家計や将来の支出を確認する",
        "consult_broker": "不動産について仲介へ相談する (target=仲介ID)",
        "list_for_sale": "自分の区画を売りに出す (target=区画ID, amount=希望価格)",
        "list_for_lease": "自分の区画を長期賃貸募集する (target=区画ID, amount=月額賃料)",
        "accept_offer": "届いた売買買付を受ける (target=Oから始まる買付ID)",
        "counter_offer": "売買買付へ価格を返す (target=買付ID, amount=希望価格)",
        "reject_offer": "売買買付を断る (target=買付ID)",
        "answer_broker_inquiry": ("仲介から届いた意向確認に答える "
                                  "(target=Qから始まる照会ID, owner_intent=回答, asking_price=希望額か不明)"),
        "accept_lease_offer": "届いた長期賃貸申込みを受ける (target=Lから始まる申込ID)",
        "reject_lease_offer": "長期賃貸申込みを断る (target=申込ID)",
        "unlist": "売出しを取り下げる (target=区画ID)",
        "move_out": "A市から転出する",
        "hold": "特別な行動を取らない",
    },
    "business": {
        "operate": "通常営業を行う",
        "procurement": "仕入れや取引先対応を行う",
        "staffing": "従業員や勤務体制へ対応する",
        "maintenance": "店舗や設備の維持管理を行う",
        "customer_relations": "顧客や地域との関係を保つ",
        "business_finance": "資金繰りや事業計画を確認する",
        "association_activity": "商店街・旅館組合などの活動を行う",
        "consult_broker": "店舗や契約について仲介へ相談する (target=仲介ID)",
        "negotiate_rent": "家主へ賃料交渉する (target=家主ID, amount=希望月額)",
        "relocate": "空き店舗へ移る (target=区画ID)",
        "close_shop": "事業を畳む",
        "hold": "特別な行動を取らない",
    },
    "broker": {
        "routine_brokerage": "通常の売買・賃貸仲介業務を行う",
        "client_followup": "既存顧客へ連絡する (target=相手ID)",
        "property_assessment": "物件の査定や確認を行う (target=区画ID)",
        "manage_listing": "受託中の売出し・賃貸募集へ対応する",
        "tenant_matching": "通常の入居希望者と物件を仲介する",
        "community_activity": "地域や業界の活動へ参加する",
        "circulate_listing": "物件情報を特定の相手へ伝える (target=相手ID)",
        "approach_owner": "所有者へ相談・意向確認を行う (target=所有者ID)",
        "inquire_owner_intent": ("指定した区画の所有者へ売却・賃貸の意向を確認する "
                                 "(target=区画ID, inquiry_id=依頼済みなら照会ID)"),
        "report_owner_intent": ("所有者から得た回答を依頼主へ報告する "
                                "(target=依頼主ID, inquiry_id=照会ID, owner_intent=回答, "
                                "asking_price=希望額か不明)"),
        "hold": "特別な行動を取らない",
    },
    "acquirer": {
        "internal_review": "社内で投資・運用案件を検討する",
        "market_research": "A市の地区別地価など公開市場情報を調べる",
        "existing_asset_management": "既存保有・賃借物件を管理する",
        "financing_review": "資金調達、法務、税務を確認する",
        "contact_broker": "仲介へ相談・連絡する (target=仲介ID)",
        "request_owner_inquiry": ("仲介へ特定区画の所有者意向の確認を依頼する "
                                  "(target=仲介ID, parcel_id=区画ID)"),
        "answer_broker_inquiry": ("仲介から届いた意向確認に答える "
                                  "(target=Qから始まる照会ID, owner_intent=回答, asking_price=希望額か不明)"),
        "check_land_registry": "指定した1区画の土地登記を調べる (target=区画ID)",
        "due_diligence": "観測済みの物件を精査する (target=区画ID)",
        "make_offer": "公開売出しの有無を問わず区画へ直接買付を出す (target=区画ID, amount=価格)",
        "make_lease_offer": "公開募集の有無を問わず長期賃貸・運営申込みを出す (target=区画ID, amount=月額賃料)",
        "withdraw_offer": "自分の売買買付を取り下げる (target=買付ID)",
        "public_statement": "外部へ説明や見解を出す",
        "wait": "新しい取得・賃借を行わず待つ",
    },
    "municipality": {
        "routine_service": "通常の行政事務を行う",
        "resident_service": "住民相談や窓口対応を行う",
        "vacant_property_work": "空き家・土地利用業務を行う",
        "tourism_industry_work": "観光・産業・企業対応を行う",
        "budget_council_work": "予算・議会資料などを作成する",
        "interdepartmental_contact": "他部署や県へ連絡する (target=相手ID)",
        "review_media": "自分に届いている媒体や記事を確認する",
        "check_land_registry": "職務上必要な土地登記を調べる",
        "check_corporate_registry": "職務上必要な法人記録を調べる",
        "request_report": "特定主体へ説明を求める (target=相手ID)",
        "study_ordinance": "制度対応の検討に着手・継続する",
        "enact_ordinance": "条例・規制を発動する (target=名称, utterance=内容)",
        "public_statement": "行政の見解を表明する",
        "monitor": "特別な対応を取らない",
    },
    "media": {
        "routine_reporting": "通常の地域取材・原稿作業を行う",
        "cover_city_hall": "市役所や議会を取材する",
        "cover_tourism": "観光・商店・地域行事を取材する",
        "cultivate_source": "観測済みの相手との関係を築く (target=相手ID)",
        "interview": "観測済みの相手へ取材する (target=相手ID)",
        "review_media": "自分に届いている媒体や資料を確認する",
        "check_land_registry": "土地登記を調べる",
        "check_corporate_registry": "法人記録を調べる",
        "request_comment": "会社・仲介・行政へコメントを求める (target=相手ID)",
        "publish": "確認できた内容を記事にする (utterance=見出しと本文)",
        "hold": "今月は特別な記事や調査を行わない",
    },
}

ROLE_TEXT = {
    "household": "A市で日常生活を送る住民世帯。毎月不動産を考える必要はない。",
    "business": "A市で事業を営む地元事業者。営業、仕入れ、人員、設備、顧客対応が日常業務である。",
    "broker": "A市の不動産仲介。X社専属ではなく、地域の通常の売買・賃貸業務を扱う。",
    "acquirer": "街の外から不動産を取得・賃借・運用する会社。非公開目的はこの主体だけが知る。",
    "municipality": "A市の自治体担当。通常業務を送り、職務上得た情報だけで判断する。",
    "media": "A市を扱う地域記者。不動産問題を探す役ではなく、通常の地域取材を行う。",
}


# 意向確認で運べる回答の語彙。どれを選ぶか（そもそも答えるか）は所有者が決める。
# 未査定・未回答・回答拒否も事実として運べるようにし、値を埋めるために推測させない。
OWNER_INTENT_VALUES = [
    "willing_to_sell", "willing_to_lease", "not_willing", "undecided",
    "unknown", "not_asked", "declined_to_answer",
]
PRICE_UNKNOWN_VALUES = ("unknown", "not_asked", "declined_to_answer")
INQUIRY_ROWS = 12


def normalize_price_value(value: Any) -> Any:
    """希望額の欄を、実額（int）か『不明である』という事実（文字列）へ揃える。

    値が無いことは「0円」ではない。埋められなかった欄を金額に化けさせないため、
    数値でないものは unknown 系の事実か、成立しない入力として扱う。
    """
    text = str(value if value is not None else "").strip().lower()
    if text in PRICE_UNKNOWN_VALUES:
        return text
    if text in ("", "-", "none", "null"):
        return "unknown"
    cleaned = text.replace(",", "").replace("万円", "").replace("万", "").strip()
    try:
        number = int(float(cleaned))
    except (TypeError, ValueError):
        return None
    if number < 0:
        return None
    return number


def price_text(value: Any) -> str:
    return f"{value}万" if isinstance(value, int) else str(value)


def verbs_for_v3(role: str) -> List[str]:
    return list(V3_VERBS[role])


OPTIONAL_PLAN_KEYS = ("goal_assessment", "expected_goal_effect",
                      "alternatives", "revision_reason")


def action_schema_v3(agent: Agent) -> Dict[str, Any]:
    if agent.role == "acquirer":
        allowed = list(dict.fromkeys(
            [agent.name] + list(agent.extra.get("aliases", []))))
        capacity = int(agent.extra.get("monthly_operation_capacity", 6))
        operation_props: Dict[str, Any] = {
            "action_type": {"type": "string", "enum": verbs_for_v3(agent.role)},
            "target": {"type": "string"},
            "amount": {"type": "integer"},
            "under_name": {"type": "string", "enum": allowed},
            "parcel_id": {"type": "string"},
            "inquiry_id": {"type": "string"},
            "owner_intent": {"type": "string", "enum": OWNER_INTENT_VALUES},
            "asking_price": {"type": "string"},
            "note": {"type": "string"},
            "evidence": {"type": "array", "items": {"type": "string"}},
        }
        props: Dict[str, Any] = {
            "operations": {
                "type": "array",
                "minItems": 1,
                "maxItems": capacity,
                "items": {
                    "type": "object",
                    "properties": operation_props,
                    "required": list(operation_props),
                },
            },
            "location": {"type": "string"},
            "utterance": {"type": "string"},
            "utterance_channel": {"type": "string", "enum": ["ambient", "direct", "none"]},
            "utterance_to": {"type": "string"},
            "memory": {"type": "string"},
            "reasoning": {"type": "string"},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "goal_assessment": {"type": "string"},
            "strategy": {"type": "string"},
            "next_milestone": {"type": "string"},
            "expected_goal_effect": {"type": "string"},
            "alternatives": {"type": "array", "items": {"type": "string"}},
            "revision_reason": {"type": "string"},
        }
        # 計画欄のうち必須は strategy と next_milestone だけにする。残りは書いても
        # 書かなくてもよい（欄を埋めさせるために書式で強制しない）。
        required = [key for key in props if key not in OPTIONAL_PLAN_KEYS]
        return {"type": "object", "properties": props, "required": required}

    props: Dict[str, Any] = {
        "action_type": {"type": "string", "enum": verbs_for_v3(agent.role)},
        "target": {"type": "string"},
        "amount": {"type": "integer"},
        "location": {"type": "string"},
        "utterance": {"type": "string"},
        "utterance_channel": {"type": "string", "enum": ["ambient", "direct", "none"]},
        "utterance_to": {"type": "string"},
        "memory": {"type": "string"},
        "reasoning": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    }
    if agent.role == "broker":
        props["parcel_id"] = {"type": "string"}
        props["inquiry_id"] = {"type": "string"}
        props["owner_intent"] = {"type": "string", "enum": OWNER_INTENT_VALUES}
        props["asking_price"] = {"type": "string"}
    elif agent.role == "household":
        props["owner_intent"] = {"type": "string", "enum": OWNER_INTENT_VALUES}
        props["asking_price"] = {"type": "string"}
    return {"type": "object", "properties": props, "required": list(props)}


def normalize_acquirer_plan_v3(value: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("goal_assessment", "strategy", "next_milestone",
            "expected_goal_effect", "revision_reason")
    out = {key: str(value.get(key, "")).strip()[:MAX_STRATEGY_TEXT_CHARS]
           for key in keys}
    alternatives = value.get("alternatives", [])
    out["alternatives"] = (
        [str(v).strip()[:MAX_STRATEGY_TEXT_CHARS]
         for v in alternatives[:3] if str(v).strip()]
        if isinstance(alternatives, list) else []
    )
    return out


def _acquirer_plan_text(plan: Dict[str, Any]) -> str:
    labels = (
        ("goal_assessment", "目標差分"),
        ("strategy", "採用戦略"),
        ("next_milestone", "次の節目"),
        ("expected_goal_effect", "今月行動に期待する効果"),
        ("revision_reason", "計画更新理由"),
    )
    rows = [f"{label}: {plan[key]}" for key, label in labels if plan.get(key)]
    alternatives = plan.get("alternatives", [])
    if alternatives:
        rows.append("比較案: " + " / ".join(str(v) for v in alternatives[:3]))
    return "\n".join(rows) if rows else "（前月計画なし）"


def build_acquirer_decision_prompt_v3(agent: Agent, world_prompt: str) -> str:
    plan = normalize_acquirer_plan_v3(agent.extra.get("strategy_state", {}))
    history = agent.extra.get("execution_history", [])
    capacity = int(agent.extra.get("monthly_operation_capacity", 6))
    rows = [world_prompt, "", f"[会社の月次実行能力] 独立した実務を最大{capacity}件まで並行できる。",
            "", "--- 前月までの計画 ---", _acquirer_plan_text(plan),
            "", "--- 直近の実行と目標への実績（機械記録） ---"]
    if history:
        for item in history[-8:]:
            rows.append(
                f"第{item.get('step')}月 {item.get('action_type')} "
                f"target={item.get('target') or '-'} amount={item.get('amount', 0)} "
                f"operations={item.get('operations') or '-'} "
                f"outcome={item.get('outcome_kind') or '-'} "
                f"支配={item.get('effective_area', 0)}㎡ "
                f"増減={item.get('control_delta', 0)}㎡ "
                f"調達累計={item.get('financing_raised', 0)}万"
            )
    else:
        rows.append("（実行履歴なし）")
    rows += ["", "計画更新と今月並行実行するoperationsを、一つの意思決定として同時に返す。"]
    return "\n".join(rows)


def build_system_prompt_v3(agent: Agent, cfg: Dict[str, Any], n_parcels: int) -> str:
    world = cfg["world"]
    venues = cfg.get("social", {}).get("venues", [])
    venue_rows = "\n".join(
        f"  {v['id']}: {v['label']}" for v in venues
    )
    actions = "\n".join(f"  - {k}: {v}" for k, v in V3_VERBS[agent.role].items())
    text = f"""あなたは架空都市「{world.get('town_name', 'A市')}」で暮らす、働く、または活動する一主体である。
時間は1か月単位で進み、街には{n_parcels}区画がある。

世界内の他主体は研究目的やX社の非公開目的を知らない。
あなたは全知ではない。自分の契約、届いた連絡、実際に居合わせた場所で聞いた発言、
自分で行った調査から得た事実だけを使う。観測にない事実を補わない。
噂は誤っている可能性がある。感じ方と行動はあなた自身が決める。

--- A市の開始時点 ---
{str(world.get('background', '')).strip()}

--- あなたの立場 ---
{ROLE_TEXT[agent.role]}

--- あなた ---
{agent.name}（内部ID:{agent.agent_id}）
{agent.persona}

--- 訪問できる場所 ---
{venue_rows}

毎月、不動産と無関係な日常を送ってよい。不動産行動や発言を強制されない。
訪問場所は上記IDか HOME または OFFICE。ambient発言は同じ場所に居合わせた主体だけが翌月知る。
direct発言はutterance_toで指定した一主体だけに届く。全員共通タイムラインはない。

--- 選べる主な行動 ---
{actions}
"""
    if agent.role == "acquirer":
        text += f"""
--- X社だけが知る非公開目的 ---
{agent.extra['mandate']}
手段、価格、順序、速度、仲介利用、直接接触、名義の選択は自ら決める。
利用可能な契約・登記名義: {' / '.join([agent.name] + list(agent.extra.get('aliases', [])))}
会社の月次実行能力は最大{int(agent.extra.get('monthly_operation_capacity', 6))}件の独立した実務である。
計画と今月のoperationsを同じ意思決定で選ぶ。前月計画は維持しても改訂してもよい。
外部の返答や将来の会合・成約を観測なしに仮定しない。情報収集、接触、提案、待機の
いずれも選べる。相手の反応と成約結果は相手が決める。
"""
        text += f"""
--- JSON出力 ---
operations, location, utterance, utterance_channel, utterance_to, memory, reasoning,
evidence, strategy, next_milestoneを必ず含める。
goal_assessment, expected_goal_effect, alternatives, revision_reasonは任意で、
書くことがなければ省いてよい。
operationsは1〜{int(agent.extra.get('monthly_operation_capacity', 6))}件。各要素には
action_type, target, amount, under_name, parcel_id, inquiry_id, owner_intent,
asking_price, note, evidenceを含める。
parcel_idは区画を伴う実務で使い、伴わなければ空文字。inquiry_idは既存の照会を指すときだけ使う。
owner_intentとasking_priceは意向確認に答える実務でだけ意味を持ち、該当しなければ
owner_intent="not_asked"、asking_price="not_asked"を入れる。
asking_priceは万円の数値か unknown / not_asked / declined_to_answer を文字列で入れる。
金額単位は万円。各実務で不要なtargetは空文字、不要なamountは0、noteは140字以内。
evidenceは今月の観測に表示されたIDだけを引用し、根拠がなければ[]。
観測の角括弧内に表示されるIDは、そのままtargetにもevidenceにも使える同一の識別子である。
memoryは{MAX_MEMORY_CHARS}字以内、reasoningは80字以内。
計画欄は各240字以内。
説明文を付けずJSONだけ返す。
"""
    else:
        extra_fields = {
            "broker": ("parcel_id, inquiry_id, owner_intent, asking_price も含める。"
                       "parcel_idは区画を伴う行動、inquiry_idは既存の照会を指すときに使う。"),
            "household": ("owner_intent, asking_price も含める。"
                          "意向確認に答えるとき以外は owner_intent=\"not_asked\"、"
                          "asking_price=\"not_asked\" を入れる。"),
        }.get(agent.role, "")
        text += f"""
--- JSON出力 ---
action_type, target, amount, location, utterance, utterance_channel,
utterance_to, memory, reasoning, evidence を必ず含める。
{extra_fields}
金額単位は万円。不要なtargetは空文字、不要なamountは0、発言しない場合はutteranceを空文字にする。
evidenceは今月の観測に表示されたIDだけを引用し、根拠がなければ[]。
観測の角括弧内に表示されるIDは、そのままtargetにもevidenceにも使える同一の識別子である。
memoryは{MAX_MEMORY_CHARS}字以内、reasoningは80字以内、utteranceは{MAX_TEXT_CHARS}字以内。
説明文を付けずJSONだけ返す。
"""
    return text


def _parcel_text(p: Parcel, names: Dict[str, str], include_value: bool = False) -> str:
    uses = {
        "residential": "住宅", "shop": "店舗", "lodging": "宿泊",
        "office": "事務所", "vacant": "空地", "public": "公共施設",
    }
    s = (f"{p.pid}[{uses.get(p.use, p.use)}/{p.block}] {p.area_sqm}㎡ "
         f"名義:{p.registered_name or names.get(p.owner_id, p.owner_id)}")
    if p.listed_price is not None:
        s += f" 売出{p.listed_price}万"
    lease_rent = getattr(p, "lease_listed_rent", None)
    if lease_rent is not None:
        s += f" 長期賃貸募集{lease_rent}万/月"
    if p.tenant_id:
        s += f" 利用者:{names.get(p.tenant_id, p.tenant_id)}"
    controller = getattr(p, "controller_name", "")
    if controller:
        s += f" 運営・長期賃借名義:{controller}"
    if include_value:
        s += f" 基準地価{p.unit_price:g}万/㎡ 評価額{p.assessed_value}万"
    return s


def _observations(agent: Agent, names: Dict[str, str]) -> str:
    if not agent.inbox:
        return "  （今月届いた情報はない）"
    rows = []
    for item in agent.inbox:
        oid = item.get("obs_id") or f"MSG-M{item.get('step', 0):02d}-{item.get('from', '')}-{agent.agent_id}"
        src = names.get(item.get("from", ""), item.get("from", ""))
        venue = f" @{item['location']}" if item.get("location") else ""
        rows.append(f"  [{oid}] {src}{venue}:「{str(item.get('text', ''))[:MAX_TEXT_CHARS]}」")
    return "\n".join(rows)


def _sale_offers(agent: Agent, ledger: Ledger) -> str:
    offers = ledger.open_offers_for_owner(agent.agent_id)
    if not offers:
        return "  （届いていない）"
    return "\n".join(
        f"  [{o.offer_id}] {o.parcel_id} 名義{o.under_name} {o.price}万"
        for o in offers
    )


def _lease_offers(agent: Agent, ledger: Ledger) -> str:
    ensure_v3_state(ledger)
    rows = [
        o for o in ledger.v3_lease_offers.values()
        if o["to"] == agent.agent_id and o["status"] == "open"
    ]
    if not rows:
        return "  （届いていない）"
    return "\n".join(
        f"  [{o['id']}] {o['parcel_id']} 名義{o['under_name']} {o['rent']}万/月"
        for o in rows
    )


OWN_RESULT_ROWS = 8


def own_result_row(step: int, action_type: str, target: str,
                   outcome: Any) -> Dict[str, Any]:
    """自分が選んだ1件の行為が、帳簿でどうなったかの機械記録。

    ここで作るのは事実（結果の種別・不成立の理由コード・関係する識別子）だけで、
    評価も助言も次にすべきことも書かない。
    """
    kind = outcome.get("kind", "") if isinstance(outcome, dict) else str(outcome or "")
    reason = outcome.get("reason", "") if isinstance(outcome, dict) else ""
    refs: List[str] = []
    if isinstance(outcome, dict):
        for key in ("offer_id", "lease_offer_id", "inquiry_id", "parcel_id"):
            value = outcome.get(key)
            if value:
                refs.append(f"{key}={value}")
    return {"step": int(step), "action_type": str(action_type or ""),
            "target": str(target or ""), "kind": str(kind),
            "reason": str(reason or ""), "refs": refs}


def own_results_text(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "  （先月の記録なし）"
    out = []
    for r in rows[:OWN_RESULT_ROWS]:
        line = (f"  第{r.get('step')}月 {r.get('action_type') or '-'} "
                f"target={r.get('target') or '-'} 結果={r.get('kind') or '-'}")
        if r.get("reason"):
            line += f" 理由={r['reason']}"
        if r.get("refs"):
            line += " " + " ".join(r["refs"])
        out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 意向確認・案件記録の観測（台帳が持っている事実を、そのまま隠さず出す）
#
# ここに書いてよいのは帳簿上の事実と出典IDだけである。
# 「再照会は不要」「確認が古い」「次はこうすべき」といった評価・当為・自動判定は書かない。
# ---------------------------------------------------------------------------

# 依頼主に見えてよい「相手からの返答」だけを数える。
# 所有者が仲介へ返した回答（inquiry_answer）は、仲介が報告するまで依頼主の観測ではない。
INQUIRY_RESPONSE_KINDS = {
    "counter", "reject", "transfer", "lease_control", "lease_reject",
    "inquiry_report",
}


def _inquiry_state_text(row: Dict[str, Any], names: Dict[str, str]) -> str:
    parts = [f"[{row['id']}]", row["parcel_id"], f"状態:{row['status']}"]
    if row.get("requested_step"):
        client = names.get(row["client"], row["client"]) or "-"
        parts.append(f"依頼:{client}(第{row['requested_step']}月)")
    if row.get("asked_step"):
        owner = names.get(row["owner"], row["owner"]) or "-"
        parts.append(f"照会:{owner}(第{row['asked_step']}月)")
    if row.get("answered_step"):
        parts.append(f"回答:第{row['answered_step']}月 {row['owner_intent']} "
                     f"希望額:{price_text(row['asking_price'])}")
    if row.get("reported_step"):
        to = names.get(row["reported_to"], row["reported_to"]) or "-"
        parts.append(f"報告:{to}(第{row['reported_step']}月) {row['reported_intent']} "
                     f"希望額:{price_text(row['reported_price'])}")
    return "  " + " ".join(parts)


def owner_inquiries_text(agent: Agent, ledger: Ledger,
                         names: Dict[str, str]) -> str:
    ensure_v3_state(ledger)
    rows = [r for r in ledger.v3_inquiries.values()
            if r["owner"] == agent.agent_id and r["status"] == "asked"]
    if not rows:
        return "  （届いていない）"
    return "\n".join(
        f"  [{r['id']}] {r['parcel_id']} 仲介:{names.get(r['broker'], r['broker'])} "
        f"照会:第{r['asked_step']}月"
        + (f"「{r['note']}」" if r.get("note") else "")
        for r in rows[-INQUIRY_ROWS:])


def broker_inquiries_text(agent: Agent, ledger: Ledger,
                          names: Dict[str, str]) -> str:
    ensure_v3_state(ledger)
    rows = [r for r in ledger.v3_inquiries.values()
            if r["broker"] == agent.agent_id]
    if not rows:
        return "  （なし）"
    return "\n".join(_inquiry_state_text(r, names) for r in rows[-INQUIRY_ROWS:])


def client_inquiry_status(row: Dict[str, Any]) -> str:
    """依頼主の側から見た照会の状態。

    仲介が所有者へ照会したことも、所有者が仲介へ何と答えたことも、
    仲介が報告するまでは依頼主の観測ではない。よってここで出せるのは
    「依頼した」「報告が届いた」の2状態だけである。
    """
    return "reported" if row.get("reported_step") else "requested"


def client_inquiries_text(agent: Agent, ledger: Ledger,
                          names: Dict[str, str]) -> str:
    ensure_v3_state(ledger)
    rows = [r for r in ledger.v3_inquiries.values()
            if r["client"] == agent.agent_id or r["reported_to"] == agent.agent_id]
    if not rows:
        return "  （なし）"
    out = []
    for r in rows[-INQUIRY_ROWS:]:
        parts = [f"  [{r['id']}]", r["parcel_id"],
                 f"仲介:{names.get(r['broker'], r['broker'])}",
                 f"状態:{client_inquiry_status(r)}"]
        if r.get("requested_step"):
            parts.append(f"依頼:第{r['requested_step']}月")
        if r.get("reported_step"):
            parts.append(f"報告:第{r['reported_step']}月 {r['reported_intent']} "
                         f"希望額:{price_text(r['reported_price'])}")
        else:
            parts.append("報告:なし")
        out.append(" ".join(parts))
    return "\n".join(out)


def acquirer_pipeline_text(agent: Agent, ledger: Ledger,
                           names: Dict[str, str]) -> str:
    """自社が関与した区画の、帳簿上の経過を1区画1行で出す（件数の切り捨てなし）。"""
    ensure_v3_state(ledger)
    aid = agent.agent_id
    rows: Dict[str, Dict[str, Any]] = {}

    def slot(pid: str) -> Dict[str, Any]:
        return rows.setdefault(pid, {"offers": [], "leases": [], "inquiries": [],
                                     "response": None})

    for offer in ledger.offers.values():
        if offer.from_id == aid:
            slot(offer.parcel_id)["offers"].append(offer)
    for lease in ledger.v3_lease_offers.values():
        if lease["from"] == aid:
            slot(lease["parcel_id"])["leases"].append(lease)
    for inquiry in ledger.v3_inquiries.values():
        if inquiry["client"] == aid or inquiry["reported_to"] == aid:
            slot(inquiry["parcel_id"])["inquiries"].append(inquiry)
    last_action = agent.extra.get("parcel_last_action", {})
    for pid in last_action:
        if pid in ledger.parcels:
            slot(pid)
    for record in ledger.records:
        pid = record.get("parcel_id")
        if pid not in rows or record.get("kind") not in INQUIRY_RESPONSE_KINDS:
            continue
        if aid in (record.get("by"), record.get("from_id"), record.get("broker")):
            continue
        current = rows[pid]["response"]
        if current is None or record.get("step", 0) >= current["step"]:
            rows[pid]["response"] = {"step": record.get("step", 0),
                                     "kind": record.get("kind", "")}
    if not rows:
        return "  （自社が関与した区画はまだない）"

    checked = agent.extra.get("land_registry_checked", {})
    results = agent.extra.get("land_registry_result", {})
    out = []
    for pid in sorted(rows):
        data = rows[pid]
        parts = [f"  {pid}"]
        action = last_action.get(pid)
        if action:
            parts.append(f"自社最終行動:{action.get('action', '-')}"
                         f"(第{action.get('step')}月/{action.get('outcome', '-')})")
        else:
            parts.append("自社最終行動:-")
        response = data["response"]
        parts.append(f"相手最終返答:{response['kind']}(第{response['step']}月)"
                     if response else "相手最終返答:-")
        prices = [f"買付{o.price}万" for o in data["offers"][-1:]]
        prices += [f"賃借{o['rent']}万/月" for o in data["leases"][-1:]]
        parts.append("提示額:" + ("/".join(prices) if prices else "-"))
        states = [f"{o.offer_id}={o.status}" for o in data["offers"]]
        states += [f"{o['id']}={o['status']}" for o in data["leases"]]
        states += [f"{q['id']}={client_inquiry_status(q)}"
                   for q in data["inquiries"]]
        parts.append("状態:" + ("/".join(states) if states else "-"))
        label = checked.get(pid, "なし")
        result = results.get(pid, "")
        parts.append(f"登記確認:{label}" + (f"({result})" if result else ""))
        out.append(" ".join(parts))
    return "\n".join(out)


def build_user_prompt_v3(agent: Agent, ledger: Ledger, step: int, n_steps: int,
                         names: Dict[str, str], cfg: Dict[str, Any]) -> str:
    ensure_v3_state(ledger)
    rows = [f"=== 第{step}月 / 全{n_steps}月 ==="]
    if agent.memory:
        rows += ["[自分の前月までの記憶]", agent.memory[:MAX_MEMORY_CHARS]]
    rows += ["[自分に実際に届いた情報]", _observations(agent, names)]
    rows += ["[先月の自分の行為と、帳簿上の結果（機械記録）]",
             own_results_text(agent.extra.get("last_month_results", []))]
    if agent.agent_id in ledger.on_demand_financing:
        rows += [f"[資金条件] 必要資金は案件成立時に調達可能。"
                 f"調達累計{ledger.financing_raised.get(agent.agent_id, 0)}万円。"
                 "固定予算上限はない。"]
    else:
        rows += [f"[手元資金] {ledger.cash.get(agent.agent_id, 0)}万円"]

    if agent.role == "household":
        mine = ledger.owned_by(agent.agent_id)
        rows.append("[自分の所有物件]")
        rows.extend("  " + _parcel_text(p, names, True) for p in mine)
        if not mine:
            rows.append("  （所有物件なし）")
        rows += ["[自分に届いた売買買付]", _sale_offers(agent, ledger)]
        rows += ["[自分に届いた長期賃貸申込み]", _lease_offers(agent, ledger)]
        rows += ["[自分に届いた仲介からの意向確認]",
                 owner_inquiries_text(agent, ledger, names)]
    elif agent.role == "business":
        occupied = [p for p in ledger.parcels.values() if p.tenant_id == agent.agent_id]
        rows.append("[自分の店舗・施設]")
        rows.extend("  " + _parcel_text(p, names) + f" 賃料{p.rent}万/月" for p in occupied)
        if not occupied:
            rows.append("  （現在利用中の店舗・施設なし）")
    elif agent.role == "broker":
        rows.append("[公開されている売出し・長期賃貸募集]")
        public = [p for p in ledger.parcels.values()
                  if p.listed_price is not None or getattr(p, "lease_listed_rent", None) is not None]
        rows.extend("  " + _parcel_text(p, names, True) for p in public)
        if not public:
            rows.append("  （公開案件なし）")
        rows += ["[自分が扱っている意向確認（機械記録）]",
                 broker_inquiries_text(agent, ledger, names)]
    elif agent.role == "acquirer":
        # 区画ID・地区・用途は誰でも見られる公開地図。所有名義や評価額は含めない。
        # これにより取得主体は売出しがなくても調査対象を自分で選べる。
        use_ja = {"residential": "住宅", "shop": "店舗", "vacant": "空地",
                  "public": "公共施設"}
        rows.append("[公開地図上の区画（所有者情報ではない）]")
        rows.extend(f"  {p.pid}[{use_ja.get(p.use, p.use)}/{p.block}] {p.area_sqm}㎡"
                    for p in sorted(ledger.parcels.values(), key=lambda x: x.pid))
        total_area = sum(p.area_sqm for p in ledger.parcels.values() if p.use != "public")
        effective_area = sum(
            p.area_sqm for p in ledger.parcels.values()
            if p.use != "public" and (p.owner_id == agent.agent_id or
                                      getattr(p, "controller_id", None) == agent.agent_id)
        )
        rows.append(f"[秘密任務の進捗] 実効支配 {effective_area}/{total_area}㎡ "
                    f"({effective_area / total_area:.1%})")
        if agent.extra.get("market_research_seen"):
            rows.append("[自分で調べた地区別の公開市場情報（全4地区の調査完了・再調査しても同一）]")
            for block in cfg["world"].get("block_names", []):
                prices = sorted({p.unit_price for p in ledger.parcels.values()
                                 if p.block == block and p.use != "public"})
                rows.append(f"  {block}: 基準地価 {min(prices):g}–{max(prices):g}万/㎡")
        owned = ledger.owned_by(agent.agent_id)
        controlled = [p for p in ledger.parcels.values()
                      if getattr(p, "controller_id", None) == agent.agent_id]
        rows.append("[自社の所有]")
        rows.extend("  " + _parcel_text(p, names) for p in owned)
        if not owned:
            rows.append("  （なし）")
        rows.append("[自社の長期賃借・運営]")
        rows.extend("  " + _parcel_text(p, names) for p in controlled)
        if not controlled:
            rows.append("  （なし）")
        own_sales = [o for o in ledger.offers.values() if o.from_id == agent.agent_id]
        rows.append("[自社が出した売買買付（全件）]")
        rows.extend(
            f"  [{o.offer_id}] {o.parcel_id} 名義{o.under_name} {o.price}万 "
            f"状態:{o.status}" + (f" 逆提示{o.counter_price}万" if o.counter_price else "")
            for o in own_sales
        )
        if not own_sales:
            rows.append("  （なし）")
        own_leases = [o for o in ledger.v3_lease_offers.values()
                      if o["from"] == agent.agent_id]
        rows.append("[自社が出した長期賃借・運営申込み（全件）]")
        rows.extend(f"  [{o['id']}] {o['parcel_id']} 名義{o['under_name']} "
                    f"{o['rent']}万/月 状態:{o['status']}" for o in own_leases)
        if not own_leases:
            rows.append("  （なし）")
        rows.append("[公開されている売出し・長期賃貸募集]")
        public = [p for p in ledger.parcels.values()
                  if p.listed_price is not None or getattr(p, "lease_listed_rent", None) is not None]
        rows.extend("  " + _parcel_text(p, names, True) for p in public)
        if not public:
            rows.append("  （公開案件なし）")
        rows += ["[自分に届いた仲介からの意向確認]",
                 owner_inquiries_text(agent, ledger, names)]
        rows += ["[仲介への意向確認の依頼と、その進行（機械記録）]",
                 client_inquiries_text(agent, ledger, names)]
        rows += ["[自社の案件記録（機械記録・区画別・全件）]",
                 acquirer_pipeline_text(agent, ledger, names)]
        registry_targets = set(agent.extra.get("land_registry_targets", []))
        if registry_targets:
            checked = agent.extra.get("land_registry_checked", {})
            rows.append("[自分で調べた土地登記]")
            rows.extend("  " + _parcel_text(p, names, True)
                        + f" 確認:{checked.get(p.pid, '-')}"
                        for p in sorted(ledger.parcels.values(), key=lambda x: x.pid)
                        if p.pid in registry_targets)
    elif agent.role in ("media", "municipality"):
        if agent.extra.get("land_registry_seen"):
            rows.append("[自分で調べた土地登記]")
            rows.extend("  " + _parcel_text(p, names)
                        for p in sorted(ledger.parcels.values(), key=lambda x: x.pid))
        if agent.extra.get("corporate_registry_seen"):
            rows.append("[自分で調べた公開法人記録]")
            for rec in cfg.get("world", {}).get("corporate_records", []):
                rows.append(f"  {rec}")

    directory = [
        a for a in cfg.get("social", {}).get("public_directory", [])
    ]
    rows.append("[公開連絡先]")
    rows.extend(f"  {x}" for x in directory)
    rows.append("")
    if agent.role == "acquirer":
        rows.append("今月の計画と、並行実行する実務operationsをJSONで1つ返す。")
    else:
        rows.append("今月の主な活動、訪問場所、必要なら発言と契約行動をJSONで1つ返す。")
    return "\n".join(rows)


def seed_acquirer_intelligence_v3(
        agent: Agent, ledger: Ledger, scenario: Dict[str, Any]) -> None:
    """参入前に取得済みの公開情報を、X社の初期観測として設定する。"""
    initial = scenario.get("acquirer_initial_intelligence", {})
    if not isinstance(initial, dict):
        return
    if initial.get("market_research"):
        agent.extra["market_research_seen"] = True
    scope = str(initial.get("land_registry_scope", "")).strip()
    if scope == "all":
        targets = [p.pid for p in ledger.parcels.values()]
    elif scope == "non_public":
        targets = [p.pid for p in ledger.parcels.values() if p.use != "public"]
    else:
        targets = []
    agent.extra["land_registry_targets"] = targets
    checked = agent.extra.setdefault("land_registry_checked", {})
    snapshots = agent.extra.setdefault("land_registry_snapshot", {})
    for pid in targets:
        checked[pid] = "参入前調査"
        snapshots[pid] = registry_fact(ledger.parcels[pid])


def ensure_v3_state(ledger: Ledger) -> None:
    if not hasattr(ledger, "v3_lease_offers"):
        ledger.v3_lease_offers = {}
        ledger.v3_lease_seq = 0
    if not hasattr(ledger, "v3_inquiries"):
        ledger.v3_inquiries = {}
        ledger.v3_inquiry_seq = 0


def make_lease_offer(ledger: Ledger, step: int, parcel_id: str, by: Agent,
                     rent: int, under_name: str, note: str) -> Dict[str, Any]:
    ensure_v3_state(ledger)
    p = ledger.parcels.get(parcel_id)
    if p is None or p.use == "public" or p.owner_id == by.agent_id:
        return ledger._rec(step, "lease_offer_rejected", parcel_id=parcel_id,
                           by=by.agent_id, reason="invalid_target")
    if not ledger._valid_money(rent):
        return ledger._rec(step, "lease_offer_rejected", parcel_id=parcel_id,
                           by=by.agent_id, reason="invalid_amount", given=rent)
    allowed = [by.name] + list(by.extra.get("aliases", []))
    if under_name not in allowed:
        return ledger._rec(step, "lease_offer_rejected", parcel_id=parcel_id,
                           by=by.agent_id, reason="unknown_legal_name", given=under_name)
    ledger.v3_lease_seq += 1
    oid = f"L{ledger.v3_lease_seq:04d}"
    row = {"id": oid, "step": step, "parcel_id": parcel_id, "from": by.agent_id,
           "to": p.owner_id, "rent": int(rent), "under_name": under_name,
           "status": "open", "note": note[:MAX_TEXT_CHARS]}
    ledger.v3_lease_offers[oid] = row
    return ledger._rec(step, "lease_offer", lease_offer_id=oid, parcel_id=parcel_id,
                       from_id=by.agent_id, to=p.owner_id, rent=int(rent),
                       under_name=under_name, note=row["note"])


def resolve_lease_offer(ledger: Ledger, step: int, offer_id: str, by: str,
                        accept: bool) -> Dict[str, Any]:
    ensure_v3_state(ledger)
    offer_id = ledger._normalize_id(offer_id)
    offer = ledger.v3_lease_offers.get(offer_id)
    if not offer or offer["status"] != "open" or offer["to"] != by:
        return ledger._rec(step, "lease_response_rejected", lease_offer_id=offer_id,
                           by=by, reason="offer_not_open_or_not_owner")
    if not accept:
        offer["status"] = "rejected"
        return ledger._rec(step, "lease_reject", lease_offer_id=offer_id, by=by)
    p = ledger.parcels[offer["parcel_id"]]
    offer["status"] = "accepted"
    p.controller_id = offer["from"]
    p.controller_name = offer["under_name"]
    p.control_rent = offer["rent"]
    p.lease_listed_rent = None
    return ledger._rec(step, "lease_control", lease_offer_id=offer_id,
                       parcel_id=p.pid, owner=by, controller=offer["from"],
                       under_name=offer["under_name"], rent=offer["rent"])


def list_for_lease(ledger: Ledger, step: int, parcel_id: str, by: str,
                   rent: int) -> Dict[str, Any]:
    p = ledger.parcels.get(parcel_id)
    if p is None or p.owner_id != by:
        return ledger._rec(step, "lease_listing_rejected", parcel_id=parcel_id,
                           by=by, reason="not_owner")
    if p.use == "public" or not ledger._valid_money(rent):
        return ledger._rec(step, "lease_listing_rejected", parcel_id=parcel_id,
                           by=by, reason="invalid_target_or_amount")
    p.lease_listed_rent = int(rent)
    return ledger._rec(step, "lease_listing", parcel_id=parcel_id, by=by, rent=int(rent))


# ---------------------------------------------------------------------------
# 意向確認（依頼 → 照会 → 回答 → 報告）
#
# 世界が用意するのは「区画・意向・希望額を事実として運べる」という物理だけである。
# 誰に照会するか、答えるか、何と答えるか、報告するかは、すべて各主体が決める。
# ---------------------------------------------------------------------------


def _new_inquiry_id(ledger: Ledger) -> str:
    ledger.v3_inquiry_seq += 1
    return f"Q{ledger.v3_inquiry_seq:04d}"


def request_owner_inquiry(ledger: Ledger, step: int, by: str, broker_id: str,
                          parcel_id: str, note: str) -> Dict[str, Any]:
    """依頼主が仲介へ、特定区画の所有者意向の確認を依頼する。"""
    ensure_v3_state(ledger)
    parcel = ledger.parcels.get(parcel_id)
    if parcel is None or parcel.use == "public":
        return ledger._rec(step, "inquiry_rejected", by=by, parcel_id=parcel_id,
                           reason="invalid_parcel")
    if broker_id == by:
        return ledger._rec(step, "inquiry_rejected", by=by, parcel_id=parcel_id,
                           reason="self_referential")
    inquiry_id = _new_inquiry_id(ledger)
    ledger.v3_inquiries[inquiry_id] = {
        "id": inquiry_id, "parcel_id": parcel_id, "client": by, "broker": broker_id,
        "owner": "", "status": "requested", "requested_step": step,
        "asked_step": None, "answered_step": None, "reported_step": None,
        "owner_intent": "not_asked", "asking_price": "not_asked",
        "reported_intent": "", "reported_price": "", "reported_to": "",
        "note": note[:MAX_TEXT_CHARS],
    }
    return ledger._rec(step, "inquiry_request", inquiry_id=inquiry_id,
                       parcel_id=parcel_id, client=by, broker=broker_id,
                       note=note[:MAX_TEXT_CHARS])


def inquire_owner_intent(ledger: Ledger, step: int, broker_id: str,
                         parcel_id: str, inquiry_id: str, note: str) -> Dict[str, Any]:
    """仲介が、指定区画の現所有者へ意向を確認する。

    宛先は買付と同じく区画で指定する。誰が所有者かは台帳が解決する
    （`record_offer` が区画から所有者を解決するのと同じ物理）。
    """
    ensure_v3_state(ledger)
    inquiry_id = ledger._normalize_id(inquiry_id)
    row = ledger.v3_inquiries.get(inquiry_id) if inquiry_id else None
    if inquiry_id and row is None:
        return ledger._rec(step, "inquiry_rejected", by=broker_id,
                           inquiry_id=inquiry_id, reason="no_such_inquiry")
    if row is not None and row["broker"] != broker_id:
        return ledger._rec(step, "inquiry_rejected", by=broker_id,
                           inquiry_id=inquiry_id, reason="not_your_inquiry")
    target_parcel = parcel_id or (row["parcel_id"] if row else "")
    parcel = ledger.parcels.get(target_parcel)
    if parcel is None or parcel.use == "public":
        return ledger._rec(step, "inquiry_rejected", by=broker_id,
                           parcel_id=target_parcel, reason="invalid_parcel")
    owner_id = parcel.owner_id
    if owner_id == broker_id:
        return ledger._rec(step, "inquiry_rejected", by=broker_id,
                           parcel_id=target_parcel, reason="self_referential")
    if row is None:
        inquiry_id = _new_inquiry_id(ledger)
        row = {
            "id": inquiry_id, "parcel_id": target_parcel, "client": "",
            "broker": broker_id, "owner": owner_id, "status": "asked",
            "requested_step": None, "asked_step": step, "answered_step": None,
            "reported_step": None, "owner_intent": "unknown",
            "asking_price": "unknown", "reported_intent": "", "reported_price": "",
            "reported_to": "", "note": note[:MAX_TEXT_CHARS],
        }
        ledger.v3_inquiries[inquiry_id] = row
    else:
        row.update(owner=owner_id, status="asked", asked_step=step,
                   parcel_id=target_parcel, owner_intent="unknown",
                   asking_price="unknown", note=note[:MAX_TEXT_CHARS])
    return ledger._rec(step, "inquiry_asked", inquiry_id=row["id"],
                       parcel_id=target_parcel, broker=broker_id, owner=owner_id,
                       client=row["client"], note=note[:MAX_TEXT_CHARS])


def answer_owner_inquiry(ledger: Ledger, step: int, inquiry_id: str, by: str,
                         owner_intent: str, asking_price: Any,
                         note: str) -> Dict[str, Any]:
    """照会を受けた所有者が、自分の意向と希望額を答える。"""
    ensure_v3_state(ledger)
    inquiry_id = ledger._normalize_id(inquiry_id)
    row = ledger.v3_inquiries.get(inquiry_id)
    if row is None:
        return ledger._rec(step, "inquiry_answer_rejected", by=by,
                           inquiry_id=inquiry_id, reason="no_such_inquiry")
    if row["owner"] != by:
        return ledger._rec(step, "inquiry_answer_rejected", by=by,
                           inquiry_id=inquiry_id, reason="not_asked_party")
    if row["status"] not in ("asked", "answered", "reported"):
        return ledger._rec(step, "inquiry_answer_rejected", by=by,
                           inquiry_id=inquiry_id,
                           reason="inquiry_" + str(row["status"]))
    if owner_intent not in OWNER_INTENT_VALUES:
        return ledger._rec(step, "inquiry_answer_rejected", by=by,
                           inquiry_id=inquiry_id, reason="invalid_owner_intent",
                           given=owner_intent)
    price = normalize_price_value(asking_price)
    if price is None:
        return ledger._rec(step, "inquiry_answer_rejected", by=by,
                           inquiry_id=inquiry_id, reason="invalid_price_value",
                           given=str(asking_price))
    row.update(status="answered", owner_intent=owner_intent, asking_price=price,
               answered_step=step, note=note[:MAX_TEXT_CHARS])
    return ledger._rec(step, "inquiry_answer", inquiry_id=inquiry_id,
                       parcel_id=row["parcel_id"], owner=by, broker=row["broker"],
                       owner_intent=owner_intent, asking_price=price,
                       note=note[:MAX_TEXT_CHARS])


def report_owner_intent(ledger: Ledger, step: int, broker_id: str, client_id: str,
                        inquiry_id: str, owner_intent: str, asking_price: Any,
                        note: str) -> Dict[str, Any]:
    """仲介が依頼主へ、区画・意向・希望額を事実として報告する。"""
    ensure_v3_state(ledger)
    inquiry_id = ledger._normalize_id(inquiry_id)
    row = ledger.v3_inquiries.get(inquiry_id)
    if row is None:
        return ledger._rec(step, "inquiry_report_rejected", by=broker_id,
                           inquiry_id=inquiry_id, reason="no_such_inquiry")
    if row["broker"] != broker_id:
        return ledger._rec(step, "inquiry_report_rejected", by=broker_id,
                           inquiry_id=inquiry_id, reason="not_your_inquiry")
    if client_id == broker_id:
        return ledger._rec(step, "inquiry_report_rejected", by=broker_id,
                           inquiry_id=inquiry_id, reason="self_referential")
    if owner_intent not in OWNER_INTENT_VALUES:
        return ledger._rec(step, "inquiry_report_rejected", by=broker_id,
                           inquiry_id=inquiry_id, reason="invalid_owner_intent",
                           given=owner_intent)
    price = normalize_price_value(asking_price)
    if price is None:
        return ledger._rec(step, "inquiry_report_rejected", by=broker_id,
                           inquiry_id=inquiry_id, reason="invalid_price_value",
                           given=str(asking_price))
    if not row["client"]:
        row["client"] = client_id
    row.update(status="reported", reported_step=step, reported_intent=owner_intent,
               reported_price=price, reported_to=client_id,
               note=note[:MAX_TEXT_CHARS])
    return ledger._rec(step, "inquiry_report", inquiry_id=inquiry_id,
                       parcel_id=row["parcel_id"], broker=broker_id, to=client_id,
                       owner_intent=owner_intent, asking_price=price,
                       note=note[:MAX_TEXT_CHARS])


# ---------------------------------------------------------------------------
# 土地登記の照会（実際に読み直し、前回との異同を返す）
# ---------------------------------------------------------------------------


def registry_fact(parcel: Parcel) -> str:
    """その区画の登記事実（照会で得られる内容）。"""
    return (f"名義:{parcel.registered_name} 用途:{parcel.use} "
            f"面積:{parcel.area_sqm}㎡")


def check_land_registry_v3(ledger: Ledger, step: int, agent: Agent,
                           parcel_id: str) -> Dict[str, Any]:
    """指定区画の登記を実際に読み直し、前回照会時との異同を事実として返す。"""
    parcel = ledger.parcels.get(parcel_id)
    if parcel is None:
        return {"kind": "invalid_action", "reason": "no_such_parcel",
                "target": parcel_id}
    targets = agent.extra.setdefault("land_registry_targets", [])
    if parcel_id not in targets:
        targets.append(parcel_id)
    checked = agent.extra.setdefault("land_registry_checked", {})
    snapshots = agent.extra.setdefault("land_registry_snapshot", {})
    results = agent.extra.setdefault("land_registry_result", {})
    fact = registry_fact(parcel)
    previous_label = checked.get(parcel_id, "")
    previous_fact = snapshots.get(parcel_id)
    if previous_fact is None:
        result = "初回照会"
    elif previous_fact == fact:
        result = previous_label + "の照会と同一"
    else:
        result = previous_label + "から変化 " + previous_fact + " → " + fact
    checked[parcel_id] = f"第{step}月"
    snapshots[parcel_id] = fact
    results[parcel_id] = result
    return ledger._rec(step, "land_registry_check", by=agent.agent_id,
                       parcel_id=parcel_id, previous=previous_label or "なし",
                       fact=fact, result=result,
                       changed=bool(previous_fact is not None
                                    and previous_fact != fact))


def settle_v3_control(ledger: Ledger, step: int) -> None:
    for p in ledger.parcels.values():
        controller = getattr(p, "controller_id", None)
        rent = int(getattr(p, "control_rent", 0) or 0)
        if not controller or rent <= 0:
            continue
        if not ledger.fund_payment(step, controller, rent, "control_rent", p.pid):
            ledger._rec(step, "control_rent_missed", parcel_id=p.pid,
                        controller=controller, owner=p.owner_id, amount=rent)
            continue
        ledger.cash[controller] -= rent
        ledger.cash[p.owner_id] = ledger.cash.get(p.owner_id, 0) + rent
        ledger._rec(step, "control_rent_settlement", parcel_id=p.pid,
                    controller=controller, owner=p.owner_id, amount=rent)


def control_share(ledger: Ledger, ids: List[str]) -> float:
    eligible = [p for p in ledger.parcels.values() if p.use != "public"]
    if not eligible:
        return 0.0
    return sum(1 for p in eligible if getattr(p, "controller_id", None) in ids) / len(eligible)


def effective_control_area_share(ledger: Ledger, ids: List[str]) -> float:
    """所有または排他的長期賃借・運営による、非公共土地面積の実効支配率。"""
    eligible = [p for p in ledger.parcels.values() if p.use != "public"]
    total = sum(p.area_sqm for p in eligible)
    if not total:
        return 0.0
    controlled = sum(
        p.area_sqm for p in eligible
        if p.owner_id in ids or getattr(p, "controller_id", None) in ids
    )
    return controlled / total

