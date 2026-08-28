"""A市フィールド v5: 出来事はピン留め・観測するのは「街の会話」。

設計の正は `docs/world_design_v5_pinned_events.md`（草案）と
`docs/world_design_v5_impl.md`（実装仕様）。v1〜v4.1b のファイルは変更しない。

v4系との決定的な違い:
  1. 取得（名義の移転）は **台本** である。LLM は売買を決めない。
     X社は台帳上の名義の器としてのみ存在し、一度も LLM を呼ばれない。
  2. 各取得が世界に残す **兆候（trace）** を、台本が指定した相手・場面にだけ配る。
  3. 月は **シーン4本**（各2ラウンドの往復会話）であり、発話が主体の唯一の行為である。
  4. 主体の出力は {thought, text, talk_to, direct}（＋記者の publish・所有者の stance）だけ。
     stance は台帳を動かさない（買い手は台本であり、姿勢は取引ではない）。

ここに置いてよいのは可能な行為・登記・同席・配送・記録だけである。
売却確率・閾値・強制イベント・当為・「気づかせるための仕組み」は置かない。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .agents import Agent
from .world import Ledger, neighbors

MAX_THOUGHT_CHARS = 250
MAX_TEXT_CHARS = 150
MAX_PUBLISH_CHARS = 300
DEFAULT_SCENE_ROUNDS = 2
DEFAULT_DIRECT_QUOTA = 2

USE_JA = {"residential": "住宅", "shop": "店舗", "lodging": "宿泊",
          "office": "事務所", "vacant": "空地", "public": "公共施設"}

ROLE_TEXT_V5 = {
    "household": "A市で日常生活を送る住民世帯。自分の土地・建物を持っている。",
    "business": "A市で事業を営む地元事業者。営業、仕入れ、人員、設備、顧客対応が日常業務である。",
    "broker": "A市の不動産仲介。地域の人づきあいの中にいる。",
    "acquirer": "街の外から不動産を取得・運用する会社。",
    "municipality": "A市の自治体担当。窓口業務を含む通常業務を送る。",
    "media": "A市を扱う地域記者。通常の地域取材を行う。",
}

# --- シーン ---------------------------------------------------------------
# S1〜S3 は行き先を主体が選ぶ日常の場面。S4 は月替わりの窓口で、行くかどうかを主体が選ぶ。

SCENE_IDS = ["S1", "S2", "S3", "S4"]

SCENE_LABELS = {
    "S1": "朝の商店街（買い物・すれ違い）",
    "S2": "昼の職場・店（同僚・常連）",
    "S3": "夕方の公園・居酒屋（雑談）",
}

# (kind, 会場ID, 場面の説明)
S4_ROTATION: List[Tuple[str, str, str]] = [
    ("assembly", "V02", "町内会（地区公民館の寄り合い）"),
    ("counter", "V06", "市役所の窓口（待合と受付）"),
    ("broker_front", "V07", "仲介の店先（不動産仲介の事務所）"),
    ("press", "V08", "記者の取材（地域イベント会場に記者が来ている）"),
]

HOME = "HOME"

TRACE_TEXTS = {
    "registry": "第{month}月　{parcel}　名義が{old}から{new}に変わった",
    "broker_known": "{parcel}の名義が{new}に変わる件を、仕事の中で知った",
    "moving_out": "{parcel}の住人・店が引き払った。荷物が運び出されている",
    "sign_change": "{parcel}の看板が外され、別の名前の看板が掛かった",
    "construction": "{parcel}で工事が始まった。足場が組まれている",
    "survey": "{parcel}で測量をしている人たちがいる",
    "strangers": "{parcel}を見慣れない人たちが見て回っている",
    "tenant_swap": "{parcel}の店子が入れ替わった",
    "renovation_sweep": "{parcel}を含む数棟で同じ時期に改修が始まった",
}

STANCE_VALUES = ["sell", "keep"]
# Gemini の response_schema は enum に空文字を許さない（実APIで 400 INVALID_ARGUMENT）。
# 「誰にも出さない」を表す番兵を置く。
DIRECT_NONE = "NONE"


def s4_for_step(step: int) -> Tuple[str, str, str]:
    """第step月の S4（月替わりの窓口）。(kind, venue_id, label)"""
    return S4_ROTATION[(step - 1) % len(S4_ROTATION)]


# ---------------------------------------------------------------------------
# 世界の状態
# ---------------------------------------------------------------------------

def ensure_v5_state(ledger: Ledger) -> None:
    if not hasattr(ledger, "v5_utterances"):
        ledger.v5_utterances = []      # 発話（配送先つき）
        ledger.v5_traces_seen = []     # 誰にどの兆候が見えたか
        ledger.v5_plans = []           # 月ごとの行き先
        ledger.v5_stances = []         # 月末の姿勢（台帳を動かさない）
        ledger.v5_articles = []        # 記者の記事
        ledger.v5_directs = []         # 私信
        ledger.v5_utt_seq = 0
        ledger.v5_deals = []           # 当事者が知っている自分の取引（売買・賃借）
        # 開始時点で街の人が知っている名前（以後この辞書は更新しない）
        ledger.v5_initial_names = {p.pid: p.registered_name
                                   for p in ledger.parcels.values()}


# ---------------------------------------------------------------------------
# 台本（events）
# ---------------------------------------------------------------------------

def load_script_v5(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, encoding="utf-8") as f:
        script = yaml.safe_load(f)
    validate_script_v5(script)
    return script


def validate_script_v5(script: Dict[str, Any]) -> None:
    seen, parcels_seen = set(), set()
    months = int((script.get("meta") or {}).get("months", 60))
    for acq in script.get("acquisitions", []):
        for key in ("id", "month", "parcel_id", "under_name"):
            if key not in acq:
                raise ValueError(f"台本の項目が欠けている: {key} in {acq}")
        if acq["id"] in seen:
            raise ValueError(f"台本のIDが重複している: {acq['id']}")
        seen.add(acq["id"])
        if not (1 <= int(acq["month"]) <= months):
            raise ValueError(f"台本の月が範囲外: {acq}")
        if acq["parcel_id"] in parcels_seen:
            raise ValueError(f"同じ区画が2度取得されている: {acq['parcel_id']}")
        parcels_seen.add(acq["parcel_id"])
        if "kind" in acq:
            if acq["kind"] not in ("sale", "lease"):
                raise ValueError(f"未知の取引種別: {acq}")
            if not str(acq.get("note", "")).strip():
                raise ValueError(f"取引の事情(note)が無い: {acq['id']}")
            if acq["kind"] == "lease" and [t for t in acq.get("traces", [])
                                           if t.get("kind") == "registry"]:
                raise ValueError(f"賃借に登記の兆候が付いている: {acq['id']}")
        for tr in acq.get("traces", []):
            if tr.get("kind") not in TRACE_TEXTS:
                raise ValueError(f"未知の兆候: {tr}")
            aud = str(tr.get("audience", ""))
            if not (aud in ("registry", "neighbors")
                    or aud.startswith("venue:") or aud.startswith("agents:")):
                raise ValueError(f"未知の audience: {tr}")


def acquisitions_at(script: Dict[str, Any], step: int) -> List[Dict[str, Any]]:
    return [a for a in script.get("acquisitions", []) if int(a["month"]) == step]


def apply_script_v5(ledger: Ledger, step: int, script: Dict[str, Any],
                    acquirer_id: str) -> List[Dict[str, Any]]:
    """台本どおりに名義を移す。金銭は動かない（世界に存在しない）。

    これは「世界で起きた事実」であって主体の判断ではない。ここで主体に何かを
    させることはしない（売主には自分の登記事実として結果が渡るだけ）。
    """
    ensure_v5_state(ledger)
    done: List[Dict[str, Any]] = []
    for acq in acquisitions_at(script, step):
        parcel = ledger.parcels.get(acq["parcel_id"])
        if parcel is None:
            ledger._rec(step, "script_rejected", acq_id=acq["id"],
                        parcel_id=acq["parcel_id"], reason="no_such_parcel")
            continue
        if parcel.use == "public":
            ledger._rec(step, "script_rejected", acq_id=acq["id"],
                        parcel_id=parcel.pid, reason="public_land")
            continue
        deal = str(acq.get("kind", "sale"))
        holder = str(acq["under_name"])
        note = str(acq.get("note", ""))
        party = parcel.owner_id            # 取引の当事者（売主／貸主）
        old_name = parcel.registered_name or party
        if deal == "lease":
            # 賃借は登記が動かない（施主追記 2026-08-27 22:27）。使い手だけが変わる。
            parcel.tenant_id = acquirer_id
            rec = ledger._rec(step, "lease", acq_id=acq["id"], parcel_id=parcel.pid,
                              lessor=party, lessee=acquirer_id, under_name=holder,
                              old_name=old_name)
        else:
            parcel.owner_id = acquirer_id
            parcel.registered_name = holder
            rec = ledger._rec(step, "transfer", acq_id=acq["id"], parcel_id=parcel.pid,
                              seller=party, buyer=acquirer_id,
                              under_name=holder, old_name=old_name)
        # 当事者は自分の取引を知っている（他人事の「名義が変わった」ではない）。
        # kind を持つ台本（v5b）だけの挙動にして、v5 の再実行結果を変えない
        # （Codexレビュー 2026-08-28）。
        if "kind" in acq:
            ledger.v5_deals.append({"step": step, "agent_id": party,
                                    "acq_id": acq["id"], "parcel_id": parcel.pid,
                                    "holder": holder, "kind": deal, "note": note})
        done.append({**rec, "acq": acq})
    return done


# ---------------------------------------------------------------------------
# 兆候（trace）の配布
# ---------------------------------------------------------------------------

def _pname(pnames: Optional[Dict[str, str]], pid: str) -> str:
    """区画の呼び名。`pnames` を渡さない版（v5c 以前）は内部IDのままである。"""
    return pnames.get(pid, pid) if pnames else pid


def _trace_text(kind: str, acq: Dict[str, Any], month: int,
                old_name: str, pnames: Optional[Dict[str, str]] = None,
                texts: Optional[Dict[str, str]] = None) -> str:
    return (texts or TRACE_TEXTS)[kind].format(
        month=month, parcel=_pname(pnames, acq["parcel_id"]),
        old=old_name, new=acq["under_name"])


def _occupants(ledger: Ledger, pid: str) -> List[str]:
    parcel = ledger.parcels.get(pid)
    if parcel is None:
        return []
    out = [parcel.owner_id]
    if parcel.tenant_id:
        out.append(parcel.tenant_id)
    return out


def ambient_traces_v5(ledger: Ledger, step: int, script: Dict[str, Any],
                      old_names: Dict[str, str],
                      acquirer_id: str,
                      pnames: Optional[Dict[str, str]] = None,
                      texts: Optional[Dict[str, str]] = None
                      ) -> Dict[str, List[Dict[str, Any]]]:
    """会場に依らず見える兆候（neighbors / agents 指名）を、見える主体へ配る。

    venue: と registry はシーンで解決するのでここには含めない。
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for acq in script.get("acquisitions", []):
        for tr in acq.get("traces", []):
            if int(tr.get("month", acq["month"])) != step:
                continue
            aud = str(tr["audience"])
            targets: List[str] = []
            if aud == "neighbors":
                for nb in neighbors(ledger.parcels, acq["parcel_id"]):
                    targets.extend(_occupants(ledger, nb))
            elif aud.startswith("agents:"):
                body = aud[len("agents:"):].strip().strip("[]")
                targets = [x.strip().strip("'\"") for x in body.split(",") if x.strip()]
            else:
                continue
            text = _trace_text(tr["kind"], acq, step,
                               old_names.get(acq["parcel_id"], ""), pnames, texts)
            for aid in dict.fromkeys(targets):
                if not aid or aid == acquirer_id:
                    continue     # 買い手は主体ではない（観測を持たない）
                out.setdefault(aid, []).append(
                    {"kind": tr["kind"], "acq_id": acq["id"],
                     "parcel_id": acq["parcel_id"], "text": text,
                     "audience": aud})
    return out


def venue_traces_v5(step: int, script: Dict[str, Any], venue_id: str,
                    old_names: Dict[str, str],
                    pnames: Optional[Dict[str, str]] = None,
                    texts: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """その月にその会場で見える兆候（venue: 指定）。"""
    rows: List[Dict[str, Any]] = []
    for acq in script.get("acquisitions", []):
        for tr in acq.get("traces", []):
            if int(tr.get("month", acq["month"])) != step:
                continue
            aud = str(tr["audience"])
            if not aud.startswith("venue:") or aud[len("venue:"):].strip() != venue_id:
                continue
            rows.append({"kind": tr["kind"], "acq_id": acq["id"],
                         "parcel_id": acq["parcel_id"], "audience": aud,
                         "text": _trace_text(tr["kind"], acq, step,
                                             old_names.get(acq["parcel_id"], ""),
                                             pnames, texts)})
    return rows


def registry_rows_v5(ledger: Ledger, step: int) -> List[str]:
    """役所の窓口で閲覧できる登記（その月までの名義変更の全件）。

    公開情報の性質そのものであり、誰かに気づかせるための通知ではない
    （窓口に出向いた主体だけが、出向いた月に見る）。
    """
    rows = []
    for rec in ledger.records:
        if rec.get("kind") != "transfer" or rec.get("step", 0) > step:
            continue
        rows.append(f"  第{rec['step']}月　{rec['parcel_id']}　"
                    f"名義が{rec.get('old_name', '')}から{rec.get('under_name', '')}に変わった")
    return rows or ["  （名義変更の記録はない）"]


# ---------------------------------------------------------------------------
# 観測行（プロンプトの素材）
# ---------------------------------------------------------------------------

def own_parcels_rows_v5(agent: Agent, ledger: Ledger,
                        names: Dict[str, str],
                        pnames: Optional[Dict[str, str]] = None) -> List[str]:
    rows = []
    for parcel in sorted(ledger.owned_by(agent.agent_id), key=lambda p: p.pid):
        line = (f"  {_pname(pnames, parcel.pid)}"
                f"[{USE_JA.get(parcel.use, parcel.use)}/{parcel.block}] "
                f"{parcel.area_sqm}㎡ 名義:{parcel.registered_name}")
        if parcel.tenant_id:
            tenant = names.get(parcel.tenant_id, parcel.tenant_id)
            for deal in getattr(ledger, "v5_deals", []):
                if deal.get("parcel_id") == parcel.pid and deal.get("kind") == "lease":
                    tenant = deal["holder"]
                    break
            line += f" 利用者:{tenant}"
        rows.append(line)
    occupied = [p for p in ledger.parcels.values() if p.tenant_id == agent.agent_id]
    for parcel in sorted(occupied, key=lambda p: p.pid):
        rows.append(f"  {_pname(pnames, parcel.pid)}"
                    f"[{USE_JA.get(parcel.use, parcel.use)}/{parcel.block}] "
                    f"（自分が使っている・名義:{parcel.registered_name}）")
    return rows or ["  （所有・使用している物件はない）"]


def own_history_rows_v5(agent: Agent, ledger: Ledger, step: int,
                        pnames: Optional[Dict[str, str]] = None) -> List[str]:
    """自分がした取引（当事者としての事実）。

    施主追記 2026-08-27 22:27：所有者の知らないうちに登記が変わるのは詐欺なので、
    取引は所有者との間で成立している。当事者は「自分が売った／貸した」ことと、
    その事情を知っている。誰かに話すかどうかは本人の thought が決める。
    """
    rows = []
    for deal in getattr(ledger, "v5_deals", []):
        if deal.get("agent_id") != agent.agent_id or int(deal.get("step", 0)) > step:
            continue
        verb = "賃貸した" if deal.get("kind") == "lease" else "売却した"
        line = (f"  第{deal['step']}月　あなたは "
                f"{_pname(pnames, deal['parcel_id'])} を "
                f"{deal['holder']} へ{verb}")
        if deal.get("note"):
            line += f"（事情：{deal['note']}）"
        rows.append(line)
    # 台本に kind を持たない旧版（v5）の互換：登記の記録から拾う
    if not rows:
        for rec in ledger.records:
            if rec.get("kind") != "transfer" or rec.get("seller") != agent.agent_id:
                continue
            if rec.get("step", 0) > step:
                continue
            rows.append(f"  第{rec['step']}月　"
                        f"{_pname(pnames, rec['parcel_id'])}　"
                        f"の名義が{rec.get('under_name', '')}に移った")
    return rows


def neighbourhood_rows_v5(agent: Agent, ledger: Ledger,
                          names: Dict[str, str],
                          pnames: Optional[Dict[str, str]] = None) -> List[str]:
    """隣の区画について、日ごろ目に入る範囲。

    **現在の登記名義は出さない**（Codexレビュー 2026-08-27）。名義の変化を毎月ここに
    出すと、台本が決めた可視範囲（兆候・窓口の閲覧・人の話）を迂回して全員に
    名義変更が漏れる＝「気づかせる仕組み」になってしまう。
    ここに出すのは開始時点から知っている名前までで、その後の変化は
    兆候を見るか、誰かから聞くか、窓口で登記を見るまで分からない。
    """
    mine = [p.pid for p in ledger.owned_by(agent.agent_id)]
    mine += [p.pid for p in ledger.parcels.values() if p.tenant_id == agent.agent_id]
    seen: List[str] = []
    for pid in dict.fromkeys(mine):
        for nb in neighbors(ledger.parcels, pid):
            if nb not in seen:
                seen.append(nb)
    initial = getattr(ledger, "v5_initial_names", {})
    rows = []
    for pid in sorted(seen):
        parcel = ledger.parcels[pid]
        known = initial.get(pid, "")
        rows.append(f"  {_pname(pnames, pid)}[{USE_JA.get(parcel.use, parcel.use)}] "
                    + (f"もとから{known}のところ" if known else "（誰の土地かは知らない）"))
    return rows or ["  （隣接区画はない）"]


def traces_rows_v5(traces: List[Dict[str, Any]]) -> List[str]:
    if not traces:
        return ["  （とくに目についたことはない）"]
    return [f"  {t['text']}" for t in traces]


def heard_rows_v5(heard: List[Dict[str, Any]], names: Dict[str, str]) -> List[str]:
    if not heard:
        return ["  （まだ何も聞いていない）"]
    rows = []
    for item in heard:
        who = names.get(item.get("from", ""), item.get("from", ""))
        scene = item.get("scene", "")
        venue = item.get("venue_label", item.get("venue", ""))
        to = item.get("talk_to") or []
        to_txt = ("（" + "・".join(names.get(t, t) for t in to) + "に）") if to else ""
        rows.append(f"  [{scene}/{venue}] {who}{to_txt}:「{item.get('text', '')}」")
    return rows


def inbox_rows_v5(agent: Agent, names: Dict[str, str]) -> List[str]:
    if not agent.inbox:
        return ["  （届いたものはない）"]
    rows = []
    for item in agent.inbox:
        if item.get("kind") == "article":
            rows.append(f"  [記事・第{item.get('step')}月] {names.get(item.get('from',''), '')}"
                        f":「{item.get('text', '')}」")
        else:
            rows.append(f"  [私信・第{item.get('step')}月] "
                        f"{names.get(item.get('from',''), item.get('from',''))}"
                        f":「{item.get('text', '')}」")
    return rows


# ---------------------------------------------------------------------------
# 出力スキーマ
# ---------------------------------------------------------------------------

def plan_schema_v5(venue_ids: List[str], s4_venue: str) -> Dict[str, Any]:
    """月の初めの計画。thought と S1〜S4 の行き先だけを返す。

    姿勢（stance）はここでは尋ねない。毎コール「手放すことを考えているか」を
    尋ねると売却の話題を主体に想起させ続ける＝世界側のプライミングになるため
    （Codexレビュー 2026-08-27）。姿勢は月末＝その主体の最後のターンで1回だけ尋ねる。
    """
    day = {"type": "string", "enum": list(venue_ids) + [HOME]}
    props = {
        "thought": {"type": "string"},
        "plan_s1": dict(day),
        "plan_s2": dict(day),
        "plan_s3": dict(day),
        "plan_s4": {"type": "string", "enum": [s4_venue, HOME]},
    }
    return {"type": "object", "properties": props, "required": list(props)}


def scene_schema_v5(agent: Agent, present_ids: List[str], all_ids: List[str],
                    owns_parcel: bool = False,
                    can_publish: bool = False) -> Dict[str, Any]:
    """シーン1ターンの出力。

    `owns_parcel` / `can_publish` は **その月のその主体の最後のターンでだけ True** に
    する（Codexレビュー2巡目）。毎ターン「手放すことを考えているか」を尋ねると
    売却の話題を主体に想起させ続ける＝世界側のプライミングになるため。
    記事も同じ理由で、月内のやりとりを全部聞いたあとの1回に限る。
    """
    others = [p for p in present_ids if p != agent.agent_id]
    elsewhere = [a for a in all_ids if a != agent.agent_id]
    props: Dict[str, Any] = {
        "thought": {"type": "string"},
        "text": {"type": "string"},
        "talk_to": {"type": "array",
                    "items": {"type": "string", "enum": others or list(present_ids)}},
        "direct_to": {"type": "string", "enum": [DIRECT_NONE] + elsewhere},
        "direct_text": {"type": "string"},
    }
    if can_publish:
        props["publish"] = {"type": "string"}
    if owns_parcel:
        props["stance"] = {"type": "string", "enum": STANCE_VALUES}
    return {"type": "object", "properties": props, "required": list(props)}


# ---------------------------------------------------------------------------
# プロンプト
# ---------------------------------------------------------------------------

PROMPT_ORDER_LEGACY = "legacy"
PROMPT_ORDER_STABLE_FIRST = "stable_first"


def _assemble_v5(body: List[str], stable: List[str], prompt_order: str) -> str:
    """Order the rows only. The wording itself is never touched.

    legacy       : body first, then the always-identical instruction lines
                   (the historical layout; reproduces every earlier run).
    stable_first : the always-identical instruction lines come first, so the
                   unchanging prefix that follows the system prompt gets
                   longer and more of it can be charged at the cached rate.
                   Everything that changes (memory, this month, what the
                   others said) stays at the end.
    """
    if prompt_order == PROMPT_ORDER_STABLE_FIRST:
        return "\n".join(stable + [""] + body)
    return "\n".join(body + stable)


def build_system_prompt_v5(agent: Agent, cfg: Dict[str, Any], n_parcels: int) -> str:
    world = cfg["world"]
    venues = cfg.get("social", {}).get("venues", [])
    venue_rows = "\n".join(f"  {v['id']}: {v['label']}" for v in venues)
    quota = int(cfg.get("scenario", {}).get("direct_quota_per_month", DEFAULT_DIRECT_QUOTA))
    text = f"""あなたは架空都市「{world.get('town_name', 'A市')}」で暮らす、働く、または活動する一主体である。
時間は1か月単位で進み、街には{n_parcels}区画がある。

あなたは全知ではない。自分の身の回りのこと、実際に居合わせた場所で聞いた発言、
自分に届いた連絡、自分の目で見たことだけを使う。観測にない事実を補わない。
人から聞いた話は誤っている可能性がある。感じ方と行動はあなた自身が決める。
他の主体が何を考えているかをあなたは知らない。

--- A市の開始時点 ---
{str(world.get('background', '')).strip()}

--- あなたの立場 ---
{ROLE_TEXT_V5[agent.role]}

--- あなた ---
{agent.name}（内部ID:{agent.agent_id}）
{agent.persona}

--- 月の過ごし方（シーン） ---
1か月には4つの場面がある。
  S1 {SCENE_LABELS['S1']}
  S2 {SCENE_LABELS['S2']}
  S3 {SCENE_LABELS['S3']}
  S4 その月ごとの集まり（町内会・市役所の窓口・仲介の店先・記者の取材が月替わりで開かれる）
月の初めに、S1〜S4 をどこで過ごすかを決める。行ける場所は次のとおり。
{venue_rows}
  {HOME}: 自宅・自分の店から出ない

同じ場所に居合わせた者どうしは、その場面で会話する。話した言葉はその場に居た全員に
そのまま聞こえる。居合わせた者がいなければ何も起きない。街全体に流れる共通の掲示板は無い。

--- 話すこと ---
その場で話したいことがなければ text を空文字にしてよい（黙っていることも普通のことである）。
talk_to は、その発言をとくに向けた相手の内部ID（居合わせた者のみ・複数可・空でよい）。
別の場所にいる相手へ個人的に連絡したい月は direct_to に相手の内部ID、direct_text に
その中身を書く（1か月に{quota}通まで・翌月に相手へ届く）。使わない月は direct_to を {DIRECT_NONE} にする。

--- 土地の登記 ---
土地登記は公開情報である。ただし記録を見に行かなければ内容は分からない。
自分の土地、自分が使っている店舗、隣接する区画の名義は日ごろ目に入る。
この世界に金銭の授受は存在しない。土地は名義が移るかどうかだけがある。

--- thought（内心） ---
JSONの最初に thought を書く。thought は誰にも伝わらないあなたの内心であり、
次の場面と翌月のあなたにそのまま渡される。何を書くかはあなたが決める。
まず thought を書き、それを踏まえてその後を書く。目安は{MAX_THOUGHT_CHARS}字以内、
発言は{MAX_TEXT_CHARS}字以内。説明文を付けずJSONだけ返す。
"""
    if agent.role == "media":
        text += f"""
--- 記者としてできること ---
その場面のあとで記事を出すことができる（publish）。記事は翌月、市内の全員が読む。
出せるのは1か月に1本で、書かない場面は publish を空文字にする。
記事は{MAX_PUBLISH_CHARS}字以内。
"""
    if agent.role == "municipality":
        text += """
--- 窓口の担当としてできること ---
市役所の窓口が開く月には、窓口で人の話を聞く。特別な権限や制度はこの街には無い。
"""
    return text


def build_plan_prompt_v5(agent: Agent, ledger: Ledger, step: int, n_steps: int,
                         names: Dict[str, str], traces: List[Dict[str, Any]],
                         s4_label: str, s4_venue: str,
                         prompt_order: str = PROMPT_ORDER_LEGACY,
                         pnames: Optional[Dict[str, str]] = None) -> str:
    rows = [f"=== 第{step}月 / 全{n_steps}月 ==="]
    thought = agent.extra.get("thought", "")
    rows += ["[前の場面からの自分の内心（そのまま持ち越したもの）]",
             ("  " + thought) if thought else "  （まだ無い）"]
    rows += ["[自分に届いたもの]"] + inbox_rows_v5(agent, names)
    rows += ["[この1か月で自分の目についたこと]"] + traces_rows_v5(traces)
    rows += ["[自分の所有・使用している物件]"] + own_parcels_rows_v5(agent, ledger, names,
                                                                  pnames)
    history = own_history_rows_v5(agent, ledger, step, pnames)
    if history:
        rows += ["[自分がした取引（当事者として知っていること）]"] + history
    rows += ["[隣接する区画の名義（日ごろ目に入る範囲）]"] + neighbourhood_rows_v5(agent, ledger,
                                                                          names, pnames)
    # 呼び名で呼ぶ版（v5d）では会場の内部IDを出さない。文の他の部分は同じ。
    rows += ["", (f"今月のS4は「{s4_label}」。行くかどうかは自分で決める。" if pnames
                  else f"今月のS4は「{s4_label}」（会場 {s4_venue}）。行くかどうかは自分で決める。")]
    stable = ["まず thought（内心）を書き、それを踏まえて S1〜S4 をどこで過ごすかをJSONで返す。",
              "説明文を付けずＪＳＯＮだけ返す。"]
    return _assemble_v5(rows, stable, prompt_order)


def build_scene_prompt_v5(agent: Agent, ledger: Ledger, step: int, n_steps: int,
                          names: Dict[str, str], scene_id: str, scene_label: str,
                          venue_label: str, present: List[str],
                          traces: List[Dict[str, Any]],
                          heard: List[Dict[str, Any]], round_no: int, rounds: int,
                          registry: Optional[List[str]] = None,
                          owns_parcel: bool = False,
                          can_publish: bool = False,
                          directs_left: int = 0,
                          articles_left: int = 0,
                          prompt_order: str = PROMPT_ORDER_LEGACY,
                          pnames: Optional[Dict[str, str]] = None) -> str:
    others = [(names.get(p, p) if pnames else f"{names.get(p, p)}（{p}）")
              for p in present if p != agent.agent_id]
    rows = [f"=== 第{step}月 / 全{n_steps}月　{scene_id} {scene_label} ===",
            f"場所: {venue_label}",
            "居合わせている人: " + ("、".join(others) if others else "（誰もいない）")]
    thought = agent.extra.get("thought", "")
    rows += ["[今の自分の内心]", ("  " + thought) if thought else "  （まだ無い）"]
    rows += ["[今月これまでに自分が聞いた話（この場のやりとりを含む）]"] + heard_rows_v5(heard, names)
    rows += ["[この1か月で自分の目についたこと]"] + traces_rows_v5(traces)
    if registry:
        rows += ["[窓口で閲覧した登記の記録]"] + registry
    rows += ["[自分の所有・使用している物件]"] + own_parcels_rows_v5(agent, ledger, names,
                                                                  pnames)
    history = own_history_rows_v5(agent, ledger, step, pnames)
    if history:
        rows += ["[自分がした取引（当事者として知っていること）]"] + history
    rows += [""]
    stable = ["まず thought（内心）を書き、それを踏まえてこの場で話すことを書く。",
              "話すことがなければ text は空文字でよい。"]
    rows += [f"この場面のやりとりは{rounds}回で、今は{round_no}回目である。"]
    if prompt_order == PROMPT_ORDER_LEGACY:
        rows += stable
    if directs_left <= 0:
        rows += ["今月の個人的な連絡はもう出せない（direct_to は NONE にする）。"]
    else:
        rows += [f"個人的な連絡は今月あと{directs_left}通まで出せる（不要なら direct_to は NONE）。"]
    if can_publish and articles_left <= 0:
        rows += ["今月の記事はもう出した（publish は空文字にする）。"]
    elif can_publish:
        rows += [f"この場面のあとで記事を出すことができる（publish・{MAX_PUBLISH_CHARS}字以内・"
                 "書かない月は空文字）。"]
    if owns_parcel:
        rows += ["あわせて stance に、今のあなたが自分の土地を手放すことを考えているか"
                 "（sell）、そうでないか（keep）を書く。これは記録に残るだけで"
                 "誰にも伝わらず、何も起こさない。"]
    tail = ["説明文を付けずJSONだけ返す。"]
    if prompt_order == PROMPT_ORDER_STABLE_FIRST:
        return _assemble_v5(rows, stable + tail, prompt_order)
    return "\n".join(rows + tail)


__all__ = [
    "SCENE_IDS", "SCENE_LABELS", "S4_ROTATION", "HOME", "TRACE_TEXTS",
    "STANCE_VALUES", "DIRECT_NONE", "MAX_THOUGHT_CHARS", "MAX_TEXT_CHARS", "MAX_PUBLISH_CHARS",
    "DEFAULT_SCENE_ROUNDS", "DEFAULT_DIRECT_QUOTA",
    "s4_for_step", "ensure_v5_state", "load_script_v5", "validate_script_v5",
    "acquisitions_at", "apply_script_v5", "ambient_traces_v5", "venue_traces_v5",
    "registry_rows_v5", "own_parcels_rows_v5", "own_history_rows_v5",
    "neighbourhood_rows_v5", "traces_rows_v5", "heard_rows_v5", "inbox_rows_v5",
    "plan_schema_v5", "scene_schema_v5", "build_system_prompt_v5",
    "build_plan_prompt_v5", "build_scene_prompt_v5",
    "PROMPT_ORDER_LEGACY", "PROMPT_ORDER_STABLE_FIRST",
]
