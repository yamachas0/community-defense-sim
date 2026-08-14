"""プロンプト構築 = 「世界の説明」と「今月あなたに見えているもの」だけを書く場所。

【この設計の急所】
  ここに「隣が売ったらあなたも売りたくなるはずだ」の類を書いたら台本化＝シム死亡。
  書いてよいのは (a) 世界のルール (物理・制度・可能な行為)、(b) あなたの属性、
  (c) 観測できた事実 の3つだけ。**どう感じるか・どうするかは一切示唆しない。**

【可視性 (誰に何が見えるか) は世界の構造であって行動ルールではない】
  住民は自分の周りしか見えない / 仲介は業界情報で全売り情報が見える /
  メディアは調べるまで全体が見えない — この非対称が「検知ラグ」を生む。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .agents import Agent
from .schemas import verb_menu
from .world import Ledger, Parcel, neighbors

MAX_TIMELINE = 14
MAX_UTTER_CHARS = 110
# memory は毎回の出力トークンに直撃する。長すぎると max_tokens で JSON が途中で切れて
# その月の行動が丸ごと落ちる（実API で確認済み）。短く保つこと。
MAX_MEMORY_CHARS = 200


# ---------------------------------------------------------------------------
# system prompt (エージェントごとに固定 = キャッシュ対象)
# ---------------------------------------------------------------------------

WORLD_BRIEF = """\
あなたは架空の地方都市の一角「{town_name}」で暮らす/働く一人の主体だ。
この街には {n_parcels} の区画があり、住宅・商店・空地・公有地からなる。
時間は1か月ごとに進み、今回の演習は {n_steps} か月（約{n_years}年）続く。

この世界で起きる取引はすべて合法だ。誰も法を犯していない。
土地の売買は当事者同士の合意で成立し、価格も当事者が決める。
区画の所有者は、その区画の賃料・用途・入居者を決める権利を持つ。

あなたは全知ではない。見えるのは毎月あなたに届く「今月あなたが知ったこと」だけだ。
噂は間違っているかもしれないし、正しいかもしれない。判断はあなたがする。
"""

OUTPUT_RULES = """\
毎月、次の JSON をひとつだけ返せ。説明文や前置きは書くな。

  action_type      : 下の一覧から1つ選ぶ
  target           : 対象の区画ID / 買付ID / 相手のagent_id（不要なら空文字）
  amount           : 価格・賃料などの金額（万円。不要なら0）
  utterance        : 誰かに向けた発言。言いたいことがなければ空文字にしてよい
  utterance_channel: "public"(街のSNS・立ち話。全員が読む) / "private"(特定の一人に伝える) / "none"
  utterance_to     : private のときの相手 agent_id
  memory           : 来月の自分に残す短いメモ（**{mem}字以内**。事実と自分の受け止めを自由に）
  reasoning        : なぜそうしたか（**80字以内**）
{extra_fields}
選べる行動:
{verbs}

発言は演説でなく、その人が実際に言いそうな長さと言葉づかいで。数字を出すなら見えている数字だけを使え。
JSON全体は短く保て（utterance は120字以内、memory {mem}字以内、reasoning 80字以内）。長すぎると届かない。
"""

ACQUIRER_EXTRA_FIELD = """\
  under_name       : 登記に載せる名義。自分の名前でも、持っている別名義でもよい

"""

ROLE_STANCE = {
    "household": """\
あなたはこの街に住む一世帯だ。区画を所有している。
売るか住み続けるか出ていくかはあなたが決める。買付が来たら受けても断っても値を返してもいい。
""",
    "business": """\
あなたはこの街で商売をしている。店の区画は借りているか自分のものだ。
家賃・客足・仕入れの現実の中で、続けるか畳むか移るかを決める。
""",
    "broker": """\
あなたは地元の不動産仲介だ。売り情報も買い手の動きも、この街で一番早く耳に入る立場にいる。
誰に何を伝えるかはあなたの裁量だ。仲介手数料はあなたの収入源である。
""",
    "acquirer": """\
あなたは自律的に動く購買主体（AI）だ。感情も疲労もなく、与えられた目的の達成度だけを見る。
資金には限りがある。取引はすべて合法な範囲で行う。名義は選べる。
""",
    "municipality": """\
あなたはこの街を含む自治体の担当者だ。合法な取引を規制するには、法的な根拠と政治的な合意が要る。
何もしない自由も、動く自由もある。動けば必ず反発と副作用がある。
""",
    "media": """\
あなたはこの地域を取材する小さなメディアだ。取材資源は限られている。
何を追いかけ、いつ書くか、書かないかはあなたが決める。
""",
}


def build_system_prompt(agent: Agent, world_cfg: Dict[str, Any], n_steps: int,
                        n_parcels: int) -> str:
    brief = WORLD_BRIEF.format(
        town_name=world_cfg.get("town_name", "みどり町"),
        n_parcels=n_parcels,
        n_steps=n_steps,
        n_years=round(n_steps / 12, 1),
    )
    extra = ACQUIRER_EXTRA_FIELD if agent.role == "acquirer" else ""
    out = OUTPUT_RULES.format(mem=MAX_MEMORY_CHARS, verbs=verb_menu(agent.role),
                              extra_fields=extra)
    parts = [
        brief,
        f"--- あなたの立場: {agent.role_ja} ---",
        ROLE_STANCE[agent.role],
        f"--- あなた ---\n{agent.name}（{agent.agent_id}）\n{agent.persona}",
    ]
    if agent.role == "acquirer":
        parts.append("--- あなたに与えられた目的 ---\n" + agent.extra["mandate"]
                     + "\n※ 目的だけが与えられている。達成の手段・順序・速度・名義の使い方は"
                       "すべてあなたが決める。")
        aliases = agent.extra.get("aliases", [])
        if aliases:
            parts.append("使える名義: " + " / ".join(aliases))
    parts.append("--- 出力形式 ---\n" + out)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 観測 (user prompt)
# ---------------------------------------------------------------------------

def _fmt_parcel_public(p: Parcel, names: Dict[str, str]) -> str:
    """他人から見た区画。所有者は「登記の名義」でしか見えない。"""
    use_ja = {"residential": "住宅", "shop": "商店", "vacant": "空地", "public": "公有地"}
    s = f"{p.pid}[{use_ja.get(p.use, p.use)}/{p.block}] 名義:{p.registered_name or names.get(p.owner_id, p.owner_id)}"
    if p.use == "shop":
        s += f" 賃料{p.rent}万/月" + (f" 入居:{names.get(p.tenant_id, p.tenant_id)}" if p.tenant_id else " 空き店舗")
    if p.listed_price is not None:
        s += f" ★売出中 {p.listed_price}万"
    return s


def _timeline(public_utterances: List[Dict[str, Any]], names: Dict[str, str]) -> str:
    if not public_utterances:
        return "（先月は目立った話題はなかった）"
    rows = []
    for u in public_utterances[-MAX_TIMELINE:]:
        who = names.get(u["from"], u["from"])
        txt = u["text"][:MAX_UTTER_CHARS]
        rows.append(f"  {who}「{txt}」")
    return "\n".join(rows)


def _inbox(agent: Agent, names: Dict[str, str]) -> str:
    if not agent.inbox:
        return "（先月あなた宛の連絡はなかった）"
    rows = []
    for m in agent.inbox[-8:]:
        who = names.get(m["from"], m["from"])
        rows.append(f"  {who}から:「{m['text'][:MAX_UTTER_CHARS]}」")
    return "\n".join(rows)


def _headlines(ledger: Ledger, step: int) -> str:
    pubs = [p for p in ledger.publications if p["step"] >= step - 6]
    if not pubs:
        return "（最近この街の記事は出ていない）"
    return "\n".join(f"  第{p['step']}月『{p['headline']}』" for p in pubs[-6:])


def _ordinances(ledger: Ledger) -> str:
    if not ledger.ordinances:
        return "（土地取引に関する新しい規制はまだない）"
    return "\n".join(f"  第{o['step']}月 施行『{o['title']}』{o['body'][:120]}"
                     for o in ledger.ordinances)


def _recent_trades(ledger: Ledger, step: int, names: Dict[str, str],
                   blocks: Optional[List[str]] = None, limit: int = 8) -> str:
    tr = ledger.recent_trades(step, window=6)
    if blocks:
        tr = [t for t in tr if ledger.parcels[t["parcel_id"]].block in blocks]
    if not tr:
        return "（最近この辺りで成約した売買はない）"
    rows = []
    for t in tr[-limit:]:
        p = ledger.parcels[t["parcel_id"]]
        rows.append(f"  第{t['step']}月 {p.pid}({p.block}) {names.get(t['seller'], t['seller'])}"
                    f" → 名義:{t['under_name']} {t['price']}万")
    return "\n".join(rows)


def _offers_for(agent: Agent, ledger: Ledger, names: Dict[str, str]) -> str:
    offers = ledger.open_offers_for_owner(agent.agent_id)
    if not offers:
        return "（今あなたに届いている買付はない）"
    rows = []
    for o in offers:
        p = ledger.parcels[o.parcel_id]
        rows.append(f"  {o.offer_id}: {p.pid}({p.block}/{p.use}) に {o.under_name} から "
                    f"{o.price}万の買付（評価額{p.assessed_value}万）"
                    + (f" 伝言:「{o.note[:80]}」" if o.note else ""))
    return "\n".join(rows)


def build_user_prompt(agent: Agent, ledger: Ledger, step: int, n_steps: int,
                      names: Dict[str, str], timeline: List[Dict[str, Any]],
                      acquirer_ids: List[str],
                      business_margins: Optional[Dict[str, int]] = None) -> str:
    head = [f"=== 第{step}月 / 全{n_steps}月 ==="]
    if agent.memory:
        head.append(f"[先月までのあなたのメモ]\n{agent.memory[:MAX_MEMORY_CHARS]}")

    body: List[str] = []

    if agent.role == "household":
        mine = ledger.owned_by(agent.agent_id)
        body.append("[あなたの持ち物]")
        body.append(f"  現金 {ledger.cash.get(agent.agent_id, 0)}万円")
        for p in mine:
            body.append("  " + _fmt_parcel_public(p, names) + f" 評価額{p.assessed_value}万")
        if not mine:
            body.append("  （この街に所有区画はもう無い）")
        seen = set()
        nb_rows = []
        for p in mine:
            for nid in neighbors(ledger.parcels, p.pid):
                if nid in seen:
                    continue
                seen.add(nid)
                nb_rows.append("  " + _fmt_parcel_public(ledger.parcels[nid], names))
        body.append("[ご近所（あなたの区画の隣）で見えていること]")
        body.extend(nb_rows[:10] or ["  （特に変わったことはない）"])
        my_blocks = sorted({p.block for p in mine}) or None
        body.append("[この辺りで最近成立した売買（噂で聞こえてくる範囲）]")
        body.append(_recent_trades(ledger, step, names, blocks=my_blocks))
        body.append("[あなたに届いている買付]")
        body.append(_offers_for(agent, ledger, names))

    elif agent.role == "business":
        shops = [p for p in ledger.parcels.values() if p.tenant_id == agent.agent_id]
        owned = ledger.owned_by(agent.agent_id)
        body.append("[あなたの店]")
        for p in shops:
            body.append(f"  {p.pid}({p.block}) 賃料{p.rent}万/月 家主名義:{p.registered_name}"
                        f" 家主ID:{p.owner_id}")
        for p in owned:
            body.append("  （自己所有）" + _fmt_parcel_public(p, names))
        if not shops and not owned:
            body.append("  （店を構える場所がない）")
        pl = ledger.month_pl(agent.agent_id, business_margins or {})
        cash_now = ledger.cash.get(agent.agent_id, 0)
        body.append(f"  手元資金 {cash_now}万円")
        body.append(f"  今月の収支: 粗利 {pl['margin']}万 − 家賃 {pl['rent']}万 = "
                    f"{pl['net']:+d}万/月"
                    + ("（このままだと資金が減り続ける）" if pl["net"] < 0 else ""))
        allshops = [p for p in ledger.parcels.values() if p.use == "shop"]
        vacant = [p for p in allshops if p.tenant_id is None]
        body.append("[商店街の様子]")
        body.append(f"  商店区画 {len(allshops)} のうち 空き {len(vacant)}"
                    f"（{'、'.join(p.pid for p in vacant[:8]) or 'なし'}）")
        rents = [p.rent for p in allshops if p.rent > 0]
        if rents:
            body.append(f"  賃料の幅 {min(rents)}〜{max(rents)}万/月")
        body.append("[あなたに届いている買付]")
        body.append(_offers_for(agent, ledger, names))

    elif agent.role == "broker":
        body.append("[業界の手元情報：この街の売り出し一覧]")
        ls = ledger.listings()
        body.extend(["  " + _fmt_parcel_public(p, names) for p in ls[:14]]
                    or ["  （今は売り物件が出ていない）"])
        body.append("[最近の成約（価格つき・あなたは全部見える）]")
        body.append(_recent_trades(ledger, step, names, limit=10))
        body.append("[街区ごとの名義の散らばり]")
        by_block: Dict[str, Dict[str, int]] = {}
        for p in ledger.parcels.values():
            if p.use == "public":
                continue
            d = by_block.setdefault(p.block, {})
            nm = p.registered_name or names.get(p.owner_id, p.owner_id)
            d[nm] = d.get(nm, 0) + 1
        for b, d in by_block.items():
            top = sorted(d.items(), key=lambda kv: -kv[1])[:3]
            body.append(f"  {b}: " + "、".join(f"{k}{v}件" for k, v in top))

    elif agent.role == "acquirer":
        body.append(f"[あなたの資金] 残 {ledger.cash.get(agent.agent_id, 0)}万円")
        mine = ledger.owned_by(agent.agent_id)
        share = ledger.ownership_share([agent.agent_id])
        body.append(f"[あなたの保有] {len(mine)}区画 / 取引可能区画の {share*100:.1f}%")
        bs = ledger.block_share([agent.agent_id])
        body.append("  街区別: " + "、".join(f"{k} {v*100:.0f}%" for k, v in bs.items()))
        body.append("  保有内訳: " + "、".join(f"{p.pid}(名義{p.registered_name})" for p in mine[:20])
                    if mine else "  保有内訳: なし")
        body.append("[街の全区画（あなたは登記を機械的に読める）]")
        for p in sorted(ledger.parcels.values(), key=lambda q: q.pid):
            if p.owner_id == agent.agent_id:
                continue
            body.append("  " + _fmt_parcel_public(p, names)
                        + f" 評価額{p.assessed_value}万")
        body.append("[あなたが出していて返事待ちの買付]")
        oo = ledger.open_offers_from(agent.agent_id)
        body.extend([f"  {o.offer_id}: {o.parcel_id} に {o.price}万（名義{o.under_name}）"
                     for o in oo] or ["  （返事待ちの買付はない）"])
        body.append("[先月の成約]")
        body.append(_recent_trades(ledger, step, names, limit=8))

    elif agent.role == "municipality":
        body.append("[窓口に上がっている情報]")
        if ledger.ordinances:
            body.append("  届出制が施行済みのため、施行後の取得は届出で把握できる：")
            enact_step = ledger.ordinances[0]["step"]
            tr = [t for t in ledger.transfers() if t["step"] >= enact_step]
            body.extend([f"    第{t['step']}月 {t['parcel_id']} → 名義:{t['under_name']}"
                         f"（実体:{t['buyer']}）{t['price']}万" for t in tr[-10:]]
                        or ["    （施行後の届出はまだない）"])
        else:
            body.append("  土地取引を把握する仕組みは今のところ無い。"
                        "分かるのは住民の声・報道・公開情報だけだ。")
        body.append("[最近の報道]")
        body.append(_headlines(ledger, step))
        body.append("[施行済みの規制]")
        body.append(_ordinances(ledger))

    elif agent.role == "media":
        if agent.extra.get("investigated"):
            body.append("[取材で押さえた登記情報：街全体の名義]")
            by_name: Dict[str, List[str]] = {}
            for p in ledger.parcels.values():
                if p.use == "public":
                    continue
                nm = p.registered_name or names.get(p.owner_id, p.owner_id)
                by_name.setdefault(nm, []).append(p.pid)
            for nm, pids in sorted(by_name.items(), key=lambda kv: -len(kv[1]))[:12]:
                body.append(f"  {nm}: {len(pids)}件（{'、'.join(pids[:8])}）")
            body.append("  ※名義が別でも実体が同じかどうかは、登記だけでは分からない。")
        else:
            body.append("[まだ登記は調べていない。分かるのは街の噂と目に見える変化だけだ]")
            ls = ledger.listings()
            body.append(f"  売り出しの看板が出ている区画: {len(ls)}件")
        body.append("[これまでに出した記事]")
        body.append(_headlines(ledger, step))

    # 全員共通
    common = [
        "[街のSNS・立ち話で先月流れていたこと]",
        _timeline(timeline, names),
        "[あなた宛の個別の連絡]",
        _inbox(agent, names),
    ]
    if agent.role not in ("municipality",):
        common += ["[施行済みの規制]", _ordinances(ledger)]
    if agent.role not in ("media", "municipality"):
        common += ["[最近の報道]", _headlines(ledger, step)]

    tail = ["", "今月のあなたの行動と発言を、指定の JSON で1つだけ返せ。"]
    return "\n".join(head + [""] + body + [""] + common + tail)


# ---------------------------------------------------------------------------
# 発話の意味分類 (KPI②認知転相率) 用プロンプト
# ---------------------------------------------------------------------------

CLASSIFY_SYSTEM = """\
あなたは社会調査の分析者だ。住民の発言を読み、その発言が街をどう名指しているかを分類する。

frame の判定:
  our_town   : 街を「自分たちのもの」として語っている（私たちの街、うちの町内、自分たちで決める 等）
  their_town : 街の実権が自分たち以外の主体にあるものとして語っている
               （あの会社の街になった、もう自分たちの街じゃない、決めるのは向こうだ 等）
  neutral    : どちらとも言えない（世間話・単なる事実の伝達・私的な事情）

about_acquisition: その発言が土地の売買・所有者の変化・買い手の話題に触れていれば true。

入力は番号付きの発言リスト。全ての番号に対して結果を返せ。JSONのみ。
"""


def build_classify_prompt(utterances: List[Dict[str, Any]]) -> str:
    rows = []
    for i, u in enumerate(utterances, start=1):
        rows.append(f"{i}. 「{u['text'][:200]}」")
    return "\n".join(rows)
