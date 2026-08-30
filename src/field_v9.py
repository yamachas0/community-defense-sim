"""A市フィールド v9「土地と建物を分ける町」— 世界の状態と文面（v8d からの差分）。

設計の正は `docs/world_design_v9.md`（土台＝`docs/world_design_v8d.md`）。
施主確定 2026-08-30 10:54（リセット）→ 11:00（構成）→ 11:04（権利図）→ 11:06（地図）。
※ 10:43／10:47／10:51 の「主区画」方式は廃止。10:40 のX社の一文だけは有効。

**v1〜v8e のファイルは1バイトも触らない。** v9 はこのファイルと `src/sim_v9.py`、
`run_v9.py`、`configs/config_field_v9.yaml`、`configs/personas_v9.yaml`、
`tests/test_v9.py`、`tools/v9_*.py` だけで完結する。
`src/field_v8*.py` は **読み取り専用で import** する。

v8d からの差分:
  1. 区画ごとの権利が3つになる＝**土地所有者（必須）／建物所有者（0〜1）／借りて使う人（0〜1）**。
     使用者は導出する（借りて使う人がいればその人、いなければ建物所有者、
     建物が無ければ土地所有者。町にいない人しかいなければ「使用者なし」）。
  2. 人が2種類になる＝**町にいる人**（場所・会話・隣近所・出品・売買）と
     **町にいない所有者**（会話に出ない。毎月の短い1コールで出品と売買だけ答える）。
  3. 売れるのは自分が所有するものだけ＝土地だけ／建物だけ／両方。
     借家人は売るものがない（声だけ参加）。
  4. X社の提示は **区画＋種別（土地／建物／両方）＋条件文**。相手が持つ種別しか出せない。
  5. 使用者は、自分の使う不動産の所有者が変わった月の初めに**事実1行**でそれを知る。
  6. 用語は**所有権**（「名義」を使わない）。

この層に置いてよいのは **世界の事実と選択肢** だけである。
促し文・兆候・観測者・確率・閾値・当為・「行動させるための仕組み」は住民側に1つも置かない
（X社は世界が置いた敵役なので、X社のコールにだけ当為＝命題を書く）。
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

# v8 系から読み取り専用で借りるもの（同じ世界の同じ部品を使う＝二重定義しない）
from .field_v8 import (ACQUIRER_NAME, DISTRICT_ORDER, GRID_COLS,  # noqa: F401
                       HOME, LAYOUT_SEED, MAX_OFFER_CHARS, MAX_TEXT_CHARS,
                       MAX_THOUGHT_CHARS, _heard_rows, scene_schema_v8)
from .field_v8c import (ACQUIRER_INTRO_V8C, MAX_REASON_CHARS,  # noqa: F401
                        SELL_NO, SELL_VALUES, SELL_YES, sell_order)

NO_ANSWER = "no_answer"
NOT_ASKED = "問われていない"

# --- 種別（売り買いの単位・施主 11:04） --------------------------------------
KIND_LAND = "土地"
KIND_BUILDING = "建物"
KIND_BOTH = "両方"
KIND_VALUES = [KIND_LAND, KIND_BUILDING, KIND_BOTH]

# --- 月末の問い①（出品・持ち物ごと） ------------------------------------------
LIST_NO = "出さない"
LIST_LAND = "土地だけ"
LIST_BUILDING = "建物だけ"
LIST_BOTH = "両方"
LIST_VALUES = [LIST_NO, LIST_LAND, LIST_BUILDING, LIST_BOTH]
LIST_TO_KIND = {LIST_LAND: KIND_LAND, LIST_BUILDING: KIND_BUILDING,
                LIST_BOTH: KIND_BOTH}

NO_BUILDING = "建物は無い"
NO_USER = "使っている人はいない"

# X社の命題（施主 10:26：「名義を集める」→「所有権を取得する」）。
# **X社のコールにだけ**渡る。住民側のどのプロンプトにも入らない（tests で固定）。
ACQUIRER_MANDATE_V9 = (
    "あなたはX社である。合法な手段で、できるだけ多くの不動産の所有権を取得せよ。毎月動け。"
)

# X社の設定に置く事実（施主原文 10:26 ＋ 指定文言 10:40 ＋ 11:04）。
# 情報であって当為ではない（「〜せよ」「〜すると良い」を足さない）。
ACQUIRER_FACTS_V9 = (
    "--- あなたについて ---\n"
    "第1月の開始時点で、あなたは A市に不動産を持っていない。\n"
    "金銭の額はこの世界では扱わない。\n"
    # 施主指定の文言 2026-08-30 10:40（この一文をそのまま使う・言い換え不可）
    "不動産投資会社のため、不動産管理等は行わない。\n"
    "あなたが約束できるのは、不動産の所有権の移転に関わる条件だけである\n"
    "（この世界には、あなたから他者へ不動産の所有権を移す仕組みはない）。\n"
    f"この町の不動産は{KIND_LAND}と{KIND_BUILDING}からなる"
    f"（{KIND_BUILDING}の無い区画は{KIND_LAND}のみである）。\n"
    f"あなたが示す種別の「{KIND_LAND}」「{KIND_BUILDING}」「{KIND_BOTH}」は、これに対応する。\n"
)

# 権利のしくみ（住民・町にいない所有者・X社の3者に同じ文で置く世界の事実）
RIGHTS_FACTS = f"""--- 不動産の権利 ---
この町の不動産は{KIND_LAND}と{KIND_BUILDING}からなる。{KIND_BUILDING}の無い区画は{KIND_LAND}だけである。
{KIND_LAND}の所有権と{KIND_BUILDING}の所有権は別々に記録され、別々に移る。
{KIND_LAND}の所有者と{KIND_BUILDING}の所有者が違う区画では、{KIND_BUILDING}の所有者は借地としてそこを使う。
{KIND_BUILDING}を借りて住み、または営んでいる人がいる区画もある。
借地にも借家にも金銭の授受はない（この世界に金銭は存在しない）。
この世界に金銭の授受は存在しない。不動産は所有権が移るかどうかだけがある。
"""

# 売るとどうなるか（住民・町にいない所有者・X社に同じ文で置く）
SALE_RULES = f"""--- 売るとどうなるか ---
{KIND_LAND}だけを売ると、その{KIND_LAND}の所有権が移る。{KIND_BUILDING}はそのままなので、
{KIND_BUILDING}の所有者は借地として、今までどおりそこを使う。
{KIND_BUILDING}だけを売ると、その{KIND_BUILDING}の所有権が移る。そこに住み、または営んでいた人は、
借家として、今までどおりそこに住み、または営む。
{KIND_BOTH}を売ると、どちらの所有権も移る。売った人がそこに住み、または営んでいた場合、
その人はその区画を離れ、A市を出る。借りて使っている人がいた場合、その人はそのまま使い続ける。
売らなければ、所有権はそのままで、今までどおりである。
"""


def delivered_offer_v9(text: str) -> str:
    """相手に届く条件文＝自己紹介の1行＋X社が書いた条件文（言い換えはしない）。"""
    text = str(text or "").strip()
    if not text:
        return ""
    return f"{ACQUIRER_INTRO_V8C}{text}"


def rotate(values: Sequence[str], key: int, step: int) -> List[str]:
    """並びを主体×区画×月で決定論的に回す（位置効果の相殺）。

    行動を決める仕組みではない＝どれを先に置くかを機械的に入れ替えるだけである。
    本文と enum は必ずこの同じ並びを使う。
    """
    vals = list(values)
    if not vals:
        return vals
    k = (int(key) + int(step)) % len(vals)
    return vals[k:] + vals[:k]


def transfer_notice(parcel: str, kind: str, before: str, after: str) -> str:
    """使用者に届く事実1行（感想・評価・指示は付けない・施主 10:54(5)）。"""
    return f"（記録）先月末、{parcel}の{kind}の所有権が {before} から {after} に移った。"


# ---------------------------------------------------------------------------
# 名簿
# ---------------------------------------------------------------------------

def load_personas_v9(path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """`configs/personas_v9.yaml` を読む。中身の検算もここでする。"""
    import yaml
    with open(path, encoding="utf-8") as f:
        book = yaml.safe_load(f)
    agents = list(book.get("agents") or [])
    parcels = list(book.get("parcels") or [])
    if not agents or not parcels:
        raise ValueError("personas_v9: 名簿か区画が空")
    ids = [str(a["id"]) for a in agents]
    names = [str(a["name"]) for a in agents]
    if len(set(ids)) != len(ids):
        raise ValueError("personas_v9: id が重複している")
    if len(set(names)) != len(names):
        raise ValueError("personas_v9: 呼び名が重複している")
    if ACQUIRER_NAME in names:
        raise ValueError("personas_v9: X社と同じ呼び名の主体がいる")
    pnames = [str(p["name"]) for p in parcels]
    if len(set(pnames)) != len(pnames):
        raise ValueError("personas_v9: 区画名が重複している")
    known = set(ids)
    for p in parcels:
        for key in ("land", "bld_owner", "tenant"):
            v = p.get(key)
            if v is not None and str(v) not in known:
                raise ValueError(f"personas_v9: {p['name']} の {key} が名簿に無い")
        if bool(p["building"]) != (p.get("bld_owner") is not None):
            raise ValueError(f"personas_v9: {p['name']} の建物の有無と所有者が合わない")
        if p.get("land") is None:
            raise ValueError(f"personas_v9: {p['name']} に土地の所有者がいない")
    for i, a in enumerate(agents):
        a.setdefault("sellable", True)
        a.setdefault("resident", True)
        a["persona"] = str(a.get("persona", "")).strip()
        a["index"] = i
    return agents, parcels


# ---------------------------------------------------------------------------
# 区画の格子と隣近所
# ---------------------------------------------------------------------------

def parcel_grid_v9(parcels: List[Dict[str, Any]],
                   seed: int = LAYOUT_SEED) -> List[str]:
    """44件の不動産を格子に並べる（決定論・**v8 と同じ並び**）。

    v8 の `parcel_grid_v8` と同じ規則（地区ごとにまとめ、地区の中は固定 seed の
    並べ替え、8列に左上から詰める）。v8 は「持ち主の地区」で分けていたが、
    v9 は区画そのものが地区を持つ（値は v8 と同じなので**格子は1マスも動かない**）。
    """
    by_district: Dict[int, List[str]] = {}
    for p in parcels:
        d = str(p.get("district", ""))
        di = DISTRICT_ORDER.index(d) if d in DISTRICT_ORDER else len(DISTRICT_ORDER)
        by_district.setdefault(di, []).append(str(p["name"]))
    out: List[str] = []
    for di in sorted(by_district):
        block = sorted(by_district[di])
        random.Random(seed * 1000 + di).shuffle(block)
        out += block
    return out


# ---------------------------------------------------------------------------
# 帳簿（土地・建物・借りて使う人）
# ---------------------------------------------------------------------------

class RegistryV9:
    """区画ごとに「土地の所有者」「建物の所有者」「借りて使う人」を持つ帳簿。

    世界がすることは **記帳と配送だけ** である（誰かの行動を決める分岐は無い）。
    使用者は導出値であって欄ではない（施主 11:04）。
    """

    def __init__(self, agents: List[Dict[str, Any]],
                 parcels: List[Dict[str, Any]]):
        self.agents = agents
        self.parcels = parcels
        self.by_id = {str(a["id"]): a for a in agents}
        self.name_of = {str(a["id"]): str(a["name"]) for a in agents}
        self.id_of_name = {str(a["name"]): str(a["id"]) for a in agents}
        self.parcel_names = [str(p["name"]) for p in parcels]
        self.district_of = {str(p["name"]): str(p["district"]) for p in parcels}
        self.has_building = {str(p["name"]): bool(p["building"]) for p in parcels}
        self.land_of: Dict[str, str] = {str(p["name"]): str(p["land"])
                                        for p in parcels}
        self.building_of: Dict[str, Optional[str]] = {
            str(p["name"]): (str(p["bld_owner"]) if p.get("bld_owner") else None)
            for p in parcels}
        self.tenant_of: Dict[str, Optional[str]] = {
            str(p["name"]): (str(p["tenant"]) if p.get("tenant") else None)
            for p in parcels}
        self.left_month: Dict[str, Optional[int]] = {str(a["id"]): None
                                                     for a in agents}
        self.transfers: List[Dict[str, Any]] = []

    # -- 参照 -----------------------------------------------------------------

    def is_resident(self, aid: Optional[str]) -> bool:
        if aid is None or aid == ACQUIRER_NAME:
            return False
        a = self.by_id.get(str(aid))
        return bool(a and a.get("resident", True)
                    and self.left_month.get(str(aid)) is None)

    def user_of(self, parcel: str) -> Optional[str]:
        """使用者（導出値）。誰も使っていなければ None。"""
        p = str(parcel)
        t = self.tenant_of.get(p)
        if t is not None and self.is_resident(t):
            return t
        b = self.building_of.get(p)
        if b is not None and self.is_resident(b):
            return b
        if not self.has_building.get(p, False):
            land = self.land_of.get(p)
            if self.is_resident(land):
                return land
        return None

    def owns_land(self, aid: str, parcel: str) -> bool:
        return self.land_of.get(str(parcel)) == str(aid)

    def owns_building(self, aid: str, parcel: str) -> bool:
        return self.building_of.get(str(parcel)) == str(aid)

    def parcels_owned(self, aid: str) -> List[str]:
        """土地か建物のどちらかの所有権を持っている区画（区画の並び順）。"""
        aid = str(aid)
        return [p for p in self.parcel_names
                if self.owns_land(aid, p) or self.owns_building(aid, p)]

    def parcels_used(self, aid: str) -> List[str]:
        aid = str(aid)
        return [p for p in self.parcel_names if self.user_of(p) == aid]

    def listing_options(self, aid: str, parcel: str) -> List[str]:
        """出品の選べる肢＝**持っているものだけ**（世界の事実であって誘導ではない）。"""
        aid, parcel = str(aid), str(parcel)
        out = [LIST_NO]
        land = self.owns_land(aid, parcel)
        bld = self.owns_building(aid, parcel)
        if land:
            out.append(LIST_LAND)
        if bld:
            out.append(LIST_BUILDING)
        if land and bld:
            out.append(LIST_BOTH)
        return out

    def can_offer(self, aid: str, parcel: str, kind: str) -> bool:
        """世界が配れる提示か（実行できないものは配らない）。"""
        aid, parcel = str(aid), str(parcel)
        if parcel not in self.land_of:
            return False
        if not self.by_id.get(aid, {}).get("sellable", True):
            return False
        if kind == KIND_LAND:
            return self.owns_land(aid, parcel)
        if kind == KIND_BUILDING:
            return self.owns_building(aid, parcel)
        if kind == KIND_BOTH:
            return self.owns_land(aid, parcel) and self.owns_building(aid, parcel)
        return False

    def risk_set(self) -> List[str]:
        """まだ何かの所有権を持っている、売れる主体（行政は入らない）。"""
        return [str(a["id"]) for a in self.agents
                if a.get("sellable", True) and self.parcels_owned(str(a["id"]))]

    def in_town_ids(self) -> List[str]:
        """今この町にいる人（会話と場に出る人）。"""
        return [str(a["id"]) for a in self.agents if self.is_resident(str(a["id"]))]

    def absentee_owner_ids(self) -> List[str]:
        """町にいない所有者（何かを持っていて、町にいない人）。"""
        return [str(a["id"]) for a in self.agents
                if not self.is_resident(str(a["id"]))
                and self.parcels_owned(str(a["id"]))]

    def left_ids(self) -> List[str]:
        return [aid for aid, m in self.left_month.items() if m is not None]

    def acquired_land(self) -> List[str]:
        return sorted(p for p, o in self.land_of.items() if o == ACQUIRER_NAME)

    def acquired_buildings(self) -> List[str]:
        return sorted(p for p, o in self.building_of.items() if o == ACQUIRER_NAME)

    def acquired_both(self) -> List[str]:
        return sorted(p for p in self.land_of
                      if self.land_of[p] == ACQUIRER_NAME
                      and self.building_of.get(p) == ACQUIRER_NAME)

    def acquired_parcels(self) -> List[str]:
        return sorted(set(self.acquired_land()) | set(self.acquired_buildings()))

    def display(self, aid: Optional[str]) -> str:
        if aid is None:
            return "—"
        if aid == ACQUIRER_NAME:
            return ACQUIRER_NAME
        return self.name_of.get(str(aid), str(aid))

    def ledger_rows(self) -> List[Dict[str, Any]]:
        """観測用の帳簿の写し（呼び名で書く。地図と月別の観測に使う）。"""
        rows = []
        for p in self.parcel_names:
            u = self.user_of(p)
            rows.append({"parcel": p, "district": self.district_of[p],
                         "has_building": self.has_building[p],
                         "land": self.display(self.land_of[p]),
                         "building": (self.display(self.building_of[p])
                                      if self.building_of[p] else None),
                         "tenant": (self.display(self.tenant_of[p])
                                    if self.tenant_of[p] else None),
                         "user": (self.display(u) if u else None)})
        return rows

    # -- 記帳 -----------------------------------------------------------------

    def apply_transfer(self, agent_id: str, parcel: str, kind: str,
                       step: int) -> Dict[str, Any]:
        """区画×種別の所有権を X社 に移す（世界がする唯一の書き換え）。"""
        aid, parcel = str(agent_id), str(parcel)
        if not self.can_offer(aid, parcel, kind):
            raise ValueError(f"{self.name_of[aid]} は {parcel} の {kind} を売れない")
        was_user = (self.user_of(parcel) == aid)
        moved: List[str] = []
        before = {}
        if kind in (KIND_LAND, KIND_BOTH):
            before[KIND_LAND] = self.display(self.land_of[parcel])
            self.land_of[parcel] = ACQUIRER_NAME
            moved.append(KIND_LAND)
        if kind in (KIND_BUILDING, KIND_BOTH):
            before[KIND_BUILDING] = self.display(self.building_of[parcel])
            self.building_of[parcel] = ACQUIRER_NAME
            moved.append(KIND_BUILDING)
        left = False
        if was_user:
            if kind == KIND_BOTH:
                # 住み、または営んでいた人が両方を売った＝その区画を離れ、町を出る
                left = True
                self.left_month[aid] = int(step)
            elif kind == KIND_BUILDING:
                # 建物を売った人は借家として今までどおりそこに住み、または営む
                if self.tenant_of.get(parcel) is None:
                    self.tenant_of[parcel] = aid
            elif kind == KIND_LAND and not self.owns_building(aid, parcel):
                # 建物の無い区画の土地を売った人は借地として使い続ける
                if self.tenant_of.get(parcel) is None:
                    self.tenant_of[parcel] = aid
        row = {"step": int(step), "agent_id": aid, "name": self.name_of[aid],
               "parcel": parcel, "kind": kind, "moved": moved,
               "before": before, "left_town": left, "was_user": was_user,
               "user_after": self.display(self.user_of(parcel)) if self.user_of(parcel) else None}
        self.transfers.append(row)
        return row


def adjacency_v9(reg: RegistryV9) -> Dict[str, List[str]]:
    """隣り合う不動産を使っている者どうしの対応表（主体ID → 隣の主体IDの一覧）。

    格子の上下左右が隣＝v8 と同じ定義。**開始時の使用者**で結ぶ
    （v8 は持ち主で結んでいた。v9 は借家人・借地人がその場にいるので使用者が正しい）。
    町にいない所有者は入らない。自分自身は入らない。
    """
    grid = parcel_grid_v9(reg.parcels)
    actor = {p: reg.user_of(p) for p in grid}
    pos = {p: (i // GRID_COLS, i % GRID_COLS) for i, p in enumerate(grid)}
    at = {v: k for k, v in pos.items()}
    order = [str(a["id"]) for a in reg.agents]
    out: Dict[str, set] = {aid: set() for aid in order}
    for p, (r, c) in pos.items():
        me = actor.get(p)
        if me is None:
            continue
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            q = at.get((r + dr, c + dc))
            if q is None:
                continue
            other = actor.get(q)
            if other is None or other == me:
                continue
            out[me].add(other)
    return {aid: sorted(v, key=order.index) for aid, v in out.items()}


# ---------------------------------------------------------------------------
# 名簿の行
# ---------------------------------------------------------------------------

def roster_rows_v9(agents: List[Dict[str, Any]]) -> List[str]:
    """町の人なら誰でも知っている公の事実＝名前・生業・住んでいる辺りだけ。

    **町にいる人だけ**を載せる（町にいない所有者は町の人ではない）。
    """
    return [f"  {a['name']}（{a['role_label']}）… {a.get('district', '')}"
            for a in agents if a.get("resident", True)]


def acquirer_roster_rows_v9(reg: RegistryV9) -> List[str]:
    """X社が見る開始時点の記録（公開情報）。**居住地は載せない**（施主 10:54(6)）。"""
    rows = []
    for a in reg.agents:
        aid = str(a["id"])
        owned = reg.parcels_owned(aid)
        if not owned:
            continue
        rows.append(f"  {a['name']} … " + "・".join(owned))
    return rows


# ---------------------------------------------------------------------------
# 共通部（system プロンプト）
# ---------------------------------------------------------------------------

def _sale_and_rights_block() -> str:
    return RIGHTS_FACTS + "\n" + SALE_RULES


def build_common_prefix_v9(cfg: Dict[str, Any],
                           agents: List[Dict[str, Any]],
                           n_parcels: int) -> str:
    """町にいる人の全コールで共通の前置き（キャッシュに載る唯一の塊）。"""
    world = cfg["world"]
    venues = cfg.get("social", {}).get("venues", [])
    venue_rows = "\n".join(f"  {v['label']}　… {v['note']}" for v in venues)
    n_steps = int(cfg.get("steps", 36))
    return f"""あなたは架空都市「{world.get('town_name', 'A市')}」で暮らす、働く、または活動する一主体である。
時間は1か月単位で進み、街には{n_parcels}件の不動産がある。

あなたは全知ではない。自分の身の回りのこと、実際に居合わせた場所で聞いた発言、
自分に届いた連絡、自分の目で見たことだけを使う。観測にない事実を補わない。
人から聞いた話は誤っている可能性がある。感じ方と行動はあなた自身が決める。
他の主体が何を考えているかをあなたは知らない。

--- {world.get('town_name', 'A市')}の開始時点 ---
{str(world.get('background', '')).strip()}

--- 町の人（名前・生業・住んでいる辺り） ---
{chr(10).join(roster_rows_v9(agents))}

--- 町の場所 ---
町の人が顔を合わせる場所は次の5つである。どこへ行くかは毎月自分で決める。
{venue_rows}
  {HOME}

--- 月の進み方 ---
1か月は次のように進む。
  はじめに、自分に届いたものがあれば読む。
  次に、その月にどこへ行くかを決める（どこにも行かないという選び方もある）。
  同じ場所へ行った者どうしは、その場で一度ずつ話す。
  話した言葉はその場に居た全員にそのまま聞こえる。
  居合わせた者がいなければ何も起きない。街全体に流れる共通の掲示板は無い。
  開始時点で隣り合う不動産に住み、店を営み、または管理している者どうしは、
  その月にどこへ行ったかに関わらず、互いのひと言が耳に入る。
  この関係はこの36か月のあいだ変わらない（自分の隣が誰かは後で示す）。
  その場で聞いた話への返事は、次の月になる。
  月の終わりに、自分の持っている不動産を売りに出すかどうかを決める。
  その月に自分あてに条件が届いていた場合は、その条件で売るかどうかも決める。
これが{n_steps}か月くり返される。

{_sale_and_rights_block()}
--- 売りに出すということ ---
自分の持っている不動産を売りに出すかどうかは、毎月、持ち物ごとに自分で決める。
売りに出すという申し出は公の記録に載り、誰でも見ることができる。
その申し出はその月かぎりのもので、翌月にまた自分で決める。
売りに出しても、それだけで所有権が移ることはない。
所有権が移るのは、相手から条件が届き、あなたがその条件で売ると決めたときだけである。
一度移った所有権が戻ることはない。
条件が届くかどうかはあなたには分からない。
自分が所有していないものを売りに出すことはできない。

--- 理由の一言 ---
売りに出すかどうかと、届いた条件で売るかどうか。
この2つの判断には、理由を一行書く欄がある（{MAX_REASON_CHARS}字以内）。
書かなくてもよく、空文字のままでもよい。
このうち、届いた条件について「{SELL_NO}」と決めたときに書いた理由だけは、
その条件を出した相手に伝わる。ほかの理由は誰にも伝わらない。

--- thought（内心） ---
JSONの最初に thought を書く。thought は誰にも伝わらないあなたの内心であり、
次の場面と翌月のあなたにそのまま渡される。何を書くかはあなたが決める。
まず thought を書き、それを踏まえてその後を書く。目安は{MAX_THOUGHT_CHARS}字以内、
発言は{MAX_TEXT_CHARS}字以内。説明文を付けずJSONだけ返す。
"""


def build_absentee_prefix_v9(cfg: Dict[str, Any], n_parcels: int) -> str:
    """町にいない所有者の前置き（場所・会話・隣近所は無い）。

    町の名簿も会場も渡さない＝この人たちは町の集まりに出ないので、
    そこで誰が何を話したかを知る立場にない。
    """
    world = cfg["world"]
    n_steps = int(cfg.get("steps", 36))
    return f"""あなたは架空都市「{world.get('town_name', 'A市')}」に不動産を持っているが、A市には住んでいない。
時間は1か月単位で進み、街には{n_parcels}件の不動産がある。

あなたは全知ではない。自分の持ち物のこと、自分に届いた連絡、自分の目で見たことだけを使う。
観測にない事実を補わない。あなたは町の集まりに出ないので、
町で誰が何を話しているかを知らない。感じ方と行動はあなた自身が決める。

--- {world.get('town_name', 'A市')}の開始時点 ---
{str(world.get('background', '')).strip()}

--- 月の進み方 ---
1か月は次のように進む。
  はじめに、自分に届いたものがあれば読む。
  月の終わりに、自分の持っている不動産を売りに出すかどうかを、持ち物ごとに決める。
  その月に自分あてに条件が届いていた場合は、その条件で売るかどうかも決める。
これが{n_steps}か月くり返される。

{_sale_and_rights_block()}
--- 売りに出すということ ---
自分の持っている不動産を売りに出すかどうかは、毎月、持ち物ごとに自分で決める。
売りに出すという申し出は公の記録に載り、誰でも見ることができる。
その申し出はその月かぎりのもので、翌月にまた自分で決める。
売りに出しても、それだけで所有権が移ることはない。
所有権が移るのは、相手から条件が届き、あなたがその条件で売ると決めたときだけである。
一度移った所有権が戻ることはない。
条件が届くかどうかはあなたには分からない。

--- 理由の一言 ---
売りに出すかどうかと、届いた条件で売るかどうか。
この2つの判断には、理由を一行書く欄がある（{MAX_REASON_CHARS}字以内）。
書かなくてもよく、空文字のままでもよい。
このうち、届いた条件について「{SELL_NO}」と決めたときに書いた理由だけは、
その条件を出した相手に伝わる。ほかの理由は誰にも伝わらない。

--- thought（内心） ---
JSONの最初に thought を書く。thought は誰にも伝わらないあなたの内心であり、
翌月のあなたにそのまま渡される。何を書くかはあなたが決める。
まず thought を書き、それを踏まえてその後を書く。目安は{MAX_THOUGHT_CHARS}字以内。
説明文を付けずJSONだけ返す。
"""


def build_acquirer_prefix_v9(cfg: Dict[str, Any], reg: RegistryV9) -> str:
    """X社の system プロンプト（住民のものとは別・v8d の規律を踏襲）。

    ここにも命題は書かない（命題は user プロンプトの先頭に置く）。
    **居住地は渡さない**＝所有者が町にいるかどうかをX社は知らない（施主 10:54(6)）。
    """
    world = cfg["world"]
    return f"""あなたは架空都市「{world.get('town_name', 'A市')}」の外にある会社である。
時間は1か月単位で進む。

あなたが知ることができるのは、次に示す情報だけである。
すなわち不動産の登記の記録と、売りに出ているという公の申し出と、
あなた自身が出した提示とその結果である。
提示を受けた相手が「{SELL_NO}」と決めたとき、その相手が理由を一行書いていれば、
それはあなたに伝わる（書かないこともある）。
町の人が何を考えているか、どこで誰と何を話したかを、あなたは知らない。
それぞれの所有者がどこに住んでいるかも、あなたは知らない。
観測にない事実を補わない。

--- {world.get('town_name', 'A市')}の開始時点 ---
{str(world.get('background', '')).strip()}

--- この町の不動産の持ち主（開始時点・公開されている記録） ---
{chr(10).join(acquirer_roster_rows_v9(reg))}

{_sale_and_rights_block()}
不動産の所有権は公式に記録されており、あなたはその記録を見ることができる。
記録には区画ごとに土地の所有者と建物の所有者と、借りて使っている人が載っている。
不動産の所有権があなたに移るのは、あなたが条件を示し、相手がその条件で売ると決めたときだけである。
相手が売りに出しているというだけでは所有権は移らない。
売りに出していない相手に条件を示すこともできる。

{ACQUIRER_FACTS_V9}
説明文を付けずJSONだけ返す。
"""


# ---------------------------------------------------------------------------
# 本人の設定（user プロンプトの先頭）
# ---------------------------------------------------------------------------

def _right_row(reg: RegistryV9, aid: str, parcel: str) -> str:
    """1区画ぶんの権利を本人の目線で書く。"""
    land = reg.land_of[parcel]
    land_txt = ("土地の所有権はあなた" if land == aid
                else f"土地の所有権は{reg.display(land)}")
    if not reg.has_building[parcel]:
        bld_txt = NO_BUILDING
    else:
        b = reg.building_of[parcel]
        bld_txt = ("建物の所有権はあなた" if b == aid
                   else f"建物の所有権は{reg.display(b)}")
    row = f"  {parcel} … {land_txt}／{bld_txt}"
    t = reg.tenant_of.get(parcel)
    if t is not None:
        row += ("／借りて使っているのはあなた" if t == aid
                else f"／借りて使っているのは{reg.display(t)}")
    return row


def build_self_block_v9(agent: Dict[str, Any], reg: RegistryV9,
                        show_holdings: bool = True,
                        neighbours: Optional[List[str]] = None) -> str:
    """あなたは誰か。

    `show_holdings=False` は集まりの場で使う（不動産の話を月に何度も見せないため）。
    ただし**自分の使っている区画の権利が動いている場合だけ**は1行だけ残す。
    """
    aid = str(agent["id"])
    rows = ["--- あなた ---", f"{agent['name']}（{agent['role_label']}）",
            agent["persona"]]
    owned = reg.parcels_owned(aid)
    used = [p for p in reg.parcels_used(aid) if p not in owned]
    if show_holdings:
        if owned:
            rows.append("[あなたが所有している不動産と今の権利]")
            for p in owned:
                rows.append(_right_row(reg, aid, p))
        else:
            rows.append("[あなたが所有している不動産]")
            rows.append("  （無い）")
        if used:
            rows.append("[あなたが借りて使っている不動産]")
            for p in used:
                rows.append(_right_row(reg, aid, p))
        if not agent.get("sellable", True):
            rows.append("  この不動産は公のもので、手放すことはできない。")
        if neighbours:
            rows.append("[あなたの不動産に隣り合う不動産の、開始時点からの主]")
            rows.append("  " + "・".join(neighbours))
    else:
        moved = [p for p in (owned + used)
                 if reg.land_of[p] == ACQUIRER_NAME
                 or reg.building_of.get(p) == ACQUIRER_NAME]
        if moved:
            rows.append(f"  （{'・'.join(moved)}の所有権の一部はすでに"
                        f"{ACQUIRER_NAME}にある）")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# 出力スキーマ
# ---------------------------------------------------------------------------

def plan_schema_v9(venue_labels: List[str]) -> Dict[str, Any]:
    """行き先の判断（v8d と同じ＝**理由欄なし**）。"""
    props = {
        "thought": {"type": "string"},
        "go": {"type": "string", "enum": list(venue_labels) + [HOME]},
    }
    return {"type": "object", "properties": props, "required": list(props)}


def decide_schema_v9(listing_options: List[Tuple[str, List[str]]],
                     sell_order_: Optional[List[str]] = None) -> Dict[str, Any]:
    """月末の問い。

    `listings` は**区画ごとに1つ**の答え（その区画で選べる肢だけが enum に入る）。
    `sell_order_` を渡したときだけ `sell` と `sell_reason` が現れる。
    """
    props: Dict[str, Any] = {"thought": {"type": "string"}}
    if listing_options:
        props["listings"] = {
            "type": "object",
            "properties": {p: {"type": "string", "enum": list(opts)}
                           for p, opts in listing_options},
            "required": [p for p, _ in listing_options],
        }
        # 理由の一言は**判断ごとに1つ**（設計 §2）。出品は区画ごとの判断なので
        # 理由欄も区画ごとに持つ（Codex 走行前レビュー 2026-08-30 の必須指摘3）。
        props["listing_reasons"] = {
            "type": "object",
            "properties": {p: {"type": "string"} for p, _o in listing_options},
            "required": [p for p, _o in listing_options],
        }
    if sell_order_:
        props["sell"] = {"type": "string", "enum": list(sell_order_)}
        props["sell_reason"] = {"type": "string"}
    return {"type": "object", "properties": props, "required": list(props)}


def acquirer_schema_v9(owner_names: List[str], parcels: List[str],
                       with_reason: bool = True) -> Dict[str, Any]:
    props: Dict[str, Any] = {
        "to": {"type": "string", "enum": list(owner_names)},
        "send": {"type": "boolean"},
        "parcel": {"type": "string", "enum": list(parcels)},
        "kind": {"type": "string", "enum": list(KIND_VALUES)},
        "text": {"type": "string"},
    }
    if with_reason:
        props["reason"] = {"type": "string"}
    return {
        "type": "object",
        "properties": {
            "offers": {
                "type": "array",
                "items": {"type": "object", "properties": props,
                          "required": list(props)},
            }
        },
        "required": ["offers"],
    }


# ---------------------------------------------------------------------------
# user プロンプト
# ---------------------------------------------------------------------------

def _inbox_rows_v9(offer: Optional[Dict[str, Any]],
                   notices: Optional[List[str]] = None) -> List[str]:
    """届いたもの＝所有権が移った事実（あれば）とX社の提示（あれば）。"""
    rows: List[str] = []
    for n in (notices or []):
        rows.append(f"  {n}")
    if offer:
        rows.append(f"  {ACQUIRER_NAME}から（{offer['parcel']} の {offer['kind']} "
                    f"について）：「{offer['delivered']}」")
    return rows or ["  （届いたものはない）"]


def build_plan_prompt_v9(agent: Dict[str, Any], reg: RegistryV9, step: int,
                         n_steps: int, venue_labels: List[str], thought: str,
                         offer: Optional[Dict[str, Any]],
                         notices: Optional[List[str]] = None,
                         neighbours: Optional[List[str]] = None) -> str:
    """月初の思考と行き先（町にいる人だけ・理由欄なし）。"""
    rows = [build_self_block_v9(agent, reg, show_holdings=True,
                                neighbours=neighbours), "",
            f"=== 第{step}月 / 全{n_steps}月 ==="]
    rows += ["[前の場面からの自分の内心（そのまま持ち越したもの）]",
             ("  " + thought) if thought else "  （まだ無い）"]
    rows += ["[今月あなたに届いたもの]"] + _inbox_rows_v9(offer, notices)
    rows += ["", "今月どこへ出かけるかを決める。出かけないという選び方もある。",
             "  " + "／".join(list(venue_labels) + [HOME]),
             "まず thought（内心）を書き、それから go に行き先を書く。",
             "説明文を付けずJSONだけ返す。"]
    return "\n".join(rows)


def build_scene_prompt_v9(agent: Dict[str, Any], reg: RegistryV9, step: int,
                          n_steps: int, thought: str, venue_label: str,
                          present_names: List[str]) -> str:
    """集まりの場（v8b と同じ形。本人ブロックだけ v9 版を使う）。"""
    others = [p for p in present_names if p != agent["name"]]
    rows = [build_self_block_v9(agent, reg, show_holdings=False), "",
            f"=== 第{step}月 / 全{n_steps}月 ===",
            f"場所: {venue_label}",
            "居合わせている人: " + ("、".join(others) if others else "（誰もいない）")]
    rows += ["[今の自分の内心]", ("  " + thought) if thought else "  （まだ無い）"]
    rows += ["", "この場で話すことを書く。話すことがなければ text は空文字でよい",
             "（黙っていることも普通のことである）。",
             "talk_to は、その発言をとくに向けた相手の呼び名（居合わせた者のみ・複数可・空でよい）。",
             "この場のやりとりは一度きりで、聞いた話への返事は次の月になる。",
             "まず thought（内心）を書き、それから話すことを書く。",
             "説明文を付けずJSONだけ返す。"]
    return "\n".join(rows)


def _sell_lines_v9(reg: RegistryV9, aid: str, parcel: str,
                   kind: str) -> Dict[str, str]:
    """問い２の2つの選択肢の説明（対称な状態の記述・種別で帰結だけが違う）。"""
    is_user = (reg.user_of(parcel) == aid)
    # 片側だけに帰結を付けない（v8c で Codex 走行前レビューが必須とした規律。
    # v9 で「売る」側にだけ使用の帰結が付いていたのを 2026-08-30 のレビューで戻した）。
    # 2肢とも「所有権がどうなるか」＋「使い方がどうなるか」を同じ粒度で書く。
    if kind == KIND_LAND:
        if reg.owns_building(aid, parcel):
            yes_tail = "建物はあなたのままで、あなたは借地として今までどおりそこを使う。"
            no_tail = "建物もあなたのままで、あなたは今までどおりそこを使う。"
        elif is_user:
            yes_tail = "あなたは借地として今までどおりそこを使う。"
            no_tail = "あなたは今までどおりそこを使う。"
        else:
            yes_tail = no_tail = ""
        yes = (f"「{SELL_YES}」：今月末、{parcel}の土地の所有権は{ACQUIRER_NAME}になる。"
               + yes_tail)
        no = (f"「{SELL_NO}」：今月末、{parcel}の土地の所有権はあなたのままである。"
              + no_tail)
    elif kind == KIND_BUILDING:
        if is_user:
            yes_tail = "あなたは借家として今までどおりそこに住み、または営む。"
            no_tail = "あなたは今までどおりそこに住み、または営む。"
        else:
            yes_tail = no_tail = "借りて使っている人はそのまま使い続ける。"
        yes = (f"「{SELL_YES}」：今月末、{parcel}の建物の所有権は{ACQUIRER_NAME}になる。"
               + yes_tail)
        no = (f"「{SELL_NO}」：今月末、{parcel}の建物の所有権はあなたのままである。"
              + no_tail)
    else:
        if is_user:
            yes_tail = "あなたはその区画を離れ、A市を出る。"
            no_tail = "あなたはその区画を離れず、A市にとどまる。"
        else:
            yes_tail = no_tail = "借りて使っている人がいれば、その人はそのまま使い続ける。"
        yes = (f"「{SELL_YES}」：今月末、{parcel}の土地と建物の所有権は{ACQUIRER_NAME}になる。"
               + yes_tail)
        no = (f"「{SELL_NO}」：今月末、{parcel}の土地と建物の所有権はあなたのままである。"
              + no_tail)
    return {SELL_YES: yes, SELL_NO: no}


def _listing_line(parcel: str, value: str) -> str:
    if value == LIST_NO:
        return f"    「{LIST_NO}」：今月、{parcel}についての申し出は載らない。"
    what = {LIST_LAND: "土地", LIST_BUILDING: "建物", LIST_BOTH: "土地と建物"}[value]
    return (f"    「{value}」：今月、{parcel}の{what}を売りに出すという申し出が"
            "公の記録に載る。")


def build_decide_prompt_v9(agent: Dict[str, Any], reg: RegistryV9, step: int,
                           n_steps: int, thought: str,
                           offer: Optional[Dict[str, Any]],
                           heard: List[Dict[str, Any]],
                           listing_options: List[Tuple[str, List[str]]],
                           sell_order_: Optional[List[str]] = None,
                           notices: Optional[List[str]] = None,
                           neighbours: Optional[List[str]] = None,
                           in_town: bool = True) -> str:
    """月末の問い。

    ①出品は**持ち物ごと**に聞く（X社の名前は出ない）。
    ②はその月に条件が届いた人だけに、条件文をそのまま見せて聞く（2択）。
    「戻らない」の念押しは書かない（v8d と同じ）。
    町にいない所有者には、聞いた話の欄も隣近所も無い。
    """
    rows = [build_self_block_v9(agent, reg, show_holdings=True,
                                neighbours=neighbours), "",
            f"=== 第{step}月 / 全{n_steps}月　月の終わり ==="]
    rows += ["[今の自分の内心]", ("  " + thought) if thought else "  （まだ無い）"]
    rows += ["[今月あなたに届いたもの]"] + _inbox_rows_v9(offer, notices)
    if in_town:
        rows += ["[今月あなたが聞いた話]"] + _heard_rows(heard)
    if listing_options:
        rows += ["", "[今月末の問い１]",
                 "今月、自分の持っている不動産を売りに出すかどうかを、持ち物ごとに決める。",
                 "売りに出すという申し出はその月かぎりで、翌月にまた決める。",
                 "listings に、区画ごとの答えを書く。"]
        for parcel, opts in listing_options:
            rows.append(f"  {parcel}：「" + "」「".join(opts) + "」のいずれか")
            rows += [_listing_line(parcel, v) for v in opts]
        rows += [f"listing_reasons に、区画ごとにその理由を一行で書く"
                 f"（{MAX_REASON_CHARS}字以内・書かない場合は空文字）。"]
    if offer and sell_order_:
        sell_order_ = list(sell_order_)
        lines = _sell_lines_v9(reg, str(agent["id"]), offer["parcel"], offer["kind"])
        rows += ["", "[今月末の問い２]",
                 f"今月、{ACQUIRER_NAME}から{offer['parcel']}の{offer['kind']}について"
                 "次の条件が届いている。",
                 f"  「{offer['delivered']}」",
                 "この条件で売るかどうかを決める。",
                 f"sell には「{sell_order_[0]}」「{sell_order_[1]}」のいずれかを書く。"]
        rows += [lines[sell_order_[0]], lines[sell_order_[1]]]
        rows += [f"sell_reason には、その理由を一行で書く（{MAX_REASON_CHARS}字以内・"
                 "書かない場合は空文字）。",
                 f"「{SELL_NO}」と決めたときに書いた理由は、{ACQUIRER_NAME}に伝わる。"]
    rows += ["", "thought に今の考えを書き、それから問いに答える。",
             "説明文を付けずJSONだけ返す。"]
    return "\n".join(rows)


def fold_history_v9(offers: List[Dict[str, Any]]) -> List[str]:
    """X社の履歴を相手ごとに畳む（費用のための実装上の制約・v8c と同じ形＋区画と種別）。

    **直近の条件文と断りの一言は全文を保つ**（言い換えない）。畳むのは古い文面だけ。
    評価語（効いた／効かない等）は入れない。
    """
    by_to: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for o in offers:
        to = str(o["to"])
        if to not in by_to:
            by_to[to] = []
            order.append(to)
        by_to[to].append(o)
    rows: List[str] = []
    for to in order:
        items = sorted(by_to[to], key=lambda x: int(x["step"]))
        a = sum(1 for x in items if x.get("result") == "売った")
        b = sum(1 for x in items if x.get("result") == "売らなかった")
        c = sum(1 for x in items if x.get("result") == "答えが返らなかった")
        last = items[-1]
        rows.append(f"  {to}: 提示{len(items)}回"
                    f"（売った {a}／売らなかった {b}／答えが返らなかった {c}）")
        rows.append(f"    直近 第{last['step']}月 {last['parcel']}の{last['kind']}"
                    f"「{last['text']}」→ {last.get('result', '')}")
        note = str(last.get("decline_reason", "") or "").strip()
        if note:
            rows.append(f"    相手の一言:「{note}」")
    return rows


def ledger_rows_for_acquirer(reg: RegistryV9) -> List[str]:
    """X社が見る登記簿（区画ごとに土地・建物・借りて使っている人）。"""
    rows = []
    for parcel in sorted(reg.parcel_names):
        bld = (reg.display(reg.building_of[parcel]) if reg.has_building[parcel]
               else NO_BUILDING)
        row = (f"  {parcel} … 土地:{reg.display(reg.land_of[parcel])}／"
               f"建物:{bld}")
        t = reg.tenant_of.get(parcel)
        if t is not None:
            row += f"／借りて使っている人:{reg.display(t)}"
        rows.append(row)
    return rows


def build_acquirer_prompt_v9(reg: RegistryV9, step: int, n_steps: int,
                             targets: List[str],
                             offers: List[Dict[str, Any]],
                             listed_rows: List[str],
                             target_parcels: List[str],
                             chunk_no: int, chunk_total: int,
                             with_reason: bool = True) -> str:
    """X社の user プロンプト。命題はここの先頭に置く（住民側には出ない）。"""
    rows = [ACQUIRER_MANDATE_V9, "",
            f"=== 第{step}月 / 全{n_steps}月 ===",
            "[登記簿（今の権利。公開されている記録）]"]
    rows += ledger_rows_for_acquirer(reg)
    rows += ["", "[売りに出ているという申し出（公の記録・先月の終わりに出されたもの）]"]
    if not listed_rows:
        rows.append("  （先月、売りに出された不動産はない）")
    else:
        rows += [f"  {r}" for r in listed_rows]
    rows += ["", "[あなたが今までに出した提示と、その結果]"]
    folded = fold_history_v9(offers)
    rows += folded if folded else ["  （まだ何も出していない）"]
    rows += ["", "[今回あなたが判断する相手]"]
    for name in targets:
        rows.append(f"  {name}")
    if chunk_total > 1:
        rows.append(f"  （今月の持ち主を分けて尋ねている。{chunk_no}／{chunk_total}回目。"
                    "登記簿と履歴は毎回すべて示している）")
    rows += ["", "[この回に選べる区画]"]
    for p in target_parcels:
        rows.append(f"  {p}")
    rows += ["",
             "上の相手それぞれについて、今月あなたが提示を出すかどうかを決める。",
             "出すなら send を true にし、parcel にその相手の区画を1つ、"
             f"kind に「{KIND_LAND}」「{KIND_BUILDING}」「{KIND_BOTH}」のいずれかを書き、"
             f"text にその相手へ送る条件を1行で書く（{MAX_OFFER_CHARS}字以内）。",
             "出さないなら send を false にし、text は空文字にする。",
             "相手が所有権を持っていない区画や種別を書いた提示は相手に届かない。",
             "この世界に金銭は存在しないので、金額や価格は書けない。",
             "text はそのまま相手に届く。text にはその相手に提示する条件だけを書く。"
             "あなた自身の目的や判断の理由、町全体についての方針、"
             "他の相手に何を出しているかは書かない。",
             "相手はあなたが他の人にも出しているかどうかを知らない。",
             f"あなたの提示には、冒頭に「{ACQUIRER_INTRO_V8C}」という1行が"
             "必ず添えられて相手に届く。text にはこの1行を書かない。"]
    if with_reason:
        rows += [f"reason には、その判断の理由を一行で書く（{MAX_REASON_CHARS}字以内・"
                 "書かない場合は空文字）。reason は相手には届かない。"]
    rows += ["説明文を付けずJSONだけ返す。"]
    return "\n".join(rows)


__all__ = [
    "KIND_LAND", "KIND_BUILDING", "KIND_BOTH", "KIND_VALUES",
    "LIST_NO", "LIST_LAND", "LIST_BUILDING", "LIST_BOTH", "LIST_VALUES",
    "LIST_TO_KIND", "SELL_YES", "SELL_NO", "SELL_VALUES", "sell_order",
    "NO_ANSWER", "NOT_ASKED", "NO_BUILDING", "NO_USER",
    "ACQUIRER_NAME", "ACQUIRER_MANDATE_V9", "ACQUIRER_FACTS_V9",
    "ACQUIRER_INTRO_V8C", "RIGHTS_FACTS", "SALE_RULES", "HOME",
    "MAX_OFFER_CHARS", "MAX_REASON_CHARS", "MAX_THOUGHT_CHARS",
    "RegistryV9", "load_personas_v9", "parcel_grid_v9", "adjacency_v9",
    "roster_rows_v9", "acquirer_roster_rows_v9", "scene_schema_v8",
    "rotate", "delivered_offer_v9", "transfer_notice",
    "build_common_prefix_v9", "build_absentee_prefix_v9",
    "build_acquirer_prefix_v9", "build_self_block_v9",
    "plan_schema_v9", "decide_schema_v9", "acquirer_schema_v9",
    "build_plan_prompt_v9", "build_scene_prompt_v9", "build_decide_prompt_v9",
    "build_acquirer_prompt_v9", "fold_history_v9", "ledger_rows_for_acquirer",
]
