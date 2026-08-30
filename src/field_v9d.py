"""A市フィールド v9d「ものさしとしてのお金」— v9c からの差分だけ。

設計の正は `docs/world_design_v9d.md`（土台＝`docs/world_design_v9c.md`）。
施主確定 2026-08-30 16:33。

**v1〜v9c のファイルは1バイトも触らない。**

v9c からの差分は4つだけ:
  1. **評価額（公開）**＝44区画の土地・建物ごとに評価額がある。誰でも見られる。
     数字は走行前に凍結する（結果を見て動かさない）。
  2. **X社の提示に金額が必須**＝区画＋種別＋金額（円）＋条件文。金額はX社が自分で決める。
  3. **X社の資金は有限**＝町の全評価額の 51%。成約ごとに減り、残額は毎月X社が知る。
     残額を超える金額の提示は世界が配らない（件数を数える）。
  4. **財布は記帳だけ**＝売った人の財布に金額が入る（開始時0）。
     財布が何かを起こすことは無い（賃料・維持費・税・買い物はこの世界に無い）。

「得か損か」はどこにも書かない。世界が置くのは**評価額と提示額という2つの数字**だけである。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .field_v9 import (ACQUIRER_INTRO_V8C, ACQUIRER_MANDATE_V9,  # noqa: F401
                       ACQUIRER_NAME, KIND_BOTH, KIND_BUILDING, KIND_LAND,
                       KIND_VALUES, MAX_OFFER_CHARS, MAX_REASON_CHARS,
                       RegistryV9, _sale_and_rights_block,
                       acquirer_roster_rows_v9, fold_history_v9)
from .field_v9b import (ACQUIRER_FACTS_V9B, UNDELIVERED_PROMISE,  # noqa: F401
                        UNDELIVERED_RIGHTS, fold_undelivered_v9b)
from .field_v9c import (LEAVE_NO, LEAVE_YES, MAX_LINE_CHARS,  # noqa: F401
                        build_absentee_prefix_v9c, build_common_prefix_v9c,
                        landlord_reply_block, time_header)

UNDELIVERED_BUDGET = "あなたの残りの資金では支払えない金額である"

# ---------------------------------------------------------------------------
# 評価額（走行前に凍結・結果を見て動かさない）
#
# 土地の単価は、別府市（大分県・人口約11万人の温泉観光都市）の公開値を土台に5段階で置く。
#   ・公示地価の平均 55,700円/m2（2026年・坪18.4万円）
#     https://e-estate.jp/land/market-price/oita/beppushi/
#     https://tochidai.info/oita/beppu/
#   ・基準地価 商業地の平均 70,240円/m2（2025年）
# A市は人口約12万人の温泉観光都市という設定なので、この水準をそのまま使う。
# 面積・延床・建物単価・残価率は、用途から常識的に置いた仮の値である（実在の物件ではない）。
# ---------------------------------------------------------------------------

# 5段階の土地単価（円/m2）
LAND_TIER: Dict[str, int] = {
    "駅前商業": 120_000,     # 中央駅前地区
    "観光沿岸": 80_000,      # 湾岸観光地区
    "温泉住宅": 55_000,      # 温泉丘陵地区（別府市の平均に相当）
    "学術生活": 40_000,      # 北部学術・生活地区
    "縁辺特殊": 20_000,      # 公園・境内地・源泉地・船着場裏
}

DISTRICT_TIER: Dict[str, str] = {
    "中央駅前地区": "駅前商業",
    "湾岸観光地区": "観光沿岸",
    "温泉丘陵地区": "温泉住宅",
    "北部学術・生活地区": "学術生活",
}

# 縁辺特殊に落とす区画（性格が地区の相場と違うもの）
EDGE_PARCELS = ("浜辺の街区公園", "神明社の境内地", "宮の下の源泉地",
                "船着場裏の駐車場", "船着場裏の資材置場")

# 用途ごとの (敷地m2, 延床m2, 建物単価 円/m2, 残価率)。建物が無い区画は延床0。
USE_SPEC: Dict[str, Tuple[int, int, int, float]] = {
    "住宅":       (180, 110, 160_000, 0.30),
    "古家":       (180, 110, 160_000, 0.15),
    "改装住宅":   (180, 110, 160_000, 0.30),
    "店舗":       (150, 120, 180_000, 0.25),
    "店舗兼住宅": (170, 150, 180_000, 0.30),
    "貸オフィス": (300, 600, 250_000, 0.45),
    "旅館":       (900, 1800, 280_000, 0.30),
    "ホテル":     (1500, 4500, 300_000, 0.40),
    "宿":         (250, 220, 200_000, 0.30),
    "社員寮":     (400, 700, 180_000, 0.35),
    "診療所":     (400, 320, 260_000, 0.45),
    "校舎":       (5000, 3000, 200_000, 0.25),
    "公民館":     (600, 400, 200_000, 0.35),
    "集会所":     (300, 180, 180_000, 0.30),
    "共同浴場":   (200, 150, 220_000, 0.25),
    "源泉地":     (300, 30, 150_000, 0.30),
    "境内地":     (800, 120, 200_000, 0.30),
    "駐車場小":   (200, 0, 0, 0.0),
    "駐車場中":   (300, 0, 0, 0.0),
    "駐車場大":   (500, 0, 0, 0.0),
    "資材置場":   (400, 0, 0, 0.0),
    "空き地":     (200, 0, 0, 0.0),
    "公園":       (1200, 0, 0, 0.0),
}

# 44区画の用途（走行前に凍結・区画名から機械的に決めたものを表として固定する）
PARCEL_USE: Dict[str, str] = {
    "湯坂上の古家": "古家",
    "湯坂上の空き店舗": "店舗",
    "駅前通りの家": "住宅",
    "観音坂の家": "住宅",
    "観音坂の空き地": "空き地",
    "湯の元の古家": "古家",
    "学園北の家": "住宅",
    "駅裏の小区画": "住宅",
    "一番街の家": "住宅",
    "一番街の資材置場": "資材置場",
    "潮見横丁の改装した古家": "改装住宅",
    "本町通りの家": "住宅",
    "本町通りの元店舗": "店舗",
    "学園南の家": "住宅",
    "湯坂下の家": "住宅",
    "波止場通りの元店舗": "店舗",
    "波止場通りの家": "住宅",
    "北団地の中古住宅": "住宅",
    "線路沿いの実家": "住宅",
    "岬道の家": "住宅",
    "学園西の家": "住宅",
    "学園西の空き家": "古家",
    "浜町の旅館": "旅館",
    "浜町の旅館の駐車場": "駐車場大",
    "駅前広場の店舗兼住宅": "店舗兼住宅",
    "湯の元前の店": "店舗",
    "学園前の貸オフィス": "貸オフィス",
    "駅裏北の月極駐車場": "駐車場中",
    "商店街裏の時間貸し駐車場": "駐車場小",
    "湾岸大通りのホテル": "ホテル",
    "湾岸大通りの社員寮": "社員寮",
    "浜辺の街区公園": "公園",
    "旧西小学校の校舎": "校舎",
    "西地区の公民館": "公民館",
    "灯台下の宿": "宿",
    "宮の下の共同浴場": "共同浴場",
    "宮の下の源泉地": "源泉地",
    "船着場裏の駐車場": "駐車場大",
    "船着場裏の資材置場": "資材置場",
    "湯坂奥の空き家": "古家",
    "神明社の境内地": "境内地",
    "神明社前の集会所": "集会所",
    "魚市場前の診療所": "診療所",
    "魚市場前の診療所の駐車場": "駐車場中",
}


def _round(v: float) -> int:
    """10万円単位に丸める（読みやすさのため・決定論）。"""
    return int(round(v / 100_000.0) * 100_000)


def tier_of(parcel: str, district: str) -> str:
    if parcel in EDGE_PARCELS:
        return "縁辺特殊"
    return DISTRICT_TIER.get(district, "温泉住宅")


def build_valuation(parcels: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """区画ごとの評価額（土地・建物）。決定論・凍結した表からしか作らない。"""
    out: Dict[str, Dict[str, Any]] = {}
    for p in parcels:
        name = str(p["name"])
        use = PARCEL_USE[name]
        area, floor, unit, rate = USE_SPEC[use]
        tier = tier_of(name, str(p["district"]))
        land = _round(area * LAND_TIER[tier])
        bld = _round(floor * unit * rate) if bool(p["building"]) else 0
        if bool(p["building"]) and bld <= 0:
            bld = 100_000
        out[name] = {"use": use, "tier": tier, "area": area, "floor": floor,
                     "land": land, "building": bld, "both": land + bld}
    return out


def yen(v: int) -> str:
    return f"{int(v):,}円"


def value_of(val: Dict[str, Dict[str, Any]], parcel: str, kind: str) -> int:
    row = val[parcel]
    if kind == KIND_LAND:
        return int(row["land"])
    if kind == KIND_BUILDING:
        return int(row["building"])
    return int(row["both"])


def total_value(val: Dict[str, Dict[str, Any]]) -> int:
    return sum(int(r["both"]) for r in val.values())


# ---------------------------------------------------------------------------
# 世界の事実（3者に同じ文で置く）
# ---------------------------------------------------------------------------

MONEY_FACTS = """--- お金 ---
この町の不動産には、土地と建物それぞれに評価額がある。評価額は公開されていて、誰でも見ることができる。
不動産の所有権が移るとき、買い手は金額を示し、その金額が売った人の記録に入る。
この世界にあるお金は、不動産の所有権が移るときに動くものだけである。
賃料はない。借地も借家も、これまでどおり金銭のやり取りなしに続く。
維持費も税もない。お金で何かを買う仕組みもない。記録に入った金額は、そのまま記録に残る。
"""


def money_block_for_owner(val: Dict[str, Dict[str, Any]], reg: RegistryV9,
                          aid: str, parcels: List[str], wallet: int) -> List[str]:
    """本人に見せる評価額（自分に関わる区画だけ）と、自分の記録の金額。"""
    rows = ["[公開されている評価額]"]
    for p in parcels:
        r = val[p]
        if reg.has_building.get(p):
            rows.append(f"  {p} … 土地 {yen(r['land'])}／建物 {yen(r['building'])}")
        else:
            rows.append(f"  {p} … 土地 {yen(r['land'])}（建物は無い）")
    rows.append(f"[あなたの記録に入っている金額] {yen(wallet)}")
    return rows


def offer_amount_row(val: Dict[str, Dict[str, Any]], parcel: str, kind: str,
                     amount: int) -> str:
    """届いた提示の金額と、その区画・種別の評価額を**並べる**（比べる語は書かない）。"""
    return (f"  {ACQUIRER_NAME}が示した金額：{yen(amount)}"
            f"（{parcel}の{kind}の評価額：{yen(value_of(val, parcel, kind))}）")


# ---------------------------------------------------------------------------
# 前置き（v9c の文＋お金の節）
# ---------------------------------------------------------------------------

def build_common_prefix_v9d(cfg, agents, n_parcels, left_ids=None) -> str:
    base = build_common_prefix_v9c(cfg, agents, n_parcels, left_ids)
    return base.replace("--- 借りて使うということ ---",
                        MONEY_FACTS + "\n--- 借りて使うということ ---")


def build_absentee_prefix_v9d(cfg, n_parcels) -> str:
    base = build_absentee_prefix_v9c(cfg, n_parcels)
    return base.replace("--- 借りて使うということ ---",
                        MONEY_FACTS + "\n--- 借りて使うということ ---")


ACQUIRER_MONEY_FACTS = """あなたが使えるお金には限りがある。残りの金額は毎月示される。
提示した金額は、相手が売ると決めたときにあなたの残りの金額から引かれる。
残りの金額を超える金額の提示は相手に届かない。
不動産の評価額は公開されていて、あなたも見ることができる。
金額をいくらにするかはあなたが決める。
"""


def build_acquirer_prefix_v9d(cfg: Dict[str, Any], reg: RegistryV9,
                              budget: int) -> str:
    world = cfg["world"]
    return f"""あなたは架空都市「{world.get('town_name', 'A市')}」の外にある会社である。
時間は1か月単位で進む。

あなたが知ることができるのは、次に示す情報だけである。
すなわち不動産の登記の記録と、公開されている評価額と、売りに出ているという公の申し出と、
あなた自身が出した提示とその結果である。
提示を受けた相手が「売らない」と決めたとき、その相手が理由を一行書いていれば、
それはあなたに伝わる（書かないこともある）。
町の人が何を考えているか、どこで誰と何を話したかを、あなたは知らない。
それぞれの所有者がどこに住んでいるかも、あなたは知らない。
観測にない事実を補わない。

--- {world.get('town_name', 'A市')}の開始時点 ---
{str(world.get('background', '')).strip()}

--- この町の不動産の持ち主（開始時点・公開されている記録） ---
{chr(10).join(acquirer_roster_rows_v9(reg))}

{_sale_and_rights_block()}
{MONEY_FACTS}
不動産の所有権は公式に記録されており、あなたはその記録を見ることができる。
記録には区画ごとに土地の所有者と建物の所有者と、借りて使っている人が載っている。
不動産の所有権があなたに移るのは、あなたが条件と金額を示し、
相手がその条件で売ると決めたときだけである。
相手が売りに出しているというだけでは所有権は移らない。
売りに出していない相手に条件を示すこともできる。

{ACQUIRER_FACTS_V9B}{ACQUIRER_MONEY_FACTS}第1月の開始時点で、あなたが使えるお金は {yen(budget)} である。
説明文を付けずJSONだけ返す。
"""


# ---------------------------------------------------------------------------
# X社の出力スキーマと user プロンプト
# ---------------------------------------------------------------------------

def acquirer_schema_v9d(owner_names: List[str], parcels: List[str],
                        with_reason: bool = True) -> Dict[str, Any]:
    props: Dict[str, Any] = {
        "to": {"type": "string", "enum": list(owner_names)},
        "send": {"type": "boolean"},
        "parcel": {"type": "string", "enum": list(parcels)},
        "kind": {"type": "string", "enum": list(KIND_VALUES)},
        "amount": {"type": "integer"},
        "text": {"type": "string"},
    }
    if with_reason:
        props["reason"] = {"type": "string"}
    return {"type": "object",
            "properties": {"offers": {"type": "array",
                                      "items": {"type": "object",
                                                "properties": props,
                                                "required": list(props)}}},
            "required": ["offers"]}


def ledger_rows_with_value(reg: RegistryV9,
                           val: Dict[str, Dict[str, Any]]) -> List[str]:
    rows = []
    for parcel in sorted(reg.parcel_names):
        r = val[parcel]
        if reg.has_building[parcel]:
            b = reg.display(reg.building_of[parcel])
            row = (f"  {parcel} … 土地:{reg.display(reg.land_of[parcel])}"
                   f"（評価額 {yen(r['land'])}）／建物:{b}（評価額 {yen(r['building'])}）")
        else:
            row = (f"  {parcel} … 土地:{reg.display(reg.land_of[parcel])}"
                   f"（評価額 {yen(r['land'])}）／建物:建物は無い")
        t = reg.tenant_of.get(parcel)
        if t is not None:
            row += f"／借りて使っている人:{reg.display(t)}"
        rows.append(row)
    return rows


def build_acquirer_prompt_v9d(reg: RegistryV9, val: Dict[str, Dict[str, Any]],
                              step: int, n_steps: int, targets: List[str],
                              offers: List[Dict[str, Any]],
                              listed_rows: List[str],
                              target_parcels: List[str],
                              chunk_no: int, chunk_total: int,
                              budget_left: int, budget_total: int,
                              spent: int,
                              with_reason: bool = True,
                              undelivered: Optional[List[Dict[str, Any]]] = None
                              ) -> str:
    """v9b の user プロンプト＋評価額・資金・金額欄。"""
    rows = [ACQUIRER_MANDATE_V9, "",
            f"=== 第{step}月 / 全{n_steps}月 ===",
            "[あなたが使えるお金]",
            f"  はじめの金額 {yen(budget_total)}／これまでに支払った金額 {yen(spent)}"
            f"／残り {yen(budget_left)}",
            "", "[登記簿（今の権利と、公開されている評価額）]"]
    rows += ledger_rows_with_value(reg, val)
    rows += ["", "[売りに出ているという申し出（公の記録・先月の終わりに出されたもの）]"]
    if not listed_rows:
        rows.append("  （先月、売りに出された不動産はない）")
    else:
        rows += [f"  {r}" for r in listed_rows]
    rows += ["", "[あなたが今までに出した提示と、その結果]"]
    folded = fold_history_v9(offers)
    rows += folded if folded else ["  （まだ何も出していない）"]
    rows += ["", "[相手に届かなかった提示（この世界で実行できないもの）]"]
    folded_u = fold_undelivered_v9b(list(undelivered or []))
    rows += folded_u if folded_u else ["  （届かなかった提示はない）"]
    rows += ["", "[今回あなたが判断する相手]"] + [f"  {n}" for n in targets]
    if chunk_total > 1:
        rows.append(f"  （今月の持ち主を分けて尋ねている。{chunk_no}／{chunk_total}回目。"
                    "登記簿と履歴は毎回すべて示している）")
    rows += ["", "[この回に選べる区画]"] + [f"  {p}" for p in target_parcels]
    rows += ["",
             "上の相手それぞれについて、今月あなたが提示を出すかどうかを決める。",
             "出すなら send を true にし、parcel にその相手の区画を1つ、"
             f"kind に「{KIND_LAND}」「{KIND_BUILDING}」「{KIND_BOTH}」のいずれかを書き、"
             "amount にその不動産に対して支払う金額を円の整数で書き、"
             f"text にその相手へ送る条件を1行で書く（{MAX_OFFER_CHARS}字以内）。",
             "出さないなら send を false にし、amount は 0、text は空文字にする。",
             "相手が所有権を持っていない区画や種別を書いた提示は相手に届かない。",
             "この世界で実行される仕組みが無い約束を含む提示も相手に届かない。",
             "あなたの残りの金額を超える金額の提示も相手に届かない。",
             "amount に書いた金額は、相手が売ると決めたときにあなたの残りの金額から引かれる。",
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
    "LAND_TIER", "DISTRICT_TIER", "EDGE_PARCELS", "USE_SPEC", "PARCEL_USE",
    "UNDELIVERED_BUDGET", "MONEY_FACTS", "ACQUIRER_MONEY_FACTS",
    "build_valuation", "value_of", "total_value", "yen", "tier_of",
    "money_block_for_owner", "offer_amount_row",
    "build_common_prefix_v9d", "build_absentee_prefix_v9d",
    "build_acquirer_prefix_v9d", "acquirer_schema_v9d",
    "ledger_rows_with_value", "build_acquirer_prompt_v9d",
]
