"""A市フィールド v9c「借りている人にも出口がある町」— v9b からの差分だけ。

設計の正は `docs/world_design_v9c.md`（土台＝`docs/world_design_v9b.md`）。
施主確定 2026-08-30 13:23（1〜3 GO）＋13:31（ペルソナ草案そのまま）。

**v1〜v9b のファイルは1バイトも触らない。** v9c はこのファイルと `src/sim_v9c.py`、
`run_v9c.py`、`configs/config_field_v9c.yaml`、`configs/personas_v9c.yaml`、
`tests/test_v9c.py`、`tools/v9c_*.py` だけで完結する。

v9b からの差分は4つだけ:
  1. 町にいない所有者14人のペルソナが厚くなった（名簿の中身だけ・情報）。
  2. 借りて使っている人にも毎月「今月、A市を出るか」を聞く（出る／出ない）。
     出た人は退場し、その区画は使う人がいなくなる。家主に翌月初、事実1行が届く。
  3. 借りて使っている人と家主のあいだに月1回の一言を通す（任意・世界は中身を見ない）。
  4. 毎月のプロンプトの冒頭に経過した時間を置く（第N月・開始から N-1 か月・年齢）。

A（実行できない約束を配らない）と B（X社の設定2行）は v9b のまま。
促し文・兆候・確率・閾値・当為は住民側に1つも置かない。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .field_v9 import (ACQUIRER_NAME, HOME, MAX_REASON_CHARS,  # noqa: F401
                       MAX_TEXT_CHARS, MAX_THOUGHT_CHARS, NO_ANSWER,
                       NOT_ASKED, RegistryV9, _heard_rows, _inbox_rows_v9,
                       _sale_and_rights_block, build_self_block_v9,
                       decide_schema_v9, rotate)
from .field_v9b import build_acquirer_prefix_v9b  # noqa: F401

# --- 借りて使っている人の月末の問い ------------------------------------------
LEAVE_YES = "出る"
LEAVE_NO = "出ない"
LEAVE_VALUES = [LEAVE_YES, LEAVE_NO]

MAX_LINE_CHARS = 60          # 家主／借りて使っている人の一言の上限
LEFT_MARK = "（転出）"        # 名簿に付ける印（施主 13:23）

AGE_RE = re.compile(r"(\d{1,3})\s*歳")


def leave_order(index: int, step: int) -> List[str]:
    """2肢の並びを主体×月で決定論的に入れ替える（位置効果の相殺だけ）。"""
    return rotate(LEAVE_VALUES, int(index), int(step))


# ---------------------------------------------------------------------------
# 経過した時間（差分4）
# ---------------------------------------------------------------------------

def start_age(agent: Optional[Dict[str, Any]]) -> Optional[int]:
    """名簿に書かれている開始時の年齢。書かれていなければ None。

    **ここで年齢を作らない**（凍結した名簿に無い事実を足さないため）。
    """
    if not agent:
        return None
    m = AGE_RE.search(str(agent.get("persona", "")))
    return int(m.group(1)) if m else None


def time_header(step: int, agent: Optional[Dict[str, Any]] = None) -> str:
    """毎月のプロンプトの冒頭に置く1行（事実だけ）。"""
    months = int(step) - 1
    age = start_age(agent)
    if age is None:
        return f"（第{step}月・開始から{months}か月）"
    return f"（第{step}月・開始から{months}か月・あなたは{age + months // 12}歳）"


# ---------------------------------------------------------------------------
# 名簿（転出の印だけが v9 との差）
# ---------------------------------------------------------------------------

def roster_rows_v9c(agents: List[Dict[str, Any]],
                    left_ids: Optional[Sequence[str]] = None) -> List[str]:
    """町の人の名簿。**町を出た人には「（転出）」を付ける**（施主 13:23）。"""
    left = set(str(x) for x in (left_ids or []))
    rows = []
    for a in agents:
        # 名簿に載るのは町の人だけ（v9 と同じ）。町を出た人は印を付けて残す。
        if not a.get("resident", True):
            continue
        mark = LEFT_MARK if str(a["id"]) in left else ""
        rows.append(f"  {a['name']}（{a['role_label']}）… {a.get('district', '')}{mark}")
    return rows


def build_common_prefix_v9c(cfg: Dict[str, Any], agents: List[Dict[str, Any]],
                            n_parcels: int,
                            left_ids: Optional[Sequence[str]] = None) -> str:
    """町にいる人の共通前置き。v9 の文と同じで、名簿の印と借りて使う人の節だけが違う。"""
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
{chr(10).join(roster_rows_v9c(agents, left_ids))}
  （{LEFT_MARK}と付いている者は、すでにA市を出ている）

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
  他人の不動産を借りて使っている者は、月の終わりに、
  その月にA市を出るかどうかを決める。
これが{n_steps}か月くり返される。

{_sale_and_rights_block()}
--- 借りて使うということ ---
他人の建物を借りて住み、または営んでいる者と、他人の土地を借りて使っている者がいる。
借りて使っている者と、その不動産を持っている者は、月に一度、互いに一言を送ることができる。
一言は相手にそのまま届く（送らないこともできる）。ほかの誰にも伝わらない。
借りて使っている者がA市を出た区画は、その区画を使う者がいなくなる。

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
売りに出すかどうかと、届いた条件で売るかどうかと、A市を出るかどうか。
これらの判断には、理由を一行書く欄がある（{MAX_REASON_CHARS}字以内）。
書かなくてもよく、空文字のままでもよい。
このうち、届いた条件について「売らない」と決めたときに書いた理由だけは、
その条件を出した相手に伝わる。ほかの理由は誰にも伝わらない。

--- thought（内心） ---
JSONの最初に thought を書く。thought は誰にも伝わらないあなたの内心であり、
次の場面と翌月のあなたにそのまま渡される。何を書くかはあなたが決める。
まず thought を書き、それを踏まえてその後を書く。目安は{MAX_THOUGHT_CHARS}字以内、
発言は{MAX_TEXT_CHARS}字以内。説明文を付けずJSONだけ返す。
"""


def build_absentee_prefix_v9c(cfg: Dict[str, Any], n_parcels: int) -> str:
    """町にいない所有者の前置き。v9 の文＋借りて使う人との一言の節だけ。"""
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
--- 借りて使うということ ---
あなたの不動産を借りて使っている者がいる場合、その者とあなたは、月に一度、
互いに一言を送ることができる。一言は相手にそのまま届く（送らないこともできる）。
ほかの誰にも伝わらない。
借りて使っている者がA市を出た区画は、その区画を使う者がいなくなる。

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
このうち、届いた条件について「売らない」と決めたときに書いた理由だけは、
その条件を出した相手に伝わる。ほかの理由は誰にも伝わらない。

--- thought（内心） ---
JSONの最初に thought を書く。thought は誰にも伝わらないあなたの内心であり、
翌月のあなたにそのまま渡される。何を書くかはあなたが決める。
まず thought を書き、それを踏まえてその後を書く。目安は{MAX_THOUGHT_CHARS}字以内。
説明文を付けずJSONだけ返す。
"""


# ---------------------------------------------------------------------------
# 借りて使っている人と家主
# ---------------------------------------------------------------------------

def tenant_ids(reg: RegistryV9) -> List[str]:
    """他人の不動産を借りて使っていて、自分は何も所有していない、町にいる人。"""
    out: List[str] = []
    for a in reg.agents:
        aid = str(a["id"])
        if not reg.is_resident(aid):
            continue
        if reg.parcels_owned(aid):
            continue
        if any(reg.tenant_of.get(p) == aid for p in reg.parcel_names):
            out.append(aid)
    return out


def tenant_parcels(reg: RegistryV9, aid: str) -> List[str]:
    return [p for p in reg.parcel_names if reg.tenant_of.get(p) == str(aid)]


def landlord_of(reg: RegistryV9, parcel: str) -> Optional[str]:
    """家主＝建物の所有者（建物があれば）、無ければ土地の所有者。X社なら None。"""
    p = str(parcel)
    who = reg.building_of.get(p) if reg.has_building.get(p) else None
    if who is None:
        who = reg.land_of.get(p)
    if who is None or who == ACQUIRER_NAME:
        return None
    return str(who)


def vacancy_notice(parcel: str, who: str) -> str:
    """家主に届く事実1行（所有権の通知と同じ文体・感想も指示も付けない）。"""
    return (f"（記録）先月末、{parcel}を借りて使っていた {who} がA市を出た。"
            "この区画を使う人はいない。")


def tenant_line_row(parcel: str, who: str, text: str) -> str:
    return f"  {parcel}を借りて使っている {who} から：「{text}」"


def landlord_line_row(parcel: str, who: str, text: str) -> str:
    return f"  {parcel}の持ち主 {who} から：「{text}」"


# ---------------------------------------------------------------------------
# 出力スキーマ
# ---------------------------------------------------------------------------

def tenant_schema_v9c(order: List[str], with_line: bool = True) -> Dict[str, Any]:
    props: Dict[str, Any] = {
        "thought": {"type": "string"},
        "leave": {"type": "string", "enum": list(order)},
        "leave_reason": {"type": "string"},
    }
    if with_line:
        props["to_landlord"] = {"type": "string"}
    return {"type": "object", "properties": props, "required": list(props)}


def decide_schema_v9c(listing_options: List[Tuple[str, List[str]]],
                      sell_order_: Optional[List[str]] = None,
                      reply_parcels: Optional[List[str]] = None) -> Dict[str, Any]:
    """v9 の月末の問い＋（借りて使っている人から一言が届いていれば）返事の欄。"""
    schema = decide_schema_v9(listing_options, sell_order_)
    if reply_parcels:
        schema["properties"]["to_tenant"] = {
            "type": "object",
            "properties": {p: {"type": "string"} for p in reply_parcels},
            "required": list(reply_parcels),
        }
        schema["required"] = list(schema["properties"])
    return schema


# ---------------------------------------------------------------------------
# user プロンプト
# ---------------------------------------------------------------------------

def build_tenant_prompt_v9c(agent: Dict[str, Any], reg: RegistryV9, step: int,
                            n_steps: int, thought: str,
                            parcels: List[str],
                            order: List[str],
                            heard: Optional[List[Dict[str, Any]]] = None,
                            notices: Optional[List[str]] = None,
                            lines: Optional[List[str]] = None,
                            with_line: bool = True) -> str:
    """借りて使っている人の月末の問い（出る／出ない・家主への一言）。"""
    rows = [build_self_block_v9(agent, reg, show_holdings=True), "",
            f"=== 第{step}月 / 全{n_steps}月　月の終わり ==="]
    rows += ["[今の自分の内心]", ("  " + thought) if thought else "  （まだ無い）"]
    inbox = list(notices or []) + list(lines or [])
    rows += ["[今月あなたに届いたもの]"]
    rows += ([("  " + x) if not x.startswith("  ") else x for x in inbox]
             if inbox else ["  （届いたものはない）"])
    rows += ["[今月あなたが聞いた話]"] + _heard_rows(heard or [])
    what = "・".join(parcels)
    rows += ["", "[今月末の問い]",
             "今月、A市を出るかどうかを決める。",
             f"leave に「{order[0]}」「{order[1]}」のいずれかを書く。"]
    lines_map = {
        LEAVE_YES: f"  「{LEAVE_YES}」：今月末、あなたは{what}を使うのをやめ、A市を出る。"
                   "以後この町にはいない。",
        LEAVE_NO: f"  「{LEAVE_NO}」：今月末、あなたは{what}を今までどおり使い、"
                  "A市にとどまる。",
    }
    rows += [lines_map[order[0]], lines_map[order[1]]]
    rows += [f"leave_reason には、その理由を一行で書く（{MAX_REASON_CHARS}字以内・"
             "書かない場合は空文字）。この理由は誰にも伝わらない。"]
    if with_line:
        rows += ["", "[今月の一言]",
                 f"to_landlord に、{what}を持っている相手への一言を書ける"
                 f"（{MAX_LINE_CHARS}字以内・書かない場合は空文字）。",
                 "書いた一言は翌月そのまま相手に届く。ほかの誰にも伝わらない。"]
    rows += ["", "thought に今の考えを書き、それから問いに答える。",
             "説明文を付けずJSONだけ返す。"]
    return "\n".join(rows)


def landlord_reply_block(reply_parcels: List[str]) -> List[str]:
    """家主の月末の問いに足す返事の欄（任意）。"""
    if not reply_parcels:
        return []
    return ["", "[今月の一言]",
            "to_tenant に、区画ごとに、借りて使っている相手への一言を書ける"
            f"（{MAX_LINE_CHARS}字以内・書かない場合は空文字）。",
            "  " + "・".join(reply_parcels),
            "書いた一言は翌月そのまま相手に届く。ほかの誰にも伝わらない。"]


__all__ = [
    "LEAVE_YES", "LEAVE_NO", "LEAVE_VALUES", "LEFT_MARK", "MAX_LINE_CHARS",
    "leave_order", "start_age", "time_header",
    "roster_rows_v9c", "build_common_prefix_v9c", "build_absentee_prefix_v9c",
    "tenant_ids", "tenant_parcels", "landlord_of", "vacancy_notice",
    "tenant_line_row", "landlord_line_row",
    "tenant_schema_v9c", "decide_schema_v9c",
    "build_tenant_prompt_v9c", "landlord_reply_block",
]
