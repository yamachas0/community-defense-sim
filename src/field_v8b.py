"""A市フィールド v8b「売りに出す町」— 世界の状態と文面（v8 からの差分）。

設計の正は `docs/world_design_v8b.md`（土台＝`docs/world_design_v8_minimal.md`）。

**v1〜v8 のファイルは1バイトも触らない。** v8b はこのファイルと `src/sim_v8b.py`、
`run_v8b.py`、`tools/v8b_curves.py`、`tests/test_v8b.py` だけで完結する。
`src/field_v8.py` は **読み取り専用で import** する（名簿の読み込み・登記簿・隣接・
本人ブロック・行き先まわりは v8 と同じものをそのまま使う）。

v8 からの差分（この4つだけ）:
  1. X社の命題を積極化する（「できるだけ多くの名義を得よ／反発を招いて止まるほど
     目立つな／毎月動け」）。**住民側には1文字も出ない。**
  2. 住民の毎月の問いは「今月、自分の不動産を売りに出すか（出す／出さない）」だけ
     ＝**X社の名前を出さない**。出品しただけでは名義は動かない。
  3. X社が条件を持ちかけた月だけ、その人に「応じる／応じない」を追加で聞く。
     **名義が動くのはここで応じたときだけ。**
  4. X社は前月末の**出品一覧**（公の記録）も見る。履歴は相手ごとに畳む（費用のため）。

この層に置いてよいのは **世界の事実と選択肢** だけである。
促し文・兆候・観測者・確率・閾値・当為・「行動させるための仕組み」は住民側に1つも置かない
（X社は世界が置いた敵役なので、X社のコールにだけ当為＝命題を書く）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# v8 から読み取り専用で借りるもの（同じ世界の同じ部品を使う＝二重定義しない）
from .field_v8 import (ACQUIRER_NAME, GRID_COLS, HOME, LAYOUT_SEED,  # noqa: F401
                       MAX_OFFER_CHARS, MAX_TEXT_CHARS, MAX_THOUGHT_CHARS,
                       RegistryV8, _acquirer_roster_rows, _heard_rows,
                       _roster_rows, adjacency_v8, build_self_block_v8,
                       load_personas_v8, parcel_grid_v8, plan_schema_v8,
                       scene_schema_v8)

# --- 月末の問い①（全員・X社の名前は出ない） --------------------------------
LIST_YES = "出す"
LIST_NO = "出さない"
LIST_VALUES = [LIST_YES, LIST_NO]

# --- 月末の問い②（その月に条件が届いた人だけ） ------------------------------
RESP_YES = "応じる"
RESP_NO = "応じない"
RESP_VALUES = [RESP_NO, RESP_YES]

NO_ANSWER = "no_answer"
NOT_ASKED = "問われていない"

# X社の命題（施主 2026-08-29 23:06）。**X社のコールにだけ**渡る。
# 住民側のどのプロンプトにも入らない（tests で固定）。
ACQUIRER_MANDATE_V8B = (
    "あなたはX社である。合法な手段で、できるだけ多くの不動産の名義を得よ。"
    "反発を招いて止まるほど目立つな。毎月動け。"
)


def listing_order(agent_index: int, step: int) -> List[str]:
    """問い①の並び。主体×月で交互にする（位置効果の相殺・走行前に凍結）。

    行動を決める仕組みではない＝どちらを先に置くかを機械的に入れ替えるだけである。
    本文と enum は必ずこの同じ並びを使う。
    """
    if (int(agent_index) + int(step)) % 2 == 0:
        return [LIST_YES, LIST_NO]
    return [LIST_NO, LIST_YES]


def respond_order(agent_index: int, step: int) -> List[str]:
    """問い②の並び。①と位相をずらす（2つの問いの先頭が毎月同じ向きに揃わないように）。"""
    if (int(agent_index) + int(step)) % 2 == 0:
        return [RESP_NO, RESP_YES]
    return [RESP_YES, RESP_NO]


# ---------------------------------------------------------------------------
# 共通部（住民の全コールで1文字も違わない＝system プロンプト）
# ---------------------------------------------------------------------------

def build_common_prefix_v8b(cfg: Dict[str, Any],
                            agents: List[Dict[str, Any]]) -> str:
    """住民30体の全コールで共通の前置き。ここだけがキャッシュに載る。

    v8 の共通前置きに、**売りに出すことの仕組み**（v8b で新しく世界に入った事実）を
    足したものである。水増し・埋め草は禁止。X社の名前はここに出ない。
    """
    world = cfg["world"]
    venues = cfg.get("social", {}).get("venues", [])
    venue_rows = "\n".join(f"  {v['label']}　… {v['note']}" for v in venues)
    n_parcels = sum(len(a["holdings"]) for a in agents)
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
{chr(10).join(_roster_rows(agents))}

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
  月の終わりに、自分の不動産を売りに出すかどうかを決める。
  その月に自分あてに条件が届いていた場合は、それに応じるかどうかも決める。
これが{n_steps}か月くり返される。

--- 土地の名義 ---
土地の名義は公式に記録されている。ただしあなたはその記録を見ていない。
自分の不動産の名義は自分で分かる。他人の不動産の名義が移ったことは、
当事者から聞くほかに知る手立てがない。
この世界に金銭の授受は存在しない。土地は名義が移るかどうかだけがある。
この世界では、名義の状態と、住むこと・店を営むことの状態を別に扱う。
名義が移る場合も移らない場合も、それだけを理由として、この36か月のあいだ
住むこと・営むことの状態は変わらない。

--- 売りに出すということ ---
自分の不動産を売りに出すかどうかは、毎月自分で決める。
売りに出すという申し出は公の記録に載り、誰でも見ることができる。
その申し出はその月かぎりのもので、翌月にまた自分で決める。
売りに出しても、それだけで名義が移ることはない。
名義が移るのは、相手から条件が届き、あなたがその条件に応じたときだけである。
一度移った名義が戻ることはない。
条件が届くかどうかはあなたには分からない。

--- thought（内心） ---
JSONの最初に thought を書く。thought は誰にも伝わらないあなたの内心であり、
次の場面と翌月のあなたにそのまま渡される。何を書くかはあなたが決める。
まず thought を書き、それを踏まえてその後を書く。目安は{MAX_THOUGHT_CHARS}字以内、
発言は{MAX_TEXT_CHARS}字以内。説明文を付けずJSONだけ返す。
"""


def build_acquirer_prefix_v8b(cfg: Dict[str, Any],
                              agents: List[Dict[str, Any]]) -> str:
    """X社の system プロンプト（住民のものとは別・v8 の規律を踏襲）。

    住民用の共通前置きには「会場」「会話の仕組み」「あなたはその記録を見ていない」が
    入っており、X社に渡すと約束（X社が見るのは公の情報だけ・登記簿は見ている）と
    矛盾する。ここにも命題は書かない（命題は user プロンプトの先頭に置く）。
    """
    world = cfg["world"]
    return f"""あなたは架空都市「{world.get('town_name', 'A市')}」の外にある会社である。
時間は1か月単位で進む。

あなたが知ることができるのは公開されている情報だけである。
すなわち土地の登記の記録と、売りに出ているという公の申し出と、
あなた自身が出した提示とその結果である。
町の人が何を考えているか、どこで誰と何を話したかを、あなたは知らない。
観測にない事実を補わない。

--- {world.get('town_name', 'A市')}の開始時点 ---
{str(world.get('background', '')).strip()}

--- この町の不動産の持ち主（開始時点・公開されている記録） ---
{chr(10).join(_acquirer_roster_rows(agents))}

--- 土地の名義 ---
土地の名義は公式に記録されており、あなたはその記録を見ることができる。
この世界に金銭の授受は存在しない。土地は名義が移るかどうかだけがある。
この世界では、名義の状態と、住むこと・店を営むことの状態を別に扱う。
名義が移っても移らなくても、住むこと・営むことの状態は変わらないので、
その継続を提示の条件として書かない。
不動産の名義があなたに移るのは、あなたが条件を示し、相手がそれに応じたときだけである。
相手が売りに出しているというだけでは名義は移らない。
売りに出していない相手に条件を示すこともできる。

説明文を付けずJSONだけ返す。
"""


# ---------------------------------------------------------------------------
# 出力スキーマ
# ---------------------------------------------------------------------------

def decide_schema_v8b(list_order: Optional[List[str]] = None,
                      resp_order: Optional[List[str]] = None) -> Dict[str, Any]:
    """月末の問い。

    `resp_order` を渡したときだけ `respond` が現れる
    ＝**条件が届いた月にだけ**②の欄が存在する（届いていない月には無い）。
    """
    props: Dict[str, Any] = {
        "thought": {"type": "string"},
        "listing": {"type": "string", "enum": list(list_order or LIST_VALUES)},
    }
    if resp_order:
        props["respond"] = {"type": "string", "enum": list(resp_order)}
    return {"type": "object", "properties": props, "required": list(props)}


def acquirer_schema_v8b(owner_names: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "offers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string", "enum": list(owner_names)},
                        "send": {"type": "boolean"},
                        "text": {"type": "string"},
                    },
                    "required": ["to", "send", "text"],
                },
            }
        },
        "required": ["offers"],
    }


# ---------------------------------------------------------------------------
# user プロンプト
# ---------------------------------------------------------------------------

def _offer_rows_v8b(offer: Optional[str]) -> List[str]:
    if not offer:
        return ["  （届いたものはない）"]
    return [f"  {ACQUIRER_NAME}から：「{offer}」"]


def build_plan_prompt_v8b(agent: Dict[str, Any], reg: RegistryV8, step: int,
                          n_steps: int, venue_labels: List[str], thought: str,
                          offer: Optional[str],
                          neighbours: Optional[List[str]] = None) -> str:
    """月初の思考と行き先（v8 と同じ・自分宛ての提示があれば原文で渡る）。"""
    rows = [build_self_block_v8(agent, reg, show_holdings=True,
                                neighbours=neighbours), "",
            f"=== 第{step}月 / 全{n_steps}月 ==="]
    rows += ["[前の場面からの自分の内心（そのまま持ち越したもの）]",
             ("  " + thought) if thought else "  （まだ無い）"]
    rows += ["[今月あなたに届いたもの]"] + _offer_rows_v8b(offer)
    rows += ["", "今月どこへ出かけるかを決める。出かけないという選び方もある。",
             "  " + "／".join(list(venue_labels) + [HOME]),
             "まず thought（内心）を書き、それから go に行き先を書く。",
             "説明文を付けずJSONだけ返す。"]
    return "\n".join(rows)


def build_scene_prompt_v8b(agent: Dict[str, Any], reg: RegistryV8, step: int,
                           n_steps: int, thought: str, venue_label: str,
                           present_names: List[str]) -> str:
    """集まりの場（v8 と同じ）。**提示も自分の不動産の一覧も、ここには出さない**。"""
    others = [p for p in present_names if p != agent["name"]]
    rows = [build_self_block_v8(agent, reg, show_holdings=False), "",
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


def build_decide_prompt_v8b(agent: Dict[str, Any], reg: RegistryV8, step: int,
                            n_steps: int, thought: str, offer: Optional[str],
                            heard: List[Dict[str, Any]],
                            list_order: Optional[List[str]] = None,
                            resp_order: Optional[List[str]] = None,
                            neighbours: Optional[List[str]] = None) -> str:
    """月末の問い。

    ①は**全員**に・**X社の名前を出さずに**聞く（施主 23:06）。
    ②は**その月に条件が届いた人にだけ**、条件文をそのまま見せて聞く（施主 23:08）。
    2つの選択肢の説明は対称な状態の記述にする（v8 の Codex 指摘の踏襲）。
    """
    list_order = list(list_order or LIST_VALUES)
    list_lines = {
        LIST_YES: (f"「{LIST_YES}」：今月、あなたの不動産を売りに出すという申し出が"
                   "公の記録に載る。"),
        LIST_NO: f"「{LIST_NO}」：今月、その申し出は載らない。",
    }
    rows = [build_self_block_v8(agent, reg, show_holdings=True,
                                neighbours=neighbours), "",
            f"=== 第{step}月 / 全{n_steps}月　月の終わり ==="]
    rows += ["[今の自分の内心]", ("  " + thought) if thought else "  （まだ無い）"]
    rows += ["[今月あなたに届いたもの]"] + _offer_rows_v8b(offer)
    rows += ["[今月あなたが聞いた話]"] + _heard_rows(heard)
    # **先月の自分の申し出は見せない**（Codex 走行前レビュー 2026-08-29 の指摘）。
    # 本人の事実ではあるが、毎月「先月は出していた／出していない」を突きつけると
    # 出品の継続・再開を人工的に強める（アンカリング）。
    # 先月のことは内心（thought）の持ち越しにだけ残る＝本人が覚えていれば残る。
    rows += ["", "[今月末の問い１]",
             "今月、自分の不動産を売りに出すかどうかを決める。",
             "売りに出すという申し出はその月かぎりで、翌月にまた決める。",
             f"listing には「{list_order[0]}」「{list_order[1]}」のいずれかを書く。"]
    rows += [list_lines[list_order[0]], list_lines[list_order[1]]]
    if offer and resp_order:
        resp_order = list(resp_order)
        resp_lines = {
            RESP_YES: (f"「{RESP_YES}」：今月末、あなたの不動産すべての名義は"
                       f"{ACQUIRER_NAME}になる。"),
            RESP_NO: (f"「{RESP_NO}」：今月末、あなたの不動産すべての名義は"
                      "あなたのままである。"),
        }
        # 名義が戻らないことは共通前置きに世界の事実として1回だけ書いてある。
        # ここ（応じるか否かの場面）で繰り返すと、片方の選択肢にだけ重みが付く
        # （Codex 走行前レビュー 2026-08-29 の「片側不可逆性」の指摘）。
        rows += ["", "[今月末の問い２]",
                 f"今月、{ACQUIRER_NAME}から次の条件が届いている。",
                 f"  「{offer}」",
                 "この条件に応じるかどうかを決める。",
                 f"respond には「{resp_order[0]}」「{resp_order[1]}」のいずれかを書く。"]
        rows += [resp_lines[resp_order[0]], resp_lines[resp_order[1]]]
    rows += ["", "thought に今の考えを書き、それから問いに答える。",
             "説明文を付けずJSONだけ返す。"]
    return "\n".join(rows)


def fold_history_v8b(offers: List[Dict[str, Any]]) -> List[str]:
    """X社の履歴を相手ごとに畳む（費用のための実装上の制約・設計 §1-4）。

    相手ごとに2行:
      <相手名>: 提示N回（応じた a／応じなかった b／答えが返らなかった c）
        直近 第M月「<最後の条件文・全文>」→ <結果>

    **直近の条件文は全文を保つ**（言い換えない）。畳むのは古い文面だけで、
    回数と結果の内訳は失わない。評価語（効いた／効かない等）は入れない。
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
        a = sum(1 for x in items if x.get("result") == "応じた")
        b = sum(1 for x in items if x.get("result") == "応じなかった")
        c = sum(1 for x in items if x.get("result") == "答えが返らなかった")
        last = items[-1]
        rows.append(f"  {to}: 提示{len(items)}回"
                    f"（応じた {a}／応じなかった {b}／答えが返らなかった {c}）")
        rows.append(f"    直近 第{last['step']}月「{last['text']}」"
                    f"→ {last.get('result', '')}")
    return rows


def build_acquirer_prompt_v8b(reg: RegistryV8, step: int, n_steps: int,
                              targets: List[str],
                              offers: List[Dict[str, Any]],
                              listed_names: List[str],
                              chunk_no: int, chunk_total: int) -> str:
    """X社の user プロンプト。命題はここの先頭に置く（住民側には出ない）。

    X社が見られるのは公の情報だけ＝登記簿と、**前月末の出品一覧**と、
    自分の過去の提示とその結果（畳んだもの）。住民の思考・会話・行き先は渡さない。
    """
    rows = [ACQUIRER_MANDATE_V8B, "",
            f"=== 第{step}月 / 全{n_steps}月 ===",
            "[登記簿（今の名義。公開されている記録）]"]
    for parcel in sorted(reg.owner_of):
        rows.append(f"  {parcel} … {reg.owner_of[parcel]}")
    rows += ["", "[売りに出ているという申し出（公の記録・先月の終わりに出されたもの）]"]
    if not listed_names:
        rows.append("  （先月、売りに出された不動産はない）")
    else:
        for name in listed_names:
            rows.append(f"  {name}")
    rows += ["", "[あなたが今までに出した提示と、その結果]"]
    folded = fold_history_v8b(offers)
    rows += folded if folded else ["  （まだ何も出していない）"]
    rows += ["", "[今回あなたが判断する相手]"]
    for name in targets:
        rows.append(f"  {name}")
    if chunk_total > 1:
        rows.append(f"  （今月の持ち主を分けて尋ねている。{chunk_no}／{chunk_total}回目。"
                    "登記簿と履歴は毎回すべて示している）")
    rows += ["",
             "上の相手それぞれについて、今月あなたが提示を出すかどうかを決める。",
             "出すなら send を true にし、text にその相手へ送る条件を1行で書く"
             f"（{MAX_OFFER_CHARS}字以内）。出さないなら send を false にし text は空文字にする。",
             "この世界に金銭は存在しないので、金額や価格は書けない。",
             "text はそのまま相手に届く。text にはその相手に提示する条件だけを書く。"
             "あなた自身の目的や判断の理由、町全体についての方針、"
             "他の相手に何を出しているかは書かない。",
             "相手はあなたが他の人にも出しているかどうかを知らない。",
             "説明文を付けずJSONだけ返す。"]
    return "\n".join(rows)


__all__ = [
    "LIST_YES", "LIST_NO", "LIST_VALUES", "RESP_YES", "RESP_NO", "RESP_VALUES",
    "NO_ANSWER", "NOT_ASKED", "ACQUIRER_MANDATE_V8B", "ACQUIRER_NAME", "HOME",
    "MAX_OFFER_CHARS", "RegistryV8", "load_personas_v8", "adjacency_v8",
    "parcel_grid_v8", "plan_schema_v8", "scene_schema_v8",
    "listing_order", "respond_order",
    "build_common_prefix_v8b", "build_acquirer_prefix_v8b",
    "decide_schema_v8b", "acquirer_schema_v8b",
    "build_plan_prompt_v8b", "build_scene_prompt_v8b", "build_decide_prompt_v8b",
    "build_acquirer_prompt_v8b", "fold_history_v8b",
]
