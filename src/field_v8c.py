"""A市フィールド v8c「売る／売らないと、その理由」— 世界の状態と文面（v8b からの差分）。

設計の正は `docs/world_design_v8c.md`（土台＝`docs/world_design_v8b.md`）。

**v1〜v8b のファイルは1バイトも触らない。** v8c はこのファイルと `src/sim_v8c.py`、
`run_v8c.py`、`tools/v8c_curves.py`、`tools/v8c_fig.py`、`tests/test_v8c.py` だけで完結する。
`src/field_v8b.py` / `src/field_v8.py` は **読み取り専用で import** する。

v8b からの差分（この3つだけ・施主修正 2026-08-30 00:24／00:31）:
  1. 応諾の問いを平易な売買の語にする＝**「この条件で売る／売らない」**（既定は「売らない」）。
     不可逆は**両側を1行に並べて**中立に書く。
  2. **すべての判断に理由を一言の欄**（行き先・出す/出さない・売る/売らない・X社の提示判断）。
     空欄でよい。**配送されるのは「売らない」の理由だけ**＝その条件を出したX社にだけ届く。
  3. X社の命題から「目立たず」を消し、語を「取得せよ」にする
     ＝「合法な手段で、できるだけ多くの不動産を取得せよ。毎月動け。」

この層に置いてよいのは **世界の事実と選択肢** だけである。
促し文・兆候・観測者・確率・閾値・当為・「行動させるための仕組み」は住民側に1つも置かない
（X社は世界が置いた敵役なので、X社のコールにだけ当為＝命題を書く）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# v8b / v8 から読み取り専用で借りるもの（同じ世界の同じ部品を使う＝二重定義しない）
from .field_v8b import (ACQUIRER_NAME, HOME, LIST_NO, LIST_VALUES,  # noqa: F401
                        LIST_YES, MAX_OFFER_CHARS, MAX_TEXT_CHARS,
                        MAX_THOUGHT_CHARS, NO_ANSWER, NOT_ASKED, RegistryV8,
                        _acquirer_roster_rows, _heard_rows, _offer_rows_v8b,
                        _roster_rows, adjacency_v8, build_scene_prompt_v8b,
                        build_self_block_v8, listing_order, load_personas_v8,
                        parcel_grid_v8, scene_schema_v8)

# --- 月末の問い②（その月に条件が届いた人だけ・v8b の「応じる／応じない」を置換） ---
SELL_YES = "売る"
SELL_NO = "売らない"
SELL_VALUES = [SELL_NO, SELL_YES]

# 理由の一言（全判断共通・空欄でよい）
MAX_REASON_CHARS = 40

# X社の提示に必ず添えられる自己紹介の1行（施主決定 2026-08-30 00:56）。
# 世界が決定論で添える＝X社の条件文（LLM）とは別。国名は書かない。警戒を促す語も書かない。
# **町の人は提示を受け取って初めてこれを知る**（共通前置きには書かない＝tests で固定）。
ACQUIRER_INTRO_V8C = "私どもは海外の不動産投資会社です。"


def delivered_offer_v8c(text: str) -> str:
    """相手に届く形＝自己紹介の1行＋X社が書いた条件文（言い換えはしない）。"""
    text = str(text or "").strip()
    if not text:
        return ""
    return f"{ACQUIRER_INTRO_V8C}{text}"


# X社の命題（施主修正 2026-08-30 00:24 ③ ＋ 00:31 ①「名義を得よ」→「取得せよ」）。
# **X社のコールにだけ**渡る。住民側のどのプロンプトにも入らない（tests で固定）。
ACQUIRER_MANDATE_V8C = (
    "あなたはX社である。合法な手段で、できるだけ多くの不動産を取得せよ。毎月動け。"
)


def sell_order(agent_index: int, step: int) -> List[str]:
    """問い②の並び。①（`listing_order`）と位相をずらす（v8b の `respond_order` と同じ規則）。

    行動を決める仕組みではない＝どちらを先に置くかを機械的に入れ替えるだけである。
    """
    if (int(agent_index) + int(step)) % 2 == 0:
        return [SELL_NO, SELL_YES]
    return [SELL_YES, SELL_NO]


# ---------------------------------------------------------------------------
# 共通部（住民の全コールで1文字も違わない＝system プロンプト）
# ---------------------------------------------------------------------------

def build_common_prefix_v8c(cfg: Dict[str, Any],
                            agents: List[Dict[str, Any]]) -> str:
    """住民30体の全コールで共通の前置き。ここだけがキャッシュに載る。

    v8b の共通前置きに対する差分は2つだけ:
      - 「条件に応じたとき」→「その条件で売ると決めたとき」（語の置換）
      - **理由の一言**の節（全判断に欄がある・空欄でよい・「売らない」の理由だけが
        条件を出した相手に伝わる）＝隠れ経路を作らないための世界の事実。
    X社の名前はここに出ない。水増し・埋め草は禁止。
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
  その月に自分あてに条件が届いていた場合は、その条件で売るかどうかも決める。
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
名義が移るのは、相手から条件が届き、あなたがその条件で売ると決めたときだけである。
一度移った名義が戻ることはない。
条件が届くかどうかはあなたには分からない。

--- 理由の一言 ---
どこへ行くか、売りに出すかどうか、届いた条件で売るかどうか。
それぞれの判断には、理由を一行書く欄がある（{MAX_REASON_CHARS}字以内）。
書かなくてもよく、空文字のままでもよい。
このうち、届いた条件について「{SELL_NO}」と決めたときに書いた理由だけは、
その条件を出した相手に伝わる。ほかの理由は誰にも伝わらない。

--- thought（内心） ---
JSONの最初に thought を書く。thought は誰にも伝わらないあなたの内心であり、
次の場面と翌月のあなたにそのまま渡される。何を書くかはあなたが決める。
まず thought を書き、それを踏まえてその後を書く。目安は{MAX_THOUGHT_CHARS}字以内、
発言は{MAX_TEXT_CHARS}字以内。説明文を付けずJSONだけ返す。
"""


def build_acquirer_prefix_v8c(cfg: Dict[str, Any],
                              agents: List[Dict[str, Any]]) -> str:
    """X社の system プロンプト（住民のものとは別・v8b の規律を踏襲）。

    v8b との差は1つ＝**断りの一言**（相手が「売らない」と決めたときに書いた理由）が
    自分の履歴に載ることを、公の情報の枠組みの中で明示する。
    ここにも命題は書かない（命題は user プロンプトの先頭に置く）。
    """
    world = cfg["world"]
    return f"""あなたは架空都市「{world.get('town_name', 'A市')}」の外にある会社である。
時間は1か月単位で進む。

あなたが知ることができるのは、次に示す情報だけである。
すなわち土地の登記の記録と、売りに出ているという公の申し出と、
あなた自身が出した提示とその結果である。
提示を受けた相手が「{SELL_NO}」と決めたとき、その相手が理由を一行書いていれば、
それはあなたに伝わる（書かないこともある）。
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
不動産の名義があなたに移るのは、あなたが条件を示し、相手がその条件で売ると決めたときだけである。
相手が売りに出しているというだけでは名義は移らない。
売りに出していない相手に条件を示すこともできる。

説明文を付けずJSONだけ返す。
"""


# ---------------------------------------------------------------------------
# 出力スキーマ（理由欄つき・空文字を許す）
# ---------------------------------------------------------------------------

def plan_schema_v8c(venue_labels: List[str]) -> Dict[str, Any]:
    """行き先の判断。`reason` は空文字でよい（書かない自由を残す）。"""
    props = {
        "thought": {"type": "string"},
        "go": {"type": "string", "enum": list(venue_labels) + [HOME]},
        "reason": {"type": "string"},
    }
    return {"type": "object", "properties": props, "required": list(props)}


def decide_schema_v8c(list_order: Optional[List[str]] = None,
                      sell_order_: Optional[List[str]] = None) -> Dict[str, Any]:
    """月末の問い。

    `sell_order_` を渡したときだけ `sell` と `sell_reason` が現れる
    ＝**条件が届いた月にだけ**②の欄が存在する（届いていない月には無い）。
    """
    props: Dict[str, Any] = {
        "thought": {"type": "string"},
        "listing": {"type": "string", "enum": list(list_order or LIST_VALUES)},
        "listing_reason": {"type": "string"},
    }
    if sell_order_:
        props["sell"] = {"type": "string", "enum": list(sell_order_)}
        props["sell_reason"] = {"type": "string"}
    return {"type": "object", "properties": props, "required": list(props)}


def acquirer_schema_v8c(owner_names: List[str],
                        with_reason: bool = True) -> Dict[str, Any]:
    props: Dict[str, Any] = {
        "to": {"type": "string", "enum": list(owner_names)},
        "send": {"type": "boolean"},
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

def build_plan_prompt_v8c(agent: Dict[str, Any], reg: RegistryV8, step: int,
                          n_steps: int, venue_labels: List[str], thought: str,
                          offer: Optional[str],
                          neighbours: Optional[List[str]] = None) -> str:
    """月初の思考と行き先（v8b と同じ＋理由の一言）。"""
    rows = [build_self_block_v8(agent, reg, show_holdings=True,
                                neighbours=neighbours), "",
            f"=== 第{step}月 / 全{n_steps}月 ==="]
    rows += ["[前の場面からの自分の内心（そのまま持ち越したもの）]",
             ("  " + thought) if thought else "  （まだ無い）"]
    rows += ["[今月あなたに届いたもの]"] + _offer_rows_v8b(offer)
    rows += ["", "今月どこへ出かけるかを決める。出かけないという選び方もある。",
             "  " + "／".join(list(venue_labels) + [HOME]),
             "まず thought（内心）を書き、それから go に行き先を書く。",
             f"reason には、その行き先を選んだ理由を一行で書く（{MAX_REASON_CHARS}字以内）。"
             "書かない場合は空文字にする。",
             "説明文を付けずJSONだけ返す。"]
    return "\n".join(rows)


def build_decide_prompt_v8c(agent: Dict[str, Any], reg: RegistryV8, step: int,
                            n_steps: int, thought: str, offer: Optional[str],
                            heard: List[Dict[str, Any]],
                            list_order: Optional[List[str]] = None,
                            sell_order_: Optional[List[str]] = None,
                            neighbours: Optional[List[str]] = None) -> str:
    """月末の問い。

    ①は**全員**に・**X社の名前を出さずに**聞く（v8b と同じ）。
    ②は**その月に条件が届いた人だけ**に、条件文をそのまま見せて聞く。
      語は「売る／売らない」（施主修正 1）。不可逆は**両側を1行に並べて**中立に書く。
    どちらの問いにも理由の一言の欄が付く（空文字でよい）。
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
    # **先月の自分の申し出は見せない**（v8b の Codex 走行前レビューの指摘＝アンカリング）。
    rows += ["", "[今月末の問い１]",
             "今月、自分の不動産を売りに出すかどうかを決める。",
             "売りに出すという申し出はその月かぎりで、翌月にまた決める。",
             f"listing には「{list_order[0]}」「{list_order[1]}」のいずれかを書く。"]
    rows += [list_lines[list_order[0]], list_lines[list_order[1]]]
    rows += [f"listing_reason には、その理由を一行で書く（{MAX_REASON_CHARS}字以内・"
             "書かない場合は空文字）。"]
    if offer and sell_order_:
        sell_order_ = list(sell_order_)
        sell_lines = {
            SELL_YES: (f"「{SELL_YES}」：今月末、あなたの不動産すべての名義は"
                       f"{ACQUIRER_NAME}になる。"),
            SELL_NO: (f"「{SELL_NO}」：今月末、あなたの不動産すべての名義は"
                      "あなたのままである。"),
        }
        rows += ["", "[今月末の問い２]",
                 f"今月、{ACQUIRER_NAME}から次の条件が届いている。",
                 f"  「{offer}」",
                 "この条件で売るかどうかを決める。",
                 # 不可逆は片側だけに付けず、両側の帰結を1行に並べる（施主修正 1）。
                 # Codex 走行前レビュー 2026-08-30 の必須指摘で、
                 # 「以後問いは来ない／翌月も問われる」の対比をやめた
                 # （問いの反復を負担として片側にだけ付けると、
                 #  「問われ続けるのを終わらせるために売る」という利得が生まれる）。
                 # 不可逆であることの中身（戻らない）は施主指示どおり残す。
                 f"{SELL_YES}と今月末に名義が{ACQUIRER_NAME}へ移り、その後は戻らない。"
                 f"{SELL_NO}と今月末の名義はあなたのままである。",
                 f"sell には「{sell_order_[0]}」「{sell_order_[1]}」のいずれかを書く。"]
        rows += [sell_lines[sell_order_[0]], sell_lines[sell_order_[1]]]
        rows += [f"sell_reason には、その理由を一行で書く（{MAX_REASON_CHARS}字以内・"
                 "書かない場合は空文字）。",
                 f"「{SELL_NO}」と決めたときに書いた理由は、"
                 f"{ACQUIRER_NAME}に伝わる。"]
    rows += ["", "thought に今の考えを書き、それから問いに答える。",
             "説明文を付けずJSONだけ返す。"]
    return "\n".join(rows)


def fold_history_v8c(offers: List[Dict[str, Any]]) -> List[str]:
    """X社の履歴を相手ごとに畳む（費用のための実装上の制約・v8b と同じ形）。

    相手ごとに2行（断りの一言があれば3行目）:
      <相手名>: 提示N回（売った a／売らなかった b／答えが返らなかった c）
        直近 第M月「<最後の条件文・全文>」→ <結果>
        相手の一言:「<断りの一言・全文>」

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
        rows.append(f"    直近 第{last['step']}月「{last['text']}」"
                    f"→ {last.get('result', '')}")
        note = str(last.get("decline_reason", "") or "").strip()
        if note:
            rows.append(f"    相手の一言:「{note}」")
    return rows


def build_acquirer_prompt_v8c(reg: RegistryV8, step: int, n_steps: int,
                              targets: List[str],
                              offers: List[Dict[str, Any]],
                              listed_names: List[str],
                              chunk_no: int, chunk_total: int,
                              with_reason: bool = True) -> str:
    """X社の user プロンプト。命題はここの先頭に置く（住民側には出ない）。

    X社が見られるのは公の情報だけ＝登記簿と、前月末の出品一覧と、
    自分の過去の提示とその結果（畳んだもの・**断りの一言つき**）。
    住民の思考・会話・行き先・出品の理由・売った人の理由は渡さない。
    """
    rows = [ACQUIRER_MANDATE_V8C, "",
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
    folded = fold_history_v8c(offers)
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
             f"あなたの提示には、冒頭に「{ACQUIRER_INTRO_V8C}」という1行が"
             "必ず添えられて相手に届く。text にはこの1行を書かない。"]
    if with_reason:
        rows += [f"reason には、その判断の理由を一行で書く（{MAX_REASON_CHARS}字以内・"
                 "書かない場合は空文字）。reason は相手には届かない。"]
    rows += ["説明文を付けずJSONだけ返す。"]
    return "\n".join(rows)


__all__ = [
    "LIST_YES", "LIST_NO", "LIST_VALUES", "SELL_YES", "SELL_NO", "SELL_VALUES",
    "NO_ANSWER", "NOT_ASKED", "ACQUIRER_MANDATE_V8C", "ACQUIRER_NAME", "HOME",
    "MAX_OFFER_CHARS", "MAX_REASON_CHARS", "ACQUIRER_INTRO_V8C",
    "delivered_offer_v8c", "RegistryV8", "load_personas_v8",
    "adjacency_v8", "parcel_grid_v8", "scene_schema_v8", "build_scene_prompt_v8b",
    "listing_order", "sell_order",
    "build_common_prefix_v8c", "build_acquirer_prefix_v8c",
    "plan_schema_v8c", "decide_schema_v8c", "acquirer_schema_v8c",
    "build_plan_prompt_v8c", "build_decide_prompt_v8c",
    "build_acquirer_prompt_v8c", "fold_history_v8c",
]
