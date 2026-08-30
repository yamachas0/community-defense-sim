"""A市フィールド v8d — 世界の状態と文面（v8c からの差分だけ）。

設計の正は `docs/world_design_v8d.md`（土台＝`docs/world_design_v8c.md`）。

**v1〜v8c のファイルは1バイトも触らない。** v8d はこのファイルと `src/sim_v8d.py`、
`run_v8d.py`、`tests/test_v8d.py` だけで完結する。
`src/field_v8c.py` / `src/field_v8b.py` は **読み取り専用で import** する。

v8c からの差分（この3つだけ・施主決定 2026-08-30 01:34）:
  1. 月末の問い２から **「売ると今月末に名義がX社へ移り、その後は戻らない。
     売らないと今月末の名義はあなたのままである。」の一文を削除**する。
     選択肢の説明（対称な状態の記述）と、共通前置きの
     「一度移った名義が戻ることはない。」は**残す**。
  2. X社の前置きに **事実を3つ**加える（情報であって当為ではない）。
     ＝A市に不動産を持っていない／金銭の額は扱わない／約束できるのは自分が実行できることだけ。
  3. 理由の一言は **出す／出さない・売る／売らない・X社の提示判断** の3つだけに付ける
     ＝**行き先の理由欄を無くす**。

この層に置いてよいのは **世界の事実と選択肢** だけである。
促し文・兆候・観測者・確率・閾値・当為・「行動させるための仕組み」は住民側に1つも置かない
（X社は世界が置いた敵役なので、X社のコールにだけ当為＝命題を書く）。

実装の作法：**文面は v8c のビルダを呼んでから、決められた1行を機械的に落とす／足す**。
こうすると「他は1文字も変わらない」ことが試験で証明できる（差分の証明可能性）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .field_v8b import HOME, MAX_OFFER_CHARS, MAX_TEXT_CHARS  # noqa: F401
from .field_v8b import (MAX_THOUGHT_CHARS, NO_ANSWER, NOT_ASKED,  # noqa: F401
                        RegistryV8, adjacency_v8, build_scene_prompt_v8b,
                        load_personas_v8, parcel_grid_v8, scene_schema_v8)
from .field_v8c import (ACQUIRER_INTRO_V8C, ACQUIRER_MANDATE_V8C,  # noqa: F401
                        ACQUIRER_NAME, LIST_NO, LIST_VALUES, LIST_YES,
                        MAX_REASON_CHARS, SELL_NO, SELL_VALUES, SELL_YES,
                        acquirer_schema_v8c, build_acquirer_prefix_v8c,
                        build_acquirer_prompt_v8c, build_common_prefix_v8c,
                        build_decide_prompt_v8c, build_plan_prompt_v8c,
                        decide_schema_v8c, delivered_offer_v8c,
                        fold_history_v8c, listing_order, plan_schema_v8c,
                        sell_order)

# --- 差分1: 問い２から落とす一文（v8c の原文・ここに1回だけ書く） -------------
IRREVERSIBLE_LINE_V8C = (
    f"{SELL_YES}と今月末に名義が{ACQUIRER_NAME}へ移り、その後は戻らない。"
    f"{SELL_NO}と今月末の名義はあなたのままである。"
)

# --- 差分2: X社の前置きに足す事実（施主 2026-08-30 01:34） -------------------
# 情報であって当為ではない（「〜せよ」「〜すると良い」を足さない）。
#
# 施主の原文からの変更は2つで、どちらも **Codex 走行前レビュー（2026-08-30 02:0x）の
# 必須指摘**（＝嘘を渡さない・実行できない例を渡さない）に従ったものである:
#   (a) 「あなたは A市に不動産を持っていない。」→ 頭に「第1月の開始時点で、」を付けた。
#       前置きは36か月固定なので、X社が取得したあとも同じ文が毎月渡る。一方で毎月の
#       user プロンプトには本物の登記簿が入る＝同じコールの中で事実が矛盾する。
#   (b) 括弧の中の例3つ（売った後も住み続けてよい期間／引き渡しの時期は相手の都合／
#       建物や土地の管理を引き受ける）→ 世界の能力の境界を書いた1行に置き換えた。
#       理由＝3つとも**この世界が持っていない仕組み**である：住むこと・営むことの継続は
#       名義に関わらず世界が保証済み（X社の前置きには「その継続を提示の条件として書かない」
#       という行が v8c から入っている＝そのままだと自己矛盾）、名義が移るのは月末で固定
#       （引き渡しの時期は選べない）、管理という状態は世界に存在しない。
#       実行できない約束の例を与えると、v8c の「他人の物件との交換」と同じ失敗
#       （＝自分にできないことを条件に書く）をこちらから教えることになる。
ACQUIRER_FACTS_V8D = (
    "--- あなたについて ---\n"
    "第1月の開始時点で、あなたは A市に不動産を持っていない。\n"
    "金銭の額はこの世界では扱わない。\n"
    "あなたが約束できるのは、自分自身が実行できることだけである\n"
    "（この世界には、あなたから他者へ不動産の名義を移す仕組みはない）。\n"
)

# --- 差分3: 行き先の理由欄を落とす（v8c の指示文の書き出し） -----------------
_PLAN_REASON_HEAD = "reason には、"

# 共通前置きの「理由の一言」節（v8c → v8d の置換。行き先を外すぶんだけ）
_REASON_BLOCK_V8C = (
    "--- 理由の一言 ---\n"
    "どこへ行くか、売りに出すかどうか、届いた条件で売るかどうか。\n"
    f"それぞれの判断には、理由を一行書く欄がある（{MAX_REASON_CHARS}字以内）。\n"
)
_REASON_BLOCK_V8D = (
    "--- 理由の一言 ---\n"
    "売りに出すかどうかと、届いた条件で売るかどうか。\n"
    f"この2つの判断には、理由を一行書く欄がある（{MAX_REASON_CHARS}字以内）。\n"
)


# ---------------------------------------------------------------------------
# 共通部（住民の全コールで1文字も違わない＝system プロンプト）
# ---------------------------------------------------------------------------

def build_common_prefix_v8d(cfg: Dict[str, Any],
                            agents: List[Dict[str, Any]]) -> str:
    """住民30体の全コールで共通の前置き。

    v8c の前置きの **「理由の一言」の最初の2行だけ** を差し替える
    （行き先の判断がその一覧から消える）。ほかは1文字も変えない。
    """
    text = build_common_prefix_v8c(cfg, agents)
    if _REASON_BLOCK_V8C not in text:
        raise RuntimeError("v8c の共通前置きの『理由の一言』節が見つからない")
    return text.replace(_REASON_BLOCK_V8C, _REASON_BLOCK_V8D, 1)


def build_acquirer_prefix_v8d(cfg: Dict[str, Any],
                              agents: List[Dict[str, Any]]) -> str:
    """X社の system プロンプト。v8c の前置きに**事実の節を1つ足すだけ**。

    足す位置は最後の「説明文を付けずJSONだけ返す。」の直前。
    """
    text = build_acquirer_prefix_v8c(cfg, agents)
    tail = "説明文を付けずJSONだけ返す。\n"
    if not text.endswith(tail):
        raise RuntimeError("v8c のX社前置きの末尾が想定と違う")
    head = text[: -len(tail)]
    return head + ACQUIRER_FACTS_V8D + "\n" + tail


# ---------------------------------------------------------------------------
# 出力スキーマ
# ---------------------------------------------------------------------------

def plan_schema_v8d(venue_labels: List[str]) -> Dict[str, Any]:
    """行き先の判断（**理由欄なし**＝差分3）。"""
    schema = plan_schema_v8c(venue_labels)
    props = {k: v for k, v in schema["properties"].items() if k != "reason"}
    return {"type": "object", "properties": props, "required": list(props)}


# 月末の問いとX社のスキーマは v8c と同一（別名で公開するだけ）
decide_schema_v8d = decide_schema_v8c
acquirer_schema_v8d = acquirer_schema_v8c


# ---------------------------------------------------------------------------
# user プロンプト
# ---------------------------------------------------------------------------

def build_plan_prompt_v8d(agent: Dict[str, Any], reg: RegistryV8, step: int,
                          n_steps: int, venue_labels: List[str], thought: str,
                          offer: Optional[str],
                          neighbours: Optional[List[str]] = None) -> str:
    """月初の思考と行き先（v8c から**理由の指示行だけ**を落とす）。"""
    text = build_plan_prompt_v8c(agent, reg, step, n_steps, venue_labels,
                                 thought, offer, neighbours=neighbours)
    lines = text.split("\n")
    kept = [ln for ln in lines if not ln.startswith(_PLAN_REASON_HEAD)]
    if len(lines) - len(kept) != 1:
        raise RuntimeError("行き先の理由の指示行がちょうど1行ではない")
    return "\n".join(kept)


def build_decide_prompt_v8d(agent: Dict[str, Any], reg: RegistryV8, step: int,
                            n_steps: int, thought: str, offer: Optional[str],
                            heard: List[Dict[str, Any]],
                            list_order: Optional[List[str]] = None,
                            sell_order_: Optional[List[str]] = None,
                            neighbours: Optional[List[str]] = None) -> str:
    """月末の問い（v8c から**「戻らない」の一文だけ**を落とす＝差分1）。"""
    text = build_decide_prompt_v8c(agent, reg, step, n_steps, thought, offer,
                                   heard, list_order=list_order,
                                   sell_order_=sell_order_,
                                   neighbours=neighbours)
    lines = text.split("\n")
    kept = [ln for ln in lines if ln != IRREVERSIBLE_LINE_V8C]
    dropped = len(lines) - len(kept)
    # 条件が届いていない月にはこの行がそもそも無い（0行）。届いた月はちょうど1行。
    if dropped > 1:
        raise RuntimeError("『戻らない』の一文が2行以上ある")
    if (offer and sell_order_) and dropped != 1:
        raise RuntimeError("『戻らない』の一文が落ちていない")
    return "\n".join(kept)


# X社の user プロンプトは v8c と同一（命題も自己紹介の扱いも変えない）
build_acquirer_prompt_v8d = build_acquirer_prompt_v8c


__all__ = [
    "LIST_YES", "LIST_NO", "LIST_VALUES", "SELL_YES", "SELL_NO", "SELL_VALUES",
    "NO_ANSWER", "NOT_ASKED", "ACQUIRER_MANDATE_V8C", "ACQUIRER_NAME", "HOME",
    "MAX_OFFER_CHARS", "MAX_REASON_CHARS", "ACQUIRER_INTRO_V8C",
    "ACQUIRER_FACTS_V8D", "IRREVERSIBLE_LINE_V8C",
    "delivered_offer_v8c", "RegistryV8", "load_personas_v8",
    "adjacency_v8", "parcel_grid_v8", "scene_schema_v8", "build_scene_prompt_v8b",
    "listing_order", "sell_order",
    "build_common_prefix_v8d", "build_acquirer_prefix_v8d",
    "plan_schema_v8d", "decide_schema_v8d", "acquirer_schema_v8d",
    "build_plan_prompt_v8d", "build_decide_prompt_v8d",
    "build_acquirer_prompt_v8d", "fold_history_v8c",
]
