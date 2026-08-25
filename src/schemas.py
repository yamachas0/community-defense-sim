"""行動の「動詞の語彙」と構造化出力スキーマ。

重要な線引き:
  ここにあるのは **世界API (どんな行為が物理的に可能か)** であって、
  「いつ・どれを選ぶか」を決めるルールではない。選択は 100% LLM。
  各動詞に発火条件・確率・閾値を持たせた瞬間にルールベース化する。禁止。
"""

from __future__ import annotations

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# 役割ごとの動詞レパートリー
# ---------------------------------------------------------------------------

VERBS: Dict[str, Dict[str, str]] = {
    "household": {
        "list_for_sale": "自分の区画を売りに出す (target=区画ID, amount=希望価格 万円)",
        "accept_offer": "届いている買付を受ける (target=買付ID)",
        "counter_offer": "届いている買付に価格を提示し返す (target=買付ID, amount=希望価格 万円)",
        "reject_offer": "届いている買付を断る (target=買付ID)",
        "unlist": "売り出しを取り下げる (target=区画ID)",
        "set_rent": "自分が貸している商店区画の賃料を改定する (target=区画ID, amount=月額 万円)",
        "move_out": "この街から転出する (区画は所有したままでも売ってからでもよい)",
        "hold": "今月は何もしない",
    },
    "business": {
        "continue": "通常営業を続ける",
        "negotiate_rent": "家主に賃料を交渉する (target=家主のagent_id, amount=希望月額 万円)",
        "close_shop": "店を畳む・撤退する",
        "relocate": "同じ街の別の空き商店を探して移る (target=区画ID)",
        "hold": "今月は判断を保留する",
    },
    "broker": {
        "circulate_listing": "売り情報を特定の相手に流す (target=相手のagent_id)",
        "approach_owner": "所有者に売却を打診する (target=所有者のagent_id)",
        "hold": "今月は動かない",
    },
    "acquirer": {
        "make_offer": "区画に買付を入れる (target=区画ID, amount=提示価格 万円)",
        "withdraw_offer": "自分が出した買付を取り下げる (target=買付ID)",
        "set_rent": "自分が所有する商店区画の賃料を改定する (target=区画ID, amount=月額 万円)",
        "redevelop": "自分が所有する区画を商業用途に建て替える (target=区画ID, amount=新しい月額賃料 万円。入居者がいれば退去になる)",
        "public_statement": "公の場で声明を出す",
        "wait": "今月は買わずに待つ",
    },
    "municipality": {
        "monitor": "様子を見る (何もしない)",
        "study_ordinance": "規制の検討に着手する / 検討状況を進める",
        "enact_ordinance": "条例・規制を発動する (target=規制名, 内容は utterance に書く)",
        "public_statement": "見解を表明する",
        "request_report": "特定の主体に説明を求める (target=agent_id)",
    },
    "media": {
        "investigate": "登記簿を調べる (次の月から街全体の所有状況が見えるようになる)",
        "publish": "記事を出す (utterance に見出しと本文)",
        "silent": "今月は書かない",
    },
}

# 妥当な action_type 一覧
def verbs_for(role: str) -> List[str]:
    return list(VERBS[role].keys())


def verb_menu(role: str) -> str:
    return "\n".join(f"  - {k}: {v}" for k, v in VERBS[role].items())


# ---------------------------------------------------------------------------
# 構造化出力スキーマ (Gemini response_schema / OpenAI json_schema 共通の内部表現)
# ---------------------------------------------------------------------------

def action_schema(role: str) -> Dict[str, Any]:
    props: Dict[str, Any] = {
        "action_type": {"type": "string", "enum": verbs_for(role)},
        "target": {"type": "string"},
        "amount": {"type": "integer"},
        "utterance": {"type": "string"},
        "utterance_channel": {"type": "string", "enum": ["public", "private", "none"]},
        "utterance_to": {"type": "string"},
        "memory": {"type": "string"},
        "reasoning": {"type": "string"},
        "evidence": {
            "type": "array",
            "items": {"type": "string"},
        },
    }
    if role == "acquirer":
        # 買い手だけが「どの名義で登記するか」を選べる。
        # 別名義の目的（用途別SPC、共同投資等）はペルソナ側の属性であり、ここでは決めない。
        props["under_name"] = {"type": "string"}
    return {
        "type": "object",
        "properties": props,
        "required": list(props.keys()),
    }


CLASSIFY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "frame": {"type": "string", "enum": ["our_town", "their_town", "neutral"]},
                    "about_acquisition": {"type": "boolean"},
                },
                "required": ["id", "frame", "about_acquisition"],
            },
        }
    },
    "required": ["results"],
}


def to_openai_json_schema(schema: Dict[str, Any], name: str) -> Dict[str, Any]:
    """内部スキーマを OpenAI Chat Completions の json_schema 形式へ変換。"""
    def strict(node: Dict[str, Any]) -> Dict[str, Any]:
        if node.get("type") == "object":
            props = {k: strict(v) for k, v in node.get("properties", {}).items()}
            return {"type": "object", "properties": props,
                    "required": list(props.keys()), "additionalProperties": False}
        if node.get("type") == "array":
            return {"type": "array", "items": strict(node["items"])}
        return node

    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": strict(schema), "strict": True},
    }
