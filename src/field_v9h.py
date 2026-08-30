"""A市フィールド v9h「目標を町に明言する買い手」— v9f からの差分だけ。

設計の正は `docs/world_design_v9h.md`（土台＝`docs/world_design_v9f.md`）。
施主確定 2026-08-30（実時刻 20:12）。v9g は不採用（ボツ）。

**v1〜v9g のファイルは1バイトも触らない。**
世界（町の人のプロンプト・名簿・門番3種・会話・出品・評価額・資金51%・
月内の提示合計の規則・面積／取得割合／残り月数の事実行・v9f の命題）は
v9f と1文字も違わない。

変わるのは2つだけ:
  1. **町の人に届く手紙の冒頭に、世界が必ず1行を添える**＝
     「私どもは、この街の不動産の過半の取得を目指しています。」
     （X社の自由文ではない。**毎通・全員に同じ文**を世界が付ける。LLM の裁量はない。）
  2. X社の設定文に、整合のための事実を1行足す（自分がそれを公言していること）。
"""

from __future__ import annotations

from typing import Any, Dict

from .field_v9 import ACQUIRER_INTRO_V8C, RegistryV9
from .field_v9d import build_acquirer_prefix_v9d

# 世界が必ず添える1行（施主の言葉・言い換え不可）
DECLARATION_V9H = "私どもは、この街の不動産の過半の取得を目指しています。"

# X社の設定文に足す事実（整合のため・行動の指示ではない）
ACQUIRER_DECLARATION_FACT_V9H = (
    f"あなたが出すすべての提示には、冒頭に「{DECLARATION_V9H}」という1行が"
    "必ず添えられて相手に届く。あなたはこの目標を町に公言している。\n"
)


def delivered_offer_v9h(text: str) -> str:
    """相手に届く条件文＝自己紹介の1行＋**明言の1行**＋X社が書いた条件文。

    X社の文は言い換えない（v9 以来の規律）。世界が足すのは決まった1行だけである。
    """
    text = str(text or "").strip()
    if not text:
        return ""
    return f"{ACQUIRER_INTRO_V8C}{DECLARATION_V9H}{text}"


def build_acquirer_prefix_v9h(cfg: Dict[str, Any], reg: RegistryV9,
                              budget: int) -> str:
    """v9d の前置き＋「公言している」の1行だけ。"""
    pre = build_acquirer_prefix_v9d(cfg, reg, budget)
    tail = "説明文を付けずJSONだけ返す。"
    if tail not in pre:
        raise ValueError("X社の前置きの形が変わっている")
    return pre.replace(tail, ACQUIRER_DECLARATION_FACT_V9H + tail)


__all__ = ["DECLARATION_V9H", "ACQUIRER_DECLARATION_FACT_V9H",
           "delivered_offer_v9h", "build_acquirer_prefix_v9h"]
