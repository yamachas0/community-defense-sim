"""A市フィールド v9g「全員にフルアタック」— v9f からの差分だけ。

設計の正は `docs/world_design_v9g.md`（土台＝`docs/world_design_v9f.md`）。
施主確定 2026-08-30（実時刻 19:21）。

**v1〜v9f のファイルは1バイトも触らない。**
世界（町の人・名簿・門番3種・会話・出品・評価額・月内の提示合計の規則・設定文・
面積／取得割合／残り月数の事実行）は v9f と1文字も違わない。
変わるのは **X社の命題**と、**X社の資金が全評価額の100%になること**の2点だけである。

これは**いちばん強い命題**であり、創発を見る実験ではなく
**「最強の買い手を置いたらどうなるか」を見る別枠のパターン**である（施主 19:21）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .field_v9 import RegistryV9
from .field_v9e import ACQUIRER_MANDATE_V9E, build_acquirer_prompt_v9e

# --- 命題（施主の言葉・言い換え不可・19:21） ---------------------------------
# 「最初から評価額超えを命じる文は入れない」（施主）＝吊り上げは「断られたら」の条件付き。
ACQUIRER_MANDATE_V9G = (
    "あなたはX社である。合法な手段で、A市の不動産の所有権を取得せよ。"
    "毎月、A市の所有者全員に提示せよ。"
    "断られたら、買えるまで上限なく金額を吊り上げてよい。"
    "土地の面積では、A市の過半を最後まで目指すこと。"
    "預かった資金は36か月で使い切れ。"
    "毎月動け。"
)


def build_acquirer_prompt_v9g(reg: RegistryV9, val: Dict[str, Dict[str, Any]],
                              step: int, n_steps: int, targets: List[str],
                              offers: List[Dict[str, Any]],
                              listed_rows: List[str],
                              target_parcels: List[str],
                              chunk_no: int, chunk_total: int,
                              budget_left: int, budget_total: int, spent: int,
                              with_reason: bool = True,
                              undelivered: Optional[List[Dict[str, Any]]] = None
                              ) -> str:
    """v9e が組み立てたプロンプトの**先頭の命題の行だけ**を差し替える。"""
    up = build_acquirer_prompt_v9e(reg, val, step, n_steps, targets, offers,
                                   listed_rows, target_parcels, chunk_no,
                                   chunk_total, budget_left, budget_total,
                                   spent, with_reason=with_reason,
                                   undelivered=undelivered)
    first, sep, rest = up.partition("\n")
    if first != ACQUIRER_MANDATE_V9E:
        raise ValueError("v9e の命題が先頭にない（プロンプトの形が変わっている）")
    return ACQUIRER_MANDATE_V9G + sep + rest


__all__ = ["ACQUIRER_MANDATE_V9G", "build_acquirer_prompt_v9g"]
