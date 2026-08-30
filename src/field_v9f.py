"""A市フィールド v9f「積んででも買えと言われた買い手」— v9e からの差分だけ。

設計の正は `docs/world_design_v9f.md`（土台＝`docs/world_design_v9e.md`）。
施主確定 2026-08-30（実時刻 17:02）。

**v1〜v9e のファイルは1バイトも触らない。**
世界（町の人・名簿・門番3種・会話・出品・評価額・資金51%・月内の提示合計の規則・
X社の設定文・面積・取得割合・残り月数の事実行）は v9e と1文字も違わない。
**変わるのは X社 に渡す命題の1行だけ**である。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .field_v9 import RegistryV9
from .field_v9e import (ACQUIRER_MANDATE_V9E,  # noqa: F401
                        build_acquirer_prompt_v9e, progress_rows_v9e)

# --- 命題（施主確定 2026-08-30 17:05・言い換え不可） --------------------------
# 17:02 の初案は算数の矛盾（資金51%＝平均1.0200倍を超えると評価額の過半は原理的に不可能）
# のため施主が差し替えた。**「過半」を消し、資金51%はそのまま**。
# 意図＝「買えるかどうかは相手次第だが、買えるまで条件を変えて頑張れ」。
# 17:09 追記＝**「土地の面積では過半を最後まで目指す」を入れる**（実効支配に要る／
# 面積の過半は最安 3億3,600万円＝資金の28.7%で理論上可能＝資金51%と矛盾しない。
# 評価額の過半は書かない＝算数上不可能なので与えない）。
ACQUIRER_MANDATE_V9F = (
    "あなたはX社である。合法な手段で、A市の不動産の所有権を取得せよ。"
    "資金の範囲で、できるだけ多くの評価額と面積を取得すること。"
    "土地の面積では、A市の過半を最後まで目指すこと。"
    "買えるかどうかは相手次第だが、買えるまで、金額を含む条件を変えて働きかけ続けること。"
    "預かった資金は36か月で使い切ることが前提であり、余らせるより、高く買うことを選べ。"
    "毎月動け。"
)


def build_acquirer_prompt_v9f(reg: RegistryV9, val: Dict[str, Dict[str, Any]],
                              step: int, n_steps: int, targets: List[str],
                              offers: List[Dict[str, Any]],
                              listed_rows: List[str],
                              target_parcels: List[str],
                              chunk_no: int, chunk_total: int,
                              budget_left: int, budget_total: int, spent: int,
                              with_reason: bool = True,
                              undelivered: Optional[List[Dict[str, Any]]] = None
                              ) -> str:
    """v9e の user プロンプトの**先頭の命題の行だけ**を差し替える。

    ほかの行（資金・取得割合・残り月数・登記簿・履歴・書き方の決まり）は
    v9e が組み立てたものをそのまま使う＝**配線を複製しない**。
    """
    up = build_acquirer_prompt_v9e(reg, val, step, n_steps, targets, offers,
                                   listed_rows, target_parcels, chunk_no,
                                   chunk_total, budget_left, budget_total,
                                   spent, with_reason=with_reason,
                                   undelivered=undelivered)
    first, sep, rest = up.partition("\n")
    if first != ACQUIRER_MANDATE_V9E:
        raise ValueError("v9e の命題が先頭にない（プロンプトの形が変わっている）")
    return ACQUIRER_MANDATE_V9F + sep + rest


__all__ = ["ACQUIRER_MANDATE_V9F", "build_acquirer_prompt_v9f"]
