"""v5e の赤（自衛の具体的な行動）と自衛レベル S1〜S3。

設計の正は `docs/world_design_v5e.md` §1〜§2。

赤＝「複数の土地が同じ相手に渡っている」という事態そのものに対して、具体的なアクションが
起きた行。主体の立場（行政かどうか）は問わない＝v5c の「行政の発話に限る」制限を外す。
青・緑・黄の定義（ルール語彙・LLM 文言）は v5c から1文字も変えない。ルール関数は
`tools/run_metrics.py` の `_v5c_rule_*` を正のままにし、ここには赤とレベルだけを置く。

ここにあるのは全て純関数で、副作用は無い。走行中の停止判定と事後集計が
**同じコード**を使うための持ち場である（判定が二本に割れないように）。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# 語彙は走行前に凍結する（docs/world_design_v5e.md §2 の表）。
# 結果を見てから動かさない。
V5E_S3_WORDS = ("禁止", "差し止め", "差止", "届出", "義務付け", "義務づけ", "条例",
                "規制", "買い戻し", "買戻", "許可制", "制限区域", "勧告", "命令",
                "要綱で", "指定して")
V5E_S2_WORDS = ("回覧", "呼びかけ", "呼び掛け", "周知", "説明会", "注意喚起",
                "知らせよう", "知らせる", "広める", "記事にし", "紙面で",
                "組合として", "町内会として", "みんなに伝え", "声をかけて回")
V5E_S1_WORDS = ("売らない", "売らん", "売りません", "貸さない", "貸さん", "貸しません",
                "断ろう", "断る", "応じない", "手放さない", "手放すのはやめ",
                "売るのはやめ", "控えよう", "見送ろう", "やめておこう")
V5E_LEVELS = ("S1", "S2", "S3")

V5E_LEVEL_WORDS = {"S1": V5E_S1_WORDS, "S2": V5E_S2_WORDS, "S3": V5E_S3_WORDS}


def rule_defense_level(text: str) -> Optional[str]:
    """ルール側の自衛レベル。S3 > S2 > S1 の順で最初に当たったもの。"""
    t = str(text or "")
    for level in ("S3", "S2", "S1"):
        if any(w in t for w in V5E_LEVEL_WORDS[level]):
            return level
    return None


def rule_red_v5e(text: str) -> bool:
    """赤のルール1次抽出。役割の制限は付けない（行政に限らない）。"""
    return rule_defense_level(text) is not None


def stage_v5e(row: Dict[str, Any]) -> Optional[str]:
    """その1行が到達した色（rule かつ LLM）。どれにも当たらなければ None。

    青・緑・黄は v5c の `_v5c_stage` と完全に同じ。赤だけ v5e の定義を使う。
    """
    if not row["classified"]:
        return None
    if row["rule_red"] and row["llm_defense"]:
        return "red"
    if row["rule_yellow"] and row["llm_same_buyer"]:
        return "yellow"
    if row["rule_green"] and row["llm_area"]:
        return "green"
    if row["rule_blue"] and row["llm_deal"]:
        return "blue"
    return None


def defense_level_of(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """赤の行の自衛レベル。赤でない行には何も返さない（色の判定には影響しない）。

    主値は LLM の `defense_level`。LLM が `none`／欠損ならルール側の最上位を使い
    `level_source="rule"` と記録する。両方無ければ `level=None`（記録は残す）。
    """
    if stage_v5e(row) != "red":
        return None
    llm_level = row.get("llm_defense_level")
    if llm_level not in V5E_LEVELS:
        llm_level = None
    rule_level = rule_defense_level(row.get("text", ""))
    if llm_level:
        level, source = llm_level, "llm"
    elif rule_level:
        level, source = rule_level, "rule"
    else:
        level, source = None, None
    return {"level": level, "level_source": source,
            "llm_level": llm_level, "rule_level": rule_level}
