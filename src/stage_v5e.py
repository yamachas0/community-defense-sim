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
# **凍結 2026-08-29 00:20（CTO）→ 2026-08-29 03:5x に S2/S3 のみ締め直した（施主指示）。**
# 締め直しの理由：第1便の3本を走らせた結果、S2/S3 に偽陽性が出た。**定義は変えていない**が、
# 実装（語彙とLLM指示）が定義に届いていなかった＝「行為の完了/告知」を要求していなかった。
# S2 は依頼・願望・対象未定の一般意向を拾い、S3 は行政以外の計画の話を拾っていた。
# **重要（正直に記録する）：この締め直しは第1便の結果を見た後に行った。**
# したがって締め直し後の S 集計は「事後の再判定」であり、第1便の3本が実際に停止したのは
# 締める前の（緩い）分類器の出力による。RESULTS には両方を並記する。S1 は変更していない。
# 最後の追加は Codex の設計レビュー（2026-08-28）が挙げた自然な言い換えの
# 取りこぼし（売却しない／譲渡しない／売却は見送／取引を凍結）で、
# 実 API を一度も走らせる前に確定させたものである。
V5E_S3_WORDS = ("売買を禁止", "取引を禁止", "禁止する", "禁止した", "禁止します",
                "差し止め", "差止", "届出を義務", "届出制", "義務付け", "義務づけ",
                "条例を制定", "条例で", "条例化", "買い戻", "許可制",
                "制限区域に指定", "規制を導入", "規制する", "規制をかけ",
                "勧告する", "勧告した", "命令する", "命令した",
                "取引を凍結", "取引の凍結")
V5E_S2_WORDS = ("回覧に載せ", "回覧で回し", "回覧を回し", "回覧板に載せ",
                "回覧板で回し", "説明会を開", "説明会をやり", "説明会をやる", "説明会をやろ", "説明会を実施",
                "説明会を行", "記事にします", "記事にする", "記事を出す",
                "記事に書き", "紙面で伝え", "紙面に載せ", "注意喚起し",
                "注意喚起する", "組合として", "町内会として", "自治会として",
                "みんなに知らせ", "皆に知らせ", "みんなに伝え", "皆に伝え",
                "声をかけて回", "呼びかけました", "呼びかける", "呼びかけよう",
                "呼びかけた")
V5E_S1_WORDS = ("売らない", "売らん", "売りません", "売却しない", "譲渡しない",
                "貸さない", "貸さん", "貸しません",
                "断ろう", "断る", "応じない", "手放さない", "手放すのはやめ",
                "売るのはやめ", "売却は見送", "控えよう", "見送ろう", "やめておこう")
V5E_LEVELS = ("S1", "S2", "S3")

V5E_LEVEL_WORDS = {"S1": V5E_S1_WORDS, "S2": V5E_S2_WORDS, "S3": V5E_S3_WORDS}


# S3 は「行政の禁止/差し止め措置」なので、**行政の立場の主体にしか成立しない**
# （施主定義）。住民・事業者・仲介・記者の行は S3 にしない。
S3_ROLES = ("municipality",)


def rule_defense_level(text: str, role: Optional[str] = None) -> Optional[str]:
    """ルール側の自衛レベル。S3 > S2 > S1 の順で最初に当たったもの。

    `role` を渡すと S3 は行政の主体だけに絞る（渡さない場合は絞らない＝
    既存の呼び出しの意味を変えない）。
    """
    t = str(text or "")
    for level in ("S3", "S2", "S1"):
        if level == "S3" and role is not None and role not in S3_ROLES:
            continue
        if any(w in t for w in V5E_LEVEL_WORDS[level]):
            return level
    return None


def rule_red_v5e(text: str, rule_blue: bool,
                 role: Optional[str] = None) -> bool:
    """赤のルール1次抽出。役割の制限は付けない（行政に限らない）。

    **青（その行が土地取引・名義・持ち主のいずれかに触れている）を必要条件にする。**
    自衛語の単独ヒットだけで赤にすると、買い手の話とまったく無関係な行政・地域の
    一般論（「説明会をやる」「条例で対応する」）がそのまま赤になる
    （Codexレビュー 2026-08-28 指摘①・docs/world_design_v5e.md §1-2）。
    赤は青の上位＝青緑黄と同じ登り方に揃える。

    `rule_blue` は呼び出し側が `tools/run_metrics.py` の `_v5c_rule_blue` で
    計算して渡す（青の判定を二重に持たないため。v5c の定義は1文字も変えていない）。
    """
    return bool(rule_blue) and rule_defense_level(text, role) is not None


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
    role = row.get("role")
    llm_level = row.get("llm_defense_level")
    if llm_level not in V5E_LEVELS:
        llm_level = None
    # LLM が S3 と言っても、行政の立場でなければ S3 にはしない（定義どおり）。
    if llm_level == "S3" and role is not None and role not in S3_ROLES:
        llm_level = None
    rule_level = rule_defense_level(row.get("text", ""), role)
    if llm_level:
        level, source = llm_level, "llm"
    elif rule_level:
        level, source = rule_level, "rule"
    else:
        level, source = None, None
    return {"level": level, "level_source": source,
            "llm_level": llm_level, "rule_level": rule_level}
