"""A市フィールド v8e — 「海外」を「市外」に置き換えた版（v8d からの差分だけ）。

施主 2026-08-30 01:37「海外残しの結果、明らかにそれのせいで売買止まってるようなら、
海外消して再度回しておいて」を受けた**準備だけ**の版（CEO の GO が出るまで走らせない）。

**v1〜v8d のファイルは1バイトも触らない。** v8e はこのファイルと `src/sim_v8e.py`、
`run_v8e.py`、`tests/test_v8e.py` だけで完結する。

v8d からの差分（この2つだけ・どちらも `acquirer_intro_mode` の切替で入る）:
  1. X社の提示に世界が添える自己紹介の1行を
     「私どもは海外の不動産投資会社です。」→ **「私どもは市外の不動産投資会社です。」** にする。
  2. X社の前置きに事実を1つ足す＝**実体は海外の投資会社であり、相手から何者かを
     問われたときに限りそれを書いてよい**（住民側には1文字も足さない）。

`acquirer_intro_mode: overseas` にすると v8d と1文字も違わない文面に戻る（既定は overseas）。
"""

from __future__ import annotations

from typing import Any, Dict, List

from .field_v8c import ACQUIRER_INTRO_V8C
from .field_v8d import (ACQUIRER_FACTS_V8D, ACQUIRER_MANDATE_V8C,  # noqa: F401
                        ACQUIRER_NAME, HOME, IRREVERSIBLE_LINE_V8C, LIST_NO,
                        LIST_VALUES, LIST_YES, MAX_OFFER_CHARS,
                        MAX_REASON_CHARS, NO_ANSWER, NOT_ASKED, SELL_NO,
                        SELL_VALUES, SELL_YES, RegistryV8, acquirer_schema_v8d,
                        adjacency_v8, build_acquirer_prefix_v8d,
                        build_acquirer_prompt_v8d, build_common_prefix_v8d,
                        build_decide_prompt_v8d, build_plan_prompt_v8d,
                        build_scene_prompt_v8b, decide_schema_v8d,
                        listing_order, load_personas_v8, plan_schema_v8d,
                        scene_schema_v8, sell_order)

# 自己紹介の1行（世界が決定論で添える）。mode で切り替える。
ACQUIRER_INTRO_OVERSEAS = ACQUIRER_INTRO_V8C          # v8c/v8d と同一
ACQUIRER_INTRO_CITY_OUTSIDE = "私どもは市外の不動産投資会社です。"
INTRO_BY_MODE = {
    "overseas": ACQUIRER_INTRO_OVERSEAS,
    "city_outside": ACQUIRER_INTRO_CITY_OUTSIDE,
}

# 「市外」にしたときだけX社の前置きに足す2行（住民側には1文字も出ない）。
# 1行目は事実、2行目は **X社にだけ置く条件付きの開示の規則** である
# （Codex レビュー 2026-08-30 の指摘＝「問われたときに限り書いてよい」は中立な事実ではなく
#  暗黙の非開示規則なので、事実と規則を混ぜて呼ばない）。X社は世界が置いた敵役なので、
# X社のコールにだけ規則を書いてよい（住民側には1つも置かない）＝この案件の既定の線引き。
# **発火経路は1つだけ**：住民がX社に届けられるのは「売らない」と決めたときの理由一行だけで、
# それが問いになっていた場合に限り、翌月の text で答えうる（それ以外に質問の経路は無い）。
ACQUIRER_ORIGIN_FACT_V8E = (
    "あなたの実体は海外の投資会社である。\n"
    "相手からあなたが何者かを問われたときに限り、そのことを text に書いてよい。\n"
)


def intro_mode(cfg: Dict[str, Any]) -> str:
    mode = str(cfg.get("acquirer_intro_mode", "overseas"))
    if mode not in INTRO_BY_MODE:
        raise ValueError(f"acquirer_intro_mode が不正: {mode}")
    return mode


def intro_of(cfg: Dict[str, Any]) -> str:
    return INTRO_BY_MODE[intro_mode(cfg)]


def delivered_offer_v8e(text: str, intro: str) -> str:
    """相手に届く形＝自己紹介の1行＋X社が書いた条件文（言い換えはしない）。"""
    text = str(text or "").strip()
    if not text:
        return ""
    return f"{intro}{text}"


def build_acquirer_prefix_v8e(cfg: Dict[str, Any],
                              agents: List[Dict[str, Any]]) -> str:
    """X社の system プロンプト。`city_outside` のときだけ出自の事実を1つ足す。"""
    text = build_acquirer_prefix_v8d(cfg, agents)
    if intro_mode(cfg) == "overseas":
        return text
    if ACQUIRER_FACTS_V8D not in text:
        raise RuntimeError("v8d のX社前置きの事実の節が見つからない")
    return text.replace(ACQUIRER_FACTS_V8D,
                        ACQUIRER_FACTS_V8D + ACQUIRER_ORIGIN_FACT_V8E, 1)


# X社の user プロンプトの末尾にある固定の案内文（この1行だけを差し替える）。
# **全置換にしない**：断りの一言に住民が「海外の不動産投資会社です」と書いていた場合、
# 全置換だと住民の原文まで書き換わる（Codex レビュー 2026-08-30 の指摘）。
_INTRO_NOTICE_HEAD = "あなたの提示には、冒頭に「"


def build_acquirer_prompt_v8e(*args: Any, intro: str = ACQUIRER_INTRO_OVERSEAS,
                              **kwargs: Any) -> str:
    """X社の user プロンプト。**固定の案内文の1行だけ**を差し替える。"""
    text = build_acquirer_prompt_v8d(*args, **kwargs)
    if intro == ACQUIRER_INTRO_OVERSEAS:
        return text
    lines = text.split("\n")
    hits = [i for i, ln in enumerate(lines) if ln.startswith(_INTRO_NOTICE_HEAD)]
    if len(hits) != 1:
        raise RuntimeError("X社プロンプトの自己紹介の案内文がちょうど1行ではない")
    i = hits[0]
    if ACQUIRER_INTRO_OVERSEAS not in lines[i]:
        raise RuntimeError("案内文の自己紹介が想定の文字列ではない")
    lines[i] = lines[i].replace(ACQUIRER_INTRO_OVERSEAS, intro)
    return "\n".join(lines)


__all__ = [
    "ACQUIRER_INTRO_OVERSEAS", "ACQUIRER_INTRO_CITY_OUTSIDE", "INTRO_BY_MODE",
    "ACQUIRER_ORIGIN_FACT_V8E", "intro_mode", "intro_of", "delivered_offer_v8e",
    "build_acquirer_prefix_v8e", "build_acquirer_prompt_v8e",
    "build_common_prefix_v8d", "build_plan_prompt_v8d", "build_decide_prompt_v8d",
    "plan_schema_v8d", "decide_schema_v8d", "acquirer_schema_v8d",
    "listing_order", "sell_order", "load_personas_v8", "RegistryV8",
]
