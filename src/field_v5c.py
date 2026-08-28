"""A市フィールド v5c: 日常の場を増やす（世界の形だけを変える層）。

設計の正は `docs/world_design_v5c_buyer_strategy.md`。v5/v5b のファイルは変更しない。

v5b との差は **会場（venue）が増えたことだけ**である。プロンプトの文面・観測の作り方・
兆候の規則・出力スキーマは v5 のものをそのまま使う（`src/field_v5.py` を import して
組み立てる＝文面が枝分かれしない）。

「誰がどこへ行きやすいか」は世界の事実であり、ペルソナ本文（年齢・世帯・生業・
付き合い）から機械的に引く。ここで決めるのは**行き先の候補一覧**までで、
毎月どこへ行くかは主体（LLM）が選ぶ。誘導する文は一切入れない。

区画属性（configs/parcels_v5c.yaml）はこの層に入らない。台本生成器と事後集計だけが
読む＝街の観測は v5b と同一である。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .agents import Agent
from .field_v5 import build_system_prompt_v5, plan_schema_v5

# 誰の生活にも出てくる場所（買い物・回覧板・バス停）。
COMMON_VENUES = ["V10", "V13", "V14"]

# 立場から決まる場所（仕事の場）。
ROLE_VENUES = {
    "household": [],
    "business": ["V01", "V08"],
    "broker": ["V07", "V01", "V02"],
    "municipality": ["V06", "V02", "V08"],
    "media": ["V08", "V01", "V02", "V06"],
    "acquirer": [],
}

# ペルソナ本文の語 → その人の生活動線に入る場所。
# 走行前に固定。語も会場も、気づきの起きやすさを見て選んだものではない。
PERSONA_VENUE_RULES = [
    (("子ども", "小学生", "中学生", "保護者", "進学", "子育て"), ["V12", "V10"]),
    (("共同浴場", "温泉"), ["V03"]),
    (("町内会", "自治会", "公民館", "地域行事", "世話役", "ボランティア"), ["V02", "V11"]),
    (("旅館", "組合", "商工会", "同業者"), ["V05"]),
    (("喫茶店", "移住", "発信", "リモート"), ["V04"]),
    (("飲食店", "常連", "地元客", "商店"), ["V01"]),
    (("通院", "介護", "看護", "病院", "医療", "体力", "段差"), ["V15"]),
    (("散歩", "退職", "引退", "自営", "設備", "配達"), ["V09"]),
    (("通勤", "交代勤務", "支店勤務", "転勤", "空港", "駅前"), ["V14"]),
    (("研究", "大学", "コワーキング", "学術"), ["V08"]),
]

# 年齢から決まる場所（高齢ほど診療所・共同浴場・神社が日常に入る）。
ELDER_AGE = 65
ELDER_VENUES = ["V15", "V03", "V11", "V09"]


def _age(persona: str) -> int:
    m = re.search(r"(\d{2})歳", persona)
    return int(m.group(1)) if m else 0


def venue_candidates(agent: Agent, all_ids: List[str]) -> List[str]:
    """その主体の生活動線に入る会場の候補（世界の事実・毎月の選択は主体がする）。"""
    out: List[str] = list(COMMON_VENUES)
    out += ROLE_VENUES.get(agent.role, [])
    text = agent.persona or ""
    for words, venues in PERSONA_VENUE_RULES:
        if any(w in text for w in words):
            out += venues
    if _age(text) >= ELDER_AGE:
        out += ELDER_VENUES
    seen = [v for v in dict.fromkeys(out) if v in all_ids]
    return sorted(seen)


def venue_candidates_for_all(agents: List[Agent],
                             all_ids: List[str]) -> Dict[str, List[str]]:
    return {a.agent_id: venue_candidates(a, all_ids) for a in agents}


def _cfg_with_venues(cfg: Dict[str, Any], venue_ids: List[str]) -> Dict[str, Any]:
    """会場一覧だけを差し替えた config の写し（プロンプトの文面は v5 と同一）。"""
    venues = [v for v in cfg.get("social", {}).get("venues", [])
              if v["id"] in set(venue_ids)]
    social = dict(cfg.get("social", {}))
    social["venues"] = venues
    out = dict(cfg)
    out["social"] = social
    return out


def build_system_prompt_v5c(agent: Agent, cfg: Dict[str, Any], n_parcels: int,
                            venue_ids: List[str]) -> str:
    """v5 の system プロンプトそのままで、行ける場所だけをその人のものにする。"""
    return build_system_prompt_v5(agent, _cfg_with_venues(cfg, venue_ids), n_parcels)


def plan_schema_v5c(venue_ids: List[str], s4_venue: str) -> Dict[str, Any]:
    return plan_schema_v5(list(venue_ids), s4_venue)


__all__ = ["COMMON_VENUES", "ROLE_VENUES", "PERSONA_VENUE_RULES", "ELDER_AGE",
           "ELDER_VENUES", "venue_candidates", "venue_candidates_for_all",
           "build_system_prompt_v5c", "plan_schema_v5c"]
