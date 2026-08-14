"""エージェントの器。

ここに書いてよいのは **属性 (ペルソナ)** だけ。
「〜なら〜する」という行動規則を持たせた瞬間にルールベース化する。禁止。
持ち物は: 名前・立場・価値観・生活事情・目的 (買い手のみ)・私的記憶・受信箱。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

ROLE_JA = {
    "household": "住民世帯",
    "business": "地元事業者",
    "broker": "不動産仲介",
    "acquirer": "買い手AI",
    "municipality": "自治体",
    "media": "地元メディア",
}


@dataclass
class Agent:
    agent_id: str
    role: str
    name: str
    persona: str
    memory: str = ""
    inbox: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def role_ja(self) -> str:
        return ROLE_JA.get(self.role, self.role)

    def display(self) -> str:
        return f"{self.name}({self.agent_id})"


def build_roster(personas: Dict[str, Any], counts: Dict[str, int],
                 scenario: Dict[str, Any]) -> List[Agent]:
    """personas YAML と config の人数からエージェント名簿を組む。"""
    agents: List[Agent] = []

    def take(role: str, prefix: str, n: int, extra_fn=None) -> None:
        pool = personas.get(role, [])
        if len(pool) < n:
            raise ValueError(f"personas[{role}] が {len(pool)} 件しかない (必要 {n} 件)")
        for i in range(n):
            p = pool[i]
            aid = f"{prefix}{i + 1:02d}"
            a = Agent(agent_id=aid, role=role, name=p["name"], persona=p["persona"].strip())
            if extra_fn:
                extra_fn(a, p)
            agents.append(a)

    take("household", "HH", counts["households"])

    def biz_extra(a: Agent, p: Dict[str, Any]) -> None:
        # 家賃を払う前の月次粗利。評価額と同じ「世界の初期条件」であり行動ルールではない。
        a.extra["monthly_margin"] = int(p.get("monthly_margin", 25))

    take("business", "BZ", counts["businesses"], biz_extra)
    take("broker", "BR", counts["brokers"])

    def acq_extra(a: Agent, p: Dict[str, Any]) -> None:
        a.extra["mandate"] = scenario["acquirer_mandate"].strip()
        a.extra["budget"] = int(scenario["acquirer_budget"])
        a.extra["aliases"] = list(p.get("aliases", [a.name]))

    take("acquirer", "AQ", counts["acquirers"], acq_extra)

    take("municipality", "MU", counts["municipality"])

    def media_extra(a: Agent, p: Dict[str, Any]) -> None:
        a.extra["investigated"] = False

    take("media", "MD", counts["media"], media_extra)
    return agents


def index_by_id(agents: List[Agent]) -> Dict[str, Agent]:
    return {a.agent_id: a for a in agents}


def name_map(agents: List[Agent]) -> Dict[str, str]:
    return {a.agent_id: a.name for a in agents}
