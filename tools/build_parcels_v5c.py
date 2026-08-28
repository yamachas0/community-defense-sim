#!/usr/bin/env python
"""v5c の区画属性（世界の事実）を1回だけ生成して YAML に凍結する。

    python tools/build_parcels_v5c.py --config configs/config_field_v5c.yaml \
        --out configs/parcels_v5c.yaml

生成規則は docs/world_design_v5c_buyer_strategy.md §2 と、下のコメントに固定してある。
**結果を見て調整しない**。走行前に1回だけ回し、出力を commit して以後は固定する。

ここで作るのは「世界の形」であって、主体の観測でも行動規則でもない：
  ・zone / frontage / size_class / use_detail / owner_profile は世界の事実
  ・visibility は **買い手（台本生成器）が取得順を決めるときだけに使う内部値**で、
    どのプロンプトにも出ない（＝街の観測は v5b と一切変わらない）。
属性は既存のペルソナ本文・既存の world 設定から機械的に導く。ペルソナは変えない。
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.agents import build_roster                    # noqa: E402
from src.world import assign_tenancies, build_town     # noqa: E402

# --- 街路の骨格（world の grid から決まる固定値・8x6 を4分する主要道路） ------
# 東西の主要道路が y=2 と y=3 の間、南北の主要道路が x=3 と x=4 の間を通る。
# 主要道路に face する列・行を「表通り」、両方に面する角を「角地」とする。
MAIN_ROAD_ROWS = (2, 3)
MAIN_ROAD_COLS = (3, 4)
# 街の中心（駅と湾岸の境目）。zone はここからの距離（チェビシェフ）で決まる。
CENTER_XY = (3.5, 1.5)
ZONE_CENTER_MAX = 1.5      # d <= 1.5 → 中心
ZONE_MIDDLE_MAX = 2.5      # d <= 2.5 → 中間、それ以外 → 郊外

SIZE_SMALL_MAX = 110       # ㎡。area_pattern_sqm に対する固定の区切り
SIZE_MID_MAX = 240

# --- ペルソナ→owner_profile / use_detail を引くための語（走行前に固定） ------
ABSENT_WORDS = ("県外", "遠方", "転勤", "移住", "支店勤務")
NO_SUCCESSOR_WORDS = ("後継", "引退", "任せて", "続けていく見通し")
OLD_HOUSE_WORDS = ("築古", "古い家", "古い住宅", "傷み", "空室", "空き家", "改装")
INHERIT_WORDS = ("相続", "実家")
SINGLE_WORDS = ("単身",)
LODGING_WORDS = ("旅館", "温泉", "宿")
OFFICE_WORDS = ("コワーキング", "資材置場", "施設")


def _age(persona: str) -> int:
    m = re.search(r"(\d{2})歳", persona)
    return int(m.group(1)) if m else 0


def _has(persona: str, words) -> bool:
    return any(w in persona for w in words)


def zone_of(x: int, y: int) -> str:
    d = max(abs(x - CENTER_XY[0]), abs(y - CENTER_XY[1]))
    if d <= ZONE_CENTER_MAX:
        return "中心"
    if d <= ZONE_MIDDLE_MAX:
        return "中間"
    return "郊外"


def frontage_of(x: int, y: int) -> str:
    row = y in MAIN_ROAD_ROWS
    col = x in MAIN_ROAD_COLS
    if row and col:
        return "角地"
    if row or col:
        return "表通り"
    return "裏通り"


def size_class_of(area: int) -> str:
    if area <= SIZE_SMALL_MAX:
        return "小"
    if area <= SIZE_MID_MAX:
        return "中"
    return "大"


def use_detail_of(parcel, owner_persona: str, tenant_persona: str) -> str:
    """区画の使われ方を、既存 use と持ち主・使い手のペルソナから機械的に細分する。"""
    if parcel.use == "public":
        return "公共施設"
    if parcel.use == "vacant":
        return "空地"
    if parcel.use == "shop":
        text = tenant_persona or owner_persona
        if _has(text, LODGING_WORDS) and "旅館" in text:
            return "旅館"
        if _has(text, OFFICE_WORDS):
            return "事業所"
        return "店舗"
    # residential
    if _has(owner_persona, OLD_HOUSE_WORDS) or _age(owner_persona) >= 70:
        return "古い住宅"
    return "住宅"


def owner_profile_of(parcel, owner_role: str, owner_persona: str,
                     tenanted: bool) -> str:
    if parcel.use == "public":
        return "公有"
    if owner_role == "business" or tenanted:
        return "営業中の店"
    age = _age(owner_persona)
    if age >= 65 and _has(owner_persona, SINGLE_WORDS):
        return "高齢単身"
    if _has(owner_persona, ABSENT_WORDS):
        return "不在地主"
    if _has(owner_persona, NO_SUCCESSOR_WORDS) or (age >= 65
                                                   and _has(owner_persona, INHERIT_WORDS)):
        return "後継者なし"
    if age >= 65:
        return "高齢単身" if _has(owner_persona, SINGLE_WORDS) else "後継者なし"
    return "現役世帯"


# 目立ちやすさ（買い手の内部値・0〜10）。表通り・角地・大・店舗・中心ほど高い。
VIS_FRONTAGE = {"表通り": 2, "角地": 3, "裏通り": 0}
VIS_SIZE = {"小": 0, "中": 1, "大": 2}
VIS_USE = {"店舗": 2, "旅館": 2, "事業所": 1, "住宅": 1, "古い住宅": 0,
           "空地": 0, "公共施設": 3}
VIS_ZONE = {"中心": 2, "中間": 1, "郊外": 0}


def visibility_of(frontage: str, size_class: str, use_detail: str, zone: str) -> int:
    return (VIS_FRONTAGE[frontage] + VIS_SIZE[size_class]
            + VIS_USE[use_detail] + VIS_ZONE[zone])


def build(cfg: dict, personas: dict) -> dict:
    agents = build_roster(personas, cfg["agents"], cfg["scenario"])
    by_id = {a.agent_id: a for a in agents}
    hh = [a.agent_id for a in agents if a.role == "household"]
    bz = [a.agent_id for a in agents if a.role == "business"]
    muni = next(a.agent_id for a in agents if a.role == "municipality")
    parcels = build_town(cfg["world"], hh, bz, muni)
    assign_tenancies(parcels, bz, 0)

    rows = []
    for p in parcels:
        owner = by_id.get(p.owner_id)
        tenant = by_id.get(p.tenant_id) if p.tenant_id else None
        owner_persona = owner.persona if owner else ""
        tenant_persona = tenant.persona if tenant else ""
        zone = zone_of(p.x, p.y)
        frontage = frontage_of(p.x, p.y)
        size_class = size_class_of(p.area_sqm)
        use_detail = use_detail_of(p, owner_persona, tenant_persona)
        profile = owner_profile_of(p, owner.role if owner else "", owner_persona,
                                   bool(p.tenant_id))
        rows.append({
            "pid": p.pid, "x": p.x, "y": p.y, "block": p.block,
            "use": p.use, "use_detail": use_detail,
            "zone": zone, "frontage": frontage,
            "area_sqm": p.area_sqm, "size_class": size_class,
            "owner_id": p.owner_id, "owner_name": owner.name if owner else "",
            "tenant_id": p.tenant_id or "",
            "owner_profile": profile,
            "visibility": visibility_of(frontage, size_class, use_detail, zone),
        })
    return {
        "meta": {
            "generated_by": "tools/build_parcels_v5c.py",
            "source": "configs/config_field_v5c.yaml + configs/personas/field_v4_1.yaml",
            "note": ("区画の属性＝世界の事実。走行前に凍結し、結果を見て調整しない。"
                     "visibility は買い手（台本生成器）の内部値であり、"
                     "どのプロンプトにも出ない＝街の観測は v5b と同一。"),
            "rules": {
                "zone": f"中心=中心点{CENTER_XY}からのチェビシェフ距離<= {ZONE_CENTER_MAX}"
                        f"／中間<= {ZONE_MIDDLE_MAX}／それ以外=郊外",
                "frontage": f"主要道路 行{MAIN_ROAD_ROWS}・列{MAIN_ROAD_COLS} の"
                            "両方に面する=角地／片方=表通り／どちらでもない=裏通り",
                "size_class": f"小<= {SIZE_SMALL_MAX}㎡／中<= {SIZE_MID_MAX}㎡／大=それ超",
                "use_detail": "既存 use ＋ 所有者・使い手のペルソナ本文の語から機械判定",
                "owner_profile": "所有者ペルソナの年齢・世帯・生業から機械判定",
                "visibility": "frontage+size+use+zone の加点（買い手の内部値）",
            },
        },
        "parcels": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_field_v5c.yaml")
    ap.add_argument("--out", default="configs/parcels_v5c.yaml")
    args = ap.parse_args()

    with open(os.path.join(ROOT, args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(ROOT, cfg["personas_file"]), encoding="utf-8") as f:
        personas = yaml.safe_load(f)

    doc = build(cfg, personas)
    out = os.path.join(ROOT, args.out)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False, width=100)

    import collections
    rows = [r for r in doc["parcels"] if r["use"] != "public"]
    print(f"[out] {out}  非公共 {len(rows)}区画 / 全{len(doc['parcels'])}")
    for key in ("zone", "frontage", "size_class", "use_detail", "owner_profile"):
        print(f"  {key:14s}", dict(collections.Counter(r[key] for r in rows).most_common()))
    print("  visibility    ",
          dict(sorted(collections.Counter(r["visibility"] for r in rows).items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
