# -*- coding: utf-8 -*-
"""最小の町の走行ログを、図を描くための素直な形に読み直す層。

図を描く側（make_figures.py）はこのモジュールの Town だけを見る。
v8b の走行フォルダからでも、v8c の emergence_v8c.json からでも同じ Town を作れる。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

# 「X社の話」の見分け方。走行ログの本文にこの語がそのまま出る。
ACQUIRER_TOKEN = "X社"

# X社の条件文から相手ごとの物件名を外して「条件の型」を数えるための整形。
_OBJECT_CLAUSE = re.compile(r"^.*?の名義を")
_OBJECT_MASK = "〈その人の持ちもの〉の名義を"


@dataclass
class Person:
    pid: str
    name: str
    role: str
    district: str
    sellable: bool
    neighbours: list = field(default_factory=list)   # pid のリスト
    sold_month: int = None
    listed_months: list = field(default_factory=list)

    @property
    def first_listed_month(self):
        return min(self.listed_months) if self.listed_months else None

    @property
    def first_move_month(self):
        cands = [m for m in (self.sold_month, self.first_listed_month) if m]
        return min(cands) if cands else None

    @property
    def outcome(self):
        if self.sold_month:
            return "sold"
        if self.listed_months:
            return "listed"
        return "stayed"


@dataclass
class Town:
    label: str
    months: int
    people: list                 # Person
    deliveries: list             # {month, to_pid, from_pid, route, venue, text, about_acquirer, about_sale}
    offers: list                 # {month, to_pid, text, template, accepted}
    monthly: list                # {month, offers_sent, listed, accepted, sold, ...}
    venues: list                 # 場の名前

    def by_id(self, pid):
        return self._index[pid]

    def __post_init__(self):
        self._index = {p.pid: p for p in self.people}


# --------------------------------------------------------------------------
# v8b の走行フォルダを読む
# --------------------------------------------------------------------------

def load_run_dir(run_dir):
    """走行フォルダ（timeline_index.json / deliveries.json / …）から Town を作る。"""
    def j(name):
        with open(os.path.join(run_dir, name), encoding="utf-8") as f:
            return json.load(f)

    index = j("timeline_index.json")
    neighbours = j("neighbours.json")
    listings = j("listings.json")
    offers_raw = j("offers.json")
    deliveries_raw = j("deliveries.json")
    monthly_raw = j("monthly.json")
    summary = j("summary.json")

    # district は各人の timeline にしか無いので、そこから拾う。
    district = {}
    for row in index:
        with open(os.path.join(run_dir, row["file"]), encoding="utf-8") as f:
            district[row["agent_id"]] = json.load(f).get("district", "")

    listed_by = {}
    for l in listings:
        listed_by.setdefault(l["agent_id"], []).append(l["step"])

    people = [
        Person(
            pid=row["agent_id"],
            name=row["name"],
            role=row["role"],
            district=district.get(row["agent_id"], ""),
            sellable=row["sellable"],
            neighbours=list(neighbours.get(row["agent_id"], [])),
            sold_month=row["sold_month"],
            listed_months=sorted(listed_by.get(row["agent_id"], [])),
        )
        for row in index
    ]

    sold_by_name = {p.name: p.sold_month for p in people if p.sold_month}

    deliveries = []
    for d in deliveries_raw:
        text = d["text"]
        # 「誰かが売った話」= その月までに実際に売った人の名前が話に出てくること。
        about_sale = any(
            d["step"] >= m and nm in text for nm, m in sold_by_name.items()
        )
        deliveries.append({
            "month": d["step"],
            "to_pid": d["to"],
            "from_pid": d.get("from_id"),
            "route": d["route"],
            "venue": d["venue_label"],
            "text": text,
            "about_acquirer": ACQUIRER_TOKEN in text,
            "about_sale": about_sale,
        })

    offers = [{
        "month": o["step"],
        "to_pid": o["to_id"],
        "text": o["text"],
        "template": _OBJECT_CLAUSE.sub(_OBJECT_MASK, o["text"], count=1),
        "accepted": bool(o["accepted"]),
        "reply": o.get("result", ""),
        # v8b には断りの一言が無いので "" のまま。v8c は decline_reason に入っている。
        "reply_reason": o.get("decline_reason", ""),
    } for o in offers_raw]

    monthly = [{
        "month": m["step"],
        "offers_sent": m.get("offers_sent", 0),
        "listed": m.get("listed_this_month", 0),
        "accepted": m.get("accepted_this_month", 0),
        "sold": m.get("sold_this_month", 0),
        "attended": m.get("attended", 0),
        "by_venue": m.get("by_venue", {}),
    } for m in monthly_raw]

    venues = []
    for m in monthly_raw:
        for v in m.get("by_venue", {}):
            if v not in venues and "行かない" not in v:
                venues.append(v)

    return Town(
        label=summary.get("run_name", os.path.basename(run_dir)),
        months=summary.get("months_run", len(monthly)),
        people=people,
        deliveries=deliveries,
        offers=offers,
        monthly=monthly,
        venues=venues,
    )


# --------------------------------------------------------------------------
# v8c の emergence_v8c.json を読む
# --------------------------------------------------------------------------
#
# 実際に出た emergence_v8c.json のトップレベルキーは
#   run / reason_candidates / heard_before_decision / contagion /
#   acquirer_adaptation / monthly
# であり、この図が要る「人・発話・提示の生レコード」は入っていない
# （集計済みの数値だけの要約ファイル）。一方 emergence_v8c.json は走行フォルダ
# （offers.json / deliveries.json / timeline_index.json / neighbours.json /
# listings.json / summary.json …）の中に置かれており、それらは v8b と
# 同じ形（フィールド名も一致）なので、load_run_dir() がそのまま使える。
# そのため emergence_v8c.json のパスを渡されたときは、
# 「隣にある生ログ一式」を走行フォルダとして読みに行く。
V8C_KEY_MAP = {
    "people": "people",
    "deliveries": "deliveries",
    "offers": "offers",
    "monthly": "monthly",
}


def load_emergence_json(path):
    """emergence_v8c.json を Town にする（将来、集計値そのものを生レコード
    形式で書き出す版が来た場合のための経路）。キー対応が合わないときは
    推測で埋めず、見つかったキーを添えて止める。"""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    missing = [k for k in V8C_KEY_MAP.values() if k not in raw]
    if missing:
        raise KeyError(
            "emergence_v8c.json のキーが V8C_KEY_MAP と合っていない。"
            f"不足={missing} / 実際のトップレベルキー={sorted(raw.keys())} "
            "→ emergence_data.py の V8C_KEY_MAP を実キー名に直すこと。"
        )

    people = [Person(**p) for p in raw[V8C_KEY_MAP["people"]]]
    return Town(
        label=raw.get("label", os.path.basename(path)),
        months=raw.get("months", max(m["month"] for m in raw[V8C_KEY_MAP["monthly"]])),
        people=people,
        deliveries=raw[V8C_KEY_MAP["deliveries"]],
        offers=raw[V8C_KEY_MAP["offers"]],
        monthly=raw[V8C_KEY_MAP["monthly"]],
        venues=raw.get("venues", []),
    )


def load(source):
    """フォルダなら走行ログ、.json なら emergence_v8c.json として読む。
    emergence_v8c.json は集計済みの要約であって生レコードを持たないため、
    同じ走行フォルダに生ログ（timeline_index.json 等）が並んでいれば、
    そちらを load_run_dir() で読む（キー不一致ではなく、そもそも
    emergence_v8c.json 単体には図に要るデータが無いための代替）。"""
    if os.path.isdir(source):
        return load_run_dir(source)
    run_dir = os.path.dirname(os.path.abspath(source))
    if os.path.exists(os.path.join(run_dir, "timeline_index.json")):
        return load_run_dir(run_dir)
    return load_emergence_json(source)


# --------------------------------------------------------------------------
# 図が共通で使う集計
# --------------------------------------------------------------------------

WINDOW = 3   # 「判断の前3か月」


def heard_counts(town, pid, m_from, m_to):
    """ある人が m_from〜m_to（両端含む）に聞いた話を、種類ごとに数える。"""
    acq = sale = 0
    for d in town.deliveries:
        if d["to_pid"] != pid or not (m_from <= d["month"] <= m_to):
            continue
        if d["about_acquirer"]:
            acq += 1
        if d["about_sale"]:
            sale += 1
    return acq, sale


def heard_before_decision(town, person):
    """判断の前3か月に聞いた件数。動かなかった人は、比べられるように
    全期間を3か月ずつ見たときの平均を使う（キャプションに明記する）。"""
    e = person.first_move_month
    if e:
        lo, hi = max(1, e - WINDOW), max(1, e - 1)
        acq, sale = heard_counts(town, person.pid, lo, hi)
        return acq, sale, (lo, hi), False
    acq, sale = heard_counts(town, person.pid, 1, town.months)
    scale = WINDOW / float(town.months)
    return acq * scale, sale * scale, (1, town.months), True
