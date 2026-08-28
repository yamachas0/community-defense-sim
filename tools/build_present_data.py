#!/usr/bin/env python
"""プレゼン画面 `present.html` が読む JSON を run_dir から焼く。

    python tools/build_present_data.py --run simulations/<run_dir> --label run94
    → <pages>/present_data_run94.json

画面に出る数値・塗りは**すべてこの JSON から**描く（HTML側に手打ちしない）。
右地図（街が語った区画）の塗り条件は `docs/present_verification.md` (a) の結論どおり：
  ①ルール1次抽出 ∧ LLM の about_acquisition ②その月までに成立した取得 ③当事者以外。
`run_metrics` の関数をそのまま再利用し、公開済み `metrics_v5.json` と突き合わせて検査する。

原文は要約しない。引用は utt_id / step / from を付けてそのまま運ぶ。
"""

from __future__ import annotations

import argparse
import ast
import collections
import datetime as _dt
import json
import math
import re
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import yaml  # noqa: E402

from run_metrics import (_read_jsonl, _v5_mentions, _v5_s4_kind,  # noqa: E402
                         _v5c_rule_blue, _v5c_rule_green, _v5c_rule_red,
                         _v5c_rule_yellow, _v5c_stage, V5C_COLORS, V5_PARCEL_RE)


def _load(run_dir, name):
    return _read_jsonl(os.path.join(run_dir, name))


# ---------------------------------------------------------------------------
# 地図の配置（マスの大きさ＝敷地面積／中央＝市街地・外側＝郊外）
# 配置は**ここで決定論に計算して JSON に入れる**。HTML は座標を描くだけ。
# ---------------------------------------------------------------------------

LAYOUT_VIEW = {"w": 1400, "h": 900}
LAYOUT_SEED = 8501
LAYOUT_R = 430.0
LAYOUT_SIDE_MIN = 34.0
LAYOUT_ZONE_F = {"中心": 0.20, "中間": 0.52, "郊外": 0.86}
LAYOUT_GAP = 7.0
LAYOUT_SQUASH = 0.78
LAYOUT_ITERS = 500
LAYOUT_PULL = 0.12
LAYOUT_NAMES_FILE = "configs/parcel_names_v5c.yaml"
LAYOUT_VENUES_FILE = "configs/venues_v5c.yaml"
LAYOUT_VENUE_R = 22.0
LAYOUT_VENUE_GAP = 6.0
VENUES_NOTE = "表示位置は装飾。世界の同席・配送・判定には使わない"


def _lcg(seed: int, n: int):
    """線形合同法。決定論の [0,1) を n 個返す（乱数源を外に持たない）。"""
    out, x = [], int(seed)
    for _ in range(n):
        x = (1103515245 * x + 12345) % (2 ** 31)
        out.append(x / float(2 ** 31))
    return out


def _overlap(cx, cy, side, a, b, gap=0.0):
    need = (side[a] + side[b]) / 2.0 + gap
    ox = need - abs(cx[b] - cx[a])
    oy = need - abs(cy[b] - cy[a])
    return min(ox, oy) if (ox > 0 and oy > 0) else 0.0


def _names_src() -> dict:
    with open(os.path.join(ROOT, *LAYOUT_NAMES_FILE.split("/")), encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_parcel_names(pids) -> dict:
    """呼び名の対応表。表に無い pid があれば黙って空にせず落とす。"""
    src = _names_src()
    land = src.get("land") or {}
    missing = [p for p in pids if p not in land]
    if missing:
        raise SystemExit("呼び名の表に無い区画がある（%s）: %s"
                         % (LAYOUT_NAMES_FILE, ", ".join(missing)))
    need = ("name", "locality", "use_word")
    bad = [p for p in pids if not all(land[p].get(k) for k in need)]
    if bad:
        raise SystemExit("呼び名の表の項目が欠けている: " + ", ".join(bad))
    return {"land": {p: dict(land[p]) for p in pids},
            "registered": dict(src.get("registered_names") or {}),
            "agents": dict(src.get("agents") or {})}


def _agent_labels() -> dict:
    """内部ID／旧表示名の両方から呼び名を引ける表を作る。"""
    src = _names_src()
    out = {}
    for aid, a in (src.get("agents") or {}).items():
        name = str(a.get("name") or "")
        if not name:
            continue
        out[aid] = name
        if a.get("old"):
            out.setdefault(str(a["old"]), name)
    return out


def _agent_homes() -> dict:
    """呼び名の由来になった本拠の区画（装飾の波紋の宛先に使う）。"""
    src = _names_src()
    out = {}
    for aid, a in (src.get("agents") or {}).items():
        if a.get("home"):
            out[aid] = str(a["home"])
            if a.get("old"):
                out.setdefault(str(a["old"]), str(a["home"]))
    return out


def _rect_gap(x, y, rcx, rcy, side):
    """点と正方形の距離（内側なら 0）。"""
    h = side / 2.0
    dx = max(abs(x - rcx) - h, 0.0)
    dy = max(abs(y - rcy) - h, 0.0)
    return math.hypot(dx, dy)


def _place_venues(cx, cy, side, pids, ccx, ccy) -> list:
    """会場は区画と同じ極座標に置き、重なったら**会場だけ**を決定論に押し出す。"""
    path = os.path.join(ROOT, *LAYOUT_VENUES_FILE.split("/"))
    with open(path, encoding="utf-8") as f:
        src = yaml.safe_load(f) or {}
    R, G = LAYOUT_VENUE_R, LAYOUT_VENUE_GAP
    out = []

    def free(x, y):
        if not (R + 2 <= x <= LAYOUT_VIEW["w"] - R - 2
                and R + 2 <= y <= LAYOUT_VIEW["h"] - R - 2):
            return False
        for pid in pids:
            if _rect_gap(x, y, cx[pid], cy[pid], side[pid]) < R + G:
                return False
        for o in out:
            if math.hypot(x - o["cx"], y - o["cy"]) < 2 * R + G:
                return False
        return True

    for v in (src.get("venues") or []):
        ang = math.radians(float(v["angle_deg"]))
        base = float(v["radius_f"]) * LAYOUT_R
        got = None
        for k in range(0, 400):                      # まず外へ
            r = base + 3.0 * k
            x, y = ccx + r * math.cos(ang), ccy + r * math.sin(ang) * LAYOUT_SQUASH
            if not (-R <= x <= LAYOUT_VIEW["w"] + R and -R <= y <= LAYOUT_VIEW["h"] + R):
                break
            if free(x, y):
                got = (x, y)
                break
        if got is None:                              # 外に空きが無ければ内へ
            for k in range(1, 200):
                r = base - 3.0 * k
                if r < 0:
                    break
                x, y = ccx + r * math.cos(ang), ccy + r * math.sin(ang) * LAYOUT_SQUASH
                if free(x, y):
                    got = (x, y)
                    break
        if got is None:
            raise SystemExit("会場の置き場所が見つからない: %s" % v.get("id"))
        out.append({"id": str(v["id"]), "label": str(v["label"]),
                    "badge": str(v["badge"]), "cx": round(got[0], 2),
                    "cy": round(got[1], 2), "r": R})
    return out


def _venue_clearance(venues, cx, cy, side, pids) -> float:
    worst = 1e9
    for i, v in enumerate(venues):
        for pid in pids:
            worst = min(worst, _rect_gap(v["cx"], v["cy"], cx[pid], cy[pid], side[pid])
                        - LAYOUT_VENUE_R)
        for w in venues[i + 1:]:
            worst = min(worst, math.hypot(v["cx"] - w["cx"], v["cy"] - w["cy"])
                        - 2 * LAYOUT_VENUE_R)
    return worst if venues else 0.0


def _build_layout(cfg: dict, run_pids) -> dict:
    """run の config が指す区画ファイルから配置を作る。合わなければ None。"""
    parcels_file = cfg.get("parcels_file")
    if not parcels_file:
        return None                    # v5/v5b の run は区画属性を持たない
    path = os.path.join(ROOT, *str(parcels_file).split("/"))
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        src = yaml.safe_load(f) or {}
    items = [dict(p) for p in (src.get("parcels") or [])]
    if not items or {str(p["pid"]) for p in items} != set(run_pids):
        return None
    items.sort(key=lambda p: str(p["pid"]))
    pids = [str(p["pid"]) for p in items]
    n = len(items)
    cols = max(int(p["x"]) for p in items) + 1
    rows = max(int(p["y"]) for p in items) + 1

    # 1. マスの辺は面積の平方根に比例させる（4分位で潰さない）
    order = sorted(items, key=lambda p: (float(p["area_sqm"]), str(p["pid"])))
    area_min = float(order[0]["area_sqm"])
    side = {str(p["pid"]): LAYOUT_SIDE_MIN * math.sqrt(float(p["area_sqm"]) / area_min)
            for p in items}
    size_breaks = [float(order[k * n // 4]["area_sqm"]) for k in (1, 2, 3)]
    cx0, cy0 = LAYOUT_VIEW["w"] / 2.0, LAYOUT_VIEW["h"] / 2.0

    # 2〜5. 初期角度・目標半径・決定論ジッター・初期座標
    u = _lcg(LAYOUT_SEED, 2 * n)
    cx, cy, rt, zone = {}, {}, {}, {}
    for i, p in enumerate(items):
        pid = pids[i]
        dx = float(p["x"]) - (cols - 1) / 2.0
        dy = float(p["y"]) - (rows - 1) / 2.0
        theta = (2 * math.pi * i / n) if (dx == 0.0 and dy == 0.0) else math.atan2(dy, dx)
        zone[pid] = str(p.get("zone") or "")
        r_target = LAYOUT_ZONE_F.get(zone[pid], LAYOUT_ZONE_F["中間"]) * LAYOUT_R
        theta += (u[2 * i] - 0.5) * 0.44
        r_target *= 1 + (u[2 * i + 1] - 0.5) * 0.12
        rt[pid] = r_target
        cx[pid] = cx0 + r_target * math.cos(theta)
        cy[pid] = cy0 + r_target * math.sin(theta) * LAYOUT_SQUASH

    def separate():
        for ia in range(n):
            a = pids[ia]
            for ib in range(ia + 1, n):
                b = pids[ib]
                pen = _overlap(cx, cy, side, a, b, LAYOUT_GAP)
                if pen <= 0:
                    continue
                ddx, ddy = cx[b] - cx[a], cy[b] - cy[a]
                d = math.hypot(ddx, ddy)
                ux, uy = (1.0, 0.0) if d < 1e-9 else (ddx / d, ddy / d)
                cx[a] -= ux * pen / 2.0
                cy[a] -= uy * pen / 2.0
                cx[b] += ux * pen / 2.0
                cy[b] += uy * pen / 2.0

    def worst():
        return max((_overlap(cx, cy, side, pids[ia], pids[ib])
                    for ia in range(n) for ib in range(ia + 1, n)), default=0.0)

    # 6. 緩和（pid 昇順で走査＝浮動小数の順序を固定する）
    for _ in range(LAYOUT_ITERS):
        separate()
        for pid in pids:
            ex, ey = cx[pid] - cx0, (cy[pid] - cy0) / LAYOUT_SQUASH
            r_now = math.hypot(ex, ey)
            ang = math.atan2(ey, ex) if r_now > 1e-9 else 0.0
            r_new = r_now + LAYOUT_PULL * (rt[pid] - r_now)
            cx[pid] = cx0 + r_new * math.cos(ang)
            cy[pid] = cy0 + r_new * math.sin(ang) * LAYOUT_SQUASH
            h = side[pid] / 2.0
            cx[pid] = min(max(cx[pid], h), LAYOUT_VIEW["w"] - h)
            cy[pid] = min(max(cy[pid], h), LAYOUT_VIEW["h"] - h)

    # 6b. 【スペックからの逸脱・要判断】§1-2 の定数のままでは §1-2-8 の検査を満たせない。
    # 中心ゾーンは 48 区画中 16 で、目標半径 0.20*R=64 の楕円の面積 1.0万に対して
    # 正方形の面積合計が 3.7万＝入りきらない。押し出された分だけ半径の戻し
    # （0.12×差）が毎回 7 単位の隙間を超えて食い込み、最大重なりが 4.48 で止まる。
    # 定数（ゾーン半径・gap・0.12）は動かさず、緩和のあとに**押し離しだけ**を
    # 決定論に追い足して重なりを解く。ゾーンの順序は下の検査で担保する。
    extra_passes = 0
    while worst() > 1.0 and extra_passes < LAYOUT_ITERS:
        separate()
        extra_passes += 1

    # 7. バウンディングボックスをビュー中央へ
    x0 = min(cx[p] - side[p] / 2.0 for p in pids)
    x1 = max(cx[p] + side[p] / 2.0 for p in pids)
    y0 = min(cy[p] - side[p] / 2.0 for p in pids)
    y1 = max(cy[p] + side[p] / 2.0 for p in pids)
    sx = LAYOUT_VIEW["w"] / 2.0 - (x0 + x1) / 2.0
    sy = LAYOUT_VIEW["h"] / 2.0 - (y0 + y1) / 2.0
    for pid in pids:
        cx[pid] += sx
        cy[pid] += sy
    ccx, ccy = cx0 + sx, cy0 + sy       # 会場・下地の装飾もこの中心を使う

    # 8. 検査（黙って歪んだ図を出さない）
    max_ov = 0.0
    for ia in range(n):
        for ib in range(ia + 1, n):
            max_ov = max(max_ov, _overlap(cx, cy, side, pids[ia], pids[ib]))
    if max_ov > 1.0:
        raise SystemExit("地図の配置でマスが重なった（max_overlap=%.3f）" % max_ov)

    by_zone = collections.defaultdict(list)
    for pid in pids:
        by_zone[zone[pid]].append(
            math.hypot(cx[pid] - ccx, (cy[pid] - ccy) / LAYOUT_SQUASH))
    attr = {str(p["pid"]): p for p in items}
    names = _load_parcel_names(pids)
    sides_sorted = sorted(side.values())
    venues = _place_venues(cx, cy, side, pids, ccx, ccy)
    return {
        "view": dict(LAYOUT_VIEW),
        "center": [round(ccx, 2), round(ccy, 2)],
        "source": str(parcels_file),
        "seed": LAYOUT_SEED,
        "size_breaks": size_breaks,
        "cells": {pid: {"cx": round(cx[pid], 2), "cy": round(cy[pid], 2),
                        "side": round(side[pid], 2), "zone": zone[pid],
                        "area_sqm": attr[pid].get("area_sqm"),
                        "size_class": attr[pid].get("size_class", ""),
                        "use": attr[pid].get("use", ""),
                        "use_detail": attr[pid].get("use_detail", ""),
                        "owner_name": attr[pid].get("owner_name", ""),
                        "frontage": attr[pid].get("frontage", ""),
                        "name": names["land"][pid]["name"],
                        "locality": names["land"][pid]["locality"],
                        "use_word": names["land"][pid]["use_word"],
                        "owner_label": names["registered"].get(
                            str(attr[pid].get("owner_name", "")),
                            str(attr[pid].get("owner_name", "")))}
                  for pid in pids},
        "naming": {"source": LAYOUT_NAMES_FILE, "n": len(pids)},
        "venues": venues,
        "venues_note": VENUES_NOTE,
        "zone_radius": {z: LAYOUT_ZONE_F[z] * LAYOUT_R for z in LAYOUT_ZONE_F},
        "checks": {"max_overlap": round(max_ov, 4),
                   "extra_separation_passes": extra_passes,
                   "side_min": round(sides_sorted[0], 2),
                   "side_max": round(sides_sorted[-1], 2),
                   "side_ratio": round(sides_sorted[-1] / sides_sorted[0], 4),
                   "draw_area_ratio": round((sides_sorted[-1] / sides_sorted[0]) ** 2, 4),
                   "venue_min_clearance": round(_venue_clearance(venues, cx, cy, side,
                                                                 pids), 4),
                   "mean_radius_by_zone": {z: round(sum(v) / len(v), 2)
                                           for z, v in sorted(by_zone.items())}},
    }


# ---------------------------------------------------------------------------
# 区画の呼び名（固有名は無い。街は区画IDか持ち主の名前で呼ぶ）— 実測で数える
# ---------------------------------------------------------------------------

def _parcel_naming(layout, utts, thoughts, articles) -> dict:
    if not layout:
        return None
    texts = ([str(u.get("text", "")) for u in utts]
             + [str(t.get("text", "")) for t in thoughts]
             + [str(a.get("text", "")) for a in articles])
    blob = "\n".join(texts)
    mentions = {}
    for pid, c in layout["cells"].items():
        name = str(c.get("owner_name") or "")
        mentions[pid] = blob.count(name) if name else 0
    return {
        "basis": "会話・内心・記事の本文に持ち主の名前（R03 等）が現れた回数を数えた実測",
        "proper_names": 0,
        "note": "区画に固有名は無い。街は区画ID か持ち主の名前で呼ぶ。",
        "name_in_speech": {pid: mentions[pid] > 0 for pid in mentions},
        "mentions": mentions,
    }


# ---------------------------------------------------------------------------
# 火が点いた瞬間（原文）— 緑・黄の初出を前後の原文つきで機械抽出
# ---------------------------------------------------------------------------

SCENE_ORDER = {"plan": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4}


def _sort_key(r) -> tuple:
    return (int(r.get("step") or 0), SCENE_ORDER.get(r.get("scene") or "", 9),
            int(r.get("round") or 0), str(r.get("utt_id") or ""),
            str(r.get("from") or ""))


def _heard_by(u):
    hb = u.get("heard_by")
    if isinstance(hb, str):
        try:
            hb = ast.literal_eval(hb)
        except (ValueError, SyntaxError):
            hb = []
    return list(hb or [])


def _ctx_row(u):
    return {"kind": "utterance", "step": int(u.get("step") or 0),
            "scene": u.get("scene", ""), "venue": u.get("venue", ""),
            "round": u.get("round", 0), "from": u.get("from", ""),
            "name": u.get("name", ""), "role": u.get("role", ""),
            "utt_id": u.get("utt_id", ""), "text": str(u.get("text", ""))}


IGNITION_CRITERIA = {
    "loose": "走行前に固定した色の定義（ルール1次抽出 ∧ LLM）どおりの初出",
    "strict_green": ("その月までに成立した取得区画IDが2つ以上、または異なる名義が2つ以上、"
                     "本文に実際に並んでいる最初の行"),
    "strict_yellow": ("その月までに成立した異なる名義が2つ以上、"
                      "本文に実際に並んでいる最初の行"),
}


def _ignition_matched(row, holders_by_step, acquired_by_step):
    step = int(row.get("step") or 0)
    text = str(row.get("text", ""))
    hit = _v5_mentions(text, holders_by_step.get(step, set()))
    parcels = sorted(p for p in hit["parcels"] if p in acquired_by_step.get(step, set()))
    return {"parcels": parcels, "holders": sorted(hit["holders"])}


def _ignition_strict_hit(row, color, holders_by_step, acquired_by_step):
    if not row.get("classified"):
        return None
    if row.get("stage") != color:
        return None
    m = _ignition_matched(row, holders_by_step, acquired_by_step)
    if color == "green":
        if not row.get("llm_area"):
            return None
        if len(m["parcels"]) < 2 and len(m["holders"]) < 2:
            return None
    else:
        if not row.get("llm_same_buyer"):
            return None
        if len(m["holders"]) < 2:
            return None
    return m


PARTY_BASIS = ("その月までに成立した取引の当事者（売主・貸主）が、その行の場に居合わせたか。"
               "同席は plans_v5.jsonl の行き先（発話・内心はその場面の同席者、"
               "自宅・計画の内心は本人だけ）から機械抽出した。本人が当事者の場合も含む")
PARTY_CAVEAT = "同席は主体のLLMが毎月選んだ行き先の結果であり、こちらで仕組んでいない"


def _party_lookup(plans, events, labels):
    """発話・内心の場に、取引の当事者本人が居合わせたかを機械抽出する。"""
    att = collections.defaultdict(set)
    for p in plans:
        step = int(p.get("step") or 0)
        for scene in ("S1", "S2", "S3", "S4"):
            v = p.get(scene)
            if v:
                att[(step, scene, str(v))].add(str(p.get("agent_id") or ""))

    def party_present(row):
        step = int(row.get("step") or row.get("month") or 0)
        who = str(row.get("from") or "")
        here = set(att.get((step, str(row.get("scene") or ""),
                            str(row.get("venue") or "")), set())) or {who}
        agents = sorted(a for a in here
                        if any(e["party"] == a and e["month"] <= step for e in events))
        parcels = sorted({e["parcel_id"] for e in events
                          if e["month"] <= step and e["party"] in agents})
        return {"present": bool(agents), "agents": agents,
                "labels": [labels.get(a, a) for a in agents], "parcels": parcels}

    return party_present


def _ignition_timeline(rows, utts, venue_labels, labels, homes,
                       holders_by_step, acquired_by_step, party_present):
    """月 → その月に初めて緑／黄に達した出来事（人ごとの初到達）。"""
    heard_of = {str(u.get("utt_id") or ""): _heard_by(u) for u in utts}
    seen, picks = set(), []
    for r in sorted(rows, key=_sort_key):
        c = r.get("stage")
        if c not in ("green", "yellow") or (r["from"], c) in seen:
            continue
        seen.add((r["from"], c))
        picks.append(r)
    timeline, excluded = collections.defaultdict(list), []
    for r in picks:
        month = int(r.get("step") or 0)
        if r.get("kind") == "article":
            # 記事は「発話」でも「内心」でもない＝この年表の2種に当てはまらないので
            # 黙って混ぜず、別に列挙する（数を隠さない）。
            excluded.append({"month": month, "color": r["stage"],
                             "agent": r["from"], "agent_label": labels.get(r["from"],
                                                                           r["from"]),
                             "kind": "article", "text": r["text"]})
            continue
        m = _ignition_matched(r, holders_by_step, acquired_by_step)
        speech = r.get("kind") == "utterance"
        timeline[str(month)].append({
            "month": month, "color": r["stage"],
            "kind": "speech" if speech else "thought",
            "agent": r["from"], "agent_label": labels.get(r["from"], r["from"]),
            "venue": r.get("venue", ""),
            "venue_label": venue_labels.get(r.get("venue", ""), r.get("venue", "")),
            "scene": r.get("scene", ""), "round": r.get("round", 0),
            "utt_id": r.get("utt_id", ""), "text": r["text"],
            "heard_by": (heard_of.get(str(r.get("utt_id") or ""), []) if speech else []),
            "homes": {a: homes[a] for a in
                      (heard_of.get(str(r.get("utt_id") or ""), []) if speech else [])
                      if a in homes},
            "parcels": m["parcels"], "holders": m["holders"],
            "party_present": party_present(r),
        })
    return dict(timeline), excluded


def _party_lens(timeline, excluded) -> dict:
    out = {"basis": PARTY_BASIS, "caveat": PARTY_CAVEAT}
    rows = [e for v in timeline.values() for e in v]
    for color in ("green", "yellow"):
        got = [e for e in rows if e["color"] == color]
        with_party = len([e for e in got if e["party_present"]["present"]])
        out[color] = {
            "first_n": len(got), "with_party": with_party,
            "share": (round(with_party / len(got), 4) if got else None),
            "excluded_articles": len([e for e in excluded if e["color"] == color]),
        }
    return out


def _ignition_empty() -> dict:
    return {"green": {"strict": None, "loose": None},
            "yellow": {"strict": None, "loose": None},
            "criteria": dict(IGNITION_CRITERIA)}


def _build_ignition(rows, utts, thoughts, traces, venue_labels,
                    holders_by_step, acquired_by_step, party_present=None) -> dict:
    utt_sorted = sorted(utts, key=_sort_key)
    rows_sorted = sorted(rows, key=_sort_key)
    out = {"criteria": dict(IGNITION_CRITERIA)}
    for color in ("green", "yellow"):
        out[color] = {"strict": None, "loose": None}
        picks = [("loose", next((r for r in rows_sorted
                                 if r.get("stage") == color), None), None)]
        strict_row, strict_matched = None, None
        for r in rows_sorted:
            m = _ignition_strict_hit(r, color, holders_by_step, acquired_by_step)
            if m is not None:
                strict_row, strict_matched = r, m
                break
        picks.append(("strict", strict_row, strict_matched))
        for slot, hit, matched in picks:
            if hit is None:
                continue
            out[color][slot] = _ignition_obj(color, hit, matched, utt_sorted,
                                             thoughts, traces, venue_labels)
            if slot == "strict" and party_present is not None:
                out[color][slot]["party_present"] = party_present(hit)
    return out


def _ignition_obj(color, hit, matched, utt_sorted, thoughts, traces,
                  venue_labels) -> dict:
    month, who = hit["step"], hit["from"]
    key = _sort_key(hit)
    same_place = [u for u in utt_sorted
                  if int(u.get("step") or 0) == month
                  and (u.get("scene") or "") == (hit.get("scene") or "")
                  and (u.get("venue") or "") == (hit.get("venue") or "")]
    if hit.get("kind") != "utterance" and not hit.get("venue"):
        same_place = []
    before = [u for u in same_place if _sort_key(u) < key][-5:]
    after = [u for u in same_place if _sort_key(u) > key][:2]
    heard = [u for u in utt_sorted
             if int(u.get("step") or 0) == month and _sort_key(u) < key
             and who in _heard_by(u)][-5:]
    own = [{"scene": t.get("scene", ""), "text": str(t.get("text", ""))}
           for t in sorted(thoughts, key=_sort_key)
           if t.get("from") == who and int(t.get("step") or 0) == month]
    tr = [{"kind": t.get("kind", ""), "parcel_id": t.get("parcel_id", ""),
           "acq_id": t.get("acq_id", ""), "text": str(t.get("text", ""))}
          for t in traces
          if t.get("agent_id") == who and int(t.get("step") or 0) == month]
    obj = {
        "color": color, "month": month, "scene": hit.get("scene", ""),
        "venue": hit.get("venue", ""),
        "venue_label": venue_labels.get(hit.get("venue", ""), hit.get("venue", "")),
        "from": who, "name": hit.get("name", ""), "role": hit.get("role", ""),
        "kind": hit.get("kind", ""), "utt_id": hit.get("utt_id", ""),
        "text": hit["text"],
        "rule": {"blue": bool(hit["rule_blue"]), "green": bool(hit["rule_green"]),
                 "yellow": bool(hit["rule_yellow"]), "red": bool(hit["rule_red"])},
        "llm": {"deal": hit["llm_deal"], "area": hit["llm_area"],
                "same_buyer": hit["llm_same_buyer"], "admin": hit["llm_admin"]},
        "context_before": [_ctx_row(u) for u in before],
        "context_after": [_ctx_row(u) for u in after],
        "heard_before": [_ctx_row(u) for u in heard],
        "own_thoughts_that_month": own,
        "traces_that_month": tr,
    }
    if matched is not None:
        obj["matched"] = matched
    return obj


def build(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    with open(os.path.join(run_dir, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    metrics_path = os.path.join(run_dir, "metrics_v5.json")
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    ledger = _load(run_dir, "ledger.jsonl")
    utts = _load(run_dir, "utterances_v5.jsonl")
    directs = _load(run_dir, "directs_v5.jsonl")
    articles = _load(run_dir, "articles_v5.jsonl")
    thoughts = _load(run_dir, "thoughts_all.jsonl")
    traces = _load(run_dir, "traces_v5.jsonl")
    plans = _load(run_dir, "plans_v5.jsonl")
    deals = _load(run_dir, "deals_v5.jsonl")
    classified = _load(run_dir, "utterances.jsonl")
    occ = _load(run_dir, "occupation_labels.jsonl")
    stage_labels = _load(run_dir, "stage_labels_v5c.jsonl")

    n_steps = int(summary.get("steps") or 0)
    version = str(cfg.get("scenario_version", "field_v5"))

    # --- 世界の形（区画） --------------------------------------------------
    world = cfg["world"]
    cols, rows = int(world["grid"]["cols"]), int(world["grid"]["rows"])
    blocks = list(world["block_names"])
    shop = {tuple(c) for c in world.get("shop_coords", [])}
    vacant = {tuple(c) for c in world.get("vacant_coords", [])}
    public = {tuple(c) for c in world.get("public_coords", [])}
    parcels = []
    idx = 0
    for y in range(rows):
        for x in range(cols):
            idx += 1
            bi = (y // (rows // 2)) * 2 + (x // (cols // 2))
            use = ("public" if (x, y) in public else
                   "shop" if (x, y) in shop else
                   "vacant" if (x, y) in vacant else "residential")
            parcels.append({"pid": f"P{idx:02d}", "x": x, "y": y,
                            "block": blocks[min(bi, len(blocks) - 1)], "use": use})

    # --- 台本が起こしたこと（登記／使い手） --------------------------------
    deal_note = {(d["parcel_id"], int(d["step"])): d for d in deals}
    events = []
    party_of = {}
    for r in ledger:
        if r.get("kind") not in ("transfer", "lease"):
            continue
        pid, month = r["parcel_id"], int(r["step"])
        party = r.get("seller") or r.get("lessor") or ""
        party_of.setdefault(pid, party)
        d = deal_note.get((pid, month), {})
        events.append({"acq_id": r.get("acq_id", ""), "parcel_id": pid, "month": month,
                       "holder": r.get("under_name", ""),
                       "kind": "lease" if r["kind"] == "lease" else "sale",
                       "party": party, "note": d.get("note", "")})
    events.sort(key=lambda e: (e["month"], e["parcel_id"]))
    acquired_month = {e["parcel_id"]: e["month"] for e in events}

    # --- 街が語った区画（3条件） -------------------------------------------
    holders_by_step, acquired_by_step = {}, {}
    hs, ps = set(), set()
    for step in range(1, n_steps + 1):
        for e in events:
            if e["month"] <= step:
                hs.add(e["holder"])
                ps.add(e["parcel_id"])
        holders_by_step[step] = set(hs)
        acquired_by_step[step] = set(ps)

    about = set()
    llm_ran = any("about_acquisition" in r for r in classified)
    for r in classified:
        if r.get("about_acquisition"):
            about.add((int(r.get("step", 0)), r.get("from", ""), str(r.get("text", ""))))

    said = ([{"ch": "utterance", "step": int(u["step"]), "from": u.get("from"),
              "name": u.get("name", ""), "scene": u.get("scene"), "venue": u.get("venue"),
              "utt_id": u.get("utt_id", ""), "text": str(u.get("text", ""))} for u in utts]
            + [{"ch": "direct", "step": int(d["step"]), "from": d.get("from"), "name": "",
                "scene": d.get("scene"), "venue": "", "utt_id": "",
                "text": str(d.get("text", ""))} for d in directs]
            + [{"ch": "article", "step": int(a["step"]), "from": a.get("from"), "name": "",
                "scene": a.get("scene"), "venue": "", "utt_id": "",
                "text": str(a.get("text", ""))} for a in articles])
    said.sort(key=lambda r: (r["step"], r["ch"], r["from"] or ""))

    # 事後LLM分類が掛かっているのは**発話だけ**（記事・私信には分類ラベルが無い）。
    # 仕様（impl §6.1）は「LLMが true とした行だけを数える」なので、
    # 右地図は**LLM分類を通った発話だけ**で塗り、記事・私信だけで名指しされた区画は
    # 別に数えて画面に出す（Codexレビュー 2026-08-28）。
    awareness = {}
    rule_only_parcels = {}
    for row in said:
        step = row["step"]
        hit = _v5_mentions(row["text"], holders_by_step.get(step, set()))
        if not hit["hit"]:
            continue
        if llm_ran and row["ch"] == "utterance"                 and (step, row["from"], row["text"]) not in about:
            continue          # LLM が「取得の話ではない」とした発話は数えない
        # 事後分類が掛かるのは発話だけ。記事・私信にはラベルが無いので、
        # 右地図には載せず「ルール抽出だけで名指しされた区画」として別に数える。
        confirmed = (row["ch"] == "utterance") or (not llm_ran)
        for pid in hit["parcels"]:
            if pid not in acquired_by_step.get(step, set()):
                continue          # その月までに成立していない取得は数えない
            if row["from"] == party_of.get(pid):
                continue          # 当事者本人は「街が語った」に数えない
            if not confirmed:
                r = rule_only_parcels.setdefault(pid, {"first_month": step,
                                                       "channels": []})
                r["first_month"] = min(r["first_month"], step)
                if row["ch"] not in r["channels"]:
                    r["channels"].append(row["ch"])
                continue
            a = awareness.setdefault(pid, {"first_month": step, "speakers_by_month": {},
                                           "speakers": []})
            a["first_month"] = min(a["first_month"], step)
            a["speakers_by_month"].setdefault(str(step), [])
            if row["from"] not in a["speakers_by_month"][str(step)]:
                a["speakers_by_month"][str(step)].append(row["from"])
            if row["from"] not in a["speakers"]:
                a["speakers"].append(row["from"])
    for pid, a in awareness.items():
        a["speakers_total"] = len(a["speakers"])
        cum, series = set(), {}
        for step in range(1, n_steps + 1):
            cum.update(a["speakers_by_month"].get(str(step), []))
            series[str(step)] = len(cum)
        a["cumulative_speakers"] = series
        del a["speakers_by_month"]

    silent = sorted(pid for pid in acquired_month if pid not in awareness)
    rule_only_only = sorted(pid for pid in rule_only_parcels if pid not in awareness)

    # --- v5b: O1（連結）が出た月を区画ごとに --------------------------------
    # 地図の紫は「公に口にされた連結」だけにする（内心だけの連結で塗ると
    #「街が語った」と読めてしまう）。内心を含む初出は別キーで持つ。
    o1_month, o1_month_any = {}, {}
    if occ:
        for r in sorted(occ, key=lambda z: int(z.get("step", 0))):
            if not r.get("links_multiple"):
                continue
            step = int(r.get("step", 0))
            text = str(r.get("text", ""))
            hit = _v5_mentions(text, holders_by_step.get(step, set()))
            named = [p for p in hit["parcels"] if p in acquired_by_step.get(step, set())]
            if not (len(named) >= 2 or (len(hit["holders"]) >= 2 and named)):
                continue          # ルール∧LLM の連結だけ
            public = r.get("kind") in ("utterance", "article")
            for pid in named:
                o1_month_any.setdefault(pid, step)
                if public:
                    o1_month.setdefault(pid, step)

    # --- v5c: 4段階の色（青／緑／黄／赤） -----------------------------------
    # 地図に塗るのは **公の場（発話・記事）で、当事者以外が、その月までに成立した取得を
    # 名指しした行** だけ（v5b の右地図と同じ3条件）。色は run_metrics と同じ
    # 「ルール1次抽出 ∧ LLM ラベル」で、区画ごとにその時点までの最高段階を採る。
    stage_by_parcel, stage_series, stage_stats, stage_rows = {}, {}, {}, []
    if stage_labels:
        role_of = {}
        for r in utts + thoughts:
            if r.get("from") and r.get("role"):
                role_of[r["from"]] = r["role"]
        rank = {c: i for i, c in enumerate(V5C_COLORS)}
        cum = {c: set() for c in V5C_COLORS}
        cum_priv = {c: set() for c in V5C_COLORS}
        series = {c: {} for c in V5C_COLORS}
        series_priv = {c: {} for c in V5C_COLORS}
        rows = []
        for r in stage_labels:
            step = int(r.get("step", 0))
            text = str(r.get("text", ""))
            role = r.get("role") or role_of.get(r.get("from"), "")
            hs = holders_by_step.get(step, set())
            ps = acquired_by_step.get(step, set())
            row = {
                "step": step, "from": r.get("from", ""), "kind": r.get("kind", ""),
                "text": text, "venue": r.get("venue", ""), "scene": r.get("scene", ""),
                "round": r.get("round", 0), "utt_id": r.get("utt_id", ""),
                "name": r.get("name", ""), "role": role,
                "classified": bool(r.get("classified", True)) and r.get("deal") is not None,
                "rule_blue": _v5c_rule_blue(text, hs, ps),
                "rule_green": _v5c_rule_green(text, hs, ps),
                "rule_yellow": _v5c_rule_yellow(text, hs, ps),
                "rule_red": _v5c_rule_red(role, text),
                "llm_deal": bool(r.get("deal")) if r.get("deal") is not None else None,
                "llm_area": bool(r.get("area")) if r.get("area") is not None else None,
                "llm_same_buyer": (bool(r.get("same_buyer"))
                                   if r.get("same_buyer") is not None else None),
                "llm_admin": bool(r.get("admin")) if r.get("admin") is not None else None,
            }
            row["stage"] = _v5c_stage(row)
            rows.append(row)
        for step in range(1, n_steps + 1):
            for row in rows:
                if row["step"] != step or not row["stage"]:
                    continue
                if row["kind"] in ("utterance", "article"):
                    cum[row["stage"]].add(row["from"])
                else:
                    cum_priv[row["stage"]].add(row["from"])
            for c in V5C_COLORS:
                series[c][str(step)] = len(cum[c])
                series_priv[c][str(step)] = len(cum_priv[c] - cum[c])
        for row in sorted(rows, key=lambda r: r["step"]):
            if not row["stage"] or row["kind"] not in ("utterance", "article"):
                continue
            for pid in {p for p in V5_PARCEL_RE.findall(row["text"])
                        if p in acquired_by_step.get(row["step"], set())}:
                if row["from"] == party_of.get(pid):
                    continue
                cur = stage_by_parcel.get(pid)
                if cur is None or rank[row["stage"]] > rank[cur["color"]]:
                    stage_by_parcel[pid] = {"color": row["stage"], "month": row["step"],
                                            "from": row["from"], "kind": row["kind"],
                                            "name": row["name"], "scene": row["scene"],
                                            "venue": row["venue"],
                                            "utt_id": row["utt_id"],
                                            "text": row["text"]}
        stage_series = {"public": series, "private_only": series_priv}
        stage_stats = {
            "rows_by_color": {c: len([r for r in rows if r["stage"] == c])
                              for c in V5C_COLORS},
            "public_agents_final": {c: len(cum[c]) for c in V5C_COLORS},
            "private_only_final": {c: len(cum_priv[c] - cum[c]) for c in V5C_COLORS},
            "unknown": len([r for r in rows if not r["classified"]]),
            "first": {c: next(({"month": r["step"], "from": r["from"],
                                "kind": r["kind"], "venue": r["venue"],
                                "text": r["text"][:180]}
                               for r in sorted(rows, key=_sort_key)
                               if r["stage"] == c), None) for c in V5C_COLORS},
            "venue_first": {},
        }
        seen_pair = set()
        vf = {c: {} for c in V5C_COLORS}
        for r in sorted(rows, key=lambda z: z["step"]):
            if not r["stage"] or (r["from"], r["stage"]) in seen_pair:
                continue
            seen_pair.add((r["from"], r["stage"]))
            key = ("記事" if r["kind"] == "article"
                   else r["venue"] or ("自宅・計画" if r["scene"] == "plan" else "—"))
            vf[r["stage"]][key] = vf[r["stage"]].get(key, 0) + 1
        stage_stats["venue_first"] = vf
        # 4色で塗るときの「語られなかった区画」＝取得済みなのに一度も色が付かなかったもの。
        stage_stats["silent"] = sorted(pid for pid in acquired_month
                                       if pid not in stage_by_parcel)
        stage_stats["basis"] = ("公の場（発話・記事）で、当事者以外が、その月までに成立した"
                                "取得を名指しした行のうち、ルール1次抽出 ∧ LLM で色が付いたもの")
        stage_rows = rows

    # --- 兆候（誰にいつ見えたか）・沈黙の区画の突き合わせ用 -----------------
    trace_rows = [{"month": int(t.get("step", 0)), "agent_id": t.get("agent_id"),
                   "scene": t.get("scene", ""), "venue": t.get("venue", ""),
                   "kind": t.get("kind"), "parcel_id": t.get("parcel_id", ""),
                   "text": t.get("text", "")}
                  for t in traces if t.get("kind") != "registry_lookup"]
    by_parcel_traces = collections.defaultdict(list)
    for t in trace_rows:
        if t["parcel_id"]:
            by_parcel_traces[t["parcel_id"]].append(t)

    silent_detail = []
    utt_by_agent_month = collections.defaultdict(list)
    for u in utts:
        utt_by_agent_month[(u.get("from"), int(u["step"]))].append(u)
    for pid in silent:
        seen = sorted(by_parcel_traces.get(pid, []), key=lambda t: t["month"])
        rows = []
        for t in seen[:4]:
            others = utt_by_agent_month.get((t["agent_id"], t["month"]), [])
            rows.append({"month": t["month"], "agent_id": t["agent_id"],
                         "kind": t["kind"], "trace_text": t["text"],
                         "said_instead": (others[0]["text"] if others else "")})
        silent_detail.append({
            "parcel_id": pid, "acquired_month": acquired_month[pid],
            "followup_months": n_steps - acquired_month[pid],
            "party": party_of.get(pid, ""), "traces_seen": len(seen),
            "rows": rows,
        })

    # --- 主体の内心と発話（Dパネル） ---------------------------------------
    def timeline(agent_id):
        out = []
        for step in range(1, n_steps + 1):
            th = [t["text"] for t in thoughts
                  if t.get("from") == agent_id and int(t.get("step", 0)) == step]
            tx = [u["text"] for u in utts
                  if u.get("from") == agent_id and int(u["step"]) == step]
            out.append({"month": step, "thought": th[-1] if th else "",
                        "texts": tx[:2]})
        return out

    focus = []
    for e in events[:40]:
        if e["party"] and e["party"] not in focus:
            focus.append(e["party"])
    focus = focus[:3]
    agent_timeline = {a: timeline(a) for a in focus}
    agent_deals = collections.defaultdict(list)
    for e in events:
        if e["party"]:
            agent_deals[e["party"]].append(
                {"parcel_id": e["parcel_id"], "month": e["month"],
                 "holder": e["holder"], "kind": e["kind"], "note": e["note"]})

    # --- 一括照会の場面（Eパネル） -----------------------------------------
    # 「窓口の月（S4）」に限らない。run94 の記者の5区画照会は S1 の市役所の待合で
    # 起きていた（Codexレビュー 2026-08-28 で判明）。**1つの発話の中に、その月までに
    # 成立した取得区画が最も多く並んだ発話**を探し、その場面を再生する。
    best = None
    for u in utts:
        m = int(u["step"])
        named = {p for p in V5_PARCEL_RE.findall(u["text"])
                 if p in acquired_by_step.get(m, set())}
        if best is None or len(named) > best[0]:
            best = (len(named), u)
    scene_replay = {"month": None, "rows": [], "article": None}
    if best and best[0] >= 2:
        trigger = best[1]
        m = int(trigger["step"])
        scene = trigger.get("scene")
        venue = trigger.get("venue")
        venue_label = {v["id"]: v["label"]
                       for v in (cfg.get("social", {}).get("venues") or [])}.get(venue, venue)
        rows = sorted([u for u in utts if int(u["step"]) == m
                       and u.get("scene") == scene and u.get("venue") == venue],
                      key=lambda u: (u.get("round", 0), u.get("utt_id", "")))
        here = [p["agent_id"] for p in plans
                if int(p.get("step", 0)) == m and p.get(scene) == venue]
        scene_replay = {
            "month": m, "scene": scene, "venue": venue, "venue_label": venue_label,
            "widest_query": best[0],
            "trigger_utt_id": trigger.get("utt_id", ""),
            "attendees": sorted(here),
            "speakers": sorted({u.get("from") for u in rows}),
            # 窓口の月に登記を見に行った主体が閲覧できた「名義変更の記録」の件数
            "registry_records": len({t.get("acq_id") or t.get("parcel_id")
                                     for t in traces
                                     if t.get("kind") == "registry_lookup"
                                     and int(t.get("step", 0)) <= m}),
            "counter_month": _v5_s4_kind(m) == "counter",
            # 一括照会の発話は必ず含める（先頭10件で切れて落ちないように）
            "rows": [{"round": u.get("round"), "from": u.get("from"),
                      "name": u.get("name"), "utt_id": u.get("utt_id"),
                      "text": u["text"],
                      "trigger": u.get("utt_id") == trigger.get("utt_id")}
                     for u in (rows[:10]
                               if any(u.get("utt_id") == trigger.get("utt_id")
                                      for u in rows[:10])
                               else [trigger] + [u for u in rows[:9]
                                                 if u.get("utt_id")
                                                 != trigger.get("utt_id")])],
            "article": None,
        }
        nxt = [a for a in articles if int(a["step"]) == m]
        if nxt:
            scene_replay["article"] = {"month": int(nxt[0]["step"]),
                                       "from": nxt[0].get("from"),
                                       "text": nxt[0]["text"]}

    # --- 台本に無い対象と登記が結びつけられた話題（Fパネル） ---------------
    venue_labels = [v["label"] for v in (cfg.get("social", {}).get("venues") or [])]
    parcel_pids = {p["pid"] for p in parcels}
    invented = None
    for label in venue_labels:
        pat = re.compile(re.escape(label) + r"[^。]{0,12}登記|登記[^。]{0,12}"
                         + re.escape(label))
        u_hits = [u for u in utts if pat.search(u["text"])]
        t_hits = [t for t in thoughts if pat.search(str(t.get("text", "")))]
        if not u_hits:
            continue
        if invented is None or len(u_hits) > invented["utterances"]:
            first = min(u_hits, key=lambda u: (int(u["step"]), u.get("utt_id", "")))
            invented = {
                "term": label,
                "is_parcel": label in parcel_pids,      # 会場名は区画IDではない
                "utterances": len(u_hits), "thoughts": len(t_hits),
                "first_month": int(first["step"]), "last_month": max(int(u["step"])
                                                                    for u in u_hits),
                "speakers": len({u.get("from") for u in u_hits}),
                "quote": {"utt_id": first.get("utt_id", ""), "month": int(first["step"]),
                          "from": first.get("from"), "text": first["text"]},
            }
    public_transfers = len([e for e in events
                            if e["parcel_id"] in {p["pid"] for p in parcels
                                                  if p["use"] == "public"}])

    # --- 画面に出す数値はすべてここから ------------------------------------
    usage = summary.get("usage", {})
    cost = (usage.get("input_tokens", 0) / 1e6 * 0.10
            + usage.get("output_tokens", 0) / 1e6 * 0.40)
    stats = {
        "run": os.path.basename(run_dir), "version": version, "steps": n_steps,
        "model": summary.get("model"), "calls": usage.get("calls", 0),
        "cost_usd": round(cost, 4),
        "parcels_total": len(parcels),
        "parcels_tradable": summary.get("parcels_tradable",
                                        len([p for p in parcels
                                             if p["use"] != "public"])),
        "agents": sum(v for k, v in (summary.get("agents") or {}).items()
                      if k != "acquirer"),
        "deals": len(events),
        "sales": len([e for e in events if e["kind"] == "sale"]),
        "leases": len([e for e in events if e["kind"] == "lease"]),
        "ownership_share": metrics.get("final_acquirer_share"),
        "control_share": metrics.get("control_share"),
        "utterances": metrics.get("utterances_spoken"),
        "silence_rate": metrics.get("silence_rate"),
        "mean_group_size": metrics.get("mean_group_size"),
        "articles": metrics.get("articles"), "directs": metrics.get("directs"),
        "noticed": len(awareness), "silent": len(silent),
        "unnoticed_ratio_all": metrics.get("unnoticed_ratio_all"),
        "cohort_unnoticed_ratio": metrics.get("cohort_unnoticed_ratio"),
        "mean_detection_lag": metrics.get("mean_detection_lag_months"),
        "detection_lag_months": metrics.get("detection_lag_months"),
        "speakers_by_month": {k: len(v) for k, v in
                              (metrics.get("speakers_by_month") or {}).items()},
        "O1_count": metrics.get("O1_count"), "O1_public": metrics.get("O1_public_count"),
        "O2_count": metrics.get("O2_count"),
        "O3_public_agents": metrics.get("O3_public_agents_final"),
        "O3_private_only": metrics.get("O3_private_only_agents_final"),
        "O3_public_share": metrics.get("O3_public_share_final"),
        "O4_article_months": metrics.get("O4_article_months"),
        "O4_assembly_months": metrics.get("O4_assembly_months"),
        "O4_counter_months": metrics.get("O4_counter_months"),
        "max_token_finishes": summary.get("max_token_finishes", 0),
        "parse_fail": (metrics.get("D3_wiring") or {}).get("parse_fail", 0),
        "misdelivered": (metrics.get("D3_wiring") or {}).get("utterances_misdelivered", 0),
        "api_errors": (metrics.get("D3_wiring") or {}).get("api_errors", 0),
        "traces_by_kind": metrics.get("traces_by_kind"),
        "rule_only_parcels": len(rule_only_only),
        "public_transfers": public_transfers,
        "cost_note": "トークン数×公開単価からの推計（請求実績ではない）",
        # v5c（4段階の色）
        "C_blue": metrics.get("C_blue_rows"), "C_green": metrics.get("C_green_rows"),
        "C_yellow": metrics.get("C_yellow_rows"), "C_red": metrics.get("C_red_rows"),
        # 「到達した最高段階」は排他的な状態で出す（一度でも達した人数は別値）。
        "C_state_public": (metrics.get("C_state_public_final")
                           if metrics.get("C_available") else None),
        "C_ever_public_agents": ({c: metrics.get("C_%s_ever_public_agents_final" % c)
                                  for c in V5C_COLORS}
                                 if metrics.get("C_available") else None),
        "C_unknown": metrics.get("C_unknown"),
        "venues": len((cfg.get("social", {}) or {}).get("venues") or []),
    }

    # 公開済み metrics との突き合わせ（画面の数字が集計と食い違わないことの担保）
    # metrics_v5 の first_mention は「登記が動いた取得（sale）」だけを追う。
    # 地図は賃借も含めた28件を塗るので、**売買だけの部分集合**で突き合わせる。
    # metrics_v5 の first_mention は「登記が動いた取得（sale）」だけを追い、
    # 記事・私信もルール抽出だけで数える。右地図は**発話のLLM確定分だけ**なので、
    # 両者は一致しない。ここでは metrics と同じ基準でも組み直し、
    # **区画IDと初出月まで**突き合わせる（件数だけの照合にしない）。
    sale_pids = {e["parcel_id"] for e in events if e["kind"] == "sale"}
    metrics_first = {v["parcel_id"]: v["month"]
                     for v in (metrics.get("first_mention") or {}).values()}
    loose = {}
    for pid, a in awareness.items():
        loose[pid] = a["first_month"]
    for pid, r in rule_only_parcels.items():
        loose[pid] = min(loose.get(pid, 10 ** 6), r["first_month"])
    loose_sale = {p: m for p, m in loose.items() if p in sale_pids}
    checks = {
        "loose_basis_matches_metrics": loose_sale == metrics_first,
        "deals_matches_metrics": (len(events) == (metrics.get("deals")
                                                  or metrics.get("acquisitions"))),
        "map_basis": "LLM分類を通った発話のみ（当事者以外・その月までに成立）",
        "metrics_basis": ("記事・私信はルール抽出のみ・売買"
                          + str(len([e for e in events if e["kind"] == "sale"]))
                          + "件のみ"),
        "noticed_map": len(awareness),
        "silent_map": len(silent),
        "noticed_metrics_basis_sale": len(loose_sale),
        "rule_only_parcels": len(rule_only_only),
    }

    venue_label_map = {v["id"]: v["label"]
                       for v in ((cfg.get("social", {}) or {}).get("venues") or [])}
    layout = _build_layout(cfg, {p["pid"] for p in parcels})
    naming = _parcel_naming(layout, utts, thoughts, articles)
    labels = _agent_labels() if layout else {}
    homes = _agent_homes() if layout else {}
    party_present = _party_lookup(plans, events, labels)
    ignition = (_build_ignition(stage_rows, utts, thoughts, traces, venue_label_map,
                                holders_by_step, acquired_by_step, party_present)
                if stage_rows else _ignition_empty())
    if stage_rows:
        timeline, excluded = _ignition_timeline(
            stage_rows, utts, venue_label_map, labels, homes,
            holders_by_step, acquired_by_step, party_present)
        lens = _party_lens(timeline, excluded)
    else:
        timeline, excluded, lens = {}, [], None

    return {
        "meta": {"generated_from": os.path.basename(run_dir),
                 "generator": "tools/build_present_data.py",
                 "schema": 7,
                 "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
                 "note": "画面の数値・塗りはこのJSONだけから描く（手打ち禁止）"},
        "grid": {"cols": cols, "rows": rows, "blocks": blocks, "parcels": parcels},
        "events": events,
        "awareness": awareness,
        "o1_month": o1_month,
        "o1_month_any": o1_month_any,
        "rule_only_parcels": rule_only_parcels,
        "invented_link": invented,
        "silent": silent,
        "silent_detail": silent_detail,
        "agent_timeline": agent_timeline,
        "agent_deals": {k: v for k, v in agent_deals.items() if k in agent_timeline},
        "scene_replay": scene_replay,
        "stage_by_parcel": stage_by_parcel,
        "stage_series": stage_series,
        "stage_stats": stage_stats,
        "prime_event": metrics.get("C_prime_event"),
        "layout": layout,
        "parcel_naming": naming,
        "ignition": ignition,
        "ignition_timeline": timeline,
        "ignition_timeline_excluded": excluded,
        "ignition_timeline_note": ("月 → その月に初めて緑／黄に達した出来事（人ごとの初到達）。"
                                   "speech は同席者に届いた発話、thought は誰にも伝わっていない内心。"
                                   "記事は2種のどちらでもないので ignition_timeline_excluded に分けた"),
        "party_lens": lens,
        "venue_labels": venue_label_map,
        "stats": stats,
        "checks": checks,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--out-dir", default=r"C:\Users\user\projects\quiet-acquisition-pages")
    ap.add_argument("--history", action="store_true",
                    help="v4系の実測（Gパネル用）を present_data_history.json に焼く")
    args = ap.parse_args()
    if args.history:
        hist = build_history(os.path.join(ROOT, "simulations"))
        out = os.path.join(args.out_dir, "present_data_history.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(hist, f, ensure_ascii=False, separators=(",", ":"))
        print("[out] " + out)
        for p in hist["phases"]:
            print("     ", p["label"],
                  [(r["transfers"], r["ownership_share"]) for r in p["runs"]])
        return 0
    data = build(args.run)
    hard = [k for k in ("loose_basis_matches_metrics", "deals_matches_metrics")
            if not data["checks"].get(k)]
    if hard:
        raise SystemExit("集計との突き合わせに失敗: " + ", ".join(hard))
    out = os.path.join(args.out_dir, f"present_data_{args.label}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    s = data["stats"]
    print(f"[out] {out}  {os.path.getsize(out) // 1024} KB")
    print(f"      deals={s['deals']} noticed={s['noticed']} silent={s['silent']} "
          f"O1={s['O1_count']} O2={s['O2_count']} checks={data['checks']}")
    return 0




# ---------------------------------------------------------------------------
# G パネル（引き算の物語 v4→v5）用：過去のランから実測値を拾う
# ---------------------------------------------------------------------------

HISTORY_RUNS = [
    ("v4.1", "金額を無くし、打診と応答だけ残した世界",
     ["2026-08-27_1715_72_field_v4_1_a_city",
      "2026-08-27_1717_73_field_v4_1_a_city",
      "2026-08-27_1720_74_field_v4_1_a_city"]),
    ("v4.1b", "そこへ「仲介に相談できる」経路を足した世界",
     ["2026-08-27_1840_81_field_v4_1b_runA",
      "2026-08-27_1843_82_field_v4_1b_runB",
      "2026-08-27_1846_83_field_v4_1b_runC"]),
]


def build_history(sim_root: str) -> dict:
    out = []
    for label, note, dirs in HISTORY_RUNS:
        rows = []
        for d in dirs:
            path = os.path.join(sim_root, d)
            if not os.path.exists(os.path.join(path, "summary.json")):
                continue
            with open(os.path.join(path, "summary.json"), encoding="utf-8") as f:
                s = json.load(f)
            ledger = _read_jsonl(os.path.join(path, "ledger.jsonl"))
            rows.append({
                "run": d,
                "steps": s.get("steps"),
                "transfers": len([r for r in ledger if r.get("kind") == "transfer"]),
                "ownership_share": (s.get("kpi") or {}).get("final_acquirer_share"),
            })
        out.append({"label": label, "note": note, "runs": rows})
    return {"meta": {"generator": "tools/build_present_data.py --history",
                     "note": "数値は各ランの ledger.jsonl / summary.json の実測"},
            "phases": out}


if __name__ == "__main__":
    raise SystemExit(main())
