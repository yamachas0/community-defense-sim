#!/usr/bin/env python
"""present.html の検収（Playwright・GPU無効）。

    python tests/test_present.py

やること：
  ・ローカルに Pages 作業ツリーを配信して 1280 と 390 で開く
  ・console エラー0・横スクロール0
  ・画面に出ている数値が present_data_*.json と一致するか
  ・自動再生が最後の月まで進むか／run 切替でデータが変わるか
  ・スクリーンショットを docs/shots_present/ に保存
外部ネットワークには出ない。ブラウザは必ず閉じる。
"""

from __future__ import annotations

import collections
import http.server
import json
import os
import math
import re
import socketserver
import sys
import threading
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES = r"C:\Users\user\projects\quiet-acquisition-pages"
SHOTS = os.path.join(ROOT, "docs", "shots_present")
SHOTS2 = os.path.join(ROOT, "docs", "shots_present2")
SHOTS3 = os.path.join(ROOT, "docs", "shots_present3")
PASS = FAIL = 0
SCENE_ORDER = {"plan": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4}


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=PAGES, **kw)

    def log_message(self, *a):
        pass


def serve():
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", 8731), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def _jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def json_checks(dC):
    """schema 7 の layout / ignition（strict・loose）/ parcel_naming を JSON だけで検査する。"""
    run_dir = os.path.join(ROOT, "simulations", dC["meta"]["generated_from"])

    # --- layout -----------------------------------------------------------
    L = dC.get("layout")
    check("layout: v5c の JSON に layout がある", bool(L))
    if not L:
        return
    check("layout: view が 1400x900", L["view"] == {"w": 1400, "h": 900}, str(L["view"]))
    check("layout: 出所と種が記録されている",
          L["source"] == "configs/parcels_v5c.yaml" and L["seed"] == 8501)
    check("layout: 全区画ぶんのマスがある",
          set(L["cells"]) == {p["pid"] for p in dC["grid"]["parcels"]},
          str(sorted(set(L["cells"]) ^ {p["pid"] for p in dC["grid"]["parcels"]})[:4]))
    need = ("cx", "cy", "side", "zone", "area_sqm", "size_class", "use",
            "use_detail", "owner_name", "frontage", "name", "locality",
            "use_word", "owner_label")
    check("layout: 各マスに必要な項目がそろっている",
          all(all(k in c for k in need) for c in L["cells"].values()))
    order = sorted(L["cells"].items(), key=lambda kv: (kv[1]["area_sqm"], kv[0]))
    sides = [c["side"] for _, c in order]
    check("layout: マスの大きさが面積の順序と矛盾しない",
          all(sides[i] <= sides[i + 1] for i in range(len(sides) - 1)), str(sides))
    amin = min(c["area_sqm"] for c in L["cells"].values())
    bad_side = [pid for pid, c in L["cells"].items()
                if abs(c["side"] - 34.0 * math.sqrt(c["area_sqm"] / amin)) > 0.01]
    check("layout: 辺が面積の平方根に比例している（34.0 × √(面積/最小面積））",
          not bad_side, str(bad_side[:4]))
    ck = L["checks"]
    check("layout: 辺の比が実測 2.53〜2.55（sqrt 比例の帰結）",
          2.53 <= ck["side_ratio"] <= 2.55, str(ck["side_ratio"]))
    check("layout: 描画面積の比が辺の比の2乗と一致",
          abs(ck["draw_area_ratio"] - ck["side_ratio"] ** 2) < 0.01,
          str((ck["draw_area_ratio"], ck["side_ratio"])))
    check("layout: side_min / side_max が実測どおり",
          abs(ck["side_min"] - min(sides)) < 0.01
          and abs(ck["side_max"] - max(sides)) < 0.01, str((ck["side_min"], ck["side_max"])))
    check("layout: マスの重なりの最大が 1.0 以下",
          L["checks"]["max_overlap"] <= 1.0, str(L["checks"]["max_overlap"]))
    ov = 0.0
    items = list(L["cells"].items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            need_d = (a["side"] + b["side"]) / 2.0
            ox = need_d - abs(b["cx"] - a["cx"])
            oy = need_d - abs(b["cy"] - a["cy"])
            if ox > 0 and oy > 0:
                ov = max(ov, min(ox, oy))
    check("layout: 座標から測り直しても重なりが 1.0 以下", ov <= 1.0, "%.3f" % ov)
    mr = L["checks"]["mean_radius_by_zone"]
    check("layout: 平均半径が 中心 < 中間 < 郊外",
          mr["中心"] < mr["中間"] < mr["郊外"], str(mr))
    for c in L["cells"].values():
        if not (c["side"] / 2 <= c["cx"] <= L["view"]["w"] - c["side"] / 2
                and c["side"] / 2 <= c["cy"] <= L["view"]["h"] - c["side"] / 2):
            check("layout: 全マスがビューに収まる", False, str(c))
            break
    else:
        check("layout: 全マスがビューに収まる", True)

    # --- 呼び名（parcel_names_v5c.yaml） --------------------------------
    with open(os.path.join(ROOT, "configs", "parcel_names_v5c.yaml"),
              encoding="utf-8") as f:
        NAMES = yaml.safe_load(f)
    check("naming: 出所と件数が JSON に書いてある",
          L.get("naming") == {"source": "configs/parcel_names_v5c.yaml",
                              "n": len(L["cells"])}, str(L.get("naming")))
    check("naming: 全48区画に名前がある",
          len(L["cells"]) == 48
          and all(c["name"] and c["locality"] and c["use_word"]
                  for c in L["cells"].values()))
    names = [c["name"] for c in L["cells"].values()]
    locs = [c["locality"] for c in L["cells"].values()]
    check("naming: 名前が全区画で一意", len(set(names)) == len(names))
    check("naming: locality が全区画で一意", len(set(locs)) == len(locs))
    check("naming: 名前が呼び名の表と1文字も違わない",
          all(L["cells"][p]["name"] == NAMES["land"][p]["name"]
              for p in L["cells"]))
    check("naming: owner_label は初期所有者（parcels_v5c の owner_name）の呼び名",
          all(c["owner_label"] == NAMES["registered_names"].get(c["owner_name"],
                                                                c["owner_name"])
              for c in L["cells"].values()))

    # --- 会場（venues_v5c.yaml・装飾） ----------------------------------
    with open(os.path.join(ROOT, "configs", "venues_v5c.yaml"),
              encoding="utf-8") as f:
        VEN = yaml.safe_load(f)["venues"]
    vs = L.get("venues") or []
    check("venues: 15か所ぶんの座標がある", len(vs) == 15 and len(VEN) == 15,
          str(len(vs)))
    check("venues: id と呼び名と1文字のバッジが表のとおり",
          [(v["id"], v["label"], v["badge"]) for v in vs]
          == [(v["id"], v["label"], v["badge"]) for v in VEN])
    check("venues: 表示位置が装飾だと JSON に書いてある",
          "装飾" in str(L.get("venues_note")) and "判定には使わない" in str(L.get("venues_note")),
          str(L.get("venues_note")))
    worst_v = 1e9
    for i, v in enumerate(vs):
        for c in L["cells"].values():
            dx = max(abs(v["cx"] - c["cx"]) - c["side"] / 2, 0.0)
            dy = max(abs(v["cy"] - c["cy"]) - c["side"] / 2, 0.0)
            worst_v = min(worst_v, math.hypot(dx, dy) - v["r"])
        for w in vs[i + 1:]:
            worst_v = min(worst_v, math.hypot(v["cx"] - w["cx"], v["cy"] - w["cy"])
                          - v["r"] - w["r"])
    check("venues: 区画とも会場どうしとも重ならない（隙間 6 以上）",
          worst_v >= 5.99, "%.3f" % worst_v)
    check("venues: 全部がビューに収まる",
          all(v["r"] <= v["cx"] <= L["view"]["w"] - v["r"]
              and v["r"] <= v["cy"] <= L["view"]["h"] - v["r"] for v in vs))

    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import build_present_data as bpd
    l1 = bpd.build(run_dir)["layout"]
    l2 = bpd.build(run_dir)["layout"]
    check("layout: 2回焼いても完全に同じ（決定論）", l1 == l2 and l1 == L)

    # --- parcel_naming ----------------------------------------------------
    pn = dC.get("parcel_naming")
    check("parcel_naming: ある", bool(pn))
    if pn:
        check("parcel_naming: 固有名は 0 件",
              pn["proper_names"] == 0 and "固有名は無い" in pn["note"])
        check("parcel_naming: 全区画ぶんの実測がある",
              set(pn["mentions"]) == set(L["cells"])
              and set(pn["name_in_speech"]) == set(L["cells"]))
        check("parcel_naming: 名前が呼ばれた回数の合計が 0 より大きい",
              sum(pn["mentions"].values()) > 0, str(sum(pn["mentions"].values())))
        check("parcel_naming: name_in_speech が mentions と整合",
              all(pn["name_in_speech"][p] == (pn["mentions"][p] > 0) for p in pn["mentions"]))

    # --- stage_by_parcel（原文を切っていない） ----------------------------
    src_text = {}
    for r in _jsonl(os.path.join(run_dir, "stage_labels_v5c.jsonl")):
        src_text.setdefault((int(r["step"]), r.get("from", ""), r.get("utt_id", ""),
                             r.get("kind", "")), []).append(str(r.get("text", "")))
    bad, nokey = [], []
    for pid, v in dC["stage_by_parcel"].items():
        key = (v["month"], v["from"], v.get("utt_id", ""), v["kind"])
        if key not in src_text:
            nokey.append(pid)
        elif v["text"] not in src_text[key]:
            bad.append(pid)
    check("stage_by_parcel: 元ログに該当行が必ずある（照合が空振りしない）",
          not nokey, str(nokey[:4]))
    check("stage_by_parcel: 原文が元ログと完全一致（160字で切れていない）",
          not bad, str(bad[:4]))
    check("stage_by_parcel: 場・発言者の情報が付いている",
          all(all(k in v for k in ("venue", "scene", "utt_id", "name"))
              for v in dC["stage_by_parcel"].values()))

    # --- ignition ---------------------------------------------------------
    ig = dC.get("ignition")
    check("ignition: ある", isinstance(ig, dict)
          and set(ig) == {"green", "yellow", "criteria"}, str(type(ig)))
    if not isinstance(ig, dict):
        return
    check("ignition: criteria が loose / strict_green / strict_yellow の3本",
          isinstance(ig.get("criteria"), dict)
          and set(ig["criteria"]) == {"loose", "strict_green", "strict_yellow"}
          and all(isinstance(v, str) and v for v in ig["criteria"].values()),
          str(ig.get("criteria")))
    utts = _jsonl(os.path.join(run_dir, "utterances_v5.jsonl"))
    utt_by_id = {u.get("utt_id"): u for u in utts}
    thoughts_src = _jsonl(os.path.join(run_dir, "thoughts_all.jsonl"))
    traces_src = _jsonl(os.path.join(run_dir, "traces_v5.jsonl"))

    def okey(o):
        u = utt_by_id.get(o["utt_id"]) or {}
        return (int(o["month"]), SCENE_ORDER.get(o["scene"], 9),
                int(u.get("round") or 0), str(o["utt_id"] or ""), str(o["from"] or ""))

    for color in ("green", "yellow"):
        pair = ig[color]
        check("ignition[%s]: strict と loose の2本立てになっている" % color,
              isinstance(pair, dict) and set(pair) == {"strict", "loose"}, str(pair))
        if not isinstance(pair, dict):
            continue
        if pair["strict"] is not None and pair["loose"] is not None:
            check("ignition[%s]: strict は loose と同じか後の行" % color,
                  okey(pair["strict"]) >= okey(pair["loose"]),
                  str((okey(pair["strict"]), okey(pair["loose"]))))
        for slot in ("strict", "loose"):
            o = pair[slot]
            tag = "ignition[%s.%s]" % (color, slot)
            if o is None:
                check("%s: 出なかった本として null" % tag, True)
                continue
            need = ("color", "month", "scene", "venue", "venue_label", "from", "name",
                    "role", "kind", "utt_id", "text", "rule", "llm", "context_before",
                    "context_after", "heard_before", "own_thoughts_that_month",
                    "traces_that_month")
            check("%s: 項目がそろっている" % tag,
                  all(k in o for k in need),
                  str([k for k in need if k not in o]))
            key = (o["month"], o["from"], o["utt_id"], o["kind"])
            check("%s: 元ログに該当行がある（照合が空振りしない）" % tag,
                  key in src_text, str(key))
            check("%s: 原文が元ログと完全一致（切り詰めていない）" % tag,
                  o["text"] in src_text.get(key, []), str(key))
            own = [str(t.get("text", "")) for t in thoughts_src
                   if t.get("from") == o["from"] and int(t.get("step", 0)) == o["month"]]
            check("%s: 本人の内心が元ログの全行と完全一致" % tag,
                  [t["text"] for t in o["own_thoughts_that_month"]] == own,
                  str(len(o["own_thoughts_that_month"])) + " vs " + str(len(own)))
            tr = [str(t.get("text", "")) for t in traces_src
                  if t.get("agent_id") == o["from"] and int(t.get("step", 0)) == o["month"]]
            check("%s: その月の兆候が元ログの全行と完全一致（登記照会も含む）" % tag,
                  [t["text"] for t in o["traces_that_month"]] == tr,
                  str(len(o["traces_that_month"])) + " vs " + str(len(tr)))
            check("%s: 直前の発言は5件以下" % tag, len(o["context_before"]) <= 5)
            check("%s: 直後の反応は2件以下" % tag, len(o["context_after"]) <= 2)
            same = {(r["step"], r["scene"], r["venue"])
                    for r in o["context_before"] + o["context_after"]}
            check("%s: 前後の発言は同じ月・同じ場面・同じ場" % tag,
                  not same or same == {(o["month"], o["scene"], o["venue"])}, str(same))
            check("%s: 聞いていた発言は5件以下" % tag,
                  len(o["heard_before"]) <= 5)
            miss = [r["utt_id"] for r in o["heard_before"]
                    if o["from"] not in (utt_by_id.get(r["utt_id"], {}).get("heard_by") or [])]
            check("%s: 聞いていた発言に本人が同席している" % tag,
                  not miss, str(miss[:3]))
            check("%s: 前後・聞いた発言の原文が元ログと一致" % tag,
                  all(utt_by_id.get(r["utt_id"], {}).get("text") == r["text"]
                      for r in o["context_before"] + o["context_after"] + o["heard_before"]))
            if slot == "loose":
                check("%s: 色が rule ∧ llm で成立している" % tag,
                      o["rule"][color]
                      and o["llm"]["area" if color == "green" else "same_buyer"],
                      str((o["rule"], o["llm"])))
                check("%s: matched を持たない" % tag, "matched" not in o)
            else:
                m = o.get("matched")
                check("%s: matched がある（並んだものを出せる）" % tag,
                      isinstance(m, dict) and set(m) == {"parcels", "holders"}, str(m))
                if isinstance(m, dict) and set(m) == {"parcels", "holders"}:
                    check("%s: 区画IDか名義のどちらかが2件以上" % tag,
                          len(m["parcels"]) >= 2 or len(m["holders"]) >= 2, str(m))
                    check("%s: matched はソート済み" % tag,
                          m["parcels"] == sorted(m["parcels"])
                          and m["holders"] == sorted(m["holders"]), str(m))
                    miss2 = [x for x in m["parcels"] + m["holders"] if x not in o["text"]]
                    check("%s: matched は本文に実在する" % tag, not miss2, str(miss2))
                check("%s: LLM ラベルが立っている" % tag,
                      bool(o["llm"]["area" if color == "green" else "same_buyer"]),
                      str(o["llm"]))

    # --- ignition_timeline / party_lens（第3弾） --------------------------
    tl = dC.get("ignition_timeline")
    lens = dC.get("party_lens")
    excl = dC.get("ignition_timeline_excluded")
    steps = dC["stats"]["steps"]
    check("ignition_timeline: 月をキーにした辞書がある", isinstance(tl, dict) and bool(tl))
    check("ignition_timeline: 除外した記事の一覧がある", isinstance(excl, list))
    rows_tl = [e for m in (tl or {}) for e in tl[m]]
    check("ignition_timeline: 月キーが 1〜%d の範囲" % steps,
          all(1 <= int(m) <= steps for m in (tl or {})), str(sorted(tl or {})[:3]))
    check("ignition_timeline: 要素の月がキーと一致",
          all(e["month"] == int(m) for m in (tl or {}) for e in tl[m]))
    need_tl = ("month", "color", "kind", "agent", "agent_label", "venue", "venue_label",
               "scene", "round", "utt_id", "text", "heard_by", "parcels", "holders",
               "party_present")
    check("ignition_timeline: 項目がそろっている",
          all(all(k in e for k in need_tl) for e in rows_tl),
          str([k for k in need_tl if rows_tl and k not in rows_tl[0]]))
    check("ignition_timeline: 色は緑か黄だけ",
          all(e["color"] in ("green", "yellow") for e in rows_tl))
    check("ignition_timeline: 種別は発話か内心だけ",
          all(e["kind"] in ("speech", "thought") for e in rows_tl))
    pairs = [(e["agent"], e["color"]) for e in rows_tl]
    check("ignition_timeline: 人ごとの初到達は1回だけ（重複しない）",
          len(set(pairs)) == len(pairs), str(len(pairs) - len(set(pairs))))
    check("ignition_timeline: 呼び名が内部IDのまま残っていない",
          all(e["agent_label"] != e["agent"] for e in rows_tl),
          str([e["agent"] for e in rows_tl if e["agent_label"] == e["agent"]][:3]))
    miss_src = [e["utt_id"] for e in rows_tl
                if (e["month"], e["agent"], e["utt_id"],
                    "utterance" if e["kind"] == "speech" else "thought") not in src_text]
    check("ignition_timeline: 元ログに該当行がある", not miss_src, str(miss_src[:3]))
    bad_text = [e["utt_id"] for e in rows_tl
                if e["text"] not in src_text.get(
                    (e["month"], e["agent"], e["utt_id"],
                     "utterance" if e["kind"] == "speech" else "thought"), [])]
    check("ignition_timeline: 原文が元ログと完全一致", not bad_text, str(bad_text[:3]))
    bad_heard = [e["utt_id"] for e in rows_tl
                 if (e["kind"] == "speech"
                     and e["heard_by"] != (utt_by_id.get(e["utt_id"], {}).get("heard_by") or []))
                 or (e["kind"] == "thought" and e["heard_by"])]
    check("ignition_timeline: 同席者はデータにある heard_by そのもの（内心は空）",
          not bad_heard, str(bad_heard[:3]))
    homes = {a: v["home"] for a, v in
             yaml.safe_load(open(os.path.join(ROOT, "configs", "parcel_names_v5c.yaml"),
                                 encoding="utf-8"))["agents"].items() if v.get("home")}
    bad_home = [e["utt_id"] for e in rows_tl
                if any(k not in e["heard_by"] or homes.get(k) != v
                       for k, v in (e.get("homes") or {}).items())]
    check("ignition_timeline: 本拠は同席者ぶんだけ・呼び名の表どおり",
          not bad_home, str(bad_home[:3]))

    led = _jsonl(os.path.join(run_dir, "ledger.jsonl"))
    deals = [{"month": int(r["step"]), "pid": r["parcel_id"],
              "party": r.get("seller") or r.get("lessor") or ""}
             for r in led if r.get("kind") in ("transfer", "lease")]
    bad_party = []
    for e in rows_tl + [ig[c]["strict"] for c in ("green", "yellow") if ig[c]["strict"]]:
        pp = e.get("party_present")
        if not isinstance(pp, dict) or set(pp) != {"present", "agents", "labels", "parcels"}:
            bad_party.append(("形", e.get("utt_id")))
            continue
        month = e["month"]
        parties = {d["party"] for d in deals if d["month"] <= month and d["party"]}
        if any(a not in parties for a in pp["agents"]):
            bad_party.append(("当事者でない", e.get("utt_id")))
        want = sorted({d["pid"] for d in deals
                       if d["month"] <= month and d["party"] in pp["agents"]})
        if pp["parcels"] != want or pp["present"] != bool(pp["agents"])                 or len(pp["labels"]) != len(pp["agents"]):
            bad_party.append(("中身", e.get("utt_id")))
    check("party_present: その月までに成立した取引の当事者だけを、区画つきで出している",
          not bad_party, str(bad_party[:3]))

    check("party_lens: 基準と但し書きがある",
          isinstance(lens, dict) and lens.get("basis") and lens.get("caveat"),
          str(lens if not isinstance(lens, dict) else list(lens)))
    for color in ("green", "yellow"):
        got = [e for e in rows_tl if e["color"] == color]
        w = len([e for e in got if e["party_present"]["present"]])
        share = round(w / len(got), 4) if got else None
        check("party_lens[%s]: 年表と数が一致" % color,
              lens[color] == {"first_n": len(got), "with_party": w, "share": share,
                              "excluded_articles": len([e for e in (excl or [])
                                                        if e["color"] == color])},
              str((lens[color], len(got), w, share)))


LABEL_JS = """() => {
  const svg = document.querySelector('#mapR svg');
  if (!svg) return {n: 0, nbad: 0, bad: [], fs: 0, mapw: 0};
  const vb = svg.viewBox.baseVal;
  const box = svg.getBoundingClientRect();
  const sc = box.width / vb.width;
  const ts = [...document.querySelectorAll('#mapR text.plabel, #mapR text.vlabel')];
  const rects = ts.map(t => { const r = t.getBoundingClientRect();
    return {t: t.textContent, x0: r.x, y0: r.y, x1: r.right, y1: r.bottom}; });
  const bad = [];
  for (let i = 0; i < rects.length; i++)
    for (let j = i + 1; j < rects.length; j++) {
      const a = rects[i], b = rects[j];
      const ox = Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0);
      const oy = Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0);
      if (ox > 0.5 && oy > 0.5) bad.push([a.t, b.t, +ox.toFixed(2), +oy.toFixed(2)]);
    }
  const fs = Math.min.apply(null, ts.map(t => parseFloat(getComputedStyle(t).fontSize) * sc));
  return {n: ts.length, nbad: bad.length, bad: bad.slice(0, 4),
          fs: +fs.toFixed(2), mapw: +box.width.toFixed(1)};
}"""


def label_checks(page, tag, min_map_w):
    """ラベルの重なり0と読める大きさを DOM から実測する（1280 と 390 の両方）。"""
    r = page.evaluate(LABEL_JS)
    check("present(%s): 地図の描画幅が %d px 以上" % (tag, min_map_w),
          r["mapw"] >= min_map_w, str(r["mapw"]))
    check("present(%s): 区画48＋会場15の名前が出ている" % tag, r["n"] == 63, str(r["n"]))
    check("present(%s): ラベルの重なりが 0" % tag, r["nbad"] == 0, str(r["bad"]))
    check("present(%s): ラベルの実効の文字が 11px 以上" % tag, r["fs"] >= 11.0, str(r["fs"]))
    return r


def open_numbers(page):
    """数表は details「詳しい数字」に畳んである＝読む前に開く。"""
    page.evaluate("() => document.querySelectorAll('details.numbers')"
                  ".forEach(d => { d.open = true; })")


def main() -> int:
    from playwright.sync_api import sync_playwright

    try:                       # 端末が cp932 でも検査名を潰さない
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    os.makedirs(SHOTS, exist_ok=True)
    os.makedirs(SHOTS2, exist_ok=True)
    os.makedirs(SHOTS3, exist_ok=True)
    with open(os.path.join(PAGES, "present_data_run94.json"), encoding="utf-8") as f:
        d94 = json.load(f)
    with open(os.path.join(PAGES, "present_data_v5bB.json"), encoding="utf-8") as f:
        dB = json.load(f)

    for lab in ("run94", "v5bA", "v5bB", "v5bC"):
        with open(os.path.join(PAGES, "present_data_%s.json" % lab), encoding="utf-8") as f:
            d = json.load(f)
        check("%s: 区画属性を持たない run は layout=null（従来の格子で描く）" % lab,
              d.get("layout") is None and d["meta"]["schema"] == 7,
              str(d["meta"]["schema"]))
        ig = d.get("ignition")
        check("%s: ignition は色の判定が無い run でも同じ形" % lab,
              isinstance(ig, dict) and set(ig) == {"green", "yellow", "criteria"}
              and all(ig[c] == {"strict": None, "loose": None}
                      for c in ("green", "yellow"))
              and set(ig["criteria"]) == {"loose", "strict_green",
                                          "strict_yellow"}, str(ig))
    for lab in ("v5cA", "v5cB", "v5cC"):
        with open(os.path.join(PAGES, "present_data_%s.json" % lab), encoding="utf-8") as f:
            json_checks(json.load(f))

    hits = []
    for rel in ("run.py", "tools/run_metrics.py", "tools/build_audit_v5c.py",
                "tools/build_parcels_v5c.py", "tools/build_events_v5c.py",
                "configs/config_field_v5c.yaml", "configs/config_field_v5c_econ.yaml"):
        path = os.path.join(ROOT, *rel.split("/"))
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                if "venues_v5c" in f.read():
                    hits.append(rel)
    for name in sorted(os.listdir(os.path.join(ROOT, "src"))):
        if not name.endswith(".py"):
            continue
        with open(os.path.join(ROOT, "src", name), encoding="utf-8") as f:
            if "venues_v5c" in f.read():
                hits.append("src/" + name)
    check("会場の表示位置は装飾＝集計・判定のコードは venues_v5c.yaml を読んでいない",
          not hits, str(hits))

    httpd = serve()
    base = "http://127.0.0.1:8731/present.html"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--disable-gpu"])
            try:
                errors = []
                page = browser.new_page(viewport={"width": 1280, "height": 900})
                page.on("console", lambda m: errors.append(m.text)
                        if m.type == "error" else None)
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.goto(base + "?run=run94", wait_until="networkidle")
                page.wait_for_selector("#mapL .cell", timeout=10000)
                open_numbers(page)
                page.click("[data-m='R']")          # 地図は1枚ずつ＝押す前に開く
                page.wait_for_timeout(150)

                check("present: console エラーが出ない", not errors, str(errors[:2]))
                w = page.evaluate("[document.documentElement.scrollWidth,"
                                  "document.documentElement.clientWidth]")
                check("present: 1280 で横スクロールが出ない", w[0] <= w[1], str(w))
                page.screenshot(path=os.path.join(SHOTS, "01_start_1280.png"),
                                full_page=True)

                s = d94["stats"]
                stats_text = page.inner_text("#stats")
                check("present: 取引の数が JSON と一致",
                      f"{s['deals']}（{s['sales']}／{s['leases']}）" in stats_text,
                      stats_text[:120])
                check("present: 語られた区画の数が JSON と一致",
                      str(s["noticed"]) in stats_text and str(s["silent"]) in stats_text)
                check("present: 実費が JSON と一致", f"${s['cost_usd']}" in stats_text)
                check("present: 突き合わせが真であることを画面に出す",
                      "一致（区画IDと初出月まで照合）" in page.inner_text("#checkline"))
                check("present: JSON の突き合わせフラグが真",
                      d94["checks"]["loose_basis_matches_metrics"]
                      and d94["checks"]["deals_matches_metrics"])
                check("present: 右地図の基準が画面に書いてある",
                      "LLM分類を通った発話" in page.inner_text("#checkline"))

                # 月送り：第0月は右地図が0
                check("present: 第0月は左右とも塗られていない",
                      "0 区画" in page.inner_text("#capL"))
                page.fill("#slider", "7")
                page.dispatch_event("#slider", "input")
                page.wait_for_timeout(150)
                capL7, capR7 = page.inner_text("#capL"), page.inner_text("#capR")
                left7 = len([e for e in d94["events"] if e["month"] <= 7])
                right7 = len([k for k, v in d94["awareness"].items()
                              if v["first_month"] <= 7])
                check("present: 第7月の左地図が台本と一致", f"{left7} 区画" in capL7, capL7)
                check("present: 第7月の右地図が集計と一致", f"{right7} 区画" in capR7, capR7)
                check("present: 第7月で左が右より進んでいる（時差が出る）",
                      int(page.inner_text("#gapnum")) == left7 - right7)
                page.screenshot(path=os.path.join(SHOTS, "02_month7_1280.png"),
                                full_page=True)

                # 自動再生が最後まで動く
                page.click("#play")
                page.wait_for_function(
                    "() => document.getElementById('monthlabel').textContent.includes('%d')"
                    % d94["stats"]["steps"], timeout=40000)
                page.wait_for_timeout(600)
                check("present: 自動再生が最後の月まで進む",
                      str(d94["stats"]["steps"]) in page.inner_text("#monthlabel"))
                silent_cells = page.eval_on_selector_all(
                    "#mapR .cell.silent", "els => els.map(e => e.dataset.pid)")
                check("present: 終了時に沈黙の区画が強調される",
                      sorted(silent_cells) == sorted(d94["silent"]),
                      f"{sorted(silent_cells)} vs {sorted(d94['silent'])}")
                page.screenshot(path=os.path.join(SHOTS, "03_end_silent_1280.png"),
                                full_page=True)

                # 沈黙の区画のポップアップ
                page.click(f"#mapR .cell[data-pid='{d94['silent'][0]}']")
                page.wait_for_timeout(300)
                check("present: 沈黙の区画をクリックすると詳細が開く",
                      page.is_visible("#dlgBody")
                      and d94["silent"][0] in page.inner_text("#dlgTitle"))
                check("present: 詳細に追跡できた月数が出る",
                      "追跡できた月数" in page.inner_text("#dlgBody"))
                page.screenshot(path=os.path.join(SHOTS, "04_silent_detail_1280.png"))
                page.click("#dlgClose")

                # 補助パネル
                for key, name in (("D", "05_panel_D"), ("E", "06_panel_E"),
                                  ("F", "07_panel_F"), ("G", "08_panel_G"),
                                  ("H", "09_panel_H")):
                    page.click(f"[role=tab][data-p='{key}']")
                    page.wait_for_timeout(200)
                    check(f"present: パネル{key} が開く",
                          page.is_visible(f"#p{key}"))
                    page.screenshot(path=os.path.join(SHOTS, f"{name}_1280.png"),
                                    full_page=True)
                check("present: 沈黙の区画がキーボードで開ける",
                      page.eval_on_selector(f"#mapR .cell[data-pid='{d94['silent'][0]}']",
                                            "el => el.tabIndex") == 0)
                v=d94.get("invented_link")
                check("present: 街が作った結びつきの数値が JSON から出る",
                      bool(v) and str(v["utterances"]) in page.inner_text("#invented")
                      and v["quote"]["text"][:24] in page.inner_text("#invented"),
                      "invented_link 無し" if not v else "")
                check("present: 引用に出典（utt_id と月）が付く",
                      bool(v) and v["quote"]["utt_id"] in page.inner_text("#invented"))
                check("present: 窓口パネルに実際の発話が載る",
                      len(page.inner_text("#replay")) > 200)
                check("present: 引き算パネルの数値が history JSON から出る",
                      "v4.1b" in page.inner_text("#history"))

                # run 切替でデータが変わる
                page.select_option("#runsel", "v5bB")
                page.wait_for_timeout(900)
                sB = dB["stats"]
                stats_b = page.inner_text("#stats")
                check("present: run 切替で数値が入れ替わる",
                      f"{sB['deals']}（{sB['sales']}／{sB['leases']}）" in stats_b,
                      stats_b[:120])
                check("present: v5b では O1／O2 が表示される",
                      str(sB["O1_count"]) in stats_b and "O2" in stats_b)
                check("present: v5b の月数が 24 になる",
                      page.get_attribute("#slider", "max") == str(sB["steps"]))
                page.screenshot(path=os.path.join(SHOTS, "10_v5bB_1280.png"),
                                full_page=True)

                # v5c：4段階の色で描けているか
                v5c_path = os.path.join(PAGES, "present_data_v5cA.json")
                if os.path.exists(v5c_path):
                    with open(v5c_path, encoding="utf-8") as f:
                        dC = json.load(f)
                    page.select_option("#runsel", "v5cA")
                    page.wait_for_timeout(900)
                    sC = dC["stats"]
                    stats_c = page.inner_text("#stats")
                    check("present: v5c の取引数が JSON と一致",
                          f"{sC['deals']}（{sC['sales']}／{sC['leases']}）" in stats_c,
                          stats_c[:120])
                    check("present: v5c は会場数（15）を出す",
                          str(sC["venues"]) in stats_c and sC["venues"] == 15)
                    check("present: v5c は到達した最高段階（排他）を出す",
                          " / ".join(str(sC["C_state_public"][c])
                                     for c in ("blue", "green", "yellow", "red"))
                          in stats_c, stats_c[:200])
                    check("present: 排他的な段階の合計が主体数と一致",
                          sum(sC["C_state_public"].values()) == sC["agents"],
                          str(sC["C_state_public"]))
                    page.fill("#slider", str(sC["steps"]))
                    page.dispatch_event("#slider", "input")
                    page.wait_for_timeout(300)
                    colors = page.eval_on_selector_all(
                        "#mapR .cell",
                        "els => els.map(e => [e.dataset.pid, e.style.background])")
                    want = {"blue": "29, 78, 216", "green": "4, 120, 87",
                            "yellow": "202, 138, 4", "red": "185, 28, 28"}
                    bad = [pid for pid, bg in colors
                           if pid in dC["stage_by_parcel"]
                           and want[dC["stage_by_parcel"][pid]["color"]] not in bg]
                    check("present: 右地図の色が JSON の色と一致する", not bad, str(bad[:4]))
                    painted = {pid for pid, bg in colors if bg}
                    check("present: 4色で塗った区画が JSON と一致する",
                          painted == set(dC["stage_by_parcel"]),
                          str(sorted(painted ^ set(dC["stage_by_parcel"]))[:4]))
                    sil = page.eval_on_selector_all(
                        "#mapR .cell.silent", "els => els.map(e => e.dataset.pid)")
                    check("present: 色が付かなかった区画が JSON と一致する",
                          sorted(sil) == sorted(dC["stage_stats"]["silent"]),
                          f"{sorted(sil)} vs {sorted(dC['stage_stats']['silent'])}")
                    open_numbers(page)
                    panel = page.inner_text("#pI")
                    st = dC["stage_stats"]
                    check("present: 色のパネルが開く", page.is_visible("#pI"))
                    check("present: 色ごとの行数が JSON と一致",
                          all(str(st["rows_by_color"][c]) in panel
                              for c in ("blue", "green", "yellow", "red")),
                          panel[:200])
                    pe = dC.get("prime_event")
                    check("present: 一等地イベントの月と区画がパネルに出る",
                          bool(pe) and f"第{pe['month']}月" in panel
                          and pe["parcel_id"] in panel)
                    page.screenshot(path=os.path.join(SHOTS, "14_v5cA_1280.png"),
                                    full_page=True)

                    # ---- 第2弾：配置・4色の帯・読み方ガイド・火が点いた瞬間 ----
                    Lc = dC["layout"]
                    check("present: 右地図が SVG で描かれている",
                          page.eval_on_selector_all(
                              "#mapR svg rect.cell", "els => els.length") == len(Lc["cells"]))
                    check("present: 左右の地図が同じ座標を使っている",
                          page.evaluate(
                              "() => {const g=id=>[...document.querySelectorAll(id+' rect.cell')]"
                              ".map(e=>[e.dataset.pid,e.getAttribute('x'),e.getAttribute('y'),"
                              "e.getAttribute('width')]);"
                              "return JSON.stringify(g('#mapL'))===JSON.stringify(g('#mapR'));}"))
                    geo = page.eval_on_selector_all(
                        "#mapR rect.cell",
                        "els => els.map(e=>[e.dataset.pid,+e.getAttribute('x'),"
                        "+e.getAttribute('y'),+e.getAttribute('width')])")
                    badgeo = [p for p, x, y, w in geo
                              if abs(x + w / 2 - Lc["cells"][p]["cx"]) > 0.01
                              or abs(y + w / 2 - Lc["cells"][p]["cy"]) > 0.01
                              or w != Lc["cells"][p]["side"]]
                    check("present: マスの位置と大きさが JSON の layout と一致",
                          not badgeo, str(badgeo[:4]))
                    labels = page.eval_on_selector_all(
                        "#mapR text.plabel", "els => els.map(e=>e.textContent)")
                    check("present: 全48区画に名前のラベルが常時出ている",
                          sorted(labels) == sorted(c["name"] for c in Lc["cells"].values()),
                          str(sorted(labels)[:3]))
                    vlabels = page.eval_on_selector_all(
                        "#mapR text.vlabel", "els => els.map(e=>e.textContent)")
                    check("present: 会場15か所の名前も常時出ている",
                          sorted(vlabels) == sorted(v["label"] for v in Lc["venues"]),
                          str(sorted(vlabels)[:3]))
                    check("present: 会場が丸で描かれている",
                          page.eval_on_selector_all(
                              "#mapR circle.venue", "els => els.length") == 15)
                    check("present: 会場の丸が layout の座標どおり",
                          page.eval_on_selector_all(
                              "#mapR circle.venue",
                              "els => els.map(e=>[e.dataset.venue,+e.getAttribute('cx'),"
                              "+e.getAttribute('cy'),+e.getAttribute('r')])")
                          == [[v["id"], v["cx"], v["cy"], v["r"]] for v in Lc["venues"]])
                    check("present: 全区画に用途のピクトが描かれている",
                          page.eval_on_selector_all(
                              "#mapR g.icon", "els => els.length") == len(Lc["cells"]))
                    legend = page.inner_text("#legendR")
                    check("present: 凡例にマスの読み方が書いてある",
                          "マスの大きさ＝敷地面積（辺は面積の平方根に比例）" in legend
                          and "中央＝市街地・外側＝郊外" in legend, legend[:160])
                    check("present: 凡例に装飾の断りが書いてある",
                          "通り・広場・地形は装飾です（判定には使っていません）" in legend
                          and "会場と本拠の位置も装飾です" in legend, legend[:400])
                    check("present: 凡例に発光の読み方が書いてある",
                          "二重リング＝この場に、取引した本人が居合わせていた" in legend
                          and "弱い光だけ＝頭の中（誰にも伝わっていない）" in legend, legend[:600])

                    # ---- 第3弾：ラベル・発光・街の下地 ----
                    label_checks(page, "1280", 1100)
                    page.screenshot(path=os.path.join(SHOTS3, "02_town_talk_1280.png"),
                                    full_page=True)
                    page.click("[data-m='L']")
                    page.wait_for_timeout(200)
                    check("present: 登記の地図に切り替えると1枚だけ出る",
                          page.is_visible("#mapboxL") and not page.is_visible("#mapboxR"))
                    same = page.evaluate(
                        "() => {const g=id=>[...document.querySelectorAll(id+' rect.cell')]"
                        ".map(e=>[e.dataset.pid,e.getAttribute('x'),e.getAttribute('y'),"
                        "e.getAttribute('width')]);"
                        "return JSON.stringify(g('#mapL'))===JSON.stringify(g('#mapR'));}")
                    check("present: 切替えても座標は同じ", same)
                    page.screenshot(path=os.path.join(SHOTS3, "01_town_registry_1280.png"),
                                    full_page=True)
                    check("present: 装飾の下地（通り・広場・地形）が描かれている",
                          page.eval_on_selector_all("#mapL g.deco .road", "e=>e.length") >= 8
                          and page.eval_on_selector_all("#mapL g.deco .ring", "e=>e.length") == 2
                          and page.eval_on_selector_all("#mapL g.deco .sea", "e=>e.length") == 1)
                    page.click("[data-m='R']")
                    page.wait_for_timeout(200)

                    # 発光：その月に初めて火が点いた人がいる月へ進める
                    tl = dC["ignition_timeline"]
                    fm = sorted(int(m) for m in tl
                                if any(e["kind"] == "speech" for e in tl[m]))[0]
                    page.fill("#slider", str(fm - 1))
                    page.dispatch_event("#slider", "input")
                    page.wait_for_timeout(250)
                    page.fill("#slider", str(fm))
                    page.dispatch_event("#slider", "input")
                    page.wait_for_timeout(120)
                    fx = page.evaluate(
                        "() => ({pulse:document.querySelectorAll('#mapR .fx circle.pulse').length,"
                        "glow:document.querySelectorAll('#mapR .fx circle.glow').length,"
                        "wave:document.querySelectorAll('#mapR .fx line.wave').length,"
                        "ring2:document.querySelectorAll('#mapR .fx circle.ring2').length,"
                        "anim:getComputedStyle(document.querySelector('#mapR .fx circle.pulse'))"
                        ".animationName})")
                    evs = tl[str(fm)]
                    check("発光: 火が点いた会場がパルスする",
                          fx["pulse"] == len([e for e in evs if e["kind"] == "speech"
                                              and e["venue"]]), str(fx))
                    check("発光: パルスに動きが付いている（既定）",
                          fx["anim"] == "pulse", str(fx["anim"]))
                    want_wave = sum(len([a for a in e["heard_by"] if a in (e.get("homes") or {})])
                                    for e in evs if e["kind"] == "speech")
                    check("発光: 波紋はデータにある同席者の本拠ぶんだけ",
                          fx["wave"] == want_wave, str((fx["wave"], want_wave)))
                    check("発光: 当事者が同席した場には二重リングが出る",
                          fx["ring2"] == len([e for e in evs
                                              if e["party_present"]["present"] and e["venue"]]),
                          str(fx))
                    page.locator("#mapboxR").screenshot(
                        path=os.path.join(SHOTS3, "04_glow_1280.png"))
                    fxlog = page.inner_text("#fxlog")
                    check("発光: 内心は「頭の中（誰にも伝わっていない）」と書く",
                          all((("頭の中（誰にも伝わっていない）" in fxlog)
                               if e["kind"] == "thought" else True) for e in evs), fxlog[:160])
                    check("発光: 同じ月の原文が下に出る",
                          all(e["text"][:20] in fxlog for e in evs), fxlog[:160])
                    still = page.eval_on_selector_all("#mapR .fx > *", "e=>e.length")
                    check("発光: 残光の間はまだ光が残っている", still > 0, str(still))
                    page.locator("#mapboxR").screenshot(
                        path=os.path.join(SHOTS3, "05_afterglow_1280.png"))
                    page.wait_for_timeout(1800)
                    check("発光: 残光のあとは消える",
                          page.eval_on_selector_all("#mapR .fx > *", "e=>e.length") == 0)
                    page.screenshot(path=os.path.join(SHOTS3, "06_fire_moment_1280.png"),
                                    full_page=True)
                    zoom = page.evaluate(
                        "() => {const r=document.querySelector('#mapR svg')"
                        ".getBoundingClientRect();"
                        "return {x:r.x+r.width*0.30,y:r.y+r.height*0.16,"
                        "width:r.width*0.40,height:r.height*0.44};}")
                    page.screenshot(path=os.path.join(SHOTS3, "08_labels_zoom_1280.png"),
                                    clip=zoom)
                    open_numbers(page)
                    page.wait_for_timeout(150)
                    check("present: 数表は「詳しい数字」に畳んである",
                          page.eval_on_selector_all("details.numbers", "e=>e.length") >= 2)
                    page.screenshot(path=os.path.join(SHOTS3, "07_numbers_open_1280.png"),
                                    full_page=True)
                    page.fill("#slider", str(sC["steps"]))   # 月を最後まで戻す
                    page.dispatch_event("#slider", "input")
                    page.wait_for_timeout(300)
                    bar = page.inner_text("#colorbar")
                    check("present: 4色の定義が常に見える帯に出ている",
                          all(t in bar for t in
                              ("青＝個別の売買の話",
                               "緑＝複数の売買を「一帯が動いている」と結びつけた",
                               "黄＝名義が違うのに同じ買い手だと結びつけた（X社にたどり着いた）",
                               "赤＝行政が規制に動いた")), bar[:200])
                    page.locator("#colorbar").screenshot(
                        path=os.path.join(SHOTS2, "07_colorbar_1280.png"))
                    page.screenshot(path=os.path.join(SHOTS2, "01_map_1280.png"),
                                    full_page=True)
                    page.locator(".tabs").screenshot(
                        path=os.path.join(SHOTS2, "06_tabs_1280.png"))

                    # 読み方ガイドの手順どおりに操作して辿り着けるか
                    guide = page.inner_text(".guide")
                    check("present: 読み方ガイドが常時表示されている",
                          "この画面の見方" in guide
                          and "右の地図で色が付いたマスを押す。" in guide, guide[:80])
                    check("present: ガイドは5〜7行",
                          5 <= page.eval_on_selector_all(".guide li", "e => e.length") <= 7)
                    hot = sorted(dC["stage_by_parcel"])[0]
                    page.click(f"#mapR rect.cell[data-pid='{hot}']")
                    page.wait_for_timeout(300)
                    dlg = page.inner_text("#dlgBody")
                    ttl = page.inner_text("#dlgTitle")
                    sbp = dC["stage_by_parcel"][hot]
                    ev = next(e for e in dC["events"] if e["parcel_id"] == hot)
                    check("ガイド手順2: 色付きマスを押すと月と色が出る",
                          hot in ttl and f"第{sbp['month']}月" in ttl, ttl)
                    check("ガイド手順2: その原文が全文出る", sbp["text"] in dlg, dlg[:120])
                    check("ガイド手順3: 登記が動いた月とその差が出る",
                          f"登記が動いた月：第{ev['month']}月" in dlg
                          and f"その差：{sbp['month'] - ev['month']} か月" in dlg, dlg[:200])
                    page.screenshot(path=os.path.join(SHOTS2, "05_parcel_dialog.png"))
                    page.click("#dlgClose")
                    page.wait_for_timeout(200)
                    ip = page.inner_text("#pJ")
                    g = dC["ignition"]["green"]["strict"]
                    check("ガイド手順4: 火が点いた瞬間のタブが開く", page.is_visible("#pJ"))
                    check("ガイド手順4: 緑の原文がそのまま出る",
                          bool(g) and g["text"] in ip, ip[:160])
                    check("ガイド手順4: 誰が・第何月・どの場かが出る",
                          bool(g) and f"第{g['month']}月" in ip and g["from"] in ip)
                    check("ガイド手順4: 並んだものが matched から出る",
                          bool(g) and all(x in ip for x in
                                          g["matched"]["parcels"] + g["matched"]["holders"]),
                          ip[:160])
                    check("ガイド手順4: 内心と発話がラベルで区別されている",
                          "口に出した（同席者に届いた）" in ip
                          or "頭の中（誰にも伝わっていない）" in ip, ip[:160])
                    y = dC["ignition"]["yellow"]["strict"]
                    check("ガイド手順4: 黄も原文で出る（出ない本はその旨）",
                          (y["text"] in ip) if y
                          else ("区画や名義が2つ以上並んだ行は無かった" in ip))
                    check("ガイド手順4: 上段が strict の説明になっている",
                          "区画や名義が実際に2つ以上並んだ、最初の発言・内心。" in ip, ip[:160])
                    check("ガイド手順4: 判定でなく原文を出す注記がある",
                          "人が読み直すと緑どまりの行が混じる" in ip)
                    check("ガイド手順4: loose は畳んである",
                          "判定の定義どおりの初出（畳んである）" in ip
                          and not page.is_visible("#pJ details.igloose > p"), ip[:160])
                    page.click("#pJ details.igloose > summary")
                    page.wait_for_timeout(200)
                    ip2 = page.inner_text("#pJ")
                    gl = dC["ignition"]["green"]["loose"]
                    check("ガイド手順4: 開くと定義どおりの初出が原文で出る",
                          bool(gl) and gl["text"] in ip2, ip2[:160])
                    check("ガイド手順4: 甘さの説明が出る",
                          "まだ何も起きていない月の世間話が入る" in ip2, ip2[:160])
                    crit = dC["ignition"]["criteria"]
                    check("ガイド手順4: strict の説明が色ごとに JSON から出る",
                          crit["strict_green"] in ip2 and crit["strict_yellow"] in ip2,
                          str(crit))
                    fire = page.eval_on_selector_all(
                        "#pJ .say.fire .who", "els => els.map(e=>e.textContent)")
                    check("ガイド手順4: 火が点いた行の見出しに第N月が出る",
                          bool(fire) and all(re.search(r"／第\d+月・", t) for t in fire),
                          str(fire[:2]))
                    check("ガイド手順4: パネルに undefined / NaN が出ない",
                          "undefined" not in ip2 and "NaN" not in ip2,
                          str([w for w in ("undefined", "NaN") if w in ip2]))
                    body = page.evaluate("() => document.body.innerText")
                    check("present: 1280 のページ全体に undefined / NaN が出ない",
                          "undefined" not in body and "NaN" not in body,
                          str([w for w in ("undefined", "NaN") if w in body]))
                    roles = page.eval_on_selector_all(
                        "#mapR rect.cell",
                        "els => els.filter(e => e.tabIndex === 0)"
                        ".map(e => [e.dataset.pid, e.getAttribute('role'),"
                        "e.getAttribute('aria-label') ? 1 : 0])")
                    check("present: 押せるマスに役割とラベルが付いている（SVG化の後退なし）",
                          bool(roles) and all(r == "button" and lab
                                              for _, r, lab in roles), str(roles[:3]))
                    page.screenshot(path=os.path.join(SHOTS2, "04_ignition_1280.png"),
                                    full_page=True)

                    # ぽちぽち（同じ月でも点灯が同時にならない）
                    delays = page.evaluate(
                        "() => [...document.querySelectorAll('#mapR rect.cell')]"
                        ".map(e => [e.dataset.pid, e.style.transitionDelay])")
                    check("present: 巻き戻し・切替の直後は遅延を持たない",
                          all(d in ("", "0s") for _, d in delays), str(delays[:3]))
                    page.click("#reset")
                    page.wait_for_timeout(200)
                    page.click("#play")
                    page.wait_for_timeout(2500)
                    page.screenshot(path=os.path.join(SHOTS2, "02_play_mid1_1280.png"),
                                    full_page=True)
                    lit1 = page.eval_on_selector_all(
                        "#mapR rect.cell", "els => els.filter(e=>e.style.fill).length")
                    page.wait_for_timeout(3500)
                    page.screenshot(path=os.path.join(SHOTS2, "03_play_mid2_1280.png"),
                                    full_page=True)
                    lit2 = page.eval_on_selector_all(
                        "#mapR rect.cell", "els => els.filter(e=>e.style.fill).length")
                    check("present: 再生中に点灯が進む", lit2 >= lit1, f"{lit1} -> {lit2}")
                    page.click("#play")
                    page.wait_for_timeout(200)

                    # 同じ月に色が付く区画が複数ある月へ「進める」と、位相差が付く
                    cnt = collections.Counter(v["month"] for v
                                              in dC["stage_by_parcel"].values())
                    mm = sorted(m for m, c in cnt.items() if c >= 2)
                    check("present: 同じ月に複数の区画が点く月がある", bool(mm), str(cnt))
                    if mm:
                        want = sorted(p for p, v in dC["stage_by_parcel"].items()
                                      if v["month"] == mm[0])
                        page.fill("#slider", str(mm[0] - 1))
                        page.dispatch_event("#slider", "input")
                        page.wait_for_timeout(200)
                        # fill 自体が input を投げる。二重に投げると同じ月で
                        # もう一度 draw が走り、位相差が 0 に戻ってしまう。
                        page.fill("#slider", str(mm[0]))
                        page.wait_for_timeout(120)
                        dl = dict(page.evaluate(
                            "() => [...document.querySelectorAll('#mapR rect.cell')]"
                            ".map(e => [e.dataset.pid, e.style.transitionDelay])"))
                        got = [dl[p] for p in want]
                        secs = [float(x.replace("s", "")) for x in got]
                        check("present: 同じ月の点灯に決定論の遅延が付く（ぽちぽち）",
                              all(0.10 <= v <= 0.40 for v in secs), str(list(zip(want, got))))
                        check("present: 同じ月でも点灯の時刻がずれる",
                              len(set(secs)) == len(secs), str(list(zip(want, got))))
                        check("present: 遅延は 0.4 秒を超えない（次の月に食い込まない）",
                              max(secs) <= 0.4001, str(max(secs)))
                        page.fill("#slider", str(mm[0] - 1))
                        page.wait_for_timeout(120)
                        back = page.evaluate(
                            "() => [...document.querySelectorAll('#mapR rect.cell')]"
                            ".map(e=>e.style.transitionDelay).filter(d=>d&&d!=='0s')")
                        check("present: 月を戻したときは遅延を消す（チカチカさせない）",
                              not back, str(back[:3]))
                    page.fill("#slider", str(sC["steps"]))
                    page.dispatch_event("#slider", "input")
                    page.wait_for_timeout(200)
                    page.click("[role=tab][data-p='D']")
                page.close()

                # 動きを止める設定（prefers-reduced-motion: reduce）
                rp = browser.new_page(viewport={"width": 1280, "height": 900},
                                      reduced_motion="reduce")
                rerr = []
                rp.on("console", lambda x: rerr.append(x.text)
                      if x.type == "error" else None)
                rp.on("pageerror", lambda e: rerr.append(str(e)))
                rp.goto(base + "?run=v5cA", wait_until="networkidle")
                rp.wait_for_selector("#mapR svg", state="attached", timeout=10000)
                rp.click("[data-m='R']")
                with open(os.path.join(PAGES, "present_data_v5cA.json"),
                          encoding="utf-8") as f:
                    dR = json.load(f)
                tlR = dR["ignition_timeline"]
                fmR = sorted(int(m) for m in tlR
                             if any(e["kind"] == "speech" for e in tlR[m]))[0]
                rp.fill("#slider", str(fmR))
                rp.dispatch_event("#slider", "input")
                rp.wait_for_timeout(400)
                anim = rp.evaluate(
                    "() => [...document.querySelectorAll('#mapR .fx circle, #mapR .fx line')]"
                    ".map(e => getComputedStyle(e).animationName)")
                check("動きを止める設定: 発光にアニメーションが掛からない",
                      bool(anim) and all(a == "none" for a in anim), str(anim[:3]))
                dly = rp.evaluate(
                    "() => [...document.querySelectorAll('#mapR rect.cell')]"
                    ".map(e => e.style.transitionDelay).filter(d => d && d !== '0s')")
                check("動きを止める設定: 点灯の位相差も付かない", not dly, str(dly[:3]))
                rp.wait_for_timeout(1800)
                check("動きを止める設定: 最終状態がそのまま残る",
                      rp.eval_on_selector_all("#mapR .fx > *", "e=>e.length") > 0)
                check("動きを止める設定: console エラーが出ない", not rerr, str(rerr[:2]))
                rp.screenshot(path=os.path.join(SHOTS3, "09_reduced_motion_1280.png"),
                              full_page=True)
                rp.close()

                # モバイル
                m = browser.new_page(viewport={"width": 390, "height": 844})
                merr = []
                m.on("console", lambda x: merr.append(x.text)
                     if x.type == "error" else None)
                m.on("pageerror", lambda e: merr.append(str(e)))
                m.goto(base + "?run=run94", wait_until="networkidle")
                m.wait_for_selector("#mapL .cell", timeout=10000)
                open_numbers(m)
                m.click("[data-m='R']")
                m.wait_for_timeout(150)
                mw = m.evaluate("[document.documentElement.scrollWidth,"
                                "document.documentElement.clientWidth]")
                check("present: 390 で横スクロールが出ない", mw[0] <= mw[1], str(mw))
                check("present: 390 でも console エラーが出ない", not merr, str(merr[:2]))
                m.screenshot(path=os.path.join(SHOTS, "11_start_390.png"),
                             full_page=True)
                m.fill("#slider", str(d94["stats"]["steps"]))
                m.dispatch_event("#slider", "input")
                m.wait_for_timeout(300)
                mw2 = m.evaluate("[document.documentElement.scrollWidth,"
                                 "document.documentElement.clientWidth]")
                check("present: 390 の最終月でも横スクロールが出ない", mw2[0] <= mw2[1],
                      str(mw2))
                check("present: 390 でも差のバーが値を持つ",
                      m.eval_on_selector("#gapfill", "el => el.style.width") not in ("", "0%"))
                m.click(f"#mapR .cell[data-pid='{d94['silent'][0]}']")
                m.wait_for_timeout(300)
                check("present: 390 で沈黙の詳細が開く", m.is_visible("#dlgBody"))
                m.screenshot(path=os.path.join(SHOTS, "12_silent_390.png"))
                m.click("#dlgClose")
                for key in ("E", "F", "G", "H"):
                    m.click(f"[role=tab][data-p='{key}']")
                    m.wait_for_timeout(150)
                mw3 = m.evaluate("[document.documentElement.scrollWidth,"
                                 "document.documentElement.clientWidth]")
                check("present: 390 でパネルを開いても横スクロールが出ない",
                      mw3[0] <= mw3[1], str(mw3))
                m.screenshot(path=os.path.join(SHOTS, "13_panels_390.png"),
                             full_page=True)
                m.select_option("#runsel", "v5cA")
                m.wait_for_timeout(1200)
                m.fill("#slider", "24")
                m.dispatch_event("#slider", "input")
                m.wait_for_timeout(600)
                mw4 = m.evaluate("[document.documentElement.scrollWidth,"
                                 "document.documentElement.clientWidth]")
                check("present: 390 の SVG 地図でも横スクロールが出ない",
                      mw4[0] <= mw4[1], str(mw4))
                label_checks(m, "390", 1100)
                m.screenshot(path=os.path.join(SHOTS3, "03_town_390.png"),
                             full_page=True)
                m.wait_for_timeout(300)
                mw5 = m.evaluate("[document.documentElement.scrollWidth,"
                                 "document.documentElement.clientWidth]")
                check("present: 390 で火が点いた瞬間を開いても横スクロールが出ない",
                      mw5[0] <= mw5[1], str(mw5))
                mbody = m.evaluate("() => document.body.innerText")
                check("present: 390 のページ全体に undefined / NaN が出ない",
                      "undefined" not in mbody and "NaN" not in mbody,
                      str([w for w in ("undefined", "NaN") if w in mbody]))
                m.screenshot(path=os.path.join(SHOTS2, "08_map_390.png"),
                             full_page=True)
                m.close()
            finally:
                browser.close()
    finally:
        httpd.shutdown()

    print()
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
