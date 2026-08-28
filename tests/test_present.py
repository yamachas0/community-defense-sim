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
import socketserver
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES = r"C:\Users\user\projects\quiet-acquisition-pages"
SHOTS = os.path.join(ROOT, "docs", "shots_present")
SHOTS2 = os.path.join(ROOT, "docs", "shots_present2")
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
    """schema 5 の layout / ignition（strict・loose）/ parcel_naming を JSON だけで検査する。"""
    run_dir = os.path.join(ROOT, "simulations", dC["meta"]["generated_from"])

    # --- layout -----------------------------------------------------------
    L = dC.get("layout")
    check("layout: v5c の JSON に layout がある", bool(L))
    if not L:
        return
    check("layout: view が 1000x760", L["view"] == {"w": 1000, "h": 760}, str(L["view"]))
    check("layout: 出所と種が記録されている",
          L["source"] == "configs/parcels_v5c.yaml" and L["seed"] == 8501)
    check("layout: 全区画ぶんのマスがある",
          set(L["cells"]) == {p["pid"] for p in dC["grid"]["parcels"]},
          str(sorted(set(L["cells"]) ^ {p["pid"] for p in dC["grid"]["parcels"]})[:4]))
    need = ("cx", "cy", "side", "zone", "area_sqm", "size_class", "use",
            "use_detail", "owner_name", "frontage")
    check("layout: 各マスに必要な項目がそろっている",
          all(all(k in c for k in need) for c in L["cells"].values()))
    order = sorted(L["cells"].items(), key=lambda kv: (kv[1]["area_sqm"], kv[0]))
    sides = [c["side"] for _, c in order]
    check("layout: マスの大きさが面積の順序と矛盾しない",
          all(sides[i] <= sides[i + 1] for i in range(len(sides) - 1)), str(sides))
    check("layout: 使っている大きさは4段階だけ",
          sorted(set(sides)) == [30, 42, 54, 68], str(sorted(set(sides))))
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
        if not (c["side"] / 2 <= c["cx"] <= 1000 - c["side"] / 2
                and c["side"] / 2 <= c["cy"] <= 760 - c["side"] / 2):
            check("layout: 全マスがビューに収まる", False, str(c))
            break
    else:
        check("layout: 全マスがビューに収まる", True)

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
    check("ignition: criteria に strict / loose の定義がある",
          isinstance(ig.get("criteria"), dict)
          and set(ig["criteria"]) == {"strict", "loose"}
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


def main() -> int:
    from playwright.sync_api import sync_playwright

    os.makedirs(SHOTS, exist_ok=True)
    os.makedirs(SHOTS2, exist_ok=True)
    with open(os.path.join(PAGES, "present_data_run94.json"), encoding="utf-8") as f:
        d94 = json.load(f)
    with open(os.path.join(PAGES, "present_data_v5bB.json"), encoding="utf-8") as f:
        dB = json.load(f)

    for lab in ("run94", "v5bA", "v5bB", "v5bC"):
        with open(os.path.join(PAGES, "present_data_%s.json" % lab), encoding="utf-8") as f:
            d = json.load(f)
        check("%s: 区画属性を持たない run は layout=null（従来の格子で描く）" % lab,
              d.get("layout") is None and d["meta"]["schema"] == 5,
              str(d["meta"]["schema"]))
        ig = d.get("ignition")
        check("%s: ignition は色の判定が無い run でも同じ形" % lab,
              isinstance(ig, dict) and set(ig) == {"green", "yellow", "criteria"}
              and all(ig[c] == {"strict": None, "loose": None}
                      for c in ("green", "yellow"))
              and set(ig["criteria"]) == {"strict", "loose"}, str(ig))
    for lab in ("v5cA", "v5cB", "v5cC"):
        with open(os.path.join(PAGES, "present_data_%s.json" % lab), encoding="utf-8") as f:
            json_checks(json.load(f))

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
                    page.click("[role=tab][data-p='I']")
                    page.wait_for_timeout(250)
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
                        "#mapR text.pid", "els => els.map(e=>e.textContent)")
                    big = sorted(p for p, c in Lc["cells"].items() if c["side"] >= 54)
                    check("present: 大きいマスにだけ区画IDのラベルが出る",
                          sorted(labels) == big,
                          str(sorted(set(labels) ^ set(big))[:4]))
                    owners = page.eval_on_selector_all(
                        "#mapR text.owner", "els => els.map(e=>e.textContent)")
                    want_owner = sorted(
                        Lc["cells"][p]["owner_name"] + "さん" for p in big
                        if dC["parcel_naming"]["name_in_speech"].get(p))
                    check("present: 街が名前で呼ぶ区画には持ち主の名前も出る",
                          sorted(owners) == want_owner,
                          str(sorted(owners)[:3]) + " vs " + str(want_owner[:3]))
                    legend = page.inner_text("#legendR")
                    check("present: 凡例にマスの読み方が書いてある",
                          "マスの大きさ＝敷地面積（4段階）／中央＝市街地・外側＝郊外" in legend,
                          legend[:120])
                    check("present: 凡例に区画の呼び名の注記が出る",
                          "区画に名前は無い。街は「P04」か「持ち主のR03さん」と呼ぶ。" in legend,
                          legend[:200])
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
                    page.click("[role=tab][data-p='J']")
                    page.wait_for_timeout(250)
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
                    page.click("[role=tab][data-p='J']")
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

                # モバイル
                m = browser.new_page(viewport={"width": 390, "height": 844})
                merr = []
                m.on("console", lambda x: merr.append(x.text)
                     if x.type == "error" else None)
                m.on("pageerror", lambda e: merr.append(str(e)))
                m.goto(base + "?run=run94", wait_until="networkidle")
                m.wait_for_selector("#mapL .cell", timeout=10000)
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
                m.click("[role=tab][data-p='J']")
                m.wait_for_timeout(300)
                mw5 = m.evaluate("[document.documentElement.scrollWidth,"
                                 "document.documentElement.clientWidth]")
                check("present: 390 で火が点いた瞬間を開いても横スクロールが出ない",
                      mw5[0] <= mw5[1], str(mw5))
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
