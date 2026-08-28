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
PASS = FAIL = 0


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


def main() -> int:
    from playwright.sync_api import sync_playwright

    os.makedirs(SHOTS, exist_ok=True)
    with open(os.path.join(PAGES, "present_data_run94.json"), encoding="utf-8") as f:
        d94 = json.load(f)
    with open(os.path.join(PAGES, "present_data_v5bB.json"), encoding="utf-8") as f:
        dB = json.load(f)

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
