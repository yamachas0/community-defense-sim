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
                check("present: 突き合わせの行に metrics 一致が出る",
                      "metrics_v5 と一致" in page.inner_text("#checkline"))

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
