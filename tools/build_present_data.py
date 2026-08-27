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
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import yaml  # noqa: E402

from run_metrics import (_read_jsonl, _v5_mentions, _v5_s4_kind,  # noqa: E402
                         V5_PARCEL_RE)


def _load(run_dir, name):
    return _read_jsonl(os.path.join(run_dir, name))


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

    awareness = {}
    for row in said:
        step = row["step"]
        hit = _v5_mentions(row["text"], holders_by_step.get(step, set()))
        if not hit["hit"]:
            continue
        if llm_ran and row["ch"] == "utterance":
            if (step, row["from"], row["text"]) not in about:
                continue
        for pid in hit["parcels"]:
            if pid not in acquired_by_step.get(step, set()):
                continue          # その月までに成立していない取得は数えない
            if row["from"] == party_of.get(pid):
                continue          # 当事者本人は「街が語った」に数えない
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

    # --- v5b: O1（連結）が出た月を区画ごとに --------------------------------
    o1_month = {}
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
            for pid in named:
                o1_month.setdefault(pid, step)

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

    # --- 窓口シーンの再現（Eパネル） ---------------------------------------
    counter_month = next((m for m in range(1, n_steps + 1)
                          if _v5_s4_kind(m) == "counter"
                          and any(u["scene"] == "S4" and int(u["step"]) == m
                                  for u in utts)), None)
    best = None
    for m in range(1, n_steps + 1):
        if _v5_s4_kind(m) != "counter":
            continue
        rows = [u for u in utts if int(u["step"]) == m and u.get("scene") == "S4"]
        named = sum(len([p for p in V5_PARCEL_RE.findall(u["text"])
                         if p in acquired_by_step.get(m, set())]) for u in rows)
        if rows and (best is None or named > best[1]):
            best = (m, named)
    scene_month = best[0] if best else counter_month
    scene_replay = {"month": scene_month, "venue": "V06", "rows": [], "article": None}
    if scene_month:
        rows = sorted([u for u in utts
                       if int(u["step"]) == scene_month and u.get("scene") == "S4"],
                      key=lambda u: (u.get("round", 0), u.get("utt_id", "")))
        scene_replay["attendees"] = sorted({u.get("from") for u in rows})
        scene_replay["registry_rows"] = len([t for t in traces
                                             if t.get("kind") == "registry_lookup"
                                             and int(t.get("step", 0)) == scene_month])
        scene_replay["rows"] = [{"round": u.get("round"), "from": u.get("from"),
                                 "name": u.get("name"), "utt_id": u.get("utt_id"),
                                 "text": u["text"]} for u in rows[:10]]
        nxt = [a for a in articles if int(a["step"]) == scene_month]
        if nxt:
            scene_replay["article"] = {"month": int(nxt[0]["step"]),
                                       "from": nxt[0].get("from"),
                                       "text": nxt[0]["text"]}

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
    }

    # 公開済み metrics との突き合わせ（画面の数字が集計と食い違わないことの担保）
    # metrics_v5 の first_mention は「登記が動いた取得（sale）」だけを追う。
    # 地図は賃借も含めた28件を塗るので、**売買だけの部分集合**で突き合わせる。
    sale_pids = {e["parcel_id"] for e in events if e["kind"] == "sale"}
    noticed_sale = len([p for p in awareness if p in sale_pids])
    silent_sale = len([p for p in silent if p in sale_pids])
    checks = {
        "noticed_sale_matches_metrics":
            noticed_sale == metrics.get("noticed_acquisitions"),
        "silent_sale_matches_metrics":
            silent_sale == len(metrics.get("unnoticed_acquisitions") or []),
        "deals_matches_metrics": (len(events) == (metrics.get("deals")
                                                  or metrics.get("acquisitions"))),
        "noticed_all_deals": len(awareness),
        "silent_all_deals": len(silent),
        "noticed_sale_only": noticed_sale,
        "silent_sale_only": silent_sale,
    }

    return {
        "meta": {"generated_from": os.path.basename(run_dir),
                 "generator": "tools/build_present_data.py",
                 "note": "画面の数値・塗りはこのJSONだけから描く（手打ち禁止）"},
        "grid": {"cols": cols, "rows": rows, "blocks": blocks, "parcels": parcels},
        "events": events,
        "awareness": awareness,
        "o1_month": o1_month,
        "silent": silent,
        "silent_detail": silent_detail,
        "agent_timeline": agent_timeline,
        "agent_deals": {k: v for k, v in agent_deals.items() if k in agent_timeline},
        "scene_replay": scene_replay,
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
