#!/usr/bin/env python
"""1ランの判定指標を、events.jsonl / ledger.jsonl / summary.json から機械的に集計する。

    python tools/run_metrics.py --run simulations/<run_dir> [--json]

run36 / run38 の診断で使った指標と同じ定義で出す（版をまたいで比較できるように）。
ここでは解釈も評価もしない。数える対象は帳簿と行動記録に実在する事実だけ。
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
from typing import Any, Dict, List

MEETING_WORDS = ("来週", "日程", "ヒアリング", "訪問", "打ち合わせ", "アポ",
                 "面談", "お伺い", "伺いま", "調整")
MONEY_RE = re.compile(r"\d+\s*万")


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --- v5: 出来事はピン留め・観測するのは街の会話 ---------------------------
# 判定と抽出の定義は docs/world_design_v5_impl.md 6章で**走行前に固定**したもの。
# 2026-08-27 Codexレビューを受けて、走行前に次を直した（生ログは不変・事後集計のみ）：
#   ・私信と記事も「言及」の対象に入れる（発話だけ見ていた）
#   ・語彙の取りこぼしを減らす／名義だけの言及を個別取得に一括帰属しない
#   ・売主の自己言及を移転月以降に限る（移転前の発言を数えていた）
#   ・右打切りを分ける（第12月の取得と第1月の取得を同じ重みで数えない）
#   ・trigger を「発言より前に本人が持っていた観測」に限り available_antecedents に改名
#   ・D2を3段階（個別/反復集中/共通の買い手）に分ける・D3を配線の不変条件で測る
# ここでは解釈も評価もしない。数えるのは記録に実在する事実だけ。

S4_KINDS = ["assembly", "counter", "broker_front", "press"]
V5_SCENES = ["S1", "S2", "S3", "S4"]
V5_HOLDER_RE = re.compile(r"[A-Z]社")
V5_PARCEL_RE = re.compile(r"P\d{2}")
V5_WORDS = ("名義", "所有者が変わ", "持ち主", "オーナー", "売られ", "売った", "売却",
            "手放し", "買い取", "買われ", "買った", "買い占め", "取得", "買収",
            "譲渡", "登記", "地上げ", "底地", "よそ者", "外の会社", "東京の会社",
            "外資", "不動産屋", "投資")
V5_FOLLOWUP_MONTHS = 3      # 右打切りの基準（走行前に固定）


def _v5_s4_kind(step: int) -> str:
    return S4_KINDS[(int(step) - 1) % len(S4_KINDS)]


def _v5_order(step, scene, rnd=0):
    """観測の前後関係。月→シーン→ラウンドの順（ambientは月の先頭）。"""
    s = V5_SCENES.index(scene) + 1 if scene in V5_SCENES else 0
    return (int(step), s, int(rnd or 0))


def _v5_mentions(text, holders):
    """走行前に固定したルールベース1次抽出（絞り込みであって判定ではない）。"""
    parcels = sorted(set(V5_PARCEL_RE.findall(text)))
    named = sorted({h for h in holders if h and h in text})
    words = [w for w in V5_WORDS if w in text]
    return {"parcels": parcels, "holders": named, "words": words,
            "hit": bool(parcels or named or words)}



# --- v5b: 「占領の認知」の判定（主判定・走行前に固定） ---------------------
# 施主 2026-08-27 22:18「観測したいのは X社が実質占領しようとしてるんじゃないか
# ということに気づくかどうかであって、1件1件の売買について気づくかどうかではない」
# ルールベース1次抽出と、走行後にLLMで付けたラベル（occupation_labels.jsonl）の
# 両方を出し、and / or の感度も併記する。「出なかった」も同じ重みで記録する。

V5B_INTENT_WORDS = ("買い占め", "買い集め", "地上げ", "乗っ取り", "支配", "一帯",
                    "街ごと", "町ごと", "次々", "相次", "まとめて買", "計画的",
                    "組織的", "同じ会社", "同一", "裏で", "占領")


def _v5b_rule_links(text, holders, acquired_pids):
    hs = {h for h in holders if h and h in text}
    ps = {p for p in V5_PARCEL_RE.findall(text) if p in acquired_pids}
    return len(hs) >= 2 or len(ps) >= 2 or (len(hs) >= 1 and len(ps) >= 2)


def _v5b_rule_intent(text):
    return any(w in text for w in V5B_INTENT_WORDS)



def _v5b_control_share(summary, n_owned, n_leased):
    """所有＋賃借で X社が実際に握っている非公共区画の割合。

    分母は summary の非公共区画数を直接使う（所有率からの逆算はしない）。
    旧ランで欠けている場合だけ所有率から復元する。
    """
    tradable = summary.get("parcels_tradable")
    if not tradable:
        share = (summary.get("kpi") or {}).get("final_acquirer_share") or 0
        if not share or not n_owned:
            return None
        tradable = round(n_owned / share)
    return round((n_owned + n_leased) / tradable, 3) if tradable else None


def occupation_metrics(run_dir, holders_by_step, acquired_by_step, deals_by_agent,
                       n_steps, n_actors):
    """占領の認知 O1〜O4。定義は docs/world_design_v5_impl.md 12章で走行前に固定。

    Codexレビュー 2026-08-28 を受けた点：
      ・分類に失敗した行は false ではなく **unknown** として数え、主値から外す
        （欠損を「気づかなかった」に化けさせない）
      ・その月までに成立した取引・名義だけで判定する（未来の取得を参照しない）
      ・O3 は「口にした人数」と「内心にとどめた人数」を分ける
    """
    labels = _read_jsonl(os.path.join(run_dir, "occupation_labels.jsonl"))
    if not labels:
        return {"O_available": False}
    rows = []
    for r in labels:
        text = str(r.get("text", ""))
        step = int(r.get("step", 0))
        hs = holders_by_step.get(step, set())
        ps = acquired_by_step.get(step, set())
        classified = bool(r.get("classified", True)) and r.get("links_multiple") is not None
        speaker = r.get("from", "")
        own = deals_by_agent.get(speaker, [])
        rows.append({
            "step": step, "from": speaker, "kind": r.get("kind", ""),
            "scene": r.get("scene", ""), "text": text,
            "rule_links": _v5b_rule_links(text, hs, ps),
            "rule_intent": _v5b_rule_intent(text),
            "llm_links": bool(r.get("links_multiple")) if classified else None,
            "llm_intent": bool(r.get("intent")) if classified else None,
            "classified": classified,
            "party_any": bool([d for d in own if int(d.get("step", 0)) <= step]),
            "party_of_mentioned": bool([d for d in own
                                        if int(d.get("step", 0)) <= step
                                        and d.get("parcel_id") in text]),
        })
    rows.sort(key=lambda r: (r["step"], r["kind"], r["from"]))
    unknown = len([r for r in rows if not r["classified"]])

    def o1(r):
        return r["classified"] and r["rule_links"] and r["llm_links"]

    def o2(r):
        return r["classified"] and r["rule_intent"] and r["llm_intent"]

    def any_o(r):
        return o1(r) or o2(r)

    def first(sel):
        for r in rows:
            if sel(r):
                return {"month": r["step"], "agent_id": r["from"], "kind": r["kind"],
                        "scene": r["scene"], "party_any": r["party_any"],
                        "party_of_mentioned": r["party_of_mentioned"],
                        "text": r["text"][:180]}
        return None

    public = ("utterance", "article")
    pub_series, priv_series = {}, {}
    cum_pub, cum_priv = set(), set()
    for step in range(1, n_steps + 1):
        for r in rows:
            if r["step"] != step or not any_o(r):
                continue
            (cum_pub if r["kind"] in public else cum_priv).add(r["from"])
        pub_series[step] = len(cum_pub)
        priv_series[step] = len(cum_priv - cum_pub)

    assembly, counter = set(), set()
    for r in rows:
        if not any_o(r) or r["kind"] != "utterance" or r["scene"] != "S4":
            continue
        kind = _v5_s4_kind(r["step"])
        if kind == "counter":
            counter.add(r["step"])
        elif kind == "assembly":
            assembly.add(r["step"])

    def months(sel, kinds=None):
        return sorted({r["step"] for r in rows
                       if sel(r) and (not kinds or r["kind"] in kinds)})

    return {
        "O_available": True,
        "O_rows_total": len(rows),
        "O_rows_classified": len(rows) - unknown,
        "O_unknown": unknown,
        "O_measurable": unknown == 0,
        # O1 連結（2つ以上の名義／区画を同じ動きとして結びつけた）
        "O1_first": first(o1),
        "O1_first_public": first(lambda r: o1(r) and r["kind"] in public),
        "O1_first_non_party": first(lambda r: o1(r) and not r["party_any"]),
        "O1_count": len([r for r in rows if o1(r)]),
        "O1_public_count": len([r for r in rows if o1(r) and r["kind"] in public]),
        "O1_by_party": {
            "party_of_mentioned": len([r for r in rows if o1(r)
                                       and r["party_of_mentioned"]]),
            "party_other_deal": len([r for r in rows if o1(r) and r["party_any"]
                                     and not r["party_of_mentioned"]]),
            "non_party": len([r for r in rows if o1(r) and not r["party_any"]]),
        },
        "O1_rule_only": len([r for r in rows if r["classified"]
                             and r["rule_links"] and not r["llm_links"]]),
        "O1_llm_only": len([r for r in rows if r["classified"]
                            and r["llm_links"] and not r["rule_links"]]),
        "O1_or_count": len([r for r in rows if r["classified"]
                            and (r["rule_links"] or r["llm_links"])]),
        "O1_agents": sorted({r["from"] for r in rows if o1(r)}),
        "O1_months": months(o1),
        # O2 意図（街ぐるみの買い集め・地上げ等）
        "O2_first": first(o2),
        "O2_first_public": first(lambda r: o2(r) and r["kind"] in public),
        "O2_count": len([r for r in rows if o2(r)]),
        "O2_public_count": len([r for r in rows if o2(r) and r["kind"] in public]),
        "O2_rule_only": len([r for r in rows if r["classified"]
                             and r["rule_intent"] and not r["llm_intent"]]),
        "O2_llm_only": len([r for r in rows if r["classified"]
                            and r["llm_intent"] and not r["rule_intent"]]),
        "O2_or_count": len([r for r in rows if r["classified"]
                            and (r["rule_intent"] or r["llm_intent"])]),
        "O2_agents": sorted({r["from"] for r in rows if o2(r)}),
        "O2_months": months(o2),
        "O1_and_O2_same_row": len([r for r in rows if o1(r) and o2(r)]),
        "O1_and_O2_same_agent": len(({r["from"] for r in rows if o1(r)}
                                     & {r["from"] for r in rows if o2(r)})),
        # O3 広がり（口にした人／内心にとどめた人を分ける）
        "O3_public_agents_by_month": pub_series,
        "O3_public_agents_final": pub_series.get(n_steps, 0),
        "O3_public_share_final": (round(pub_series.get(n_steps, 0) / n_actors, 3)
                                  if n_actors else None),
        "O3_public_agents_at_12": pub_series.get(min(12, n_steps), 0),
        "O3_private_only_agents_final": priv_series.get(n_steps, 0),
        # O4 公の場
        "O4_article_months": months(any_o, kinds=("article",)),
        "O4_assembly_months": sorted(assembly),
        "O4_counter_months": sorted(counter),
        "O4_O1_public_months": months(lambda r: o1(r) and r["kind"] in public),
        "O4_O2_public_months": months(lambda r: o2(r) and r["kind"] in public),
        "O4_private_only": (not months(any_o, kinds=public)
                            and bool(months(any_o, kinds=("thought",)))),
    }


def metrics_v5(run_dir: str) -> Dict[str, Any]:
    events = _read_jsonl(os.path.join(run_dir, "events.jsonl"))
    ledger = _read_jsonl(os.path.join(run_dir, "ledger.jsonl"))
    utts = _read_jsonl(os.path.join(run_dir, "utterances_v5.jsonl"))
    traces = _read_jsonl(os.path.join(run_dir, "traces_v5.jsonl"))
    stances = _read_jsonl(os.path.join(run_dir, "stances_v5.jsonl"))
    articles = _read_jsonl(os.path.join(run_dir, "articles_v5.jsonl"))
    directs = _read_jsonl(os.path.join(run_dir, "directs_v5.jsonl"))
    plans = _read_jsonl(os.path.join(run_dir, "plans_v5.jsonl"))
    thoughts = _read_jsonl(os.path.join(run_dir, "thoughts_all.jsonl"))
    classified = _read_jsonl(os.path.join(run_dir, "utterances.jsonl"))
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    n_steps = int(summary.get("steps") or 0)
    scen_version = "field_v5"
    cfgp = os.path.join(run_dir, "config.yaml")
    if os.path.exists(cfgp):
        for line in open(cfgp, encoding="utf-8"):
            if line.startswith("scenario_version:"):
                scen_version = line.split(":",1)[1].strip(); break

    deals = _read_jsonl(os.path.join(run_dir, "deals_v5.jsonl"))
    leases = [r for r in ledger if r.get("kind") == "lease"]
    acqs = [r for r in ledger if r.get("kind") == "transfer"]
    holders = sorted({str(a.get("under_name", "")) for a in acqs + leases
                      if a.get("under_name")})
    by_parcel = {a["parcel_id"]: a for a in acqs}
    acq_id_of = {a["parcel_id"]: (a.get("acq_id") or a["parcel_id"]) for a in acqs}
    sellers = {a["parcel_id"]: a.get("seller", "") for a in acqs}

    # LLM の事後分類（走行中の主体には一切見せていない）。
    # 解析に失敗した分は False ではなく unknown として数える。
    about, unknown_cls = set(), 0
    llm_ran = any("about_acquisition" in r for r in classified)
    for r in classified:
        key = (int(r.get("step", 0)), r.get("from", ""), str(r.get("text", "")))
        if r.get("about_acquisition"):
            about.add(key)
        elif r.get("frame") in (None, "", "unclassified"):
            unknown_cls += 1

    # --- 言及の母集団＝発話＋私信＋記事 ------------------------------------
    said = []
    for u in utts:
        said.append({"kind": "utterance", "step": int(u["step"]), "from": u.get("from"),
                     "role": u.get("role"), "scene": u.get("scene"), "round": u.get("round"),
                     "venue": u.get("venue"), "text": str(u.get("text", "")),
                     "heard_by": u.get("heard_by", [])})
    for d in directs:
        said.append({"kind": "direct", "step": int(d["step"]), "from": d.get("from"),
                     "role": "", "scene": d.get("scene"), "round": 9, "venue": "",
                     "text": str(d.get("text", "")), "heard_by": [d.get("to")]})
    for a in articles:
        said.append({"kind": "article", "step": int(a["step"]), "from": a.get("from"),
                     "role": "media", "scene": a.get("scene"), "round": 9, "venue": "",
                     "text": str(a.get("text", "")), "heard_by": []})
    said.sort(key=lambda r: _v5_order(r["step"], r.get("scene") or "", r.get("round")))

    cross = collections.Counter()
    for row in said:
        if row["kind"] != "utterance":
            continue
        r_hit = _v5_mentions(row["text"], holders)["hit"]
        l_hit = (int(row["step"]), row["from"], row["text"]) in about
        cross[("rule+" if r_hit else "rule-") + "/" + ("llm+" if l_hit else "llm-")] += 1
    llm_only = [row for row in said
                if row["kind"] == "utterance"
                and not _v5_mentions(row["text"], holders)["hit"]
                and (int(row["step"]), row["from"], row["text"]) in about]

    mention_rows, edges, form = [], 0, collections.Counter()
    knowers_by_month = collections.defaultdict(set)
    holder_only_rows = []
    for row in said:
        hit = _v5_mentions(row["text"], holders)
        if not hit["hit"]:
            continue
        key = (row["step"], row["from"], row["text"])
        if llm_ran and row["kind"] == "utterance" and key not in about:
            continue
        row["_hit"] = hit
        mention_rows.append(row)
        edges += len(row["heard_by"])
        knowers_by_month[row["step"]].add(row["from"])
        if hit["parcels"]:
            for pid in hit["parcels"]:
                rec = by_parcel.get(pid)
                if rec and int(rec.get("step", 0)) <= row["step"]:
                    ok = (not hit["holders"] or rec.get("under_name") in hit["holders"])
                    form["fact" if ok else "error"] += 1
                else:
                    form["error"] += 1
        elif hit["holders"]:
            form["holder_only"] += 1
            holder_only_rows.append(row)
        else:
            form["unspecific"] += 1

    # --- 初認知（当事者＝売主を除く・区画が名指しされたものだけ） -----------
    first_mention = {}
    for row in mention_rows:
        for pid in row["_hit"]["parcels"]:
            rec = by_parcel.get(pid)
            if rec is None or int(rec.get("step", 0)) > row["step"]:
                continue
            acq_id = acq_id_of[pid]
            if acq_id in first_mention or row["from"] == sellers.get(pid):
                continue
            here = _v5_order(row["step"], row.get("scene") or "", row.get("round"))
            ante = sorted(
                [t for t in traces
                 if t.get("agent_id") == row["from"] and t.get("parcel_id") == pid
                 and _v5_order(t.get("step", 0), t.get("scene") or "") < here],
                key=lambda t: _v5_order(t.get("step", 0), t.get("scene") or ""))
            heard = [m for m in mention_rows
                     if row["from"] in (m["heard_by"] or [])
                     and pid in m["_hit"]["parcels"]
                     and _v5_order(m["step"], m.get("scene") or "", m.get("round")) < here]
            first_mention[acq_id] = {
                "parcel_id": pid, "transfer_month": int(rec.get("step", 0)),
                "month": row["step"], "agent_id": row["from"], "role": row["role"],
                "channel": row["kind"], "scene": row.get("scene"), "venue": row.get("venue"),
                "lag_months": row["step"] - int(rec.get("step", 0)),
                "available_antecedents": [t.get("kind") for t in ante][:4],
                "heard_from": [m["from"] for m in heard][:4],
            }

    optimistic = set(first_mention)
    for row in holder_only_rows:
        for a in acqs:
            if (int(a["step"]) <= row["step"]
                    and a.get("under_name") in row["_hit"]["holders"]
                    and row["from"] != a.get("seller")):
                optimistic.add(acq_id_of[a["parcel_id"]])

    all_ids = [acq_id_of[a["parcel_id"]] for a in acqs]
    noticed = set(first_mention)
    unnoticed = [a for a in all_ids if a not in noticed]
    # 右打切り：追跡できた月数が V5_FOLLOWUP_MONTHS 以上の取得だけのコホート
    cohort = [acq_id_of[a["parcel_id"]] for a in acqs
              if n_steps - int(a["step"]) >= V5_FOLLOWUP_MONTHS]
    cohort_unnoticed = [a for a in cohort if a not in noticed]

    # --- 明るみに出たか（3段階） -------------------------------------------
    article_months = sorted({r["step"] for r in mention_rows if r["kind"] == "article"})
    assembly = collections.defaultdict(set)
    counter = []
    for row in mention_rows:
        if row["kind"] != "utterance" or row.get("scene") != "S4":
            continue
        kind = _v5_s4_kind(row["step"])
        if kind == "assembly":
            assembly[row["step"]].add(row["from"])
        elif kind == "counter":
            counter.append(row["step"])
    assembly_months = sorted(m for m, who in assembly.items() if len(who) >= 3)
    counter_months = sorted(set(counter))

    # 反復・集中の認知＝1つの発言に2区画以上、または名義＋区画で「まとめて」語る
    concentration = [r for r in mention_rows
                     if len(r["_hit"]["parcels"]) >= 2
                     or (r["_hit"]["holders"] and r["_hit"]["parcels"])]
    # 共通の買い手として結び付けた認知＝1つの発言に2つ以上の名義が出る
    common_buyer = [r for r in mention_rows if len(r["_hit"]["holders"]) >= 2]

    # --- 元所有者の沈黙・発言（移転月以降に限る） --------------------------
    seller_spoke = {}
    for pid, seller in sellers.items():
        month0 = int(by_parcel[pid].get("step", 0))
        rows = [r for r in mention_rows
                if r["from"] == seller and r["step"] >= month0
                and (pid in r["_hit"]["parcels"]
                     or by_parcel[pid].get("under_name") in r["_hit"]["holders"])]
        th = [t for t in thoughts
              if t.get("from") == seller and int(t.get("step", 0)) >= month0
              and (pid in str(t.get("text", ""))
                   or by_parcel[pid].get("under_name", "@") in str(t.get("text", "")))]
        seller_spoke[f"{seller}/{pid}"] = {
            "transfer_month": month0,
            "spoke_month": min([r["step"] for r in rows]) if rows else None,
            "thought_month": min([int(t["step"]) for t in th]) if th else None,
        }

    # --- D1: 当事者以外の3主体以上が、その区画を名指しで語った --------------
    by_acq_speakers = collections.defaultdict(set)
    for row in mention_rows:
        for pid in row["_hit"]["parcels"]:
            rec = by_parcel.get(pid)
            if rec is None or int(rec.get("step", 0)) > row["step"]:
                continue
            if row["from"] == sellers.get(pid):
                continue
            by_acq_speakers[acq_id_of[pid]].add(row["from"])
    d1 = sorted(k for k, who in by_acq_speakers.items() if len(who) >= 3)

    # --- D3: 配線の不変条件（行動の結果とは分ける） ------------------------
    plan_by = {(int(p["step"]), p["agent_id"]): p for p in plans}
    rounds_seen = collections.Counter()
    turn_by = collections.defaultdict(set)
    for e in events:
        if e.get("action_type") != "utterance":
            continue
        rounds_seen[(e["step"], e["scene"], e["venue"])] = max(
            rounds_seen[(e["step"], e["scene"], e["venue"])], int(e.get("round", 0)))
        turn_by[(e["step"], e["scene"], e["venue"], e.get("round"))].add(e["agent_id"])
    expected = collections.defaultdict(set)
    for (st, aid), p in plan_by.items():
        for s in V5_SCENES:
            if p.get(s) and p[s] != "HOME":
                expected[(st, s, p[s])].add(aid)
    conv_groups = {k: v for k, v in expected.items() if len(v) >= 2}
    missing_turns = [k for k, who in conv_groups.items()
                     for r in range(1, (rounds_seen.get(k) or 0) + 1)
                     if turn_by.get((k[0], k[1], k[2], r)) != who]
    bad_delivery = [u for u in utts
                    if sorted(u["heard_by"] + [u["from"]])
                    != sorted(conv_groups.get((u["step"], u["scene"], u["venue"]), set()))]
    scenes_with_talk = {(u["step"], u["scene"]) for u in utts}
    parse_fail = len([e for e in events if e.get("action_type") == "PARSE_FAIL"])
    max_tok = summary.get("max_token_finishes", 0)
    # 判定はその月までに成立した取引・名義だけを使う（未来の取得を参照しない）。
    holders_by_step, acquired_by_step = {}, {}
    hs, ps = set(), set()
    for step in range(1, n_steps + 1):
        for r in acqs + leases:
            if int(r.get("step", 0)) <= step:
                hs.add(str(r.get("under_name", "")))
                ps.add(r.get("parcel_id"))
        holders_by_step[step] = set(hs)
        acquired_by_step[step] = set(ps)
    deals_by_agent = collections.defaultdict(list)
    for d in deals:
        deals_by_agent[d.get("agent_id")].append(d)
    acquired_pids = ({a["parcel_id"] for a in acqs} | {r["parcel_id"] for r in leases})
    n_actors = sum(v for k, v in (summary.get("agents") or {}).items()
                   if k != "acquirer")
    occ = occupation_metrics(run_dir, holders_by_step, acquired_by_step,
                             deals_by_agent, n_steps, n_actors)

    n_classified = len([r for r in classified if "about_acquisition" in r])
    d3 = {
        "api_errors": summary.get("usage", {}).get("errors", 0),
        "transfers_recorded": len(acqs),
        "classified_rows": n_classified,
        "classifier_unknown": unknown_cls,
        "parse_fail": parse_fail,
        "max_token_finishes": max_tok,
        "groups_missing_turns": len(missing_turns),
        "utterances_misdelivered": len(bad_delivery),
        "conversation_groups": len(conv_groups),
        "scenes_with_conversation": len(scenes_with_talk),
        "rounds_max": max(rounds_seen.values()) if rounds_seen else 0,
        "occupation_unknown": occ.get("O_unknown", 0),
        "ok": (parse_fail == 0 and max_tok == 0 and not missing_turns
               and not bad_delivery and bool(conv_groups)
               and summary.get("usage", {}).get("errors", 0) == 0
               and (not llm_ran or (n_classified == len(utts) and unknown_cls == 0))
               and occ.get("O_unknown", 0) == 0),
    }

    with open(os.path.join(run_dir, "edges_v5.jsonl"), "w", encoding="utf-8") as f:
        for row in mention_rows:
            for dst in (row["heard_by"] or []):
                f.write(json.dumps({"step": row["step"], "from": row["from"],
                                    "to": dst, "channel": row["kind"],
                                    "scene": row.get("scene"), "venue": row.get("venue"),
                                    "parcels": row["_hit"]["parcels"],
                                    "holders": row["_hit"]["holders"]},
                                   ensure_ascii=False) + "\n")

    talk_events = [e for e in events if e.get("action_type") == "utterance"]
    usage = summary.get("usage", {})
    lags = [v["lag_months"] for v in first_mention.values()]
    return {
        "run": os.path.basename(run_dir),
        "version": scen_version,
        "steps": n_steps,
        "model": summary.get("model"),
        "calls": usage.get("calls", 0),
        "errors": usage.get("errors", 0),
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "invalid_actions": summary.get("invalid_actions", 0),
        "llm_classified": llm_ran,
        "classifier_unknown": unknown_cls,
        # 出来事（台本）
        "acquisitions": len(acqs),
        "leases": len(leases),
        "deals": len(deals),
        "acquisition_months": sorted(int(a["step"]) for a in acqs),
        "lease_months": sorted(int(r["step"]) for r in leases),
        "control_share": _v5b_control_share(summary, len(acqs), len(leases)),
        "holders": holders,
        "final_acquirer_share": summary.get("kpi", {}).get("final_acquirer_share"),
        # 会話
        "utterance_turns": len(talk_events),
        "utterances_spoken": len(utts),
        "silence_rate": (round(1 - len(utts) / len(talk_events), 3)
                         if talk_events else None),
        "mean_group_size": (round(sum(len(v) for v in conv_groups.values())
                                  / len(conv_groups), 2) if conv_groups else 0),
        "directs": len(directs),
        "articles": len(articles),
        "home_rate": (round(sum(1 for p in plans for s in V5_SCENES
                                if p.get(s) == "HOME") / max(1, len(plans) * 4), 3)
                      if plans else None),
        # 気づき
        "mention_rows": len(mention_rows),
        "mention_by_channel": dict(collections.Counter(r["kind"] for r in mention_rows)),
        "propagation_edges": edges,
        "speakers_by_month": {m: sorted(v) for m, v in sorted(knowers_by_month.items())},
        "rule_llm_cross": dict(cross),
        "llm_only_utterances": len(llm_only),
        "first_mention": first_mention,
        "noticed_acquisitions": len(noticed),
        "unnoticed_acquisitions": unnoticed,
        "unnoticed_ratio_all": (round(len(unnoticed) / len(all_ids), 3)
                                if all_ids else None),
        "unnoticed_ratio_optimistic": (
            round(len([a for a in all_ids if a not in optimistic]) / len(all_ids), 3)
            if all_ids else None),
        "cohort_followup_months": V5_FOLLOWUP_MONTHS,
        "cohort_size": len(cohort),
        "cohort_unnoticed_ratio": (round(len(cohort_unnoticed) / len(cohort), 3)
                                   if cohort else None),
        "detection_lag_months": sorted(lags),
        "mean_detection_lag_months": (round(sum(lags) / len(lags), 2) if lags else None),
        "reference_accuracy": dict(form),
        "holder_only_mentions": len(holder_only_rows),
        # 明るみ（3段階）
        "article_months": article_months,
        "assembly_months": assembly_months,
        "counter_months": counter_months,
        "concentration_mentions": len(concentration),
        "common_buyer_mentions": len(common_buyer),
        "registry_lookup_agents": sorted({t["agent_id"] for t in traces
                                          if t.get("kind") == "registry_lookup"}),
        "traces_by_kind": dict(collections.Counter(t.get("kind") for t in traces)),
        "seller_first_mention_of_own_sale": seller_spoke,
        "sellers_silent": len([v for v in seller_spoke.values()
                               if v["spoke_month"] is None]),
        # 姿勢（台帳を動かさない観測値）
        "stance_sell": len([s for s in stances if s.get("stance") == "sell"]),
        "stance_keep": len([s for s in stances if s.get("stance") == "keep"]),
        # 判定（走行前に固定した定義）
        "D1_exposed_acquisitions": d1,
        "D2_individual": bool(article_months or assembly_months or counter_months),
        "D2_concentration": bool(concentration),
        "D2_common_buyer": bool(common_buyer),
        "D3_wiring": d3,
        **occ,
    }


def metrics_v41(run_dir: str) -> Dict[str, Any]:
    """v4.1（金額のない世界）の集計。判定基準は docs/world_design_v4_impl.md と同じ定義。"""
    events = _read_jsonl(os.path.join(run_dir, "events.jsonl"))
    ledger = _read_jsonl(os.path.join(run_dir, "ledger.jsonl"))
    thoughts = _read_jsonl(os.path.join(run_dir, "thoughts.jsonl"))
    deliveries = _read_jsonl(os.path.join(run_dir, "deliveries.jsonl"))
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)

    kinds = collections.Counter(r.get("kind", "") for r in ledger)
    offer_records = [r for r in ledger if r.get("kind") == "offer"]
    transfers = [r for r in ledger if r.get("kind") == "transfer"]
    acquirer_ids = {e["agent_id"] for e in events if e.get("role") == "acquirer"}
    valid_transfers = [r for r in transfers if r.get("buyer") in acquirer_ids
                       and r.get("buyer") != r.get("seller")]

    offer_ops = [op for e in events if e.get("role") == "acquirer"
                 for op in e.get("operations", []) or []
                 if op.get("action_type") == "make_offer"]
    offer_fail = collections.Counter(
        op.get("outcome", {}).get("reason", "")
        for op in offer_ops
        if str(op.get("outcome", {}).get("kind", "")).endswith(
            ("_rejected", "invalid_action")))

    response_ops = [op for e in events if e.get("action_type") == "responses"
                    for op in e.get("operations", []) or []]
    decisions = collections.Counter(op.get("action_type", "") for op in response_ops)
    sell_decisions = decisions.get("sell", 0)
    sell_transfer = sum(1 for op in response_ops
                        if op.get("outcome", {}).get("kind") == "transfer")
    sell_filing = sum(1 for op in response_ops
                      if op.get("outcome", {}).get("kind") == "filing_required")
    sell_failed = sum(1 for op in response_ops
                      if op.get("outcome", {}).get("kind") == "response_rejected"
                      and op.get("action_type") == "sell")
    filed_ids = {r.get("offer_id") for r in ledger if r.get("kind") == "filing_required"}
    filed_completed = [r for r in transfers if r.get("offer_id") in filed_ids]

    parse_fail = sum(1 for e in events if e.get("action_type") == "PARSE_FAIL")
    offer_steps = sorted({r["step"] for r in offer_records})
    called = {(e["step"], e["agent_id"]) for e in events
              if e.get("action_type") == "responses"}
    about = [t for t in thoughts if t.get("about_acquisition")]
    about_people = sorted({t["from"] for t in about})
    about_without_own_offer = [t for t in about
                               if (t["step"], t["from"]) not in called]
    money_rows = [r for r in ledger
                  if any(k in r for k in ("price", "amount", "rent", "fee"))]

    # --- 初回反応と owner-month 反応を分ける（長期holdが分母を膨らませないように）---
    first_response: Dict[str, str] = {}
    for e in sorted(events, key=lambda x: (x.get("step", 0), x.get("agent_id", ""))):
        if e.get("action_type") != "responses":
            continue
        for op in e.get("operations", []) or []:
            oid = str(op.get("target", "")).strip().upper().strip("[]")
            if oid and oid not in first_response:
                first_response[oid] = op.get("action_type", "")
    first_decisions = collections.Counter(first_response.values())
    implicit_holds = sum(1 for e in events if e.get("action_type") == "responses"
                         for op in e.get("operations", []) or []
                         if op.get("implicit"))

    # --- 届出（条例）による実現遅延を offer 単位で照合する ---
    filing_rows = [r for r in ledger if r.get("kind") == "filing_required"]
    transfer_by_offer = {r.get("offer_id"): r for r in transfers}
    filing_delays = []
    for row in filing_rows:
        done = transfer_by_offer.get(row.get("offer_id"))
        filing_delays.append({
            "offer_id": row.get("offer_id"), "parcel_id": row.get("parcel_id"),
            "sell_step": row.get("step"), "due_step": row.get("due_step"),
            "transfer_step": done.get("step") if done else None,
            "realized_delay": ((done.get("step") - row.get("step")) if done else None),
        })
    pending_at_end = [r["offer_id"] for r in filing_delays
                      if r["transfer_step"] is None]

    # --- 認知の転相を主体ごとに見る（総件数ではなく個人内の初回転換）---
    first_transition = {}
    for t in sorted(thoughts, key=lambda x: (x.get("step", 0), x.get("from", ""))):
        if t.get("about_acquisition") and t.get("from") not in first_transition:
            first_transition[t["from"]] = t.get("step")
    transition_steps = sorted(first_transition.values())

    # --- 噂（取得を扱う発話）の初出月 ---
    rumor_steps = sorted({d.get("step") for d in deliveries
                          if d.get("kind") in ("ambient", "direct")
                          and any(w in str(d.get("text", ""))
                                  for w in ("名義", "買い", "取得", "外資", "X社",
                                            "まとめて", "手放"))})

    # --- 既に世界にあった相談経路（direct 私信）の量 ---
    #     v4.1b の「専用の相談プロトコル」と比べるための基準値。
    roles = {e["agent_id"]: e.get("role", "") for e in events}
    owner_to_broker = [d for d in deliveries if d.get("kind") == "direct"
                       and roles.get(d.get("from", "")) in ("household", "business")
                       and roles.get(d.get("to", "")) == "broker"]
    broker_to_owner = [d for d in deliveries if d.get("kind") == "direct"
                       and roles.get(d.get("from", "")) == "broker"
                       and roles.get(d.get("to", "")) in ("household", "business")]

    # --- 名義の分散（面積シェアのハーフィンダール指数）---
    name_area = collections.Counter()
    for r in valid_transfers:
        name_area[r.get("under_name", "")] += 1
    total_named = sum(name_area.values()) or 1
    name_hhi = round(sum((v / total_named) ** 2 for v in name_area.values()), 4)

    return {
        "run_dir": os.path.basename(run_dir.rstrip("/\\")),
        "scenario": "field_v4_1",
        "steps": summary.get("steps"),
        "model": summary.get("model"),
        # --- 配線 ---
        "calls": summary.get("usage", {}).get("calls"),
        "api_errors": summary.get("usage", {}).get("errors"),
        "input_tokens": summary.get("usage", {}).get("input_tokens"),
        "output_tokens": summary.get("usage", {}).get("output_tokens"),
        "parse_fail": parse_fail,
        "truncated": summary.get("truncated_responses"),
        "invalid_actions": summary.get("invalid_actions"),
        "offers_returned_by_acquirer": len(offer_ops),
        "offers_recorded": len(offer_records),
        "offer_rejection_reasons": dict(offer_fail),
        "responders_called": sum(1 for e in events if e.get("action_type") == "responses"),
        "responses": dict(decisions),
        "sell_decisions": sell_decisions,
        "sell_immediate_transfer": sell_transfer,
        "sell_filing_required": sell_filing,
        "sell_rejected": sell_failed,
        "sell_reconciles": sell_decisions == (sell_transfer + sell_filing + sell_failed),
        "filed_transfer_completed": len(filed_completed),
        "filing_void": kinds.get("filing_void", 0),
        "offer_void": kinds.get("offer_void", 0),
        "response_rejected": kinds.get("response_rejected", 0),
        "ledger_rows_with_money_fields": len(money_rows),
        "deliveries_by_kind": dict(collections.Counter(d.get("kind", "")
                                                       for d in deliveries)),
        # --- 世界で起きたこと ---
        "months_with_offers": len(offer_steps),
        "first_offer_step": offer_steps[0] if offer_steps else None,
        "months_with_transfers": len({r["step"] for r in valid_transfers}),
        "first_transfer_step": (min(r["step"] for r in valid_transfers)
                                if valid_transfers else None),
        "transfers": len(valid_transfers),
        "withdrawn": kinds.get("withdraw", 0),
        "kept": kinds.get("keep", 0),
        "held": kinds.get("hold", 0),
        "ownership_share": summary.get("kpi", {}).get("final_acquirer_share"),
        "area_share": summary.get("kpi", {}).get("final_acquirer_area_share"),
        "under_names_used": dict(collections.Counter(r.get("under_name", "")
                                                     for r in valid_transfers)),
        "thoughts": len(thoughts),
        "thought_frames": dict(collections.Counter(t.get("frame", "") for t in thoughts)),
        "thoughts_about_acquisition": len(about),
        "residents_aware": len(about_people),
        "aware_without_own_offer": len(about_without_own_offer),
        "first_aware_step": min((t["step"] for t in about), default=None),
        "ordinances": [{"step": r["step"], "by": r.get("by"), "title": r.get("title"),
                        "threshold_sqm": r.get("threshold_sqm"),
                        "delay_months": r.get("delay_months")}
                       for r in ledger if r.get("kind") == "ordinance"],
        "publications": kinds.get("publication", 0),
        "publications_about_acquisition": sum(
            1 for r in ledger if r.get("kind") == "publication"
            and r.get("about_acquisition")),
        "investigations": kinds.get("investigate", 0),
        # --- 判定（事前固定） ---
        "verdict_acquisition_progresses": bool(
            len({r["step"] for r in valid_transfers}) >= 2
            and (summary.get("kpi", {}).get("final_acquirer_area_share") or 0) > 0),
        "verdict_residents_notice": bool(len(about_people) >= 2
                                         and len(about_without_own_offer) >= 1),
        # --- 追加集計（Codexレビュー 2026-08-27 の指摘反映）---
        "offers_first_month": sum(1 for r in offer_records if r.get("step") == 1),
        "first_response_by_offer": dict(first_decisions),
        "owner_month_responses": dict(decisions),
        "implicit_holds": implicit_holds,
        "response_not_recorded": kinds.get("response_not_recorded", 0),
        "ordinance_same_step_conflict": kinds.get("ordinance_same_step_conflict", 0),
        "filing_delays": filing_delays,
        "filings_pending_at_end": pending_at_end,
        "first_transition_step_by_agent": first_transition,
        "median_first_transition_step": (
            transition_steps[len(transition_steps) // 2] if transition_steps else None),
        "first_rumor_step": rumor_steps[0] if rumor_steps else None,
        "under_name_area_hhi": name_hhi,
        "owner_to_broker_direct": len(owner_to_broker),
        "broker_to_owner_direct": len(broker_to_owner),
        "owners_using_direct_to_broker": len({d.get("from") for d in owner_to_broker}),
    }


def metrics_v41b(run_dir: str) -> Dict[str, Any]:
    """v4.1b（相談経路を戻した版）の集計。v4.1 の全指標に相談・回答・条例実績を足す。"""
    base = metrics_v41(run_dir)
    base["scenario"] = "field_v4_1b"
    events = _read_jsonl(os.path.join(run_dir, "events.jsonl"))
    ledger = _read_jsonl(os.path.join(run_dir, "ledger.jsonl"))
    deliveries = _read_jsonl(os.path.join(run_dir, "deliveries.jsonl"))

    consults = [r for r in ledger if r.get("kind") == "consult"]
    advices = [r for r in ledger if r.get("kind") == "advice"]
    consult_rejects = collections.Counter(
        r.get("reason", "") for r in ledger if r.get("kind") == "consult_rejected")
    advice_rejects = collections.Counter(
        r.get("reason", "") for r in ledger if r.get("kind") == "advice_rejected")
    # 台帳に載らない不成立（相手が空・非object・月次上限超過など）は events 側にある。
    for e in events:
        for op in e.get("operations", []) or []:
            outcome = op.get("outcome", {}) or {}
            if str(outcome.get("kind", "")) != "invalid_action":
                continue
            if op.get("action_type") == "consult":
                consult_rejects[str(outcome.get("reason", ""))] += 1
            elif op.get("action_type") == "advise":
                advice_rejects[str(outcome.get("reason", ""))] += 1
    consult_step = {r.get("consult_id"): r.get("step") for r in consults}
    answered_ids = [r.get("consult_id") for r in advices]
    latency = [int(r.get("step", 0)) - int(consult_step.get(r.get("consult_id"), 0))
               for r in advices if r.get("consult_id") in consult_step]

    # 回答が実際に届いた所有者と、その月
    advice_delivered: Dict[str, int] = {}
    for d in deliveries:
        if d.get("kind") != "advice":
            continue
        who = d.get("to", "")
        step = int(d.get("step", 0))
        if who and (who not in advice_delivered or step < advice_delivered[who]):
            advice_delivered[who] = step

    # 助言を読めるのは届いた翌月から。読了前後で owner-month の判断を分けて数える。
    advice_read_step = {who: step + 1 for who, step in advice_delivered.items()}
    decided_after_advice: Dict[str, List[str]] = {}
    before = collections.Counter()
    after = collections.Counter()
    for e in sorted(events, key=lambda x: (x.get("step", 0), x.get("agent_id", ""))):
        if e.get("action_type") != "responses":
            continue
        who = e.get("agent_id", "")
        read = advice_read_step.get(who)
        readable = read is not None and int(e.get("step", 0)) >= read
        for op in e.get("operations", []) or []:
            decision = op.get("action_type", "")
            if decision not in ("sell", "keep", "hold"):
                continue
            (after if readable else before)[decision] += 1
            if readable and decision in ("sell", "keep"):
                decided_after_advice.setdefault(who, []).append(
                    f"M{e.get('step')}:{decision}")

    # 相談の語が所有者の内心に出た回数（v4.1 の 58/70/60 と直接比べるため）
    thoughts_all = _read_jsonl(os.path.join(run_dir, "thoughts_all.jsonl"))
    consult_word = sum(1 for t in thoughts_all
                       if t.get("role") in ("household", "business")
                       and "仲介" in str(t.get("text", "")))

    # 条例が空振りしたか（閾値 > 区画の最大面積）
    areas: List[int] = []
    cfg_path = os.path.join(run_dir, "config.yaml")
    if os.path.exists(cfg_path):
        import yaml
        with open(cfg_path, encoding="utf-8") as f:
            run_cfg = yaml.safe_load(f) or {}
        areas = [int(x) for x in
                 (run_cfg.get("world", {}) or {}).get("area_pattern_sqm", []) or []]
    max_area = max(areas) if areas else None
    ordinances = base.get("ordinances", [])
    base.update({
        "consults_recorded": len(consults),
        "consult_rejection_reasons": dict(consult_rejects),
        "owners_who_consulted": len({r.get("by") for r in consults}),
        "consult_steps": sorted({int(r.get("step", 0)) for r in consults}),
        "advices_recorded": len(advices),
        "advice_rejection_reasons": dict(advice_rejects),
        "consults_answered": len(set(answered_ids)),
        "consult_answer_rate": (round(len(set(answered_ids)) / len(consults), 3)
                                if consults else None),
        "answer_latency_steps": sorted(latency),
        "advices_delivered": sum(1 for d in deliveries if d.get("kind") == "advice"),
        "owners_who_received_advice": len(advice_delivered),
        "terminal_decisions_after_first_advice_read": decided_after_advice,
        "owners_deciding_after_advice_read": len(decided_after_advice),
        "owner_month_decisions_before_advice_read": dict(before),
        "owner_month_decisions_after_advice_read": dict(after),
        "unanswered_consults_at_end": len(consults) - len(set(answered_ids)),
        "advices_unread_at_end": sum(
            1 for d in deliveries if d.get("kind") == "advice"
            and int(d.get("step", 0)) >= int(base.get("steps") or 0)),
        "broker_mentions_in_owner_thoughts": consult_word,
        "max_parcel_area_sqm": max_area,
        "ordinance_thresholds_vs_max_parcel": [
            {"step": o.get("step"), "threshold_sqm": o.get("threshold_sqm"),
             "exceeds_every_parcel": (None if max_area is None
                                      else bool((o.get("threshold_sqm") or 0) >= max_area))}
            for o in ordinances],
        "filings_triggered": sum(1 for r in ledger
                                 if r.get("kind") == "filing_required"),
    })
    return base

def metrics_v4(run_dir: str) -> Dict[str, Any]:
    """v4（同期3フェーズ）の集計。配線の指標と、世界で起きたことの指標を分けて出す。"""
    events = _read_jsonl(os.path.join(run_dir, "events.jsonl"))
    ledger = _read_jsonl(os.path.join(run_dir, "ledger.jsonl"))
    feelings = _read_jsonl(os.path.join(run_dir, "feelings.jsonl"))
    deliveries = _read_jsonl(os.path.join(run_dir, "deliveries.jsonl"))
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)

    kinds = collections.Counter(r.get("kind", "") for r in ledger)
    offer_records = [r for r in ledger if r.get("kind") == "offer"]
    transfers = [r for r in ledger if r.get("kind") == "transfer"]
    fees = [r for r in ledger if r.get("kind") == "broker_fee"]

    offer_ops = [op for e in events if e.get("role") == "acquirer"
                 for op in e.get("operations", []) or []
                 if op.get("action_type") == "make_offer"]
    offer_fail = collections.Counter(
        op.get("outcome", {}).get("reason", "")
        for op in offer_ops if str(op.get("outcome", {}).get("kind", "")).endswith(
            ("_rejected", "invalid_action")))
    via_counter = collections.Counter(op.get("via", "") for op in offer_ops)

    response_ops = [op for e in events if e.get("action_type") == "responses"
                    for op in e.get("operations", []) or []]
    decisions = collections.Counter(op.get("action_type", "") for op in response_ops)
    decision_results = collections.Counter(
        op.get("outcome", {}).get("kind", "") for op in response_ops)

    parse_fail = sum(1 for e in events if e.get("action_type") == "PARSE_FAIL")

    acquirer_ids = {e["agent_id"] for e in events if e.get("role") == "acquirer"}
    valid_transfers = [r for r in transfers if r.get("buyer") in acquirer_ids
                       and r.get("buyer") != r.get("seller")]
    filed_ids = {r.get("offer_id") for r in ledger if r.get("kind") == "filing_required"}
    filed_completed = [r for r in transfers if r.get("offer_id") in filed_ids]
    accept_decisions = sum(1 for op in response_ops if op.get("action_type") == "accept")
    accept_immediate = sum(1 for op in response_ops
                           if op.get("outcome", {}).get("kind") == "transfer")
    accept_filing = sum(1 for op in response_ops
                        if op.get("outcome", {}).get("kind") == "filing_required")
    accept_failed = sum(1 for op in response_ops
                        if op.get("outcome", {}).get("kind") == "accept_rejected")
    # 仲介経由の有効成立1件につき手数料が1件で、金額が契約料率と一致するか
    fee_by_offer = collections.Counter(r.get("offer_id") for r in fees)
    fee_duplicates = sum(1 for n in fee_by_offer.values() if n > 1)
    fee_expected = 0
    for record in valid_transfers + filed_completed:
        channel_fee = [r for r in fees if r.get("offer_id") == record.get("offer_id")]
        if channel_fee:
            fee_expected += int(round(record.get("price", 0) * channel_fee[0].get("rate", 0)))
    # 「その月に買付が届いていた所有者」＝応答フェーズに呼ばれた所有者
    called = {(e["step"], e["agent_id"]) for e in events
              if e.get("action_type") == "responses"}
    offer_steps = sorted({r["step"] for r in offer_records})
    transfer_steps = sorted({r["step"] for r in transfers})

    about = [f for f in feelings if f.get("about_acquisition")]
    about_people = sorted({f["from"] for f in about})
    about_without_own_offer = [f for f in about
                               if (f["step"], f["from"]) not in called]
    frames = collections.Counter(f.get("frame", "") for f in feelings)

    ordinances = [{"step": r["step"], "by": r.get("by"), "title": r.get("title"),
                   "threshold_sqm": r.get("threshold_sqm"),
                   "delay_months": r.get("delay_months")}
                  for r in ledger if r.get("kind") == "ordinance"]

    acquirer_steps = sorted({e["step"] for e in events if e.get("role") == "acquirer"})
    no_offer_months = sum(1 for e in events if e.get("role") == "acquirer"
                          for op in e.get("operations", []) or []
                          if op.get("action_type") == "no_offer")

    return {
        "run_dir": os.path.basename(run_dir.rstrip("/\\")),
        "scenario": "field_v4",
        "steps": summary.get("steps"),
        "model": summary.get("model"),
        # --- 配線（ここが崩れていたら結果は解釈できない） ---
        "calls": summary.get("usage", {}).get("calls"),
        "api_errors": summary.get("usage", {}).get("errors"),
        "input_tokens": summary.get("usage", {}).get("input_tokens"),
        "output_tokens": summary.get("usage", {}).get("output_tokens"),
        "parse_fail": parse_fail,
        "truncated": summary.get("truncated_responses"),
        "invalid_actions": summary.get("invalid_actions"),
        "offers_returned_by_acquirer": len(offer_ops),
        "offers_recorded": len(offer_records),
        "offer_rejection_reasons": dict(offer_fail),
        "offers_via": dict(via_counter),
        "responders_called": sum(1 for e in events if e.get("action_type") == "responses"),
        "responses": dict(decisions),
        "response_outcomes": dict(decision_results),
        "accept_decisions": accept_decisions,
        "accept_immediate_transfer": accept_immediate,
        "accept_filing_required": accept_filing,
        "accept_rejected_outcome": accept_failed,
        "accept_reconciles": accept_decisions == (accept_immediate + accept_filing
                                                  + accept_failed),
        "filed_transfer_completed": len(filed_completed),
        "filing_void": kinds.get("filing_void", 0),
        "offer_void": kinds.get("offer_void", 0),
        "no_response_rejected": kinds.get("no_response_rejected", 0),
        "transfers_total": kinds.get("transfer", 0),
        "transfers_valid_acquirer": len(valid_transfers),
        "fee_duplicates": fee_duplicates,
        "fee_total_matches_rate": fee_expected == sum(r.get("amount", 0) for r in fees),
        "broker_fee_count": len(fees),
        "broker_fee_total": sum(r.get("amount", 0) for r in fees),
        "deliveries_by_kind": dict(collections.Counter(d.get("kind", "")
                                                       for d in deliveries)),
        # --- 世界で起きたこと（設計の指標） ---
        "months_with_offers": len(offer_steps),
        "first_offer_step": offer_steps[0] if offer_steps else None,
        "months_with_transfers": len({r["step"] for r in valid_transfers}),
        "first_transfer_step": (min(r["step"] for r in valid_transfers)
                                if valid_transfers else None),
        "transfers": len(valid_transfers),
        "no_offer_months": no_offer_months,
        "acquirer_active_months": len(acquirer_steps),
        "ownership_share": summary.get("kpi", {}).get("final_acquirer_share"),
        "area_share": summary.get("kpi", {}).get("final_acquirer_area_share"),
        "under_names_used": dict(collections.Counter(r.get("under_name", "")
                                                     for r in transfers)),
        "financing_raised": sum(r.get("amount", 0) for r in ledger
                                if r.get("kind") == "financing_raised"),
        "feelings": len(feelings),
        "feeling_frames": dict(frames),
        "feelings_about_acquisition": len(about),
        "residents_aware": len(about_people),
        "aware_without_own_offer": len(about_without_own_offer),
        "first_aware_step": min((f["step"] for f in about), default=None),
        "ordinances": ordinances,
        "publications": kinds.get("publication", 0),
        "publications_about_acquisition": sum(
            1 for r in ledger if r.get("kind") == "publication"
            and r.get("about_acquisition")),
        "investigations": kinds.get("investigate", 0),
        # --- 判定（docs/world_design_v4_impl.md の事前定義） ---
        "verdict_acquisition_progresses": bool(
            len({r["step"] for r in valid_transfers}) >= 2
                                               and (summary.get("kpi", {})
                                                    .get("final_acquirer_area_share") or 0) > 0),
        "verdict_residents_notice": bool(len(about_people) >= 2
                                         and len(about_without_own_offer) >= 1),
    }


def metrics(run_dir: str) -> Dict[str, Any]:
    events = _read_jsonl(os.path.join(run_dir, "events.jsonl"))
    ledger = _read_jsonl(os.path.join(run_dir, "ledger.jsonl"))
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)

    acquirer_ids = {e["agent_id"] for e in events if e.get("role") == "acquirer"}
    operations: List[Dict[str, Any]] = []
    for event in events:
        if event.get("role") != "acquirer":
            continue
        for op in event.get("operations", []) or []:
            operations.append({**op, "step": event["step"]})

    verbs = collections.Counter(op.get("action_type", "") for op in operations)
    registry_targets = collections.Counter(
        op.get("target", "") for op in operations
        if op.get("action_type") == "check_land_registry")
    duplicate_registry = sum(max(0, n - 1) for n in registry_targets.values())
    preparation = sum(verbs[v] for v in (
        "internal_review", "market_research", "financing_review",
        "existing_asset_management", "check_land_registry", "due_diligence",
        "contact_broker", "public_statement", "wait"))

    direct = [e for e in events
              if e.get("utterance") and e.get("utterance_channel") == "direct"]
    meeting = [e for e in direct if any(w in e["utterance"] for w in MEETING_WORDS)]
    with_money = [e for e in direct if MONEY_RE.search(e["utterance"])]

    no_such_offer = sum(1 for r in ledger
                        if r.get("kind") == "accept_rejected"
                        and r.get("reason") == "no_such_offer")
    kinds = collections.Counter(r.get("kind", "") for r in ledger)
    first_month = collections.Counter(
        op.get("action_type", "") for op in operations if op["step"] == 1)

    plans = [e.get("plan") for e in events if e.get("plan")]
    repeated = {}
    if plans:
        for key in ("strategy", "next_milestone", "goal_assessment",
                    "expected_goal_effect", "revision_reason"):
            same = sum(1 for a, b in zip(plans, plans[1:])
                       if a.get(key, "") and a.get(key, "") == b.get(key, ""))
            repeated[key] = f"{same}/{max(0, len(plans) - 1)}"

    return {
        "run_dir": os.path.basename(run_dir.rstrip("/\\")),
        "steps": summary.get("steps"),
        "acquirer_model": summary.get("acquirer_model"),
        "calls": summary.get("usage", {}).get("calls"),
        "input_tokens": summary.get("usage", {}).get("input_tokens"),
        "output_tokens": summary.get("usage", {}).get("output_tokens"),
        "api_errors": summary.get("usage", {}).get("errors"),
        "invalid_actions": summary.get("invalid_actions"),
        "truncated": summary.get("truncated_responses"),
        "ownership_share": summary.get("kpi", {}).get("final_acquirer_share"),
        "effective_control_area_share": summary.get("kpi", {}).get(
            "final_effective_control_area_share"),
        "transfers": kinds.get("transfer", 0),
        "lease_controls": kinds.get("lease_control", 0),
        "acquirer_operations": len(operations),
        "operation_verbs": dict(verbs.most_common()),
        "preparation_ratio": (round(preparation / len(operations), 3)
                              if operations else None),
        "first_month_operations": dict(first_month),
        "registry_checks": sum(registry_targets.values()),
        "registry_distinct_parcels": len(registry_targets),
        "registry_duplicate_checks": duplicate_registry,
        "make_offer": verbs.get("make_offer", 0),
        "make_lease_offer": verbs.get("make_lease_offer", 0),
        "direct_messages": len(direct),
        "direct_with_meeting_words": len(meeting),
        "direct_with_money": len(with_money),
        "accept_no_such_offer": no_such_offer,
        "inquiry_request": kinds.get("inquiry_request", 0),
        "inquiry_asked": kinds.get("inquiry_asked", 0),
        "inquiry_answer": kinds.get("inquiry_answer", 0),
        "inquiry_report": kinds.get("inquiry_report", 0),
        "land_registry_check_records": kinds.get("land_registry_check", 0),
        "plan_repeat_vs_previous_month": repeated,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, nargs="+")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    rows = []
    for run in args.run:
        cfg_path = os.path.join(run, "config.yaml")
        version = ""
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                for line in f:
                    if line.startswith("scenario_version:"):
                        version = line.split(":", 1)[1].strip()
                        break
        if version in ("field_v5", "field_v5b"):
            rows.append(metrics_v5(run))
        elif version == "field_v4_1b":
            rows.append(metrics_v41b(run))
        elif version == "field_v4_1":
            rows.append(metrics_v41(run))
        elif version == "field_v4":
            rows.append(metrics_v4(run))
        else:
            rows.append(metrics(run))
    for run, row in zip(args.run, rows):
        if str(row.get("version", "")).startswith("field_v5"):
            with open(os.path.join(run, "metrics_v5.json"), "w", encoding="utf-8") as f:
                json.dump(row, f, ensure_ascii=False, indent=2)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    for row in rows:
        print("=" * 66)
        for key, value in row.items():
            print(f"{key:34s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
