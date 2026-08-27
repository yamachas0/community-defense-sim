"""KPI6 の観測。

全て「帳簿の記録」と「エージェントが実際に発した言葉」からの事後集計であり、
シミュレーションの進行には一切影響しない（観測が世界を動かさない）。

  ① 所有集中度   : 買い手シェア % と HHI
  ② 認知転相率   : 住民の公開発話のうち「あの主体の街」フレームの割合（LLM分類）
  ③ 売却カスケード: 直前W月以内に隣接区画が売れた直後の売却＝誘発売却の数と最大連鎖長
  ④ 手遅れ度     : 規制発動時点の買い手シェア
  ⑤ 検知ラグ     : 買収を扱う初報道の時点の買い手シェア
  ⑥ 生活KPI     : 地元事業者の残存率・住民の転出率
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .world import Ledger, neighbors


def step_metrics(ledger: Ledger, step: int, acquirer_ids: List[str],
                 household_ids: List[str], business_ids: List[str],
                 cascade_window: int = 3) -> Dict[str, Any]:
    share = ledger.ownership_share(acquirer_ids)
    hhi = ledger.hhi()
    casc = cascade_stats(ledger, until_step=step, window=cascade_window)
    n_hh = len(household_ids) or 1
    n_bz = len(business_ids) or 1
    open_shops = [p for p in ledger.parcels.values() if p.use == "shop" and p.tenant_id]
    all_shops = [p for p in ledger.parcels.values() if p.use == "shop"]
    return {
        "step": step,
        "acquirer_share": round(share, 4),
        "hhi": round(hhi, 4),
        "block_share": {k: round(v, 4) for k, v in ledger.block_share(acquirer_ids).items()},
        "transfers_cum": len(ledger.transfers()),
        "cascade_induced": casc["induced"],
        "cascade_max_chain": casc["max_chain"],
        "business_survival": round(1 - len(ledger.closed_businesses) / n_bz, 4),
        "resident_outflow": round(len(ledger.moved_out) / n_hh, 4),
        "shops_occupied": len(open_shops),
        "shops_total": len(all_shops),
        "mean_shop_rent": round(
            sum(p.rent for p in all_shops) / len(all_shops), 1) if all_shops else 0.0,
        "ordinance_enacted": bool(ledger.ordinances),
        "published_acquisition": any(p["about_acquisition"] for p in ledger.publications),
    }


def cascade_stats(ledger: Ledger, until_step: int, window: int = 3) -> Dict[str, int]:
    """隣接区画の直近売却に続いた売却＝誘発売却を数える。

    因果を主張するものではなく「隣で売れた直後に売れた」という時空間的近接の観測。
    """
    transfers = [t for t in ledger.transfers() if t["step"] <= until_step]
    by_parcel_step = [(t["parcel_id"], t["step"]) for t in transfers]
    induced = 0
    chain_len: Dict[str, int] = {}
    max_chain = 0
    for pid, st in by_parcel_step:
        nb = set(neighbors(ledger.parcels, pid))
        prior = [(q, s) for q, s in by_parcel_step
                 if q in nb and st - window <= s < st]
        if prior:
            induced += 1
            best = max(chain_len.get(q, 1) for q, _ in prior)
            chain_len[pid] = best + 1
        else:
            chain_len[pid] = 1
        max_chain = max(max_chain, chain_len[pid])
    return {"induced": induced, "max_chain": max_chain}


def late_index(ledger: Ledger, acquirer_ids: List[str],
               share_by_step: Dict[int, float]) -> Optional[Dict[str, Any]]:
    """④手遅れ度: 最初の規制発動時点の買い手シェア。

    2つ出す。全主体は月初の状態を見て同時に判断するので、判断者が実際に見えていたのは
    **前月末**のシェア (share_observable)。同月末シェア (share_at_enactment) には、
    その月に同時進行で成立した取得も含まれる。どちらも記録して取り違えを防ぐ。
    """
    if not ledger.ordinances:
        return None
    o = ledger.ordinances[0]
    return {"step": o["step"], "title": o["title"],
            "share_at_enactment": share_by_step.get(o["step"]),
            "share_observable": share_by_step.get(o["step"] - 1, 0.0)}


def detection_lag(ledger: Ledger, share_by_step: Dict[int, float],
                  about_steps: Optional[List[int]] = None) -> Optional[Dict[str, Any]]:
    """⑤検知ラグ: 買収を扱う初報道の時点の買い手シェア。

    about_steps を渡すと、そちら (LLM分類の結果) を「買収を扱った報道」の判定に使う。
    渡さない場合は publish 時のキーワード判定にフォールバックする。
    """
    if about_steps is not None:
        pubs = [p for p in ledger.publications if p["step"] in set(about_steps)]
    else:
        pubs = [p for p in ledger.publications if p["about_acquisition"]]
    if not pubs:
        return None
    p = min(pubs, key=lambda r: r["step"])
    return {"step": p["step"], "headline": p["headline"],
            "share_at_first_report": share_by_step.get(p["step"]),
            "share_observable": share_by_step.get(p["step"] - 1, 0.0)}


# ---------------------------------------------------------------------------
# ② 認知転相率 — LLM による発話の意味分類
# ---------------------------------------------------------------------------

def classify_utterances(client, utterances: List[Dict[str, Any]], batch: int = 25,
                        temperature: float = 0.0, max_tokens: int = 1400) -> List[Dict[str, Any]]:
    """住民・事業者の公開発話を our_town / their_town / neutral に分類する。

    分類器はシミュレーション本体と分離され、結果は世界に戻らない（純粋な観測）。
    """
    from .prompts import CLASSIFY_SYSTEM, build_classify_prompt
    from .schemas import CLASSIFY_SCHEMA

    out: List[Dict[str, Any]] = []
    for i in range(0, len(utterances), batch):
        chunk = utterances[i:i + batch]
        raw = client.generate(
            CLASSIFY_SYSTEM,
            build_classify_prompt(chunk),
            schema=CLASSIFY_SCHEMA,
            temperature=temperature,
            max_tokens=max_tokens,
            tag="classify",
        )
        parsed: Dict[int, Dict[str, Any]] = {}
        try:
            data = json.loads(raw) if raw else {}
            for r in data.get("results", []):
                parsed[int(r["id"])] = r
        except Exception:
            parsed = {}
        for j, u in enumerate(chunk, start=1):
            r = parsed.get(j, {})
            out.append({**u,
                        "frame": r.get("frame", "unclassified"),
                        "about_acquisition": bool(r.get("about_acquisition", False))})
    return out


def classify_publications(client, publications: List[Dict[str, Any]],
                         batch: int = 25) -> List[int]:
    """報道が土地取得を扱っているかを LLM に判定させ、該当する step のリストを返す。

    publish 時のキーワード判定は取りこぼし・誤検知があるため、⑤検知ラグの確定には
    こちらの事後分類を使う（観測であって世界には戻らない）。
    """
    if not publications:
        return []
    rows = [{"step": p["step"], "text": p["headline"]} for p in publications]
    classified = classify_utterances(client, rows, batch=batch)
    return sorted({c["step"] for c in classified if c.get("about_acquisition")})


def cognition_series(classified: List[Dict[str, Any]], n_steps: int) -> List[Dict[str, Any]]:
    """step ごとの認知転相率（their/(our+their)）と累積値。"""
    per: Dict[int, Dict[str, int]] = {}
    for u in classified:
        d = per.setdefault(u["step"], {"our_town": 0, "their_town": 0, "neutral": 0})
        f = u.get("frame")
        if f in d:
            d[f] += 1
    rows = []
    cum_our = cum_their = 0
    for s in range(1, n_steps + 1):
        d = per.get(s, {"our_town": 0, "their_town": 0, "neutral": 0})
        cum_our += d["our_town"]
        cum_their += d["their_town"]
        denom = d["our_town"] + d["their_town"]
        cum_denom = cum_our + cum_their
        rows.append({
            "step": s,
            "our_town": d["our_town"],
            "their_town": d["their_town"],
            "neutral": d["neutral"],
            "shift_rate": round(d["their_town"] / denom, 4) if denom else None,
            "shift_rate_cum": round(cum_their / cum_denom, 4) if cum_denom else None,
        })
    return rows

# ---------------------------------------------------------------------------
# v5b: 「占領の認知」の事後分類（観測であって世界には戻らない）
# ---------------------------------------------------------------------------

OCCUPATION_SYSTEM = """あなたは会話ログの分類器である。架空都市A市の住民たちの発言・内心を読み、
2つの点だけを判定する。判定に使ってよいのは渡された文そのものだけで、背景を推測しない。

links_multiple: その文が、2つ以上の会社名（A社・B社・C社・D社など）または2つ以上の区画（P01形式）を
  「同じ相手の動き」「関連する一連の出来事」として結びつけているなら true。
  単に1件の取引や1社に触れているだけなら false。並べただけで関連づけていないなら false。
intent: その文が、地区や街ぐるみで買い集める・地上げ・乗っ取り・支配・「この辺一帯」「街ごと」
  といった、広がりのある意図や動きに言及しているなら true。個別の売買や空き家の話だけなら false。

JSONだけを返す。"""


OCCUPATION_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "integer"},
            "links_multiple": {"type": "boolean"},
            "intent": {"type": "boolean"},
        }, "required": ["id", "links_multiple", "intent"]}},
    },
    "required": ["results"],
}


def build_occupation_prompt(rows: List[Dict[str, Any]]) -> str:
    out = ["次の文をそれぞれ判定して results を返す。"]
    for i, r in enumerate(rows, start=1):
        out.append(f"{i}. {str(r.get('text', ''))[:400]}")
    return "\n".join(out)


def classify_occupation(client, rows: List[Dict[str, Any]],
                        batch: int = 25) -> List[Dict[str, Any]]:
    """発話・内心が『複数を結びつけているか』『街ぐるみの意図に触れたか』を事後分類する。"""
    out: List[Dict[str, Any]] = []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        raw = client.generate(OCCUPATION_SYSTEM, build_occupation_prompt(chunk),
                              schema=OCCUPATION_SCHEMA, temperature=0.0,
                              max_tokens=1400, tag="classify_occupation")
        parsed: Dict[int, Dict[str, Any]] = {}
        try:
            for r in (json.loads(raw) if raw else {}).get("results", []):
                parsed[int(r["id"])] = r
        except Exception:
            parsed = {}
        for j, row in enumerate(chunk, start=1):
            r = parsed.get(j)
            out.append({**row,
                        "links_multiple": bool(r.get("links_multiple")) if r else None,
                        "intent": bool(r.get("intent")) if r else None,
                        "classified": r is not None})
    return out
