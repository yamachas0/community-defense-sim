"""A市フィールド v6: 町の人に「中立な行動の選択肢」を3段だけ置く層。

設計の正は `docs/world_design_v6_two_worlds.md`。
v1〜v5e2 のファイル・設定・成果物は変えない（v6 は足すだけ）。

v5e2 との差はこれで全部である。

  1. 登記簿に `refusal` の1列（持ち主が「当面売らない」を宣言中か）。決定論。
  2. 主体の毎月の出力に行動欄（enum・既定は先頭の「変えない」「なし」）。
     個人 `sell_intent` ／ 集団・公的 `public_act` ／ 行政 `measure`。
  3. 回覧板・町内会の議題・役場への申入れ・行政の措置は、**翌月の紙**として
     街の全員に届く（配送は記事とまったく同じ型＝私信ではない・公の記録）。
  4. 買い手の台本は機械的に従う：予定の土地が `refusal` ならその月は買えず次の予定へ。

ここに置いてよいのは**選択肢**（可能な行為）だけである。促し文・兆候・観測者・
確率・閾値・当為・「行動させるための仕組み」は1つも置かない。
採点用の LLM は v6 には存在しない（config の `kpi.classify_utterances: false`）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .field_v5 import acquisitions_at
from .field_v5d import scene_schema_v5d
from .world import Ledger

# --- 行動の選択肢（enum・先頭が既定＝「変えない／なし」）---------------------
# 並びも文言も中立にする。どれかを勧める語・強める語は入れない。

SELL_INTENT_KEEP = "変えない"
SELL_INTENT_REFUSE = "当面売らない"
SELL_INTENT_CLEAR = "売らないを解除"
SELL_INTENT_VALUES = [SELL_INTENT_KEEP, SELL_INTENT_REFUSE, SELL_INTENT_CLEAR]

PUBLIC_ACT_NONE = "なし"
PUBLIC_ACT_CIRCULAR = "回覧板に載せる"
PUBLIC_ACT_ASSEMBLY = "町内会の議題にする"
PUBLIC_ACT_PETITION = "役場へ申し入れる"
PUBLIC_ACT_VALUES = [PUBLIC_ACT_NONE, PUBLIC_ACT_CIRCULAR,
                     PUBLIC_ACT_ASSEMBLY, PUBLIC_ACT_PETITION]

MEASURE_NONE = "なし"
MEASURE_DESK = "相談窓口を開く"
MEASURE_BRIEFING = "説明会を開く"
MEASURE_STUDY = "条例・届出の検討を始める"
MEASURE_VALUES = [MEASURE_NONE, MEASURE_DESK, MEASURE_BRIEFING, MEASURE_STUDY]

# 紙の見出し（翌月に街の全員が見るときの表示）。
PAPER_LABELS = {
    PUBLIC_ACT_CIRCULAR: "回覧板",
    PUBLIC_ACT_ASSEMBLY: "町内会の議題",
    PUBLIC_ACT_PETITION: "役場への申入れ",
}
MEASURE_PAPER_LABEL = "役場からのお知らせ"

MAX_ACT_TEXT_CHARS = 150

# 行動の欄（v6b ではこれだけを必須から外す）
ACTION_FIELDS = ("sell_intent", "public_act", "public_act_text",
                 "measure", "measure_text")


# --- 出力スキーマ -----------------------------------------------------------

def scene_schema_v6(agent_name: str, present_names: List[str],
                    all_names: List[str], owns_parcel: bool = False,
                    can_publish: bool = False, ask_actions: bool = False,
                    is_municipality: bool = False,
                    required_actions: bool = True) -> Dict[str, Any]:
    """`scene_schema_v5d` に行動欄を足しただけ（構造も既存の項目も同一）。

    `ask_actions` は**その主体のその月の最後のターンでだけ** True にする
    （stance と同じ理由＝毎ターン尋ねると売却の話題を想起させ続ける＝
    世界側のプライミングになる）。
    """
    schema = scene_schema_v5d(agent_name, present_names, all_names,
                              owns_parcel, can_publish)
    props = schema["properties"]
    if ask_actions:
        if owns_parcel:
            props["sell_intent"] = {"type": "string", "enum": list(SELL_INTENT_VALUES)}
        if is_municipality:
            props["measure"] = {"type": "string", "enum": list(MEASURE_VALUES)}
            props["measure_text"] = {"type": "string"}
        else:
            props["public_act"] = {"type": "string", "enum": list(PUBLIC_ACT_VALUES)}
            props["public_act_text"] = {"type": "string"}
    if required_actions:
        schema["required"] = list(props)
    else:
        # v6b（任意回答）：欄はあるが**必須にしない**＝毎月の問いかけが無く、
        # 本人が使いたい月にだけ書く。書かなければ既定（変えない／なし）として扱う。
        schema["required"] = [k for k in props
                              if k not in ACTION_FIELDS]
    return schema


# --- プロンプトに足す文（中立・これで全文）----------------------------------

def action_rows_v6(owns_parcel: bool, is_municipality: bool) -> List[str]:
    """最後のターンにだけ足す行動欄の説明。選択肢を示すだけで、促さない。

    文言は Codex レビュー（2026-08-29 走行前）を受けて中立側へ寄せてある：
    「何かするか」「取る措置」のような行動を意識させる問いかけの形をやめ、
    「街の全員が見られる」という到達力の宣伝もやめた。残しているのは
    **選ばれたものが世界でどうなるか**という事実だけである（それが無いと
    選択の意味が分からない＝選べない）。
    """
    rows: List[str] = []
    if owns_parcel:
        rows += [f"sell_intent は次のどれかを書く（{SELL_INTENT_KEEP}／"
                 f"{SELL_INTENT_REFUSE}／{SELL_INTENT_CLEAR}）。書かなければ"
                 f"「{SELL_INTENT_KEEP}」として扱う。",
                 f"「{SELL_INTENT_REFUSE}」は、解除するまで自分が持っている土地の"
                 "売買が成立しないことを意味する（貸し借りには関わらない）。"]
    if is_municipality:
        rows += [f"measure は次のどれかを書く（{MEASURE_NONE}／{MEASURE_DESK}／"
                 f"{MEASURE_BRIEFING}／{MEASURE_STUDY}）。書かなければ"
                 f"「{MEASURE_NONE}」として扱う。",
                 f"「{MEASURE_NONE}」以外のときは measure_text に1行書く。"
                 "その1行は翌月、紙として街に配られる。"]
    else:
        rows += [f"public_act は次のどれかを書く（{PUBLIC_ACT_NONE}／"
                 f"{PUBLIC_ACT_CIRCULAR}／{PUBLIC_ACT_ASSEMBLY}／"
                 f"{PUBLIC_ACT_PETITION}）。書かなければ「{PUBLIC_ACT_NONE}」として扱う。",
                 f"「{PUBLIC_ACT_NONE}」以外のときは public_act_text に1行書く。"
                 "その1行は翌月、紙として街に配られる。"]
    return rows


# --- v6b：選択肢の存在を「世界の知識」として1回だけ書く -------------------
# v6 は毎月の最後のターンで行動欄の説明を出していた（＝毎月かならず答える）。
# v6b はそれをやめ、**system プロンプトに1回だけ**制度の存在を書く。
# 毎月の問いかけはせず、欄も必須にしない（本人が使いたい月にだけ書く）。
# 文面は制度の説明だけで、使うことも使わないことも勧めない。

WORLD_KNOWLEDGE_V6B = f"""
--- この街にある手続き ---
土地の持ち主は、自分の土地を「{SELL_INTENT_REFUSE}」としておくことができる。その間、
その人の土地の売買は成立しない。この扱いはあとで解除できる。
街に向けた手続きとして、「{PUBLIC_ACT_CIRCULAR}」「{PUBLIC_ACT_ASSEMBLY}」
「{PUBLIC_ACT_PETITION}」がある。その内容は翌月、紙として街に配られる。
市役所の担当が行えるものとして、「{MEASURE_DESK}」「{MEASURE_BRIEFING}」
「{MEASURE_STUDY}」がある。
どの手続きを使うか、またはいずれも使わないかは、各人が決める。
"""

# v6b では、行政ロール向けの既存の一文と制度の説明が文字どおり矛盾する
# （「特別な権限や制度はこの街には無い」と「条例・届出の検討」が同居できない）。
# **v5d の原文は旧世界では正しいので変えず**、v6b の組み立て側でこの一文だけを外す
# （Codexレビュー 2026-08-29 走行前）。
MUNI_LINE_V5D = "市役所の待合で人の話を聞くことがある。特別な権限や制度はこの街には無い。"
MUNI_LINE_V6B = "市役所の待合で人の話を聞くことがある。"


def system_prompt_v6b(text: str) -> str:
    """v6b の system プロンプト＝v5d の文面 − 矛盾する一文 ＋ 制度の説明1回。"""
    return text.replace(MUNI_LINE_V5D, MUNI_LINE_V6B) + WORLD_KNOWLEDGE_V6B



# --- 登記簿の refusal 列 ----------------------------------------------------

def is_refused(ledger: Ledger, parcel_id: str) -> bool:
    parcel = ledger.parcels.get(parcel_id)
    return bool(getattr(parcel, "refusal", False)) if parcel is not None else False


def set_refusal(ledger: Ledger, step: int, agent_id: str,
                refuse: bool) -> List[str]:
    """その主体が持つ全区画の `refusal` を書き換え、変わった区画IDを返す。

    自分の土地にだけ効く（他人の土地には触れない）。決定論であり、
    ここで主体に何かをさせることはしない。
    """
    changed: List[str] = []
    for pid in sorted(ledger.parcels):
        parcel = ledger.parcels[pid]
        if parcel.owner_id != agent_id:
            continue
        if bool(getattr(parcel, "refusal", False)) == bool(refuse):
            continue
        parcel.refusal = bool(refuse)
        ledger._rec(step, "refusal_set" if refuse else "refusal_clear",
                    by=agent_id, parcel_id=pid)
        changed.append(pid)
    return changed


# --- 買い手の台本の従い方 ---------------------------------------------------

def blocked_acquisitions_v6(ledger: Ledger, step: int,
                            script: Dict[str, Any]) -> List[Dict[str, Any]]:
    """その月の予定のうち、対象が「当面売らない」で買えないもの。

    **売買（sale）だけを止める。** 賃借（lease）は止めない＝選択肢の文言が
    「当面売らない」だからである（文言と実装を一致させる・Codexレビュー
    2026-08-29 走行前指摘）。台本の45件のうち賃借は止まらない。
    """
    return [acq for acq in acquisitions_at(script, step)
            if str(acq.get("kind", "sale")) == "sale"
            and is_refused(ledger, str(acq["parcel_id"]))]


def script_without(script: Dict[str, Any],
                   acq_ids: Any) -> Dict[str, Any]:
    """指定の取得を外した台本の写し（元の台本は書き換えない）。

    順番も月も動かさない＝**買えなかった取得は消えるだけで、後で買い直さない**
    （買い手に新しい行動を足さないため）。
    """
    ids = {str(a) for a in acq_ids}
    if not ids:
        return script
    out = dict(script)
    out["acquisitions"] = [a for a in script.get("acquisitions", [])
                           if str(a["id"]) not in ids]
    return out


# --- 紙（公の記録）----------------------------------------------------------

def paper_row(step: int, agent_id: str, role: str, name: str,
              act: str, label: str, text: str) -> Dict[str, Any]:
    return {"step": step, "from": agent_id, "role": role, "name": name,
            "act": act, "label": label, "text": text[:MAX_ACT_TEXT_CHARS]}


def first_month(rows: List[Dict[str, Any]],
                pred=None) -> Optional[int]:
    """条件に当たる最初の月（無ければ None）。集計の共通部品。"""
    months = [int(r.get("step", 0)) for r in rows if (pred is None or pred(r))]
    return min(months) if months else None


__all__ = [
    "SELL_INTENT_KEEP", "SELL_INTENT_REFUSE", "SELL_INTENT_CLEAR",
    "SELL_INTENT_VALUES", "PUBLIC_ACT_NONE", "PUBLIC_ACT_CIRCULAR",
    "PUBLIC_ACT_ASSEMBLY", "PUBLIC_ACT_PETITION", "PUBLIC_ACT_VALUES",
    "MEASURE_NONE", "MEASURE_DESK", "MEASURE_BRIEFING", "MEASURE_STUDY",
    "MEASURE_VALUES", "PAPER_LABELS", "MEASURE_PAPER_LABEL",
    "MAX_ACT_TEXT_CHARS", "scene_schema_v6", "action_rows_v6",
    "is_refused", "set_refusal", "blocked_acquisitions_v6", "script_without",
    "paper_row", "first_month", "ACTION_FIELDS", "WORLD_KNOWLEDGE_V6B", "system_prompt_v6b",
]
