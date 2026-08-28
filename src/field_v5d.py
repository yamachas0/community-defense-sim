"""A市フィールド v5d: 役場の窓口を廃止し、土地と持ち主を街の呼び名で呼ぶ層。

設計の正は `docs/world_design_v5d.md`。v5 / v5b / v5c のファイルと成果物は変えない
（共通コードに触れた箇所は既定値で従来どおりの結果になる）。

v5c との差は3つだけで、どれも **世界が住民に渡している入力の非現実を消す** ものである。
気づかせる経路・観測者・機構・語彙・確率・閾値は1本も足していない。

  1. 市役所の窓口（S4 の counter）を廃止する。S4 は町内会／仲介の店先／記者の取材の
     3か月周期になり、`registry_rows_v5`（窓口で閲覧できる登記の全件）は一度も呼ばれない。
     会場としての「市役所の待合」(V06) は S1〜S3 の行き先として残る。
  2. 土地と主体を内部ID（P01 / HH01 / V01）ではなく街の呼び名で呼ぶ。
     対応表は `configs/parcel_names_v5c.yaml` の1枚だけ（画面とシミュで食い違わないため）。
  3. 兆候の文言のうち、名義の移転では起きない書き方を直す（看板・使い手）。
     どの兆候が実際に残るかは台本（configs/events_v5d_seed85.yaml）が決める。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .agents import Agent
from .field_v5 import (DIRECT_NONE, HOME, MAX_PUBLISH_CHARS, MAX_TEXT_CHARS,
                       MAX_THOUGHT_CHARS, ROLE_TEXT_V5, SCENE_LABELS,
                       STANCE_VALUES, TRACE_TEXTS)

# --- S4（月替わりの集まり）------------------------------------------------
# 施主 2026-08-28 20:41「普通わざわざ窓口行って登記確認しない」。counter を落とす。
S4_ROTATION_V5D: List[Tuple[str, str, str]] = [
    ("assembly", "V02", "町内会（地区公民館の寄り合い）"),
    ("broker_front", "V07", "仲介の店先（不動産仲介の事務所）"),
    ("press", "V08", "記者の取材（地域イベント会場に記者が来ている）"),
]

S4_KINDS_V5D = [k for k, _v, _l in S4_ROTATION_V5D]


def s4_for_step_v5d(step: int) -> Tuple[str, str, str]:
    return S4_ROTATION_V5D[(int(step) - 1) % len(S4_ROTATION_V5D)]


# --- 兆候の文言 -------------------------------------------------------------
# 名義が移っただけでは看板は替わらない／賃借の対象は使い手が居ない区画に限られる
# （docs/world_design_v5d.md §3）。文言だけを直す。規則・本数は台本が決める。
TRACE_TEXTS_V5D = dict(TRACE_TEXTS)
TRACE_TEXTS_V5D["sign_change"] = "{parcel}に見慣れない名前の看板が掛かった"
TRACE_TEXTS_V5D["tenant_swap"] = "{parcel}を使う人が変わった"


# --- 呼び名の対応表 ---------------------------------------------------------

# 対応表 configs/parcel_names_v5c.yaml は記者の内部IDを `ME01/ME02` と書いているが、
# 実際の名簿（src/agents.build_roster）の記者は `MD01/MD02` である。表は凍結されている
# ので、読み込み側で内部IDだけを名簿に合わせる（呼び名そのものは1文字も変えない）。
# 表のキーを直すかどうかは要施主・CTO判断（docs/world_design_v5d.md §1-2）。
AGENT_ID_ALIAS = {"ME01": "MD01", "ME02": "MD02"}


def load_names_v5d(path: str) -> Dict[str, Dict[str, str]]:
    """`configs/parcel_names_v5c.yaml` を読む（v5d 用の別表は作らない）。

    返すのは3つの辞書だけ：
      parcels    区画の内部ID → 土地の名前
      agents     主体の内部ID → 呼び名
      registered v5c までの登記名義の表示 → v5d の表示（法人名義は同じ値）
    """
    import yaml
    with open(path, encoding="utf-8") as f:
        book = yaml.safe_load(f)
    parcels = {pid: str(row["name"]) for pid, row in (book.get("land") or {}).items()}
    agents = {AGENT_ID_ALIAS.get(aid, aid): str(row["name"])
              for aid, row in (book.get("agents") or {}).items()}
    registered = {str(k): str(v) for k, v in (book.get("registered_names") or {}).items()}
    if len(set(parcels.values())) != len(parcels):
        raise ValueError("土地の名前が重複している")
    if len(set(agents.values())) != len(agents):
        raise ValueError("主体の呼び名が重複している")
    return {"parcels": parcels, "agents": agents, "registered": registered}


def venue_labels_v5d(cfg: Dict[str, Any]) -> Dict[str, str]:
    """会場の内部ID → 表示（会場名だけ。V番号は出さない）。"""
    labels = {str(v["id"]): str(v["label"]) for v in cfg.get("social", {}).get("venues", [])}
    if len(set(labels.values())) != len(labels):
        raise ValueError("会場名が重複している")
    return labels


# --- 出力スキーマ（構造は v5 と同一・enum の値だけが呼び名になる）-----------

def scene_schema_v5d(agent_name: str, present_names: List[str],
                     all_names: List[str], owns_parcel: bool = False,
                     can_publish: bool = False) -> Dict[str, Any]:
    """`scene_schema_v5` と同じ構造。並ぶ値が内部IDでなく呼び名なだけである。"""
    others = [p for p in present_names if p != agent_name]
    elsewhere = [a for a in all_names if a != agent_name]
    props: Dict[str, Any] = {
        "thought": {"type": "string"},
        "text": {"type": "string"},
        "talk_to": {"type": "array",
                    "items": {"type": "string", "enum": others or list(present_names)}},
        "direct_to": {"type": "string", "enum": [DIRECT_NONE] + elsewhere},
        "direct_text": {"type": "string"},
    }
    if can_publish:
        props["publish"] = {"type": "string"}
    if owns_parcel:
        props["stance"] = {"type": "string", "enum": STANCE_VALUES}
    return {"type": "object", "properties": props, "required": list(props)}


# --- system プロンプト -------------------------------------------------------
# v5 の文面の写しであり、直したのは docs/world_design_v5d.md §2 が挙げた3か所と、
# 内部IDを出していた2か所（あなた／talk_to・direct_to の相手の指定）だけである。

def build_system_prompt_v5d(agent: Agent, cfg: Dict[str, Any], n_parcels: int,
                            venue_ids: List[str]) -> str:
    world = cfg["world"]
    labels = venue_labels_v5d(cfg)
    venue_rows = "\n".join(f"  {labels[v]}" for v in venue_ids if v in labels)
    quota = int(cfg.get("scenario", {}).get("direct_quota_per_month", 2))
    text = f"""あなたは架空都市「{world.get('town_name', 'A市')}」で暮らす、働く、または活動する一主体である。
時間は1か月単位で進み、街には{n_parcels}区画がある。

あなたは全知ではない。自分の身の回りのこと、実際に居合わせた場所で聞いた発言、
自分に届いた連絡、自分の目で見たことだけを使う。観測にない事実を補わない。
人から聞いた話は誤っている可能性がある。感じ方と行動はあなた自身が決める。
他の主体が何を考えているかをあなたは知らない。

--- A市の開始時点 ---
{str(world.get('background', '')).strip()}

--- あなたの立場 ---
{ROLE_TEXT_V5[agent.role]}

--- あなた ---
{agent.name}
{agent.persona}

--- 月の過ごし方（シーン） ---
1か月には4つの場面がある。
  S1 {SCENE_LABELS['S1']}
  S2 {SCENE_LABELS['S2']}
  S3 {SCENE_LABELS['S3']}
  S4 その月ごとの集まり（町内会・仲介の店先・記者の取材が月替わりで開かれる）
月の初めに、S1〜S4 をどこで過ごすかを決める。行ける場所は次のとおり。
{venue_rows}
  {HOME}: 自宅・自分の店から出ない

同じ場所に居合わせた者どうしは、その場面で会話する。話した言葉はその場に居た全員に
そのまま聞こえる。居合わせた者がいなければ何も起きない。街全体に流れる共通の掲示板は無い。

--- 話すこと ---
その場で話したいことがなければ text を空文字にしてよい（黙っていることも普通のことである）。
talk_to は、その発言をとくに向けた相手の呼び名（居合わせた者のみ・複数可・空でよい）。
別の場所にいる相手へ個人的に連絡したい月は direct_to に相手の呼び名、direct_text に
その中身を書く（1か月に{quota}通まで・翌月に相手へ届く）。使わない月は direct_to を {DIRECT_NONE} にする。

--- 土地の登記 ---
土地の名義は公式に記録されている。ただしあなたはその記録を見ていない。
名義が変わったことは、当事者から聞くか、目に見える様子から察するほかない。
自分の土地、自分が使っている店舗、隣接する区画の名義は日ごろ目に入る。
この世界に金銭の授受は存在しない。土地は名義が移るかどうかだけがある。

--- thought（内心） ---
JSONの最初に thought を書く。thought は誰にも伝わらないあなたの内心であり、
次の場面と翌月のあなたにそのまま渡される。何を書くかはあなたが決める。
まず thought を書き、それを踏まえてその後を書く。目安は{MAX_THOUGHT_CHARS}字以内、
発言は{MAX_TEXT_CHARS}字以内。説明文を付けずJSONだけ返す。
"""
    if agent.role == "media":
        text += f"""
--- 記者としてできること ---
その場面のあとで記事を出すことができる（publish）。記事は翌月、市内の全員が読む。
出せるのは1か月に1本で、書かない場面は publish を空文字にする。
記事は{MAX_PUBLISH_CHARS}字以内。
"""
    if agent.role == "municipality":
        text += """
--- 市役所の担当としてできること ---
市役所の待合で人の話を聞くことがある。特別な権限や制度はこの街には無い。
"""
    return text


__all__ = ["S4_ROTATION_V5D", "S4_KINDS_V5D", "s4_for_step_v5d", "TRACE_TEXTS_V5D",
           "load_names_v5d", "venue_labels_v5d", "scene_schema_v5d",
           "build_system_prompt_v5d"]
