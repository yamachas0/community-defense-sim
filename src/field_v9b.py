"""A市フィールド v9b「実行される約束だけが届く町」— v9 からの差分だけ。

設計の正は `docs/world_design_v9b.md`（土台＝`docs/world_design_v9.md`）。
施主確定 2026-08-30 13:10（A＋B・お金の概念は入れない・住民側は変えない）。

**v1〜v9 のファイルは1バイトも触らない。** v9b はこのファイルと `src/sim_v9b.py`、
`run_v9b.py`、`configs/config_field_v9b.yaml`、`tests/test_v9b.py`、`tools/v9b_*.py`
だけで完結する。`src/field_v9.py` は **読み取り専用で import** する。

v9 からの差分は2つだけ:
  A. **この世界で実行される仕組みが無い約束を含む提示を、世界が配らない。**
     （v9 の「相手が持っていない区画・種別の提示を配らない」と同じ原理の門番）
     配らなかった件数を数え、翌月 X社 に「配られなかった」事実と理由だけを返す。
  B. X社の設定に施主指定の事実を2行足す（情報であって当為ではない）。

町の人の側は1文字も変えない（前置き・行き先・会話・出品・売買・理由の一言）。
A の判定は**この世界の物理**であって、X社に攻略法を渡すものではない
（どの語が引っかかったかは返さない）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .field_v9 import (ACQUIRER_FACTS_V9, ACQUIRER_INTRO_V8C,  # noqa: F401
                       ACQUIRER_MANDATE_V9, ACQUIRER_NAME, KIND_BOTH,
                       KIND_BUILDING, KIND_LAND, KIND_VALUES, MAX_OFFER_CHARS,
                       MAX_REASON_CHARS, RegistryV9, _sale_and_rights_block,
                       acquirer_roster_rows_v9, acquirer_schema_v9,
                       fold_history_v9, ledger_rows_for_acquirer)
from .field_v9 import SELL_NO  # noqa: F401

# ---------------------------------------------------------------------------
# B. X社の設定に足す事実（施主文言 2026-08-30 13:10・言い換え不可）
# ---------------------------------------------------------------------------

ACQUIRER_FACTS_ADDED_V9B = (
    "土地と建物のセットにこだわらない。土地だけ、建物だけの取得でもよい。\n"
    "この世界で所有権の移転後に実行されるのは、使用者がそのまま使い続けること"
    "（借地・借家）だけである。支援・改修・管理・金銭の提供を実行する仕組みはない。\n"
)

# 既存の行は一字一句そのまま（末尾に足すだけ）。
ACQUIRER_FACTS_V9B = ACQUIRER_FACTS_V9 + ACQUIRER_FACTS_ADDED_V9B

# ---------------------------------------------------------------------------
# A. 「この世界で実行できない約束」の語彙表（走行前に凍結・結果を見てから足さない）
#    設計 docs/world_design_v9b.md §1-3 と同じもの。
# ---------------------------------------------------------------------------

# 強い語＝どんな言い方でもこの世界に実行する仕組みが無いもの（21語）
PROMISE_STRONG: Tuple[str, ...] = (
    "費用", "負担", "補助", "助成", "資金", "出資", "融資", "金銭", "対価",
    "謝礼", "報酬",
    "改修", "修繕", "改築", "建て替え", "建替え",
    "雇用", "就労", "提携", "紹介", "斡旋",
)

# 弱い語＝打ち消し、または「今までどおり使い続ける」と一緒なら真になりうるもの（11語）
PROMISE_WEAK: Tuple[str, ...] = (
    "支援", "サポート", "援助", "管理", "運営", "維持", "保証", "補償",
    "便宜", "手配", "提供",
)

# 打ち消しの言い方＝その節は約束ではない（16通り）
NEGATIONS: Tuple[str, ...] = (
    "ません",          # 丁寧な打ち消しの全般（行いません／生じません／ありません…）
    "行わない", "しない", "できない", "ない。", "ないため", "ないので",
    "ないこと", "持たない", "求めない", "不要", "一切ない", "存在しない",
    "無い", "ございません", "致しません",
)

# 実行される事実の言い方＝弱い語と同じ節にあれば真（12通り）
EXECUTED_FACTS: Tuple[str, ...] = (
    "使い続け", "住み続け", "営み続け", "借り続け", "借地", "借家",
    "引き続き住", "引き続き使", "引き続き営",
    "今までどおり", "これまでどおり", "変わらず",
)

# 節の区切り
CLAUSE_SEPARATORS = "。、\n！？!?"

# 配らなかった理由（X社に返す文言はこの2つだけ・評価語を入れない）
UNDELIVERED_RIGHTS = "相手がその区画のその種別の所有権を持っていない"
UNDELIVERED_PROMISE = "この世界で実行される仕組みが無い約束が含まれている"


def split_clauses(text: str) -> List[str]:
    """条件文を節に切る（決定論・句読点と改行だけで切る）。"""
    out: List[str] = []
    buf = ""
    for ch in str(text or ""):
        if ch in CLAUSE_SEPARATORS:
            # 「ない。」を打ち消しとして拾えるよう、句点は節の側に残す
            buf += ch if ch == "。" else ""
            if buf.strip():
                out.append(buf)
            buf = ""
        else:
            buf += ch
    if buf.strip():
        out.append(buf)
    return out


def undeliverable_promise(text: str) -> bool:
    """この世界で実行される仕組みが無い約束を含むか（決定論・意味の推定はしない）。

    設計 §1-2 の規則をそのまま実装する:
      1. 節に打ち消しがあれば約束ではない → 素通し
      2. 強い語があれば → 配らない
      3. 弱い語があり、実行される事実の言い方が無ければ → 配らない
    """
    for clause in split_clauses(text):
        if any(n in clause for n in NEGATIONS):
            continue
        if any(w in clause for w in PROMISE_STRONG):
            return True
        if any(w in clause for w in PROMISE_WEAK):
            if not any(f in clause for f in EXECUTED_FACTS):
                return True
    return False


def promise_word_hits(text: str) -> Dict[str, int]:
    """条件文に現れた語の回数（観測用・判定には使わない）。"""
    t = str(text or "")
    hits: Dict[str, int] = {}
    for w in list(PROMISE_STRONG) + list(PROMISE_WEAK):
        c = t.count(w)
        if c:
            hits[w] = c
    return hits


# ---------------------------------------------------------------------------
# X社の system プロンプト（B を反映・それ以外は v9 と同一）
# ---------------------------------------------------------------------------

def build_acquirer_prefix_v9b(cfg: Dict[str, Any], reg: RegistryV9) -> str:
    """v9 の `build_acquirer_prefix_v9` と同じ文＋ B の2行だけが違う。"""
    world = cfg["world"]
    return f"""あなたは架空都市「{world.get('town_name', 'A市')}」の外にある会社である。
時間は1か月単位で進む。

あなたが知ることができるのは、次に示す情報だけである。
すなわち不動産の登記の記録と、売りに出ているという公の申し出と、
あなた自身が出した提示とその結果である。
提示を受けた相手が「{SELL_NO}」と決めたとき、その相手が理由を一行書いていれば、
それはあなたに伝わる（書かないこともある）。
町の人が何を考えているか、どこで誰と何を話したかを、あなたは知らない。
それぞれの所有者がどこに住んでいるかも、あなたは知らない。
観測にない事実を補わない。

--- {world.get('town_name', 'A市')}の開始時点 ---
{str(world.get('background', '')).strip()}

--- この町の不動産の持ち主（開始時点・公開されている記録） ---
{chr(10).join(acquirer_roster_rows_v9(reg))}

{_sale_and_rights_block()}
不動産の所有権は公式に記録されており、あなたはその記録を見ることができる。
記録には区画ごとに土地の所有者と建物の所有者と、借りて使っている人が載っている。
不動産の所有権があなたに移るのは、あなたが条件を示し、相手がその条件で売ると決めたときだけである。
相手が売りに出しているというだけでは所有権は移らない。
売りに出していない相手に条件を示すこともできる。

{ACQUIRER_FACTS_V9B}
説明文を付けずJSONだけ返す。
"""


# ---------------------------------------------------------------------------
# 配られなかった提示を X社 に返す（事実と理由だけ・翌月）
# ---------------------------------------------------------------------------

def fold_undelivered_v9b(rows: List[Dict[str, Any]]) -> List[str]:
    """届かなかった提示を相手ごとに畳む（v9 の履歴と同じ形・評価語を入れない）。"""
    by_to: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for r in rows:
        to = str(r["to"])
        if to not in by_to:
            by_to[to] = []
            order.append(to)
        by_to[to].append(r)
    out: List[str] = []
    for to in order:
        items = sorted(by_to[to], key=lambda x: int(x["step"]))
        last = items[-1]
        out.append(f"  {to}: 届かなかった {len(items)}回")
        out.append(f"    直近 第{last['step']}月 {last['parcel']}の{last['kind']}"
                   f"「{last['text']}」")
        out.append(f"    → 届かなかった理由: {last['why']}")
    return out


def build_acquirer_prompt_v9b(reg: RegistryV9, step: int, n_steps: int,
                              targets: List[str],
                              offers: List[Dict[str, Any]],
                              listed_rows: List[str],
                              target_parcels: List[str],
                              chunk_no: int, chunk_total: int,
                              with_reason: bool = True,
                              undelivered: Optional[List[Dict[str, Any]]] = None
                              ) -> str:
    """X社の user プロンプト。v9 との差は2つだけ:
       ①届かなかった提示の節を足す ②届かない条件の告知を1行足す。"""
    rows = [ACQUIRER_MANDATE_V9, "",
            f"=== 第{step}月 / 全{n_steps}月 ===",
            "[登記簿（今の権利。公開されている記録）]"]
    rows += ledger_rows_for_acquirer(reg)
    rows += ["", "[売りに出ているという申し出（公の記録・先月の終わりに出されたもの）]"]
    if not listed_rows:
        rows.append("  （先月、売りに出された不動産はない）")
    else:
        rows += [f"  {r}" for r in listed_rows]
    rows += ["", "[あなたが今までに出した提示と、その結果]"]
    folded = fold_history_v9(offers)
    rows += folded if folded else ["  （まだ何も出していない）"]
    rows += ["", "[相手に届かなかった提示（この世界で実行できないもの）]"]
    folded_u = fold_undelivered_v9b(list(undelivered or []))
    rows += folded_u if folded_u else ["  （届かなかった提示はない）"]
    rows += ["", "[今回あなたが判断する相手]"]
    for name in targets:
        rows.append(f"  {name}")
    if chunk_total > 1:
        rows.append(f"  （今月の持ち主を分けて尋ねている。{chunk_no}／{chunk_total}回目。"
                    "登記簿と履歴は毎回すべて示している）")
    rows += ["", "[この回に選べる区画]"]
    for p in target_parcels:
        rows.append(f"  {p}")
    rows += ["",
             "上の相手それぞれについて、今月あなたが提示を出すかどうかを決める。",
             "出すなら send を true にし、parcel にその相手の区画を1つ、"
             f"kind に「{KIND_LAND}」「{KIND_BUILDING}」「{KIND_BOTH}」のいずれかを書き、"
             f"text にその相手へ送る条件を1行で書く（{MAX_OFFER_CHARS}字以内）。",
             "出さないなら send を false にし、text は空文字にする。",
             "相手が所有権を持っていない区画や種別を書いた提示は相手に届かない。",
             "この世界で実行される仕組みが無い約束を含む提示も相手に届かない。",
             "この世界に金銭は存在しないので、金額や価格は書けない。",
             "text はそのまま相手に届く。text にはその相手に提示する条件だけを書く。"
             "あなた自身の目的や判断の理由、町全体についての方針、"
             "他の相手に何を出しているかは書かない。",
             "相手はあなたが他の人にも出しているかどうかを知らない。",
             f"あなたの提示には、冒頭に「{ACQUIRER_INTRO_V8C}」という1行が"
             "必ず添えられて相手に届く。text にはこの1行を書かない。"]
    if with_reason:
        rows += [f"reason には、その判断の理由を一行で書く（{MAX_REASON_CHARS}字以内・"
                 "書かない場合は空文字）。reason は相手には届かない。"]
    rows += ["説明文を付けずJSONだけ返す。"]
    return "\n".join(rows)


__all__ = [
    "ACQUIRER_FACTS_ADDED_V9B", "ACQUIRER_FACTS_V9B",
    "PROMISE_STRONG", "PROMISE_WEAK", "NEGATIONS", "EXECUTED_FACTS",
    "UNDELIVERED_RIGHTS", "UNDELIVERED_PROMISE",
    "split_clauses", "undeliverable_promise", "promise_word_hits",
    "build_acquirer_prefix_v9b", "fold_undelivered_v9b",
    "build_acquirer_prompt_v9b", "acquirer_schema_v9",
]
