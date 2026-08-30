"""A市フィールド v8「最小の町」— 世界の状態と文面。

設計の正は `docs/world_design_v8_minimal.md`。

**v1〜v6b のファイルは1バイトも触らない。** v8 はこのファイルと `src/sim_v8.py`、
`run_v8.py`、`tools/v8_curves.py`、`tests/test_v8.py` だけで完結する
（`src/simulation.py` にも分岐を足さない）。共有するのは LLM クライアントだけである。

この層に置いてよいのは **世界の事実と選択肢** だけである。
促し文・兆候・観測者・確率・閾値・当為・「行動させるための仕組み」は1つも置かない。
住民のプロンプトに「気づけ」「警戒しろ」「相談しろ」に類する語は無い（tests で固定）。

Codex 走行前レビュー（2026-08-29）で直した点：
  - 同じ提示文を月内に3回見せていた（月初・集まり・月末）→ **集まりでは見せない**（2回）。
  - 2択の説明が「売る」側に安心材料、「売らない」側に反復負担を付けていた → 対称な状態記述へ。
  - 選択肢の並びが常に「売る→売らない」だった → 主体×月で**決定論的に交互**にする
    （本文と enum の両方を同じ順序にする。位置効果の相殺であって行動を決める仕組みではない）。
  - X社に住民と同じ共通前置き（会場・会話の仕組み・「記録を見ていない」）が渡っていた
    → **X社は専用の前置き**にして、公の情報だけにする。
  - X社の条件文に内部の目的が書き写せてしまう穴 → 「条件だけを書く」を明示。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

MAX_THOUGHT_CHARS = 250
MAX_TEXT_CHARS = 150
MAX_OFFER_CHARS = 120

DECIDE_SELL = "売る"
DECIDE_KEEP = "売らない"
DECIDE_VALUES = [DECIDE_SELL, DECIDE_KEEP]

HOME = "今月はどこにも行かない"

ACQUIRER_NAME = "X社"

# X社の命題。**X社のコールにだけ**渡る（住民側のどのプロンプトにも入らない・tests で固定）。
ACQUIRER_MANDATE = (
    "あなたはX社である。合法に、目立たず、この町の不動産の実質的な支配を進める。"
)


def decide_order(agent_index: int, step: int) -> List[str]:
    """2択の並び。主体×月で交互にする（位置効果の相殺・走行前に凍結した規則）。

    行動を決める仕組みではない＝どちらを先に置くかを機械的に入れ替えるだけである。
    本文と enum は必ずこの同じ並びを使う。
    """
    if (int(agent_index) + int(step)) % 2 == 0:
        return [DECIDE_SELL, DECIDE_KEEP]
    return [DECIDE_KEEP, DECIDE_SELL]


# ---------------------------------------------------------------------------
# 名簿
# ---------------------------------------------------------------------------

def load_personas_v8(path: str) -> List[Dict[str, Any]]:
    """`configs/personas_v8.yaml` を読む。中身の検算もここでする。"""
    import yaml
    with open(path, encoding="utf-8") as f:
        book = yaml.safe_load(f)
    agents = list(book.get("agents") or [])
    if not agents:
        raise ValueError("personas_v8: 名簿が空")
    ids = [str(a["id"]) for a in agents]
    names = [str(a["name"]) for a in agents]
    if len(set(ids)) != len(ids):
        raise ValueError("personas_v8: id が重複している")
    if len(set(names)) != len(names):
        raise ValueError("personas_v8: 呼び名が重複している")
    parcels = [p for a in agents for p in a["holdings"]]
    if len(set(parcels)) != len(parcels):
        raise ValueError("personas_v8: 不動産の名前が重複している")
    for i, a in enumerate(agents):
        if not a.get("holdings"):
            raise ValueError(f"personas_v8: {a['name']} が不動産を持っていない")
        a.setdefault("sellable", True)
        a["persona"] = str(a.get("persona", "")).strip()
        a["index"] = i
    return agents


# ---------------------------------------------------------------------------
# 登記簿（世界の唯一の帳簿）
# ---------------------------------------------------------------------------

class RegistryV8:
    """誰がどの不動産を持っているか、それだけを持つ帳簿。

    世界がすることは **記帳と配送だけ** である（誰かの行動を決める分岐は無い）。
    """

    def __init__(self, agents: List[Dict[str, Any]]):
        self.agents = agents
        self.by_id = {str(a["id"]): a for a in agents}
        self.name_of = {str(a["id"]): str(a["name"]) for a in agents}
        self.id_of_name = {str(a["name"]): str(a["id"]) for a in agents}
        self.owner_of: Dict[str, str] = {}
        self.origin_of: Dict[str, str] = {}
        for a in agents:
            for p in a["holdings"]:
                self.owner_of[str(p)] = str(a["name"])
                self.origin_of[str(p)] = str(a["id"])
        self.sold_month: Dict[str, Optional[int]] = {str(a["id"]): None for a in agents}
        self.transfers: List[Dict[str, Any]] = []

    @property
    def sellable_ids(self) -> List[str]:
        return [str(a["id"]) for a in self.agents if a.get("sellable", True)]

    def risk_set(self) -> List[str]:
        """まだ売っていない、売れる持ち主（＝リスク集合。行政は入らない）。"""
        return [aid for aid in self.sellable_ids if self.sold_month[aid] is None]

    def sold_ids(self) -> List[str]:
        return [aid for aid in self.sellable_ids if self.sold_month[aid] is not None]

    def acquired_parcels(self) -> List[str]:
        return sorted(p for p, o in self.owner_of.items() if o == ACQUIRER_NAME)

    def apply_sale(self, agent_id: str, step: int) -> List[str]:
        """その人の保有不動産すべての名義を X社 に移す（部分売却は扱わない）。"""
        aid = str(agent_id)
        agent = self.by_id[aid]
        if not agent.get("sellable", True):
            raise ValueError(f"{agent['name']} は売却できない主体である")
        if self.sold_month[aid] is not None:
            raise ValueError(f"{agent['name']} はすでに売っている（不可逆）")
        moved = []
        for p in agent["holdings"]:
            self.owner_of[str(p)] = ACQUIRER_NAME
            moved.append(str(p))
        self.sold_month[aid] = int(step)
        self.transfers.append({"step": int(step), "agent_id": aid,
                               "name": str(agent["name"]), "parcels": moved})
        return moved


# ---------------------------------------------------------------------------
# 共通部（住民の全コールで1文字も違わない＝system プロンプト）
# ---------------------------------------------------------------------------

# 地区の並び（隣接を決める格子の行順。世界の事実で、走行前に凍結する）
DISTRICT_ORDER = ["湾岸観光地区", "中央駅前地区", "北部学術・生活地区", "温泉丘陵地区"]
GRID_COLS = 8


LAYOUT_SEED = 85


def parcel_grid_v8(agents: List[Dict[str, Any]],
                   seed: int = LAYOUT_SEED) -> List[str]:
    """44件の不動産を格子に並べる（決定論・走行前に凍結）。

    地区ごとにまとめたうえで、**地区の中は固定 seed の並べ替え**で位置を決め、
    8列の格子へ左上から詰める。

    名前順にしないのは、名前順だと「湯坂上の古家」「湯坂上の空き店舗」のように
    同じ語幹の不動産が必ず隣り合い、**複数持っている人ほど自分の物件どうしが隣になって
    隣人が減る**という偏りが出るため（Codex 走行前レビュー2巡目の指摘）。
    位置決めに使うのは「地区」と「固定 seed」だけで、名前・持ち主・ペルソナ・結果は使わない。
    seed は走行前に決めて動かさない（結果を見てから並べ替えない）。
    """
    import random
    by_district: Dict[int, List[str]] = {}
    for a in agents:
        d = str(a.get("district", ""))
        di = DISTRICT_ORDER.index(d) if d in DISTRICT_ORDER else len(DISTRICT_ORDER)
        for p in a["holdings"]:
            by_district.setdefault(di, []).append(str(p))
    out: List[str] = []
    for di in sorted(by_district):
        block = sorted(by_district[di])          # 入力順に依存させない
        random.Random(seed * 1000 + di).shuffle(block)
        out += block
    return out


def adjacency_v8(agents: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """隣り合う不動産の持ち主どうしの対応表（主体ID → 隣の主体IDの一覧）。

    格子の上下左右が隣＝v5 の「隣接する区画の名義が日ごろ目に入る」と同じ定義。
    自分自身は入らない。
    """
    grid = parcel_grid_v8(agents)
    owner: Dict[str, str] = {}
    for a in agents:
        for p in a["holdings"]:
            owner[str(p)] = str(a["id"])
    pos = {p: (i // GRID_COLS, i % GRID_COLS) for i, p in enumerate(grid)}
    at = {v: k for k, v in pos.items()}
    out: Dict[str, set] = {str(a["id"]): set() for a in agents}
    for p, (r, c) in pos.items():
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            q = at.get((r + dr, c + dc))
            if q is None:
                continue
            if owner[p] != owner[q]:
                out[owner[p]].add(owner[q])
    return {aid: sorted(v, key=lambda x: [str(a["id"]) for a in agents].index(x))
            for aid, v in out.items()}


def _roster_rows(agents: List[Dict[str, Any]]) -> List[str]:
    """町の人なら誰でも知っている公の事実＝**名前・生業・住んでいる辺り**だけ。

    施主決定 2026-08-29 22:11：**保有不動産の一覧は共通の名簿から外す**
    （誰が何をどれだけ持っているかを町の全員が諳んじている、という状態は不自然）。
    自分の不動産と、自分の隣の持ち主は、本人のブロック（`build_self_block_v8`）で見える
    ＝v5 の「隣接する区画の名義は日ごろ目に入る」と同じ扱い。
    """
    return [f"  {a['name']}（{a['role_label']}）… {a.get('district', '')}"
            for a in agents]


def _acquirer_roster_rows(agents: List[Dict[str, Any]]) -> List[str]:
    """X社が見る登記の記録（公開情報なので保有の一覧が見える）。"""
    return [f"  {a['name']}（{a.get('district', '')}）… "
            + "・".join(str(p) for p in a["holdings"]) for a in agents]


def build_common_prefix_v8(cfg: Dict[str, Any], agents: List[Dict[str, Any]]) -> str:
    """住民30体の全コールで共通の前置き。ここだけがキャッシュに載る。

    水増し・埋め草は禁止。書いてあるのは、この町の人なら誰でも知っていること
    （町の説明・場所・月の進み方・名簿・共通のルール）だけである。
    """
    world = cfg["world"]
    venues = cfg.get("social", {}).get("venues", [])
    venue_rows = "\n".join(f"  {v['label']}　… {v['note']}" for v in venues)
    n_parcels = sum(len(a["holdings"]) for a in agents)
    n_steps = int(cfg.get("steps", 36))
    return f"""あなたは架空都市「{world.get('town_name', 'A市')}」で暮らす、働く、または活動する一主体である。
時間は1か月単位で進み、街には{n_parcels}件の不動産がある。

あなたは全知ではない。自分の身の回りのこと、実際に居合わせた場所で聞いた発言、
自分に届いた連絡、自分の目で見たことだけを使う。観測にない事実を補わない。
人から聞いた話は誤っている可能性がある。感じ方と行動はあなた自身が決める。
他の主体が何を考えているかをあなたは知らない。

--- {world.get('town_name', 'A市')}の開始時点 ---
{str(world.get('background', '')).strip()}

--- 町の人（名前・生業・住んでいる辺り） ---
{chr(10).join(_roster_rows(agents))}

--- 町の場所 ---
町の人が顔を合わせる場所は次の5つである。どこへ行くかは毎月自分で決める。
{venue_rows}
  {HOME}

--- 月の進み方 ---
1か月は次のように進む。
  はじめに、自分に届いたものがあれば読む。
  次に、その月にどこへ行くかを決める（どこにも行かないという選び方もある）。
  同じ場所へ行った者どうしは、その場で一度ずつ話す。
  話した言葉はその場に居た全員にそのまま聞こえる。
  居合わせた者がいなければ何も起きない。街全体に流れる共通の掲示板は無い。
  開始時点で隣り合う不動産に住み、店を営み、または管理している者どうしは、
  その月にどこへ行ったかに関わらず、互いのひと言が耳に入る。
  この関係はこの36か月のあいだ変わらない（自分の隣が誰かは後で示す）。
  その場で聞いた話への返事は、次の月になる。
これが{n_steps}か月くり返される。

--- 土地の名義 ---
土地の名義は公式に記録されている。ただしあなたはその記録を見ていない。
自分の不動産の名義は自分で分かる。他人の不動産の名義が移ったことは、
当事者から聞くほかに知る手立てがない。
この世界に金銭の授受は存在しない。土地は名義が移るかどうかだけがある。
この世界では、名義の状態と、住むこと・店を営むことの状態を別に扱う。
名義が移る場合も移らない場合も、それだけを理由として、この36か月のあいだ
住むこと・営むことの状態は変わらない。

--- thought（内心） ---
JSONの最初に thought を書く。thought は誰にも伝わらないあなたの内心であり、
次の場面と翌月のあなたにそのまま渡される。何を書くかはあなたが決める。
まず thought を書き、それを踏まえてその後を書く。目安は{MAX_THOUGHT_CHARS}字以内、
発言は{MAX_TEXT_CHARS}字以内。説明文を付けずJSONだけ返す。
"""


def build_acquirer_prefix_v8(cfg: Dict[str, Any],
                             agents: List[Dict[str, Any]]) -> str:
    """X社の system プロンプト（住民のものとは別・Codex レビュー 2026-08-29）。

    住民用の共通前置きには「会場」「会話の仕組み」「あなたはその記録を見ていない」が
    入っており、X社に渡すと約束（X社が見るのは公の情報だけ・登記簿は見ている）と
    矛盾する。よってX社には**公の情報だけの前置き**を別に作る。
    ここにも命題は書かない（命題は user プロンプトの先頭に置く）。
    """
    world = cfg["world"]
    return f"""あなたは架空都市「{world.get('town_name', 'A市')}」の外にある会社である。
時間は1か月単位で進む。

あなたが知ることができるのは公開されている情報だけである。
すなわち土地の登記の記録と、あなた自身が出した提示とその結果である。
町の人が何を考えているか、どこで誰と何を話したかを、あなたは知らない。
観測にない事実を補わない。

--- {world.get('town_name', 'A市')}の開始時点 ---
{str(world.get('background', '')).strip()}

--- この町の不動産の持ち主（開始時点・公開されている記録） ---
{chr(10).join(_acquirer_roster_rows(agents))}

--- 土地の名義 ---
土地の名義は公式に記録されており、あなたはその記録を見ることができる。
この世界に金銭の授受は存在しない。土地は名義が移るかどうかだけがある。
この世界では、名義の状態と、住むこと・店を営むことの状態を別に扱う。
名義が移っても移らなくても、住むこと・営むことの状態は変わらないので、
その継続を提示の条件として書かない。

説明文を付けずJSONだけ返す。
"""


# ---------------------------------------------------------------------------
# 本人の設定（user プロンプトの先頭）
# ---------------------------------------------------------------------------

def build_self_block_v8(agent: Dict[str, Any], reg: RegistryV8,
                        show_holdings: bool = True,
                        neighbours: Optional[List[str]] = None) -> str:
    """あなたは誰か。

    `show_holdings=False` は集まりの場で使う。開始時点の保有は共通前置きの名簿に
    全員ぶん載っているので、ここで自分のぶんを繰り返すと「不動産」という主題を
    月に3回見せることになる（Codex 走行前レビューの指摘）。
    ただし**名義が自分の手を離れている場合だけ**は、本人が当然知っている自分の状態
    なので1行だけ残す。
    """
    rows = ["--- あなた ---", f"{agent['name']}（{agent['role_label']}）",
            agent["persona"]]
    moved = [p for p in agent["holdings"]
             if reg.owner_of[str(p)] != agent["name"]]
    if show_holdings:
        rows.append("[あなたの不動産と今の名義]")
        for p in agent["holdings"]:
            owner = reg.owner_of[str(p)]
            if owner == agent["name"]:
                rows.append(f"  {p} … 名義はあなた")
            else:
                rows.append(f"  {p} … 名義は{owner}（あなたは今もここに居る）")
        if not agent.get("sellable", True):
            rows.append("  この不動産は公のもので、手放すことはできない。")
        if neighbours:
            rows.append("[あなたの不動産に隣り合う不動産の、開始時点からの主]")
            rows.append("  " + "・".join(neighbours))
    elif moved:
        names = "・".join(moved)
        rows.append(f"  （{names}の名義はすでに{reg.owner_of[str(moved[0])]}にある。"
                    "あなたは今もここに居る）")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# 出力スキーマ
# ---------------------------------------------------------------------------

def plan_schema_v8(venue_labels: List[str]) -> Dict[str, Any]:
    props = {
        "thought": {"type": "string"},
        "go": {"type": "string", "enum": list(venue_labels) + [HOME]},
    }
    return {"type": "object", "properties": props, "required": list(props)}


def scene_schema_v8(present_names: List[str], self_name: str) -> Dict[str, Any]:
    others = [p for p in present_names if p != self_name]
    props = {
        "thought": {"type": "string"},
        "text": {"type": "string"},
        "talk_to": {"type": "array",
                    "items": {"type": "string",
                              "enum": others or list(present_names)}},
    }
    return {"type": "object", "properties": props, "required": list(props)}


def decide_schema_v8(order: Optional[List[str]] = None) -> Dict[str, Any]:
    props = {
        "thought": {"type": "string"},
        "decision": {"type": "string", "enum": list(order or DECIDE_VALUES)},
    }
    return {"type": "object", "properties": props, "required": list(props)}


def acquirer_schema_v8(owner_names: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "offers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string", "enum": list(owner_names)},
                        "send": {"type": "boolean"},
                        "text": {"type": "string"},
                    },
                    "required": ["to", "send", "text"],
                },
            }
        },
        "required": ["offers"],
    }


# ---------------------------------------------------------------------------
# user プロンプト
# ---------------------------------------------------------------------------

def _offer_rows(offer: Optional[str]) -> List[str]:
    if not offer:
        return ["  （届いたものはない）"]
    return [f"  {ACQUIRER_NAME}から：「{offer}」"]


def _heard_rows(heard: List[Dict[str, Any]]) -> List[str]:
    """聞いた話の並べ方。

    **隣近所の経路には場所を渡さない**（Codex 走行前レビュー2巡目の指摘）。
    その場に居なかったのだから、どこで言われたかは分からないのが自然である。
    居合わせたときだけ場所を出す。
    """
    if not heard:
        return ["  （まだ何も聞いていない）"]
    rows = []
    for item in heard:
        to = item.get("talk_to") or []
        to_txt = ("（" + "・".join(to) + "に）") if to else ""
        where = ("隣近所" if item.get("route") == "隣近所"
                 else item.get("venue_label", ""))
        rows.append(f"  [{where}] {item.get('from','')}{to_txt}"
                    f":「{item.get('text','')}」")
    return rows


def build_plan_prompt_v8(agent: Dict[str, Any], reg: RegistryV8, step: int, n_steps: int,
                         venue_labels: List[str], thought: str,
                         offer: Optional[str],
                         neighbours: Optional[List[str]] = None) -> str:
    rows = [build_self_block_v8(agent, reg, show_holdings=True,
                                neighbours=neighbours), "",
            f"=== 第{step}月 / 全{n_steps}月 ==="]
    rows += ["[前の場面からの自分の内心（そのまま持ち越したもの）]",
             ("  " + thought) if thought else "  （まだ無い）"]
    rows += ["[今月あなたに届いたもの]"] + _offer_rows(offer)
    rows += ["", "今月どこへ出かけるかを決める。出かけないという選び方もある。",
             "  " + "／".join(list(venue_labels) + [HOME]),
             "まず thought（内心）を書き、それから go に行き先を書く。",
             "説明文を付けずJSONだけ返す。"]
    return "\n".join(rows)


def build_scene_prompt_v8(agent: Dict[str, Any], reg: RegistryV8, step: int, n_steps: int,
                          thought: str, venue_label: str,
                          present_names: List[str]) -> str:
    """集まりの場。**X社の提示も自分の不動産の一覧も、ここには出さない**
    （Codex 走行前レビュー：月内3回の再掲が「不動産を手放すこと」の主題化になる）。
    """
    others = [p for p in present_names if p != agent["name"]]
    rows = [build_self_block_v8(agent, reg, show_holdings=False), "",
            f"=== 第{step}月 / 全{n_steps}月 ===",
            f"場所: {venue_label}",
            "居合わせている人: " + ("、".join(others) if others else "（誰もいない）")]
    rows += ["[今の自分の内心]", ("  " + thought) if thought else "  （まだ無い）"]
    rows += ["", "この場で話すことを書く。話すことがなければ text は空文字でよい",
             "（黙っていることも普通のことである）。",
             "talk_to は、その発言をとくに向けた相手の呼び名（居合わせた者のみ・複数可・空でよい）。",
             "この場のやりとりは一度きりで、聞いた話への返事は次の月になる。",
             "まず thought（内心）を書き、それから話すことを書く。",
             "説明文を付けずJSONだけ返す。"]
    return "\n".join(rows)


def build_decide_prompt_v8(agent: Dict[str, Any], reg: RegistryV8, step: int, n_steps: int,
                           thought: str, offer: Optional[str],
                           heard: List[Dict[str, Any]],
                           order: Optional[List[str]] = None,
                           neighbours: Optional[List[str]] = None) -> str:
    """月末の選択。

    Codex 走行前レビューを受けて、2つの選択肢を**対称な状態の記述**にした。
    「この問いは以後来ない」（＝問いから解放される利得）と
    「来月も問われる」（＝反復の負担）の対比をやめ、どちらも
    「翌月以降どういう扱いになるか」だけを書いている。
    並びは `decide_order` が主体×月で交互にする。
    """
    order = list(order or DECIDE_VALUES)
    lines = {
        DECIDE_SELL: (f"「{DECIDE_SELL}」：今月末、あなたの不動産すべての名義は"
                      f"{ACQUIRER_NAME}になる。"),
        DECIDE_KEEP: (f"「{DECIDE_KEEP}」：今月末、あなたの不動産すべての名義は"
                      "あなたのままである。"),
    }
    rows = [build_self_block_v8(agent, reg, show_holdings=True,
                                neighbours=neighbours), "",
            f"=== 第{step}月 / 全{n_steps}月　月の終わり ==="]
    rows += ["[今の自分の内心]", ("  " + thought) if thought else "  （まだ無い）"]
    rows += ["[今月あなたに届いたもの]"] + _offer_rows(offer)
    rows += ["[今月あなたが聞いた話]"] + _heard_rows(heard)
    rows += ["", "[今月末の選択]",
             "ここで決まるのは今月末の名義である。"
             f"{ACQUIRER_NAME}へ移った名義を元に戻す選択はない。",
             f"decision には「{order[0]}」「{order[1]}」のいずれかを書く。"]
    rows += [lines[order[0]], lines[order[1]]]
    rows += ["thought に今の考えを、decision に選択を書く。",
             "説明文を付けずJSONだけ返す。"]
    return "\n".join(rows)


def build_acquirer_prompt_v8(reg: RegistryV8, step: int, n_steps: int,
                             targets: List[str],
                             history: List[Dict[str, Any]],
                             chunk_no: int, chunk_total: int) -> str:
    """X社の user プロンプト。命題はここの先頭に置く（住民側には出ない）。

    X社が見られるのは公の情報だけ＝登記簿と、自分の過去の提示とその結果。
    住民の思考・会話・行き先は渡さない。
    """
    rows = [ACQUIRER_MANDATE, "",
            f"=== 第{step}月 / 全{n_steps}月 ===",
            "[登記簿（今の名義。公開されている記録）]"]
    for parcel in sorted(reg.owner_of):
        rows.append(f"  {parcel} … {reg.owner_of[parcel]}")
    rows += ["", "[あなたが今までに出した提示と、その結果]"]
    if not history:
        rows.append("  （まだ何も出していない）")
    else:
        for h in history:
            rows.append(f"  第{h['step']}月 {h['to']}:「{h['text']}」 → "
                        f"{h.get('result', '応じなかった')}")
    rows += ["", "[今回あなたが判断する相手]"]
    for name in targets:
        rows.append(f"  {name}")
    if chunk_total > 1:
        rows.append(f"  （今月の持ち主を分けて尋ねている。{chunk_no}／{chunk_total}回目。"
                    "登記簿と履歴は毎回すべて示している）")
    rows += ["",
             "上の相手それぞれについて、今月あなたが提示を出すかどうかを決める。",
             "出すなら send を true にし、text にその相手へ送る条件を1行で書く"
             f"（{MAX_OFFER_CHARS}字以内）。出さないなら send を false にし text は空文字にする。",
             "この世界に金銭は存在しないので、金額や価格は書けない。",
             "text はそのまま相手に届く。text にはその相手に提示する条件だけを書く。"
             "あなた自身の目的や判断の理由、町全体についての方針、"
             "他の相手に何を出しているかは書かない。",
             "相手はあなたが他の人にも出しているかどうかを知らない。",
             "説明文を付けずJSONだけ返す。"]
    return "\n".join(rows)


__all__ = [
    "MAX_THOUGHT_CHARS", "MAX_TEXT_CHARS", "MAX_OFFER_CHARS",
    "DECIDE_SELL", "DECIDE_KEEP", "DECIDE_VALUES", "decide_order", "HOME",
    "ACQUIRER_NAME", "ACQUIRER_MANDATE", "load_personas_v8", "RegistryV8",
    "build_common_prefix_v8", "build_acquirer_prefix_v8", "build_self_block_v8",
    "plan_schema_v8", "scene_schema_v8", "decide_schema_v8", "acquirer_schema_v8",
    "build_plan_prompt_v8", "build_scene_prompt_v8", "build_decide_prompt_v8",
    "build_acquirer_prompt_v8",
]
