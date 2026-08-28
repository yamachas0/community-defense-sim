# API費の節約 第1便 — 実装と実測

作成 2026-08-28（CTO）。発注＝施主採用の3案
**①分類器を Gemini Batch API へ ②共通前置きのキャッシュ最大化 ③出力上限の最小化**。

守った前提（発注書の絶対ルール）：
**プロンプトの意味内容・分類ルール・台本・世界設計は1バイトも変えていない。**
変えたのは「呼び方（同期／Batch）」「並び順（毎回同じ指示文を先頭へ）」「出力上限」の3つだけで、
どれも **config のスイッチ**であり、**既定は従来どおり**（`configs/config_field_v5c.yaml` は無改造で
v5c を再現する）。節約設定は別ファイル `configs/config_field_v5c_econ.yaml`。

---

## 0. 結論（先に）

| 案 | 実測にもとづく判定 | 24か月1本あたりの効き |
|---|---|---|
| ① 分類器を Batch API へ | **採用**。判定は50/50で完全一致・トークンも同一・請求だけ半分 | **−15.3%**（実測スモークでは −14.3%） |
| ② キャッシュ最大化（明示） | **採用不可**。前置き 929 トークンに対し API の最低が **2,048** | ±0 |
| ② キャッシュ最大化（並び替え） | スイッチは作ったが**既定 off**。前置きは 929→972 トークンにしか伸びず、暗黙キャッシュの最低 1,024 に**届かない**。1か月スモークのキャッシュ読みも 9.4%→8.0% で改善せず | ±0（実測） |
| ③ 出力上限の最小化 | **削減額ゼロ**。上限 2,200 は一度も当たっていない（平均出力 294・打切り0） | ±0 |

**3案あわせて 24か月3本 $4.89 → $4.14（84.7%＝1/1.18）。**
`docs/cost_breakdown_v5c.md` の「A案＝1/1.4」は **キャッシュを35%まで効かせられる前提**だったが、
その前提が実測で崩れた（§3）。**1/1.4 も 1/3 も、この前置きの長さでは届かない。**

届かせるには、前置きを長くする（＝記憶の渡し方＝実験条件の変更）か、
コール数・巡数・同席者を削る（＝観測条件の変更）しかない。どちらも
「狙った動きを出す仕組みは入れない」の裏返しなので、**本便では触っていない**。

---

## 1. 何を実装したか

| ファイル | 変更 |
|---|---|
| `src/llm_batch.py`（新規） | Gemini Batch API のジョブ実行層。inlined requests で投げ、ジョブ名を `run_dir/batch_jobs/<tag>.json` に保存し、ポーリングして結果を**元の順序で**返す。取れなかった行は `None` を返す |
| `src/llm_client_factory.py` | `GeminiClient.generate_many()`（Batch か同期並列かを config で切替・**取れなかった行は同期で埋め直す**）、`close_caches()`、キャッシュ成否のカウンタ、`MockClient` / `OpenAIClient` にも同じ `generate_many` |
| `src/kpi.py` | 3つの分類器（`classify` / `classify_occupation` / `classify_stage_v5c`）を `generate_many` 経由に。**渡す system / user / schema / temperature / max_tokens / チャンク分割はすべて従来と同一**（テストで固定） |
| `src/field_v5.py` | `prompt_order`（`legacy` / `stable_first`）。`stable_first` は毎回同じ指示文を先頭へ移すだけで、**行の集合は legacy と完全に同一**（テストで固定） |
| `src/simulation.py` | `llm.thought_max_tokens`（主体コールの出力上限・既定は `llm.max_tokens`）、`llm.prompt_order`、Batch ジョブ台帳の置き場、`summary.json` の `saving` ブロック、走行後のキャッシュ片付け |
| `configs/config_field_v5c_econ.yaml`（新規） | 節約設定。`config_field_v5c.yaml` との差は **`llm:` と `run_name` だけ**（テストで固定） |
| `tools/cost_saving_v1.py`（新規） | ランの `summary.json` から費用を実測。**`|batch` が付いた tag は 0.5 倍**で合算する |
| `tools/verify_batch_classify.py`（新規） | 同じ行を同期と Batch の両方に流し、判定の一致率・往復時間・usage を出す |
| `tests/test_cost_saving.py`（新規・43件） | 上の「変えていない」をすべて実際に動かして固定する |

**config のスイッチ**（既定は全部 従来どおり）：

```yaml
llm:
  batch_classify: false     # 事後の分類器を Batch へ
  batch_agents: false       # 主体のコールを Batch へ
  prompt_order: legacy      # legacy | stable_first
  thought_max_tokens: <llm.max_tokens と同じ>
  enable_cache: false
```

---

## 2. mock の全テスト（実APIを叩く前に通した）

```
python tests/test_ledger.py        -> 619 passed, 0 failed
python tests/test_v5c.py           ->  73 passed, 0 failed
python tests/test_present.py       -> 344 passed, 0 failed
python tests/test_cost_saving.py   ->  43 passed, 0 failed   （本便で新設）
python run.py --config configs/config_field_v5c.yaml      --provider mock --quiet   -> 24か月完走
python run.py --config configs/config_field_v5c_econ.yaml --provider mock --quiet   -> 24か月完走
```

**既定が従来どおりであることの証拠（バイト一致）**：改修前のmockラン
`simulations/2026-08-28_1012_100_v5c_mock3` に保存されている `config.yaml` をそのまま
再実行し（`simulations/2026-08-28_2031_100_v5c_mock3`）、9本の出力を `cmp` で比較した。

```
SAME  events.jsonl  utterances_v5.jsonl  thoughts_all.jsonl  plans_v5.jsonl
SAME  traces_v5.jsonl  deals_v5.jsonl  kpi.jsonl  stage_labels_v5c.jsonl  occupation_labels.jsonl
```

9本すべて**1バイトも違わない**。プロンプト・観測・分類の出力は変わっていない。

---

## 3. ②キャッシュ最大化 — なぜ採用不可なのか（API の返答そのもの）

`countTokens` API で実測した前置きの長さ（`_scratch/system_prompt_tokens.json`）：

- **主体の system prompt（26体）: 844〜962 トークン・中央値 897・合計 23,269**
- 分類器の system prompt: `CLASSIFY_SYSTEM` **185** / `OCCUPATION_SYSTEM` **215** / `STAGE_SYSTEM` **352**

この前置きで明示キャッシュを作ろうとすると、API がこう返す：

```
400 INVALID_ARGUMENT
Cached content is too small. total_token_count=929, min_total_token_count=2048
```

**明示キャッシュの最低は 2,048 トークン。前置きは 929。半分にも届いていない。**
分類器が実ランで「キャッシュ 0%」だったのも同じ理由である（185〜352 トークン）。

並び替え（`stable_first`）で先頭に移せる「毎回同じ指示文」は **43 トークン**しかなく、
共通の前置きは **929 → 972 トークン**にしかならない。
Gemini 2.5 Flash-Lite の**暗黙キャッシュの最低は 1,024 トークン**なので、これでも届かない。

前置きを 2,048 トークンまで伸ばせばキャッシュは効くが、それは
**世界説明・ルール・人物設定を書き足す＝主体に渡す情報を変える**ことであり、
発注書の「プロンプトの意味内容は不変」に真正面から反する。**やらない。**

> 実ランに 18〜21% のキャッシュ読みが現れているのは、**同じシーンの1巡目と2巡目**のように
> user prompt の先頭まで一致する組み合わせで暗黙キャッシュが当たっているためで、
> system prompt 単独では最低トークン数に届いていない。並び替えでこの当たりが増えるかも
> 1か月スモークで測った＝**9.4%（legacy）→ 8.0%（stable_first）で改善しなかった**（§5.1b）。
> よって `prompt_order` の既定は `legacy` に戻してある（スイッチとテストは残す）。

---

## 4. ③出力上限 — 削減額はゼロ

実API 1か月スモーク（従来設定・`simulations/2026-08-28_2036_100_field_v5c_a_city`）の実測：

- calls **221**／output **65,011** トークン ＝ **1コールあたり平均 294**
- `max_token_finishes` **0**／`truncated_responses` **0**

**上限 2,200 には一度も当たっていない。** 課金は生成した分だけなので、
上限を 1,400 に下げても**請求は1円も減らず**、長い内心が切れる危険だけが増える。
スイッチ `llm.thought_max_tokens` は作ったが、**推奨値は据え置きの 2,200**。

（なお「内心の目安 250字」はプロンプト本文の文言であり、変えれば実験条件の変更になる。触っていない。）

---

## 5. ①分類器の Batch 化 — 実測

### 5.1 費用（1か月スモーク・従来設定の実測内訳）

```
=== 2026-08-28_2036_100_field_v5c_a_city  1か月  gemini-2.5-flash-lite
    calls 221  input 449,830（キャッシュ読み 9.4%）  output 65,011   $0.0672
    住民        calls=108 in=225,910 cached= 9.5%  $0.0299  (44.5%)
    分類器（事後） calls= 35 in= 71,127 cached= 0.0%  $0.0185  (27.6%)
    事業者      calls= 32 in= 63,387 cached= 9.7%  $0.0082  (12.3%)
    行政        calls= 18 in= 35,790 cached=15.5%  $0.0042  ( 6.2%)
    仲介        calls= 14 in= 25,995 cached=11.3%  $0.0032  ( 4.8%)
    記者        calls= 14 in= 27,621 cached=22.3%  $0.0031  ( 4.6%)
```

分類器は費用の **27.6%**（24か月の実ラン3本では 30.7〜31.3%）。ここが半額になる。

### 5.1b 従来 vs 節約 — 1か月スモークの実測（`tools/cost_saving_v1.py`）

```
=== 2026-08-28_2036_100_field_v5c_a_city（従来）
    設定: prompt_order=legacy cache=False batch=[] agent_max_tokens=2200
    calls 221（Batch 0）  input 449,830（キャッシュ読み 9.4%）  output 65,011   $0.0672
    打切り max_token_finishes=0 truncated=0 batch_fallback=0
    分類器  calls=35 in=71,127 cached=0.0%  $0.0185 (27.6%)

=== 2026-08-28_2044_100_field_v5c_econ（節約）
    設定: prompt_order=stable_first cache=False batch=['classify'] agent_max_tokens=2200
    calls 209（うちBatch 35）  input 434,783（キャッシュ読み 8.0%）  output 67,519   $0.0576
    打切り max_token_finishes=0 truncated=0 batch_fallback=0
    分類器  calls=35 in=77,085 cached=0.0%  $0.0098 (17.0%)
    Batch ジョブ 5本  往復 中央値 301s  最短 117s  最長 728s

=== 比較（1か月あたり）  従来 $0.0672 → 節約 $0.0576  ＝ 86%（1/1.17）
```

- **分類器 $0.0185 → $0.0098**（入力が 71k→77k と増えたうえで半額＝Batch が効いている）。
- **打切りは両方とも 0**（`max_token_finishes` / `truncated` / Batchの同期埋め戻し すべて0）。
- **キャッシュ読みは 9.4% → 8.0% で改善しなかった**＝`stable_first` は効かない（§3のとおり）。
  この差はラン間のばらつきの範囲。
- 2本は別々の世界（LLMのサンプリングが違う）なのでコール数は 221 と 209 で違う。
  比べているのは**1か月あたりの費用**であって、同じ世界の再現ではない。

### 5.2 24か月3本への効き（既存の実ラン3本の実測トークンに 0.5 を掛けたもの）

```
runA  $1.7239 -> $1.4633  (84.9%)   分類器分 $0.5213
runB  $1.6375 -> $1.3869  (84.7%)   分類器分 $0.5012
runC  $1.5293 -> $1.2903  (84.4%)   分類器分 $0.4780
3本合計  $4.8907 -> $4.1405  (84.7% ＝ 1/1.18)
```

### 5.3 判定の一致（Batch と同期で結果が変わらないか）

同じ50行を、同じ関数・同じプロンプト・同じスキーマ・同じ温度で、同期と Batch の
両方に流した（`python tools/verify_batch_classify.py 2026-08-28_2036_100_field_v5c_a_city --rows 50`、
結果は `_scratch/batch_agreement.json`）。

| 分類器 | 判定の一致 | 同期 | Batch |
|---|---|---|---|
| `classify`（frame / about_acquisition） | **50/50 = 100%** | 2.7s | 178.7s |
| `classify_occupation`（links_multiple / intent） | **50/50 = 100%** | 2.1s | 178.7s |
| `classify_stage_v5c`（deal / area / same_buyer / admin） | **50/50 = 100%** | 3.0s | 360.5s |

さらに、**消費トークンまで完全に一致した**：

```
usage_sync  : calls 6  input 6,127  output 5,294  errors 0
usage_batch : calls 6  input 6,127  output 5,294  errors 0   （同期フォールバック 0件）
```

同じ入力・同じ出力・同じトークン数で、**請求だけが半分**になる。
分類は事後の観測であって世界には戻らないので、創発は1ミリも歪まない。

### 5.4 往復時間 — `batch_agents` は不採用

Batch API は返却時刻を保証しない。実測した往復時間：

| ジョブ | 件数 | 往復 |
|---|---|---|
| 極小の確認（JSONを1個返すだけ） | 2 | **536.5s（8.9分）** |
| `classify` | 2 | 178.7s |
| `classify_occupation` | 2 | 178.7s |
| `classify_stage_v5c` | 2 | 360.5s |
| 節約設定の1か月スモーク（5本） | 各2 | 中央値 **301s**・最短 117s・最長 **728s** |

**件数ではなくキューの待ちで決まる**（2件でも9分かかった）。発注書の判定条件は
「1ラウンド3分以内なら採用」だったが、**中央値 301s・最長 728s で条件を満たさない**。

主体のコールは「前の巡の発話が次の巡の入力」なので、24か月×(4シーン×2巡＋計画)＝
**216回のバッチ投入が直列**になる。中央値 301s で **216 × 301s ≒ 18時間/本**、
最長側の 728s なら **44時間/本**。3本なら数日。
→ **`batch_agents` は既定 false（不採用）**。スイッチは残してあるので、
施主が「時間はかけていい」と判断した時だけ true にすればよい（§7）。

**再開できることは実APIで確認済み**：ジョブ投入後にプロセスを落とし、同じ tag で
起動し直したところ、`run_dir/batch_jobs/<tag>.json` のジョブ名から取り直して
新しいジョブを作らずに結果を回収した（`reused: true` / `returned: 2`）。

---

## 6. 再現手順

```powershell
# mock（課金なし）
python tests/test_cost_saving.py
python run.py --config configs/config_field_v5c_econ.yaml --provider mock --quiet

# 費用の実測（従来 と 節約 を並べる）
python tools/cost_saving_v1.py <従来のrun_dir> <節約のrun_dir>

# Batch と同期で判定が一致するか（実APIを叩く＝課金が走る）
python tools/verify_batch_classify.py <run_dir> --rows 50
```

---

## 7. 残リスク・要施主判断

### 実測でわかった、報告すべきこと

1. **`docs/cost_breakdown_v5c.md` の A案（1/1.4）・B案（1/3）は前提が崩れた。**
   どちらも「キャッシュを35〜55%まで効かせる」前提だったが、**前置き 929 トークンでは
   キャッシュに載らない**（明示の最低 2,048／暗黙の最低 1,024）。上の文書の§5の試算表は
   この点で楽観だった。**正しい数字は本便の実測（1/1.18）である。**
2. **思考トークン（thinking）の隠れコストは無い。** 実測で `thoughts_token_count` は
   返らず、`total_token_count = prompt + candidates` だった＝2.5 Flash-Lite は
   思考オフで、いまの usage 集計に取りこぼしは無い。

### 施主の新目標「24か月×3本を $1」について（実測トークンにもとづく試算）

`tools/cost_saving_v1.py` と同じ単価を、実ラン3本の実測トークンに掛けたもの。

```
                                                      3本合計   基準比
現状                                                   $4.8907   100.0%   1/1.00
[本便で採用] 分類器のみ Batch                             $4.1405    84.7%   1/1.18
+ 主体も Batch（1本あたり実時間 18〜44時間）                  $2.4453    50.0%   1/2.00
+ stage をルール先行にして候補行だけ LLM へ                   $2.3823    48.7%   1/2.05
+ classify_occupation を廃止                           $2.1458    43.9%   1/2.28
```

各案の「創発を歪めないか」の一行判定：

- **主体も Batch（`batch_agents: true`）** — **歪めない**（同じプロンプト・同じ順序を
  非同期で投げるだけ）。代償は**実時間だけ**（18〜44時間/本）。**$1 に向けた最大の一手。**
- **分類器のルール先行（候補行だけ LLM へ）** — **歪めない**（事後の観測。世界に戻らない）。
  しかも `_v5c_stage` は `rule AND llm` なので、**ルールに当たらない行は LLM が何を返しても
  色にならない＝結果は完全に同一**。ただし削減幅は小さい：実測のルール当たり率は
  **76.7% / 82.9% / 79.6%**（runA/B/C）で、落とせる行は**2割しかない**（−2.2%）。
- **`classify_occupation` を廃止** — **歪めない**（事後の観測）。代償は v5b と並べる
  比較指標（links_multiple / intent）を1つ失うこと。**要施主判断。**
- **明示キャッシュのために前置きを 2,048 トークンまで足す** — **歪める。** 主体に渡す
  情報そのものが変わる。**やらない。**
- **同席者を絞る／巡数を減らす／記憶を要約する** — **歪める。** 観測条件そのもの。**やらない。**
- **モデルを下げる** — 施主指示で据え置き。今回は触っていない。

**結論：歪めない策を全部足しても $2.15（1/2.28）で、$1 には届かない。**
$1 にするなら、残るのは**本数か月数を減らす**という正直な道しかない：

- 上の最良（全Batch＋ルール先行＋occupation廃止）で **3本→1本 ＝ $0.72**
- 同じく **3本→2本 ＝ $1.43**
- 24か月→12か月は、記憶が短くなる分だけ比例より安くなるはずだが、**実測していないので
  数字は書かない**（12か月の実ランを1本回せば確定する）。

### 要施主判断（3点）

1. **`batch_agents` を on にするか。** 費用は $4.14 → $2.45（1/2.0）。代償は1本 18〜44時間。
   創発は歪まない。夜間〜週末に回すなら現実的（ジョブIDは保存済みなので途中で落ちても再開できる）。
2. **`classify_occupation` を廃止するか**（−$0.24／3本・v5b比較指標を1つ失う）。
3. **$1 に本当に寄せるなら、本数を3→1（または2）にするか、月数を24→12にするか。**
   ここは費用ではなく**研究の設計**の判断なので、CTOでは決めない。

### 未検証・残リスク

- 24か月の本走は**まだ回していない**（発注外）。上の $4.14 / $2.45 は
  **実ラン3本の実測トークンに単価を掛けた試算**であり、実走の請求ではない。
- Batch の往復時間は Google 側のキューに依存する。18〜44時間/本は**今日の実測から引いた見込み**で、
  日によって変わる。`batch_agents` を on にするなら、まず1か月スモークで実時間を測るべき。
- `prompt_order: stable_first` は**効果が無いことを確認した**ので既定 legacy に戻した。
  スイッチとテストは残してある（将来 前置きが長くなったら意味を持つ）。

---

## 8. Codex レビュー（gpt-5.6-sol）と、それを受けて直したこと

レビュー結果の生文は `_scratch/codex_cost1.md`。総評は「条件付きで意味内容不変は守れている。
ただし Batch 台帳の内容照合不足を直すまで本番採用は不可」。指摘のうち **4件を直した**：

| 指摘 | 重大度 | 対応 |
|---|---|---|
| 台帳の再利用条件が「tag と件数」だけで、中身・model・schema・温度・上限が違っても古い結果を掴む | 高 | **直した。** `request_fingerprint()`（model＋system＋user＋schema＋温度＋上限＋thinking を順序込みで SHA-256）を台帳に保存し、**一致したときだけ**再利用する |
| Batch の空応答が同期フォールバックされない | 中 | **直した。** 空文字は「取れなかった行」として同期で埋め直す |
| Batch を使わない設定でも分類器が並列呼び出しになっていた（＝既定が従来どおりでない） | 中 | **直した。** `batch_kinds` に入っていない枠は**従来どおり1件ずつ順番に**呼ぶ。並列化は Batch の埋め直しのときだけ |
| Batch 経路の `latency_sec` にジョブ全体の時間が主体の数だけ複製されていた | 中 | **直した。** Batch では `latency_sec` を `null` にし、ジョブ全体の時間は `summary.saving.batch_jobs` にだけ残す |
| usage の tag が `...\|batch` になり従来と完全一致でない | 中 | **仕様として残す。** 半額の印がないと集計側が請求を再現できない。role / slot は保持されるので `tools/cost_saving_v1.py` は両方を読める |
| `stable_first` は行の集合は同じでも末尾強調が変わりうる | 中 | **既定 off**（`legacy`）。実測でも効果ゼロだったので使わない |
| テストが改修前の golden 文字列と比べていない | 低 | **強化した。** 場面プロンプトの16通りの条件すべてで「既定＝legacy」「stable_first は並べ替えのみ」を固定。加えて改修前の mock ランとの**バイト一致**（10本）を再確認 |

修正後の再検証：

```
python tests/test_cost_saving.py   ->  51 passed, 0 failed
python tests/test_ledger.py        -> 619 passed, 0 failed
python tests/test_v5c.py           ->  73 passed, 0 failed
python tests/test_present.py       -> 344 passed, 0 failed
改修前 mock ランとのバイト一致  -> events / utterances_v5 / thoughts_all / plans_v5 /
   traces_v5 / deals_v5 / kpi / stage_labels_v5c / occupation_labels / deliveries の10本すべて SAME
```
