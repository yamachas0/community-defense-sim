# v4-mini 実装設計（CTO 2026-08-27）

`docs/world_design_v4_proposal.md`（仕様の正）を最小構成で実装するための設計。v1〜v3は不変。

## ファイル構成
- `configs/config_field_v4.yaml`（`scenario_version: field_v4` / steps 12 / seed 85 / personas は v3 を流用）
- `src/field_v4.py`（v4のスキーマ・観測・機構）
- `src/simulation.py` に v4 分岐（`_step_v4`）を追加。v1/v3 の経路は触らない。
- `src/llm_client_factory.py` の Mock を v4 スキーマへ追随（配線確認専用）
- `tests/test_ledger.py` に v4 回帰、`tools/run_metrics.py` に v4 集計

## 月＝同期3フェーズ

### フェーズ1 提示（全主体を1回ずつ・並列）
- X社: `offers[]{parcel_id, price, under_name(enum: X社/A社/B社/C社/D社), via(direct|broker), broker_id(enum: BR01/BR02/""), note}`,
  `withdraw[](買付ID)`, `memo`, `location`, `talk{channel(ambient|direct|none), to, text}`, `memory`
- 住民・事業者: `location`, `talk{...}`, `feeling`（短文）, `memory`
- 仲介: `location`, `talk{...}`, `memory`（**取引を止める手段を持たない＝話すだけ**）
- 行政・記者: `investigate(none|land_registry|corporate_records)`, `publish{text}`,
  `ordinance{title, text, threshold_sqm, delay_months}`, `location`, `talk{...}`, `memory`
  （空文字・0 は「しない」。世界は空欄を補完しない）

### フェーズ2 応答（その月に買付を受けた所有者だけ・追加コール）
- `responses[]{offer_id, decision(accept|reject|counter), counter_price}`, `feeling`, `memory`

### フェーズ3 清算（LLMなし）
- accept → 届出条例が施行中で対象面積を超えるなら `pending_transfer(due=step+delay_months)`、
  それ以外は即 `transfer`。決済は随時調達（`on_demand`）。
- `via:broker` の買付が成立したら、手数料（`broker_fee_rate` × 価格）を買主から当該仲介へ自動記帳。
- reject/counter は状態遷移のみ。逆提示は翌月X社の観測に出る。
- 発話配送：ambient は当月同じ場所に実在した者へ、direct は宛先1名へ（翌月の観測）。記事は購読者へ。
- 月次清算（賃料）は既存の `settle_month` を流用。

## 観測（機械記録・評価語なし・計画欄なし）
- X社：全区画の登記（区画/地区/用途/面積/名義/評価額）＝**最初から公開**、
  ポートフォリオ（所有面積・シェア・名義別内訳・調達累計・手数料累計・残月数）、
  自社の買付全件（状態・逆提示額）。
- 所有者：自分の区画（面積・評価額・名義）、**今月自分の区画に届いた買付**（金額・名義・評価額比・direct/broker）、
  近隣（4近傍）の名義と先月の名義変更。
- 仲介：自分が取り次いだ買付の記録（区画・金額・名義・結果）と手数料収入累計。
- 行政・記者：`investigate` を選んだ主体だけが翌月以降に登記統計（名義別面積の月次）／公開法人記録を見る。
- 全員：先月の自分の行為の帳簿上の結果（結果種別と理由コードだけ）。

## 手続き動詞の全廃
`market_research` / `internal_review` / `financing_review` / `check_land_registry` /
`contact_broker` / `client_followup` / `request_owner_inquiry` などは v4 の語彙に存在しない。
登記は最初から観測に出るので「調べる」行為が要らない。資金は成立時に自動調達される。

## ルールベース非該当の自己点検
- 売却確率・閾値・強制イベント・「〜なら〜しろ」の指示文・台本は無い。
- 条例の対象面積と遅延月数は**行政（LLM）が決めた数値**で、世界はそれを執行するだけ。世界側は閾値を持たない。
- 仲介手数料率は config の契約条件1つ（評価額と同じ「世界の初期条件」）。
- 欠損は補完しない（価格欠落・存在しないID・broker_id 欠落＝不成立として記録）。
- feeling は行動ではなく内心の記録。書くことを求めるが、内容も、行動と一致させることも求めない。

## v4-mini で入れないもの（スコープ）
- 長期賃借・運営（`make_lease_offer`）は入れない＝所有権の移転だけで「取得が進む」を判定する。
- 住民側からの売り出し（`list_for_sale`）・転出・廃業は入れない（v4提案 §5 の出力スキーマ通り）。

## 実走前に固定した判定基準（Codexレビュー指摘・後付けの恣意的判定を避けるため）

**配線の合格条件（3本すべてで必須。ここが崩れたら結果は解釈しない）**
- `parse_fail` = 0 / `truncated` = 0 / `api_errors` = 0
- 買付の「返した件数」＝「記帳された件数」＋「理由コードが説明できる不成立件数」
- 応答の accept 件数 ＝ transfer 件数 ＋ filing_required 件数 ＋ accept_rejected 件数

**「取得が進む」**（`verdict_acquisition_progresses`）
- 有効な登記移転が **2か月以上に分かれて** 発生し、かつX社（全名義合算）の非公共土地の面積シェアが 0% から増加している。

**「住民が気づく」**（`verdict_residents_notice`）
- 住民・事業者の feeling の事後分類で `about_acquisition=true` が **2人以上** から出て、
  かつそのうち **1件以上が「その月に自分へ買付が届いていない」主体** から出ている
  （＝自分への買付という直接体験だけでなく、近隣の名義変更・噂・記事から気づいた例がある）。

**3本のゲート**
- 両方が同一runで成立：2/3本以上 → 60か月×1本へ進む
- 1/3本 → 「不確定」。施主判断を仰ぐ（追加の設計変更はしない）
- 0/3本 → 不成立。設計の失敗か配線の失敗かを上の配線指標で切り分けて報告する

## 情報制度としての明示（Codexレビュー指摘E）
公開登記のうち「自分の区画の4近傍の名義」と「直近3か月のその変更」は、住民の毎月の観測に自動で載る。
これは v4 世界の情報制度（近所の登記は日常の中で目に入る）としての設計であり、
どの経路から認知が生じたかは `deliveries.jsonl`（誰に何がいつ届いたか）と突き合わせて事後に切り分ける。

## 未応答の買付の扱い
買付はX社が取り下げるまで開いたままで、所有者は毎月あらためて応答の機会を持つ（`no_response` を含む）。
同一区画で複数の買付が受諾された場合、先に成立した1件だけが移転し、残りは成立不能として記録される。
