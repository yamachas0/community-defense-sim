#!/usr/bin/env python
"""帳簿と記帳ディスパッチの回帰テスト。

    python tests/test_ledger.py

外部依存なし・API を叩かない。Codex レビューで指摘された不具合をそれぞれ1件ずつ
固定している（直したつもりで戻る事故を防ぐ）。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation import (_AMOUNT_REQUIRED, _has_amount, _is_rejected,  # noqa: E402
                            _parse_action, _repair_truncated_json)
from src.agents import Agent  # noqa: E402
from src.field_v3 import (action_schema_v3, build_acquirer_decision_prompt_v3,
                          normalize_acquirer_plan_v3, normalize_price_value,
                          build_system_prompt_v3, build_user_prompt_v3, control_share,
                          ensure_v3_state, list_for_lease, make_lease_offer,
                          acquirer_pipeline_text, answer_owner_inquiry,
                          client_inquiries_text,
                          check_land_registry_v3, inquire_owner_intent,
                          registry_view_rows,
                          own_result_row, own_results_text, report_owner_intent,
                          request_owner_inquiry,
                          resolve_lease_offer, seed_acquirer_intelligence_v3)  # noqa: E402

from src.field_v4 import (active_ordinance, build_phase1_prompt_v4,
                          build_phase2_prompt_v4, build_system_prompt_v4,
                          channel_text, enact_ordinance_v4, ensure_v4_state,
                          execute_pending_transfers_v4, filing_delay_for,
                          incoming_offers_rows, neighbourhood_rows,
                          own_offers_rows, owners_with_offers, phase1_schema_v4,
                          phase2_schema_v4, record_offer_v4, registry_rows,
                          registry_stats_rows, respond_to_offer_v4,
                          close_competing_offers_v4, own_results_text_v4,
                          broker_relay_rows)  # noqa: E402

from src.prompts import build_system_prompt, build_user_prompt  # noqa: E402
from src.schemas import action_schema  # noqa: E402
from src.world import Ledger, Parcel  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def mk() -> Ledger:
    parcels = [
        Parcel("P01", 0, 0, "北町", "residential", "HH01", 2400, registered_name="住民A"),
        Parcel("P02", 1, 0, "北町", "shop", "HH01", 3000, rent=18, tenant_id="BZ01",
               registered_name="住民A"),
        Parcel("P03", 2, 0, "北町", "residential", "HH02", 2400, registered_name="住民B"),
        Parcel("P04", 3, 0, "北町", "public", "MU01", 2400, registered_name="市"),
        Parcel("P05", 4, 0, "北町", "shop", "HH02", 3000, rent=0, tenant_id=None,
               registered_name="住民B"),
    ]
    return Ledger(parcels, {"HH01": 900, "HH02": 900, "BZ01": 400, "AQ01": 60000, "MU01": 0})


print("== 金額の成立条件 ==")
L = mk()
check("負の価格の買付は不成立",
      L.record_offer(1, "P01", "AQ01", -5000)["kind"] == "offer_rejected")
check("0円の買付は不成立",
      L.record_offer(1, "P01", "AQ01", 0)["kind"] == "offer_rejected")
check("予算超過の買付は不成立",
      L.record_offer(1, "P01", "AQ01", 999999)["kind"] == "offer_rejected")
L.enable_on_demand_financing(["AQ01"])
fo = L.record_offer(1, "P01", "AQ01", 999999)
check("随時調達主体は残高超過の買付を提示できる", fo["kind"] == "offer")
ft = L.record_accept(2, fo["offer_id"], "HH01")
check("受諾時に不足資金を調達して成約する", ft["kind"] == "transfer")
check("外部調達額が台帳に残る", L.financing_raised["AQ01"] == 939999)
check("負の賃料は不成立",
      L.record_rent_change(1, "P02", "HH01", -10)["kind"] == "rent_rejected")
check("賃料0は不成立（欠損の0が通らないこと）",
      L.record_rent_change(1, "P02", "HH01", 0)["kind"] == "rent_rejected")
check("住宅に賃料は設定できない",
      L.record_rent_change(1, "P01", "HH01", 20)["kind"] == "rent_rejected")
check("正常な賃料改定は成立",
      L.record_rent_change(1, "P02", "HH01", 40)["kind"] == "rent_change")
check("賃料が実際に反映される", L.parcels["P02"].rent == 40)
check("公有地への買付は不成立",
      L.record_offer(1, "P04", "AQ01", 3000)["kind"] == "offer_rejected")

print("== 売買の流れ ==")
L = mk()
o = L.record_offer(1, "P01", "AQ01", 3000, under_name="みどり緑地開発")
check("買付が成立", o["kind"] == "offer")
oid = o["offer_id"]
check("他人は受諾できない", L.record_accept(2, oid, "HH02")["kind"] == "accept_rejected")
t = L.record_accept(2, oid, "HH01")
check("所有者は受諾できる", t["kind"] == "transfer")
check("所有権が移る", L.parcels["P01"].owner_id == "AQ01")
check("登記名義は alias になる", L.parcels["P01"].registered_name == "みどり緑地開発")
check("買い手の資金が減る", L.cash["AQ01"] == 60000 - 3000)
check("売主に入金される", L.cash["HH01"] == 900 + 3000)
check("同じ買付は二度受諾できない",
      L.record_accept(3, oid, "AQ01")["kind"] == "accept_rejected")

print("== 二重売却の防止 ==")
L = mk()
L.cash["HH01"] = 5000          # 競合する買い手として資金を持たせる
a = L.record_offer(1, "P03", "AQ01", 3000)["offer_id"]
b = L.record_offer(1, "P03", "HH01", 3100)["offer_id"]
L.record_accept(2, a, "HH02")
check("同一区画の残った買付は無効化される", L.offers[b].status == "void")
check("無効化後は受諾できない", L.record_accept(2, b, "HH02")["kind"] == "accept_rejected")

print("== 逆提示（counter）==")
L = mk()
oid = L.record_offer(1, "P01", "AQ01", 2500)["offer_id"]
c = L.record_counter(2, oid, "HH01", 3800)
check("逆提示が成立", c["kind"] == "counter")
check("元の買付は countered になる", L.offers[oid].status == "countered")
check("逆提示は公開の売り出しに変換されない", L.parcels["P01"].listed_price is None)
check("逆提示額が保持される", L.offers[oid].counter_price == 3800)
check("買い手から逆提示が見える",
      [o.offer_id for o in L.counters_for("AQ01")] == [oid])
check("countered な買付はもう受諾できない",
      L.record_accept(3, oid, "HH01")["kind"] == "accept_rejected")

print("== 廃業・転出は一度きり ==")
L = mk()
check("廃業が成立", L.record_close(1, "BZ01")["kind"] == "business_closed")
check("入居関係が解消される", L.parcels["P02"].tenant_id is None)
check("二度目の廃業は不成立", L.record_close(2, "BZ01")["kind"] == "close_rejected")
check("廃業後は移転できない",
      L.record_relocate(3, "BZ01", "P05")["kind"] == "relocate_rejected")
check("転出が成立", L.record_move_out(1, "HH01")["kind"] == "move_out")
check("二度目の転出は不成立",
      L.record_move_out(2, "HH01")["kind"] == "move_out_rejected")

print("== 月次清算 ==")
L = mk()
margins = {"BZ01": 32}
L.settle_month(1, margins)
check("借主から賃料が引かれる", L.cash["BZ01"] == 400 + 32 - 18)
check("家主に賃料が入る", L.cash["HH01"] == 900 + 18)
before = L.cash["BZ01"]
L.record_close(2, "BZ01")
L.settle_month(2, margins)
check("廃業後は粗利も賃料も動かない", L.cash["BZ01"] == before)

L2 = mk()
L2.record_redevelop(1, "P02", "HH01", 60)   # 入居者を追い出す再開発
check("再開発で入居者が退去する", L2.parcels["P02"].tenant_id is None)
cash_before = L2.cash["BZ01"]
L2.settle_month(2, {"BZ01": 32})
check("店を失った月は営業粗利が立たない", L2.cash["BZ01"] == cash_before)

print("== 売り出し・取り下げ ==")
L = mk()
check("他人の区画は売りに出せない",
      L.record_listing(1, "P01", "HH02", 3000)["kind"] == "listing_rejected")
L.record_listing(1, "P01", "HH01", 3000)
check("売り出しが反映される", L.parcels["P01"].listed_price == 3000)
check("取り下げが成立", L.record_unlist(2, "P01", "HH01")["kind"] == "unlist")
check("取り下げで両フィールドが消える",
      L.parcels["P01"].listed_price is None and L.parcels["P01"].listed_at_step is None)
check("売り出していないものは取り下げられない",
      L.record_unlist(3, "P01", "HH01")["kind"] == "unlist_rejected")

print("== 応答の解析 ==")
check("正常な JSON", _parse_action('{"action_type":"hold"}') == {"action_type": "hold"})
check("配列は行動として成立しない（dict以外を弾く）", _parse_action("[1,2,3]") is None)
check("文字列は行動として成立しない", _parse_action('"hold"') is None)
check("数値は行動として成立しない", _parse_action("42") is None)
check("空応答", _parse_action("") is None)
trunc = '{\n "action_type": "set_rent",\n "target": "P02",\n "reasoning": "途中で切れ'
r = _parse_action(trunc)
check("切れた JSON からフィールドを回収する", r is not None and r["action_type"] == "set_rent")
check("回収したことが記録される", r is not None and r.get("_truncated") is True)
check("欠けた amount は 0 で代行されない（捏造しない）", not _has_amount(r or {}))
check("set_rent は amount 必須動詞", "set_rent" in _AMOUNT_REQUIRED)
esc = '{"action_type":"hold","utterance":"彼は\\"action_type\\": \\"move_out\\" と言った"}'
p = _parse_action(esc)
check("発話内に埋め込まれた擬似JSONに騙されない", p is not None and p["action_type"] == "hold")

print("== 不成立の判定 ==")
check("_rejected は不成立", _is_rejected({"kind": "offer_rejected"}))
check("invalid_action は不成立", _is_rejected({"kind": "invalid_action"}))
check("parse_fail は不成立", _is_rejected({"kind": "parse_fail"}))
check("成立した記帳は不成立でない", not _is_rejected({"kind": "transfer"}))

print("== 集計 ==")
L = mk()
check("公有地はシェアの母数から除く（取引可能4区画）",
      abs(L.ownership_share(["HH01"]) - 0.5) < 1e-9,
      f"got {L.ownership_share(['HH01'])}")
check("HHI は 0-1", 0 < L.hhi() <= 1)

print("== フィールドv2の構造化出力 ==")
schema = action_schema("household")
check("evidence は必須の配列",
      "evidence" in schema["required"]
      and schema["properties"]["evidence"]["type"] == "array"
      and schema["properties"]["evidence"]["items"]["type"] == "string")
acquirer = Agent("AQ01", "acquirer", "海外不動産投資会社", "長期保有型の投資会社",
                 extra={"mandate": "長期収益を目指す", "aliases": ["国内SPC"]})
system_prompt = build_system_prompt(
    acquirer,
    {"town_name": "A市", "background": "湾の反対側でスペースポート化が進む。"},
    60,
    48,
)
check("世界背景が全エージェントのsystem promptに入る",
      "湾の反対側でスペースポート化が進む。" in system_prompt)
check("買い手をAIに固定しない",
      "買い手AI" not in acquirer.role_ja and "購買主体（AI）" not in system_prompt)

print("== 観測IDの追跡可能性 ==")
L = mk()
offer = L.record_offer(1, "P01", "AQ01", 2500, under_name="国内SPC")
names = {"HH01": "住民A", "HH02": "住民B", "BZ01": "店", "AQ01": "取得主体",
         "MU01": "市"}
household = Agent("HH01", "household", "住民A", "所有者")
buyer_prompt = build_user_prompt(acquirer, L, 2, 60, names, [], ["AQ01"])
owner_prompt = build_user_prompt(household, L, 2, 60, names, [], ["AQ01"])
offer_obs_id = f"[OFFER-{offer['offer_id']}]"
check("同じ買付IDが売主と取得主体の双方に表示される",
      offer_obs_id in owner_prompt and offer_obs_id in buyer_prompt)
L.record_accept(2, offer["offer_id"], "HH01")
L.ordinances.append({"step": 2, "title": "届出条例", "body": "取引を届け出る"})
municipality = Agent("MU01", "municipality", "市", "自治体担当")
municipality_prompt = build_user_prompt(municipality, L, 3, 60, names, [], ["AQ01"])
check("自治体の取引観測にも成約IDが表示される",
      "[SALE-P01-M02]" in municipality_prompt)

print("== フィールドv3の世界機構 ==")
resident_v3 = Agent("HH01", "household", "R01", "日常生活を送る住民")
buyer_v3 = Agent("AQ01", "acquirer", "X社", "海外の不動産会社",
                 extra={"mandate": "A市で秘密裏に不動産を狙う",
                        "aliases": ["A社", "B社", "C社", "D社"]})
cfg_v3 = {
    "world": {"town_name": "A市", "background": "日常が続く。"},
    "social": {"venues": [{"id": "V01", "label": "飲食店"},
                           {"id": "V02", "label": "公民館"}]},
}
resident_schema = action_schema_v3(resident_v3)
buyer_schema = action_schema_v3(buyer_v3)
buyer_operation_schema = buyer_schema["properties"]["operations"]
check("住民には不動産以外の日常行動がある",
      all(v in resident_schema["properties"]["action_type"]["enum"]
          for v in ("routine", "work", "community_activity")))
check("発話は同席か直接連絡だけ",
      resident_schema["properties"]["utterance_channel"]["enum"]
      == ["ambient", "direct", "none"])
check("場所が構造化出力の必須項目", "location" in resident_schema["required"])
check("X社の名義はX社とA社からD社だけ",
      buyer_operation_schema["items"]["properties"]["under_name"]["enum"]
      == ["X社", "A社", "B社", "C社", "D社"])
resident_system = build_system_prompt_v3(resident_v3, cfg_v3, 48)
buyer_system = build_system_prompt_v3(buyer_v3, cfg_v3, 48)
check("秘密目的はX社だけに表示",
      "秘密裏に不動産を狙う" not in resident_system
      and "秘密裏に不動産を狙う" in buyer_system)
check("v3の利用者向けプロンプトにAI表記がない",
      "AI" not in resident_system and "AI" not in buyer_system)
check("X社は計画と世界内行動を同じスキーマで返す",
      all(key in buyer_schema["required"] for key in (
          "operations", "strategy", "next_milestone"))
      and buyer_operation_schema["maxItems"] == 6)
buyer_v3.extra["strategy_state"] = {"strategy": "面積効率を比較する"}
buyer_v3.extra["execution_history"] = [{
    "step": 1, "action_type": "market_research", "target": "", "amount": 0,
    "outcome_kind": "market_research", "effective_area": 0,
    "operations": "market_research:-=>market_research",
    "control_delta": 0, "cash": 60000,
}]
decision_prompt = build_acquirer_decision_prompt_v3(buyer_v3, "第2月の観測")
check("自己要約と別に計画・実行実績が残る",
      "面積効率を比較する" in decision_prompt
      and "market_research" in decision_prompt and "増減=0㎡" in decision_prompt)
check("実行と計画の分離したレビューを使わない",
      "一つの意思決定として同時に返す" in decision_prompt)
intel_ledger = mk()
seed_acquirer_intelligence_v3(
    buyer_v3, intel_ledger,
    {"acquirer_initial_intelligence": {
        "market_research": True, "land_registry_scope": "non_public"}})
check("X社は参入前に一定の公開市場・登記情報を把握",
      buyer_v3.extra.get("market_research_seen") is True
      and set(buyer_v3.extra.get("land_registry_targets", []))
      == {p.pid for p in intel_ledger.parcels.values() if p.use != "public"})

L = mk()
ensure_v3_state(L)
listed = list_for_lease(L, 1, "P01", "HH01", 22)
lease = make_lease_offer(L, 2, "P01", buyer_v3, 24, "C社", "長期利用希望")
accepted = resolve_lease_offer(L, 3, lease.get("lease_offer_id", ""), "HH01", True)
check("住民が自分の物件を賃貸募集できる", listed["kind"] == "lease_listing")
check("長期賃貸申込みと受諾が成立",
      lease["kind"] == "lease_offer" and accepted["kind"] == "lease_control")
check("所有者を変えずに運営・賃借主体を別記録",
      L.parcels["P01"].owner_id == "HH01"
      and getattr(L.parcels["P01"], "controller_id", None) == "AQ01")
check("所有率と別の支配率を集計", abs(control_share(L, ["AQ01"]) - 0.25) < 1e-9)
print("== 案1: 買付・賃借IDの表記統一 ==")
names_v3 = {"HH01": "R01", "HH02": "R02", "BZ01": "T01", "AQ01": "X社", "MU01": "G01"}
cfg_id = {"world": {"town_name": "A市", "background": "日常が続く。",
                    "block_names": ["北町"]},
          "social": {"venues": [{"id": "V01", "label": "飲食店"}],
                     "public_directory": ["HH01: R01"]}}
L = mk()
L.enable_on_demand_financing(["AQ01"])
oid_a = L.record_offer(1, "P01", "AQ01", 2500, under_name="X社")["offer_id"]
owner_v3 = Agent("HH01", "household", "R01", "所有者")
owner_view = build_user_prompt_v3(owner_v3, L, 2, 60, names_v3, cfg_id)
check("観測の買付IDは角括弧1トークンだけ",
      f"[{oid_a}] P01" in owner_view and "OFFER-" not in owner_view,
      owner_view)
check("角括弧付きの買付IDで受諾できる",
      L.record_accept(2, f"[{oid_a}]", "HH01")["kind"] == "transfer")
L = mk()
L.enable_on_demand_financing(["AQ01"])
oid_b = L.record_offer(1, "P01", "AQ01", 2500, under_name="X社")["offer_id"]
check("旧表記 OFFER- 付きでも同一の買付に解決する",
      L.record_accept(2, f"OFFER-{oid_b}", "HH01")["kind"] == "transfer")
L = mk()
L.enable_on_demand_financing(["AQ01"])
oid_c = L.record_offer(1, "P01", "AQ01", 2500, under_name="X社")["offer_id"]
check("素のIDでも受諾できる",
      L.record_accept(2, oid_c, "HH01")["kind"] == "transfer")
L = mk()
L.enable_on_demand_financing(["AQ01"])
oid_d = L.record_offer(1, "P01", "AQ01", 2500, under_name="X社")["offer_id"]
check("存在しない買付IDは今までどおり不成立",
      L.record_accept(2, "[O9999]", "HH01")["kind"] == "accept_rejected")
check("角括弧付きの買付IDで拒否・逆提示・取下げも解決する",
      L.record_counter(2, f"[{oid_d}]", "HH01", 3000)["kind"] == "counter")
L = mk()
L.enable_on_demand_financing(["AQ01"])
oid_e = L.record_offer(1, "P01", "AQ01", 2500, under_name="X社")["offer_id"]
check("角括弧付きで取下げできる",
      L.record_withdraw(2, f"[{oid_e}]", "AQ01")["kind"] == "withdraw")
L = mk()
buyer_id_test = Agent("AQ01", "acquirer", "X社", "外部会社",
                      extra={"mandate": "m", "aliases": ["A社"]})
ensure_v3_state(L)
lease_id = make_lease_offer(L, 1, "P01", buyer_id_test, 24, "A社", "希望")["lease_offer_id"]
owner_lease_view = build_user_prompt_v3(owner_v3, L, 2, 60, names_v3, cfg_id)
check("観測の賃借申込みIDも角括弧1トークンだけ",
      f"[{lease_id}] P01" in owner_lease_view and "LEASE-" not in owner_lease_view)
check("角括弧・旧表記どちらでも賃借申込みに解決する",
      resolve_lease_offer(L, 2, f"[LEASE-{lease_id}]", "HH01", True)["kind"] == "lease_control")

print("== 案6: 自分の行為の帰結が本人へ返る ==")
L = mk()
rejected = L.record_accept(2, "O9999", "HH01")
row = own_result_row(2, "accept_offer", "O9999", rejected)
check("不成立の理由コードが機械記録として残る",
      row["kind"] == "accept_rejected" and row["reason"] == "no_such_offer")
check("関係する識別子も残る", "offer_id=O9999" in row["refs"])
text = own_results_text([row])
check("不成立の記録に評価語も当為も混ざらない",
      "理由=no_such_offer" in text
      and not any(w in text for w in ("すべき", "しろ", "望ましい", "有効", "非効率")))
resident_r = Agent("HH01", "household", "R01", "所有者")
resident_r.extra["last_month_results"] = [row]
view_r = build_user_prompt_v3(resident_r, L, 3, 60, names_v3, cfg_id)
check("翌月の観測に先月の自分の行為と結果が載る",
      "[先月の自分の行為と、帳簿上の結果（機械記録）]" in view_r
      and "accept_rejected" in view_r)
view_empty = build_user_prompt_v3(Agent("HH02", "household", "R02", "所有者"),
                                  L, 1, 60, names_v3, cfg_id)
check("記録がない月は空である事実だけを返す", "（先月の記録なし）" in view_empty)
buyer_r = Agent("AQ01", "acquirer", "X社", "外部会社",
                extra={"mandate": "m", "aliases": ["A社"],
                       "last_month_results": [own_result_row(
                           2, "make_offer", "P01",
                           {"kind": "offer_rejected", "reason": "already_owner"})]})
view_b = build_user_prompt_v3(buyer_r, L, 3, 60, names_v3, cfg_id)
check("X社にも同じ機械記録が返る",
      "offer_rejected" in view_b and "理由=already_owner" in view_b)

print("== 案2: 仲介が区画・意向・希望額を事実として運ぶ ==")
L = mk()
L.enable_on_demand_financing(["AQ01"])
buyer_q = Agent("AQ01", "acquirer", "X社", "外部会社",
                extra={"mandate": "m", "aliases": ["A社"]})
broker_q = Agent("BR01", "broker", "B01", "地域の仲介")
owner_q = Agent("HH01", "household", "R01", "所有者")
names_q = {"HH01": "R01", "HH02": "R02", "BZ01": "T01", "BR01": "B01",
           "AQ01": "X社", "MU01": "G01"}
req = request_owner_inquiry(L, 1, "AQ01", "BR01", "P01", "P01の意向を確認したい")
qid = req.get("inquiry_id", "")
check("依頼が台帳に記録される", req["kind"] == "inquiry_request" and qid.startswith("Q"))
check("公共区画への依頼は不成立",
      request_owner_inquiry(L, 1, "AQ01", "BR01", "P04", "")["kind"] == "inquiry_rejected")
asked = inquire_owner_intent(L, 2, "BR01", "P01", qid, "いかがでしょう")
check("仲介の照会で依頼が進む",
      asked["kind"] == "inquiry_asked" and L.v3_inquiries[qid]["status"] == "asked")
check("宛先は区画で指定し、現所有者は台帳が解決する",
      asked["owner"] == "HH01")
check("存在しない区画への照会は不成立",
      inquire_owner_intent(L, 2, "BR01", "P99", "", "")["reason"] == "invalid_parcel")
check("他社の照会は進められない",
      inquire_owner_intent(L, 2, "BR02", "P01", qid, "")["reason"]
      == "not_your_inquiry")
owner_q.extra["inbox"] = []
owner_view_q = build_user_prompt_v3(owner_q, L, 3, 60, names_q, cfg_id)
check("照会は所有者の観測へ台帳の記録として届く",
      f"[{qid}] P01" in owner_view_q and "仲介:B01" in owner_view_q)
check("照会を受けていない主体は答えられない",
      answer_owner_inquiry(L, 3, qid, "HH02", "willing_to_sell", "3000", "")["reason"]
      == "not_asked_party")
check("数値でない希望額は事実として成立しない",
      answer_owner_inquiry(L, 3, qid, "HH01", "willing_to_sell", "応相談", "")["reason"]
      == "invalid_price_value")
ans = answer_owner_inquiry(L, 3, qid, "HH01", "willing_to_sell", "3200", "条件次第")
check("所有者の回答が区画・意向・希望額として台帳に載る",
      ans["kind"] == "inquiry_answer" and ans["asking_price"] == 3200
      and ans["owner_intent"] == "willing_to_sell")
rep = report_owner_intent(L, 4, "BR01", "AQ01", qid, "willing_to_sell", "3200", "報告")
check("仲介の報告が依頼主宛に台帳へ載る",
      rep["kind"] == "inquiry_report" and rep["to"] == "AQ01")
client_view = build_user_prompt_v3(buyer_q, L, 5, 60, names_q, cfg_id)
check("報告は依頼主の観測へ機械記録として届く",
      f"[{qid}]" in client_view and "willing_to_sell" in client_view
      and "希望額:3200万" in client_view)
broker_view = build_user_prompt_v3(broker_q, L, 5, 60, names_q, cfg_id)
check("仲介は自分が扱う照会の進行状況を観測できる",
      f"[{qid}]" in broker_view and "状態:reported" in broker_view)
L2 = mk()
q2 = inquire_owner_intent(L2, 1, "BR01", "P01", "", "")["inquiry_id"]
for value in ("unknown", "not_asked", "declined_to_answer"):
    fresh = inquire_owner_intent(L2, 1, "BR01", "P01", "", "")["inquiry_id"]
    check(f"未回答・拒否も事実として運べる（{value}）",
          answer_owner_inquiry(L2, 2, fresh, "HH01", value, value, "")["kind"]
          == "inquiry_answer")
check("依頼のない自発的な照会も成立する", L2.v3_inquiries[q2]["client"] == "")
check("存在しない照会IDへの回答は不成立",
      answer_owner_inquiry(L2, 2, "Q9999", "HH01", "unknown", "unknown", "")["reason"]
      == "no_such_inquiry")
check("他社の照会は報告できない",
      report_owner_intent(L2, 3, "BR02", "AQ01", q2, "unknown", "unknown", "")["reason"]
      == "not_your_inquiry")
broker_schema = action_schema_v3(broker_q)
check("仲介の出力に区画・照会・意向・希望額の欄がある",
      all(k in broker_schema["required"]
          for k in ("parcel_id", "inquiry_id", "owner_intent", "asking_price")))
check("意向の語彙に unknown / not_asked / 回答拒否が含まれる",
      all(v in broker_schema["properties"]["owner_intent"]["enum"]
          for v in ("unknown", "not_asked", "declined_to_answer")))

print("== 案3: 案件パイプラインと登記の再照会 ==")
L3 = mk()
L3.enable_on_demand_financing(["AQ01"])
buyer_p = Agent("AQ01", "acquirer", "X社", "外部会社",
                extra={"mandate": "m", "aliases": ["A社"]})
seed_acquirer_intelligence_v3(
    buyer_p, L3, {"acquirer_initial_intelligence": {
        "market_research": True, "land_registry_scope": "non_public"}})
check("参入前に取得した登記は確認月が『参入前調査』として残る",
      buyer_p.extra["land_registry_checked"]["P01"] == "参入前調査")
first = check_land_registry_v3(L3, 5, buyer_p, "P01")
check("再照会は no-op ではなく、前回との異同を返す",
      first["kind"] == "land_registry_check"
      and first["result"] == "参入前調査の照会と同一" and first["changed"] is False)
offer_p = L3.record_offer(6, "P01", "AQ01", 3000, under_name="A社")
L3.record_accept(7, offer_p["offer_id"], "HH01")
second = check_land_registry_v3(L3, 8, buyer_p, "P01")
check("名義が動いていれば差異が返る",
      second["changed"] is True and "から変化" in second["result"])
check("存在しない区画の照会は不成立",
      check_land_registry_v3(L3, 8, buyer_p, "P99")["reason"] == "no_such_parcel")
buyer_p.extra["parcel_last_action"] = {
    "P01": {"step": 6, "action": "make_offer", "outcome": "offer", "amount": 3000}}
pipeline = acquirer_pipeline_text(buyer_p, L3, names_q)
check("案件記録に自社の行動・相手の返答・提示額・状態・登記確認月が並ぶ",
      all(k in pipeline for k in ("自社最終行動:make_offer", "相手最終返答:transfer",
                                  "提示額:買付3000万", "状態:", "登記確認:第8月")))
check("案件記録に評価語・当為・自動判定が混ざらない",
      not any(w in pipeline for w in (
          "再照会", "不要", "古い", "すべき", "望ましい", "優先", "確率",
          "要再調査", "推奨")))
many = mk()
many.enable_on_demand_financing(["AQ01"])
big_buyer = Agent("AQ01", "acquirer", "X社", "外部会社",
                  extra={"mandate": "m", "aliases": ["A社"]})
for i in range(14):
    many.record_offer(i + 1, "P01" if i % 2 else "P03", "AQ01", 2000 + i,
                      under_name="X社")
big_view = build_user_prompt_v3(big_buyer, many, 20, 60, names_q, cfg_id)
check("自社の買付は件数を切り捨てずに全件出る",
      all(f"[O{n:04d}]" in big_view for n in range(1, 15))
      and "[自社が出した売買買付（全件）]" in big_view)

print("== 案4: 計画欄は strategy と next_milestone だけ必須 ==")
plan_schema = action_schema_v3(buyer_v3)
check("必須の計画欄は2つだけ",
      all(k in plan_schema["required"] for k in ("strategy", "next_milestone"))
      and not any(k in plan_schema["required"] for k in (
          "goal_assessment", "expected_goal_effect", "alternatives",
          "revision_reason")))
check("任意の計画欄はスキーマに残る（書いてもよい）",
      all(k in plan_schema["properties"] for k in (
          "goal_assessment", "expected_goal_effect", "alternatives",
          "revision_reason")))
sparse_plan = normalize_acquirer_plan_v3(
    {"strategy": "面積の大きい区画から当たる", "next_milestone": "第9月までに3件"})
check("任意欄が無くてもテレメトリの形は保たれる（無ければ空）",
      sparse_plan["strategy"] == "面積の大きい区画から当たる"
      and sparse_plan["goal_assessment"] == ""
      and sparse_plan["alternatives"] == [])
check("計画欄の埋め方を書式で強制しない",
      "alternativesは2〜3件" not in build_system_prompt_v3(buyer_v3, cfg_v3, 48))

print("== 案2: 仲介が報告するまで依頼主には回答が見えない ==")
L4 = mk()
buyer_v = Agent("AQ01", "acquirer", "X社", "外部会社",
                extra={"mandate": "m", "aliases": ["A社"]})
q4 = request_owner_inquiry(L4, 1, "AQ01", "BR01", "P01", "")["inquiry_id"]
inquire_owner_intent(L4, 2, "BR01", "P01", q4, "")
answer_owner_inquiry(L4, 3, q4, "HH01", "willing_to_sell", "4200", "")
client_before = client_inquiries_text(buyer_v, L4, names_q)
check("所有者の回答は、仲介の報告前は依頼主の観測に出ない",
      "willing_to_sell" not in client_before and "4200" not in client_before
      and "状態:requested" in client_before and "報告:なし" in client_before)
buyer_v.extra["parcel_last_action"] = {
    "P01": {"step": 1, "action": "request_owner_inquiry",
            "outcome": "inquiry_request", "amount": 0}}
pipe_before = acquirer_pipeline_text(buyer_v, L4, names_q)
check("案件記録にも回答の存在は出ない",
      "相手最終返答:-" in pipe_before and f"{q4}=requested" in pipe_before)
report_owner_intent(L4, 4, "BR01", "AQ01", q4, "willing_to_sell", "4200", "")
client_after = client_inquiries_text(buyer_v, L4, names_q)
check("報告が届いてはじめて意向と希望額が依頼主の観測になる",
      "willing_to_sell" in client_after and "希望額:4200万" in client_after
      and "状態:reported" in client_after)
check("案件記録の相手最終返答も報告で更新される",
      "相手最終返答:inquiry_report(第4月)" in acquirer_pipeline_text(buyer_v, L4, names_q))

print("== 案3: 登記は照会した時点の内容として見える ==")
L5 = mk()
L5.enable_on_demand_financing(["AQ01"])
buyer_r5 = Agent("AQ01", "acquirer", "X社", "外部会社",
                 extra={"mandate": "m", "aliases": ["A社"]})
seed_acquirer_intelligence_v3(
    buyer_r5, L5, {"acquirer_initial_intelligence": {
        "market_research": True, "land_registry_scope": "non_public"}})
before = " | ".join(registry_view_rows(buyer_r5, L5))
check("参入前調査の内容と確認月が並ぶ",
      "名義:住民A" in before and "確認:参入前調査" in before)
o5 = L5.record_offer(2, "P01", "AQ01", 3000, under_name="A社")
L5.record_accept(3, o5["offer_id"], "HH01")
stale = " | ".join(registry_view_rows(buyer_r5, L5))
check("再照会していない区画の名義変更は自動的には見えない",
      "P01[北町] 名義:住民A" in stale)
check_land_registry_v3(L5, 4, buyer_r5, "P01")
fresh = " | ".join(registry_view_rows(buyer_r5, L5))
check("再照会した区画だけ内容と確認月が更新される",
      "P01[北町] 名義:A社" in fresh and "確認:第4月" in fresh)

print("== Codexレビュー反映: 状態遷移と情報配送の穴を塞ぐ ==")
L6 = mk()
buyer6 = Agent("AQ01", "acquirer", "X社", "外部会社",
               extra={"mandate": "m", "aliases": ["A社"]})
q6 = request_owner_inquiry(L6, 1, "AQ01", "BR01", "P01", "")["inquiry_id"]
check("回答が届く前に報告はできない",
      report_owner_intent(L6, 2, "BR01", "AQ01", q6, "willing_to_sell", "3000", "")["reason"]
      == "inquiry_requested")
inquire_owner_intent(L6, 2, "BR01", "P01", q6, "")
check("照会しただけの段階でも報告はできない",
      report_owner_intent(L6, 3, "BR01", "AQ01", q6, "willing_to_sell", "3000", "")["reason"]
      == "inquiry_asked")
answer_owner_inquiry(L6, 3, q6, "HH01", "undecided", "unknown", "")
check("依頼済みの照会を別の相手へは報告できない",
      report_owner_intent(L6, 4, "BR01", "HH02", q6, "undecided", "unknown", "")["reason"]
      == "not_the_client")
check("回答が届いた照会は依頼主へ報告できる",
      report_owner_intent(L6, 4, "BR01", "AQ01", q6, "undecided", "unknown", "")["kind"]
      == "inquiry_report")
check("同じ回答を二度は返せない（再回答で状態を巻き戻さない）",
      answer_owner_inquiry(L6, 5, q6, "HH01", "willing_to_sell", "5000", "")["reason"]
      == "inquiry_reported")
check("進行済みの照会は再照会で巻き戻せない",
      inquire_owner_intent(L6, 5, "BR01", "P01", q6, "")["reason"] == "inquiry_reported")
L7 = mk()
q7 = request_owner_inquiry(L7, 1, "AQ01", "BR01", "P01", "")["inquiry_id"]
check("依頼済み照会の区画は差し替えられない",
      inquire_owner_intent(L7, 2, "BR01", "P03", q7, "")["reason"] == "parcel_mismatch")
check("希望額の欄が空なら不成立（欠損をunknownに補完しない）",
      normalize_price_value("") is None and normalize_price_value(None) is None)
check("明示された unknown は事実として通る",
      normalize_price_value("unknown") == "unknown")
check("接頭辞と種別が食い違うIDは解決しない",
      Ledger._normalize_id("[LEASE-O0001]", "O") == "LEASE-O0001")
check("種別の合う旧表記はこれまでどおり解決する",
      Ledger._normalize_id("[OFFER-O0001]", "O") == "O0001")
parse_row = own_result_row(3, "", "", {"kind": "parse_fail",
                                       "reason": "unparseable_response"})
check("世界の中には帳簿の事実だけを返す（パース失敗の内部事情を出さない）",
      parse_row["kind"] == "not_recorded"
      and parse_row["reason"] == "no_action_recorded")

print("== 依頼主の文面も仲介へ届く ==")
L8 = mk()
broker8 = Agent("BR01", "broker", "B01", "仲介")
q8 = request_owner_inquiry(L8, 1, "AQ01", "BR01", "P01",
                           "P01について売却または賃貸の意向を確認してほしい")["inquiry_id"]
view8 = build_user_prompt_v3(broker8, L8, 2, 60, names_q, cfg_id)
check("依頼主が書いた文面が仲介の観測に届く",
      "売却または賃貸の意向を確認してほしい" in view8 and f"[{q8}] P01" in view8)

# ---------------------------------------------------------------------------
# v4（同期3フェーズ・手続き動詞なし）
# ---------------------------------------------------------------------------

print("== v4: 買付の記帳と経路 ==")


def mk4(fee_rate: float = 0.03) -> Ledger:
    parcels = [
        Parcel("P01", 0, 0, "北町", "residential", "HH01", 2400, area_sqm=100,
               registered_name="住民A"),
        Parcel("P02", 1, 0, "北町", "residential", "HH01", 6000, area_sqm=400,
               registered_name="住民A"),
        Parcel("P03", 2, 0, "北町", "residential", "HH02", 2400, area_sqm=120,
               registered_name="住民B"),
        Parcel("P04", 3, 0, "北町", "public", "MU01", 2400, area_sqm=200,
               registered_name="市"),
    ]
    ledger = Ledger(parcels, {"HH01": 900, "HH02": 900, "BR01": 100, "AQ01": 0,
                              "MU01": 0})
    ensure_v4_state(ledger)
    ledger.v4_fee_rate = fee_rate
    ledger.enable_on_demand_financing(["AQ01"])
    return ledger


AQ4 = Agent("AQ01", "acquirer", "X社", "取得会社")
AQ4.extra["aliases"] = ["A社", "B社"]
AQ4.extra["mandate"] = "この会社だけが知る目的"
AQ4.extra["monthly_offer_capacity"] = 8
HH4 = Agent("HH01", "household", "R01", "住民")
MU4 = Agent("MU01", "municipality", "G01", "行政")
NAMES4 = {"HH01": "R01", "HH02": "R02", "BR01": "B01", "AQ01": "X社", "MU01": "G01"}
CFG4 = {
    "world": {"town_name": "A市", "background": "背景", "block_names": ["北町"],
              "corporate_records": ["A社: 記録"]},
    "social": {"venues": [{"id": "V01", "label": "飲食店"}],
               "public_directory": ["AQ01: X社"]},
    "scenario": {},
}

L4 = mk4()
bad_name = record_offer_v4(L4, 1, AQ4, "P01", 3000, "Z社", "direct", "", ["BR01"])
check("使えない名義の買付は成立しない",
      bad_name["kind"] == "offer_rejected" and bad_name["reason"] == "unknown_legal_name")
missing_broker = record_offer_v4(L4, 1, AQ4, "P01", 3000, "A社", "broker", "", ["BR01"])
check("仲介経由なのに仲介が指定されていない買付は成立しない（世界が割り当てない）",
      missing_broker["kind"] == "offer_rejected"
      and missing_broker["reason"] == "unknown_broker")
bad_price = record_offer_v4(L4, 1, AQ4, "P01", "たかい", "A社", "direct", "", ["BR01"])
check("金額でない価格は成立しない（0に補完しない）",
      bad_price["kind"] == "offer_rejected" and bad_price["reason"] == "invalid_amount")
public_offer = record_offer_v4(L4, 1, AQ4, "P04", 3000, "A社", "direct", "", ["BR01"])
check("公有地への買付は成立しない", public_offer["kind"] == "offer_rejected")

ok_offer = record_offer_v4(L4, 1, AQ4, "P01", 3000, "A社", "broker", "BR01", ["BR01"],
                           "ご検討ください")
oid4 = ok_offer["offer_id"]
check("仲介経由の買付が記帳され、経路が残る",
      ok_offer["kind"] == "offer" and ok_offer["via"] == "broker"
      and channel_text(L4, oid4, NAMES4) == "仲介B01経由")
check("買付は出した月のうちに所有者の応答対象になる",
      owners_with_offers(L4, 1).get("HH01", [])[0].offer_id == oid4)

print("== v4: 応答と清算 ==")
no_res = respond_to_offer_v4(L4, 1, oid4, "HH01", "no_response", 0)
check("答えないことも選択として記録され、買付は開いたままになる",
      no_res["kind"] == "offer_no_response" and L4.offers[oid4].status == "open")
counter4 = respond_to_offer_v4(L4, 1, oid4, "HH01", "counter", 4000)
check("逆提示は台帳の状態遷移として記帳される",
      counter4["kind"] == "counter" and L4.offers[oid4].counter_price == 4000)
check("逆提示は買付を出した側の観測に出る",
      any("逆提示4000万" in row for row in own_offers_rows(AQ4, L4, NAMES4)))

L5 = mk4()
o5 = record_offer_v4(L5, 1, AQ4, "P01", 3000, "A社", "broker", "BR01", ["BR01"])["offer_id"]
before_broker = L5.cash["BR01"]
accepted = respond_to_offer_v4(L5, 1, o5, "HH01", "accept", 0)
check("受諾で所有権が移り、名義は買い手が選んだものになる",
      accepted["kind"] == "transfer" and L5.parcels["P01"].owner_id == "AQ01"
      and L5.parcels["P01"].registered_name == "A社")
check("仲介経由の成立で手数料が自動記帳される（契約条件の3%）",
      L5.cash["BR01"] - before_broker == 90
      and L5.v4_broker_fees.get("BR01") == 90)
check("必要資金は成立時に調達される", L5.financing_raised.get("AQ01", 0) >= 3000)

L6 = mk4()
o6 = record_offer_v4(L6, 1, AQ4, "P03", 2500, "A社", "direct", "", ["BR01"])["offer_id"]
before6 = L6.cash["BR01"]
respond_to_offer_v4(L6, 1, o6, "HH02", "accept", 0)
check("直接の買付では仲介手数料が発生しない", L6.cash["BR01"] == before6)

L7 = mk4()
a7 = record_offer_v4(L7, 1, AQ4, "P01", 3000, "A社", "direct", "", ["BR01"])["offer_id"]
b7 = record_offer_v4(L7, 1, AQ4, "P01", 3200, "B社", "direct", "", ["BR01"])["offer_id"]
respond_to_offer_v4(L7, 1, a7, "HH01", "accept", 0)
second = respond_to_offer_v4(L7, 1, b7, "HH01", "accept", 0)
check("同じ区画は二重に成立しない（先に成立した1件だけが移転する）",
      second["kind"] == "accept_rejected" and L7.parcels["P01"].owner_id == "AQ01")

print("== v4: 条例（制定されれば機構として効く） ==")
L8 = mk4()
bad_ord = enact_ordinance_v4(L8, 1, "MU01", "", "本文", 100, 2)
check("条文が空の条例は成立しない",
      bad_ord["kind"] == "ordinance_rejected" and bad_ord["reason"] == "missing_text")
bad_param = enact_ordinance_v4(L8, 1, "MU01", "届出条例", "本文", 100, -1)
check("成立しない数値の条例は記帳されない",
      bad_param["kind"] == "ordinance_rejected"
      and bad_param["reason"] == "invalid_parameters")
ord8 = enact_ordinance_v4(L8, 2, "MU01", "届出条例", "300㎡超は届出", 300, 2)
check("条例は制定した月には施行されない（全主体は同月を並行して判断している）",
      ord8["effective_step"] == 3 and active_ordinance(L8, 2) is None
      and filing_delay_for(L8, L8.parcels["P02"], 2) == 0)
check("施行後は対象面積を超える取得にだけ届出の遅延がかかる",
      filing_delay_for(L8, L8.parcels["P02"], 3) == 2
      and filing_delay_for(L8, L8.parcels["P01"], 3) == 0)
enact_ordinance_v4(L8, 3, "MU01", "改正届出条例", "50㎡超は届出", 50, 1)
check("後から制定された条例が有効になる（上書き）",
      active_ordinance(L8, 4)["threshold_sqm"] == 50
      and filing_delay_for(L8, L8.parcels["P01"], 4) == 1)

L9 = mk4()
enact_ordinance_v4(L9, 1, "MU01", "届出条例", "300㎡超は届出", 300, 2)
o9 = record_offer_v4(L9, 2, AQ4, "P02", 5000, "A社", "broker", "BR01", ["BR01"])["offer_id"]
filed = respond_to_offer_v4(L9, 2, o9, "HH01", "accept", 0)
check("届出が要る取得は、受諾しても即日には成立しない",
      filed["kind"] == "filing_required" and filed["due_step"] == 4
      and L9.parcels["P02"].owner_id == "HH01")
check("届出期間中は成立しない", execute_pending_transfers_v4(L9, 3) == []
      and L9.parcels["P02"].owner_id == "HH01")
done9 = execute_pending_transfers_v4(L9, 4)
check("届出期間を終えた月に成立し、手数料もその時点で発生する",
      done9 and done9[0]["kind"] == "transfer"
      and L9.parcels["P02"].owner_id == "AQ01"
      and L9.v4_broker_fees.get("BR01") == 150)

L10 = mk4()
enact_ordinance_v4(L10, 1, "MU01", "届出条例", "300㎡超は届出（遅延なし）", 300, 0)
o10 = record_offer_v4(L10, 2, AQ4, "P02", 5000, "A社", "direct", "", ["BR01"])["offer_id"]
check("遅延0か月の届出制度では、その月のうちに成立する",
      respond_to_offer_v4(L10, 2, o10, "HH01", "accept", 0)["kind"] == "transfer")

print("== v4: 観測とスキーマ ==")
schema_aq = phase1_schema_v4(AQ4, CFG4, ["BR01", "BR02"])
offers_schema = schema_aq["properties"]["offers"]
check("買付を出さない月が許される（最低件数を課さない）",
      "minItems" not in offers_schema and offers_schema["maxItems"] == 8)
check("経路の仲介は選択肢から選ぶ（自由入力で取りこぼさない）",
      offers_schema["items"]["properties"]["broker_id"]["enum"] == ["none", "BR01", "BR02"])
def _enum_values(node):
    found = []
    if isinstance(node, dict):
        if "enum" in node:
            found.extend(node["enum"])
        for value in node.values():
            found.extend(_enum_values(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_enum_values(value))
    return found


check("スキーマの選択肢に空文字を入れない（実APIが400で拒否する）",
      all(str(v).strip() != "" for role_agent in
          (AQ4, HH4, MU4, Agent("MD01", "media", "J01", "記者"),
           Agent("BR01", "broker", "B01", "仲介"))
          for v in _enum_values(phase1_schema_v4(role_agent, CFG4, ["BR01", "BR02"])))
      and all(str(v).strip() != "" for v in _enum_values(phase2_schema_v4())))
check("『取次ぎ先なし』は none と書く（direct の指定と矛盾しない）",
      record_offer_v4(mk4(), 1, AQ4, "P01", 3000, "A社", "direct", "none",
                      ["BR01"])["kind"] == "offer")
check("名義は使える登記名義からしか選べない",
      offers_schema["items"]["properties"]["under_name"]["enum"] == ["X社", "A社", "B社"])
check("応答には『今月は答えない』が含まれる",
      "no_response" in phase2_schema_v4()["properties"]["responses"]["items"]
      ["properties"]["decision"]["enum"])
check("仲介は話すだけで、取引を止める行為を持たない",
      set(phase1_schema_v4(Agent("BR01", "broker", "B01", "仲介"), CFG4, ["BR01"])
          ["properties"]) == {"location", "utterance", "utterance_channel",
                              "utterance_to", "memory"})
check("条例を制定できるのは行政だけ（記者のスキーマには無い）",
      "ordinance_title" not in phase1_schema_v4(
          Agent("MD01", "media", "J01", "記者"), CFG4, ["BR01"])["properties"]
      and "ordinance_title" in phase1_schema_v4(MU4, CFG4, ["BR01"])["properties"])

system_v4 = build_system_prompt_v4(AQ4, CFG4, 48)
banned_verbs = ("check_land_registry", "market_research", "internal_review",
                "financing_review", "contact_broker", "client_followup",
                "request_owner_inquiry", "due_diligence")
check("v4に手続き動詞は存在しない",
      not any(v in system_v4 for v in banned_verbs))
check("v4のプロンプトに「AI」表記はない", "AI" not in system_v4)
check("買付を出すことを指示していない",
      "必ず買付" not in system_v4 and "買付を出せ" not in system_v4)

L11 = mk4()
view_aq = build_phase1_prompt_v4(AQ4, L11, 1, 12, NAMES4, CFG4)
check("登記は最初から公開情報として観測に出る（調べる行為は要らない）",
      "[公開されている土地登記（全区画）]" in view_aq and "P01" in view_aq
      and "名義:住民A" in view_aq)
check("計画欄・代替案・根拠IDの必須はない",
      "alternatives" not in view_aq and "goal_assessment" not in view_aq
      and "evidence" not in view_aq)
for banned in ("すべき", "推奨", "優先度", "不要", "望ましい"):
    check(f"観測に評価語・当為が混ざらない（{banned}）", banned not in view_aq)

record_offer_v4(L11, 1, AQ4, "P01", 3000, "A社", "broker", "BR01", ["BR01"])
respond_to_offer_v4(L11, 1, "O0001", "HH01", "accept", 0)
view_hh = build_phase1_prompt_v4(HH4, L11, 2, 12, NAMES4, CFG4)
check("住民には近隣の名義と、その変更が観測に出る",
      "[近隣の土地登記（公開情報）]" in view_hh and "P02" in view_hh)
check("登記統計は名義別の面積として集計できる",
      any("A社" in row for row in registry_stats_rows(L11, NAMES4)))
check("全区画の登記は誰が見ても同じ事実である",
      len(registry_rows(L11, NAMES4)) == len(L11.parcels))

L12 = mk4()
o12 = record_offer_v4(L12, 1, AQ4, "P01", 3600, "A社", "direct", "", ["BR01"])["offer_id"]
view_p2 = build_phase2_prompt_v4(HH4, L12, 1, 12, NAMES4,
                                 owners_with_offers(L12, 1)["HH01"])
check("所有者には金額・名義・経路・評価額との比が事実として出る",
      f"[{o12}]" in view_p2 and "名義:A社" in view_p2 and "直接" in view_p2
      and "1.50倍" in view_p2)
check("応答フェーズにも評価語・当為は書かない",
      "すべき" not in view_p2 and "推奨" not in view_p2)

print("== v4: Codex実装レビューで直した欠陥の固定 ==")

L20 = mk4()
check("存在しない買付には『答えない』も記帳できない",
      respond_to_offer_v4(L20, 1, "O9999", "HH01", "no_response", 0)["reason"]
      == "offer_not_open")
o20 = record_offer_v4(L20, 1, AQ4, "P03", 2500, "A社", "direct", "", ["BR01"])["offer_id"]
check("他人に届いた買付には『答えない』も記帳できない",
      respond_to_offer_v4(L20, 1, o20, "HH01", "no_response", 0)["reason"] == "not_owner")

L21 = mk4()
a21 = record_offer_v4(L21, 1, AQ4, "P01", 3000, "A社", "direct", "", ["BR01"])["offer_id"]
b21 = record_offer_v4(L21, 1, AQ4, "P01", 3300, "B社", "direct", "", ["BR01"])["offer_id"]
respond_to_offer_v4(L21, 1, a21, "HH01", "accept", 0)
check("成立した区画に残る他の買付は、その場で成立不能として終端する",
      L21.offers[b21].status == "void")
check("買い手が所有者になった区画で、自分の買付を自分で受けることはできない",
      respond_to_offer_v4(L21, 2, b21, "AQ01", "accept", 0)["kind"] == "accept_rejected"
      and sum(1 for r in L21.records if r["kind"] == "transfer") == 1)
check("成立不能になった買付は応答対象に残らない",
      "AQ01" not in owners_with_offers(L21, 2))

L22 = mk4()
enact_ordinance_v4(L22, 1, "MU01", "届出条例", "300㎡超は届出", 300, 2)
x22 = record_offer_v4(L22, 2, AQ4, "P02", 5000, "A社", "direct", "", ["BR01"])["offer_id"]
y22 = record_offer_v4(L22, 2, AQ4, "P02", 5200, "B社", "direct", "", ["BR01"])["offer_id"]
respond_to_offer_v4(L22, 2, x22, "HH01", "accept", 0)
respond_to_offer_v4(L22, 2, y22, "HH01", "accept", 0)
done22 = execute_pending_transfers_v4(L22, 4)
kinds22 = [r["kind"] for r in done22]
check("同じ区画の届出待ちが2件あっても、成立するのは1件だけ",
      kinds22.count("transfer") == 1
      and sum(1 for r in L22.records if r["kind"] == "transfer") == 1)
check("成立できなかった届出待ちは成立不能として終端する",
      L22.offers[y22].status == "void" and L22.v4_pending == [])
check("届出を終えた成立は、届出を経た成立として印が残る",
      any(r.get("filed") for r in done22 if r["kind"] == "transfer"))

L23 = mk4()
o23 = record_offer_v4(L23, 1, AQ4, "P01", 3000, "A社", "direct", "", ["BR01"])["offer_id"]
L23.v4_ordinance = {"step": 0, "effective_step": 1, "by": "MU01", "title": "t",
                    "body": "b", "threshold_sqm": 50, "delay_months": 2}
o23b = record_offer_v4(L23, 1, AQ4, "P03", 2500, "A社", "direct", "", ["BR01"])["offer_id"]
respond_to_offer_v4(L23, 1, o23b, "HH02", "accept", 0)
L23.parcels["P03"].owner_id = "HH01"      # 届出の間に所有者が変わった状況を作る
void23 = execute_pending_transfers_v4(L23, 3)
check("届出の間に所有者が変わった案件は成立させない",
      void23 and void23[0]["kind"] == "filing_void"
      and L23.parcels["P03"].owner_id == "HH01")

L24 = mk4()
mixed = record_offer_v4(L24, 1, AQ4, "P01", 3000, "A社", "direct", "BR01", ["BR01"])
check("直接の買付に取次ぎ先を入れた指定は、黙って捨てずに不成立にする",
      mixed["kind"] == "offer_rejected"
      and mixed["reason"] == "broker_not_allowed_for_direct")

L25 = mk4()
o25 = record_offer_v4(L25, 1, AQ4, "P01", 3000, "A社", "broker", "BR01", ["BR01"])["offer_id"]
respond_to_offer_v4(L25, 1, o25, "HH01", "accept", 0)
BR25 = Agent("BR01", "broker", "B01", "仲介")
relay25 = broker_relay_rows(BR25, L25, NAMES4)
check("仲介の記録は、取り次いだ時点の相手のまま残る（成立後の名義に書き換えない）",
      any("取次ぎ先:R01" in row for row in relay25))
view_br = build_phase1_prompt_v4(BR25, L25, 2, 12, NAMES4, CFG4)
check("仲介は全区画の登記を配られない（自分が扱った案件と手数料だけ）",
      "[公開されている土地登記（全区画）]" not in view_br
      and "[自分が取り次いだ買付（機械記録）]" in view_br)

many = [own_result_row(1, "make_offer", f"P{i:02d}", {"kind": "offer"}) for i in range(1, 13)]
check("自分の行為の結果は8件で切り捨てない",
      own_results_text_v4(many).count("make_offer") == 12)

L26 = mk4()
o26 = record_offer_v4(L26, 1, AQ4, "P01", 3600, "A社", "direct", "", ["BR01"])["offer_id"]
HH26 = Agent("HH01", "household", "R01", "住民")
HH26.memory = "先月は仲介から連絡があった"
inbox26 = [{"from": "BR01", "text": "湾岸で名義が変わっているようです", "step": 1,
            "obs_id": "TALK-V01-M01-BR01", "location": "V01"}]
view26 = build_phase2_prompt_v4(HH26, L26, 1, 12, NAMES4,
                                owners_with_offers(L26, 1)["HH01"], inbox26)
check("応答フェーズでも、その月にすでに届いていた情報と記憶が見えている",
      "湾岸で名義が変わっている" in view26 and "先月は仲介から連絡があった" in view26)

print()
print(f"RESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
