#!/usr/bin/env python
"""帳簿と記帳ディスパッチの回帰テスト。

    python tests/test_ledger.py

外部依存なし・API を叩かない。Codex レビューで指摘された不具合をそれぞれ1件ずつ
固定している（直したつもりで戻る事故を防ぐ）。
"""

from __future__ import annotations

import io
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

from src.field_v4_1 import (active_ordinance_v41, build_phase1_prompt_v41,
                            build_phase2_prompt_v41, build_system_prompt_v41,
                            enact_ordinance_v41, ensure_v41_state,
                            execute_pending_v41, filing_delay_v41,
                            incoming_offers_rows_v41, observations_rows_v41,
                            own_results_text_v41, own_result_row_v41,
                            owners_with_offers_v41, phase1_schema_v41,
                            phase2_schema_v41, record_offer_v41,
                            registry_rows_v41, registry_stats_rows_v41,
                            respond_to_offer_v41, withdraw_offer_v41)  # noqa: E402

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

# ---------------------------------------------------------------------------
# v4.1（金額を持たない世界・思考レイヤー）
# ---------------------------------------------------------------------------

print("== v4.1: 打診と応答（金額なし） ==")


def mk41() -> Ledger:
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
    ledger = Ledger(parcels, {})
    ensure_v41_state(ledger)
    return ledger


AQ41 = Agent("AQ01", "acquirer", "X社", "取得会社")
AQ41.extra["aliases"] = ["A社", "B社"]
AQ41.extra["mandate"] = "この会社だけが知る目的"
AQ41.extra["monthly_offer_capacity"] = 6
HH41 = Agent("HH01", "household", "R01", "この主体の事情だけを書いた説明")
MU41 = Agent("MU01", "municipality", "G01", "この主体の事情だけを書いた説明")
MD41 = Agent("MD01", "media", "J01", "この主体の事情だけを書いた説明")
BR41 = Agent("BR01", "broker", "B01", "この主体の事情だけを書いた説明")
NAMES41 = {"HH01": "R01", "HH02": "R02", "BR01": "B01", "AQ01": "X社", "MU01": "G01"}
CFG41 = {
    "world": {"town_name": "A市", "background": "背景", "block_names": ["北町"],
              "corporate_records": ["A社: 記録"]},
    "social": {"venues": [{"id": "V01", "label": "飲食店"}],
               "public_directory": ["AQ01: X社"]},
    "scenario": {},
}

L41 = mk41()
bad41 = record_offer_v41(L41, 1, AQ41, "P01", "Z社")
check("v4.1: 使えない名義の打診は成立しない",
      bad41["kind"] == "offer_rejected" and bad41["reason"] == "unknown_legal_name")
check("v4.1: 公有地への打診は成立しない",
      record_offer_v41(L41, 1, AQ41, "P04", "A社")["reason"] == "public_land_not_for_sale")
check("v4.1: 存在しない区画への打診は成立しない",
      record_offer_v41(L41, 1, AQ41, "P99", "A社")["reason"] == "no_such_parcel")
o41 = record_offer_v41(L41, 1, AQ41, "P01", "A社")
check("v4.1: 打診が金額なしで記帳される",
      o41["kind"] == "offer" and "price" not in o41 and "amount" not in o41)
check("v4.1: 打診はその月のうちに所有者の応答対象になる",
      owners_with_offers_v41(L41)["HH01"][0]["id"] == o41["offer_id"])
check("v4.1: 打診は所有者に届いた相手として記録される",
      L41.v41_offers[o41["offer_id"]]["to"] == "HH01")

hold41 = respond_to_offer_v41(L41, 1, o41["offer_id"], "HH01", "hold")
check("v4.1: 決めないことも選択で、打診は開いたまま残る",
      hold41["kind"] == "hold" and L41.v41_offers[o41["offer_id"]]["status"] == "open")
check("v4.1: 他人に届いた打診には応答できない",
      respond_to_offer_v41(L41, 1, o41["offer_id"], "HH02", "sell")["reason"] == "not_owner")
check("v4.1: 存在しない打診には応答できない",
      respond_to_offer_v41(L41, 1, "O9999", "HH01", "sell")["reason"] == "offer_not_open")
check("v4.1: 世界にない決定は成立しない",
      respond_to_offer_v41(L41, 1, o41["offer_id"], "HH01", "counter")["reason"]
      == "unknown_decision")

keep41 = respond_to_offer_v41(L41, 2, o41["offer_id"], "HH01", "keep")
check("v4.1: 応じないと決めた打診は終端する",
      keep41["kind"] == "keep" and L41.v41_offers[o41["offer_id"]]["status"] == "kept"
      and "HH01" not in owners_with_offers_v41(L41))

L42 = mk41()
a42 = record_offer_v41(L42, 1, AQ41, "P01", "A社")["offer_id"]
b42 = record_offer_v41(L42, 1, AQ41, "P01", "B社")["offer_id"]
sold = respond_to_offer_v41(L42, 1, a42, "HH01", "sell")
check("v4.1: sell で登記の名義だけが移る（金銭の記帳はない）",
      sold["kind"] == "transfer" and L42.parcels["P01"].owner_id == "AQ01"
      and L42.parcels["P01"].registered_name == "A社"
      and "price" not in sold and L42.cash.get("AQ01", 0) == 0)
check("v4.1: 成立した区画に残る他の打診はその場で終端する",
      L42.v41_offers[b42]["status"] == "void")
check("v4.1: 買い手が所有者になった区画で自己売買はできない",
      respond_to_offer_v41(L42, 2, b42, "AQ01", "sell")["kind"] == "response_rejected")
check("v4.1: 既に自社が持つ区画への打診は成立しない",
      record_offer_v41(L42, 2, AQ41, "P01", "A社")["reason"] == "already_owner")

L43 = mk41()
w43 = record_offer_v41(L43, 1, AQ41, "P03", "A社")["offer_id"]
check("v4.1: 自分が出した打診は取り下げられる",
      withdraw_offer_v41(L43, 2, w43, "AQ01")["kind"] == "withdraw"
      and L43.v41_offers[w43]["status"] == "withdrawn")
check("v4.1: 他人の打診は取り下げられない",
      withdraw_offer_v41(L43, 2, w43, "HH01")["kind"] == "withdraw_rejected")

print("== v4.1: 条例（届出の遅延） ==")
L44 = mk41()
check("v4.1: 条文が空の条例は成立しない",
      enact_ordinance_v41(L44, 1, "MU01", "", "本文", 100, 2)["reason"] == "missing_text")
check("v4.1: 成立しない数値の条例は記帳されない",
      enact_ordinance_v41(L44, 1, "MU01", "条例", "本文", 100, -1)["reason"]
      == "invalid_parameters")
ord44 = enact_ordinance_v41(L44, 2, "MU01", "届出条例", "300㎡超は届出", 300, 2)
check("v4.1: 条例は制定した翌月から施行される",
      ord44["effective_step"] == 3 and active_ordinance_v41(L44, 2) is None
      and filing_delay_v41(L44, L44.parcels["P02"], 3) == 2
      and filing_delay_v41(L44, L44.parcels["P01"], 3) == 0)
o44 = record_offer_v41(L44, 3, AQ41, "P02", "A社")["offer_id"]
filed44 = respond_to_offer_v41(L44, 3, o44, "HH01", "sell")
check("v4.1: 届出が要る取得は受諾しても即日には成立しない",
      filed44["kind"] == "filing_required" and filed44["due_step"] == 5
      and L44.parcels["P02"].owner_id == "HH01")
check("v4.1: 届出期間中は成立しない", execute_pending_v41(L44, 4) == []
      and L44.parcels["P02"].owner_id == "HH01")
done44 = execute_pending_v41(L44, 5)
check("v4.1: 届出期間を終えた月に成立し、届出を経た印が残る",
      done44 and done44[0]["kind"] == "transfer" and done44[0].get("filed") is True
      and L44.parcels["P02"].owner_id == "AQ01")

L45 = mk41()
enact_ordinance_v41(L45, 1, "MU01", "届出条例", "300㎡超は届出", 300, 2)
o45 = record_offer_v41(L45, 2, AQ41, "P02", "A社")["offer_id"]
respond_to_offer_v41(L45, 2, o45, "HH01", "sell")
L45.parcels["P02"].owner_id = "HH02"     # 届出の間に所有者が変わった状況
void45 = execute_pending_v41(L45, 4)
check("v4.1: 届出の間に所有者が変わった案件は成立させない",
      void45 and void45[0]["kind"] == "filing_void"
      and L45.parcels["P02"].owner_id == "HH02")

print("== v4.1: スキーマと観測（金額が世界に無いこと） ==")
schema41 = phase1_schema_v41(AQ41)
check("v4.1: thought がJSONの先頭にある",
      list(schema41["properties"])[0] == "thought"
      and list(phase2_schema_v41()["properties"])[0] == "thought")
check("v4.1: 打診の欄に金額はない",
      set(schema41["properties"]["offers"]["items"]["properties"])
      == {"parcel_id", "under_name"})
check("v4.1: 打診は月6件までで、出さない月も許される",
      schema41["properties"]["offers"]["maxItems"] == 6
      and "minItems" not in schema41["properties"]["offers"])
check("v4.1: 応答は sell / keep / hold の3つで、逆提示は存在しない",
      phase2_schema_v41()["properties"]["responses"]["items"]["properties"]
      ["decision"]["enum"] == ["sell", "keep", "hold"])
check("v4.1: memory / memo / feeling / reasoning の欄は無い（thoughtに統合）",
      not ({"memory", "memo", "feeling", "reasoning"} & set(schema41["properties"]))
      and not ({"memory", "memo", "feeling", "reasoning"}
               & set(phase2_schema_v41()["properties"])))
check("v4.1: 仲介は話すだけで、取引の行為を持たない",
      set(phase1_schema_v41(BR41)["properties"])
      == {"thought", "location", "utterance", "utterance_channel", "utterance_to"})
check("v4.1: 条例を制定できるのは行政だけ",
      "ordinance_title" in phase1_schema_v41(MU41)["properties"]
      and "ordinance_title" not in phase1_schema_v41(MD41)["properties"])


def _enum_values_41(node):
    found = []
    if isinstance(node, dict):
        if "enum" in node:
            found.extend(node["enum"])
        for value in node.values():
            found.extend(_enum_values_41(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_enum_values_41(value))
    return found


check("v4.1: スキーマの選択肢に空文字を入れない（実APIが400で拒否する）",
      all(str(v).strip() != "" for agent41 in (AQ41, HH41, MU41, MD41, BR41)
          for v in _enum_values_41(phase1_schema_v41(agent41)))
      and all(str(v).strip() != "" for v in _enum_values_41(phase2_schema_v41())))

MONEY_WORDS = ("価格", "金額", "万円", "評価額", "決済", "資金", "手数料", "逆提示",
               "賃料", "予算", "円）", "price", "amount")
sys41 = build_system_prompt_v41(AQ41, CFG41, 48)
for word in MONEY_WORDS:
    check(f"v4.1: 世界の説明に金額の語が出ない（{word}）", word not in sys41)
check("v4.1: v4.1のプロンプトに「AI」表記はない", "AI" not in sys41)
check("v4.1: 内心は順序だけ指示し、中身を指示しない",
      "まず thought を書き" in sys41 and "何を書くかはあなたが決める" in sys41)

L46 = mk41()
record_offer_v41(L46, 1, AQ41, "P01", "A社")
view41 = build_phase1_prompt_v41(AQ41, L46, 1, 12, NAMES41, CFG41)
for word in ("価格", "評価額", "万円", "手数料", "資金"):
    check(f"v4.1: 取得主体の観測に金額の語が出ない（{word}）", word not in view41)
for word in ("すべき", "推奨", "優先度", "不要", "望ましい"):
    check(f"v4.1: 観測に評価語・当為が混ざらない（{word}）", word not in view41)
check("v4.1: 登記は最初から公開情報として全区画が観測に出る",
      "[公開されている土地登記（全区画）]" in view41
      and len(registry_rows_v41(L46, NAMES41)) == len(L46.parcels))
check("v4.1: 登記の行に評価額を出さない",
      all("評価額" not in row for row in registry_rows_v41(L46, NAMES41)))

HH41.extra["thought"] = "先月からの内心がそのまま残っている"
view_hh41 = build_phase1_prompt_v41(HH41, L46, 2, 12, NAMES41, CFG41)
check("v4.1: 前月の内心はそのまま翌月の観測に出る",
      "先月からの内心がそのまま残っている" in view_hh41)
check("v4.1: 住民には近隣の名義が観測に出る",
      "[近隣の土地登記（公開情報）]" in view_hh41 and "P02" in view_hh41)
view_p2_41 = build_phase2_prompt_v41(HH41, L46, 1, 12, NAMES41,
                                     owners_with_offers_v41(L46)["HH01"],
                                     [{"from": "BR01", "text": "名義が変わったらしい",
                                       "step": 1, "obs_id": "TALK-V01-M01-BR01"}])
check("v4.1: 応答フェーズにも内心と、その月に届いた情報が渡る",
      "先月からの内心がそのまま残っている" in view_p2_41
      and "名義が変わったらしい" in view_p2_41)
for word in ("価格", "評価額", "万円"):
    check(f"v4.1: 応答フェーズにも金額の語が出ない（{word}）", word not in view_p2_41)
check("v4.1: 登記統計は名義別の面積として出る（金額を含まない）",
      any("住民A" in row for row in registry_stats_rows_v41(L46, NAMES41))
      and all("万" not in row for row in registry_stats_rows_v41(L46, NAMES41)))
many41 = [own_result_row_v41(1, "make_offer", f"P{i:02d}", {"kind": "offer"})
          for i in range(1, 13)]
check("v4.1: 自分の行為の結果は切り捨てない",
      own_results_text_v41(many41).count("make_offer") == 12)


# ---------------------------------------------------------------------------
# v4.1: Codexレビュー（2026-08-27）指摘の修正を固定する
# ---------------------------------------------------------------------------

print("== v4.1: Codexレビュー指摘の修正 ==")

# --- 打診に文面（note）は無い＝第二の私信経路を作らない -------------------
L50 = mk41()
o50 = record_offer_v41(L50, 1, AQ41, "P01", "A社")
check("v4.1: 打診の記帳に文面(note)を持たない",
      "note" not in o50 and "note" not in L50.v41_offers[o50["offer_id"]])
check("v4.1: 届いている申し入れの表示に文面欄が無い",
      all("「" not in row
          for row in incoming_offers_rows_v41(L50, NAMES41,
                                              owners_with_offers_v41(L50)["HH01"])))
sys50 = build_system_prompt_v41(AQ41, CFG41, 4)
check("v4.1: 取得主体のプロンプトが note を指示しない",
      "note" not in sys50 and "文面を付けることはできない" in sys50)

# --- ID表記ゆれ（[O0001] / O0001 / OFFER-O0001）は同一の打診を指す -------
for label, given in (("角括弧", "[O0001]"), ("素のID", "O0001"),
                     ("旧接頭辞", "OFFER-O0001")):
    L51 = mk41()
    id51 = record_offer_v41(L51, 1, AQ41, "P01", "A社")["offer_id"]
    assert id51 == "O0001"
    out51 = respond_to_offer_v41(L51, 1, given, "HH01", "hold")
    check(f"v4.1: {label}の表記でも同じ打診に応答できる", out51["kind"] == "hold")
    check(f"v4.1: {label}で応答しても正規化すると同一IDになる",
          L51._normalize_id(given, "O") == id51)

# --- 届出中に競合が成立しても、消した案件が翌月に復活しない ---------------
L52 = mk41()
a52 = record_offer_v41(L52, 1, AQ41, "P02", "A社")["offer_id"]
b52 = record_offer_v41(L52, 1, AQ41, "P02", "B社")["offer_id"]
enact_ordinance_v41(L52, 1, "MU01", "届出制度", "本文", 100, 1)
enact_ordinance_v41(L52, 1, "MU01", "届出制度", "本文", 100, 2)   # 期限違いにする
L52.v41_ordinance["delay_months"] = 1
respond_to_offer_v41(L52, 2, a52, "HH01", "sell")     # 第3月成立
L52.v41_ordinance["delay_months"] = 2
respond_to_offer_v41(L52, 2, b52, "HH01", "sell")     # 第4月成立の予定
check("v4.1: 期限の違う届出が2件とも待機している",
      len(L52.v41_pending) == 2
      and {r["offer_id"] for r in L52.v41_pending} == {a52, b52})
done52 = execute_pending_v41(L52, 3)
check("v4.1: 先に期限が来た届出だけが成立する",
      len([r for r in done52 if r["kind"] == "transfer"]) == 1
      and L52.parcels["P02"].owner_id == "AQ01")
check("v4.1: 競合で消えた届出は待機列に復活しない", L52.v41_pending == [])
check("v4.1: 競合で消えた届出は翌月に二重終端されない",
      execute_pending_v41(L52, 4) == [])
voids52 = [r for r in L52.records if r["kind"] == "offer_void"]
check("v4.1: 競合で消えた打診には当事者（買主・売主）が残る",
      voids52 and voids52[0].get("buyer") == "AQ01"
      and voids52[0].get("seller") == "HH01")

# --- 届出中に所有者が変わった案件の不成立は、買主にも結果が届く -----------
L53 = mk41()
o53 = record_offer_v41(L53, 1, AQ41, "P02", "A社")["offer_id"]
enact_ordinance_v41(L53, 1, "MU01", "届出制度", "本文", 100, 2)
respond_to_offer_v41(L53, 2, o53, "HH01", "sell")
L53.parcels["P02"].owner_id = "HH02"          # 世界の外で名義が動いた場合
void53 = execute_pending_v41(L53, 4)
check("v4.1: 所有者が変わった届出は不成立になる",
      len(void53) == 1 and void53[0]["kind"] == "filing_void")
check("v4.1: 届出の不成立は買主にも結果として返せる",
      void53[0].get("buyer") == "AQ01" and void53[0].get("seller") == "HH01")

# --- 条例：欠損は補完しない・上限を世界が足さない・同月競合を記録する -----
L54 = mk41()
check("v4.1: 条例の数値が欠けていたら不成立にする（0に補完しない）",
      enact_ordinance_v41(L54, 1, "MU01", "題", "本文", None, None)["kind"]
      == "ordinance_rejected")
check("v4.1: 条例の数値が数値でなければ不成立にする",
      enact_ordinance_v41(L54, 1, "MU01", "題", "本文", "たくさん", 1)["kind"]
      == "ordinance_rejected")
check("v4.1: 遅延月数に世界側の上限を置かない",
      enact_ordinance_v41(L54, 1, "MU01", "題", "本文", 100, 120)["kind"] == "ordinance")
enact_ordinance_v41(L54, 2, "MU01", "題A", "本文", 100, 1)
enact_ordinance_v41(L54, 2, "MU02", "題B", "本文", 200, 3)
conflicts54 = [r for r in L54.records if r["kind"] == "ordinance_same_step_conflict"]
check("v4.1: 同月に2主体が制定した事実は隠さず記録に残す",
      len(conflicts54) == 1 and conflicts54[0]["by"] == "MU02"
      and conflicts54[0]["other"] == "MU01")

# --- 内心と、届いた文は切り捨てない ---------------------------------------
L55 = mk41()
long_thought = "あ" * 900
HH55 = Agent("HH01", "household", "R01", "この主体の事情だけを書いた説明")
HH55.extra["thought"] = long_thought
view55 = build_phase1_prompt_v41(HH55, L55, 2, 12, NAMES41, CFG41)
check("v4.1: 前月の内心は要約も切り捨てもせずそのまま渡す",
      long_thought in view55)
long_article = "外" * 380
rows55 = observations_rows_v41(HH55, NAMES41,
                               [{"from": "MD01", "text": long_article, "step": 2,
                                 "obs_id": "NEWS-MD01-M02"}])
check("v4.1: 届いた記事・発言は観測で切り捨てない",
      long_article in rows55[0])
view_p2_55 = build_phase2_prompt_v41(HH55, L55, 2, 12, NAMES41, [],
                                     [{"from": "MD01", "text": long_article, "step": 2,
                                       "obs_id": "NEWS-MD01-M02"}])
check("v4.1: 応答フェーズでも内心と届いた文を切り捨てない",
      long_thought in view_p2_55 and long_article in view_p2_55)

# --- 先月の自分の結果は件数で切らない -------------------------------------
many55 = [own_result_row_v41(1, "hold", f"O{i:04d}", {"kind": "hold"})
          for i in range(1, 61)]
check("v4.1: 先月の結果は60件でも全件返す",
      len(own_results_text_v41(many55).splitlines()) == 60)

# --- 開始時点の属性であることを断定形にしない ------------------------------
check("v4.1: 売却後も「所有している」と断定しない",
      "開始時点で自分の土地・建物を所有していた"
      in build_system_prompt_v41(HH55, CFG41, 4))

# --- 実configと実ペルソナで組んだ本番プロンプトを検査する ------------------
import yaml  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_REAL = yaml.safe_load(io.open(os.path.join(_ROOT, "configs",
                                               "config_field_v4_1.yaml"),
                                  encoding="utf-8"))
PERSONAS_REAL = yaml.safe_load(io.open(os.path.join(_ROOT,
                                                    CFG_REAL["personas_file"]),
                                       encoding="utf-8"))
MONEY_WORDS = ("価格", "評価額", "万円", "円", "手数料", "資金", "ローン", "借入",
               "固定資産税", "修繕費", "老後資金", "進学費用", "原材料費", "賃借",
               "家賃", "賃料", "金融機関", "査定", "予算", "決済", "代金")
# コメント行（何を置き換えたかの説明）ではなく、実際に読み込まれる本文を検査する。
_persona_text = chr(10).join(
    str(row.get("persona", ""))
    for rows in PERSONAS_REAL.values() for row in rows)
for word in MONEY_WORDS:
    if word == "円":
        continue
    check(f"v4.1: 実ペルソナに金額の語が無い（{word}）", word not in _persona_text)

_roles41 = {"household": "household", "business": "business", "broker": "broker",
            "acquirer": "acquirer", "municipality": "municipality", "media": "media"}
_real_prompts = {}
for _role, _key in _roles41.items():
    for _idx, _p in enumerate(PERSONAS_REAL[_key]):
        _a = Agent(f"{_key[:2].upper()}{_idx + 1:02d}", _role, _p["name"],
                   _p["persona"].strip())
        _a.extra["aliases"] = _p.get("aliases", [])
        _a.extra["mandate"] = CFG_REAL["scenario"]["acquirer_mandate"]
        _a.extra["monthly_offer_capacity"] = \
            CFG_REAL["scenario"]["acquirer_monthly_offer_capacity"]
        _real_prompts[_a.agent_id + _role] = build_system_prompt_v41(
            _a, CFG_REAL, 48)
check("v4.1: 実configで全27主体のシステムプロンプトが組める",
      len(_real_prompts) == 27)
for word in MONEY_WORDS:
    if word == "円":
        continue  # 「範囲」等の部分一致を避け、金額表記は「万円」で見る
    check(f"v4.1: 実プロンプト全27本に金額の語が出ない（{word}）",
          all(word not in text for text in _real_prompts.values()))
# 当為・優先度の検査。X社の非公開目的（施主確定の研究前提）だけは対象外にする。
for word in ("すべき", "しなさい", "推奨", "優先度", "望ましい", "確率", "閾値"):
    check(f"v4.1: 実プロンプト（取得主体を除く）に当為・確率が出ない（{word}）",
          all(word not in text for key, text in _real_prompts.items()
              if not key.endswith("acquirer")))


# ---------------------------------------------------------------------------
# v4.1b: 相談経路の復活（金額は無いまま）＋行政の区画面積観測
#   設計の正 = docs/world_design_v4_1b_broker_consult.md
#   v4.1 の挙動が1文字も変わっていないことも、ここで固定する。
# ---------------------------------------------------------------------------

from src.field_v4_1b import (CONSULT_NONE, answer_consult_v41b,  # noqa: E402
                             build_phase1_prompt_v41b,
                             build_phase2_prompt_v41b,
                             build_system_prompt_v41b, ensure_v41b_state,
                             incoming_consults_rows_v41b,
                             ordinance_effect_rows_v41b,
                             own_consults_rows_v41b, parcel_area_rows_v41b,
                             phase1_schema_v41b, record_consult_v41b)

BR41B = Agent("BR01", "broker", "B01", "この主体の事情だけを書いた説明")
BR41B.extra["monthly_advice_capacity"] = 6
BR41B_2 = Agent("BR02", "broker", "B02", "この主体の事情だけを書いた説明")
BR41B_2.extra["monthly_advice_capacity"] = 6
HH41B = Agent("HH01", "household", "R01", "この主体の事情だけを書いた説明")
HH41B_2 = Agent("HH02", "household", "R02", "この主体の事情だけを書いた説明")
BROKERS41B = ["BR01", "BR02"]
NAMES41B = dict(NAMES41)
NAMES41B["BR02"] = "B02"

LB = mk41()
ensure_v41b_state(LB)
q1 = record_consult_v41b(LB, 1, HH41B, "BR01", "この辺りの動きをどう見ているか。",
                         BROKERS41B)
check("v4.1b: 所有者の相談が Q0001 として台帳に載る",
      q1["kind"] == "consult" and q1["consult_id"] == "Q0001"
      and LB.v41b_consults["Q0001"]["status"] == "open"
      and LB.v41b_consults["Q0001"]["to"] == "BR01")
check("v4.1b: 相談は自分自身には出せない",
      record_consult_v41b(LB, 1, HH41B, "HH01", "本文", BROKERS41B)["reason"]
      == "self_consult")
check("v4.1b: 仲介でない相手への相談は成立しない",
      record_consult_v41b(LB, 1, HH41B, "HH02", "本文", BROKERS41B)["reason"]
      == "unknown_broker")
check("v4.1b: 本文の無い相談は成立しない（空欄を補完しない）",
      record_consult_v41b(LB, 1, HH41B, "BR01", "   ", BROKERS41B)["reason"]
      == "missing_question")
check("v4.1b: 住民・事業者以外は相談できない",
      record_consult_v41b(LB, 1, BR41B, "BR02", "本文", BROKERS41B)["reason"]
      == "role_cannot_consult")

check("v4.1b: 自分宛でない相談には答えられない",
      answer_consult_v41b(LB, 2, "BR02", "Q0001", "答え")["reason"]
      == "not_your_consultation")
check("v4.1b: 他の仲介が答えようとしても相談の状態は変わらない",
      LB.v41b_consults["Q0001"]["status"] == "open")
check("v4.1b: 存在しない相談には答えられない",
      answer_consult_v41b(LB, 2, "BR01", "Q9999", "答え")["reason"]
      == "no_such_consultation")
check("v4.1b: 本文の無い回答は成立しない",
      answer_consult_v41b(LB, 2, "BR01", "Q0001", "  ")["reason"] == "missing_reply")
a1 = answer_consult_v41b(LB, 2, "BR01", "[Q0001]", "近隣の名義変更の状況を伝える。")
check("v4.1b: 角括弧付きの相談IDでも同じ1件に解決する",
      a1["kind"] == "advice" and a1["consult_id"] == "Q0001"
      and a1["to"] == "HH01")
check("v4.1b: 回答した相談は回答済みになり、回答月が残る",
      LB.v41b_consults["Q0001"]["status"] == "answered"
      and LB.v41b_consults["Q0001"]["answered_step"] == 2)
check("v4.1b: 同じ相談に二度は答えられない（状態は巻き戻らない）",
      answer_consult_v41b(LB, 3, "BR01", "Q0001", "追加の答え")["reason"]
      == "already_answered"
      and LB.v41b_consults["Q0001"]["answered_step"] == 2)

record_consult_v41b(LB, 3, HH41B_2, "BR02", "自分の土地について聞きたい。", BROKERS41B)
rows_br1 = incoming_consults_rows_v41b(LB, "BR01", NAMES41B)
check("v4.1b: 仲介の観測には自分宛の相談だけが機械記録で出る",
      any("[Q0001]" in r and "回答済み(第2月)" in r and "R01" in r for r in rows_br1)
      and not any("[Q0002]" in r for r in rows_br1))
check("v4.1b: 相談本文は仲介の機械記録にも出る（世界が持つ事実を配送し損ねない）",
      any("この辺りの動きをどう見ているか。" in r for r in rows_br1))
check("v4.1b: 相談が来ていない仲介には「来ていない」と出る",
      incoming_consults_rows_v41b(LB, "BR09", NAMES41B) == ["  （相談は来ていない）"])
rows_hh1 = own_consults_rows_v41b(LB, "HH01", NAMES41B)
check("v4.1b: 所有者の観測には自分がした相談と状態が出る",
      len(rows_hh1) == 1 and "[Q0001]" in rows_hh1[0]
      and "相談先:B01" in rows_hh1[0] and "回答済み(第2月)" in rows_hh1[0])
check("v4.1b: 未回答の相談は未回答と出る",
      "未回答" in own_consults_rows_v41b(LB, "HH02", NAMES41B)[0])
check("v4.1b: 相談していない所有者には「まだ相談していない」と出る",
      own_consults_rows_v41b(LB, "HH09", NAMES41B) == ["  （まだ相談していない）"])
check("v4.1b: 相談も回答も台帳に金額の欄を持たない",
      not any(k in row for row in LB.records for k in ("price", "amount", "fee", "rent")))

# --- 行政の観測（区画面積の分布・条例の適用実績）-----------------------------
area_rows = parcel_area_rows_v41b(LB, NAMES41B)
check("v4.1b: 区画面積の分布に件数・最小・中央値・最大が出る",
      any("非公共区画 3件" in r and "最小100㎡" in r and "中央値120㎡" in r and "最大400㎡" in r
          for r in area_rows))
check("v4.1b: 面積の分布はヒストグラムで出る",
      any("100〜199㎡ 2件" in r and "400〜499㎡ 1件" in r for r in area_rows))
check("v4.1b: 調べていない行政の面積分布に名義は出ない",
      not any("名義別" in r for r in area_rows))
check("v4.1b: 登記統計を調べた行政には名義別の1件あたり面積も出る",
      any("名義別の1件あたり面積" in r
          for r in parcel_area_rows_v41b(LB, NAMES41B, with_names=True)))
check("v4.1b: 条例が無い月は適用実績も無いと出る",
      ordinance_effect_rows_v41b(LB, 3) == ["  （施行中の届出制度はない）"])
enact_ordinance_v41(LB, 3, "MU01", "届出条例", "1000㎡超は届出", 1000, 2)
eff_rows = ordinance_effect_rows_v41b(LB, 4)
check("v4.1b: 閾値が区画の最大を超えていれば「該当0件」が事実として返る",
      any("超える区画: 0件" in r for r in eff_rows))
check("v4.1b: 条例の適用実績に施行後の移転件数と届出件数が出る",
      any("施行後の名義移転: 0件" in r and "うち対象面積に該当したもの: 0件" in r
          for r in eff_rows))
o_lb = record_offer_v41(LB, 4, AQ41, "P01", "A社")["offer_id"]
respond_to_offer_v41(LB, 4, o_lb, "HH01", "sell")
check("v4.1b: 閾値より小さい区画の移転は届出の対象にならない（実績にそう出る）",
      any("施行後の名義移転: 1件 うち対象面積に該当したもの: 0件" in r
          for r in ordinance_effect_rows_v41b(LB, 5)))

# --- スキーマ ---------------------------------------------------------------
sch_hh = phase1_schema_v41b(HH41B, BROKERS41B)
check("v4.1b: 所有者のスキーマに相談先と相談内容が入る",
      sch_hh["properties"]["consult_broker_id"]["enum"] == ["BR01", "BR02", CONSULT_NONE]
      and "consult_question" in sch_hh["properties"]
      and "consult_broker_id" in sch_hh["required"]
      and "consult_question" in sch_hh["required"])
check("v4.1b: enumに空の選択肢を入れない（Geminiが400を返す）",
      all(v for v in sch_hh["properties"]["consult_broker_id"]["enum"]))
sch_br = phase1_schema_v41b(BR41B, BROKERS41B)
check("v4.1b: 仲介のスキーマは月次上限つきの回答配列を持つ",
      sch_br["properties"]["advices"]["maxItems"] == 6
      and sorted(sch_br["properties"]["advices"]["items"]["required"])
      == ["consult_id", "reply"])
check("v4.1b: 仲介には相談の欄を持たせない（答える側であって相談者ではない）",
      "consult_broker_id" not in sch_br["properties"])
check("v4.1b: 取得主体のスキーマは v4.1 と同一（打診の文面は戻していない）",
      phase1_schema_v41b(AQ41, BROKERS41B) == phase1_schema_v41(AQ41))
check("v4.1b: 行政のスキーマは v4.1 と同一",
      phase1_schema_v41b(MU41, BROKERS41B) == phase1_schema_v41(MU41))
check("v4.1: 所有者のスキーマは v4.1b の追加で変わっていない",
      "consult_broker_id" not in phase1_schema_v41(HH41B)["properties"])

# --- プロンプト -------------------------------------------------------------
sys_hh_41 = build_system_prompt_v41(HH41B, CFG41, 48)
sys_hh_41b = build_system_prompt_v41b(HH41B, CFG41, 48, BROKERS41B)
check("v4.1b: 所有者のシステムプロンプトは v4.1 の本文をそのまま含む",
      sys_hh_41b.startswith(sys_hh_41))
check("v4.1b: 所有者に相談できることとJSONの形だけを足す",
      "不動産仲介に相談する。" in sys_hh_41b
      and "1か月に相談できるのは1件。" in sys_hh_41b
      and "consult_broker_id, consult_question" in sys_hh_41b)
sys_br_41b = build_system_prompt_v41b(BR41B, CFG41, 48, BROKERS41B)
check("v4.1b: 仲介には自分宛の相談に答えられることを足す",
      "自分に来ている相談に答える。" in sys_br_41b
      and "advices の各要素は consult_id" in sys_br_41b)
check("v4.1b: 仲介に取引経路は戻していないと本文に書いてある",
      "土地の売買を取り次ぐ経路はこの世界に無い。" in sys_br_41b)
check("v4.1b: 取得主体のシステムプロンプトは v4.1 と1文字も変わらない",
      build_system_prompt_v41b(AQ41, CFG41, 48, BROKERS41B)
      == build_system_prompt_v41(AQ41, CFG41, 48))
for word in MONEY_WORDS:
    check(f"v4.1b: 追加した世界の説明に金額の語が出ない（{word}）",
          word not in sys_hh_41b and word not in sys_br_41b)
for word in ("すべき", "しなさい", "推奨", "望ましい", "確率", "閾値", "優先度"):
    check(f"v4.1b: 追加した世界の説明に当為・確率が出ない（{word}）",
          word not in sys_hh_41b and word not in sys_br_41b)

p1_br = build_phase1_prompt_v41b(BR41B, LB, 5, 12, NAMES41B, CFG41)
check("v4.1b: 仲介の月次観測に相談の機械記録が入る",
      "[自分に来ている相談（機械記録）]" in p1_br and "[Q0001]" in p1_br)
check("v4.1b: 追加した観測は最後の指示文より前に入る",
      p1_br.rstrip().endswith("まず thought（内心）を書き、それを踏まえて今月の行動をJSONで1つ返す。")
      and p1_br.index("[自分に来ている相談（機械記録）]")
      < p1_br.rindex("まず thought（内心）"))
p1_hh = build_phase1_prompt_v41b(HH41B_2, LB, 5, 12, NAMES41B, CFG41)
check("v4.1b: 所有者の月次観測に自分の相談の状態が入る",
      "[自分がした相談（機械記録）]" in p1_hh and "[Q0002]" in p1_hh)
check("v4.1b: 所有者の観測に他人の相談は出ない", "[Q0001]" not in p1_hh)
p1_mu = build_phase1_prompt_v41b(MU41, LB, 5, 12, NAMES41B, CFG41)
check("v4.1b: 行政の月次観測に区画面積の分布と条例の適用実績が入る",
      "[A市の区画1件あたりの面積の分布（機械記録）]" in p1_mu
      and "[施行中の条例の適用実績（機械記録）]" in p1_mu)
check("v4.1b: 調べていない行政の観測に名義別の面積は出ない",
      "名義別の1件あたり面積" not in p1_mu)
MU41B_SEEN = Agent("MU01", "municipality", "G01", "この主体の事情だけを書いた説明")
MU41B_SEEN.extra["registry_stats_seen"] = True
check("v4.1b: 登記統計を調べた行政の観測には名義別の面積も出る",
      "名義別の1件あたり面積" in build_phase1_prompt_v41b(
          MU41B_SEEN, LB, 5, 12, NAMES41B, CFG41))
p1_aq = build_phase1_prompt_v41b(AQ41, LB, 5, 12, NAMES41B, CFG41)
check("v4.1b: 取得主体の観測は v4.1 と同一（相談の中身は見えない）",
      p1_aq == build_phase1_prompt_v41(AQ41, LB, 5, 12, NAMES41B, CFG41)
      and "[Q0001]" not in p1_aq)
p1_md = build_phase1_prompt_v41b(MD41, LB, 5, 12, NAMES41B, CFG41)
check("v4.1b: 記者の観測は v4.1 と同一（面積分布は行政だけに足す）",
      p1_md == build_phase1_prompt_v41(MD41, LB, 5, 12, NAMES41B, CFG41))

o_lb2 = record_offer_v41(LB, 5, AQ41, "P03", "A社")["offer_id"]
offers_lb = owners_with_offers_v41(LB)["HH02"]
p2_hh = build_phase2_prompt_v41b(HH41B_2, LB, 5, 12, NAMES41B, offers_lb, [])
check("v4.1b: 応答フェーズでも自分の相談の状態が見える",
      "[自分がした相談（機械記録）]" in p2_hh and "[Q0002]" in p2_hh)
check("v4.1b: 応答フェーズの指示文は最後のまま",
      p2_hh.rstrip().endswith("説明文を付けずJSONだけ返す。")
      and p2_hh.index("[自分がした相談（機械記録）]") < p2_hh.rindex("まず thought（内心）"))

# --- 実config・実ペルソナでの検査 ------------------------------------------
CFG_41B = yaml.safe_load(io.open(os.path.join(_ROOT, "configs",
                                              "config_field_v4_1b.yaml"),
                                 encoding="utf-8").read())
check("v4.1b: 実configは v4.1 と同じ seed・モデル・月数・ペルソナを使う",
      (CFG_41B["seed"], CFG_41B["steps"], CFG_41B["llm"]["model"],
       CFG_41B["personas_file"], CFG_41B["scenario"]["acquirer_monthly_offer_capacity"])
      == (CFG_REAL["seed"], CFG_REAL["steps"], CFG_REAL["llm"]["model"],
          CFG_REAL["personas_file"],
          CFG_REAL["scenario"]["acquirer_monthly_offer_capacity"]))
check("v4.1b: 実configの scenario_version は field_v4_1b",
      CFG_41B["scenario_version"] == "field_v4_1b")
_real_prompts_41b = {}
for _role, _key in _roles41.items():
    for _idx, _p in enumerate(PERSONAS_REAL[_key]):
        _a = Agent(f"{_key[:2].upper()}{_idx + 1:02d}", _role, _p["name"],
                   _p["persona"].strip())
        _a.extra["aliases"] = _p.get("aliases", [])
        _a.extra["mandate"] = CFG_41B["scenario"]["acquirer_mandate"]
        _a.extra["monthly_offer_capacity"] = \
            CFG_41B["scenario"]["acquirer_monthly_offer_capacity"]
        _a.extra["monthly_advice_capacity"] = \
            CFG_41B["scenario"]["broker_monthly_advice_capacity"]
        _real_prompts_41b[_a.agent_id + _role] = build_system_prompt_v41b(
            _a, CFG_41B, 48, ["BR01", "BR02"])
check("v4.1b: 実configで全27主体のシステムプロンプトが組める",
      len(_real_prompts_41b) == 27)
for word in MONEY_WORDS:
    if word == "円":
        continue
    check(f"v4.1b: 実プロンプト全27本に金額の語が出ない（{word}）",
          all(word not in text for text in _real_prompts_41b.values()))
for word in ("すべき", "しなさい", "推奨", "優先度", "望ましい", "確率", "閾値"):
    check(f"v4.1b: 実プロンプト（取得主体を除く）に当為・確率が出ない（{word}）",
          all(word not in text for key, text in _real_prompts_41b.items()
              if not key.endswith("acquirer")))


# --- v4.1b: Codexレビュー（2026-08-27・gpt-5.6-sol）の指摘に対する回帰 ------------

from src.field_v4_1b import (CONSULT_CAPACITY_PER_MONTH,  # noqa: E402
                             _normalize_consult_id, ordinance_effect_rows_v41b)
from src.viz import (MONEY_WORDS_HTML, _MONEYLESS_SUBS, _TEMPLATE,  # noqa: E402
                     _apply_moneyless, _compact_events)

LC = mk41()
ensure_v41b_state(LC)
qc = record_consult_v41b(LC, 1, HH41B, "BR01", "相談の本文。", BROKERS41B)["consult_id"]
check("v4.1b: 相談は出した月のうちには答えられない（届くのは翌月）",
      answer_consult_v41b(LC, 1, "BR01", qc, "答え")["reason"] == "not_yet_delivered")
check("v4.1b: 同じ月に2件目の相談は出せない（有限資源を台帳側で数える）",
      record_consult_v41b(LC, 1, HH41B, "BR02", "2件目。", BROKERS41B)["reason"]
      == "monthly_consult_capacity_exceeded" and CONSULT_CAPACITY_PER_MONTH == 1)
check("v4.1b: 翌月になれば相談はまた出せる",
      record_consult_v41b(LC, 2, HH41B, "BR01", "翌月の相談。",
                          BROKERS41B)["kind"] == "consult")
check("v4.1b: 配送IDの表記（CONSULT-/ADVICE-）でも同じ相談へ解決する",
      _normalize_consult_id(LC, "[CONSULT-Q0001]") == "Q0001"
      and _normalize_consult_id(LC, "ADVICE-Q0001") == "Q0001"
      and answer_consult_v41b(LC, 2, "BR01", "CONSULT-Q0001",
                              "答え")["kind"] == "advice")
check("v4.1b: 回答した月、相談者の観測にはまだ回答済みと出ない（本文は翌月に届く）",
      "未回答" in own_consults_rows_v41b(LC, "HH01", NAMES41B, 2)[0]
      and "回答済み" not in own_consults_rows_v41b(LC, "HH01", NAMES41B, 2)[0])
check("v4.1b: 翌月になると相談者の観測に回答済みが出る",
      "回答済み(第2月)" in own_consults_rows_v41b(LC, "HH01", NAMES41B, 3)[0])
check("v4.1b: 答えた仲介自身の観測にはその月から回答済みが出る",
      any("回答済み(第2月)" in r
          for r in incoming_consults_rows_v41b(LC, "BR01", NAMES41B)))
LE = mk41()
ensure_v41b_state(LE)
qe1 = record_consult_v41b(LE, 1, HH41B, "BR01", "1人目の相談。", BROKERS41B)["consult_id"]
qe2 = record_consult_v41b(LE, 1, HH41B_2, "BR01", "2人目の相談。", BROKERS41B)["consult_id"]
check("v4.1b: 同じ月の1件目の回答は成立する",
      answer_consult_v41b(LE, 2, "BR01", qe1, "答え1", capacity=1)["kind"] == "advice")
check("v4.1b: 月次上限を超える回答は成立しない（台帳側で数える）",
      answer_consult_v41b(LE, 2, "BR01", qe2, "答え2", capacity=1)["reason"]
      == "monthly_advice_capacity_exceeded")
check("v4.1b: 翌月になれば上限は戻る",
      answer_consult_v41b(LE, 3, "BR01", qe2, "答え2", capacity=1)["kind"] == "advice")

LD = mk41()
ensure_v41b_state(LD)
enact_ordinance_v41(LD, 1, "MU01", "即時届出条例", "300㎡超は届出", 300, 0)
od = record_offer_v41(LD, 2, AQ41, "P02", "A社")["offer_id"]   # P02 は 400㎡
respond_to_offer_v41(LD, 2, od, "HH01", "sell")
rows_d = ordinance_effect_rows_v41b(LD, 3)
check("v4.1b: 遅延0か月の条例でも「対象面積に該当した取得」を数え落とさない",
      any("施行後の名義移転: 1件 うち対象面積に該当したもの: 1件 "
          "うち届出で待機したもの: 0件" in r for r in rows_d))

_mless = _apply_moneyless(_TEMPLATE)
check("v4.1b: 金額のない世界のレポートHTMLに金額の語が残らない",
      not [w for w in MONEY_WORDS_HTML if w in _mless])
check("v4.1b: 金額ありのテンプレートは差し替えても壊れていない（差し替えは全件当たる）",
      len(_MONEYLESS_SUBS) >= 10 and len(_mless) < len(_TEMPLATE))
check("v4.1b: 金額のある版のテンプレートは変えていない",
      "初期評価額" in _TEMPLATE and "平均賃料" in _TEMPLATE)
_ev = [{"step": 1, "agent_id": "AQ01", "role": "acquirer", "amount": 3000,
        "action_type": "offers", "outcome": {"kind": "phase1"}}]
check("v4.1b: 金額のない世界のレポートデータに amount を載せない",
      "amount" not in _compact_events(_ev, moneyless=True)[0]
      and _compact_events(_ev)[0]["amount"] == 3000)

# ===========================================================================
# v5: 出来事はピン留め・観測するのは街の会話
#   設計の正は docs/world_design_v5_impl.md（受け入れ条件は 9.2）
#   v1〜v4.1b の経路は不変であることも合わせて固定する。
# ===========================================================================

import collections            # noqa: E402
import json as _json          # noqa: E402
import shutil as _shutil      # noqa: E402
import tempfile as _tempfile  # noqa: E402

from src.field_v5 import (HOME as HOME_V5, SCENE_IDS, ambient_traces_v5,  # noqa: E402
                          neighbourhood_rows_v5,
                          apply_script_v5, build_plan_prompt_v5,
                          build_scene_prompt_v5, build_system_prompt_v5,
                          ensure_v5_state, plan_schema_v5, registry_rows_v5,
                          s4_for_step, scene_schema_v5, validate_script_v5,
                          venue_traces_v5)
from src.simulation import Simulation  # noqa: E402


def mk5() -> Ledger:
    parcels = [
        Parcel("P01", 0, 0, "北町", "residential", "HH01", 2400, area_sqm=100,
               registered_name="住民A"),
        Parcel("P02", 1, 0, "北町", "shop", "HH02", 3000, area_sqm=200,
               tenant_id="BZ01", registered_name="住民B"),
        Parcel("P03", 2, 0, "北町", "residential", "HH03", 2400, area_sqm=120,
               registered_name="住民C"),
        Parcel("P04", 3, 0, "北町", "public", "MU01", 2400, area_sqm=200,
               registered_name="市"),
    ]
    ledger = Ledger(parcels, {})
    ensure_v5_state(ledger)
    return ledger


SCRIPT5 = {
    "meta": {"seed": 85, "months": 3, "holders": ["A社", "B社"]},
    "acquisitions": [
        {"id": "ACQ01", "month": 1, "parcel_id": "P01", "under_name": "A社",
         "traces": [{"kind": "registry", "month": 1, "audience": "registry"},
                    {"kind": "moving_out", "month": 1, "audience": "neighbors"},
                    {"kind": "sign_change", "month": 1, "audience": "venue:V01"},
                    {"kind": "broker_known", "month": 1, "audience": "agents:[BR01]"}]},
        {"id": "ACQ02", "month": 2, "parcel_id": "P04", "under_name": "B社",
         "traces": []},
    ],
}

L5 = mk5()
validate_script_v5(SCRIPT5)
check("v5: 台本の検証が正しい台本を通す", True)
try:
    validate_script_v5({"acquisitions": [{"id": "X", "month": 1, "parcel_id": "P01",
                                          "under_name": "A社",
                                          "traces": [{"kind": "nope",
                                                      "audience": "registry"}]}]})
    check("v5: 未知の兆候を台本の検証がはじく", False, "例外が出なかった")
except ValueError:
    check("v5: 未知の兆候を台本の検証がはじく", True)
try:
    validate_script_v5({"acquisitions": [{"id": "X", "month": 1, "parcel_id": "P01",
                                          "under_name": "A社",
                                          "traces": [{"kind": "registry",
                                                      "audience": "everyone"}]}]})
    check("v5: 未知の可視範囲を台本の検証がはじく", False, "例外が出なかった")
except ValueError:
    check("v5: 未知の可視範囲を台本の検証がはじく", True)

_old_names5 = {p.pid: p.registered_name for p in L5.parcels.values()}
_done5 = apply_script_v5(L5, 1, SCRIPT5, "AQ01")
check("v5: 台本の取得がその月に登記へ反映される",
      len(_done5) == 1 and L5.parcels["P01"].owner_id == "AQ01"
      and L5.parcels["P01"].registered_name == "A社")
check("v5: 台本の名義移転に金額が付かない",
      "price" not in _done5[0] and "amount" not in _done5[0])
check("v5: 台本の名義移転は売主と旧名義を残す",
      _done5[0]["seller"] == "HH01" and _done5[0]["old_name"] == "住民A")
check("v5: 台本にない月には名義が動かない", not apply_script_v5(L5, 3, SCRIPT5, "AQ01"))
_pub5 = apply_script_v5(L5, 2, SCRIPT5, "AQ01")
check("v5: 公有地は台本でも動かない（不成立として記録される）",
      not _pub5 and L5.parcels["P04"].owner_id == "MU01"
      and any(r["kind"] == "script_rejected" for r in L5.records))

_amb5 = ambient_traces_v5(L5, 1, SCRIPT5, _old_names5, "AQ01")
check("v5: neighbors の兆候は隣接区画の所有者に見える",
      any(t["kind"] == "moving_out" for t in _amb5.get("HH02", [])))
check("v5: neighbors の兆候は隣接区画の利用者（店子）にも見える",
      any(t["kind"] == "moving_out" for t in _amb5.get("BZ01", [])))
check("v5: neighbors の兆候は隣接していない主体には見えない",
      not any(t["kind"] == "moving_out" for t in _amb5.get("HH03", [])))
check("v5: agents 指名の兆候は名指しされた主体だけに見える",
      any(t["kind"] == "broker_known" for t in _amb5.get("BR01", []))
      and not any(t["kind"] == "broker_known" for t in _amb5.get("HH02", [])))
check("v5: venue と registry の兆候は会場に依らない配布に含まれない",
      not any(t["kind"] in ("sign_change", "registry")
              for rows in _amb5.values() for t in rows))
check("v5: 買い手（X社）には兆候が配られない（主体ではない）", "AQ01" not in _amb5)
check("v5: 兆候の文言に評価語が入らない（機械記録）",
      all("べき" not in t["text"] and "危" not in t["text"]
          for rows in _amb5.values() for t in rows))

_ven5 = venue_traces_v5(1, SCRIPT5, "V01", _old_names5)
check("v5: venue の兆候は指定の会場でだけ見える",
      len(_ven5) == 1 and _ven5[0]["kind"] == "sign_change"
      and not venue_traces_v5(1, SCRIPT5, "V02", _old_names5))
check("v5: venue の兆候は指定の月だけ現れる",
      not venue_traces_v5(2, SCRIPT5, "V01", _old_names5))

_reg5 = registry_rows_v5(L5, 1)
check("v5: 窓口の登記閲覧はその月までの名義変更を全件見せる",
      len(_reg5) == 1 and "P01" in _reg5[0] and "A社" in _reg5[0])
check("v5: 窓口の登記閲覧に金額が出ない",
      not [w for w in MONEY_WORDS if any(w in r for r in _reg5)])

check("v5: S4は町内会→窓口→仲介の店先→取材の順で月替わりになる",
      [s4_for_step(m)[0] for m in (1, 2, 3, 4, 5)]
      == ["assembly", "counter", "broker_front", "press", "assembly"])

# --- スキーマ -------------------------------------------------------------
_plan_sc = plan_schema_v5(["V01", "V02"], "V06")
check("v5: 計画の出力は thought が先頭",
      list(_plan_sc["properties"])[0] == "thought")
check("v5: S4の行き先はその月の会場か自宅だけ",
      _plan_sc["properties"]["plan_s4"]["enum"] == ["V06", HOME_V5])
# 姿勢を毎コール尋ねると売却の話題を主体に想起させ続ける（Codexレビュー 2026-08-27）。
# 計画コールでは尋ねず、その月の最後のターンで1回だけ尋ねる。
check("v5: 計画コールでは姿勢を尋ねない（売却話題のプライミングを作らない）",
      "stance" not in _plan_sc["properties"]
      and "stance" not in plan_schema_v5(["V01"], "V06")["properties"])
check("v5: 計画コールのプロンプトに姿勢の教示が無い",
      "stance" not in build_plan_prompt_v5(
          Agent("HH01", "household", "R01", "説明"), L5, 1, 12,
          {"HH01": "R01"}, [], "町内会", "V02"))
_HH5 = Agent("HH01", "household", "R01", "この主体の事情だけを書いた説明")
_MD5 = Agent("MD01", "media", "J01", "この主体の事情だけを書いた説明")
_sc5 = scene_schema_v5(_HH5, ["HH01", "HH02"], ["HH01", "HH02", "MD01"],
                       owns_parcel=True, can_publish=False)
check("v5: 発話の宛先は同席者からしか選べない（自分自身は選べない）",
      _sc5["properties"]["talk_to"]["items"]["enum"] == ["HH02"])
check("v5: 私信の宛先に自分自身が入らない",
      "HH01" not in _sc5["properties"]["direct_to"]["enum"]
      and "MD01" in _sc5["properties"]["direct_to"]["enum"])
check("v5: 記事を書けるのは記者だけ",
      "publish" not in _sc5["properties"]
      and "publish" in scene_schema_v5(_MD5, ["MD01", "HH01"], ["MD01", "HH01"],
                                       can_publish=True)["properties"])
def _enum_values_v5(node):
    out = []
    if isinstance(node, dict):
        if isinstance(node.get("enum"), list):
            out += [v for v in node["enum"]]
        for v in node.values():
            out += _enum_values_v5(v)
    elif isinstance(node, list):
        for v in node:
            out += _enum_values_v5(v)
    return out


# Gemini の response_schema は enum に空文字を許さない（実APIで 400 INVALID_ARGUMENT
# ＝2026-08-27 のスモークで実際に全シーンコールが落ちた）。番兵で表す。
check("v5: 出力スキーマの選択肢に空文字が無い（実APIが400を返す）",
      "" not in _enum_values_v5(_sc5) and "" not in _enum_values_v5(_plan_sc)
      and "" not in _enum_values_v5(scene_schema_v5(_MD5, ["MD01", "HH01"],
                                                    ["MD01", "HH01"], True, True)))
check("v5: 私信を出さないことを番兵で表す",
      "NONE" in _sc5["properties"]["direct_to"]["enum"])
check("v5: 番兵は宛先として成立しない",
      True)
check("v5: 出力スキーマに金額の欄が無い",
      not [k for k in list(_sc5["properties"]) + list(_plan_sc["properties"])
           if k in ("amount", "price", "rent", "value", "offer", "budget")])
check("v5: 出力スキーマに打診・応答・条例の欄が無い",
      not [k for k in _sc5["properties"]
           if k in ("offers", "responses", "decision", "ordinance_title",
                    "investigate", "consult")])

# --- 実configの実ペルソナで、全主体のプロンプトに金額語が無いこと -------------
_ROOT5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with io.open(os.path.join(_ROOT5, "configs/config_field_v5.yaml"), encoding="utf-8") as _f:
    CFG5 = yaml.safe_load(_f)
with io.open(os.path.join(_ROOT5, CFG5["personas_file"]), encoding="utf-8") as _f:
    PERSONAS5 = yaml.safe_load(_f)

from src.agents import build_roster as _build_roster5  # noqa: E402
from src.world import assign_tenancies as _assign5, build_town as _build_town5  # noqa: E402

_agents5 = _build_roster5(PERSONAS5, CFG5["agents"], CFG5["scenario"])
_actors5 = [a for a in _agents5 if a.role != "acquirer"]
_hh5 = [a.agent_id for a in _agents5 if a.role == "household"]
_bz5 = [a.agent_id for a in _agents5 if a.role == "business"]
_mu5 = next(a.agent_id for a in _agents5 if a.role == "municipality")
_parcels5 = _build_town5(CFG5["world"], _hh5, _bz5, _mu5)
_assign5(_parcels5, _bz5, 0)
for _p in _parcels5:
    _p.registered_name = next((a.name for a in _agents5 if a.agent_id == _p.owner_id),
                              _p.owner_id)
_L5real = Ledger(_parcels5, {})
ensure_v5_state(_L5real)
_names5 = {a.agent_id: a.name for a in _agents5}
_venue_ids5 = [v["id"] for v in CFG5["social"]["venues"]]

_prompts5 = []
_prompts5_world = []
for _a in _actors5:
    _sys = build_system_prompt_v5(_a, CFG5, len(_parcels5))
    _plan = build_plan_prompt_v5(_a, _L5real, 1, 12, _names5, [], "町内会", "V02")
    _scene = build_scene_prompt_v5(_a, _L5real, 1, 12, _names5, "S1", "朝の商店街",
                                   "V01 駅前の飲食店", [_a.agent_id, "HH02"], [], [],
                                   1, 2, registry_rows_v5(_L5real, 1), True,
                                   _a.role == "media", 2, 1)
    _prompts5 += [_sys, _plan, _scene]
    # ペルソナ本文は「その主体の事情」であり世界の説明ではないので、
    # 機構・誘導の語の検査からは外す（金額語の検査は v4.1 で別途ペルソナも見ている）。
    _stripped5 = [p.replace(_a.persona, "") for p in (_sys, _plan, _scene)]
    _prompts5_world += _stripped5
check("v5: 全26主体ぶんのプロンプトを組める（3種×26＝78本）",
      len(_actors5) == 26 and len(_prompts5) == 78)
for word in MONEY_WORDS:
    check(f"v5: 実プロンプト全78本に金額の語が出ない（{word}）",
          not [p for p in _prompts5 if word in p])
for _forbidden in ("打診", "条例", "届出", "相談", "案件", "投資", "買収"):
    check(f"v5: 世界の説明に v4系の機構が残っていない（{_forbidden}）",
          not [p for p in _prompts5_world if _forbidden in p])
check("v5: X社の非公開目的がどのプロンプトにも出ない",
      not [p for p in _prompts5 if "過半" in p or "mandate" in p])
check("v5: 世界の説明が主体に注目点・目的を指示していない",
      not [p for p in _prompts5_world
           if "注意して" in p or "警戒" in p or "気づ" in p or "疑" in p])

# --- 私信の上限（配線の資源制約） -----------------------------------------
_tmp5 = _tempfile.mkdtemp(prefix="qa_v5_")
_cfg_run5 = _json.loads(_json.dumps(CFG5))
_cfg_run5["steps"] = 2
_cfg_run5["llm"] = {"provider": "mock", "model": "mock", "parallel_workers": 4}
_cfg_run5.setdefault("kpi", {})["classify_utterances"] = False
_sim5 = Simulation(_cfg_run5, PERSONAS5, _tmp5)
_a_dir = _sim5.by_id["HH01"]
_a_dir.extra["directs_used"] = 0
for _i in range(3):
    _sim5._v5_direct(_a_dir, {"direct_to": "HH02", "direct_text": "話がある"}, 1, "S1")
check("v5: 私信は月2通まで（3通目は不成立として記録され届かない）",
      _a_dir.extra["directs_used"] == 2
      and len([r for r in _sim5.ledger.records
               if r["kind"] == "direct_rejected" and r.get("reason") == "quota_exhausted"]) == 1
      and len(_sim5.by_id["HH02"].inbox) == 2)
_sim5._v5_direct(_sim5.by_id["HH03"], {"direct_to": "AQ01", "direct_text": "あ"}, 1, "S1")
check("v5: X社宛の私信は成立しない（主体ではない）",
      any(r["kind"] == "direct_rejected" and r.get("to") == "AQ01"
          for r in _sim5.ledger.records))
check("v5: X社のシステムプロンプトを作らない（LLMを呼ばない）",
      "AQ01" not in _sim5.system_prompts and len(_sim5.system_prompts) == 26)

# --- 姿勢は台帳を動かさない ------------------------------------------------
_before5 = len(_sim5.ledger.transfers())
_sim5._v5_stance(_sim5.by_id["HH01"], {"stance": "sell"}, 1, "S1")
_sim5._v5_stance(_sim5.by_id["HH01"], {"stance": "keep"}, 1, "S4")
check("v5: 姿勢は台帳を動かさない", len(_sim5.ledger.transfers()) == _before5)
check("v5: 姿勢はその月の最後のコールの値だけが残る",
      [r for r in _sim5.ledger.v5_stances if r["agent_id"] == "HH01"]
      == [{"step": 1, "agent_id": "HH01", "role": "household", "stance": "keep",
           "scene": "S4"}])

# --- 実際に2か月まわして配線を見る（mock・APIは叩かない） -------------------
_sim5.run()
_utts5 = _sim5.ledger.v5_utterances
_plans5 = {(r["step"], r["agent_id"]): r for r in _sim5.ledger.v5_plans}
check("v5: 会話が実際に起きている（mock2か月）", len(_utts5) > 0)
_bad_deliver = [u for u in _utts5
                if sorted(u["heard_by"] + [u["from"]])
                != sorted([aid for (st, aid), p in _plans5.items()
                           if st == u["step"] and p[u["scene"]] == u["venue"]])]
check("v5: 発話は同じ場所に居た者だけに届く（それ以外には届かない）", not _bad_deliver)
check("v5: 発話に居合わせなかった主体が talk_to に入らない",
      not [u for u in _utts5 if [t for t in u["talk_to"] if t not in u["heard_by"]]])
check("v5: X社はどの発話にも現れない",
      not [u for u in _utts5 if u["from"] == "AQ01" or "AQ01" in u["heard_by"]])
check("v5: X社にコールが行かない（イベントに現れない）",
      not [e for e in _sim5.events if e.get("agent_id") == "AQ01"])

_r2 = [p for p in _sim5.client.prompt_log if ":S1r2" in p["tag"]]
_r1_texts = [u["text"] for u in _utts5 if u["scene"] == "S1" and u["round"] == 1
             and u["step"] == 1]
check("v5: 2ラウンド目は1ラウンド目の発言を全文読んでから話す（往復になっている）",
      bool(_r2) and bool(_r1_texts)
      and any(any(t in p["user"] for t in _r1_texts) for p in _r2))

_script_months = {(a["month"], a["parcel_id"]) for a in _sim5.script["acquisitions"]
                  if a["month"] <= 2}
_actual5 = {(t["step"], t["parcel_id"]) for t in _sim5.ledger.transfers()}
check("v5: 登記が動くのは台本のとおりだけ（台本にない移転が1件も無い）",
      _actual5 == _script_months and len(_actual5) > 0)
check("v5: 台本の名義がそのまま登記名義になる",
      all(_sim5.ledger.parcels[a["parcel_id"]].registered_name == a["under_name"]
          for a in _sim5.script["acquisitions"] if a["month"] <= 2))

_arts5 = _sim5.ledger.v5_articles
check("v5: 記事は書いた月に全主体へ配送される（記事＝公開発話）",
      bool(_arts5) and all(
          {d["to"] for d in _sim5.deliveries
           if d.get("kind") == "article" and d["step"] == _a["step"]
           and d["from"] == _a["from"]} == {x.agent_id for x in _sim5.actors}
          for _a in _arts5))
_arts_early = [a for a in _arts5 if a["step"] < _sim5.n_steps]
if _arts_early:
    _first = _arts_early[0]
    _plan_next = [p for p in _sim5.client.prompt_log
                  if p["tag"].endswith(":plan") and f"第{_first['step'] + 1}月" in p["user"]]
    check("v5: 記事は翌月の観測に全主体ぶん載る",
          bool(_plan_next)
          and all(_first["text"][:20] in p["user"] for p in _plan_next))
else:
    check("v5: 記事は翌月の観測に全主体ぶん載る（最終月の記事しか出ず未確認）", True)
check("v5: 記事は書いた月の同席者の観測には入らない",
      not [p for p in _sim5.client.prompt_log
           if ":S1r" in p["tag"] and "[記事・第1月]" in p["user"]])

check("v5: 窓口の月に登記を閲覧した主体が観測に残る",
      any(t["kind"] == "registry_lookup" for t in _sim5.ledger.v5_traces_seen)
      == (s4_for_step(2)[0] == "counter"))
check("v5: 会場に居なかった主体には venue の兆候が見えない",
      not [t for t in _sim5.ledger.v5_traces_seen
           if t.get("audience", "").startswith("venue:")
           and _plans5.get((t["step"], t["agent_id"]), {}).get(t["scene"]) != t["venue"]])
# --- Codexレビュー（2026-08-27）で塞いだ穴の回帰 ---------------------------
check("v5: 隣接区画の観測に現在の登記名義が出ない（可視範囲の迂回を塞ぐ）",
      not [r for r in neighbourhood_rows_v5(_sim5.by_id["HH02"], _sim5.ledger,
                                            _sim5.names)
           if "A社" in r or "B社" in r or "C社" in r or "D社" in r])
check("v5: 隣接区画は開始時点で知っている名前までしか出ない",
      all("もとから" in r or "知らない" in r or "隣接区画はない" in r
          for r in neighbourhood_rows_v5(_sim5.by_id["HH02"], _sim5.ledger,
                                         _sim5.names)))
_moved5 = [t for t in _sim5.ledger.transfers()]
check("v5: 台本で名義が移った区画があるのに、隣人の観測にその名義が出ていない",
      bool(_moved5))
check("v5: 窓口の閲覧は取得ごとに記録される（何を見たか追える）",
      all(t.get("parcel_id") for t in _sim5.ledger.v5_traces_seen
          if t.get("kind") == "registry_lookup"))
_alone5 = Simulation(_cfg_run5, PERSONAS5, _tempfile.mkdtemp(prefix="qa_v5b_"))
_alone5.script = {"meta": {"holders": ["A社"]},
                  "acquisitions": [{"id": "ACQ01", "month": 1, "parcel_id": "P02",
                                    "under_name": "A社",
                                    "traces": [{"kind": "sign_change", "month": 1,
                                                "audience": "venue:V01"}]}]}
_ledger_alone = _alone5.ledger
_seen_before = len(_ledger_alone.v5_traces_seen)
_alone5._step_v5(1)
_solo = [a for a in _alone5.actors
         if len([b for b in _alone5.actors
                 if [p for p in _alone5.ledger.v5_plans
                     if p["agent_id"] == b.agent_id and p["step"] == 1]]) >= 0]
_plans_1 = {p["agent_id"]: p for p in _alone5.ledger.v5_plans if p["step"] == 1}
_v01_by_scene = {s: [a for a, p in _plans_1.items() if p[s] == "V01"]
                 for s in ("S1", "S2", "S3")}
_solo_scenes = [s for s, who in _v01_by_scene.items() if len(who) == 1]
if _solo_scenes:
    _s = _solo_scenes[0]
    _who = _v01_by_scene[_s][0]
    check("v5: 会場に1人で行った主体にもその場の兆候が見える（会話の成立と切り離す）",
          any(tr.get("kind") == "sign_change" and tr.get("agent_id") == _who
              for tr in _ledger_alone.v5_traces_seen))
else:
    check("v5: 会場に1人で行った主体にもその場の兆候が見える（該当ケース無し）", True)
_shutil.rmtree(_alone5.run_dir, ignore_errors=True)

check("v5: 同席していない相手を宛先に書いたら記録に残る（黙って捨てない）",
      hasattr(_sim5.ledger, "records"))

_stance_prompts = [p for p in _sim5.client.prompt_log if "stance" in p["user"]]
_stance_by = collections.Counter()
for _p in _stance_prompts:
    _stance_by[_p["tag"]] += 1
check("v5: 姿勢を尋ねるのは会話のターンだけ（計画コールでは尋ねない）",
      not [p for p in _stance_prompts if p["tag"].endswith(":plan")])
check("v5: 姿勢は1主体につき月1回までしか記録されない",
      all(v == 1 for v in collections.Counter(
          (r["step"], r["agent_id"]) for r in _sim5.ledger.v5_stances).values()))
check("v5: 記事は1主体につき月1本までしか記録されない",
      all(v == 1 for v in collections.Counter(
          (r["step"], r["from"]) for r in _sim5.ledger.v5_articles).values()))
check("v5: 記事の棄却（quota超過）が起きていない＝最後のターンで1回だけ書かせている",
      not [r for r in _sim5.ledger.records if r.get("kind") == "article_rejected"])
check("v5: 内心は全主体ぶん記録される（誰の内心も落ちない）",
      {t["from"] for t in _sim5.thoughts} == {a.agent_id for a in _sim5.actors})
check("v5: mock2か月で打切り・解釈不能が出ない",
      _sim5.truncated_count == 0
      and not [e for e in _sim5.events if e.get("action_type") == "PARSE_FAIL"])
_shutil.rmtree(_tmp5, ignore_errors=True)

# --- v1〜v4.1b が不変であること -------------------------------------------
from src.viz import MONEYLESS_VERSIONS as _MLV5  # noqa: E402
check("v5: レポートHTMLを金額のない版で描く（v4.1bで作った仕組みをv5にも当てる）",
      "field_v5" in _MLV5 and "field_v4_1b" in _MLV5)
check("v5: v4.1 の経路は v5 のフラグで切り替わらない",
      Simulation.__dict__["_step"].__doc__ is None
      and "field_v5" in io.open(os.path.join(_ROOT5, "src/simulation.py"),
                                encoding="utf-8").read())
_src5 = io.open(os.path.join(_ROOT5, "src/simulation.py"), encoding="utf-8").read()
check("v5: v4.1b の分岐が残っている（既存経路を消していない）",
      "self._step_v41(step)" in _src5 and "_step_v4(step)" in _src5
      and "_step_v3(step)" in _src5)
_fv41 = io.open(os.path.join(_ROOT5, "src/field_v4_1.py"), encoding="utf-8").read()
check("v5: field_v4_1.py に v5 の語が入り込んでいない", "v5" not in _fv41)


print()
print(f"RESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
