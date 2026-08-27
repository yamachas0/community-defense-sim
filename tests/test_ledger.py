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
                          build_system_prompt_v3, build_user_prompt_v3, control_share,
                          ensure_v3_state, list_for_lease, make_lease_offer,
                          own_result_row, own_results_text,
                          resolve_lease_offer, seed_acquirer_intelligence_v3)  # noqa: E402

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
          "operations", "strategy", "next_milestone", "expected_goal_effect"))
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

print()
print(f"RESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
