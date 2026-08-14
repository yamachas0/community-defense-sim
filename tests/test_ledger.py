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

print()
print(f"RESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
