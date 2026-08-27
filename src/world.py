"""World ledger (登記簿) — the ONLY deterministic layer of this simulation.

設計原則 / Design invariant
--------------------------
    「誰かの行動を決めるコードはゼロ。行動の結果を記帳するコードだけ書く。」
    (No code decides anyone's behaviour. Code only books the consequences.)

このモジュールがやってよいこと:
  - 区画・所有者・価格・賃料・現金・予算の記帳 (bookkeeping)
  - 記帳結果から導出できる集計 (シェア・HHI 等) の計算
  - 「誰に何が見えるか」という世界の構造 (可視性チャネル) の定義

このモジュールがやってはいけないこと:
  - 地価を式で動かす / 売る確率を持つ / 閾値で誰かを行動させる
  - 拡散率・同調率などの行動パラメータを持つ
  価格はすべてエージェント (LLM) が提示・合意した実額のみを記帳する。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------
# Parcel
# --------------------------------------------------------------------------

@dataclass
class Parcel:
    """1区画。地番・用途・所有者・賃料を持つ帳簿上の実体。"""

    pid: str
    x: int
    y: int
    block: str
    use: str                     # "residential" | "shop" | "vacant" | "public"
    owner_id: str
    assessed_value: int          # 初期評価額 (万円)。世界の初期条件であり、以後コードは動かさない
    area_sqm: int = 100           # 土地面積 (㎡)。世界の初期条件
    unit_price: float = 24.0      # 地区・用途を反映した基準地価 (万円/㎡)
    rent: int = 0                # 月額賃料 (万円)。shop のみ。所有者(LLM)が改定する
    tenant_id: Optional[str] = None
    registered_name: str = ""    # 登記上の名義。買い手がalias を使えば実体(owner_id)と食い違う
    listed_price: Optional[int] = None   # 売り出し中の希望価格 (万円)
    listed_at_step: Optional[int] = None

    def label(self) -> str:
        use_ja = {"residential": "住宅", "shop": "商店", "vacant": "空地", "public": "公有地"}
        return f"{self.pid}({use_ja.get(self.use, self.use)}/{self.block})"


@dataclass
class Offer:
    """買付の申し込み。step t に出され、step t+1 に相手の観測へ載る。"""

    offer_id: str
    step: int
    parcel_id: str
    from_id: str          # 実際の申込主体 (agent_id)
    under_name: str       # 名義 (買い手が alias を使えば別名になる)
    price: int
    status: str = "open"  # open | countered | accepted | rejected | withdrawn | void
    note: str = ""
    counter_price: Optional[int] = None   # 売主からの逆提示額。買い手にだけ見える
    counter_step: Optional[int] = None


# --------------------------------------------------------------------------
# Ledger
# --------------------------------------------------------------------------

class Ledger:
    """世界の帳簿。全ての状態変化はここを通り、records に一行ずつ残る。"""

    def __init__(self, parcels: List[Parcel], cash: Dict[str, int], seed_note: str = ""):
        self.parcels: Dict[str, Parcel] = {p.pid: p for p in parcels}
        self.cash: Dict[str, int] = dict(cash)          # agent_id -> 万円
        self.records: List[Dict[str, Any]] = []
        self.offers: Dict[str, Offer] = {}
        self._offer_seq = 0
        self.ordinances: List[Dict[str, Any]] = []      # 自治体が発動した規制
        self.publications: List[Dict[str, Any]] = []    # メディアの報道
        self.moved_out: Dict[str, int] = {}             # agent_id -> step (転出)
        self.closed_businesses: Dict[str, int] = {}     # agent_id -> step (廃業/撤退)
        self.seed_note = seed_note
        self.on_demand_financing: set[str] = set()
        self.financing_raised: Dict[str, int] = {}

    # -- bookkeeping primitives -------------------------------------------

    def _rec(self, step: int, kind: str, **kw) -> Dict[str, Any]:
        row = {"step": step, "kind": kind, **kw}
        self.records.append(row)
        return row

    def enable_on_demand_financing(self, agent_ids: List[str]) -> None:
        self.on_demand_financing.update(agent_ids)
        for agent_id in agent_ids:
            self.financing_raised.setdefault(agent_id, 0)

    def fund_payment(self, step: int, agent_id: str, amount: int,
                     purpose: str, reference: str = "") -> bool:
        """随時調達型なら決済時の不足額を調達し、台帳に残す。"""
        available = self.cash.get(agent_id, 0)
        if available >= amount:
            return True
        if agent_id not in self.on_demand_financing:
            return False
        raised = amount - available
        self.cash[agent_id] = available + raised
        self.financing_raised[agent_id] = self.financing_raised.get(agent_id, 0) + raised
        self._rec(step, "financing_raised", agent_id=agent_id, amount=raised,
                  purpose=purpose, reference=reference)
        return True

    @staticmethod
    def _valid_money(v: Any, allow_zero: bool = False) -> bool:
        """金額として物理的に成立するか。

        負の価格・負の賃料は「安く売る」ではなく帳簿を壊す入力なので成立しない。
        これは行動の良し悪しの判断ではなく、取引が物理的に成立するかの判定である
        （自分の持ち物でない区画を売れないのと同じ扱い）。
        """
        try:
            iv = int(v)
        except (TypeError, ValueError):
            return False
        if iv < 0:
            return False
        if iv == 0 and not allow_zero:
            return False
        return iv <= 100_000_000

    def record_listing(self, step: int, parcel_id: str, by: str, price: int) -> Dict[str, Any]:
        p = self.parcels.get(parcel_id)
        if p is None or p.owner_id != by:
            return self._rec(step, "listing_rejected", parcel_id=parcel_id, by=by,
                             reason="not_owner")
        if not self._valid_money(price):
            return self._rec(step, "listing_rejected", parcel_id=parcel_id, by=by,
                             reason="invalid_amount", given=price)
        p.listed_price = int(price)
        p.listed_at_step = step
        return self._rec(step, "listing", parcel_id=parcel_id, by=by, price=int(price))

    def record_unlist(self, step: int, parcel_id: str, by: str) -> Dict[str, Any]:
        p = self.parcels.get(parcel_id)
        if p is None or p.owner_id != by:
            return self._rec(step, "unlist_rejected", parcel_id=parcel_id, by=by,
                             reason="not_owner")
        if p.listed_price is None:
            return self._rec(step, "unlist_rejected", parcel_id=parcel_id, by=by,
                             reason="not_listed")
        p.listed_price = None
        p.listed_at_step = None
        return self._rec(step, "unlist", parcel_id=parcel_id, by=by)

    def record_offer(self, step: int, parcel_id: str, from_id: str, price: int,
                     under_name: str = "", note: str = "") -> Dict[str, Any]:
        p = self.parcels.get(parcel_id)
        if p is None:
            return self._rec(step, "offer_rejected", parcel_id=parcel_id, from_id=from_id,
                             reason="no_such_parcel")
        if p.owner_id == from_id:
            return self._rec(step, "offer_rejected", parcel_id=parcel_id, from_id=from_id,
                             reason="already_owner")
        if p.use == "public":
            return self._rec(step, "offer_rejected", parcel_id=parcel_id, from_id=from_id,
                             reason="public_land_not_for_sale")
        if not self._valid_money(price):
            return self._rec(step, "offer_rejected", parcel_id=parcel_id, from_id=from_id,
                             reason="invalid_amount", given=price)
        if (from_id not in self.on_demand_financing
                and self.cash.get(from_id, 0) < int(price)):
            return self._rec(step, "offer_rejected", parcel_id=parcel_id, from_id=from_id,
                             reason="insufficient_funds", given=price,
                             budget=self.cash.get(from_id, 0))
        self._offer_seq += 1
        oid = f"O{self._offer_seq:04d}"
        offer = Offer(offer_id=oid, step=step, parcel_id=parcel_id, from_id=from_id,
                      under_name=under_name or from_id, price=int(price), note=note)
        self.offers[oid] = offer
        return self._rec(step, "offer", offer_id=oid, parcel_id=parcel_id, from_id=from_id,
                         under_name=offer.under_name, price=offer.price, to=p.owner_id, note=note)

    def record_counter(self, step: int, offer_id: str, by: str, price: int) -> Dict[str, Any]:
        offer = self.offers.get(offer_id)
        if offer is None or offer.status != "open":
            return self._rec(step, "counter_rejected", offer_id=offer_id, by=by,
                             reason="offer_not_open")
        p = self.parcels[offer.parcel_id]
        if p.owner_id != by:
            return self._rec(step, "counter_rejected", offer_id=offer_id, by=by,
                             reason="not_owner")
        if not self._valid_money(price):
            return self._rec(step, "counter_rejected", offer_id=offer_id, by=by,
                             reason="invalid_amount", given=price)
        # 私的な逆提示。公開の売り出しには変換しない（買い手にだけ届く）。
        offer.status = "countered"
        offer.counter_price = int(price)
        offer.counter_step = step
        return self._rec(step, "counter", offer_id=offer_id, parcel_id=offer.parcel_id,
                         by=by, to=offer.from_id, price=int(price))

    def record_accept(self, step: int, offer_id: str, by: str) -> Dict[str, Any]:
        offer = self.offers.get(offer_id)
        if offer is None:
            return self._rec(step, "accept_rejected", offer_id=offer_id, by=by,
                             reason="no_such_offer")
        if offer.status != "open":
            return self._rec(step, "accept_rejected", offer_id=offer_id, by=by,
                             reason=f"offer_{offer.status}")
        p = self.parcels[offer.parcel_id]
        if p.owner_id != by:
            return self._rec(step, "accept_rejected", offer_id=offer_id, by=by,
                             reason="not_owner")
        if not self.fund_payment(step, offer.from_id, offer.price,
                                 "property_purchase", offer.offer_id):
            offer.status = "void"
            return self._rec(step, "accept_rejected", offer_id=offer_id, by=by,
                             reason="buyer_insufficient_funds")
        buyer_cash = self.cash.get(offer.from_id, 0)
        # 所有権移転を記帳
        seller = p.owner_id
        p.owner_id = offer.from_id
        p.registered_name = offer.under_name
        p.listed_price = None
        p.listed_at_step = None
        self.cash[offer.from_id] = buyer_cash - offer.price
        self.cash[seller] = self.cash.get(seller, 0) + offer.price
        offer.status = "accepted"
        # 同一区画に残っている他の open offer は成立不能になるので void
        for o in self.offers.values():
            if o.parcel_id == offer.parcel_id and o.status == "open":
                o.status = "void"
        return self._rec(step, "transfer", offer_id=offer_id, parcel_id=offer.parcel_id,
                         seller=seller, buyer=offer.from_id, under_name=offer.under_name,
                         price=offer.price)

    def record_reject(self, step: int, offer_id: str, by: str) -> Dict[str, Any]:
        offer = self.offers.get(offer_id)
        if offer is None or offer.status != "open":
            return self._rec(step, "reject_rejected", offer_id=offer_id, by=by,
                             reason="offer_not_open")
        p = self.parcels[offer.parcel_id]
        if p.owner_id != by:
            return self._rec(step, "reject_rejected", offer_id=offer_id, by=by,
                             reason="not_owner")
        offer.status = "rejected"
        return self._rec(step, "reject", offer_id=offer_id, parcel_id=offer.parcel_id, by=by)

    def record_withdraw(self, step: int, offer_id: str, by: str) -> Dict[str, Any]:
        offer = self.offers.get(offer_id)
        if offer is None or offer.status != "open" or offer.from_id != by:
            return self._rec(step, "withdraw_rejected", offer_id=offer_id, by=by,
                             reason="offer_not_open_or_not_yours")
        offer.status = "withdrawn"
        return self._rec(step, "withdraw", offer_id=offer_id, parcel_id=offer.parcel_id, by=by)

    def record_rent_change(self, step: int, parcel_id: str, by: str, rent: int) -> Dict[str, Any]:
        p = self.parcels.get(parcel_id)
        if p is None or p.owner_id != by:
            return self._rec(step, "rent_rejected", parcel_id=parcel_id, by=by, reason="not_owner")
        if p.use != "shop":
            return self._rec(step, "rent_rejected", parcel_id=parcel_id, by=by,
                             reason="not_a_shop", use=p.use)
        if not self._valid_money(rent):
            return self._rec(step, "rent_rejected", parcel_id=parcel_id, by=by,
                             reason="invalid_amount", given=rent)
        old = p.rent
        p.rent = int(rent)
        return self._rec(step, "rent_change", parcel_id=parcel_id, by=by, old=old, new=p.rent,
                         tenant=p.tenant_id)

    def record_tenancy_end(self, step: int, agent_id: str, reason: str) -> List[Dict[str, Any]]:
        out = []
        for p in self.parcels.values():
            if p.tenant_id == agent_id:
                p.tenant_id = None
                out.append(self._rec(step, "tenancy_end", parcel_id=p.pid, tenant=agent_id,
                                     owner=p.owner_id, reason=reason))
        return out

    def record_relocate(self, step: int, agent_id: str, parcel_id: str) -> Dict[str, Any]:
        p = self.parcels.get(parcel_id)
        if agent_id in self.closed_businesses:
            return self._rec(step, "relocate_rejected", agent_id=agent_id, parcel_id=parcel_id,
                             reason="already_closed")
        if p is None or p.use != "shop":
            return self._rec(step, "relocate_rejected", agent_id=agent_id, parcel_id=parcel_id,
                             reason="not_a_shop")
        if p.tenant_id is not None:
            return self._rec(step, "relocate_rejected", agent_id=agent_id, parcel_id=parcel_id,
                             reason="occupied")
        self.record_tenancy_end(step, agent_id, reason="relocate")
        p.tenant_id = agent_id
        return self._rec(step, "relocate", agent_id=agent_id, parcel_id=parcel_id,
                         owner=p.owner_id, rent=p.rent)

    def record_redevelop(self, step: int, parcel_id: str, by: str, new_rent: int) -> Dict[str, Any]:
        p = self.parcels.get(parcel_id)
        if p is None or p.owner_id != by:
            return self._rec(step, "redevelop_rejected", parcel_id=parcel_id, by=by,
                             reason="not_owner")
        if not self._valid_money(new_rent):
            return self._rec(step, "redevelop_rejected", parcel_id=parcel_id, by=by,
                             reason="invalid_amount", given=new_rent)
        old_use, old_tenant = p.use, p.tenant_id
        if old_tenant:
            self.record_tenancy_end(step, old_tenant, reason="redevelopment")
        p.use = "shop"
        p.rent = int(new_rent)
        return self._rec(step, "redevelop", parcel_id=parcel_id, by=by, old_use=old_use,
                         new_use="shop", rent=p.rent, evicted=old_tenant)

    def record_close(self, step: int, agent_id: str, note: str = "") -> Dict[str, Any]:
        if agent_id in self.closed_businesses:
            return self._rec(step, "close_rejected", agent_id=agent_id, reason="already_closed",
                             closed_at=self.closed_businesses[agent_id])
        self.closed_businesses.setdefault(agent_id, step)
        self.record_tenancy_end(step, agent_id, reason="closed")
        return self._rec(step, "business_closed", agent_id=agent_id, note=note)

    def record_move_out(self, step: int, agent_id: str, note: str = "") -> Dict[str, Any]:
        if agent_id in self.moved_out:
            return self._rec(step, "move_out_rejected", agent_id=agent_id,
                             reason="already_moved_out", moved_at=self.moved_out[agent_id])
        self.moved_out.setdefault(agent_id, step)
        self.record_tenancy_end(step, agent_id, reason="moved_out")
        return self._rec(step, "move_out", agent_id=agent_id, note=note)

    def record_ordinance(self, step: int, by: str, title: str, body: str) -> Dict[str, Any]:
        self.ordinances.append({"step": step, "by": by, "title": title, "body": body})
        return self._rec(step, "ordinance", by=by, title=title, body=body)

    def record_study(self, step: int, by: str, note: str) -> Dict[str, Any]:
        return self._rec(step, "ordinance_study", by=by, note=note)

    def record_publication(self, step: int, by: str, headline: str, body: str,
                           about_acquisition: bool) -> Dict[str, Any]:
        self.publications.append({"step": step, "by": by, "headline": headline,
                                  "about_acquisition": about_acquisition})
        return self._rec(step, "publication", by=by, headline=headline, body=body,
                         about_acquisition=about_acquisition)

    def record_note(self, step: int, by: str, kind: str, note: str) -> Dict[str, Any]:
        """帳簿を動かさない行為 (静観・調査・発話のみ 等) の記帳。"""
        return self._rec(step, kind, by=by, note=note)

    def settle_month(self, step: int, business_margins: Dict[str, int]) -> Dict[str, Any]:
        """月次の清算。賃料が実際に借主から家主へ動く。

        これは行動ではなく契約の履行＝会計処理。エージェントは誰も「払うかどうか」を
        選ばない（契約しているから払う）。
        粗利 (business_margins) は各事業者の初期条件として与えられる世界の数値であり、
        評価額と同じ扱い。**資金が尽きても自動で閉店させない** — 続けるか畳むかは
        その事業者 (LLM) が決める。ここで閉店させたらルールベースになる。
        """
        occupying = {p.tenant_id for p in self.parcels.values()
                     if p.use == "shop" and p.tenant_id}
        occupying |= {p.owner_id for p in self.parcels.values()
                      if p.use == "shop" and p.owner_id in business_margins}
        moved = []
        for biz_id, margin in business_margins.items():
            if biz_id in self.closed_businesses:
                continue
            if biz_id not in occupying:
                # 店を構えていない月は営業粗利が立たない
                moved.append({"agent": biz_id, "margin": 0, "reason": "no_shop"})
                continue
            self.cash[biz_id] = self.cash.get(biz_id, 0) + int(margin)
            moved.append({"agent": biz_id, "margin": int(margin)})
        rents = []
        for p in self.parcels.values():
            if p.use != "shop" or not p.tenant_id or p.rent <= 0:
                continue
            if p.tenant_id in self.closed_businesses:
                continue
            self.cash[p.tenant_id] = self.cash.get(p.tenant_id, 0) - p.rent
            self.cash[p.owner_id] = self.cash.get(p.owner_id, 0) + p.rent
            rents.append({"parcel": p.pid, "tenant": p.tenant_id,
                          "owner": p.owner_id, "rent": p.rent})
        return self._rec(step, "settlement", margins=moved, rents=rents)

    def month_pl(self, agent_id: str, business_margins: Dict[str, int]) -> Dict[str, int]:
        """その事業者の今月の収支内訳（観測に載せる用）。"""
        margin = int(business_margins.get(agent_id, 0))
        rent = sum(p.rent for p in self.parcels.values()
                   if p.tenant_id == agent_id and p.use == "shop")
        return {"margin": margin, "rent": rent, "net": margin - rent}

    # -- derived views (集計。記帳結果からの導出のみ) -----------------------

    def open_offers_for_owner(self, owner_id: str) -> List[Offer]:
        return [o for o in self.offers.values()
                if o.status == "open" and self.parcels[o.parcel_id].owner_id == owner_id]

    def open_offers_from(self, from_id: str) -> List[Offer]:
        return [o for o in self.offers.values() if o.status == "open" and o.from_id == from_id]

    def counters_for(self, from_id: str) -> List[Offer]:
        """自分が出した買付に返ってきた逆提示（買い手にだけ見える私的な提示）。"""
        return [o for o in self.offers.values()
                if o.status == "countered" and o.from_id == from_id]

    def owned_by(self, owner_id: str) -> List[Parcel]:
        return [p for p in self.parcels.values() if p.owner_id == owner_id]

    def listings(self) -> List[Parcel]:
        return [p for p in self.parcels.values() if p.listed_price is not None]

    def transfers(self) -> List[Dict[str, Any]]:
        return [r for r in self.records if r["kind"] == "transfer"]

    def recent_trades(self, step: int, window: int = 6) -> List[Dict[str, Any]]:
        return [r for r in self.transfers() if step - window <= r["step"] <= step]

    def ownership_share(self, owner_ids: List[str]) -> float:
        """対象所有者が持つ区画の割合 (公有地を除く)。"""
        tradable = [p for p in self.parcels.values() if p.use != "public"]
        if not tradable:
            return 0.0
        n = sum(1 for p in tradable if p.owner_id in owner_ids)
        return n / len(tradable)

    def hhi(self) -> float:
        """所有集中度 (Herfindahl-Hirschman Index, 0-1)。1 に近いほど1者に集中。"""
        tradable = [p for p in self.parcels.values() if p.use != "public"]
        if not tradable:
            return 0.0
        counts: Dict[str, int] = {}
        for p in tradable:
            counts[p.owner_id] = counts.get(p.owner_id, 0) + 1
        total = len(tradable)
        return sum((c / total) ** 2 for c in counts.values())

    def block_share(self, owner_ids: List[str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        blocks: Dict[str, List[Parcel]] = {}
        for p in self.parcels.values():
            if p.use == "public":
                continue
            blocks.setdefault(p.block, []).append(p)
        for b, ps in blocks.items():
            out[b] = sum(1 for p in ps if p.owner_id in owner_ids) / len(ps)
        return out

    def snapshot(self) -> Dict[str, Any]:
        return {
            "parcels": {pid: asdict(p) for pid, p in self.parcels.items()},
            "cash": dict(self.cash),
            "financing_raised": dict(self.financing_raised),
        }

    def owner_map(self) -> Dict[str, str]:
        return {pid: p.owner_id for pid, p in self.parcels.items()}

    def dump_records(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# World construction
# --------------------------------------------------------------------------

def build_town(world_cfg: Dict[str, Any], household_ids: List[str],
               business_ids: List[str], municipality_id: str) -> List[Parcel]:
    """config の街定義から区画リストを組む。

    区画の初期割当は決定論だが、これは「世界の初期条件」であって行動ルールではない。
    """
    cols = int(world_cfg["grid"]["cols"])
    rows = int(world_cfg["grid"]["rows"])
    block_names = list(world_cfg["block_names"])
    shop_coords = {tuple(c) for c in world_cfg.get("shop_coords", [])}
    vacant_coords = {tuple(c) for c in world_cfg.get("vacant_coords", [])}
    public_coords = {tuple(c) for c in world_cfg.get("public_coords", [])}
    base_value = int(world_cfg.get("base_assessed_value", 2400))
    base_unit_price = float(world_cfg.get("base_unit_price", 24.0))
    area_pattern = [int(v) for v in world_cfg.get(
        "area_pattern_sqm", [70, 90, 110, 140, 180, 240, 320, 450]
    )]
    block_premium = world_cfg.get("block_premium", {})

    parcels: List[Parcel] = []
    idx = 0
    hh_cursor = 0
    for y in range(rows):
        for x in range(cols):
            idx += 1
            pid = f"P{idx:02d}"
            bi = (y // (rows // 2)) * 2 + (x // (cols // 2))
            block = block_names[min(bi, len(block_names) - 1)]
            if (x, y) in public_coords:
                use, owner = "public", municipality_id
            elif (x, y) in shop_coords:
                use, owner = "shop", household_ids[hh_cursor % len(household_ids)]
                hh_cursor += 1
            elif (x, y) in vacant_coords:
                use, owner = "vacant", household_ids[hh_cursor % len(household_ids)]
                hh_cursor += 1
            else:
                use, owner = "residential", household_ids[hh_cursor % len(household_ids)]
                hh_cursor += 1
            area = area_pattern[(idx - 1) % len(area_pattern)]
            unit_price = base_unit_price * float(block_premium.get(block, 1.0))
            if use == "shop":
                unit_price *= 1.25
            elif use == "vacant":
                unit_price *= 0.7
            value = int(round(area * unit_price))
            parcels.append(Parcel(pid=pid, x=x, y=y, block=block, use=use, owner_id=owner,
                                  assessed_value=value, area_sqm=area,
                                  unit_price=round(unit_price, 2)))
    return parcels


def assign_tenancies(parcels: List[Parcel], business_ids: List[str],
                     initial_rent: int) -> None:
    """商店区画に地元事業者をテナントとして入居させる (初期条件)。"""
    shops = [p for p in parcels if p.use == "shop"]
    for i, biz in enumerate(business_ids):
        if i < len(shops):
            shops[i].tenant_id = biz
            shops[i].rent = initial_rent


def neighbors(parcels: Dict[str, Parcel], pid: str) -> List[str]:
    """4近傍の区画 ID。可視性チャネル (誰に何が見えるか) の定義に使う。"""
    p = parcels[pid]
    out = []
    for q in parcels.values():
        if q.pid == pid:
            continue
        if abs(q.x - p.x) + abs(q.y - p.y) == 1:
            out.append(q.pid)
    return out
