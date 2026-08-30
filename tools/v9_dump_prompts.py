#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v9 の全プロンプト原文を1枚に書き出す（決定論・LLM を呼ばない）。

    python tools/v9_dump_prompts.py > docs/v9_prompts.txt

走行前レビュー（Codex・施主確認）に出すための素材。実装を直したら必ず作り直す。
"""
from __future__ import annotations

import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import field_v9 as F  # noqa: E402

BAR = "=" * 78


def head(t: str) -> str:
    return f"{BAR}\n### {t}\n{BAR}"


def main() -> int:
    with open(os.path.join(ROOT, "configs", "config_field_v9.yaml"),
              encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    agents, parcels = F.load_personas_v9(
        os.path.join(ROOT, "configs", "personas_v9.yaml"))
    reg = F.RegistryV9(agents, parcels)
    n = len(reg.parcel_names)
    steps = int(cfg["steps"])
    venue_labels = [str(v["label"]) for v in cfg["social"]["venues"]]
    out = []

    out.append(head("住民の共通前置き（system・全コール共通）"))
    out.append(F.build_common_prefix_v9(cfg, agents, n))
    out.append(head("町にいない所有者の前置き（system）"))
    out.append(F.build_absentee_prefix_v9(cfg, n))
    out.append(head("X社の前置き（system）"))
    out.append(F.build_acquirer_prefix_v9(cfg, reg))

    a = reg.by_id["A"]                      # 町にいる所有者（2区画）
    nb = ["駅前通りの持ち主さん"]
    offer = {"parcel": "湯坂上の古家", "kind": F.KIND_BOTH,
             "delivered": (F.ACQUIRER_INTRO_V8C
                           + "ご都合に合わせて手続きを進めます。")}
    notices = ["（記録）先月末、湯坂上の空き店舗の建物の所有権が "
               "湯坂上のご夫婦 から X社 に移った。"]
    heard = [{"from": "駅前通りの持ち主さん", "venue_label": "公園",
              "text": "最近どうですか。"}]

    out.append(head("月初（行き先）user プロンプト＋所有者変更の通知つき"))
    out.append(F.build_plan_prompt_v9(a, reg, 7, steps, venue_labels, "今の内心。",
                                      offer, notices, nb))
    out.append(head("場面（会話）user プロンプト"))
    out.append(F.build_scene_prompt_v9(a, reg, 7, steps, "今の内心。", "公園",
                                       ["湯坂上のご夫婦", "駅前通りの持ち主さん"]))

    opts_a = [(p, reg.listing_options("A", p))
              for p in reg.parcels_owned("A")]
    out.append(head("月末の問い（町にいる人・提示あり）"))
    out.append(F.build_decide_prompt_v9(a, reg, 7, steps, "今の内心。", offer,
                                        heard, opts_a, [F.SELL_YES, F.SELL_NO],
                                        None, nb, True))

    cid = next(x["id"] for x in agents if not x.get("resident", True))
    c = reg.by_id[cid]
    opts_c = [(p, reg.listing_options(cid, p))
              for p in reg.parcels_owned(cid)]
    o_parcel = opts_c[0][0]
    kind = (F.KIND_LAND if reg.can_offer(cid, o_parcel, F.KIND_LAND)
            else F.KIND_BUILDING)
    offer_c = {"parcel": o_parcel, "kind": kind,
               "delivered": (F.ACQUIRER_INTRO_V8C
                             + "所有権の移転についてご相談させてください。")}
    out.append(head("月末の問い（町にいない所有者・提示あり）"))
    out.append(F.build_decide_prompt_v9(c, reg, 7, steps, "今の内心。", offer_c,
                                        [], opts_c, [F.SELL_NO, F.SELL_YES],
                                        None, None, False))

    names = [x["name"] for x in agents if x.get("sellable", True) and reg.parcels_owned(x["id"])][:10]
    parcels_for = []
    for x in agents:
        if x["name"] in names:
            parcels_for += list(reg.parcels_owned(x["id"]))
    out.append(head("X社の user プロンプト（1塊目）"))
    out.append(F.build_acquirer_prompt_v9(
        reg, 7, steps, names, [], ["湯坂上の古家（土地）"],
        sorted(set(parcels_for)), 1, 5, bool(cfg.get("acquirer_reason", True))))

    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
