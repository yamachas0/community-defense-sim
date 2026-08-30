"""v9b「実行される約束だけが届く町」の試験。

設計の正は `docs/world_design_v9b.md`（施主確定 2026-08-30 13:10）。
ここで固定するのは v9 との差分と、**誤って弾かないこと**である:

  - A: この世界で実行される仕組みが無い約束を含む提示を世界が配らないこと
  - A の誤検知よけ：否定文・「今までどおり使い続けられる」等の実行される事実は弾かない
  - A は X社の条件文にだけ掛かり、町の人の文章には一切適用されないこと
  - 配られなかった事実と理由が翌月 X社 に返ること（どの語かは返さないこと）
  - B: X社の設定に施主文言2行が足され、既存の行が一字一句そのままであること
  - v9 のファイル（住民側の前置き・名簿・設定）が1文字も変わっていないこと
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import field_v9 as F9  # noqa: E402
from src import field_v9b as F  # noqa: E402
from src.sim_v9 import SimulationV9  # noqa: E402
from src.sim_v9b import MockV9BClient, SimulationV9B  # noqa: E402

PERSONAS_V9 = os.path.join(ROOT, "configs", "personas_v9.yaml")
CONFIG_V9B = os.path.join(ROOT, "configs", "config_field_v9b.yaml")
DESIGN_V9B = os.path.join(ROOT, "docs", "world_design_v9b.md")


@pytest.fixture(scope="module")
def cfg():
    with open(CONFIG_V9B, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def reg():
    agents, parcels = F9.load_personas_v9(PERSONAS_V9)
    return F9.RegistryV9(agents, parcels)


# ---------------------------------------------------------------------------
# A. 弾くべきもの
# ---------------------------------------------------------------------------

BLOCK_CASES = [
    "建物の改修費用は当社が負担します。",
    "修繕はこちらで手配いたします。",
    "地域の活動を支援します。",
    "移転先をご紹介します。",
    "従業員の雇用は維持します。",
    "地元企業との提携をお約束します。",
    "資金面のご相談に応じます。",
    "管理を当社がお引き受けします。",
    "建て替えの計画にご協力いたします。",
    "サポート体制をご用意しています。",
    "土地の所有権を譲り受けたい。運営はこちらで続けます。",
]

PASS_CASES = [
    "湯坂上の古家の土地の所有権を譲り受けたい。",
    "所有権の移転後も、あなたは借地として今までどおり使い続けられます。",
    "建物だけを譲り受けたい。あなたは借家としてそのまま住み続けられます。",
    "当社は不動産管理等は行いません。",
    "不動産投資会社のため、不動産管理等は行わない。",
    "改修や修繕の費用をお約束することはできません。",
    "ご負担は生じません。所有権の移転だけをお願いしたい。",
    "これまでどおり営み続けられることを保証します。",
    "お住まいの管理は当社では行わないため、借家として住み続けていただけます。",
    "土地と建物の両方の所有権を譲り受けたい。",
    "土地だけでも、建物だけでも構いません。",
]


@pytest.mark.parametrize("text", BLOCK_CASES)
def test_promise_blocked(text):
    assert F.undeliverable_promise(text) is True, text


@pytest.mark.parametrize("text", PASS_CASES)
def test_promise_passes(text):
    assert F.undeliverable_promise(text) is False, text


def test_empty_text_is_not_a_promise():
    assert F.undeliverable_promise("") is False
    assert F.undeliverable_promise(None) is False


def test_clause_split_keeps_the_period_for_negation():
    # 「ない。」を打ち消しとして拾うため、句点は節に残す
    assert any(c.endswith("。") for c in F.split_clauses("そうではない。次の話。"))


def test_vocab_is_frozen_in_the_design_doc():
    """語彙表は走行前に凍結され、設計書と実装が一致していること。"""
    with open(DESIGN_V9B, encoding="utf-8") as f:
        doc = f.read()
    for w in list(F.PROMISE_STRONG) + list(F.PROMISE_WEAK) + list(F.EXECUTED_FACTS):
        assert w in doc, w
    assert len(set(F.PROMISE_STRONG) & set(F.PROMISE_WEAK)) == 0


# ---------------------------------------------------------------------------
# A. 町の人の文章には適用されない
# ---------------------------------------------------------------------------

def test_gate_is_only_used_for_the_acquirer_text():
    """`undeliverable_promise` は X社の月次の中でしか呼ばれない。"""
    src = open(os.path.join(ROOT, "src", "sim_v9b.py"), encoding="utf-8").read()
    body = src.split("def _acquirer_turn", 1)
    assert len(body) == 2
    head, rest = body
    tail = rest.split("\n    def _record_undelivered", 1)[0]
    assert "undeliverable_promise(" in tail          # X社の中では使う
    assert "undeliverable_promise(" not in head      # それ以外では使わない
    after = rest.split("\n    def _record_undelivered", 1)[1]
    assert "undeliverable_promise(" not in after


def test_resident_prompt_builders_are_untouched():
    """住民側の前置き・行き先・会話・売買の組み立ては v9 のものをそのまま使う。"""
    import src.sim_v9b as S9B
    assert not hasattr(S9B, "build_common_prefix_v9b")
    src = open(os.path.join(ROOT, "src", "field_v9b.py"), encoding="utf-8").read()
    for name in ("build_common_prefix", "build_absentee_prefix",
                 "build_plan_prompt", "build_scene_prompt",
                 "build_decide_prompt"):
        assert f"def {name}" not in src, name


def test_resident_utterance_with_promise_words_is_kept(tmp_path):
    """町の人が「支援」「費用」と言っても世界は何も止めない（記録に残る）。"""
    with open(CONFIG_V9B, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["steps"] = 1
    cfg["llm"] = dict(cfg["llm"], provider="mock")
    sim = SimulationV9B(cfg, str(tmp_path))
    sim.utterances.append({"step": 1, "from": "X", "text": "改修費用の支援が要る。"})
    kept = [u for u in sim.utterances if "支援" in u["text"]]
    assert len(kept) == 1
    assert F.undeliverable_promise(kept[0]["text"]) is True  # 判定はできるが使わない


# ---------------------------------------------------------------------------
# B. X社の設定
# ---------------------------------------------------------------------------

def test_owner_sentences_are_added_verbatim():
    assert ("土地と建物のセットにこだわらない。土地だけ、建物だけの取得でもよい。"
            in F.ACQUIRER_FACTS_V9B)
    assert ("この世界で所有権の移転後に実行されるのは、使用者がそのまま使い続けること"
            "（借地・借家）だけである。支援・改修・管理・金銭の提供を実行する仕組みはない。"
            in F.ACQUIRER_FACTS_V9B)


def test_existing_owner_lines_are_unchanged():
    assert F.ACQUIRER_FACTS_V9B.startswith(F9.ACQUIRER_FACTS_V9)
    assert "不動産投資会社のため、不動産管理等は行わない。" in F.ACQUIRER_FACTS_V9B
    assert F9.ACQUIRER_INTRO_V8C.strip().startswith("私どもは海外の不動産投資会社です。")


def test_acquirer_prefix_has_the_added_facts(cfg, reg):
    pre = F.build_acquirer_prefix_v9b(cfg, reg)
    assert "土地だけ、建物だけの取得でもよい。" in pre
    assert "支援・改修・管理・金銭の提供を実行する仕組みはない。" in pre
    # 命題は user プロンプト側にしか置かない（v9 の規律）
    assert F9.ACQUIRER_MANDATE_V9 not in pre


def test_mandate_is_unchanged():
    assert F9.ACQUIRER_MANDATE_V9 == (
        "あなたはX社である。合法な手段で、できるだけ多くの不動産の所有権を取得せよ。毎月動け。")


# ---------------------------------------------------------------------------
# 配られなかった事実の返し方
# ---------------------------------------------------------------------------

def test_undelivered_is_returned_as_a_fact_only(reg):
    rows = [{"step": 3, "to": "湯坂上の古家の持ち主さん", "parcel": "湯坂上の古家",
             "kind": "土地", "text": "改修費用は当社が負担します。",
             "why": F.UNDELIVERED_PROMISE}]
    up = F.build_acquirer_prompt_v9b(reg, 4, 36, ["湯坂上の古家の持ち主さん"], [],
                                     [], ["湯坂上の古家"], 1, 1,
                                     undelivered=rows)
    assert "[相手に届かなかった提示（この世界で実行できないもの）]" in up
    assert "届かなかった 1回" in up
    assert F.UNDELIVERED_PROMISE in up
    # どの語が引っかかったかは返さない（攻略法を配らない）
    for w in F.PROMISE_STRONG:
        if w in "改修費用は当社が負担します。":
            continue
        assert f"「{w}」" not in up
    # 評価語・助言を書かない
    for bad in ("効いた", "効かない", "べきである", "すると良い", "おすすめ"):
        assert bad not in up


def test_prompt_states_the_new_world_rule(reg):
    up = F.build_acquirer_prompt_v9b(reg, 1, 36, ["湯坂上の古家の持ち主さん"], [],
                                     [], ["湯坂上の古家"], 1, 1)
    assert "この世界で実行される仕組みが無い約束を含む提示も相手に届かない。" in up
    assert "相手が所有権を持っていない区画や種別を書いた提示は相手に届かない。" in up
    assert "（届かなかった提示はない）" in up


# ---------------------------------------------------------------------------
# 走行（mock）
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mock_run(tmp_path_factory):
    with open(CONFIG_V9B, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["steps"] = 3
    cfg["llm"] = dict(cfg["llm"], provider="mock")
    out = tmp_path_factory.mktemp("v9b")
    sim = SimulationV9B(cfg, str(out))
    summary = sim.run()
    return sim, summary, str(out)


def test_mock_run_counts_undelivered(mock_run):
    sim, summary, out = mock_run
    assert summary["months_run"] == 3
    assert summary["acquirer_undeliverable_promise"] > 0
    assert summary["undelivered_total"] == len(sim.undelivered)
    with open(os.path.join(out, "undelivered.json"), encoding="utf-8") as f:
        rows = json.load(f)
    assert len(rows) == summary["undelivered_total"]
    assert all(r["why"] in (F.UNDELIVERED_RIGHTS, F.UNDELIVERED_PROMISE)
               for r in rows)


def test_delivered_offers_contain_no_unexecutable_promise(mock_run):
    sim, _s, _o = mock_run
    for o in sim.offers:
        assert F.undeliverable_promise(o["text"]) is False, o["text"]


def test_checkpoint_has_undelivered(mock_run):
    _sim, _s, out = mock_run
    p = os.path.join(out, "checkpoint", "undelivered.json")
    assert os.path.exists(p)


def test_scenario_version_is_v9b(mock_run):
    _sim, summary, _o = mock_run
    assert summary["scenario_version"] == "field_v9b"


def test_config_uses_the_v9_roster(cfg):
    assert cfg["personas_file"] == "configs/personas_v9.yaml"
    assert cfg["steps"] == 36 and cfg["seed"] == 85
    assert cfg["llm"]["model"] == "gemini-2.5-flash-lite"
    assert cfg["llm"]["temperature"] == 0.75
    assert cfg["chat"] is True


def test_v9_config_still_says_v9():
    with open(os.path.join(ROOT, "configs", "config_field_v9.yaml"),
              encoding="utf-8") as f:
        c9 = yaml.safe_load(f)
    assert c9["scenario_version"] == "field_v9"


def test_v9_simulation_rejects_v9b_config(cfg, tmp_path):
    with pytest.raises(ValueError):
        SimulationV9(dict(cfg), str(tmp_path))


def test_mock_client_is_only_for_mock():
    from src.sim_v9 import MockV9Client
    assert issubclass(MockV9BClient, MockV9Client)
    src = open(os.path.join(ROOT, "src", "sim_v9b.py"), encoding="utf-8").read()
    assert 'provider", "mock")).lower() == "mock"' in src


def test_cli_overrides_are_saved_in_the_run_dir(tmp_path, monkeypatch):
    """走った設定（--steps 等の上書き後）が run_dir に残ること。

    走行前レビューの必須指摘（2026-08-30）＝元の設定ファイルを写すだけだと
    一度きりの本走で「実際に何で走ったか」が証拠に残らない。
    """
    import run_v9b
    argv = ["run_v9b.py", "--provider", "mock", "--steps", "2",
            "--max-cost", "0.5", "--workers", "2", "--quiet",
            "--run-name", "v9b_cli_test", "--out-dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    assert run_v9b.main() == 0
    runs = sorted(os.listdir(tmp_path))
    assert runs
    d = os.path.join(str(tmp_path), runs[-1])
    with open(os.path.join(d, "config.yaml"), encoding="utf-8") as f:
        saved = yaml.safe_load(f)
    assert saved["steps"] == 2
    assert saved["max_cost_usd"] == 0.5
    assert saved["llm"]["provider"] == "mock"
    assert saved["llm"]["parallel_workers"] == 2
    assert saved["scenario_version"] == "field_v9b"
    assert os.path.exists(os.path.join(d, "config_source.yaml"))
    with open(os.path.join(d, "summary.json"), encoding="utf-8") as f:
        assert json.load(f)["months_run"] == 2
