"""v8d の月ループ（v8c からの差分だけ）。

設計の正は `docs/world_design_v8d.md`。
**既存の `src/simulation.py`・`src/sim_v8.py`・`src/sim_v8b.py`・`src/sim_v8c.py`・
`src/llm_client_factory.py` には一切触らない**
（`SimulationV8C` と `GeminiClient` を **継承** して差分だけ足す）。

世界の差分は3つだけ（`src/field_v8d.py` の docstring 参照）。
このファイルが足すのはそれに加えて **機械の都合だけ** である:
  - LLM 呼び出しの request timeout（60秒）と、タイムアウト時の1回だけの再試行
  - **月ごとのチェックポイント書き出し**（途中で落ちてもそこまでが残る）
  - 月の進みを stdout に1行出す（無人ランの見張り用）

やっていないこと（意図的に）:
  - 誰かの行動を条件分岐で決める
  - 答えが無かったときに既定値を書き込むこと
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional

from .field_v8d import (HOME, MAX_REASON_CHARS, NO_ANSWER,  # noqa: F401
                        acquirer_schema_v8d,
                        build_acquirer_prefix_v8d, build_acquirer_prompt_v8d,
                        build_common_prefix_v8d, build_decide_prompt_v8d,
                        build_plan_prompt_v8d, decide_schema_v8d,
                        plan_schema_v8d)
from .llm_client_factory import MAX_RATE_LIMIT_RETRIES, GeminiClient, _retry_backoff
from .sim_v8b import CostLimitReached
from .sim_v8c import (RESULT_NO_ANSWER, RESULT_NOT_SOLD, RESULT_SOLD,  # noqa: F401
                      MockV8CClient, SimulationV8C)

logger = logging.getLogger(__name__)

DEFAULT_REQUEST_TIMEOUT_SEC = 60.0
TIMEOUT_RETRIES = 1          # タイムアウトしたら1回だけやり直す（施主指示）


def _is_timeout(err: Exception) -> bool:
    msg = str(err).lower()
    name = type(err).__name__
    return ("timeout" in msg or "timed out" in msg
            or "deadline" in msg or "ReadTimeout" in name
            or "Timeout" in name)


def _is_transient(err: Exception) -> bool:
    msg = str(err).lower()
    name = type(err).__name__
    return ("429" in msg or "rate" in msg or "quota" in msg
            or "resource_exhausted" in msg or "503" in msg
            or "unavailable" in msg or "disconnect" in msg
            or "connection" in msg or "remoteprotocolerror" in msg
            or name in ("ResourceExhausted", "RemoteProtocolError",
                        "ConnectError", "ConnectionError"))


class TimeoutGeminiClient(GeminiClient):
    """`GeminiClient` に **request timeout** を足しただけの版（v8d 専用）。

    v8c の本走1回目が第11月で墜落した（HANDOFF 第21便）。落ちた原因はメモリだが、
    「1本のコールが返らないまま走行が固まる」経路も同じくらい怖いので、
    HTTP の層で 60 秒の締切を入れる。

    **見張り用のスレッドは足さない**（v8c の墜落は `ThreadPoolExecutor` の
    スレッド生成失敗＝メモリ由来なので、コールごとにスレッドを増やす実装は採らない）。
    締切は google-genai の `HttpOptions(timeout=...)`＝httpx の timeout で掛ける。

    再試行の方針:
      - **タイムアウト**：1回だけやり直す（施主指示）。それでも駄目なら空文字を返す
        ＝「答えが返らなかった」事実として数える（既定値で埋めない）。
      - レート制限・接続断などの一時障害：v8c と同じ回数・同じ待ち方（世界に無関係）。
    """

    def __init__(self, request_timeout_sec: float = DEFAULT_REQUEST_TIMEOUT_SEC,
                 api_key_env: str = "GOOGLE_API_KEY", **kwargs: Any):
        super().__init__(api_key_env=api_key_env, **kwargs)
        self.request_timeout_sec = float(request_timeout_sec)
        self.timeout_retries = 0     # タイムアウトしてやり直した回数
        self.timeout_giveups = 0     # やり直しても駄目だった回数
        self.http_timeout_applied = False
        if self.backend == "genai":
            try:
                from google import genai as new_genai        # type: ignore
                from google.genai import types as genai_types  # type: ignore
                self._client = new_genai.Client(
                    api_key=os.environ.get(api_key_env),
                    http_options=genai_types.HttpOptions(
                        timeout=int(self.request_timeout_sec * 1000)))
                self.http_timeout_applied = True
            except Exception as e:  # noqa: BLE001
                logger.warning("request timeout を設定できなかった（素の client で続行）: %s", e)

    def generate(self, system_prompt: str, user_prompt: str,
                 schema: Optional[Dict[str, Any]] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, tag: str = "agent") -> str:
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        timeouts = 0
        others = 0
        last_err: Optional[Exception] = None
        while True:
            try:
                if self.backend == "genai":
                    text, usage = self._gen_new(system_prompt, user_prompt, schema,
                                                temperature, max_tokens)
                else:
                    text, usage = self._gen_old(system_prompt, user_prompt, schema,
                                                temperature, max_tokens)
                self.usage.add(tag, **usage)
                return text
            except Exception as e:  # noqa: BLE001
                last_err = e
                if _is_timeout(e):
                    if timeouts < TIMEOUT_RETRIES:
                        timeouts += 1
                        self.timeout_retries += 1
                        logger.warning("request timeout（%.0f秒）。1回だけやり直す: %s",
                                       self.request_timeout_sec, e)
                        continue
                    self.timeout_giveups += 1
                    logger.error("request timeout でやり直しても駄目だった: %s", e)
                    self.usage.add(tag, error=True)
                    return ""
                if _is_transient(e) and others < MAX_RATE_LIMIT_RETRIES - 1:
                    wait = _retry_backoff(others)
                    others += 1
                    logger.warning("一時障害（%d/%d）。%d秒待って再試行: %s",
                                   others, MAX_RATE_LIMIT_RETRIES - 1, wait, e)
                    time.sleep(wait)
                    continue
                logger.error("Gemini API error: %s", last_err)
                self.usage.add(tag, error=True)
                return ""


class MockV8DClient(MockV8CClient):
    """v8d 用の mock（配線と集計を通すためだけのもの）。

    v8c の mock との差は1つ＝**行き先に理由欄が無い**（スキーマに無いので書かない）。
    """

    def generate(self, system_prompt: str, user_prompt: str,
                 schema: Optional[Dict[str, Any]] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, tag: str = "agent") -> str:
        raw = super().generate(system_prompt, user_prompt, schema, temperature,
                               max_tokens, tag)
        props = (schema or {}).get("properties", {})
        if "go" in props and "reason" not in props:
            obj = json.loads(raw)
            obj.pop("reason", None)
            return json.dumps(obj, ensure_ascii=False)
        return raw


# 月ごとに書き出すもの（属性名 → ファイル名）。**上書き**で書く。
CHECKPOINT_FILES = (
    ("monthly", "monthly.json"),
    ("offers", "offers.json"),
    ("decisions", "decisions.json"),
    ("plans", "plans.json"),
    ("utterances", "utterances.json"),
    ("listings", "listings.json"),
    ("deliveries", "deliveries.json"),
    ("acquirer_raw", "acquirer_raw.json"),
)


class SimulationV8D(SimulationV8C):

    def __init__(self, cfg: Dict[str, Any], run_dir: str):
        if cfg.get("scenario_version") != "field_v8d":
            raise ValueError("config の scenario_version が field_v8d ではない")
        # 親（v8c）は field_v8c を要求するので、版名だけ差し替えて渡し、あとで戻す。
        super().__init__({**cfg, "scenario_version": "field_v8c"}, run_dir)
        self.cfg = cfg
        # 前置きは v8d のもので置き換える（住民側にX社の名前も命題も出ない）
        self.common_prefix = build_common_prefix_v8d(cfg, self.agents)
        self.acquirer_prefix = build_acquirer_prefix_v8d(cfg, self.agents)
        provider = str(cfg["llm"].get("provider", "mock")).lower()
        if provider == "mock":
            self.client = MockV8DClient(seed=self.seed, usage=self.usage)
        elif provider == "google":
            self.client = TimeoutGeminiClient(
                request_timeout_sec=float(cfg["llm"].get(
                    "request_timeout_sec", DEFAULT_REQUEST_TIMEOUT_SEC)),
                model=str(cfg["llm"].get("model", "gemini-2.5-flash-lite")),
                temperature=float(cfg["llm"].get("temperature", 0.75)),
                max_tokens=int(cfg["llm"].get("max_tokens", 2200)),
                enable_cache=bool(cfg["llm"].get("enable_cache", False)),
                usage=self.usage,
                parallel_workers=int(cfg["llm"].get("parallel_workers", 4)))
        self.checkpoint_dir = os.path.join(run_dir, "checkpoint")
        self.checkpoints_written = 0
        # 行き先の答えが返らなかった回数（HOME で埋めずに別に数える）
        self.plan_no_answer = 0

    # -- 月初の行き先（理由欄なし＝差分3） -------------------------------------

    def _plan_turn(self, step: int, offers: Dict[str, str]) -> Dict[str, str]:
        """月初の思考と行き先。

        v8c との差は2つ:
          1. **理由の一言を聞かない**（差分3）。
          2. **答えが返らなかった行き先を「今月はどこにも行かない」で埋めない**
             （Codex 走行前レビューの必須指摘。v8/v8b/v8c は欠損を HOME に丸めていた＝
             欠損を既定行動に変換していた。60秒の締切を入れた v8d では空応答が
             現実に起こりうるので、ここで分ける。v8c 本走での発生は0件なので、
             実測が0のあいだは v8c と同じ結果になる）。
        """
        items = []
        for a in self.agents:
            aid = str(a["id"])
            up = build_plan_prompt_v8d(a, self.reg, step, self.n_steps,
                                       self.all_labels, self.thought[aid],
                                       offers.get(aid),
                                       neighbours=self.neighbour_names.get(aid))
            items.append((aid, up, plan_schema_v8d(self.all_labels), ":plan"))
        raws = self._call(items, "plan")

        go: Dict[str, str] = {}
        for a in self.agents:
            aid = str(a["id"])
            act = self._read(aid, raws.get(aid, ""), step, "plan")
            where = NO_ANSWER          # 既定で埋めない（答えが無かった事実を残す）
            if act is not None:
                self.thought[aid] = str(act.get("thought", "") or "")[:600]
                value = str(act.get("go", "") or "").strip()
                if value in self.label_to_venue:
                    where = self.label_to_venue[value]
                elif value == HOME:
                    where = HOME
                else:
                    self.invalid_venue += 1
            if where == NO_ANSWER:
                self.plan_no_answer += 1
            go[aid] = where
            self.plans.append({"step": step, "agent_id": aid,
                               "name": a["name"], "go": where,
                               "go_label": (NO_ANSWER if where == NO_ANSWER
                                            else self.venue_labels.get(where, HOME)),
                               "thought": self.thought[aid]})
        return go

    def _scene_turn(self, step, go, offers):
        """場の集まり。**答えが返らなかった人はその月の観測から外す**。

        - どの場にも居ない（居合わせの発話を聞かない・自分も話さない）
        - **隣近所の配送も受け取らない**（Codex 走行前レビュー2巡目の必須指摘。
          「居合わせからは外すが隣近所からは届く」という中途半端な扱いをしない）
        v8c の実装をそのまま使うため、`NO_ANSWER` は「家」と同じ扱いで渡してから、
        その人あての配送を落とす（記録は `plans.json` に `no_answer` のまま残る）。
        """
        absent = {aid for aid, v in go.items() if v == NO_ANSWER}
        heard = super()._scene_turn(
            step, {aid: (HOME if v == NO_ANSWER else v) for aid, v in go.items()},
            offers)
        if absent:
            for aid in absent:
                heard[aid] = []
            self.deliveries = [d for d in self.deliveries
                               if not (d["step"] == step and d["to"] in absent)]
        return heard

    # -- 月末の問い（「戻らない」の一文なし＝差分1） ----------------------------

    def _decide_prompt(self, a, step, thought, offer, heard, list_order,
                       sell_order_, neighbours):
        return build_decide_prompt_v8d(a, self.reg, step, self.n_steps, thought,
                                       offer, heard, list_order=list_order,
                                       sell_order_=sell_order_,
                                       neighbours=neighbours)

    def _decide_turn(self, step, offers, heard):
        # 親（v8c）の実装をそのまま使いたいので、文面のビルダだけ差し替える。
        import src.sim_v8c as _v8c
        original = _v8c.build_decide_prompt_v8c
        _v8c.build_decide_prompt_v8c = build_decide_prompt_v8d
        try:
            return super()._decide_turn(step, offers, heard)
        finally:
            _v8c.build_decide_prompt_v8c = original

    # -- チェックポイント -------------------------------------------------------

    def _checkpoint(self, step: int) -> None:
        """その月までの記録を丸ごと書き出す（上書き）。落ちてもここまでは残る。"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        for attr, name in CHECKPOINT_FILES:
            obj = getattr(self, attr, None)
            if obj is None:
                continue
            tmp = os.path.join(self.checkpoint_dir, name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            os.replace(tmp, os.path.join(self.checkpoint_dir, name))
        row = {
            "step": step,
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cost_usd": round(self._cost_so_far(), 4),
            "offers_sent": self.monthly[-1]["offers_sent"] if self.monthly else 0,
            "listed_this_month": (self.monthly[-1]["listed_this_month"]
                                  if self.monthly else 0),
            "sold_this_month": (self.monthly[-1]["sold_this_month"]
                                if self.monthly else 0),
            "parcels_cum": self.monthly[-1]["parcels_cum"] if self.monthly else 0,
            "utterances_cum": len(self.utterances),
            "timeout_retries": getattr(self.client, "timeout_retries", 0),
            "timeout_giveups": getattr(self.client, "timeout_giveups", 0),
        }
        with open(os.path.join(self.checkpoint_dir, "log.jsonl"), "a",
                  encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.checkpoints_written += 1
        # 無人ランの見張り用に stdout へ1行（logging は stderr へ出る）
        print(f"[v8d] m{step:02d}/{self.n_steps} 提示{row['offers_sent']} "
              f"出品{row['listed_this_month']} 売却{row['sold_this_month']} "
              f"累計{row['parcels_cum']}区画 ${row['cost_usd']:.4f}", flush=True)

    def _step(self, step: int) -> None:
        super()._step(step)
        # 月別集計の行き先を数え直す（親は「家でない＝出席」と数えるので、
        # 答えが返らなかった人が出席に混ざってしまう）。
        if self.monthly:
            m = self.monthly[-1]
            na = sum(1 for p in self.plans
                     if p["step"] == step and p["go"] == NO_ANSWER)
            m["no_answer_go"] = na
            m["attended"] = max(0, int(m["attended"]) - na)
            if na:
                m["by_venue"][NO_ANSWER] = na
        try:
            self._checkpoint(step)
        except Exception as e:  # noqa: BLE001
            # チェックポイントの失敗で走行を落とさない（記録は finalize でも書く）
            logger.warning("チェックポイントの書き出しに失敗（続行）: %s", e)

    # -- 集計と出力 ------------------------------------------------------------

    def _finalize(self, elapsed: float) -> Dict[str, Any]:
        summary = super()._finalize(elapsed)
        summary["scenario_version"] = "field_v8d"
        summary["checkpoints_written"] = self.checkpoints_written
        summary["plan_no_answer"] = self.plan_no_answer
        summary["request_timeout_sec"] = getattr(self.client, "request_timeout_sec",
                                                 None)
        summary["http_timeout_applied"] = getattr(self.client,
                                                  "http_timeout_applied", None)
        summary["timeout_retries"] = getattr(self.client, "timeout_retries", 0)
        summary["timeout_giveups"] = getattr(self.client, "timeout_giveups", 0)
        with open(os.path.join(self.run_dir, "summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary


__all__ = ["SimulationV8D", "MockV8DClient", "TimeoutGeminiClient",
           "CostLimitReached", "RESULT_SOLD", "RESULT_NOT_SOLD",
           "RESULT_NO_ANSWER"]
