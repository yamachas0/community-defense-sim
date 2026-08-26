"""Provider-switching LLM clients.

前回ハッカソン (2d-multi-places-simulation-on-fire-public/llm_client_factory.py) の
GeminiClient を移植し、以下だけ変更した:
  - スキーマを system prompt から推測せず **呼び出し側が明示的に渡す** 形に一般化
  - usage (input / cached / output tokens) を UsageMeter に集約
  - 新SDK `google.genai` があればそちらを使い、無ければ旧 `google.generativeai` に落ちる
  - API を一切叩かない MockClient を同居させ、E2E をオフラインで通せるようにした

共通インタフェース:
    generate(system_prompt, user_prompt, schema=None, temperature=None,
             max_tokens=None, cache_key=None) -> str   (JSON文字列)
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import random
import re
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_RATE_LIMIT_RETRIES = 4


def _retry_backoff(attempt: int) -> int:
    return 2 ** attempt  # 1, 2, 4, 8


# ---------------------------------------------------------------------------
# Usage accounting
# ---------------------------------------------------------------------------

class UsageMeter:
    """API 呼び出しとトークン消費の集計。コスト見積りの実測値はここから出す。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.input_tokens = 0
        self.cached_tokens = 0
        self.output_tokens = 0
        self.errors = 0
        self.by_tag: Dict[str, Dict[str, int]] = {}

    def add(self, tag: str, input_tokens: int = 0, cached_tokens: int = 0,
            output_tokens: int = 0, error: bool = False) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += int(input_tokens)
            self.cached_tokens += int(cached_tokens)
            self.output_tokens += int(output_tokens)
            if error:
                self.errors += 1
            slot = self.by_tag.setdefault(tag, {"calls": 0, "input": 0, "cached": 0, "output": 0})
            slot["calls"] += 1
            slot["input"] += int(input_tokens)
            slot["cached"] += int(cached_tokens)
            slot["output"] += int(output_tokens)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "cached_tokens": self.cached_tokens,
            "output_tokens": self.output_tokens,
            "errors": self.errors,
            "by_tag": self.by_tag,
        }


# ---------------------------------------------------------------------------
# Schema conversion helpers
# ---------------------------------------------------------------------------

def _uppercase_types(node: Any) -> Any:
    """google-genai (新SDK) の types.Schema は JSON Schema の type を大文字で受ける。"""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "type" and isinstance(v, str):
                out[k] = v.upper()
            else:
                out[k] = _uppercase_types(v)
        return out
    if isinstance(node, list):
        return [_uppercase_types(v) for v in node]
    return node


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

class GeminiClient:
    """Google Gemini client.

    backend:
      - "genai"       : google-genai (新SDK, 推奨)
      - "generativeai": google-generativeai (旧SDK, 前回ハッカソンで使用)
    どちらも同じ generate() を提供する。
    """

    CACHE_TTL_SECONDS = 3600

    def __init__(self, model: str = "gemini-2.5-flash-lite", temperature: float = 0.9,
                 max_tokens: int = 420, enable_cache: bool = False,
                 usage: Optional[UsageMeter] = None, api_key_env: str = "GOOGLE_API_KEY"):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_cache = enable_cache
        self.usage = usage or UsageMeter()
        self._cache_handles: Dict[str, Any] = {}
        self._cache_lock = threading.Lock()
        self.backend = None
        self._client = None
        self._genai = None
        self._types = None

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{api_key_env} が未設定。.env に GOOGLE_API_KEY を入れるか "
                f"provider: mock で実行しろ。"
            )
        try:
            from google import genai as new_genai            # type: ignore
            from google.genai import types as genai_types    # type: ignore
            self._client = new_genai.Client(api_key=api_key)
            self._types = genai_types
            self.backend = "genai"
        except Exception as e_new:
            try:
                import google.generativeai as old_genai       # type: ignore
                old_genai.configure(api_key=api_key)
                self._genai = old_genai
                self.backend = "generativeai"
                logger.warning("google-genai が使えないため旧SDK google-generativeai を使用: %s", e_new)
            except Exception as e_old:
                raise RuntimeError(
                    f"Gemini SDK を初期化できない。google-genai / google-generativeai の"
                    f"どちらも失敗: new={e_new} old={e_old}"
                )

    # -- explicit context cache (最小トークン数を下回ると作成に失敗する。失敗時は素通し) --

    def _get_cache(self, system_prompt: str):
        key = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
        handle = self._cache_handles.get(key, "missing")
        if handle != "missing":
            return handle or None
        with self._cache_lock:
            handle = self._cache_handles.get(key, "missing")
            if handle != "missing":
                return handle or None
            try:
                if self.backend == "genai":
                    cache = self._client.caches.create(
                        model=self.model,
                        config=self._types.CreateCachedContentConfig(
                            system_instruction=system_prompt,
                            ttl=f"{self.CACHE_TTL_SECONDS}s",
                        ),
                    )
                    self._cache_handles[key] = cache.name
                    logger.info("Gemini cache created: %s", cache.name)
                    return cache.name
                else:
                    from google.generativeai import caching  # type: ignore
                    model_name = self.model if self.model.startswith("models/") else f"models/{self.model}"
                    cache = caching.CachedContent.create(
                        model=model_name,
                        system_instruction=system_prompt,
                        ttl=datetime.timedelta(seconds=self.CACHE_TTL_SECONDS),
                    )
                    self._cache_handles[key] = cache
                    logger.info("Gemini cache created: %s", cache.name)
                    return cache
            except Exception as e:
                logger.warning("Gemini cache 作成失敗 (uncached で続行): %s", e)
                self._cache_handles[key] = False
                return None

    def generate(self, system_prompt: str, user_prompt: str,
                 schema: Optional[Dict[str, Any]] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 tag: str = "agent") -> str:
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens

        last_err = None
        for attempt in range(MAX_RATE_LIMIT_RETRIES):
            try:
                if self.backend == "genai":
                    text, usage = self._gen_new(system_prompt, user_prompt, schema,
                                                temperature, max_tokens)
                else:
                    text, usage = self._gen_old(system_prompt, user_prompt, schema,
                                                temperature, max_tokens)
                self.usage.add(tag, **usage)
                return text
            except Exception as e:
                msg = str(e).lower()
                name = type(e).__name__
                transient = ("429" in msg or "rate" in msg or "quota" in msg
                             or "resource_exhausted" in msg or "503" in msg
                             or "unavailable" in msg or name == "ResourceExhausted")
                if transient and attempt < MAX_RATE_LIMIT_RETRIES - 1:
                    last_err = e
                    wait = _retry_backoff(attempt)
                    logger.warning("Gemini rate/quota hit (%d/%d). %ds 待って再試行",
                                   attempt + 1, MAX_RATE_LIMIT_RETRIES, wait)
                    time.sleep(wait)
                    continue
                logger.error("Gemini API error: %s", e)
                self.usage.add(tag, error=True)
                return ""
        logger.error("Gemini: %d 回リトライして諦めた: %s", MAX_RATE_LIMIT_RETRIES, last_err)
        self.usage.add(tag, error=True)
        return ""

    def _gen_new(self, system_prompt, user_prompt, schema, temperature, max_tokens):
        cfg: Dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        cache_name = self._get_cache(system_prompt) if self.enable_cache else None
        if cache_name:
            cfg["cached_content"] = cache_name
        else:
            cfg["system_instruction"] = system_prompt
        if schema is not None:
            cfg["response_mime_type"] = "application/json"
            cfg["response_schema"] = _uppercase_types(schema)
        resp = self._client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=self._types.GenerateContentConfig(**cfg),
        )
        um = getattr(resp, "usage_metadata", None)
        usage = {
            "input_tokens": getattr(um, "prompt_token_count", 0) or 0,
            "cached_tokens": getattr(um, "cached_content_token_count", 0) or 0,
            "output_tokens": getattr(um, "candidates_token_count", 0) or 0,
        }
        text = getattr(resp, "text", None) or ""
        return text.strip(), usage

    def _gen_old(self, system_prompt, user_prompt, schema, temperature, max_tokens):
        gen_config: Dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if schema is not None:
            gen_config["response_mime_type"] = "application/json"
            gen_config["response_schema"] = schema
        cache = self._get_cache(system_prompt) if self.enable_cache else None
        if cache is not None:
            model = self._genai.GenerativeModel.from_cached_content(cached_content=cache)
        else:
            model = self._genai.GenerativeModel(self.model, system_instruction=system_prompt)
        resp = model.generate_content(user_prompt, generation_config=gen_config)
        um = getattr(resp, "usage_metadata", None)
        usage = {
            "input_tokens": getattr(um, "prompt_token_count", 0) or 0,
            "cached_tokens": getattr(um, "cached_content_token_count", 0) or 0,
            "output_tokens": getattr(um, "candidates_token_count", 0) or 0,
        }
        text = getattr(resp, "text", None) or ""
        return text.strip(), usage

    def count_tokens(self, text: str) -> int:
        """countTokens API。生成を伴わないメータリング呼び出し。"""
        if self.backend == "genai":
            r = self._client.models.count_tokens(model=self.model, contents=text)
            return int(getattr(r, "total_tokens", 0) or 0)
        m = self._genai.GenerativeModel(self.model)
        r = m.count_tokens(text)
        return int(getattr(r, "total_tokens", 0) or 0)


# ---------------------------------------------------------------------------
# OpenAI (比較用。既定では使わない)
# ---------------------------------------------------------------------------

class OpenAIClient:
    def __init__(self, model: str = "gpt-5.4-nano", temperature: float = 0.9,
                 max_tokens: int = 420, usage: Optional[UsageMeter] = None):
        from openai import OpenAI  # type: ignore
        from .schemas import to_openai_json_schema  # noqa: F401  (呼び出し側で使う)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.usage = usage or UsageMeter()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY が未設定")
        self._client = OpenAI(api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str,
                 schema: Optional[Dict[str, Any]] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, tag: str = "agent") -> str:
        from .schemas import to_openai_json_schema
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_completion_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if schema is not None:
            kwargs["response_format"] = to_openai_json_schema(schema, "agent_action")
        if not str(self.model).startswith("gpt-5"):
            kwargs["temperature"] = temperature
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            logger.error("OpenAI API error: %s", e)
            self.usage.add(tag, error=True)
            return ""
        u = getattr(resp, "usage", None)
        self.usage.add(
            tag,
            input_tokens=getattr(u, "prompt_tokens", 0) or 0,
            cached_tokens=getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0,
            output_tokens=getattr(u, "completion_tokens", 0) or 0,
        )
        ch = resp.choices[0] if resp.choices else None
        return (ch.message.content or "").strip() if ch and ch.message else ""


# ---------------------------------------------------------------------------
# Mock (API を一切叩かない。E2E 配線確認とコスト見積り用)
# ---------------------------------------------------------------------------

_PARCEL_RE = re.compile(r"\bP\d{2}\b")
_OFFER_RE = re.compile(r"\bO\d{4}\b")
_AGENT_RE = re.compile(r"\b(?:HH|BZ|BR|AQ|MU|MD)\d{2}\b")
_VENUE_RE = re.compile(r"\bV\d{2}\b")
_STEP_RE = re.compile(r"第\s*(\d+)\s*月")
_OBSERVATION_ID_RE = re.compile(r"\[((?:SALE|UTT|MSG|NEWS|OFFER|LEASE|TALK)-[^\]]+)\]")


class MockClient:
    """スタブ LLM。

    **これはシミュレーションの一部ではない。** 配線 (帳簿・ログ・KPI・可視化) を
    オフラインで通すためのテストダブルであり、ここにある確率や step 依存の分岐は
    「もっともらしい入力を作るためだけ」のもの。本番ランでは一切使わない。
    プロンプトの実文字列はそのまま受け取るので、トークン数の実測にも使える。
    """

    def __init__(self, seed: int = 42, usage: Optional[UsageMeter] = None,
                 model: str = "mock", **_: Any):
        self.model = model
        self.usage = usage or UsageMeter()
        self.backend = "mock"
        self._rng_lock = threading.Lock()
        self._seed = seed
        self.prompt_log: List[Dict[str, Any]] = []

    def _rng(self, key: str) -> random.Random:
        h = hashlib.sha256(f"{self._seed}:{key}".encode("utf-8")).hexdigest()
        return random.Random(int(h[:16], 16))

    def generate(self, system_prompt: str, user_prompt: str,
                 schema: Optional[Dict[str, Any]] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, tag: str = "agent") -> str:
        with self._rng_lock:
            self.prompt_log.append({
                "tag": tag,
                "system_chars": len(system_prompt),
                "user_chars": len(user_prompt),
                "system": system_prompt,
                "user": user_prompt,
            })
        self.usage.add(tag, input_tokens=0, output_tokens=0)

        if schema is not None and "strategy" in schema.get("properties", {}) \
                and "action_type" not in schema.get("properties", {}):
            return json.dumps({
                "situation_assessment": "mock strategy assessment",
                "strategy": "mock strategy",
                "next_milestone": "mock milestone",
                "success_measure": "mock measure",
                "alternatives": ["mock alternative A", "mock alternative B"],
                "selection_basis": "mock basis",
                "revision_reason": "mock revision",
            }, ensure_ascii=False)
        if schema is not None and "results" in schema.get("properties", {}):
            return self._mock_classify(user_prompt)
        role = self._role_from_schema(schema)
        return self._mock_action(role, system_prompt, user_prompt, schema)

    @staticmethod
    def _role_from_schema(schema: Optional[Dict[str, Any]]) -> str:
        if not schema:
            return "household"
        enum = schema.get("properties", {}).get("action_type", {}).get("enum", [])
        for role_key, marker in (("acquirer", "make_offer"), ("municipality", "enact_ordinance"),
                                 ("media", "publish"), ("broker", "circulate_listing"),
                                 ("business", "close_shop"), ("household", "list_for_sale")):
            if marker in enum:
                return role_key
        return "household"

    def _mock_classify(self, user_prompt: str) -> str:
        ids = [int(m) for m in re.findall(r"^\s*(\d+)\.", user_prompt, flags=re.M)]
        rng = self._rng("classify:" + user_prompt[:64])
        results = [{"id": i,
                    "frame": rng.choice(["our_town", "their_town", "neutral"]),
                    "about_acquisition": rng.random() < 0.6} for i in ids]
        return json.dumps({"results": results}, ensure_ascii=False)

    def _mock_action(self, role: str, system_prompt: str, user_prompt: str,
                     schema: Optional[Dict[str, Any]] = None) -> str:
        parcels = sorted(set(_PARCEL_RE.findall(user_prompt)))
        offers = sorted(set(_OFFER_RE.findall(user_prompt)))
        agents = sorted(set(_AGENT_RE.findall(user_prompt)))
        observed_ids = list(dict.fromkeys(_OBSERVATION_ID_RE.findall(user_prompt)))
        m = _STEP_RE.search(user_prompt)
        step = int(m.group(1)) if m else 1
        rng = self._rng(f"{role}:{step}:{system_prompt[:48]}:{len(user_prompt)}")
        prog = min(1.0, step / 36.0)

        act = {"action_type": "hold", "target": "", "amount": 0, "utterance": "",
               "utterance_channel": "none", "utterance_to": "", "memory": "",
               "reasoning": "", "evidence": []}

        if role == "acquirer":
            if parcels and rng.random() < 0.85:
                act.update(action_type="make_offer", target=rng.choice(parcels),
                           amount=int(2400 * (1.0 + 0.5 * rng.random())))
            else:
                act.update(action_type="wait")
        elif role == "household":
            if offers and rng.random() < 0.45:
                act.update(action_type="accept_offer", target=rng.choice(offers))
            elif offers and rng.random() < 0.5:
                act.update(action_type="counter_offer", target=rng.choice(offers),
                           amount=int(3000 + 2000 * rng.random()))
            elif parcels and rng.random() < 0.15:
                act.update(action_type="list_for_sale", target=rng.choice(parcels),
                           amount=int(2800 + 1500 * rng.random()))
            elif rng.random() < 0.05 * prog:
                act.update(action_type="move_out")
            else:
                act.update(action_type="hold")
            if rng.random() < 0.6:
                act.update(utterance="近所の家がまた売れたらしい。この街、どうなるんだろう。",
                           utterance_channel="public")
        elif role == "business":
            if rng.random() < 0.1 * prog:
                act.update(action_type="close_shop")
            elif agents and rng.random() < 0.3:
                act.update(action_type="negotiate_rent", target=rng.choice(agents),
                           amount=int(18 + 10 * rng.random()))
            else:
                act.update(action_type="continue")
            if rng.random() < 0.5:
                act.update(utterance="家賃の話が来た。正直しんどい。", utterance_channel="public")
        elif role == "broker":
            if agents:
                act.update(action_type=rng.choice(["circulate_listing", "approach_owner"]),
                           target=rng.choice(agents),
                           utterance="いい話がありますよ。", utterance_channel="private",
                           utterance_to=rng.choice(agents))
            else:
                act.update(action_type="hold")
        elif role == "municipality":
            if prog > 0.6 and rng.random() < 0.35:
                act.update(action_type="enact_ordinance", target="土地取得届出条例",
                           utterance="一定規模以上の取得に事前届出を求める条例を施行します。",
                           utterance_channel="public")
            elif prog > 0.3:
                act.update(action_type="study_ordinance",
                           utterance="実態把握を進めています。", utterance_channel="public")
            else:
                act.update(action_type="monitor")
        elif role == "media":
            if prog > 0.35 and rng.random() < 0.4:
                act.update(action_type="publish", target="",
                           utterance="【調査】町の一角、所有者が同じ名義に集中",
                           utterance_channel="public")
            elif rng.random() < 0.5:
                act.update(action_type="investigate")
            else:
                act.update(action_type="silent")

        # v3 の追加フィールドへ適合させる。Mockは配線確認専用で、世界の行動規則ではない。
        properties = (schema or {}).get("properties", {})
        allowed_actions = properties.get("action_type", {}).get("enum", [])
        if allowed_actions and act["action_type"] not in allowed_actions:
            fallback = {
                "business": "operate",
                "media": "routine_reporting",
                "household": "hold",
                "broker": "hold",
                "municipality": "monitor",
                "acquirer": "wait",
            }
            act["action_type"] = fallback[role]
            act["target"] = ""
            act["amount"] = 0
        if "location" in properties:
            venues = sorted(set(_VENUE_RE.findall(system_prompt)))
            act["location"] = rng.choice(venues + ["HOME", "OFFICE"])
            act["utterance_channel"] = {
                "public": "ambient", "private": "direct",
            }.get(act["utterance_channel"], act["utterance_channel"])
        if "under_name" in properties:
            legal_names = properties["under_name"].get("enum", [])
            act["under_name"] = rng.choice(legal_names) if legal_names else ""
        act["memory"] = f"step{step}: {act['action_type']} を選んだ (mock)"
        act["reasoning"] = "mock client — 実際の判断はしていない"
        evidence = []
        if observed_ids:
            evidence.append(rng.choice(observed_ids))
        if act.get("target") in parcels:
            evidence.append(act["target"])
        act["evidence"] = list(dict.fromkeys(evidence))
        return json.dumps(act, ensure_ascii=False)

    def count_tokens(self, text: str) -> int:
        raise RuntimeError("MockClient は count_tokens を持たない")


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

def create_llm_client(llm_config: Dict[str, Any], usage: Optional[UsageMeter] = None):
    provider = (llm_config.get("provider") or "mock").lower()
    model = llm_config.get("model")
    temperature = float(llm_config.get("temperature", 0.9))
    max_tokens = int(llm_config.get("max_tokens", 420))

    if provider == "mock":
        return MockClient(seed=int(llm_config.get("seed", 42)), usage=usage)
    if provider == "google":
        return GeminiClient(model=model or "gemini-2.5-flash-lite", temperature=temperature,
                            max_tokens=max_tokens,
                            enable_cache=bool(llm_config.get("enable_cache", False)),
                            usage=usage)
    if provider == "openai":
        return OpenAIClient(model=model or "gpt-5.4-nano", temperature=temperature,
                            max_tokens=max_tokens, usage=usage)
    raise ValueError(f"Unknown LLM provider: {provider}")
