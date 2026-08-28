"""Gemini Batch API のジョブ実行層（費用の節約だけを目的にした配線）。

**世界には一切触れない。** ここがするのは「同じプロンプトを、同期ではなく Batch API で
投げる」ことだけである。system prompt・user prompt・schema・temperature・max_tokens は
呼び出し側から受け取ったものをそのまま渡す（1バイトも作らない・書き換えない）。

Batch API は同期と同じモデル・同じリクエスト形式で、料金が 50% になる代わりに
返却時刻の保証が無い。したがって:

  - ジョブ名は `run_dir/batch_jobs/<tag>.json` に必ず保存する（途中で落ちても再取得できる）
  - 応答が欠けた行・エラーだった行は、呼び出し側が同期にフォールバックする
  - 往復時間を計測して stats に返す（採否の判断材料は実測だけにする）

usage は同期と同じ UsageMeter に入れるが、**tag に `|batch` を付ける**。
料金は同期の半額なので、集計側（tools/cost_breakdown_v5c.py）はこの印で 0.5 を掛ける。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# inlined requests は1ジョブあたりのリクエスト総サイズに上限がある（20MB）。
# 1件あたり数KBなので 200 件で切っておけば安全側。
MAX_REQUESTS_PER_JOB = 200

TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED", "JOB_STATE_PARTIALLY_SUCCEEDED",
}


def request_fingerprint(model: str, requests: List[Dict[str, Any]]) -> str:
    """このジョブが「まったく同じ問い合わせ」かどうかの指紋。

    件数と tag が同じでも中身が違えば別ジョブである（Codexレビュー 2026-08-28・重大度高）。
    model・system・user・schema・temperature・max_tokens・thinking_budget を
    **リクエストの順序込み**で SHA-256 にする。1つでも違えば台帳は再利用されない。
    """
    unit = chr(31)      # フィールドの区切り（本文には現れない制御文字）
    rec = chr(30)       # リクエストの区切り
    h = hashlib.sha256()
    h.update(("model=" + str(model)).encode("utf-8"))
    for r in requests:
        for key in ("system_prompt", "user_prompt", "temperature", "max_tokens",
                    "thinking_budget"):
            h.update((unit + key + "=" + repr(r.get(key))).encode("utf-8"))
        h.update((unit + "schema="
                  + json.dumps(r.get("schema"), sort_keys=True,
                               ensure_ascii=False)).encode("utf-8"))
        h.update(rec.encode("utf-8"))
    return h.hexdigest()


def _state_name(job: Any) -> str:
    state = getattr(job, "state", None)
    return str(getattr(state, "name", state) or "")


class BatchRunner:
    """google-genai の batches API を「同じ順序で結果を返す関数」に包んだもの。"""

    def __init__(self, client: Any, types: Any, model: str,
                 jobs_dir: Optional[str] = None,
                 poll_interval: float = 5.0,
                 poll_interval_max: float = 30.0,
                 timeout_sec: float = 3600.0) -> None:
        self._client = client
        self._types = types
        self.model = model
        self.jobs_dir = jobs_dir
        self.poll_interval = float(poll_interval)
        self.poll_interval_max = float(poll_interval_max)
        self.timeout_sec = float(timeout_sec)
        # 採否の判断に使う実測（往復時間・ジョブ数・フォールバック件数）
        self.stats: List[Dict[str, Any]] = []

    # -- ジョブ台帳（再開可能にするための唯一の状態） ----------------------

    def _record_path(self, tag: str) -> Optional[str]:
        if not self.jobs_dir:
            return None
        os.makedirs(self.jobs_dir, exist_ok=True)
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in tag)
        return os.path.join(self.jobs_dir, f"{safe}.json")

    def _load_record(self, tag: str) -> Dict[str, Any]:
        path = self._record_path(tag)
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:  # 壊れた台帳で走行を止めない
            logger.warning("batch job record 読込失敗 %s: %s", path, e)
            return {}

    def _save_record(self, tag: str, data: Dict[str, Any]) -> None:
        path = self._record_path(tag)
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("batch job record 保存失敗 %s: %s", path, e)

    # -- リクエスト組み立て（内容は呼び出し側のものをそのまま使う） ----------

    def _inlined(self, req: Dict[str, Any]) -> Any:
        cfg: Dict[str, Any] = {
            "system_instruction": req["system_prompt"],
            "temperature": req["temperature"],
            "max_output_tokens": req["max_tokens"],
        }
        if req.get("schema") is not None:
            cfg["response_mime_type"] = "application/json"
            cfg["response_schema"] = req["schema"]
        if req.get("thinking_budget") is not None:
            cfg["thinking_config"] = self._types.ThinkingConfig(
                thinking_budget=req["thinking_budget"])
        return self._types.InlinedRequest(
            model=self.model,
            contents=req["user_prompt"],
            config=self._types.GenerateContentConfig(**cfg),
        )

    # -- 実行 ---------------------------------------------------------------

    def run(self, tag: str, requests: List[Dict[str, Any]]
            ) -> Tuple[List[Optional[str]], List[Optional[Dict[str, int]]]]:
        """requests と同じ長さ・同じ順序で (テキスト, usage) を返す。

        取れなかった行は (None, None)。呼び出し側が同期で埋め直す。
        """
        texts: List[Optional[str]] = [None] * len(requests)
        usages: List[Optional[Dict[str, int]]] = [None] * len(requests)
        if not requests:
            return texts, usages

        chunks = [(i, requests[i:i + MAX_REQUESTS_PER_JOB])
                  for i in range(0, len(requests), MAX_REQUESTS_PER_JOB)]
        for part, (offset, chunk) in enumerate(chunks):
            sub_tag = tag if len(chunks) == 1 else f"{tag}.p{part}"
            got_texts, got_usages, stat = self._run_one(sub_tag, chunk)
            for k in range(len(chunk)):
                texts[offset + k] = got_texts[k]
                usages[offset + k] = got_usages[k]
            self.stats.append(stat)
        return texts, usages

    def _run_one(self, tag: str, chunk: List[Dict[str, Any]]):
        n = len(chunk)
        texts: List[Optional[str]] = [None] * n
        usages: List[Optional[Dict[str, int]]] = [None] * n
        stat: Dict[str, Any] = {"tag": tag, "requests": n, "job_name": "",
                                "state": "", "elapsed_sec": 0.0, "reused": False,
                                "returned": 0, "error": ""}

        record = self._load_record(tag)
        fingerprint = request_fingerprint(self.model, chunk)
        job_name = (record.get("job_name")
                    if (record.get("requests") == n
                        and record.get("request_hash") == fingerprint)
                    else None)
        if record.get("job_name") and not job_name:
            logger.info("batch %s: 台帳のジョブは中身が違うので使わない（作り直す）", tag)
        started = time.time()

        try:
            if job_name:
                job = self._client.batches.get(name=job_name)
                stat["reused"] = True
                # 再開時は投入時刻からの経過が本当の往復時間なので、それを使う。
                started = float(record.get("submitted_at", started))
            else:
                job = self._client.batches.create(
                    model=self.model,
                    src=[self._inlined(r) for r in chunk],
                    config={"display_name": f"quiet-{tag}-{int(time.time())}"},
                )
                job_name = getattr(job, "name", "")
                self._save_record(tag, {"job_name": job_name, "requests": n,
                                        "request_hash": fingerprint,
                                        "submitted_at": started, "tag": tag})
            stat["job_name"] = job_name or ""

            wait = self.poll_interval
            while _state_name(job) not in TERMINAL_STATES:
                if time.time() - started > self.timeout_sec:
                    stat["state"] = _state_name(job)
                    stat["error"] = "timeout"
                    stat["elapsed_sec"] = round(time.time() - started, 1)
                    logger.warning("batch %s が %.0fs 経っても終わらない。同期に落とす",
                                   tag, self.timeout_sec)
                    return texts, usages, stat
                time.sleep(wait)
                wait = min(self.poll_interval_max, wait * 1.5)
                job = self._client.batches.get(name=job_name)

            stat["state"] = _state_name(job)
            stat["elapsed_sec"] = round(time.time() - started, 1)
            if stat["state"] not in ("JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"):
                stat["error"] = str(getattr(job, "error", "") or stat["state"])
                return texts, usages, stat

            responses = list(getattr(getattr(job, "dest", None),
                                     "inlined_responses", None) or [])
            for i, item in enumerate(responses[:n]):
                if getattr(item, "error", None):
                    continue
                resp = getattr(item, "response", None)
                if resp is None:
                    continue
                text = (getattr(resp, "text", None) or "").strip()
                if not text:
                    # 空応答は「取れなかった行」として扱う。どのスキーマも空文字を
                    # 正当な構造化出力とは認めない（Codexレビュー 2026-08-28）。
                    continue
                um = getattr(resp, "usage_metadata", None)
                texts[i] = text
                usages[i] = {
                    "input_tokens": getattr(um, "prompt_token_count", 0) or 0,
                    "cached_tokens": getattr(um, "cached_content_token_count", 0) or 0,
                    "output_tokens": getattr(um, "candidates_token_count", 0) or 0,
                }
            stat["returned"] = sum(1 for t in texts if t is not None)
        except Exception as e:
            stat["error"] = f"{type(e).__name__}: {e}"
            stat["elapsed_sec"] = round(time.time() - started, 1)
            logger.warning("batch %s 失敗（同期に落とす）: %s", tag, e)
        return texts, usages, stat


__all__ = ["BatchRunner", "MAX_REQUESTS_PER_JOB", "TERMINAL_STATES",
           "request_fingerprint"]
