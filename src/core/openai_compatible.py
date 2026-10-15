from __future__ import annotations

import json
import http.client
import re
import socket
import ssl
import threading
import time
import urllib.parse
from typing import Any

from .llama_manager import (
    DEFAULT_GENERATION_TIMEOUT_SECONDS,
    LlamaConnectionError,
    LlamaError,
    LlamaTimeoutError,
)
from .prompt_engine import clean_model_output


LLM_PROVIDER_LLAMA_CPP = "llama_cpp"
LLM_PROVIDER_OPENAI_COMPATIBLE = "openai_compatible"
DEFAULT_OPENAI_BASE_URL = "http://127.0.0.1:8000/v1"


def normalize_openai_base_url(value: str) -> str:
    """Normalize an OpenAI-compatible API base URL without accepting credentials."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("OpenAI-compatible server URL is required.")
    parsed = urllib.parse.urlsplit(value.strip())
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Enter an HTTP or HTTPS API base URL without credentials or query parameters.")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("The API URL contains an invalid port.") from exc
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions") or path.endswith("/models"):
        raise ValueError("Enter the API base URL, for example http://192.168.1.20:1234/v1.")
    return urllib.parse.urlunsplit(
        (parsed.scheme.lower(), parsed.netloc, path, "", "")
    ).rstrip("/")


class OpenAICompatibleClient:
    """Small synchronous client for OpenAI-compatible chat completion servers."""

    def __init__(self, base_url: str, model: str, api_key: str = "") -> None:
        self.base_url = normalize_openai_base_url(base_url)
        self.model = str(model or "").strip()
        if not self.model:
            raise ValueError("OpenAI-compatible model name is required.")
        self.api_key = str(api_key or "").strip()
        self.last_response_metrics: dict[str, int | float] = {}
        self._connection_lock = threading.Lock()
        self._active_connection: http.client.HTTPConnection | None = None
        self._cancelled = False

    @property
    def _api_root(self) -> str:
        path = urllib.parse.urlsplit(self.base_url).path
        if path.endswith("/v1"):
            return self.base_url
        return f"{self.base_url}/v1"

    def cancel(self) -> None:
        with self._connection_lock:
            self._cancelled = True
            connection = self._active_connection
        if connection is not None:
            connection.close()

    def _request(
        self,
        endpoint: str,
        *,
        payload: dict[str, Any] | None,
        timeout: float,
    ) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        body: bytes | None = None
        if payload is not None:
            request_payload = dict(payload)
            request_payload["model"] = self.model
            request_payload["stream"] = False
            body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        parsed = urllib.parse.urlsplit(
            f"{self._api_root}/{endpoint.lstrip('/')}"
        )
        connection_type = (
            http.client.HTTPSConnection
            if parsed.scheme == "https"
            else http.client.HTTPConnection
        )
        connection_options: dict[str, Any] = {"timeout": timeout}
        if parsed.scheme == "https":
            connection_options["context"] = ssl.create_default_context()
        connection = connection_type(
            parsed.hostname,
            parsed.port,
            **connection_options,
        )
        request_path = urllib.parse.urlunsplit(("", "", parsed.path, parsed.query, ""))
        with self._connection_lock:
            if self._cancelled:
                raise LlamaError("The OpenAI-compatible request was cancelled.")
            self._active_connection = connection
        try:
            connection.request(
                "POST" if body is not None else "GET",
                request_path,
                body=body,
                headers=headers,
            )
            response = connection.getresponse()
            status = response.status
            raw = response.read()
        except (TimeoutError, socket.timeout) as exc:
            raise LlamaTimeoutError("OpenAI-compatible server request timed out.") from exc
        except (http.client.HTTPException, ConnectionError, OSError) as exc:
            if self._cancelled:
                raise LlamaError("The OpenAI-compatible request was cancelled.") from exc
            raise LlamaConnectionError(
                "Could not connect to the configured OpenAI-compatible server."
            ) from exc
        finally:
            connection.close()
            with self._connection_lock:
                if self._active_connection is connection:
                    self._active_connection = None
        if self._cancelled:
            raise LlamaError("The OpenAI-compatible request was cancelled.")
        if status < 200 or status >= 300:
            raise self._http_error(status, raw)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LlamaError("The OpenAI-compatible server returned invalid JSON.") from exc
        if not isinstance(data, dict):
            raise LlamaError("The OpenAI-compatible server returned an invalid response.")
        return data

    def chat_completion(
        self,
        payload: dict[str, Any],
        timeout: float = DEFAULT_GENERATION_TIMEOUT_SECONDS,
    ) -> str:
        started_at = time.monotonic()
        request_payload = dict(payload)
        request_payload["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            data = self._request(
                "chat/completions",
                payload=request_payload,
                timeout=timeout,
            )
        except LlamaConnectionError as exc:
            if (
                exc.status not in {400, 422}
                or "chat_template_kwargs" not in str(exc).casefold()
            ):
                raise
            data = self._request("chat/completions", payload=payload, timeout=timeout)
        try:
            choice = data["choices"][0]
            message = choice["message"]
            content = message["content"]
            if isinstance(content, list):
                content = "".join(
                    str(item.get("text", ""))
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
        except (KeyError, IndexError, TypeError) as exc:
            raise LlamaError("The OpenAI-compatible server returned an invalid chat response.") from exc
        if not isinstance(content, str) or not content.strip():
            if (
                isinstance(message, dict)
                and isinstance(message.get("reasoning_content"), str)
                and isinstance(choice, dict)
                and choice.get("finish_reason") == "length"
            ):
                raise LlamaError(
                    "The server used the output token limit for reasoning and returned no final response. "
                    "This server may not support chat_template_kwargs; disable reasoning in its server settings."
                )
            raise LlamaError("The OpenAI-compatible server returned an empty response.")
        usage = data.get("usage")
        self.last_response_metrics = {
            key: value
            for key, value in {
                "prompt_tokens": usage.get("prompt_tokens") if isinstance(usage, dict) else None,
                "generated_tokens": usage.get("completion_tokens") if isinstance(usage, dict) else None,
                "request_seconds": time.monotonic() - started_at,
            }.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
        }
        return clean_model_output(content)

    def test_connection(self, timeout: float = 10.0) -> None:
        """Check the standard models endpoint; this sends no prompt or image data."""
        data = self._request("models", payload=None, timeout=timeout)
        if not isinstance(data.get("data"), list):
            raise LlamaError("The server responded, but /v1/models did not return a model list.")

    def _http_error(self, status: int, body: bytes) -> LlamaConnectionError:
        message = ""
        try:
            payload = json.loads(body.decode("utf-8"))
            error = payload.get("error", payload) if isinstance(payload, dict) else {}
            if isinstance(error, dict):
                message = str(error.get("message") or "")
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
            pass
        message = re.sub(r"\s+", " ", message).strip()
        if self.api_key:
            message = message.replace(self.api_key, "[redacted]")
        if len(message) > 300:
            message = message[:300] + "…"
        detail = f": {message}" if message else ""
        return LlamaConnectionError(
            f"OpenAI-compatible server returned HTTP {status}{detail}",
            status=status,
        )
