"""Small OpenAI-compatible JSON client for API-based runs."""

from __future__ import annotations

import ast
import json
import os
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from openai import DefaultHttpxClient, OpenAI
import yaml

from compresstab.table.store import approx_token_count

DEFAULT_BASE_URLS: list[str] = []


@dataclass
class UsageRecord:
    model: str
    base_url: str
    input_tokens_est: int
    output_tokens_est: int
    total_tokens_est: int
    measurement: str = "estimated"


class LLMResponseFormatError(ValueError):
    """The relay returned content that could not be parsed as one JSON object."""


def is_transient_api_error(exc: Exception) -> bool:
    """Return whether an API failure is safe to retry without changing meaning."""

    names = {cls.__name__ for cls in type(exc).__mro__}
    if names & {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    }:
        return True
    status_code = getattr(exc, "status_code", None)
    return isinstance(status_code, int) and (status_code == 429 or 500 <= status_code < 600)


class JsonLLMClient:
    def __init__(
        self,
        *,
        env_file: str | Path | None = None,
        timeout: float | None = None,
        max_completion_tokens: int = 2048,
        base_url: str | None = None,
        api_key: str | None = None,
        local_only: bool = False,
        disable_thinking: bool = False,
        json_mode: bool = True,
    ):
        if env_file and not local_only:
            load_env_file(env_file)
        explicit_key = str(api_key or "").strip()
        resolved_key = explicit_key or os.environ.get("OPENAI_API_KEY", "").strip()
        if not resolved_key:
            raise RuntimeError("OPENAI_API_KEY is not set; export it or pass api_key")
        if base_url:
            urls = [str(base_url).strip().rstrip("/")]
        else:
            explicit_urls = os.environ.get("OPENAI_BASE_URLS", "").strip()
            explicit_url = os.environ.get("OPENAI_BASE_URL", "").strip()
            if explicit_urls:
                urls = [url.strip().rstrip("/") for url in explicit_urls.split(",") if url.strip()]
            elif explicit_url:
                # A singular endpoint is an explicit routing decision. Adding
                # an implicit relay here can multiply every request timeout.
                urls = [explicit_url.rstrip("/")]
            else:
                raise RuntimeError("Set OPENAI_BASE_URL or pass an explicit base_url")
        if local_only:
            if len(urls) != 1 or urlparse(urls[0]).hostname not in {"127.0.0.1", "localhost", "::1"}:
                raise ValueError("local_only requires exactly one loopback base_url")
        timeout = (
            float(timeout)
            if timeout is not None
            else float(os.environ.get("COMPRESSTAB_CLIENT_TIMEOUT", "60"))
        )
        trust_env_proxy = os.environ.get("COMPRESSTAB_TRUST_ENV_PROXY", "").strip().lower() in {
            "1", "true", "yes", "on",
        }
        credentials = [(resolved_key, url) for url in urls]
        self.clients = [
            OpenAI(
                api_key=key,
                base_url=url,
                timeout=timeout,
                max_retries=0,
                # Experiment hosts often inherit a stale localhost proxy. API
                # endpoints are explicit, so ambient proxy routing is opt-in.
                http_client=DefaultHttpxClient(trust_env=trust_env_proxy),
            )
            for key, url in dict.fromkeys(credentials)
        ]
        self.max_completion_tokens = max(1, int(max_completion_tokens))
        self.request_timeout = timeout
        self.disable_thinking = bool(disable_thinking)
        self.json_mode = bool(json_mode)
        self.usage: list[UsageRecord] = []

    def chat_json(self, *, model: str, system_prompt: str, user_prompt: str, temperature: float = 0.0, max_attempts: int = 3) -> tuple[dict[str, Any], str]:
        last_error: Exception | None = None
        last_format_error: LLMResponseFormatError | None = None
        adaptive_prompt = user_prompt
        for attempt in range(max(1, max_attempts)):
            for client in self.clients:
                try:
                    request = {
                        "model": model,
                        "temperature": temperature,
                        "max_tokens": self.max_completion_tokens,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": adaptive_prompt},
                        ],
                        "timeout": getattr(
                            self,
                            "request_timeout",
                            float(os.environ.get("COMPRESSTAB_REQUEST_TIMEOUT", "120")),
                        ),
                    }
                    if getattr(self, "json_mode", True):
                        request["response_format"] = {"type": "json_object"}
                    if getattr(self, "disable_thinking", False):
                        request["extra_body"] = {
                            "chat_template_kwargs": {"enable_thinking": False}
                        }
                    try:
                        response = client.chat.completions.create(**request)
                    except Exception as exc:
                        message = str(exc).lower()
                        if "response_format" not in request or not any(
                            term in message for term in ("response_format", "json mode", "json_object")
                        ):
                            raise
                        # Keep compatibility with relays that do not expose JSON mode.
                        request.pop("response_format")
                        response = client.chat.completions.create(**request)
                    raw = (response.choices[0].message.content or "").strip()
                    try:
                        blob = extract_json_blob(raw)
                        data = parse_json_object(blob)
                    except (ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
                        snippet = raw[:1800] if raw else "<empty response>"
                        last_format_error = LLMResponseFormatError(
                            f"invalid JSON response ({exc}); raw={snippet}"
                        )
                        adaptive_prompt = (
                            user_prompt
                            + "\n\nYour previous response was not valid JSON. Repair only its JSON syntax "
                            + "and return one complete object matching the requested schema. Do not add prose "
                            + "or Markdown. Previous response:\n"
                            + snippet
                        )
                        raise last_format_error from exc
                    estimated_input = approx_token_count(system_prompt) + approx_token_count(adaptive_prompt)
                    estimated_output = approx_token_count(raw)
                    provider_usage = getattr(response, "usage", None)
                    prompt_tokens = getattr(provider_usage, "prompt_tokens", None)
                    completion_tokens = getattr(provider_usage, "completion_tokens", None)
                    has_provider_usage = isinstance(prompt_tokens, int) and isinstance(completion_tokens, int)
                    input_tokens = prompt_tokens if has_provider_usage else estimated_input
                    output_tokens = completion_tokens if has_provider_usage else estimated_output
                    self.usage.append(
                        UsageRecord(
                            model=model,
                            base_url=str(getattr(client, "base_url", "")),
                            input_tokens_est=input_tokens,
                            output_tokens_est=output_tokens,
                            total_tokens_est=input_tokens + output_tokens,
                            measurement="provider" if has_provider_usage else "estimated",
                        )
                    )
                    return data, raw
                except Exception as exc:
                    last_error = exc
                    if not isinstance(exc, LLMResponseFormatError) and not is_transient_api_error(exc):
                        # Permanent 4xx and method errors must fail closed. They
                        # are neither endpoint-selection signals nor evidence
                        # that should alter the next claim-producing prompt.
                        raise
            if attempt + 1 < max_attempts:
                time.sleep(1.0 + attempt)
        if last_format_error is not None:
            raise last_format_error
        assert last_error is not None
        raise last_error

    def token_totals(self) -> dict[str, int]:
        return {
            "input": sum(r.input_tokens_est for r in self.usage),
            "output": sum(r.output_tokens_est for r in self.usage),
            "total": sum(r.total_tokens_est for r in self.usage),
            "calls": len(self.usage),
            "provider_measured_calls": sum(r.measurement == "provider" for r in self.usage),
        }


def load_env_file(path: str | Path) -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def extract_json_blob(text: str) -> str:
    text = (text or "").strip()
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if fenced:
        text = fenced[0].strip()
    start = text.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
    raise ValueError(f"model did not return a JSON object: {text[:300]}")


def parse_json_object(blob: str) -> dict[str, Any]:
    """Parse one model mapping while tolerating common JSON syntax noise."""

    candidates = [blob]
    if "\\'" in blob:
        # Apostrophes do not need escaping inside JSON double-quoted strings.
        candidates.append(blob.replace("\\'", "'"))
    errors: list[Exception] = []
    for candidate in candidates:
        for loader in (json.loads, yaml.safe_load, _literal_eval):
            try:
                value = loader(candidate)
            except (ValueError, SyntaxError, json.JSONDecodeError, yaml.YAMLError) as exc:
                errors.append(exc)
                continue
            if isinstance(value, dict):
                return value
            errors.append(ValueError("JSON-like response is not an object"))
    raise ValueError(str(errors[-1]) if errors else "JSON-like response is not an object")


def _literal_eval(text: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.literal_eval(text)
