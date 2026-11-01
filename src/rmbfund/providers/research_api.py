"""Thin HTTP clients for a caller-selected research model API.

The client sends public research messages only. It does not search, fetch source
material, read dotenv files, retry, persist responses, or log credentials.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from ..domain import ValidationError
from ..research_settings import normalize_research_api_settings, resolve_research_api_key

RequestCallable = Callable[..., Any]


class ResearchAPIError(RuntimeError):
    """Safe, short API failure; response bodies are deliberately not retained."""

    def __init__(self, message: str, *, code: str = "api_error", status_code: int | None = None):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True)
class ResearchAPIResponse:
    text: str
    model: str | None
    usage: dict[str, Any]
    request_id: str | None = None


def _endpoint(base_url: str, protocol: str) -> str:
    suffix = "/chat/completions" if protocol == "openai_chat_completions" else "/messages"
    return base_url if base_url.endswith(suffix) else base_url + suffix


def _response_status(response: Any) -> int | None:
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _json(response: Any) -> Mapping:
    try:
        value = response.json()
    except (ValueError, TypeError, AttributeError) as exc:
        raise ResearchAPIError("研究API返回不是JSON", code="invalid_response",
                               status_code=_response_status(response)) from exc
    if not isinstance(value, Mapping):
        raise ResearchAPIError("研究API返回JSON不是对象", code="invalid_response",
                               status_code=_response_status(response))
    return value


def _error_for_status(response: Any) -> ResearchAPIError:
    status = _response_status(response)
    if status in {401, 403}:
        message = "研究API认证失败"
        code = "authentication_failed"
    elif status == 429:
        message = "研究API请求过于频繁"
        code = "rate_limited"
    elif status is not None and status >= 500:
        message = "研究API服务暂时不可用"
        code = "upstream_unavailable"
    else:
        message = "研究API请求被拒绝"
        code = "request_rejected"
    return ResearchAPIError(message, code=code, status_code=status)


def _text_from_openai(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        pieces = []
        for item in value:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                pieces.append(item["text"])
        if pieces:
            return "".join(pieces)
    raise ResearchAPIError("研究API未返回可读的模型文本", code="invalid_response")


def _text_from_anthropic(value: Any) -> str:
    if not isinstance(value, list):
        raise ResearchAPIError("研究API未返回可读的模型内容", code="invalid_response")
    pieces = [item["text"] for item in value
              if isinstance(item, Mapping) and item.get("type") == "text"
              and isinstance(item.get("text"), str)]
    if not pieces:
        raise ResearchAPIError("研究API未返回可读的模型文本", code="invalid_response")
    return "".join(pieces)


def _usage(value: Any) -> dict[str, Any]:
    """Retain only numeric usage fields; never retain response text or metadata."""
    if not isinstance(value, Mapping):
        return {}
    result = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens",
                "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
        item = value.get(key)
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            result[key] = item
    return result


class ResearchAPIClient:
    """One-shot client for one explicitly selected protocol and endpoint."""

    def __init__(self, settings: Mapping, *, api_key: str | None = None,
                 environ: Mapping | None = None):
        try:
            self.settings = normalize_research_api_settings(settings)
            self.api_key = resolve_research_api_key(api_key=api_key, environ=environ)
        except ValidationError as exc:
            raise ResearchAPIError(str(exc), code="unconfigured") from exc
        if not self.settings["base_url"] or not self.settings["model"] or not self.api_key:
            raise ResearchAPIError("研究API未配置完整地址、模型或Key", code="unconfigured")

    def complete(self, messages: list[dict], *, request: RequestCallable | None = None,
                 system: str | None = None) -> ResearchAPIResponse:
        protocol = self.settings["protocol"]
        endpoint = _endpoint(self.settings["base_url"], protocol)
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if protocol == "openai_chat_completions":
            headers["Authorization"] = f"Bearer {self.api_key}"
            body: dict[str, Any] = {
                "model": self.settings["model"],
                "messages": ([{"role": "system", "content": system}] if system else []) + messages,
                "max_tokens": self.settings["max_tokens"],
            }
        elif protocol == "anthropic_messages":
            headers.update({"x-api-key": self.api_key, "anthropic-version": "2023-06-01"})
            body = {"model": self.settings["model"], "messages": messages,
                    "max_tokens": self.settings["max_tokens"]}
            if system:
                body["system"] = system
        else:  # Kept unreachable by settings normalization; useful for future adapters.
            raise ResearchAPIError("研究API协议未适配", code="unsupported_protocol")
        try:
            if request is None:
                # This application's explicit key is the only credential source.
                with httpx.Client(trust_env=False, follow_redirects=False) as client:
                    response = client.request("POST", endpoint, headers=headers, json=body,
                                              timeout=self.settings["timeout"])
            else:
                response = request("POST", endpoint, headers=headers, json=body,
                                   timeout=self.settings["timeout"])
        except httpx.HTTPError as exc:
            raise ResearchAPIError("研究API网络请求失败", code="network_error") from exc
        except ResearchAPIError:
            raise
        except Exception as exc:
            raise ResearchAPIError("研究API请求执行失败", code="request_error") from exc
        status = _response_status(response)
        if status is None or status < 200 or status >= 300:
            raise _error_for_status(response)
        payload = _json(response)
        if protocol == "openai_chat_completions":
            choices = payload.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
                raise ResearchAPIError("研究API缺少choices", code="invalid_response", status_code=status)
            message = choices[0].get("message")
            if not isinstance(message, Mapping):
                raise ResearchAPIError("研究API缺少message", code="invalid_response", status_code=status)
            stop = choices[0].get("finish_reason")
            if stop in {"length", "content_filter", "tool_calls", "function_call"} or message.get("refusal"):
                raise ResearchAPIError("研究API输出被截断、拒绝或请求了未提供的工具", code="incomplete_output")
            text = _text_from_openai(message.get("content"))
        else:
            if payload.get("stop_reason") in {"max_tokens", "refusal", "tool_use", "pause_turn"}:
                raise ResearchAPIError("研究API输出未形成完整研究", code="incomplete_output")
            text = _text_from_anthropic(payload.get("content"))
        model = payload.get("model")
        if not isinstance(model, str) or not model or len(model) > 200 or self.api_key in model:
            model = None
        return ResearchAPIResponse(text=text, model=model, usage=_usage(payload.get("usage")))


def complete_research(*, settings: Mapping, messages: list[dict], api_key: str | None = None,
                      request: RequestCallable | None = None, system: str | None = None,
                      environ: Mapping | None = None) -> ResearchAPIResponse:
    """Convenience function used by the runner and integration callers."""
    return ResearchAPIClient(settings, api_key=api_key, environ=environ).complete(
        messages, request=request, system=system,
    )
