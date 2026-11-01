"""Explicit, non-secret configuration for a user-triggered research API call.

No configuration discovery, credential profiles, dotenv, or connectivity probes.
The caller chooses ``research-api.json``; a missing path means not configured.
"""
from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit

from .domain import ValidationError

RESEARCH_API_KEY_ENV = "RMBFUND_RESEARCH_API_KEY"
DEFAULT_SETTINGS = {
    "protocol": "openai_chat_completions", "base_url": "", "model": "",
    "timeout": 120.0, "max_tokens": 16000,
}


def normalize_research_api_settings(config: Mapping | None = None) -> dict:
    """Validate only the five public settings; a key never belongs in this mapping.

    ``base_url`` is the API base (for example the user's supplied ``.../v1``), or
    the exact ``.../chat/completions`` endpoint. No host or model is guessed.
    """
    if config is None:
        config = {}
    if not isinstance(config, Mapping) or config.keys() - DEFAULT_SETTINGS.keys():
        raise ValidationError("研究API配置只接受protocol/base_url/model/timeout/max_tokens，不保存Key")
    value = {**DEFAULT_SETTINGS, **config}
    if value["protocol"] not in {"openai_chat_completions", "anthropic_messages"}:
        raise ValidationError("研究API协议须明确为openai_chat_completions或anthropic_messages，不自动切换")
    for key in ("base_url", "model"):
        if not isinstance(value[key], str) or len(value[key]) > 2048:
            raise ValidationError(f"研究API的{key}须为有界文本")
        value[key] = value[key].strip()
        if any(ord(char) < 32 for char in value[key]):
            raise ValidationError(f"研究API的{key}不能包含控制字符")
    if value["base_url"]:
        try:
            parts = urlsplit(value["base_url"])
            valid = (parts.scheme in {"https", "http"} and parts.hostname
                     and not parts.username and not parts.password
                     and not parts.query and not parts.fragment)
            _ = parts.port
        except ValueError:
            valid = False
        if not valid:
            raise ValidationError("研究API地址须为不含Key、用户凭据或查询参数的HTTP(S)地址")
        value["base_url"] = value["base_url"].rstrip("/")
    timeout = value["timeout"]
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or timeout <= 0):
        raise ValidationError("研究API timeout须为正的有限秒数")
    value["timeout"] = float(timeout)
    tokens = value["max_tokens"]
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
        raise ValidationError("研究API max_tokens须为正整数")
    return value


def read_research_api_settings(path: str | Path | None = None, *,
                               config: Mapping | None = None) -> dict:
    """Read one explicitly selected non-secret JSON, or normalize a pure mapping.

    This function never reads environment variables and never makes a request.
    With neither argument it returns the unconfigured defaults, not a search of
    the working directory. Saving UI settings is the caller's responsibility.
    """
    if path is not None and config is not None:
        raise ValidationError("研究API配置须明确选择文件或mapping之一")
    if path is None:
        return normalize_research_api_settings(config)
    target = Path(path)
    if target.name != "research-api.json":
        raise ValidationError("只读取明确指定的research-api.json非秘密配置")
    if not target.exists():
        return normalize_research_api_settings()
    try:
        config = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValidationError("research-api.json无法读取为非秘密JSON配置") from exc
    return normalize_research_api_settings(config)


def resolve_research_api_key(*, api_key: str | None = None,
                             environ: Mapping | None = None) -> str:
    """Only this application's explicit injection or its one named environment key."""
    value = ((os.environ if environ is None else environ).get(RESEARCH_API_KEY_ENV, "")
             if api_key is None else api_key)
    if not isinstance(value, str):
        raise ValidationError("研究API Key须由服务端环境变量或本次显式输入提供")
    return value.strip()


def is_configured(settings: Mapping | None, *, api_key: str | None = None,
                  environ: Mapping | None = None) -> bool:
    """Local presence/format check, not a paid API health check."""
    try:
        value = normalize_research_api_settings(settings)
        return bool(value["base_url"] and value["model"]
                    and resolve_research_api_key(api_key=api_key, environ=environ))
    except ValidationError:
        return False
