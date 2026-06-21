"""
Switchable LLM provider.

Usage:
    from llm import get_llm
    llm = get_llm(temperature=0.2)

The provider is chosen from config.settings.llm_provider ("anthropic" | "openai").
You can override it per-call by passing provider= explicitly.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from langchain_core.language_models import BaseChatModel

from config import settings


def get_llm(
    temperature: float = settings.synthesis_temperature,
    provider: Optional[Literal["anthropic", "openai"]] = None,
    streaming: bool = False,
) -> BaseChatModel:
    """Return a LangChain chat model for the configured (or requested) provider."""
    prov = provider or settings.llm_provider

    if prov == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.anthropic_model,
            anthropic_api_key=settings.anthropic_api_key,
            temperature=temperature,
            streaming=streaming,
            max_tokens=4096,
        )

    elif prov == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            openai_api_key=settings.openai_api_key,
            temperature=temperature,
            streaming=streaming,
        )

    else:
        raise ValueError(f"Unknown LLM provider: {prov!r}. Use 'anthropic' or 'openai'.")


# Synchronous alias — identical signature
get_llm_sync = get_llm


# ── Lightweight factory helpers ──────────────────────────────────────────────

def get_router_llm() -> BaseChatModel:
    """Fast, cheap model for routing/classification decisions."""
    prov = settings.llm_provider
    if prov == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            anthropic_api_key=settings.anthropic_api_key,
            temperature=0,
            max_tokens=512,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=settings.openai_api_key,
            temperature=0,
        )


def get_sql_llm() -> BaseChatModel:
    """Fast lightweight model for NL→SQL translation (structured task, no creativity needed)."""
    prov = settings.llm_provider
    if prov == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            anthropic_api_key=settings.anthropic_api_key,
            temperature=0,
            max_tokens=512,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=settings.openai_api_key,
            temperature=0,
        )


def get_synthesis_llm(streaming: bool = False) -> BaseChatModel:
    """High-quality model for final answer synthesis."""
    return get_llm(temperature=settings.synthesis_temperature, streaming=streaming)
