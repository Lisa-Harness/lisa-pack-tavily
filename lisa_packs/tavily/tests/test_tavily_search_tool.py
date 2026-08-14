"""Tests for the Tavily provider web_search tool contract."""

from __future__ import annotations

import pytest
from kernel.models import ToolExecutionContext


@pytest.mark.asyncio
async def test_tavily_search_returns_friendly_error_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lisa_packs.tavily.tools.tavily_search import tavily_search as ws

    monkeypatch.setattr(ws, "is_configured", lambda: False)

    ctx = ToolExecutionContext(
        tool_id="web_search",
        parameters={"query": "cloud cost"},
    )
    result = await ws.execute(ctx)

    assert (
        result.error
        == "Tavily search is not configured. Set TAVILY_SEARCH_API_KEY to enable it."
    )


@pytest.mark.asyncio
async def test_tavily_search_returns_error_on_empty_query() -> None:
    from lisa_packs.tavily.tools.tavily_search import tavily_search as ws

    ctx = ToolExecutionContext(tool_id="web_search", parameters={})
    result = await ws.execute(ctx)

    assert result.error == "web_search requires a query."


@pytest.mark.asyncio
async def test_tavily_search_returns_grounded_payload_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lisa_packs.tavily.models import WebSearchResponse, WebSearchResult
    from lisa_packs.tavily.tools.tavily_search import tavily_search as ws

    async def _fake_search(query: str) -> WebSearchResponse:
        return WebSearchResponse(
            query=query,
            generated_at="2026-08-12T00:00:00+00:00",
            results=[
                WebSearchResult(
                    name="Cloud Cost Guardrails",
                    url="https://example.com/guardrails",
                    snippet="Use budgets and alerts for spend guardrails.",
                    display_url="example.com/guardrails",
                )
            ],
        )

    monkeypatch.setattr(ws, "is_configured", lambda: True)
    monkeypatch.setattr(ws, "search", _fake_search)

    ctx = ToolExecutionContext(
        tool_id="web_search",
        parameters={"query": "cloud cost"},
    )
    result = await ws.execute(ctx)

    assert result.error == ""
    assert result.output_json["response"]["query"] == "cloud cost"
    assert (
        result.output_json["response"]["results"][0]["url"]
        == "https://example.com/guardrails"
    )
