"""Tavily Search tool provider for the web_search contract.

Uses the Tavily REST API (POST https://api.tavily.com/search) with bearer
token authentication.  The response ``results`` array contains objects with
``title``, ``url``, and ``content`` fields that are mapped to the pack's
generic ``WebSearchResult`` model.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import httpx
from kernel.models import ToolExecutionContext, ToolResult
from pydantic import BaseModel, Field

from lisa_packs.tavily.models import WebSearchResponse, WebSearchResult

# A provider JSON payload: string keys with arbitrary values.
_JsonPayload = dict[str, Any]


__all__ = ["Parameters", "execute", "is_configured", "search"]

_DEFAULT_ENDPOINT = "https://api.tavily.com/search"
_DEFAULT_MAX_RESULTS = 5
_DEFAULT_SEARCH_DEPTH = "basic"
_DEFAULT_TOPIC = "general"
_SEARCH_TIMEOUT_SECONDS: float = 15.0


class Parameters(BaseModel):
    """Structured input parameters for the web_search tool."""

    query: str = Field(..., description="The search query to execute.")
    count: int = Field(default=5, ge=1, le=20, description="Max results per query.")


def _get_api_key() -> str:
    return os.environ.get("TAVILY_SEARCH_API_KEY", "").strip()


def _get_endpoint() -> str:
    return os.environ.get("TAVILY_SEARCH_ENDPOINT", _DEFAULT_ENDPOINT).strip() or _DEFAULT_ENDPOINT


def _get_max_results() -> int:
    raw = os.environ.get("TAVILY_SEARCH_MAX_RESULTS", "").strip()
    if not raw:
        return _DEFAULT_MAX_RESULTS
    try:
        return max(1, min(20, int(raw)))
    except ValueError:
        return _DEFAULT_MAX_RESULTS


def _get_search_depth() -> str:
    return (
        os.environ.get("TAVILY_SEARCH_SEARCH_DEPTH", _DEFAULT_SEARCH_DEPTH).strip()
        or _DEFAULT_SEARCH_DEPTH
    )


def _get_topic() -> str:
    return os.environ.get("TAVILY_SEARCH_TOPIC", _DEFAULT_TOPIC).strip() or _DEFAULT_TOPIC


def is_configured() -> bool:
    """Return True when a Tavily search API key is present in the environment."""
    return bool(_get_api_key())


async def search(query: str, client: httpx.AsyncClient | None = None) -> WebSearchResponse:
    """Execute one Tavily Search query and return structured results.

    Raises:
        RuntimeError: when TAVILY_SEARCH_API_KEY is not configured.
        httpx.HTTPStatusError: when the search API returns a non-2xx response.
    """
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError(
            "Tavily search is not configured. Set TAVILY_SEARCH_API_KEY to enable it."
        )

    payload: dict[str, object] = {
        "query": query,
        "max_results": _get_max_results(),
        "search_depth": _get_search_depth(),
        "topic": _get_topic(),
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    async def _do_request(c: httpx.AsyncClient) -> WebSearchResponse:
        response = await c.post(
            _get_endpoint(),
            json=payload,
            headers=headers,
            timeout=_SEARCH_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return _parse_response(query, response.json())

    if client is not None:
        return await _do_request(client)

    async with httpx.AsyncClient() as managed_client:
        return await _do_request(managed_client)


def _parse_response(query: str, data: _JsonPayload) -> WebSearchResponse:
    raw_values: list[_JsonPayload] = data.get("results", [])

    results: list[WebSearchResult] = []
    for item in raw_values:
        url = item.get("url", "")
        if not url:
            continue
        results.append(
            WebSearchResult(
                name=item.get("title", ""),
                url=url,
                snippet=item.get("content", ""),
                display_url=url,
            )
        )

    return WebSearchResponse(
        query=query,
        generated_at=datetime.now(UTC).isoformat(),
        results=results,
    )


async def execute(ctx: ToolExecutionContext) -> ToolResult:
    """Execute the web_search tool — called by the tool registry dispatcher."""
    from pydantic import ValidationError

    try:
        params = Parameters.model_validate(ctx.parameters or {})
        query = params.query.strip()
    except ValidationError:
        query = ""
    if not query:
        query = (ctx.query or "").strip()
    if not query:
        return ToolResult(
            tool_id=ctx.tool_id or "web_search",
            error="web_search requires a query.",
        )

    if not is_configured():
        return ToolResult(
            tool_id=ctx.tool_id or "web_search",
            error="Tavily search is not configured. Set TAVILY_SEARCH_API_KEY to enable it.",
        )

    try:
        response = await search(query)
    except Exception as exc:  # pylint: disable=broad-except  # intentional at tool boundary
        return ToolResult(
            tool_id=ctx.tool_id or "web_search",
            error=f"Tavily search failed: {type(exc).__name__}: {exc}",
        )

    return ToolResult(
        tool_id=ctx.tool_id or "web_search",
        output_text=render_search_results(response),
        output_json={"response": response.model_dump(mode="json")},
    )


def render_search_results(response: WebSearchResponse) -> str:
    """Render a WebSearchResponse as a text block for model consumption."""
    if not response.results:
        return f"No web results found for: {response.query}"

    lines: list[str] = []
    for i, result in enumerate(response.results, 1):
        lines.append(f"{i}. {result.name}")
        lines.append(f"   {result.display_url or result.url}")
        if result.snippet:
            lines.append(f"   {result.snippet}")
        lines.append("")
    lines.append(f"Tavily Search | {len(response.results)} result(s)")
    return "\n".join(lines)
