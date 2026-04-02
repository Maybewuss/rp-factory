"""统一 LLM 调用层 — 封装 OpenAI-compatible API 调用。"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from rp_factory.config import LLMEndpoint

logger = logging.getLogger(__name__)


class LLMClient:
    """对单个 LLM 端点的异步调用封装。"""

    def __init__(self, endpoint: LLMEndpoint) -> None:
        self.endpoint = endpoint
        self._client = AsyncOpenAI(
            api_key=endpoint.api_key or "placeholder",
            base_url=endpoint.base_url,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=16))
    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        n: int = 1,
    ) -> list[dict[str, Any]]:
        """发送 chat completion 请求，返回 n 条回复。

        每条回复为 ``{"content": str, "reasoning_content": str | None}``。
        """
        temp = temperature if temperature is not None else self.endpoint.temperature
        mt = max_tokens if max_tokens is not None else self.endpoint.max_tokens

        resp = await self._client.chat.completions.create(
            model=self.endpoint.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temp,
            max_tokens=mt,
            n=n,
        )

        results: list[dict[str, Any]] = []
        for choice in resp.choices:
            msg = choice.message
            results.append({
                "content": msg.content or "",
                "reasoning_content": getattr(msg, "reasoning_content", None),
            })
        return results

    async def chat_single(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """便捷方法：只取第一条回复。"""
        results = await self.chat(messages, n=1, **kwargs)
        return results[0]

    async def chat_parallel(
        self,
        messages: list[dict[str, str]],
        n: int,
        max_concurrent: int = 4,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """并发发送 n 次独立请求（适用于 Best-of-N 采样）。"""
        sem = asyncio.Semaphore(max_concurrent)

        async def _one() -> dict[str, Any]:
            async with sem:
                return await self.chat_single(messages, **kwargs)

        return await asyncio.gather(*[_one() for _ in range(n)])


def parse_json_response(text: str) -> dict[str, Any]:
    """尝试从 LLM 返回文本中提取 JSON 对象。"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # skip opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return json.loads(text)
