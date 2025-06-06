import logging
from collections.abc import AsyncIterable
from typing import Any, Literal

import httpx
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)


class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str


class BeachAgent:
    """Beach Search Agent that uses Gemini for web searches."""

    SYSTEM_INSTRUCTION = (
        "You are a specialized assistant for beach information. "
        "Use the provided web search results to respond accurately."
    )

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self) -> None:
        logger.info("Initializing BeachAgent using Gemini for internet search ...")
        self.model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17")

    async def _search_web(self, query: str) -> str:
        """Retrieve information from the internet using DuckDuckGo."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://api.duckduckgo.com/",
                    params={"q": query, "format": "json", "no_redirect": 1},
                )
                resp.raise_for_status()
                data = resp.json()
            snippets = []
            if data.get("AbstractText"):
                snippets.append(data["AbstractText"])
            for topic in data.get("RelatedTopics", [])[:3]:
                if isinstance(topic, dict) and topic.get("Text"):
                    snippets.append(topic["Text"])
            return "\n".join(snippets)
        except Exception as exc:  # pragma: no cover - network errors
            logger.error(f"Web search failed: {exc}")
            return ""

    async def ainvoke(self, query: str, sessionId: str) -> dict[str, Any]:
        logger.info(f"BeachAgent.ainvoke called with query: '{query}', sessionId: '{sessionId}'")
        web_results = await self._search_web(query)
        prompt = f"{self.SYSTEM_INSTRUCTION}\nWeb results:\n{web_results}\nAnswer the question: {query}"
        try:
            ai_msg = await self.model.ainvoke(prompt)
            return {
                "is_task_complete": True,
                "require_user_input": False,
                "content": ai_msg.content,
            }
        except Exception as exc:  # pragma: no cover - API errors
            logger.error(f"Gemini invocation failed: {exc}")
            return {
                "is_task_complete": True,
                "require_user_input": False,
                "content": "Failed to retrieve information.",
            }

    async def stream(self, query: str, sessionId: str) -> AsyncIterable[Any]:
        logger.info(f"BeachAgent.stream called with query: '{query}', sessionId: '{sessionId}'")
        web_results = await self._search_web(query)
        prompt = f"{self.SYSTEM_INSTRUCTION}\nWeb results:\n{web_results}\nAnswer the question: {query}"
        try:
            async for chunk in self.model.astream(prompt):
                if chunk.content:
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": chunk.content,
                    }
            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": "",
            }
        except Exception as exc:  # pragma: no cover - API errors
            logger.error(f"Error during streaming: {exc}")
            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": "Streaming error occurred.",
            }
