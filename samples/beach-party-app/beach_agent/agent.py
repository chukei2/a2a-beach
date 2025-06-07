import logging
from collections.abc import AsyncIterable
from typing import Any, Literal

# import httpx # Removed as it's no longer used
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
        logger.info(
            "Initializing BeachAgent using Gemini for internet search ..."
        )
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-04-17"
        )

    async def _search_web(self, query: str) -> str:
        """Retrieve information from the internet using Gemini."""
        logger.info(
            f"Performing web search using Gemini for query: '{query}'"
        )
        try:
            # Prompt for Gemini to perform a web search and summarize findings
            search_prompt = (
                f"""
あなたは、優秀なビーチ情報アシスタントです。以下のクエリに基づいて、インターネットを検索し、関連する情報を収集してください。
クエリ: {query}
あなたの回答は、ビーチに関する情報を提供することに焦点を当ててください。
あまり熟考せず、迅速に検索結果をまとめてください。
                """
            )
            # Use the existing Gemini model to get search-like results
            ai_msg = await self.model.ainvoke(search_prompt)

            if ai_msg.content and isinstance(ai_msg.content, str):
                logger.info(
                    f"Gemini search returned content of length: "
                    f"{len(ai_msg.content)}"
                )
                return ai_msg.content
            else:
                logger.warning(
                    "Gemini search returned no content or content in an "
                    "unexpected format."
                )
                return ""  # Return empty string if no usable content
        except Exception as exc:  # pragma: no cover - network or API errors
            logger.error(f"Gemini-based web search failed: {exc}")
            # Return empty string on failure
            return ""

    async def ainvoke(self, query: str, sessionId: str) -> dict[str, Any]:
        logger.info(
            f"BeachAgent.ainvoke called with query: '{query}', "
            f"sessionId: '{sessionId}'"
        )
        web_results = await self._search_web(query)
        prompt = (
            f"{self.SYSTEM_INSTRUCTION}\n"
            f"Web results:\n{web_results}\n"
            f"Answer the question: {query}"
        )
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
        logger.info(
            f"BeachAgent.stream called with query: '{query}', "
            f"sessionId: '{sessionId}'"
        )
        web_results = await self._search_web(query)
        prompt = (
            f"{self.SYSTEM_INSTRUCTION}\n"
            f"Web results:\n{web_results}\n"
            f"Answer the question: {query}"
        )
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
