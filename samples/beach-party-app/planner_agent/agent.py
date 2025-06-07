import json
import sys
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from collections.abc import AsyncGenerator

class PlannerAgent:
    """ planner Agent."""

    def __init__(self):
        """Initialize the planner dialogue model"""
        try:
            with open("config.json") as f:
                config = json.load(f)
            if not os.getenv(config["api_key"]):
                print(f'{config["api_key"]} environment variable not set.')
                sys.exit(1)
            api_key = os.getenv(config["api_key"])

            self.model = ChatOpenAI(
                model=config["model_name"],
                base_url=config["base_url"],
                api_key=api_key,
                temperature=0.7  # Control the generation randomness (0-2, higher values indicate greater randomness)
            )
        except FileNotFoundError:
            print("Error: The configuration file config.json cannot be found.")
            exit()
        except KeyError as e:
            print(f"The configuration file is missing required fields: {e}")
            exit()

    async def stream(self, query: str) -> AsyncGenerator[str, None]:

        """Stream the response of the large model back to the client. """
        try:
            # ビーチパーティーの計画を支援するための会話履歴を初期化
            messages = [
                SystemMessage(
                    content="""
あなたは、ビーチパーティーの計画を支援する専門家です。ユーザーが楽しく、現実的なビーチパーティーを計画できるように、以下の点に注意して回答してください。
- ビーチの選択、アクティビティ、食事、飲み物、装飾、音楽など、パーティーのすべての側面を考慮してください。
- ユーザーの予算、参加者の好み、天候、季節など、現実的な制約を考慮してください。
- 地元の文化や習慣を尊重し、パーティーが楽しく、かつ安全であるように配慮してください。
- ユーザーがパーティーを計画する際に役立つ具体的なアドバイスや提案を提供してください。
                """
                )
            ]

            # Add the user message to the history.
            messages.append(HumanMessage(content=query))

            # Invoke the model in streaming mode to generate a response.
            for chunk in self.model.stream(messages):
                # Return the text content block.
                if hasattr(chunk, 'content') and chunk.content:
                    yield {'content': chunk.content, 'done': False}
            yield {'content': '', 'done': True}

        except Exception as e:
            print(f"error：{str(e)}")
            yield {'content': 'Sorry, an error occurred while processing your request.', 'done': True}


