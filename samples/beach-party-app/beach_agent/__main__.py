import os
import sys
from typing import Dict, Any
import asyncio
from contextlib import asynccontextmanager

import click
import uvicorn

from .agent import BeachAgent  # Renamed from BeachAgent
from .agent_executor import BeachAgentExecutor  # Renamed from BeachAgentExecutor
from dotenv import load_dotenv

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from a2a.server.tasks import InMemoryTaskStore
from langchain_core.tools import tool

load_dotenv(override=True)

app_context: Dict[str, Any] = {}


@asynccontextmanager
async def app_lifespan(context: Dict[str, Any]):
    """Minimal lifespan manager for the Beach Agent server."""
    context.clear()
    try:
        yield
    finally:
        context.clear()


@click.command()
@click.option(
    "--host", "host", default="localhost", help="Hostname to bind the server to."
)
@click.option(
    "--port", "port", default=10002, type=int, help="Port to bind the server to."
)
@click.option("--log-level", "log_level", default="info", help="Uvicorn log level.")
def cli_main(host: str, port: int, log_level: str):
    """Command Line Interface to start the Beach Agent server."""  # Updated docstring
    if not os.getenv("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    async def run_server_async():
        async with app_lifespan(app_context):
            beach_agent_executor = BeachAgentExecutor()

            request_handler = DefaultRequestHandler(
                agent_executor=beach_agent_executor,  # Renamed variable
                task_store=InMemoryTaskStore(),
            )

            # Create the A2AServer instance
            a2a_server = A2AStarletteApplication(
                agent_card=get_agent_card(host, port), http_handler=request_handler
            )

            # Get the ASGI app from the A2AServer instance
            asgi_app = a2a_server.build()

            config = uvicorn.Config(
                app=asgi_app,
                host=host,
                port=port,
                log_level=log_level.lower(),
                lifespan="auto",
            )

            uvicorn_server = uvicorn.Server(config)

            print(
                f"Starting Uvicorn server at http://{host}:{port} with log-level {log_level}..."
            )
            try:
                await uvicorn_server.serve()
            except KeyboardInterrupt:
                print("Server shutdown requested (KeyboardInterrupt).")
            finally:
                print("Uvicorn server has stopped.")
                # The app_lifespan's finally block handles mcp_client shutdown

    try:
        asyncio.run(run_server_async())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            print(
                "Critical Error: Attempted to nest asyncio.run(). This should have been prevented.",
                file=sys.stderr,
            )
        else:
            print(f"RuntimeError in cli_main: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred in cli_main: {e}", file=sys.stderr)
        sys.exit(1)


def get_agent_card(host: str, port: int):
    """Returns the Agent Card for the Beach Agent."""  # Updated docstring
    capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
    skill = AgentSkill(
        id="beach_search",  # Updated skill id
        name="Search for beaches",  # Updated skill name
        description="Helps with beach search and related questions",  # Updated skill description
        tags=["beach information", "beach search"],  # Updated skill tags
        examples=[
            "Please find a beach in California with good surfing conditions for tomorrow.",  # Updated example
            "What are the amenities at Bondi Beach?",  # Updated example
            "Show me family-friendly beaches near San Diego."  # Updated example
        ],
    )
    return AgentCard(
        name="Beach Agent",  # Updated agent name
        description="Helps with searching for beaches and answering related questions",  # Updated agent description
        url=f"http://{host}:{port}/",
        version="1.0.0",
        defaultInputModes=BeachAgent.SUPPORTED_CONTENT_TYPES,  # Renamed class
        defaultOutputModes=BeachAgent.SUPPORTED_CONTENT_TYPES,  # Renamed class
        capabilities=capabilities,
        skills=[skill],
    )


if __name__ == "__main__":
    cli_main()
