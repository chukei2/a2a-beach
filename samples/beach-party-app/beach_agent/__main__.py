import os
import sys
from typing import Dict, Any, List
import asyncio
from contextlib import asynccontextmanager

import click
import uvicorn

from agent import BeachAgent  # Renamed from BeachAgent
from agent_executor import BeachAgentExecutor  # Renamed from BeachAgentExecutor
from dotenv import load_dotenv

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from a2a.server.tasks import InMemoryTaskStore
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import tool

load_dotenv(override=True)

SERVER_CONFIGS = {
    "bnb": {
        "command": "npx",
        "args": ["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"],
        "transport": "stdio",
    },
}

# Fallback data for local beach search when MCP tools are unavailable
LOCAL_BEACHES = [
    {
        "name": "Bondi Beach",
        "location": "Sydney, Australia",
        "features": "surfing, amenities, family friendly",
    },
    {
        "name": "Waikiki Beach",
        "location": "Honolulu, Hawaii",
        "features": "resort area, gentle waves",
    },
    {
        "name": "Santa Monica Beach",
        "location": "California, USA",
        "features": "pier, boardwalk, great for sunset",
    },
]


@tool
def local_beach_search(query: str) -> str:
    """Fallback beach search implemented locally.

    Args:
        query: Free form search string describing the desired beach or location.

    Returns:
        Textual description of matching beaches from a small local dataset.
    """
    query_lower = query.lower()
    matches = [
        b
        for b in LOCAL_BEACHES
        if query_lower in f"{b['name']} {b['location']} {b['features']}".lower()
    ]
    if not matches:
        return "No local beach information found for your query."
    return "\n".join(
        f"{b['name']} ({b['location']}) - {b['features']}" for b in matches
    )

app_context: Dict[str, Any] = {}


@asynccontextmanager
async def app_lifespan(context: Dict[str, Any]):
    """Manages the lifecycle of shared resources like the MCP client and tools."""
    print("Lifespan: Initializing MCP client and tools...")

    # This variable will hold the MultiServerMCPClient instance
    mcp_client_instance: MultiServerMCPClient | None = None
    mcp_tools: List[Any] = []

    try:
        mcp_client_instance = MultiServerMCPClient(SERVER_CONFIGS)
        mcp_tools = await mcp_client_instance.get_tools()
        if not mcp_tools:
            raise ValueError("No MCP tools returned")
        context["using_fallback_tools"] = False
        print(
            f"Lifespan: MCP Tools preloaded successfully ({len(mcp_tools)} tools found)."
        )
    except Exception as e:
        print(
            f"Lifespan: Failed to load MCP tools: {e}. Falling back to local search.",
            file=sys.stderr,
        )
        if mcp_client_instance and hasattr(mcp_client_instance, "__aexit__"):
            try:
                await mcp_client_instance.__aexit__(None, None, None)
            except Exception:
                pass
        mcp_client_instance = None
        mcp_tools = [local_beach_search]
        context["using_fallback_tools"] = True

    context["mcp_tools"] = mcp_tools

    try:
        yield  # Application runs here
    finally:
        print("Lifespan: Shutting down MCP client...")
        if (
            mcp_client_instance
        ):  # Check if the MultiServerMCPClient instance was created
            # The original code called __aexit__ on the MultiServerMCPClient instance
            # (which was mcp_client_manager). We assume this is still the correct cleanup method.
            if hasattr(mcp_client_instance, "__aexit__"):
                try:
                    print(
                        f"Lifespan: Calling __aexit__ on {type(mcp_client_instance).__name__} instance..."
                    )
                    await mcp_client_instance.__aexit__(None, None, None)
                    print("Lifespan: MCP Client resources released via __aexit__.")
                except Exception as e:
                    print(
                        f"Lifespan: Error during MCP client __aexit__: {e}",
                        file=sys.stderr,
                    )
            else:
                # This would be unexpected if only the context manager usage changed.
                # Log an error as this could lead to resource leaks.
                print(
                    f"Lifespan: CRITICAL - {type(mcp_client_instance).__name__} instance does not have __aexit__ method for cleanup. Resource leak possible.",
                    file=sys.stderr,
                )
        else:
            # This case means MultiServerMCPClient() constructor likely failed or was not reached.
            print(
                "Lifespan: MCP Client instance was not created, no shutdown attempt via __aexit__."
            )

        # Clear the application context as in the original code.
        print("Lifespan: Clearing application context.")
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
            if app_context.get("using_fallback_tools"):
                print(
                    "Warning: MCP tools unavailable. Using local beach search tool.",
                    file=sys.stderr,
                )
            
            # Initialize BeachAgentExecutor with preloaded tools # Updated comment
            beach_agent_executor = BeachAgentExecutor(  # Renamed variable
                mcp_tools=app_context.get("mcp_tools", [])
            )

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
