import subprocess
import sys
from pathlib import Path

import click

APP_DIR = Path(__file__).resolve().parent


def _spawn(cmd: list[str]) -> subprocess.Popen:
    """Start a subprocess in the sample directory."""
    return subprocess.Popen(cmd, cwd=APP_DIR)


@click.group(help="Utilities for running the beach party sample")
def cli() -> None:
    """Command group for agent management."""
    pass


@cli.command(help="Start only the host agent")
def host() -> None:
    subprocess.run([sys.executable, "host_agent/app.py"], cwd=APP_DIR, check=True)


@cli.command(help="Start the weather agent")
def weather() -> None:
    subprocess.run([sys.executable, "-m", "weather_agent"], cwd=APP_DIR, check=True)


@cli.command(help="Start the beach agent")
def beach() -> None:
    subprocess.run([sys.executable, "-m", "beach_agent"], cwd=APP_DIR, check=True)


@cli.command(name="all", help="Start weather, beach and host agents")
def start_all() -> None:
    processes = [
        _spawn([sys.executable, "-m", "weather_agent"]),
        _spawn([sys.executable, "-m", "beach_agent"]),
    ]
    try:
        subprocess.run([sys.executable, "host_agent/app.py"], cwd=APP_DIR, check=True)
    finally:
        for proc in processes:
            proc.terminate()


if __name__ == "__main__":
    cli()
