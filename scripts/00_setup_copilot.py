"""
Script 00: Get GitHub Copilot API token via gh CLI.
Writes COPILOT_TOKEN to .env. Auto-refreshes (tokens expire ~30 min).
Also provides get_client() used by all other scripts.
"""

import subprocess, json, sys, os
from pathlib import Path

ROOT     = Path(__file__).parent.parent
ENV_PATH = ROOT / ".env"


GITHUB_MODELS_URL = "https://models.inference.ai.azure.com"


def get_copilot_token() -> str:
    """Get GitHub PAT via gh CLI for use with GitHub Models API."""
    try:
        gh_token = subprocess.check_output(
            ["gh", "auth", "token"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: gh CLI not found or not authenticated.")
        print("  Install: https://cli.github.com")
        print("  Auth:    gh auth login")
        sys.exit(1)

    lines = []
    if ENV_PATH.exists():
        lines = [l for l in ENV_PATH.read_text().splitlines()
                 if not l.startswith("COPILOT_TOKEN=")]
    lines.append(f"COPILOT_TOKEN={gh_token}")
    ENV_PATH.write_text("\n".join(lines) + "\n")
    print(f"  Token set: {gh_token[:12]}...  (using GitHub Models API)")
    return gh_token


def get_client():
    """OpenAI-compatible client pointed at GitHub Models API."""
    from dotenv import load_dotenv
    load_dotenv(ENV_PATH)
    token = os.getenv("COPILOT_TOKEN") or get_copilot_token()
    from openai import OpenAI
    return OpenAI(api_key=token, base_url=GITHUB_MODELS_URL)


if __name__ == "__main__":
    print("=== Step 0: Refreshing Copilot Token ===")
    get_copilot_token()
