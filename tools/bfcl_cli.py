#!/usr/bin/env python3
"""Official BFCL CLI with this repository's additional model registrations."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "envs/bfcl/gorilla/berkeley-function-call-leaderboard"))


def main():
    from bfas.bfcl_azure import register_azure_model
    from bfas.bfcl_ollama import register_ollama_models

    register_azure_model()
    register_ollama_models()
    # The historical DeepSeek entry uses the harness's generic OpenAI handler.
    # Supply its Ollama credentials locally, without a paid credential probe.
    if "generate" in sys.argv and "--model" in sys.argv:
        model = sys.argv[sys.argv.index("--model") + 1]
        if "deepseek" in model or model.startswith("gpt-"):
            from bfas.bfcl_teacher import ollama_credentials

            os.environ["OPENAI_BASE_URL"], os.environ["OPENAI_API_KEY"] = ollama_credentials()
    from bfcl_eval.__main__ import cli

    cli()


if __name__ == "__main__":
    main()
