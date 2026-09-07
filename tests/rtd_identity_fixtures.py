"""Real source selectors in small synthetic BFCL checkouts; no code execution."""
from pathlib import Path

from bfas.rtd.identity import EVALUATION_TOOLS
from bfas.rtd.scoring_scope import PYTHON_SCOPES, SHELL


ROOT = Path(__file__).resolve().parents[1]


def tool_content(name):
    return (ROOT/name).read_text() if name in {*PYTHON_SCOPES, SHELL} else 'fixture'


def put_tools(root):
    for name in EVALUATION_TOOLS:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tool_content(name))
