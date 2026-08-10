"""Resource-limited execution for generated Python solutions."""

import math
import resource
import subprocess
import sys
import tempfile


_WRAPPER = """\
import sys

namespace = {}
exec(compile(sys.stdin.read(), "<solution>", "exec"), namespace)
print(repr(float(namespace["solution"]())))
"""


def run_solution(code: str, timeout_s=3, mem_mb=512) -> float | None:
    """Run ``solution()`` in an isolated, resource-limited subprocess."""
    cpu_s = max(1, math.ceil(timeout_s))
    mem_bytes = int(mem_mb * 1024 * 1024)

    def apply_limits():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))

    try:
        with tempfile.TemporaryDirectory(prefix="tc-verifier-") as temp_cwd:
            result = subprocess.run(
                [sys.executable, "-I", "-c", _WRAPPER],
                input=code,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=temp_cwd,
                timeout=timeout_s,
                preexec_fn=apply_limits,
                check=False,
            )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None

    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None
