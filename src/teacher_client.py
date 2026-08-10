"""Generate verified teacher trajectories with the Ollama cloud chat API."""
import argparse
import json
import re
import signal
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent.parent
SECRETS_FILE = Path("/home/xueqi/hq/secrets/llm_apis.env")
API_URL = "https://ollama.com/api/chat"
DEFAULT_LIMIT = 100
DEFAULT_RPM = 20
MAX_RETRIES = 5
REQUEST_TIMEOUT = 180

DOMAIN_INSTRUCTIONS = {
    "gsm8k-code": (
        "Solve by writing a Python function solution() that DERIVES the numeric "
        "answer step by step: one intermediate variable per arithmetic step, "
        "then return the final variable. Do NOT precompute the answer in your "
        "head — no bare `return <number>`. Output ONLY the code."
    ),
    "pandas": "Solve with a Python/pandas code snippet. Output ONLY the code.",
    "sql": "Answer with a single SQL query. Output ONLY the SQL.",
}


class ExecutionTimeout(TimeoutError):
    pass


class DiscardOutput:
    def write(self, text):
        return len(text)

    def flush(self):
        pass


class RateLimiter:
    """Space request starts evenly so their rate never exceeds the RPM target."""

    def __init__(self, rpm):
        self.interval = 60.0 / rpm
        self.next_request = 0.0

    def wait(self):
        now = time.monotonic()
        delay = self.next_request - now
        if delay > 0:
            time.sleep(delay)
        started = time.monotonic()
        self.next_request = started + self.interval


def load_api_key(path=SECRETS_FILE):
    path = Path(path)
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise RuntimeError(f"could not read API secrets file: {path}") from exc

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() != "OLLAMA_API_KEY":
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not value:
            break
        return value
    raise RuntimeError(f"OLLAMA_API_KEY is missing from {path}")


def sanitize_model(model):
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("._-")
    if not sanitized:
        raise ValueError("model name has no usable filename characters")
    return sanitized


def extract_code(text):
    fenced = re.search(
        r"(?P<fence>`{3,}|~{3,})[^\r\n]*\r?\n(?P<body>.*?)(?P=fence)",
        text,
        flags=re.DOTALL,
    )
    if fenced:
        return fenced.group("body").strip()

    inline_fenced = re.search(r"```(.*?)```", text, flags=re.DOTALL)
    if inline_fenced:
        body = inline_fenced.group(1).strip()
        lines = body.splitlines()
        if lines and lines[0].strip().lower() in {"python", "py", "sql"}:
            body = "\n".join(lines[1:]).strip()
        return body
    return text.strip()


def _alarm_handler(*_):
    raise ExecutionTimeout()


def exec_solution(code, timeout=3):
    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout)
    try:
        namespace = {}
        with redirect_stdout(DiscardOutput()), redirect_stderr(DiscardOutput()):
            exec(code, namespace)  # noqa: S102 - required verification
            value = namespace["solution"]()
        return float(value)
    except (Exception, SystemExit, KeyboardInterrupt):
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def is_degenerate_solution(code):
    """A trajectory must carry computation: reject constant-return solutions."""
    if re.search(r"return\s+[-+]?[\d.]+\s*$", code, flags=re.MULTILINE):
        return True
    return len(re.findall(r"^\s*\w+\s*=", code, flags=re.MULTILINE)) < 2


def verify_gsm8k(candidate, reference):
    if is_degenerate_solution(candidate):
        return False
    predicted = exec_solution(candidate, timeout=3)
    gold = exec_solution(reference, timeout=3)
    if predicted is None or gold is None:
        return False
    return abs(predicted - gold) <= 1e-4


def pandas_compiles(code):
    try:
        compile(code, "<teacher-pandas>", "exec")
        return True
    except (SyntaxError, ValueError, TypeError):
        return False


def is_single_sql_statement(code):
    """Check statement count while ignoring semicolons in quotes and comments."""
    count = 0
    has_content = False
    quote = None
    i = 0
    while i < len(code):
        char = code[i]
        following = code[i + 1] if i + 1 < len(code) else ""

        if quote is not None:
            has_content = True
            if char == quote:
                if following == quote:
                    i += 2
                    continue
                quote = None
            i += 1
            continue

        if char == "-" and following == "-":
            newline = code.find("\n", i + 2)
            i = len(code) if newline < 0 else newline + 1
            continue
        if char == "/" and following == "*":
            comment_end = code.find("*/", i + 2)
            if comment_end < 0:
                return False
            i = comment_end + 2
            continue
        if char in {"'", '"', "`"}:
            quote = char
            has_content = True
        elif char == "[":
            quote = "]"
            has_content = True
        elif char == ";":
            if has_content:
                count += 1
                has_content = False
        elif not char.isspace():
            has_content = True
        i += 1

    if quote is not None:
        return False
    if has_content:
        count += 1
    return count == 1


def build_teacher_prompt(domain, prompt):
    return f"{DOMAIN_INSTRUCTIONS[domain]}\n\nProblem:\n{prompt}"


def retry_delay(attempt):
    return min(2 ** (attempt + 1), 60)


def request_teacher(model, prompt, api_key, limiter):
    headers = {"Authorization": f"Bearer {api_key}"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    for attempt in range(MAX_RETRIES + 1):
        limiter.wait()
        retryable = False
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json=body,
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code == 429 or response.status_code >= 500:
                error = f"HTTP {response.status_code} after retries"
                retryable = True
            elif not 200 <= response.status_code < 300:
                return None, f"HTTP {response.status_code}"
            else:
                try:
                    content = response.json()["message"]["content"]
                    if not isinstance(content, str):
                        raise TypeError("message content is not text")
                    return content, None
                except (ValueError, KeyError, TypeError):
                    error = "invalid API response after retries"
                    retryable = True
        except requests.RequestException as exc:
            error = f"request error after retries: {type(exc).__name__}"
            retryable = True

        if not retryable or attempt == MAX_RETRIES:
            return None, error
        time.sleep(retry_delay(attempt))

    return None, "request failed after retries"


def load_seeds(domain, limit):
    path = ROOT / "data" / "pilot" / f"{domain}.jsonl"
    rows = []
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            if len(rows) >= limit:
                break
            try:
                row = json.loads(line)
                if not isinstance(row["prompt"], str) or not isinstance(
                    row["response"], str
                ):
                    raise TypeError
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"invalid seed row at {path}:{line_number}") from exc
            rows.append(row)
    return rows


def load_completed_prompts(path):
    prompts = set()
    if not path.exists():
        return prompts
    with path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row.get("prompt"), str):
                prompts.add(row["prompt"])
    return prompts


def needs_leading_newline(path):
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as f:
        f.seek(-1, 2)
        return f.read(1) != b"\n"


def make_result(domain, prompt, model, response, code, verified, error=None):
    result = {
        "domain": domain,
        "prompt": prompt,
        "teacher": model,
        "response": response,
        "code": code,
        "verified": verified,
        "ts": datetime.now(UTC).isoformat(),
    }
    if error is not None:
        result["error"] = error
    return result


def generate(model, domain, limit, rpm):
    seeds = load_seeds(domain, limit)
    output = ROOT / "data" / "pool_v0" / sanitize_model(model) / f"{domain}.jsonl"
    completed = load_completed_prompts(output)
    pending = []
    for row in seeds:
        if row["prompt"] in completed:
            continue
        pending.append(row)
        completed.add(row["prompt"])
    output.parent.mkdir(parents=True, exist_ok=True)

    api_key = load_api_key() if pending else None
    limiter = RateLimiter(rpm)
    done = verified_count = failed = 0
    leading_newline = needs_leading_newline(output)
    with output.open("a") as f:
        if leading_newline:
            f.write("\n")
        for seed in pending:
            teacher_prompt = build_teacher_prompt(domain, seed["prompt"])
            response, error = request_teacher(model, teacher_prompt, api_key, limiter)
            if error is not None:
                result = make_result(
                    domain, seed["prompt"], model, None, None, None, error
                )
                failed += 1
            else:
                code = extract_code(response)
                row_error = None
                if domain == "gsm8k-code":
                    verified = verify_gsm8k(code, seed["response"])
                    verified_count += int(verified)
                elif domain == "pandas":
                    verified = None
                    if not pandas_compiles(code):
                        row_error = "pandas compile check failed"
                else:
                    verified = None
                    if not is_single_sql_statement(code):
                        row_error = "SQL must be one non-empty statement"
                if row_error is not None:
                    failed += 1
                result = make_result(
                    domain,
                    seed["prompt"],
                    model,
                    response,
                    code,
                    verified,
                    row_error,
                )

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            done += 1
            if done % 10 == 0:
                print(
                    f"done={done} verified={verified_count} failed={failed}",
                    flush=True,
                )

    if done % 10 != 0 or done == 0:
        print(f"done={done} verified={verified_count} failed={failed}", flush=True)


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--domain", required=True, choices=DOMAIN_INSTRUCTIONS)
    parser.add_argument("--limit", type=positive_int, default=DEFAULT_LIMIT)
    parser.add_argument("--rpm", type=positive_int, default=DEFAULT_RPM)
    return parser.parse_args()


def main():
    args = parse_args()
    generate(args.model, args.domain, args.limit, args.rpm)


if __name__ == "__main__":
    main()
