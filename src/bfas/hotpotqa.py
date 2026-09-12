"""Shared HotpotQA ReAct protocol, Wikipedia environment, and CPU verifier.

Wikipedia behavior follows ysymyth/ReAct/wikienv.py (MIT); see the attribution
in prompts/hotpotqa_react_LICENSE.txt. Dataset context is never shown to agents.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
from ipaddress import ip_address
import json
import os
from pathlib import Path
import re
import string
import time
from urllib.parse import urlencode, urlsplit

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "envs/hotpotqa/data"
CACHE = ROOT / "envs/hotpotqa/cache"
PROMPT_FILE = ROOT / "prompts/hotpotqa_react_6shot.txt"
MAX_STEPS = 7
MAX_TOKENS = 100  # Original ReAct notebook's visible completion limit.
PROMPT_VERSION = "react6-" + hashlib.sha256(PROMPT_FILE.read_bytes()).hexdigest()[:12]
WIKI_VERSION = "react-html-v1"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def file_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def load_manifest(split):
    if split not in ("train", "dev"):
        raise ValueError("HotpotQA split must be train or dev")
    name = "support" if split == "train" else "eval"
    return json.loads((ROOT / f"configs/hotpotqa_{name}_split.json").read_text())


def load_questions(split, data_dir=DATA):
    manifest = load_manifest(split)
    path = Path(data_dir) / manifest["source"].rsplit("/", 1)[-1]
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run scripts/setup_hotpotqa.sh")
    with path.open(encoding="utf-8") as stream:
        rows = json.load(stream)
    ids = [r["_id"] for r in rows]
    digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    if digest != manifest["source_ids_sha256"] or len(set(ids)) != len(ids):
        raise ValueError(f"{path}: question inventory/order differs from frozen split")
    selected = set(manifest["ids"])
    by_id = {r["_id"]: {k: r[k] for k in ("_id", "question", "answer", "type")}
             for r in rows if r["_id"] in selected}
    return [by_id[task_id] for task_id in manifest["ids"]]


def normalize_answer(text):
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_metrics(prediction, gold):
    """Official HotpotQA answer EM/F1, including yes/no/noanswer handling."""
    pred, target = normalize_answer(prediction), normalize_answer(gold)
    em = float(pred == target)
    if pred != target and ({pred, target} & {"yes", "no", "noanswer"}):
        return {"em": em, "f1": 0.0}
    p, t = pred.split(), target.split()
    common = sum((Counter(p) & Counter(t)).values())
    f1 = 2 * common / (len(p) + len(t)) if common else 0.0
    return {"em": em, "f1": f1}


def parse_action(text, step=None):
    """Accept one standalone action, or one numbered ReAct Action line.

    Reject ambiguous/multiple actions and fabricated observations. Never search
    inside a Thought line for a tool invocation.
    """
    if re.search(r"(?im)^\s*Observation\s*\d*\s*:", text):
        return None
    lines = text.strip().splitlines()
    actions = []
    for line in lines:
        match = re.fullmatch(r"\s*(?:Action\s*(\d*)\s*:\s*)?(search|lookup|finish)\[([^\n]*)\]\s*", line, re.I)
        if match:
            number, verb, argument = match.groups()
            if number and step is not None and int(number) != step:
                return None
            # Nested brackets are legitimate Wikipedia queries (disambiguation).
            if "][" in argument or re.search(r"\]\s*(?:search|lookup|finish)\[", argument, re.I):
                return None
            actions.append((verb.lower(), argument.strip()))
    return actions[0] if len(actions) == 1 else None


def build_messages(question, transcript="", step=1, *, action_thought=None):
    text = PROMPT_FILE.read_text(encoding="utf-8") + f"Question: {question}\n" + transcript
    text += f"Thought {step}:"
    if action_thought is not None:
        text += f" {action_thought}\nAction {step}:"
    return [{"role": "user", "content": text}]


class OfflineCacheMiss(RuntimeError):
    record = None


class WikiError(RuntimeError):
    pass


class Wikipedia:
    """ReAct's HTTPS search wrapper with immutable per-query disk snapshots.

    Each episode owns its cursor; workers share only the locked cache files.
    A failed search preserves the previous page, matching the original wrapper.
    """

    def __init__(self, cache_dir=CACHE, *, offline=False, timeout=20, retries=2, fetch=None):
        self.cache_dir = Path(cache_dir)
        self.offline, self.timeout, self.retries = offline, timeout, retries
        if timeout <= 0 or retries < 0:
            raise ValueError("Wikipedia timeout must be positive and retries nonnegative")
        self.fetch = fetch or self._fetch
        self.reset()

    def reset(self):
        self.page = None
        self.keyword = None
        self.matches = []
        self.cursor = 0
        self.queries = []

    @staticmethod
    def _fetch(query, timeout):
        import requests
        response = requests.get("https://en.wikipedia.org/w/index.php",
                                params={"search": query}, timeout=timeout,
                                headers={"User-Agent": "BFAS-HotpotQA/1.0 (ReAct research reproduction)"})
        response.raise_for_status()
        return response.text

    @staticmethod
    def _parse(html):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        results = soup.find_all("div", class_="mw-search-result-heading")
        if results:
            return {"page": None, "titles": [p.get_text().strip() for p in results][:5]}
        paragraphs = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
        if any("may refer to:" in p for p in paragraphs):
            return {"disambiguation": True}
        page = "".join(p + ("\n" if not p.endswith("\n") else "")
                       for p in paragraphs if len(p.split(" ")) > 2)
        if not page:
            raise WikiError("Wikipedia returned no article or search results; response was not cached")
        return {"page": page, "titles": []}

    def _snapshot(self, query):
        key = hashlib.sha256(query.encode("utf-8")).hexdigest()
        path = self.cache_dir / f"{key}.json"
        with file_lock(path.with_suffix(".lock")):
            if path.exists():
                record = json.loads(path.read_text(encoding="utf-8"))
                if record.get("query") != query or record.get("version") != WIKI_VERSION:
                    raise WikiError(f"Wikipedia cache identity mismatch: {path}")
                actual = hashlib.sha256(json.dumps(record["result"], sort_keys=True).encode()).hexdigest()
                if actual != record.get("sha256"):
                    raise WikiError(f"Wikipedia cache checksum mismatch: {path}")
            else:
                if self.offline:
                    raise OfflineCacheMiss(f"Wikipedia query {query!r} is not cached at {path} (--offline)")
                for attempt in range(self.retries + 1):
                    try:
                        result = self._parse(self.fetch(query, self.timeout))
                        break
                    except Exception as exc:
                        if attempt == self.retries:
                            raise WikiError(f"Wikipedia query {query!r} failed after {attempt + 1} requests: {exc}") from exc
                        time.sleep(min(2 ** attempt, 4))
                record = {"version": WIKI_VERSION, "query": query, "result": result,
                          "url": "https://en.wikipedia.org/w/index.php?" + urlencode({"search": query}),
                          "sha256": hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()}
                write_json(path, record)
        self.queries.append({"query": query, "sha256": record["sha256"]})
        return record["result"]

    @staticmethod
    def sentences(page):
        return [s.strip() + "." for p in page.split("\n") if p.strip()
                for s in p.strip().split(". ") if s.strip()]

    def search(self, entity):
        query = entity
        for _ in range(3):  # Bound the original wrapper's recursive disambiguation.
            result = self._snapshot(query)
            if result.get("disambiguation"):
                query = "[" + query + "]"
                continue
            if result["page"] is None:
                return f"Could not find {query}. Similar: {result['titles']}."
            self.page = result["page"]
            self.keyword, self.matches, self.cursor = None, [], 0
            return " ".join(self.sentences(self.page)[:5])
        raise WikiError(f"Wikipedia disambiguation loop for {entity!r}")

    def lookup(self, keyword):
        if keyword != self.keyword:
            self.keyword, self.cursor = keyword, 0
            self.matches = [s for s in self.sentences(self.page or "") if keyword.lower() in s.lower()]
        if self.cursor >= len(self.matches):
            return "No more results.\n"
        self.cursor += 1
        return f"(Result {self.cursor} / {len(self.matches)}) " + self.matches[self.cursor - 1]


def episode_stream(question, wiki, *, temperature=0.0):
    """Yield model requests and tool calls; drivers send their results back.

    The original notebook retries missing action format once, without consuming
    another environment step. Both calls are retained for accounting/training.
    """
    wiki.reset()
    transcript, prediction, error = "", "", None
    history, calls, steps, badcalls, finished = [], 0, 0, 0, False
    offline_miss = None
    try:
        for step in range(1, MAX_STEPS + 1):
            entry = {"step": step, "thought": "", "response": None, "fallback_response": None,
                     "action": None, "observation": None}
            history.append(entry)
            messages = build_messages(question["question"], transcript, step)
            calls += 1
            reply = yield ("generate", messages, [f"\nObservation {step}:"], temperature)
            entry["response"] = reply
            action = parse_action(reply, step)
            thought_text = re.split(r"(?im)^\s*Action\s*\d*\s*:", reply, maxsplit=1)[0]
            if action is None or not re.search(r"(?im)^\s*Action\s*\d*\s*:", reply):
                thought_text = reply.split("\n")[0] if action is None else ""
            thought = re.sub(r"^\s*Thought\s*\d*:\s*", "", thought_text).strip()
            entry["thought"] = thought
            fallback_reply = None
            if action is None:
                badcalls += 1
                calls += 1
                fallback_reply = yield ("generate", build_messages(question["question"], transcript, step,
                                                          action_thought=thought), ["\n"], temperature)
                entry["fallback_response"] = fallback_reply
                action = parse_action(fallback_reply, step)
            steps = step
            if action is None:
                observation = "Invalid action: " + (fallback_reply or reply)
                action_text = fallback_reply or reply
            else:
                verb, argument = action
                action_text = f"{verb}[{argument}]"
                entry["action"] = action_text
                if verb == "finish":
                    prediction, finished = argument, True
                    observation = "Episode finished."
                else:
                    observation = yield ("tool", verb, argument)
            entry.update(action=action_text, observation=observation)
            transcript += f"Thought {step}: {thought}\nAction {step}: {action_text}\nObservation {step}: {observation}\n"
            if finished:
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, OfflineCacheMiss):
            offline_miss = exc
        entry["observation"] = error
        transcript += (f"Thought {entry['step']}: {entry['thought']}\n"
                       f"Action {entry['step']}: {entry['action'] or ''}\n"
                       f"Observation {entry['step']}: {error}\n")
    scores = answer_metrics(prediction, question["answer"]) if finished else {"em": 0.0, "f1": 0.0}
    record = {"task_id": question["_id"], "question": question["question"], "gold": question["answer"],
            "category": question["type"], "prediction": prediction, **scores,
            "verified": scores["em"] == 1.0, "checker_verified": scores["em"] == 1.0,
            "steps": steps, "model_calls": calls, "format_failures": badcalls,
            "finished": finished, "error": error, "history": history, "transcript": transcript,
            "termination_reason": ("offline_cache_miss" if offline_miss else "error" if error else
                                   "finish" if finished else "step_limit"),
            "wiki_queries": list(wiki.queries), "prompt_version": PROMPT_VERSION}
    if offline_miss is not None:
        offline_miss.record = record
        raise offline_miss
    return record


def run_episode(question, wiki, generate, *, temperature=0.0):
    """Synchronous adapter driver for the shared episode state machine."""
    stream = episode_stream(question, wiki, temperature=temperature)
    try:
        event = next(stream)
        while True:
            try:
                value = (generate(*event[1:]) if event[0] == "generate"
                         else getattr(wiki, event[1])(event[2]))
            except Exception as exc:
                event = stream.throw(exc)
            else:
                event = stream.send(value)
    except StopIteration as done:
        return done.value
    finally:
        stream.close()


def make_client(base_url):
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY", "").strip() or os.environ.get("BFAS_STUDENT_API_KEY", "").strip()
    if not api_key:
        host = urlsplit(base_url).hostname or ""
        try:
            local = ip_address(host).is_private
        except ValueError:
            local = host == "localhost" or host.endswith(".localhost")
        if not local:
            raise ValueError("OPENAI_API_KEY or BFAS_STUDENT_API_KEY is required for non-local servers")
        api_key = "EMPTY"  # OpenAI-compatible local servers need no credential.
    return OpenAI(base_url=base_url, api_key=api_key, timeout=120, max_retries=0)


def student_generator(client, model):
    luna = model.startswith("gpt-5.6-luna")
    service_tier = os.environ.get("BFAS_OPENAI_SERVICE_TIER", "").strip() or None
    if luna and service_tier not in {None, "flex", "priority"}:
        raise ValueError("BFAS_OPENAI_SERVICE_TIER must be flex, priority, or unset")

    def generate(messages, stop, temperature):
        if luna:
            options = {"max_completion_tokens": MAX_TOKENS}
            if service_tier is not None:
                options["service_tier"] = service_tier
        else:
            options = {"temperature": temperature, "max_tokens": MAX_TOKENS, "stop": stop}
        response = client.chat.completions.create(model=model, messages=messages, **options)
        reply = response.choices[0].message.content or ""
        if luna:
            for marker in stop:
                reply = reply.split(marker, 1)[0]
        return reply
    return generate


def compute_metrics(records, config):
    records = list(records)
    n = len(records)
    mean = lambda key: sum(r[key] for r in records) / n if n else 0.0
    return {"n": n, "em": mean("em"), "f1": mean("f1"), "mean_steps": mean("steps"),
            "headline": mean("em"), "mean_score": mean("em"),
            "errored_episodes": sum(bool(r["error"]) for r in records),
            "per_category": {c: sum(r["em"] for r in records if r["category"] == c)
                             / sum(r["category"] == c for r in records)
                             for c in sorted({r["category"] for r in records})},
            "task_ids": [r["task_id"] for r in records], "config": config}
