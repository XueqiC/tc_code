"""Local evaluation data for the existing HotpotQA ReAct harness.

No download, training, or teacher dependencies. Scoring fields and annotations
are kept separate; the harness renders only the public question string.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from . import hotpotqa as hp

DATA = hp.ROOT / "envs/multihop/data"
MANIFEST_DIR = hp.ROOT / "configs"
VERSION = "multihop-local-v1"
DATASETS = {
    "2wiki": {"source": "2wiki_dev.json", "split": "dev", "count": 12576, "level": "sentence"},
    "musique": {"source": "musique_validation.json", "split": "validation", "count": 2417, "level": "paragraph"},
    "bamboogle": {"source": "bamboogle_test.json", "split": "test", "count": 125, "level": "none"},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def read_rows(path, *, source_bytes=None):
    """Accept a JSON array or JSON Lines regardless of filename extension."""
    text = (Path(path).read_bytes() if source_bytes is None else source_bytes).decode("utf-8")
    rows = json.loads(text) if text.lstrip().startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or not rows or any(not isinstance(r, dict) for r in rows):
        raise ValueError(f"{path}: expected a nonempty inventory of objects")
    return rows


def normalize_row(dataset, row, index):
    """Return (question/scoring allowlist, audit-only annotations).

    Invalid rows fail the inventory; they are never filtered or replaced.
    Bamboogle has no source ID, so use its zero-based source ordinal, bound to
    the complete source checksum in the frozen manifest.
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown multi-hop dataset: {dataset}")
    if dataset == "bamboogle":
        question = dict(_id=f"bamboogle-{index:06d}", question=row["Question"],
                        answer=row["Answer"], type="multihop")
        annotations = {"annotation_level": "none"}
    elif dataset == "musique":
        if type(row["answerable"]) is not bool:
            raise ValueError("MuSiQue answerable must be a boolean (no default or filtering)")
        aliases = row["answer_aliases"]
        if not isinstance(aliases, list) or any(not isinstance(a, str) for a in aliases):
            raise ValueError("MuSiQue answer_aliases must be a list of strings")
        question = dict(_id=row["id"], question=row["question"], answer=row["answer"],
                        type="multihop", answer_aliases=list(aliases), answerable=row["answerable"])
        paragraphs = []
        for ordinal, paragraph in enumerate(row["paragraphs"]):
            if type(paragraph["is_supporting"]) is not bool:
                raise ValueError("MuSiQue is_supporting must be a boolean")
            if paragraph["is_supporting"]:
                paragraphs.append(dict(title=paragraph["title"], paragraph_index=paragraph.get("idx", ordinal),
                                       text=paragraph["paragraph_text"]))
        annotations = dict(annotation_level="paragraph", supporting_paragraphs=paragraphs)
    else:
        question = {k: row[k] for k in ("_id", "question", "answer", "type")}
        annotations = sentence_annotations(row)
    if any(not isinstance(question[k], str) for k in ("_id", "question", "answer", "type")):
        raise ValueError(f"{dataset} row {index}: ID, question, answer, and type must be strings")
    if not question["_id"] or not question["question"]:
        raise ValueError(f"{dataset} row {index}: empty ID or question")
    return question, annotations


def sentence_annotations(row):
    facts = row.get("supporting_facts", [])
    titles = {title for title, _ in facts}
    return dict(annotation_level="sentence", supporting_facts=facts,
                context=[c for c in row.get("context", []) if c[0] in titles])


def normalize_rows(dataset, rows):
    pairs = [normalize_row(dataset, row, i) for i, row in enumerate(rows)]
    questions = [q for q, _ in pairs]
    ids = [q["_id"] for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{dataset}: duplicate source IDs")
    return questions, {q["_id"]: annotations for q, annotations in pairs}


def manifest_name(dataset):
    return f"multihop_{dataset}_eval_split.json"


def make_manifest(dataset, rows, source_sha256):
    """Freeze the full inventory once, independent of any requested n."""
    spec = DATASETS[dataset]
    questions, _ = normalize_rows(dataset, rows)
    if len(questions) != spec["count"]:
        raise ValueError(f"{dataset}: expected {spec['count']} rows, got {len(questions)}; no rows dropped")
    source_ids = [q["_id"] for q in questions]
    size = len(source_ids)
    return dict(version=VERSION, dataset=dataset, source=spec["source"], split=spec["split"],
                source_count=size, source_sha256=source_sha256, source_ids_sha256=digest(source_ids),
                normalized_rows_sha256=digest(questions), seed=0, sample_size=size,
                selection=f"random.Random(0).sample(source_ids, {size})",
                rule=(f"First n IDs, in stored order, from configs/{manifest_name(dataset)}. "
                      f"That {spec['split']} inventory is random.Random(0).sample(source_ids, {size}). "
                      "No filtering, replacement, or selection by student outcomes."),
                answerable_counts=dict(Counter(str(q.get("answerable", True)).lower() for q in questions)),
                rows_with_aliases=sum(bool(q.get("answer_aliases")) for q in questions),
                excluded_rows=0, ids=random.Random(0).sample(source_ids, size))


def load_dataset(dataset, *, data_dir=DATA, manifest_dir=MANIFEST_DIR):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown multi-hop dataset: {dataset}")
    source = Path(data_dir) / DATASETS[dataset]["source"]
    manifest = json.loads((Path(manifest_dir) / manifest_name(dataset)).read_text())
    source_bytes = source.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != manifest["source_sha256"]:
        raise ValueError(f"{dataset}: source checksum differs from frozen inventory")
    rows = read_rows(source, source_bytes=source_bytes)
    expected = make_manifest(dataset, rows, manifest["source_sha256"])
    if expected != manifest:
        raise ValueError(f"{dataset}: frozen inventory/order or normalization differs")
    questions, annotations = normalize_rows(dataset, rows)
    by_id = {q["_id"]: q for q in questions}
    return [by_id[task_id] for task_id in manifest["ids"]], annotations, manifest


def select_questions(dataset, n=None, *, data_dir=DATA, manifest_dir=MANIFEST_DIR):
    questions, annotations, manifest = load_dataset(dataset, data_dir=data_dir, manifest_dir=manifest_dir)
    n = len(questions) if n is None else n
    if type(n) is not int or not 1 <= n <= len(questions):
        raise ValueError(f"--n must be within 1..{len(questions)}; no filtering, replacement, or resampling")
    questions = questions[:n]
    ids = [q["_id"] for q in questions]
    selection = {k: v for k, v in manifest.items() if k not in {"ids", "selection"}}
    selection.update(task_ids=ids, selected_rows_sha256=digest(questions),
                     selected_answerable_counts=dict(Counter(str(q.get("answerable", True)).lower() for q in questions)))
    return questions, {task_id: annotations[task_id] for task_id in ids}, selection


def hotpotqa_annotations(questions, *, data_dir=None):
    """Read local sentence annotations separately; missing annotations stay unknown."""
    source = (hp.DATA if data_dir is None else Path(data_dir)) / hp.load_manifest("dev")["source"].rsplit("/", 1)[-1]
    try:
        rows = read_rows(source)
    except (OSError, ValueError) as exc:
        return {q["_id"]: dict(annotation_level="sentence", audit_error=f"{type(exc).__name__}: {exc}")
                for q in questions}
    by_id = {q["_id"]: q for q in questions}
    return {r["_id"]: sentence_annotations(r) for r in rows if r.get("_id") in by_id
            and all(r.get(k) == by_id[r["_id"]][k] for k in ("question", "answer", "type"))}
