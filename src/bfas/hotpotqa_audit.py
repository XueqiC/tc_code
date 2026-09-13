"""Post-episode lexical audit only; never used by the HotpotQA environment.

Adapted from the existing HotpotQA check's tools/hotpotqa_step_audit.py
(mech-hotpotqa worktree). See docs/multihop_evaluation.md for annotation levels.
"""
from __future__ import annotations

import re

VERSION = "hotpotqa-step-audit-v2"
MATCH_RULE = "unicode-word-sequence-casefold-v1"
END_REASONS = (
    "normal_action", "invalid_action", "empty_retrieval", "step_limit_reached",
    "finish_called", "harness_error",
)
FIELD_APPLICABILITY = {
    "all_sets": ["query_previous_observation_overlap (shared strings/count)",
                 "step_end_reason", "episode_end_reason"],
    "sentence": ["supporting_fact_matches", "all_supporting_facts_seen_before_step",
                 "query_previous_observation_overlap.supporting_title_reuse"],
    "paragraph": ["supporting_paragraph_matches", "all_supporting_paragraphs_seen_before_step",
                  "query_previous_observation_overlap.supporting_title_reuse"],
    "none": [],
    "interpretation": "Lexical exposure only, not semantic sufficiency or a required retrieval path; null means unknown/inapplicable, never zero coverage.",
}


def tokens(text):
    return list(re.finditer(r"\w+", text))


def matching_spans(text, needle):
    """Contiguous casefolded Unicode words; offsets index the original string.

    Ignore punctuation/whitespace between words, but never skip a word, stem,
    resolve an alias, or accept a partial word. Keep every occurrence, including
    overlapping ones. Spans are half-open Python character offsets, not bytes;
    leading/trailing punctuation is outside the span.
    """
    haystack, target = tokens(text), [m.group().casefold() for m in tokens(needle)]
    if not target:
        return []
    folded = [m.group().casefold() for m in haystack]
    size = len(target)
    return [{"start": haystack[i].start(), "end": haystack[i + size - 1].end()}
            for i in range(len(haystack) - size + 1) if folded[i:i + size] == target]


def supporting_facts(annotations):
    if annotations.get("annotation_level") == "paragraph":
        return [dict(fact_index=i, title=p["title"], paragraph_index=p["paragraph_index"],
                     text=p["text"] if isinstance(p["text"], str) and tokens(p["text"]) else None,
                     status="available" if isinstance(p["text"], str) and tokens(p["text"]) else "unavailable")
                for i, p in enumerate(annotations.get("supporting_paragraphs", []))]
    pages = dict(annotations.get("context") or [])
    facts = []
    for index, (title, sentence_index) in enumerate(annotations.get("supporting_facts") or []):
        sentences = pages.get(title, [])
        text = (sentences[sentence_index] if type(sentence_index) is int
                and 0 <= sentence_index < len(sentences) else None)
        available = isinstance(text, str) and bool(tokens(text))
        facts.append({"fact_index": index, "title": title, "sentence_index": sentence_index,
                      "text": text if available else None,
                      "status": "available" if available else "unavailable"})
    return facts


def empty_retrieval(observation):
    # No returned page/passage; suggestions can still be useful. This is not a
    # semantic utility verdict and must never establish taxonomy S2 by itself.
    return not observation or observation.startswith((
        "Could not find ", "No more results.", "No matching page or passage was found.",
        "Retrieval disabled.",
    ))


def annotate_episode(record, annotations, parse_action):
    """Attach a parallel step_audit only AFTER the stream and scoring finish.

    No history/transcript/request, outcome, or eligibility fields are modified.
    parse_action is the existing harness parser, passed in to keep this module
    independent of the environment and its prompt builder.
    """
    facts = supporting_facts(annotations)
    complete = bool(facts) and all(f["status"] == "available" for f in facts)
    level = annotations.get("annotation_level", "sentence")
    status = ("unavailable" if not facts or not any(f["status"] == "available" for f in facts)
              else "coarser_than_sentence_level" if level == "paragraph" else "available")
    record["supporting_facts_audit"] = {
        "version": VERSION, "match_rule": MATCH_RULE,
        "status": status, "annotation_level": level, "annotation_complete": complete,
        "facts": facts, "error": annotations.get("audit_error"),
    }
    terminal = ("harness_error" if record.get("error") or record.get("prefix_restored") is False
                else "finish_called" if record.get("finished") else "step_limit_reached")
    record["episode_end_reason"] = terminal
    history, seen, audit = record["history"], set(), []
    for i, h in enumerate(history):
        parsed = parse_action(h.get("action") or "", h["step"])
        observation = h.get("observation")
        last = i == len(history) - 1
        if last and terminal == "harness_error":
            reason = "harness_error"
        elif parsed is None:
            reason = "invalid_action"
        elif parsed[0] == "finish":
            reason = "finish_called"
        elif empty_retrieval(observation):
            reason = "empty_retrieval"
        elif last and terminal == "step_limit_reached":
            reason = "step_limit_reached"
        else:
            reason = "normal_action"
        usable = parsed and parsed[0] in {"search", "lookup"} and reason in {
            "normal_action", "step_limit_reached"}
        matches = []
        for fact in facts:
            spans = matching_spans(observation, fact["text"]) if usable and fact["text"] else []
            matches.append({k: fact[k] for k in ("fact_index", "title", "sentence_index", "paragraph_index") if k in fact} | {
                "status": "unavailable" if fact["text"] is None else "matched" if spans else "no_match",
                "spans": spans,
            })
        previous = history[i - 1] if i else None
        retrieval = bool(parsed and parsed[0] in {"search", "lookup"})
        query = parsed[1] if retrieval else None
        previous_text = previous.get("observation") if previous else None
        overlap_status = ("not_retrieval" if not retrieval else "no_previous_observation" if previous is None
                  else "previous_observation_unavailable" if not isinstance(previous_text, str) else "compared")
        overlap = {"status": overlap_status, "query": query,
                   "previous_step": previous["step"] if previous else None,
                   "shared_strings": [], "shared_query_strings": [], "count": 0,
                   "supporting_title_reuse": [] if facts else None}
        if overlap_status == "compared":
            # Distinct casefolded words, in first-occurrence observation order;
            # preserve the actual surface spelling on each side. No stopwords.
            query_words = {}
            for m in tokens(query):
                query_words.setdefault(m.group().casefold(), m.group())
            shared = set()
            for m in tokens(previous_text):
                word = m.group().casefold()
                if word in query_words and word not in shared:
                    shared.add(word)
                    overlap["shared_strings"].append(m.group())
                    overlap["shared_query_strings"].append(query_words[word])
            overlap["count"] = len(shared)
            # Annotated page titles are an explicit lexical entity candidate
            # inventory, not NER or a claim that this next query was required.
            for title in dict.fromkeys(f["title"] for f in facts):
                visible = matching_spans(previous_text, title)
                if visible:
                    reused = matching_spans(query, title)
                    overlap["supporting_title_reuse"].append({
                        "title": title, "observation_spans": visible, "query_spans": reused,
                        "reused": bool(reused),
                    })
        audit.append({"step": h["step"], "annotation_status": status, "annotation_level": level,
                      "annotation_complete": complete,
                      "supporting_fact_matches": matches if level == "sentence" and facts else None,
                      "supporting_paragraph_matches": matches if level == "paragraph" and facts else None,
                      "query_previous_observation_overlap": overlap,
                      "step_end_reason": reason, "episode_end_reason": terminal,
                      "all_supporting_facts_seen_before_step": len(seen) == len(facts) if complete and level == "sentence" else None,
                      "all_supporting_paragraphs_seen_before_step": len(seen) == len(facts) if complete and level == "paragraph" else None})
        seen.update(m["fact_index"] for m in matches if m["status"] == "matched")
    record["step_audit"] = audit
