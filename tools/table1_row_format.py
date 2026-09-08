"""Render purchased evidence in the legacy trainer format, without model/API calls.

The audit snapshots remain immutable provenance. BFCL/ALFWorld raw states are
read only for purchased IDs, with file hashes pinned by the frozen audit ledger.
"""
from __future__ import annotations

from copy import deepcopy
import json

from tools import table1_common as io
from tools.bfcl_demo_pool import serialize


ROW_FORMAT = 'legacy-messages-v1'
BANKS = {'bfcl': 'v1_1_bfcl', 'alfworld': 'v1_alfworld_c26'}
TEMPLATE_MARKERS = ('<|im_start|>', '<|im_end|>', '<think>', '</think>')


def render_legacy_row(row, messages, response):
    """Same pseudo-tag serialization as bfcl_demo_pool; template at encode only."""
    if not messages or not isinstance(response, str) or not response:
        raise ValueError('legacy row requires messages and a non-empty response')
    result = deepcopy(row)
    result.update(messages=deepcopy(messages), prompt=serialize(messages), response=response)
    if any(marker in result['prompt'] for marker in TEMPLATE_MARKERS):
        raise ValueError('applied chat template / think marker in legacy prompt')
    if response != row['response']:
        result['token_hint'] = max(len(response) // 4, 1)
    return result


def bfcl_messages(state, task_id, historical):
    """Call the exact old demo producer's official system/function-doc helper.

    task_json sorts object keys. Recover original function key order from the
    sealed generator object or demo's sealed <tools> JSON, then verify the
    definitions against task_json. No chat-template text enters the output.
    """
    from bfcl_eval.model_handler.utils import system_prompt_pre_processing_chat_model

    task = json.loads(state['task_json'])
    messages = json.loads(state['history_json'])
    functions = historical.get('function')
    if functions is None:
        start = state['prompt'].index('<tools>\n') + len('<tools>\n')
        end = state['prompt'].index('\n</tools>', start)
        functions = [json.loads(line) for line in state['prompt'][start:end].splitlines()]
    if functions != task['function'] or messages != task['question']:
        raise ValueError('sealed BFCL state/function definitions disagree')
    return system_prompt_pre_processing_chat_model(messages, deepcopy(functions), task_id)


def bfcl_response(kind, historical, text):
    """Keep demo result bytes; express concrete generated calls as legacy FC JSON.

    Do not serialize checker ground_truth alternatives as argument values or
    synthesize a refusal. The sealed behavior already contains concrete calls.
    """
    if kind == 'demo_attempt':
        answer = historical['result']
        return answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
    if kind != 'generator_item':
        raise ValueError(f'unsupported BFCL package kind: {kind}')
    if not text:
        if historical.get('ground_truth') != []:
            raise ValueError('unexplained empty teacher response')
        return '[]'
    calls, remaining = [], text.strip()
    decoder = json.JSONDecoder()
    while remaining:
        if not remaining.startswith('<tool_call>'):
            raise ValueError('unexpected sealed generator response framing')
        remaining = remaining[len('<tool_call>'):].lstrip()
        call, end = decoder.raw_decode(remaining)
        remaining = remaining[end:].lstrip()
        if not remaining.startswith('</tool_call>'):
            raise ValueError('incomplete sealed generator tool call')
        remaining = remaining[len('</tool_call>'):].lstrip()
        if (set(call) != {'name', 'arguments'} or not isinstance(call['name'], str)
                or not call['name'] or not isinstance(call['arguments'], dict)):
            raise ValueError('invalid sealed generator tool call')
        calls.append({call['name']: json.dumps(call['arguments'], ensure_ascii=False)})
    return json.dumps(calls, ensure_ascii=False)


def read_bank_payload(benchmark, package_id, owned, sources):
    """Authorization and frozen source hash checks precede content use."""
    if package_id not in owned:
        raise PermissionError('cannot reveal an unpurchased package')
    bank = io.ROOT / 'data/rtd' / BANKS[benchmark]
    name = f'data/rtd/{BANKS[benchmark]}/sealed/{package_id}.json'
    # read_sealed also validates the ID before opening its file.
    payload = io.read_sealed(bank, package_id, owned)
    if io.file_hash(io.ROOT / name) != sources[name]:
        raise ValueError('purchased raw snapshot differs from frozen source hash')
    return payload


def render_package(benchmark, snapshot, payload=None):
    package, old_rows = snapshot['package'], snapshot['rows']
    if benchmark == 'appworld':
        # The audit sealed the exact trace prefixes and teacher text already.
        return [render_legacy_row(row, row['messages'], row['response']) for row in old_rows]
    if payload is None:
        raise ValueError('raw sealed package required for BFCL/ALFWorld rendering')
    if (payload['cost'] != package['recorded_cost']
            or payload['provenance']['kind'] != package['kind']):
        raise ValueError('raw package identity/cost mismatch')
    if not old_rows:
        return []  # Failed attempts stay charged and never become positives.
    behaviors, historical = payload['behaviors'], payload['historical_response']
    if len(behaviors) != len(old_rows):
        raise ValueError('sealed behavior/positive row count mismatch')
    rendered = []
    for i, (row, behavior) in enumerate(zip(old_rows, behaviors)):
        if benchmark == 'bfcl':
            if historical['id'] != row['task_id']:
                raise ValueError('BFCL row does not belong to sealed task')
            messages = bfcl_messages(behavior['state'], row['task_id'], historical)
            response = bfcl_response(package['kind'], historical, behavior['text'])
        elif benchmark == 'alfworld':
            turns = historical['demo']['turns']
            if len(turns) != len(old_rows) or historical['task_id'] != row['task_id']:
                raise ValueError('ALFWorld row does not belong to sealed episode')
            # Original command-only context matches the retained command target;
            # the later RTD expert/THOUGHT/ACTION prompt is a different policy.
            messages, response = turns[i]['context'], turns[i]['target']
            if response != behavior['text']:
                raise ValueError('ALFWorld retained target disagrees with sealed behavior')
        else:
            raise ValueError(f'unsupported benchmark: {benchmark}')
        rendered.append(render_legacy_row(row, messages, response))
    return rendered
