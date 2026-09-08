"""Render purchased evidence for the trainer, without model/API calls.

The audit snapshots remain immutable provenance. BFCL/ALFWorld raw states are
read only for purchased IDs, with file hashes pinned by the frozen audit ledger.
"""
from __future__ import annotations

from copy import deepcopy
import json

from tools import table1_common as io
from tools.bfcl_demo_pool import serialize


LEGACY_ROW_FORMAT = 'legacy-messages-v1'
NATIVE_ROW_FORMAT = 'native-fc'
ROW_FORMATS = (NATIVE_ROW_FORMAT, LEGACY_ROW_FORMAT)
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


def row_format_for(benchmark, row_format=None):
    row_format = row_format or (NATIVE_ROW_FORMAT if benchmark == 'bfcl' else LEGACY_ROW_FORMAT)
    if row_format not in ROW_FORMATS or (row_format == NATIVE_ROW_FORMAT and benchmark != 'bfcl'):
        raise ValueError(f'unsupported row format for {benchmark}: {row_format}')
    return row_format


def bfcl_inputs(state, historical):
    """Recover the ordered, purchased handler inputs; never re-sort tool keys.

    task_json sorts object keys. Recover original function key order from the
    sealed generator object or demo's sealed <tools> JSON, then verify the
    definitions against task_json. This also preserves the deployment bytes.
    """
    task = json.loads(state['task_json'])
    messages = json.loads(state['history_json'])
    functions = historical.get('function')
    if functions is None:
        start = state['prompt'].index('<tools>\n') + len('<tools>\n')
        end = state['prompt'].index('\n</tools>', start)
        functions = [json.loads(line) for line in state['prompt'][start:end].splitlines()]
    if functions != task['function'] or messages != task['question']:
        raise ValueError('sealed BFCL state/function definitions disagree')
    return messages, functions


def bfcl_messages(state, task_id, historical):
    """Call the exact old demo producer's official system/function-doc helper."""
    from bfcl_eval.model_handler.utils import system_prompt_pre_processing_chat_model

    messages, functions = bfcl_inputs(state, historical)
    return system_prompt_pre_processing_chat_model(messages, deepcopy(functions), task_id)


def bfcl_native_prompt(state, historical):
    """Verify the sealed RTD state against the actual FC non-thinking renderer.

    __new__ avoids OSSHandler.__init__, which constructs an API client. The
    formatting method is pure Python and does not use instance configuration.
    As in rtd.bank.full_state / rtd.return_gradient.bfcl_task_rollout, complete
    the handler's assistant marker with the existing thinking_off helper.
    """
    from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
    from bfas.cc_pairs import thinking_off

    messages, functions = bfcl_inputs(state, historical)
    handler = QwenFCHandler.__new__(QwenFCHandler)
    prompt = thinking_off(handler._format_prompt(messages, functions))
    if prompt != state['prompt']:
        raise ValueError('sealed BFCL prompt differs from native FC deployment rendering')
    return prompt


def bfcl_native_rejected(response):
    """Convert cached decoded FC lists; keep native emissions / plain text.

    No new student sampling or repair of malformed failed calls. Strip only
    terminal chat EOS markers, since encode() adds the termination token.
    """
    from bfas.rtd.bank import _response_text

    if isinstance(response, str):
        while response.endswith(('<|im_end|>', '<|endoftext|>')):
            response = response.rsplit('<|', 1)[0]
        try:
            decoded = json.loads(response)
        except json.JSONDecodeError:
            return response
        if not isinstance(decoded, list):
            return response
        response = decoded
    if isinstance(response, list) and response and all(
            isinstance(call, dict) and set(call) == {'name', 'arguments'} for call in response):
        response = [{call['name']: call['arguments']} for call in response]
    return _response_text(response)


def render_native_row(row, behavior, historical):
    """The exact state.prompt / behavior.text consumed by RTD score_behavior."""
    prompt = bfcl_native_prompt(behavior['state'], historical)
    response = behavior['text']
    if not isinstance(response, str):
        raise ValueError('native continuation must be text')
    if response.endswith(('<|im_end|>', '<|endoftext|>')):
        raise ValueError('native continuation must not contain a trailing EOS')
    result = deepcopy(row)
    result.update(messages=[], prompt=prompt, response=response)
    if not response:
        if historical.get('ground_truth') != [] and historical.get('result') != []:
            raise ValueError('unexplained empty native teacher response')
        # RTD renders zero calls as empty text plus EOS. This explicit marker
        # permits only these purchased abstentions through load_pool; it never
        # attributes invented refusal prose or a legacy [] target to a teacher.
        result['_native_fc_empty_response'] = True
    return result


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


def render_package(benchmark, snapshot, payload=None, row_format=None):
    row_format = row_format_for(benchmark, row_format)
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
            if row_format == NATIVE_ROW_FORMAT:
                rendered.append(render_native_row(row, behavior, historical))
                continue
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
