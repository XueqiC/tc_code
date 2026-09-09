"""Offline awb3 deployment rendering of purchased AppWorld trace prefixes.

awb3 uses appworld_eval, not the newer official few-shot ReAct scaffold.
The cached base tokenizer supplies the opening think tag; targets supply only
the remainder of the empty block. No model, bridge, or environment is run.
"""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

from tools import table1_common as io


APPWORLD_NATIVE_ROW_FORMAT = 'native-appworld-awb3-v1'
ASSISTANT_MARKER = '<|im_start|>assistant\n'
THINK_OPEN = '<think>\n'
THINK_REMAINDER = '\n</think>\n\n'


@lru_cache(maxsize=1)
def base_tokenizer():
    from transformers import AutoTokenizer
    from transformers.utils.hub import cached_file

    config = cached_file(io.INITIAL_CHECKPOINT['model'], 'tokenizer_config.json',
                         local_files_only=True)
    return AutoTokenizer.from_pretrained(Path(config).parent, local_files_only=True,
                                        trust_remote_code=False)


def evaluation_prompt(messages, tokenizer=None):
    """Same kwargs as awb3 generate_reply (APPWORLD_THINK unset, no tools kwarg).

    Docs are discovered through code and retained as user observations. Passing
    a tools schema or enable_thinking=False would change the deployment input.
    """
    tokenizer = tokenizer if tokenizer is not None else base_tokenizer()
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not prompt.endswith(ASSISTANT_MARKER + THINK_OPEN):
        raise ValueError('AppWorld base template boundary changed; re-audit emission contract')
    return prompt


def teacher_emission(response):
    if not isinstance(response, str) or not response:
        raise ValueError('AppWorld teacher continuation must be non-empty text')
    if any(marker in response for marker in ('<think>', '</think>', '<|im_end|>', '<|endoftext|>')):
        raise ValueError('unexpected framing in sealed AppWorld teacher continuation')
    return THINK_REMAINDER + response


def rejected_emission(response):
    """Cached failed replies already continue the open think block.

    Keep the parsed student text, including reasoning and malformed code. The
    old evaluator strips leading/trailing whitespace; raw emission is absent.
    """
    if not isinstance(response, str) or not response or '<think>' in response:
        raise ValueError('unexpected cached AppWorld rejection framing')
    if any(marker in response for marker in ('<|im_end|>', '<|endoftext|>')):
        raise ValueError('cached AppWorld rejection contains EOS')
    return response


def render_rows(snapshot):
    from tools.bfcl_demo_pool import serialize

    rows = snapshot['rows']
    if not rows:
        return []
    previous = None
    rendered = []
    for row in rows:
        messages = row['messages']
        if (row['task_id'] != snapshot['package']['task_id']
                or row['teacher'] != snapshot['package']['teacher']
                or len(messages) != row['turn_index']
                or row['prompt'] != serialize(messages)
                or [m['role'] for m in messages] != ['system', 'user'] +
                   ['assistant', 'user'] * ((len(messages) - 2) // 2)
                or messages[1]['content'] != 'Begin by consulting the API documentation.'):
            raise ValueError('sealed AppWorld row is not an awb3 trace prefix')
        if previous is not None and (
                messages[:-2] != previous['messages']
                or messages[-2] != dict(role='assistant', content=previous['response'])):
            raise ValueError('AppWorld episode prefixes are not consecutive')
        result = deepcopy(row)
        result.update(messages=[], prompt=evaluation_prompt(messages),
                      response=teacher_emission(row['response']))
        rendered.append(result)
        previous = row
    return rendered


def annotate_manifest(manifest):
    manifest['deployment'] = dict(
        scaffold='awb3_appworld_eval', evaluator='src/appworld_eval.py',
        campaign='scripts/awb3_hpg.slurm',
        template_sha256=io.digest(base_tokenizer().chat_template),
        template_kwargs=dict(add_generation_prompt=True), tools_argument=None,
        generation_suffix=ASSISTANT_MARKER + THINK_OPEN,
        teacher_target_prefix=THINK_REMAINDER,
        history='sealed code and truncated execution observations, rendered by base chat template',
        official_react_scaffold_equivalent=False,
        recorded_awb3_prompt_validation='unavailable: no awb3 evaluation input transcripts in this worktree')
    manifest['cost_confidence'] = 'estimated'
    for arm in manifest['arms'].values():
        arm['cost_confidence'] = 'estimated' if arm['C_m'] else 'exact'
    ddpo = manifest['arms']['ddpo']
    ddpo.update(status='requires new teacher calls', trainer_status='requires_new_teacher_calls',
                rank_pairs=0)
    manifest['arms']['pbsd_agent']['note'] = (
        'c references purchased same-task first turns; state_prompt is the consuming row prompt; '
        'source_state_prompt retains evidence context; appworld_train does not consume c')
    manifest['arms']['pbsd_insp']['cached_verdict_rule'] = (
        'aw_pbsd_pool: not (passed_tests > failed_tests); raw sample records unavailable here')
    manifest['arms']['pbsd_insp']['rejected_rendering'] = (
        'cached decoded/stripped continuation of the open think block, unchanged')
