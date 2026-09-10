"""Configured student rendering, shared with the deployment converters."""
from ..cc_pairs import render_prompt, thinking_off


def bfcl_prompt(config, messages, functions, tokenizer=None):
    if config.get('student_call_format', 'qwen') == 'gemma4':
        from tools.bfcl_pool_render_gemma4 import template_render
        if tokenizer is None:
            from .bank_build import load_student_tokenizer
            tokenizer = load_student_tokenizer(config)
        return template_render(tokenizer, messages, functions)
    return thinking_off(render_prompt([messages], functions))


def bfcl_row(config, row, tokenizer):
    if config.get('student_call_format', 'qwen') == 'gemma4':
        from tools.bfcl_pool_render_gemma4 import render_row
        return render_row(row, tokenizer, verify=True)
    from tools.bfcl_pool_render_gemma4 import row_context, assistant_message
    from ..cc_pairs import render_calls
    context = row_context(row)
    assistant = assistant_message(row['response'], context['functions'])
    calls = assistant.get('tool_calls', [])
    target = render_calls([{c['function']['name']: c['function']['arguments']} for c in calls]) if calls else assistant['content']
    return dict(row, prompt=bfcl_prompt(config, context['messages'], context['functions'], tokenizer),
                response=target, _render_context=context)


def termination_ids(backend):
    """Stop at native turn/handoff tokens while retaining their scored likelihood."""
    eos = backend.tokenizer.eos_token_id
    config = getattr(backend, 'student_config', {})
    if 'gemma-4' not in config.get('student', '') and config.get('student_call_format') != 'gemma4':
        return (eos,)
    tokens = ['<turn|>']
    if config.get('benchmark', 'bfcl') == 'bfcl':
        tokens.append('<|tool_response>')
    ids = [eos] + [backend.tokenizer.convert_tokens_to_ids(t) for t in tokens]
    if any(type(i) is not int or i == backend.tokenizer.unk_token_id for i in ids):
        raise ValueError('configured student lacks native termination tokens')
    return tuple(dict.fromkeys(ids))
