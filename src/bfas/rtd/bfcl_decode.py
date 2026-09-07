"""RTD-only guards for sampled BFCL actions; the campaign handler is untouched."""
import json
import re
from types import MethodType


# Restrict recovery to model-response decoding. Backend, task/environment and
# checker infrastructure failures must still propagate to the training worker.
DECODE_ERRORS = (ValueError, KeyError, TypeError, SyntaxError, AttributeError)


def guard_rtd_decoding(handler, record):
    """Retain malformed text in history and use BFCL's failed-decode turn path.

    Qwen's extractor silently drops invalid JSON and admits invalid call shapes.
    The latter poison history and crash _format_prompt on the next user turn.
    Validate before that history is built; never repair arguments or resample.
    Plain assistant chat remains legal (including multi-turn completion messages).
    """
    extract = handler._extract_tool_calls
    parse = handler._parse_query_response_prompting

    def checked_extract(_handler, response):
        if not isinstance(response, str):
            raise TypeError('model response must be text')
        if not response.strip():
            raise ValueError('empty model response')
        # Match the official Qwen framing exactly, including its newlines.
        pattern = r'<tool_call>\n(.*?)\n</tool_call>'
        bodies = re.findall(pattern, response, re.DOTALL)
        outside_calls = re.sub(pattern, '', response, flags=re.DOTALL)
        if '<tool_call>' in outside_calls or '</tool_call>' in outside_calls:
            raise ValueError('incomplete or invalid tool-call framing')
        for body in bodies:
            call = json.loads(body)
            if not isinstance(call, dict):
                raise TypeError('tool call must be an object')
            name, arguments = call['name'], call['arguments']
            if not isinstance(name, str) or not name:
                raise TypeError('tool-call name must be a nonempty string')
            if not isinstance(arguments, dict):
                raise TypeError('tool-call arguments must be an object')
        return extract(response)

    def checked_parse(_handler, response):
        # Read the backend envelope outside recovery: only sampled text is an
        # action, not a broken backend response or missing usage metadata.
        text = response.choices[0].text
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        if not isinstance(text, str):
            raise TypeError('backend response text must be a string')
        try:
            return parse(response)
        except DECODE_ERRORS as exc:
            record(exc, 'parse_response')
            # Preserve the raw action as assistant content, so the next prompt
            # never indexes a malformed structured call. decode_execute below
            # still raises and BFCL advances to its next turn without execution.
            return dict(model_responses=text, reasoning_content='',
                model_responses_message_for_chat_history=dict(role='assistant', content=text),
                input_token=prompt_tokens, output_token=completion_tokens)

    handler._extract_tool_calls = MethodType(checked_extract, handler)
    handler._parse_query_response_prompting = MethodType(checked_parse, handler)
    for name in ('decode_ast', 'decode_execute'):
        decode = getattr(handler, name)

        def checked_decode(_handler, *args, _decode=decode, _stage=name, **kwargs):
            try:
                return _decode(*args, **kwargs)
            except DECODE_ERRORS as exc:
                record(exc, _stage)
                raise  # BFCL's own turn/checker boundary handles failed decode.

        setattr(handler, name, MethodType(checked_decode, handler))
