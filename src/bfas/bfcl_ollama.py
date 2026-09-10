"""Ollama-hosted native FC models, registered without editing the BFCL checkout."""

import os
import threading
import time
import uuid
from pathlib import Path

from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
from bfcl_eval.model_handler.base_handler import BaseHandler
from openai import OpenAI

from .bfcl_teacher import ollama_credentials
from .ledger import append_record


class OllamaOpenAIHandler(OpenAICompletionsHandler):
    def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs):
        BaseHandler.__init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        base_url, key = ollama_credentials()
        self.client = OpenAI(base_url=base_url, api_key=key, max_retries=0)
        # BFCL shares one handler among inference threads.
        self._usage = threading.local()

    def inference(self, test_entry, include_input_log, exclude_state_log):
        self._usage.task_id = test_entry["id"]
        self._usage.attempt_id = uuid.uuid4().hex
        self._usage.input_tokens = 0
        self._usage.output_tokens = 0
        self._record_usage(0, 0)
        try:
            result, metadata = super().inference(test_entry, include_input_log, exclude_state_log)
            metadata["bfas_attempt_id"] = self._usage.attempt_id
            return result, metadata
        except Exception as exc:
            # The harness's outer exception handler otherwise discards all usage,
            # including successful earlier steps in a failed multi-turn episode.
            return f"Error during inference: {exc}", {
                "error": type(exc).__name__,
                "bfas_attempt_id": self._usage.attempt_id,
                "input_token_count": self._usage.input_tokens,
                "output_token_count": self._usage.output_tokens,
            }
        finally:
            self._usage.task_id = None

    def _record_usage(self, input_tokens, output_tokens):
        if path := os.environ.get("BFAS_BFCL_USAGE_LOG"):
            append_record(Path(path), {
                "id": self._usage.task_id,
                "bfas_attempt_id": self._usage.attempt_id,
                "input_token_count": input_tokens,
                "output_token_count": output_tokens,
            })

    def generate_with_backoff(self, **kwargs):
        # Each BFAS attempt has one owner; disable hidden SDK retries and let
        # collection budget subsequent attempts, including failures.
        start = time.monotonic()
        response = self.client.chat.completions.create(**kwargs)
        if getattr(self._usage, "task_id", None) is not None:
            usage = response.usage
            self._usage.input_tokens += usage.prompt_tokens
            self._usage.output_tokens += usage.completion_tokens
            self._record_usage(usage.prompt_tokens, usage.completion_tokens)
        return response, time.monotonic() - start

    def _query_FC(self, inference_data):
        request = {
            "model": self.model_name,
            "messages": inference_data["message"],
            "temperature": self.temperature,
        }
        if inference_data["tools"]:
            request["tools"] = inference_data["tools"]
        inference_data["inference_input_log"] = dict(request)
        return self.generate_with_backoff(**request)

    def _parse_query_response_FC(self, api_response):
        response = super()._parse_query_response_FC(api_response)
        message = api_response.choices[0].message
        if not message.tool_calls:
            response["model_responses"] = message.content or ""
        # completion_tokens already includes any reasoning tokens.
        response["output_token"] = api_response.usage.completion_tokens
        return response

    def decode_ast(self, result, language, has_tool_call_tag):
        if isinstance(result, str):
            return []
        return super().decode_ast(result, language, has_tool_call_tag)

    def decode_execute(self, result, has_tool_call_tag):
        if isinstance(result, str):
            return []
        return super().decode_execute(result, has_tool_call_tag)


def register_ollama_models():
    from bfcl_eval.constants.model_config import (
        MODEL_CONFIG_MAPPING, ModelConfig, api_inference_model_map,
    )
    from bfcl_eval.constants.supported_models import SUPPORTED_MODELS

    for model, display, org in (
        ("gpt-oss:120b", "GPT-OSS 120B", "OpenAI"),
        ("mistral-large-3:675b", "Mistral Large 3 675B", "Mistral AI"),
    ):
        name = f"ollama/{model}-FC"
        config = ModelConfig(
            model_name=model, display_name=f"{display} Ollama (FC)",
            url="https://ollama.com", org=org, license="Apache-2.0",
            model_handler=OllamaOpenAIHandler, is_fc_model=True, underscore_to_dot=True,
        )
        MODEL_CONFIG_MAPPING[name] = api_inference_model_map[name] = config
        if name not in SUPPORTED_MODELS:
            SUPPORTED_MODELS.append(name)
