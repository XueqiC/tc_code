"""API-hosted native FC models, registered without editing the BFCL checkout."""

import os
import threading
import time
import uuid
from pathlib import Path

from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
from bfcl_eval.model_handler.base_handler import BaseHandler
from openai import OpenAI

from .bfcl_teacher import PROVIDERS, provider_credentials
from .ledger import append_record


class OpenAICompatibleHandler(OpenAICompletionsHandler):
    def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs):
        BaseHandler.__init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        self.provider = registry_name.split("/", 1)[0]
        base_url, key = provider_credentials(self.provider)
        self.service_tier = None
        if self.provider == "openai":
            self.service_tier = os.environ.get("BFAS_OPENAI_SERVICE_TIER", "").strip() or None
            if self.service_tier not in {None, "flex", "priority"}:
                raise ValueError("Invalid BFAS_OPENAI_SERVICE_TIER: expected 'flex' or 'priority'")
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

    def _record_usage(self, input_tokens, output_tokens, *, cached_tokens=None):
        if path := os.environ.get("BFAS_BFCL_USAGE_LOG"):
            record = {
                "id": self._usage.task_id,
                "bfas_attempt_id": self._usage.attempt_id,
                "input_token_count": input_tokens,
                "output_token_count": output_tokens,
            }
            if cached_tokens is not None:
                record["cached_tokens"] = cached_tokens
            append_record(Path(path), record)

    def generate_with_backoff(self, **kwargs):
        # Each BFAS attempt has one owner; disable hidden SDK retries and let
        # collection budget subsequent attempts, including failures.
        start = time.monotonic()
        response = self.client.chat.completions.create(**kwargs)
        if getattr(self._usage, "task_id", None) is not None:
            usage = response.usage
            self._usage.input_tokens += usage.prompt_tokens
            self._usage.output_tokens += usage.completion_tokens
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            self._record_usage(
                usage.prompt_tokens, usage.completion_tokens,
                cached_tokens=getattr(prompt_details, "cached_tokens", None),
            )
        return response, time.monotonic() - start

    def _query_FC(self, inference_data):
        request = {
            "model": self.model_name,
            "messages": inference_data["message"],
        }
        # Luna rejects nonzero temperature. Keep self.temperature for BFCL's
        # result naming, but leave sampling to the official OpenAI endpoint.
        if not (self.provider == "openai" and self.model_name == "gpt-5.6-luna"):
            request["temperature"] = self.temperature
        if self.service_tier is not None:
            request["service_tier"] = self.service_tier
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


def register_api_models():
    from bfcl_eval.constants.model_config import (
        MODEL_CONFIG_MAPPING, ModelConfig, api_inference_model_map,
    )
    from bfcl_eval.constants.supported_models import SUPPORTED_MODELS

    for name, display, org, license in (
        ("ollama/gpt-oss:120b-FC", "GPT-OSS 120B Ollama", "OpenAI", "Apache-2.0"),
        ("ollama/mistral-large-3:675b-FC", "Mistral Large 3 675B Ollama", "Mistral AI", "Apache-2.0"),
        ("openai/gpt-5.4-FC", "GPT-5.4 OpenAI", "OpenAI", "Proprietary"),
        ("openai/gpt-5.6-luna-FC", "GPT-5.6 Luna OpenAI", "OpenAI", "Proprietary"),
        ("openrouter/openai/gpt-5.4-FC", "GPT-5.4 OpenRouter", "OpenAI", "Proprietary"),
        ("openrouter/anthropic/claude-sonnet-5-FC", "Claude Sonnet 5 OpenRouter", "Anthropic", "Proprietary"),
        ("openrouter/google/gemini-3.1-pro-preview-FC", "Gemini 3.1 Pro Preview OpenRouter", "Google", "Proprietary"),
        ("openrouter/openai/gpt-5.6-luna-FC", "GPT-5.6 Luna OpenRouter", "OpenAI", "Proprietary"),
    ):
        prefix, model = name.split("/", 1)
        config = ModelConfig(
            model_name=model.removesuffix("-FC"), display_name=f"{display} (FC)",
            url=PROVIDERS[prefix][2], org=org, license=license,
            model_handler=OpenAICompatibleHandler, is_fc_model=True, underscore_to_dot=True,
        )
        MODEL_CONFIG_MAPPING[name] = api_inference_model_map[name] = config
        if name not in SUPPORTED_MODELS:
            SUPPORTED_MODELS.append(name)


# Preserve imports used by existing collectors and tests.
OllamaOpenAIHandler = OpenAICompatibleHandler
register_ollama_models = register_api_models
