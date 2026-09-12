"""Azure Chat Completions teacher using the official BFCL FC protocol."""

import os

from appworld_teacher import AZURE_OPENAI_API_VERSION, _azure_credentials
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
from bfcl_eval.model_handler.base_handler import BaseHandler
from openai import AzureOpenAI


class AzureOpenAIHandler(OpenAICompletionsHandler):
    def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs):
        # Do not construct the parent's public OpenAI client: Azure credentials
        # and deployment routing are independent of OPENAI_* / Ollama settings.
        BaseHandler.__init__(
            self, model_name, temperature, registry_name, is_fc_model, **kwargs
        )
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        endpoint, key = _azure_credentials()
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=key,
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", AZURE_OPENAI_API_VERSION),
        )
        self.reasoning_effort = os.environ.get("BFAS_BFCL_AZURE_REASONING_EFFORT", "none")
        if self.reasoning_effort not in {"none", "low", "medium", "high", "xhigh"}:
            raise ValueError("Invalid BFAS_BFCL_AZURE_REASONING_EFFORT")

    def _query_FC(self, inference_data):
        request = {
            "model": self.model_name,  # Azure deployment name, without azure/ or -FC
            "messages": inference_data["message"],
            "reasoning_effort": self.reasoning_effort,
            "store": False,
        }
        # GPT-5.4 accepts sampling parameters only with reasoning disabled.
        if self.reasoning_effort == "none":
            request["temperature"] = self.temperature
        if inference_data["tools"]:
            request["tools"] = inference_data["tools"]
        inference_data["inference_input_log"] = dict(request)
        return self.generate_with_backoff(**request)

    def _parse_query_response_FC(self, api_response):
        response = super()._parse_query_response_FC(api_response)
        message = api_response.choices[0].message
        if not message.tool_calls:
            response["model_responses"] = message.content or ""
        # completion_tokens already includes completion_tokens_details.reasoning_tokens.
        # Keep the provider total intact; BaseHandler writes output_token_count.
        response["output_token"] = api_response.usage.completion_tokens
        return response

    def decode_ast(self, result, language, has_tool_call_tag):
        if isinstance(result, str):
            return []  # Native no-call text is a valid BFCL irrelevance answer.
        return super().decode_ast(result, language, has_tool_call_tag)

    def decode_execute(self, result, has_tool_call_tag):
        if isinstance(result, str):
            return []
        return super().decode_execute(result, has_tool_call_tag)


def register_azure_model():
    """Extend model_config in-process; keep the shared harness checkout untouched."""
    from bfcl_eval.constants.model_config import (
        MODEL_CONFIG_MAPPING, ModelConfig, api_inference_model_map,
    )
    from bfcl_eval.constants.supported_models import SUPPORTED_MODELS

    name = "azure/gpt-5.4-FC"
    config = ModelConfig(
        model_name="gpt-5.4",
        display_name="GPT-5.4 Azure (FC)",
        url="https://developers.openai.com/api/docs/models/gpt-5.4",
        org="OpenAI", license="Proprietary",
        model_handler=AzureOpenAIHandler,
        is_fc_model=True, underscore_to_dot=True,
    )
    api_inference_model_map[name] = config
    MODEL_CONFIG_MAPPING[name] = config
    if name not in SUPPORTED_MODELS:
        SUPPORTED_MODELS.append(name)
