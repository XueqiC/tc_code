#!/usr/bin/env python3
"""Run the vendored tau2 CLI with a guard around its LiteLLM calls."""
from __future__ import annotations

import os
import json
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    from bfas.tau2_budget import (
        RequestBudget, charge_response, cost_usd, is_luna_model,
        register_luna_price, request_usage_bound, response_field, response_usage, service_tier,
    )

    config = os.environ.get("BFAS_TAU2_BUDGET_CONFIG")
    budget = RequestBudget(Path(config)) if config else None
    tier = budget.service_tier if budget else service_tier()
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import litellm

    register_luna_price(litellm, tier)
    from tau2.evaluator import evaluator_nl_assertions
    from tau2.utils import llm_utils

    original = llm_utils.completion

    def completion(**kwargs):
        if is_luna_model(kwargs.get("model")):
            # Register dated aliases before LiteLLM validates tool_choice.
            if kwargs["model"].rsplit("/", 1)[-1] != "gpt-5.6-luna":
                register_luna_price(litellm, tier, kwargs["model"])
        if budget is not None:
            return budget.call(original, **kwargs)
        luna = is_luna_model(kwargs.get("model"))
        if luna:
            kwargs["model"] = "openai/" + kwargs["model"].strip().rsplit("/", 1)[-1]
            kwargs.pop("temperature", None)
            maximum = kwargs.pop("max_tokens", 2048)
            kwargs.setdefault("max_completion_tokens", maximum)
            kwargs.update(service_tier=tier, num_retries=0, max_retries=0)
            request_usage_bound(kwargs)  # estimates require an enforced output limit
        response = original(**kwargs)
        charge = charge_response(response, kwargs, tier) if luna else None
        if (kwargs.get("metadata") or {}).get("bfas_purpose") == "teacher_judge":
            directory = llm_utils.llm_log_dir.get()
            simulation_id = next(part[4:] for part in directory.parts if part.startswith("sim_"))
            row = {"simulation_id": simulation_id, **charge}
            with Path(os.environ["BFAS_TAU2_JUDGE_USAGE_PATH"]).open("a") as handle:
                handle.write(json.dumps(row) + "\n")
        return response

    llm_utils.completion = completion
    original_cost = llm_utils.get_response_cost
    original_usage = llm_utils.get_response_usage

    def get_response_cost(response):
        charge = response_field(response, "bfas_charge")
        if charge is not None:
            return charge["estimated_usd"]
        if is_luna_model(response_field(response, "model")):
            return cost_usd(response_usage(response), tier)
        return original_cost(response)

    def get_response_usage(response):
        charge = response_field(response, "bfas_charge")
        if charge is not None:
            return charge["usage"]
        return original_usage(response)

    # Native reporting must not consult a second, potentially stale price map.
    llm_utils.get_response_cost = get_response_cost
    llm_utils.get_response_usage = get_response_usage
    # Some pinned retail tasks require this native LLM grading step. Preserve
    # the rubric and evaluator, price its Luna calls alongside the two actors.
    evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = "openai/gpt-5.6-luna"
    evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {
        "base_url": "https://api.openai.com/v1", "service_tier": tier,
        "max_completion_tokens": budget.config["max_completion_tokens"] if budget else 2048,
        "metadata": {"bfas_purpose": "teacher_judge"},
    }
    # tau2 imports this module's generate function; its globals now use the guard.
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
