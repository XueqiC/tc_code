"""An enumerable CPU policy on real C26-B fixtures, for the unmodified RTD math.

Only the backend/environment boundaries are substituted. Source, paid bank,
parser, complete action scoring, LOO, pilot, VJP, ledger and StateStore are real.
"""
import json
from pathlib import Path

import torch

from bfas.rtd.functional_step import lora_parameters
from bfas.rtd.persistence import atomic_json
from bfas.rtd.benchmarks.registry import ALFWorldFeedbackContext
from test_rtd_alfworld_rollout import TinyBackend, TinyTokenizer, EnumerableEnv, render
from test_rtd_alfworld_state import FakeStepper


class RunnerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_theta = torch.nn.Parameter(torch.tensor([.35, -.2, .4, .1], dtype=torch.float64))
        self.eval()

    def save_pretrained(self, directory, *, safe_serialization):
        from safetensors.torch import save_file
        Path(directory).mkdir(parents=True)
        save_file({n: p.detach().contiguous() for n, p in lora_parameters(self).items()},
                  str(Path(directory) / 'adapter_model.safetensors'))
        atomic_json(Path(directory) / 'adapter_config.json', dict(peft_type='LORA', fixture=True))


class RunnerTokenizer(TinyTokenizer):
    def save_pretrained(self, directory):
        atomic_json(Path(directory) / 'tokenizer_config.json', dict(fixture=True, chat_template='fixture'))

    def encode(self, text, *, add_special_tokens=False):
        # Authored teacher targets may be tokenized; student sampled IDs are
        # always consumed directly by score_source/checked_score_action.
        if text in ('go to table 1', 'put apple 1 on table 1'):
            return [11, 21]
        return super().encode(text, add_special_tokens=add_special_tokens)


class RunnerBackend(TinyBackend):
    def __init__(self):
        super().__init__()
        self.model, self.tokenizer = RunnerModel(), RunnerTokenizer()

    def values(self, prompt_ids, action_ids, parameters):
        return super().values(prompt_ids, action_ids, {'theta': parameters['lora_theta']})

    def sample_action(self, prompt, parameters, generator, *, temperature=1., top_p=1.):
        # TinyBackend's sampler names theta, while the production runner enforces
        # LoRA-only names. Translate that single coordinate name at the boundary.
        class Coordinates(dict):
            def __getitem__(self, name):
                return super().__getitem__('lora_theta' if name == 'theta' else name)
        return super().sample_action(prompt, Coordinates(parameters), generator,
                                     temperature=temperature, top_p=top_p)

    def score_source(self, source, parameters):
        return self.values(self.tokenizer.encode(source.behavior.state.prompt), source.token_ids, parameters).sum()

    def score_behavior(self, behavior, parameters):
        return self.values(self.tokenizer.encode(behavior.state.prompt),
                           (*self.tokenizer.encode(behavior.text), 0), parameters).sum()

    def initial_hidden(self, prompt, initial_parameters, *, initial_snapshot_id):
        assert self.identity(initial_parameters) == initial_snapshot_id
        index = json.loads(prompt)['index']
        return torch.tensor([1., index, len(prompt) / 1000., .3], dtype=torch.float64)

    def source_kl(self, actions, source_parameters, updated_parameters):
        values = []
        for action in actions:
            old = torch.stack([self.values(action.prompt_ids, (10+z, 20+a, 0), source_parameters).sum()
                               for z in (0, 1) for a in (0, 1)])
            new = torch.stack([self.values(action.prompt_ids, (10+z, 20+a, 0), updated_parameters).sum()
                               for z in (0, 1) for a in (0, 1)])
            values.append((old.exp() * (old-new)).sum().clamp_min(0))
        return torch.stack(values).mean()


class RunnerEnv(EnumerableEnv):
    def observation(self):
        if not self.commands:
            return FakeStepper(self.request).observation(0)
        return super().observation()


def feedback_context(self, round_number, backend, journal):
    return ALFWorldFeedbackContext(self.protocol, round_number, render, RunnerEnv, journal)
