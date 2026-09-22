"""CPU-only K=32 seed acceptance and exposure/RNG checks; no training or banks."""
import argparse
from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT)]

from bfas.rtd.baselines import alfworld_cli
from bfas.rtd.baselines.alfworld_curriculum import exposure_schedule
from bfas.rtd.baselines.paper_data import TeacherRow
from bfas.rtd.baselines.paper_seeds import seed_training, training_seed
from bfas.rtd.baselines.pi1 import exposure_plan
from bfas.rtd.persistence import digest
from tools import alf_pi1_train


@pytest.mark.parametrize("method", ["pi1_ce", "smartad", "sad"])
@pytest.mark.parametrize("seed", [0, 1, 2, -1, 3])
def test_cli_seed_choices_without_running_training(monkeypatch, method, seed):
    class Parsed(Exception):
        pass

    original = argparse.ArgumentParser.parse_args

    def parse_only(parser, *args, **kwargs):
        # Exercise the actual parser, then stop before any CLI side effects.
        raise Parsed(original(parser, *args, **kwargs))

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_only)
    argv = ["--seed", str(seed), "--output", "unused"]
    main = alf_pi1_train.main
    if method != "pi1_ce":
        main = alfworld_cli.main
        argv = ["train", "--method", method, *argv, "--config", "unused.yaml",
                "--model-path", "unused-model", "--gpu-uuid", "unused"]
    accepted = seed in (0, 1, 2)
    with pytest.raises(Parsed if accepted else SystemExit) as result:
        main(argv)
    if accepted:
        assert result.value.args[0].seed == seed
    else:
        assert result.value.code == 2


# Captured from the unmodified implementation before widening the CLI choices.
SEED_01_PLAN_HASHES = {
    "pi1_ce": ("3b5151e0a3759ce78fdb1190146332c1b492de11ccdfff13134bc81887bf2380",
               "de918b1f096d10183c31f067ecec49b2adabc7868a9f339c837c98670966e6ec"),
    "smartad": ("a001304c53c268af108eab6696fa028ea2a18f827fee0ca9efd75d459acdad89",
                "4ef75db3aafbf0fb4d6e9bf9b55bb2ec791524058cfbd214206591f6e8bb7862"),
    "sad": ("ab66cd3007842aa16249cfb76f1fc3f12e44417addd0189407e92e190fd73e13",
            "fa2bea2440e049ac8b11a7df9179fc01d075d8a6e94d66ff6e46d65b87ae0053"),
}


@pytest.mark.parametrize("method", ["pi1_ce", "smartad", "sad"])
def test_three_seed_orders_preserve_rows_endpoints_and_seed_01_plans(method):
    rows = [TeacherRow(package, "task", "parent", i, "prompt", "ACTION: look", "alfworld", False)
            for package, length in [("long", 5), ("short", 1), ("medium", 3)] for i in range(length)]
    costs = list(range(2, len(rows)+2))
    config = dict(exposure_passes=[3, 10], supervised_tokens_per_update=17)

    def plan(seed):
        if method == "pi1_ce":
            return exposure_plan(costs, config, seed)
        return exposure_schedule(rows, costs, config, seed, method)

    plans = [plan(seed) for seed in (0, 1, 2)]
    assert tuple(digest(p) for p in plans[:2]) == SEED_01_PLAN_HASHES[method]
    orders = []
    for seed, current in enumerate(plans):
        assert current == plan(seed)
        order = [i for batch in current["batches"] for i in batch["indices"]]
        orders.append(tuple(order))
        assert Counter(rows[i] for i in order) == Counter({r: 10 for r in rows})
        for start in range(0, len(order), len(rows)):
            assert sorted(order[start:start+len(rows)]) == list(range(len(rows)))
        assert {b["endpoint"]: b["cumulative_tokens"] for b in current["batches"] if b["endpoint"]} == {
            3: 3*sum(costs), 10: 10*sum(costs)}
    assert len(set(orders)) == 3


def test_three_seed_rng_streams_repeat_and_differ_on_cpu():
    saved = random.getstate(), np.random.get_state(), torch.get_rng_state()

    def draws(seed):
        assert training_seed({"training_seed": seed}) == seed
        seed_training(seed)
        return (tuple(random.random() for _ in range(8)),
                tuple(np.random.random(8)), tuple(torch.rand(8, device="cpu").tolist()))

    try:
        streams = [draws(seed) for seed in (0, 1, 2)]
        assert streams == [draws(seed) for seed in (0, 1, 2)]
        for library in range(3):
            assert len({stream[library] for stream in streams}) == 3
        assert not torch.cuda.is_initialized()
    finally:
        random.setstate(saved[0])
        np.random.set_state(saved[1])
        torch.set_rng_state(saved[2])
