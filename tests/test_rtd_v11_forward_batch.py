"""D13: actual Qwen3.5 linear/full attention + nonzero LoRA, CPU FP32."""
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from test_rtd_v11_generation_batch import backend, rng
from bfas.rtd.checkpointing import enable_gradient_checkpointing
from bfas.rtd.forward_batch import POSITION_CHUNK, plan_forwards, token_row
from bfas.rtd.functional_step import lora_parameters, snapshot
from bfas.rtd.generation_batch import GenerationBatch
from bfas.rtd.persistence import ComputeJournal, digest
from bfas.rtd.return_gradient import ActionTrace, IncompleteRolloutError
from bfas.rtd.runtime import HFGenerateBackend
from bfas.rtd.source_scoring import source_gradient_pair
from bfas.rtd.transport import Behavior, FullState, SourceSample


def traces(b, p):
    # Same-state pair, mixed EOS/cap, unequal lengths, prompt length one,
    # >128-token recurrent chunk boundary, and action projection chunk boundary.
    specs = [((0, 1)*67, (2, 1, 3)), ((0, 1)*67, (1,)*37),
             ((2,)*127, (0, 3)), ((1,), (3,)), ((0, 1), (2, 0))]
    return tuple(ActionTrace(prompt, action, 3, '', 0., b.backend_id, b.identity(p),
                             truncated=action[-1] != 3) for prompt, action in specs)


def enable(b, **kwargs):
    b.generation_batch = GenerationBatch(forward_prompts_per_batch=8, **kwargs)


def test_qwen35_every_scored_position_and_right_padding(backend, monkeypatch):
    p = {n: v+.03 for n, v in snapshot(lora_parameters(backend.model)).items()}
    actions = traces(backend, p)
    with torch.no_grad():
        expected = [backend.score_action(a, p, return_details=True) for a in actions]
    calls, head_shapes = [], []
    model = backend.model.get_base_model()
    def spy(module, args, kwargs):
        calls.append((kwargs['input_ids'].clone(), kwargs['attention_mask'].clone(), torch.is_grad_enabled()))
    handle = backend.model.register_forward_pre_hook(spy, with_kwargs=True)
    head_hook = model.lm_head.register_forward_hook(lambda m, a, o: head_shapes.append(o.shape))
    enable(backend)
    try:
        with torch.no_grad():
            got = backend.score_actions_batch(actions, p, return_details=True)
    finally:
        handle.remove(); head_hook.remove()
    assert len(calls) < len(actions) and any(len(ids) > 1 for ids, _, _ in calls)
    assert any(bool((mask == 0).any()) for _, mask, _ in calls)
    for ids, mask, grad in calls:
        assert not grad and (mask[:, 1:] <= mask[:, :-1]).all() and mask[:, 0].all()
        assert ids.numel() <= backend.generation_batch.max_batch_tokens
    assert all(len(shape) == 2 and shape[0] <= POSITION_CHUNK for shape in head_shapes)
    for a, before, after in zip(actions, expected, got):
        assert len(after[1]) == len(a.action_ids)
        torch.testing.assert_close(after[1], before[1], atol=1e-4, rtol=0)
        assert torch.equal(after[0], after[1].sum())
        assert before[2] == after[2]  # Existing journal fields/dtypes unchanged.


def test_shared_prompt_pair_one_forward_and_length_budget(backend):
    p = snapshot(lora_parameters(backend.model))
    actions = traces(backend, p)[:2]
    enable(backend, max_batch_tokens=1000)
    assert len(plan_forwards(actions, backend.generation_batch, 512)) == 1
    policy = GenerationBatch(max_batch_tokens=100, forward_prompts_per_batch=1)
    rows = [token_row((1,)*n, (2, 3), 3, False) for n in (80, 20, 500, 23, 41, 20)]
    groups = plan_forwards(rows, policy, 512)
    assert sorted(r['index'] for g in groups for r in g) == list(range(len(rows)))
    for g in groups:
        width = max(len(r['ids']) for r in g)
        assert len(g)*width <= 100 or len(g) == 1
        assert width <= 2*min(len(r['ids']) for r in g)
        assert len({r['prompt_index'] for r in g}) <= 1
    with pytest.raises(IncompleteRolloutError):
        plan_forwards([token_row((1,)*511, (2, 3), 3, False)], policy, 512)
    for a in (token_row((), (3,), 3, False), token_row((1,), (3, 3), 3, False),
              token_row((1,), (2,), 3, False), token_row((1,), (3,), 3, True)):
        with pytest.raises(ValueError):
            plan_forwards([a], policy, 512)


def test_conditional_kl_matches_original_per_action_order_and_zero(backend, tmp_path):
    p = snapshot(lora_parameters(backend.model))
    q = {n: (v+.3).requires_grad_(True) for n, v in p.items()}
    actions = traces(backend, p)
    with torch.no_grad():
        expected = torch.stack([backend._action_kl(a, p, q) for a in actions])
    enable(backend)
    backend.journal = ComputeJournal(tmp_path/'kl.jsonl', cuda=False)
    with torch.no_grad():
        got = backend.source_kl(actions, p, q)
        singles = torch.stack([backend.source_kl([a], p, q) for a in actions])
        zero = backend.source_kl(actions, p, p)
    torch.testing.assert_close(singles, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(got, expected.mean(), atol=1e-6, rtol=1e-5)
    assert zero.item() == 0 and not got.requires_grad
    first = next(r for r in backend.journal.events if r['kind'] == 'compute_begin')
    assert first['forward_passes'] < 2*len(actions)
    with torch.no_grad(), pytest.raises(ValueError, match='policy mismatch'):
        backend.source_kl([replace(actions[0], policy_id='wrong')], p, q)


def test_no_grad_prefetch_preserves_checks_single_use_and_failure_cleanup(backend, tmp_path):
    p = snapshot(lora_parameters(backend.model))
    actions = backend.sample_actions((0, 1), 2, p, rng())
    enable(backend)
    backend.journal = ComputeJournal(tmp_path/'scores.jsonl', cuda=False)
    original = snapshot(lora_parameters(backend.model))
    global_rng = torch.get_rng_state().clone()
    with backend.prefetch_scores(actions, p):
        assert torch.is_grad_enabled()
        # Training always obtains a real, independent graph; leaves the queue intact.
        assert backend.score_action(actions[0], p).requires_grad
        with torch.no_grad():
            for a in actions:
                backend.checked_score_action(a, p, expected_prompt_ids=a.prompt_ids)
            with pytest.raises(AssertionError, match='already consumed'):
                backend.score_action(actions[-1], p)
    assert backend._pending_forward_scores is None
    assert torch.equal(global_rng, torch.get_rng_state())
    assert all(torch.equal(original[n], v) for n, v in lora_parameters(backend.model).items())
    checks = [r for r in backend.journal.events if r['kind'] == 'score_consistency']
    assert [r['action_ids'] for r in checks] == [a.action_ids for a in actions]
    assert all(r['passed'] for r in checks)
    bad = replace(actions[0], generation_logprob=actions[0].generation_logprob-10,
                  generation_token_logprobs=tuple(v-10 if i == 0 else v
                      for i, v in enumerate(actions[0].generation_token_logprobs)))
    with pytest.raises(ValueError, match='likelihood differs'):
        with backend.prefetch_scores([bad, actions[1]], p), torch.no_grad():
            backend.checked_score_action(bad, p)
    assert backend._pending_forward_scores is None
    assert not [r for r in backend.journal.events if r['kind'] == 'score_consistency'][-1]['passed']
    with pytest.raises(AssertionError, match='not fully consumed'):
        with backend.prefetch_scores(actions, p):
            pass
    with pytest.raises(AssertionError, match='policy mismatch'):
        with backend.prefetch_scores(actions, p), torch.no_grad():
            backend.score_tokens(actions[0].prompt_ids, actions[0].action_ids,
                {n: v+.1 for n, v in p.items()}, eos_token_id=3, truncated=actions[0].truncated)


def test_per_unit_hard_soft_teacher_gradients_stay_byte_identical(backend):
    from bfas.rtd.alpha_d import ExposureRecord, SourcePair, components
    enable_gradient_checkpointing(backend.model)
    p = snapshot(lora_parameters(backend.model))
    q = {n: (v+.04).detach().requires_grad_(True) for n, v in p.items()}
    state = FullState.create({'q': 'q'}, [{'role': 'user', 'content': 'q'}], '0 1', 'parent')
    sources = tuple(SourceSample(Behavior(state, ''), backend.identity(p), ids, 3, -1., truncated=ids[-1] != 3)
                    for ids in ((1, 3), (2, 1, 0)))
    pair = SourcePair(ExposureRecord(None, 0, state, Behavior(state, '2 1')), sources, ('draw1', 'draw2'))
    expected = components(pair, backend, q, p)
    enable(backend)
    got = components(pair, backend, q, p)
    for before, after in zip(expected, got):
        for n in p:
            assert torch.equal(before[n], after[n])
    for src in sources:
        _, soft, _ = source_gradient_pair(backend, src, p, p)
        assert all(not g.any() for g in soft.values())


def test_d12_sampling_identity_rng_and_default_config_unchanged(backend):
    p = snapshot(lora_parameters(backend.model))
    original = backend.sample_actions_batch([(0, 1), (2, 1)], 2, p, rng())
    b = HFGenerateBackend(backend.model, backend.tokenizer, base_checkpoint_hash='tiny-qwen35',
        harness_hash='d12', tokenizer_hash='integer-vocab', max_action_tokens=8, max_context_tokens=512,
        generation_batch=GenerationBatch(forward_prompts_per_batch=8))
    assert b.backend_id == backend.backend_id and b.identity(p) == backend.identity(p)
    assert b.sample_actions_batch([(0, 1), (2, 1)], 2, p, rng()) == original
    assert GenerationBatch.from_config({}) is None
    assert GenerationBatch().config() == dict(prompts_per_batch=8, max_batch_tokens=16384)
    for v in (-1, True, 1.2):
        with pytest.raises(ValueError):
            GenerationBatch(forward_prompts_per_batch=v)
    with pytest.raises(ValueError, match='no-grad'):
        b.score_actions_batch(traces(b, p), p)


def test_d7_cross_state_source_journals_and_midphase_recovery(backend, tmp_path):
    from test_rtd_v11_generation_batch import test_real_source_pair_checks_and_durable_midphase_resume
    enable(backend)
    test_real_source_pair_checks_and_durable_midphase_resume(backend, tmp_path)
    assert backend._pending_generation is None and backend._pending_forward_scores is None
    calls = [r for r in backend.journal.events if r['kind'] == 'compute_begin'
             and r['operation'] == 'teacher_forced_forward']
    assert all(r['sequences'] == 4 for r in calls) and len(calls) == 3


def test_failed_batched_forward_restores_head_hook_and_parameters(backend, monkeypatch):
    enable(backend)
    head = backend.model.get_output_embeddings()
    hooks = dict(head._forward_pre_hooks)
    p = snapshot(lora_parameters(backend.model))
    def fail(*args, **kwargs):
        raise RuntimeError('backbone failure')
    monkeypatch.setattr(backend.model.get_base_model().model, 'forward', fail)
    with torch.no_grad(), pytest.raises(RuntimeError, match='backbone failure'):
        backend.score_actions_batch(traces(backend, p), p)
    assert dict(head._forward_pre_hooks) == hooks
    assert all(torch.equal(p[n], v) for n, v in lora_parameters(backend.model).items())


def test_d12_d13_source_pair_hashes_and_journal_order_match(backend, tmp_path):
    from test_rtd_v11_generation_batch import test_real_source_pair_checks_and_durable_midphase_resume
    test_real_source_pair_checks_and_durable_midphase_resume(backend, tmp_path/'d12')
    before = backend.journal.events
    enable(backend)
    test_real_source_pair_checks_and_durable_midphase_resume(backend, tmp_path/'d13')
    after = backend.journal.events
    kinds = {'alpha_d_source_pair', 'source_sample', 'score_consistency'}
    assert [r['kind'] for r in before if r['kind'] in kinds] == [r['kind'] for r in after if r['kind'] in kinds]
    for kind, keys in [('alpha_d_source_pair', ('sample_hashes', 'action_hashes', 'draw_ids', 'rng_before', 'rng_after', 'source_id')),
                       ('source_sample', ('action', 'source_id', 'state_hash'))]:
        assert [{k: r[k] for k in keys} for r in before if r['kind'] == kind] == [
               {k: r[k] for k in keys} for r in after if r['kind'] == kind]


def test_benchmark_same_actions_export_and_d12_import(backend, tmp_path):
    import json
    from tools.rtd_v11_forward_bench import load_actions
    from bfas.rtd.scoring import score_diagnostic
    p = snapshot(lora_parameters(backend.model))
    actions = backend.sample_actions((0, 1), 2, p, rng())*40
    export = tmp_path/'actions.json'
    export.write_text(json.dumps(dict(actions=[asdict(a) for a in actions])))
    assert load_actions(export) == actions
    with torch.no_grad():
        details = [backend.score_action(a, p, return_details=True) for a in actions[:2]]
    records = [dict(kind='score_consistency', context='benchmark_batched',
        **score_diagnostic(a, v, s, m, backend.score_tolerance))
        for a, (s, v, m) in zip(actions[:2], details)]*40
    d12 = tmp_path/'compute.jsonl'
    d12.write_text(''.join(json.dumps(r)+'\n' for r in records))
    assert load_actions(d12) == actions


def test_benchmark_nonzero_exit_on_one_bad_token_or_memory(backend, tmp_path, monkeypatch):
    import json
    from tools import rtd_v11_forward_bench as bench
    p = snapshot(lora_parameters(backend.model))
    actions = backend.sample_actions((0, 1), 2, p, rng())*40
    enable(backend)
    export = tmp_path/'actions.json'
    export.write_text(json.dumps(dict(actions=[asdict(a) for a in actions])))
    args = SimpleNamespace(actions=export, out=tmp_path, order='unbatched-first', atol=1e-4)
    for error, peak, code in ((0., 39., 0), (.001, 39., 1), (0., 40., 1)):
        def score(b, actions, params, *, batched):
            values = [list(a.generation_token_logprobs) for a in actions]
            if batched:
                values[0][0] += error
            return dict(wall_seconds=1., peak_allocated_gb=peak, generation_score_guard_passed=True), values
        monkeypatch.setattr(bench, 'score_case', score)
        assert bench.run_forward_benchmark(backend, [], None, p, args, {}) == code
    assert not bench.compare_logprobs([[-1.]], [[float('nan')]], atol=1e-4)['passed']
    with pytest.raises(ValueError, match='coverage'):
        bench.compare_logprobs([[-1.]], [[-1., -1.]], atol=1e-4)
