"""Offline tiny Gemma 4 multimodal-wrapper oracle for the production score paths."""
import pytest
import torch

from bfas.rtd.functional_step import lora_parameters, snapshot
from bfas.rtd.checkpointing import enable_gradient_checkpointing
from bfas.rtd.generation_batch import GenerationBatch
from bfas.rtd.runtime import HFGenerateBackend
from bfas.rtd.scoring import ScoreTolerance
from test_rtd_v11_generation_batch import Tokenizer


def tiny_gemma4_backend(dtype, *, shared_layers=2, window=8, attention='eager'):
    from peft import LoraConfig, get_peft_model
    from transformers import Gemma4UnifiedConfig, Gemma4UnifiedTextConfig, Gemma4UnifiedForConditionalGeneration
    torch.manual_seed(42)
    config = Gemma4UnifiedConfig(text_config=Gemma4UnifiedTextConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, num_global_key_value_heads=1,
        head_dim=16, global_head_dim=32, num_kv_shared_layers=shared_layers, attention_k_eq_v=True,
        layer_types=['sliding_attention', 'full_attention']*2, sliding_window=window,
        max_position_embeddings=window+128, final_logit_softcapping=30.,
        use_bidirectional_attention='vision',
        eos_token_id=3, bos_token_id=0, pad_token_id=3), attn_implementation=attention)
    base = Gemma4UnifiedForConditionalGeneration(config).to(dtype=dtype)
    # Nonzero configured dropout must remain disabled by eval in BOTH paths.
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=.2,
        task_type='CAUSAL_LM', target_modules=[
            'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])).eval()
    for p in lora_parameters(model).values():
        p.data = p.data.float()
    backend = HFGenerateBackend(model, Tokenizer(), base_checkpoint_hash='tiny-gemma4-unified',
        harness_hash='cpu-scoring', tokenizer_hash='integer-vocab', max_action_tokens=12,
        max_context_tokens=window+128, generation_batch=GenerationBatch(), score_tolerance=ScoreTolerance())
    enable_gradient_checkpointing(model)
    return backend


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('shared_layers', [0, 2])
@pytest.mark.parametrize('lengths', [(5, 7), (7, 8), (9, 12)])
def test_gemma4_scoring_sampling_logits_padding_positions_cache_and_softcap(
        dtype, shared_layers, lengths, monkeypatch):
    backend = tiny_gemma4_backend(dtype, shared_layers=shared_layers)
    model, base = backend.model, backend.model.get_base_model()
    resident = snapshot(lora_parameters(model))
    parameters = {k: v+.03 for k, v in resident.items()}
    raw_logits, calls, probabilities, outputs = [], [], [], []
    def observe(module, args, kwargs, output):
        raw_logits.append(output.logits[:, -1].float().detach().clone())
        calls.append(dict(shape=tuple(kwargs['input_ids'].shape), use_cache=kwargs['use_cache'],
                          positions=kwargs['position_ids'].clone()))
    hook = base.register_forward_hook(observe, with_kwargs=True)
    multinomial, generate = torch.multinomial, model.generate
    def sample(probs, *args, **kwargs):
        probabilities.append(probs.clone())
        return multinomial(probs, *args, **kwargs)
    def capture(**kwargs):
        output = generate(**kwargs)
        outputs.append(output)
        return output
    monkeypatch.setattr(torch, 'multinomial', sample)
    monkeypatch.setattr(model, 'generate', capture)
    prompts = [tuple(i % 3 for i in range(n)) for n in lengths]
    try:
        groups = backend.sample_actions_batch(prompts, 2, parameters, torch.Generator().manual_seed(7))
    finally:
        hook.remove()
    output, = outputs
    assert calls[0]['shape'] == (4, max(lengths)) and all(c['use_cache'] for c in calls)
    assert all(c['shape'] == (4, 1) for c in calls[1:])
    assert type(output.past_key_values).__name__ == 'DynamicCache'
    assert not any(m.training for m in model.modules())
    for name, p in lora_parameters(model).items():
        torch.testing.assert_close(p, resident[name], rtol=0, atol=0)
    for row, prompt in enumerate([p for p in prompts for _ in range(2)]):
        assert calls[0]['positions'][row, -len(prompt):].tolist() == list(range(len(prompt)))
        for step, call in enumerate(calls[1:]):
            assert call['positions'][row].tolist() == [len(prompt)+step]
    assert len(raw_logits) == len(output.scores) == len(probabilities)
    for raw, scores, probs in zip(raw_logits, output.scores, probabilities):
        torch.testing.assert_close(raw, scores, rtol=0, atol=0)
        torch.testing.assert_close(scores.softmax(-1), probs, rtol=0, atol=0)
    for row, action in enumerate(a for group in groups for a in group):
        expected = [float(scores[row].log_softmax(-1)[token])
                    for scores, token in zip(output.scores, action.action_ids)]
        assert action.generation_token_logprobs == tuple(expected)
        # The scorer uses the same installed LoRA values, the exact token IDs,
        # and native model softcap; only batch/cache/sequence shapes differ.
        score, diagnostic = backend.checked_score_action(action, parameters,
            expected_prompt_ids=action.prompt_ids)
        assert diagnostic['passed']
        if dtype == torch.float32:
            assert diagnostic['max_abs_difference'] < 1e-4
        else:
            assert diagnostic['mean_abs_difference'] < .01
        assert diagnostic['scoring_backend']['logprob_dtype'] == 'torch.float32'
        assert diagnostic['scoring_backend']['attention'] == action.generation_metadata['attention'] == 'eager'
        assert diagnostic['scoring_backend']['cache_type'] is None
        assert action.generation_metadata['cache_type'] == 'DynamicCache'
        assert score.requires_grad


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_gemma4_scoring_crosses_production_1024_window(dtype):
    backend = tiny_gemma4_backend(dtype, shared_layers=0, window=1024)
    parameters = {k: v+.03 for k, v in snapshot(lora_parameters(backend.model)).items()}
    prompts = [tuple(i % 3 for i in range(n)) for n in (1021, 1023)]
    actions = backend.sample_actions_batch(prompts, 1, parameters, torch.Generator().manual_seed(7))
    for group in actions:
        action, = group
        assert len(action.prompt_ids)+len(action.action_ids)-1 > 1024
        # The production source check is no-grad and forward batching is off.
        with torch.no_grad():
            _, diagnostic = backend.checked_score_action(action, parameters,
                expected_prompt_ids=action.prompt_ids)
        if dtype == torch.float32:
            assert diagnostic['max_abs_difference'] < 1e-4
        else:
            assert diagnostic['mean_abs_difference'] < .01


@pytest.mark.parametrize('attention', ['eager', 'sdpa'])
@pytest.mark.parametrize('path', ['serial', 'batched_generation', 'batched_forward'])
def test_gemma4_scoring_reports_actual_attention(attention, path):
    backend = tiny_gemma4_backend(torch.float32, shared_layers=0, attention=attention)
    if path == 'serial':
        backend.generation_batch = None
    elif path == 'batched_forward':
        backend.generation_batch = GenerationBatch(forward_prompts_per_batch=2)
    parameters = snapshot(lora_parameters(backend.model))
    action = backend.sample_action('0 1 2 0 1', parameters, torch.Generator().manual_seed(7))
    with torch.no_grad():
        _, values, metadata = backend.score_action(action, parameters, return_details=True)
    # A literal 'eager' in a journal is not evidence of the runtime setting.
    assert action.generation_metadata['attention'] == attention
    assert metadata['attention'] == attention
    assert action.generation_metadata['cache_type'] == 'DynamicCache'
    assert metadata['cache_type'] is None
    torch.testing.assert_close(values, torch.tensor(action.generation_token_logprobs), rtol=0, atol=1e-4)
