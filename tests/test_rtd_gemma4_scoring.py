"""Offline tiny Gemma 4 multimodal-wrapper oracle for the production score paths."""
import pytest
import torch

from bfas.rtd.functional_step import lora_parameters, snapshot
from bfas.rtd.generation_batch import GenerationBatch
from bfas.rtd.runtime import HFGenerateBackend
from bfas.rtd.scoring import ScoreTolerance
from test_rtd_v11_generation_batch import Tokenizer


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_gemma4_scoring_sampling_logits_padding_positions_cache_and_softcap(dtype, monkeypatch):
    from peft import LoraConfig, get_peft_model
    from transformers import Gemma4UnifiedConfig, Gemma4UnifiedTextConfig, Gemma4UnifiedForConditionalGeneration
    torch.manual_seed(42)
    config = Gemma4UnifiedConfig(text_config=Gemma4UnifiedTextConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, num_global_key_value_heads=1,
        head_dim=16, global_head_dim=16, num_kv_shared_layers=2, attention_k_eq_v=True,
        layer_types=['sliding_attention', 'full_attention']*2, sliding_window=8,
        max_position_embeddings=128, final_logit_softcapping=10.,
        eos_token_id=3, bos_token_id=0, pad_token_id=3), attn_implementation='eager')
    base = Gemma4UnifiedForConditionalGeneration(config).to(dtype=dtype)
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.,
        task_type='CAUSAL_LM', target_modules=['q_proj', 'v_proj', 'o_proj'])).eval()
    for p in lora_parameters(model).values():
        p.data = p.data.float()
    backend = HFGenerateBackend(model, Tokenizer(), base_checkpoint_hash='tiny-gemma4-unified',
        harness_hash='cpu-scoring', tokenizer_hash='integer-vocab', max_action_tokens=12,
        max_context_tokens=128, generation_batch=GenerationBatch(), score_tolerance=ScoreTolerance())
    parameters = {k: v+.03 for k, v in snapshot(lora_parameters(model)).items()}
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
    prompts = [(0, 1, 2)*3, (0, 1, 2)*4]
    try:
        groups = backend.sample_actions_batch(prompts, 2, parameters, torch.Generator().manual_seed(7))
    finally:
        hook.remove()
    output, = outputs
    assert calls[0]['shape'] == (4, 12) and all(c['use_cache'] for c in calls)
    assert all(c['shape'] == (4, 1) for c in calls[1:])
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
            assert diagnostic['max_abs_difference'] < 2e-5
        assert diagnostic['scoring_backend']['logprob_dtype'] == 'torch.float32'
        assert score.requires_grad
