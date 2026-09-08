"""Versioned batch configuration; obsolete empty-prior keys are rejected."""
from .caps import V11_CLASS_CAPS, V11_BUDGET_BASIS

REMOVED_KEYS = {'max_new_packages_per_decision', 'replay_prior_mass', 'insertion_fraction',
                'acquisition_entropy_temperature'}
V11_DEFAULTS = dict(method='rtd_v1_1', protocol_version='1.1.0', budget_basis=V11_BUDGET_BASIS,
    exposure_slots_per_window=40, max_new_packages_per_window=20, slots_per_step=40,
    acquisition_posterior_refresh='persistent', replay_bank_path='data/rtd/v1_1_bfcl',
    replay_public_cap_output_tokens_by_class=V11_CLASS_CAPS,
    replay_cap_scope='sealed_bank_class_content_envelope', output_root='results/rtd_v1_1',
    drift_reference_packages=4, value_noise_floor=1e-8)


def v11_config(config, canonical):
    if REMOVED_KEYS & config.keys():
        raise ValueError('obsolete v1.0 acquisition keys in v1.1: ' + ', '.join(sorted(REMOVED_KEYS & config.keys())))
    result = dict(config)
    for key, value in V11_DEFAULTS.items():
        result.setdefault(key, value)
    # slots_per_step is the actual reference/replay exposure, not a dead alias.
    if 'slots_per_step' not in config:
        result['slots_per_step'] = result['exposure_slots_per_window']
    E, K, rounds = result['exposure_slots_per_window'], result['max_new_packages_per_window'], result.get('rounds', 3)
    if type(E) is not int or E < 1 or type(K) is not int or not 0 <= K <= E or result['slots_per_step'] != E:
        raise ValueError('positive E, 0 <= K <= E, and slots_per_step=E required')
    if type(rounds) is not int or rounds not in (2, 3):
        raise ValueError('v1.1 supports rounds 2 or 3')
    result['rounds'] = rounds
    fractions = [0.10, 0.25, 0.50][:rounds]
    result.setdefault('budget_checkpoints_bank_fraction', fractions)
    if result['budget_checkpoints_bank_fraction'] != fractions:
        raise ValueError('budget checkpoints must match the two/three-round schedule')
    if type(result['drift_reference_packages']) is not int or result['drift_reference_packages'] < 1:
        raise ValueError('positive purchased drift reference count required')
    if not 0 < result['value_noise_floor'] <= 1:
        raise ValueError('positive finite insertion noise floor <= 1 required')
    frozen = {k: v for k, v in canonical.items() if k not in REMOVED_KEYS} | V11_DEFAULTS
    return result, frozen
