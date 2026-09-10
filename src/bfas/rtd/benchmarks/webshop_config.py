"""WebShop uses the common RTD v1.1 validator and bank audit."""
from .config import data_identity


def validate_config(config):
    from ..cli import validate_config as validate
    if config.get('benchmark') != 'webshop':
        raise ValueError('benchmark: webshop required')
    return validate(config)


def bank_audit(root, config):
    from ..bank_build import validate_state_certificate
    from pathlib import Path
    bank = Path(root)/config['replay_bank_path']
    return validate_state_certificate(bank, benchmark='webshop', student=config['student'])
