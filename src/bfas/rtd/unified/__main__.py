"""Validate/describe a P0 configuration on CPU; P1-P3 experiments are planned."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import yaml

from .engine import UnifiedConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = UnifiedConfig.from_config(yaml.safe_load(args.config.read_text()))
    print(json.dumps(dict(status='P0; experiments PLANNED', selected=asdict(config)), indent=2))


if __name__ == '__main__':
    main()
