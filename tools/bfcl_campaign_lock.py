#!/usr/bin/env python3
"""Lease campaign resources before the shell starts export or generation."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bfas.rtd.evaluation_lock import (evaluation_lock, inherited_lock, port_lock_path,
                                     reserve_port, tag_lock_path)


def main(argv):
    gpu, requested, *tags = argv
    port = None if requested == 'auto' else int(requested)
    leaderboard = ROOT / 'envs/bfcl/gorilla/berkeley-function-call-leaderboard'
    # BFCL calls load_dotenv(override=True). Refuse settings that would silently
    # redirect a leased port/GPU or erase the listener's ownership token.
    dotenv = leaderboard / '.env'
    if dotenv.exists():
        protected = re.findall(
            r"^\s*(?:export\s+)?['\"]?(LOCAL_SERVER_PORT|LOCAL_SERVER_ENDPOINT|CUDA_VISIBLE_DEVICES|"
            r"CUDA_DEVICE_ORDER|BFCL_PROJECT_ROOT|BFCLSTD_\w+)['\"]?\s*=", dotenv.read_text(), re.M)
        if protected:
            raise ValueError('campaign resource overrides in BFCL .env: ' + ', '.join(sorted(set(protected))))
    status = 0
    for tag in tags:
        with ExitStack() as stack:
            tag_path = tag_lock_path(ROOT, tag)
            if 'BFCLSTD_TAG_LOCK_FD' in os.environ:
                tag_fd = inherited_lock(os.environ['BFCLSTD_TAG_LOCK_FD'], tag_path)
                port_fd = inherited_lock(os.environ['BFCLSTD_PORT_LOCK_FD'], port_lock_path(ROOT, port))
                selected = port
            else:
                tag_fd = stack.enter_context(evaluation_lock(tag_path, tag=tag))
                selected, port_fd = stack.enter_context(reserve_port(ROOT, tag=tag, port=port, gpu_uuid=gpu))
            attempt = uuid.uuid4().hex
            result, score = f'result_p{selected}_{attempt}', f'score_p{selected}_{attempt}'
            # Exclusive creation: even a UUID collision must fail closed.
            (leaderboard / result).mkdir()
            (leaderboard / score).mkdir()
            env = dict(os.environ, BFCLSTD_LOCKED='1', BFCLSTD_RUN_ID=attempt,
                       BFCLSTD_RESULT_SUB=result, BFCLSTD_SCORE_SUB=score,
                       BFCLSTD_TAG_LOCK_FD=str(tag_fd), BFCLSTD_PORT_LOCK_FD=str(port_fd),
                       LOCAL_SERVER_ENDPOINT='127.0.0.1', LOCAL_SERVER_PORT=str(selected),
                       BFCL_PROJECT_ROOT=str(leaderboard))
            print('[bfclstd] resources ' + json.dumps(dict(tag=tag, port=selected,
                  attempt=attempt, result_dir=result, score_dir=score)), flush=True)
            completed = subprocess.run(['bash', str(ROOT / 'tools/bfcl_std_campaign.sh'),
                gpu, str(selected), tag], env=env, pass_fds=(tag_fd, port_fd))
            if completed.returncode:
                status = completed.returncode
    return status


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
