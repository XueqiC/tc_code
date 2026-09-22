#!/usr/bin/env python3
"""Launch one shared vLLM 0.27.1 server and publish its ready identity.

Run with envs/vllm-serve/.venv/bin/python. This supervisor stays in the
foreground; SIGINT/SIGTERM stop its own process group. A new --state-dir holds
server.log, server.json, hardware.json and port. Model/tokenizer are local,
merged snapshots. No installation or model export is performed.
"""
from __future__ import annotations

import argparse
import http.client
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid


def check_port(port):
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("an explicit port in 1..65535 is required")
    # Bind without SO_REUSEADDR/SO_REUSEPORT before doing any GPU work. vLLM
    # also binds without port reuse, closing the check/start race fail-closed.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise ValueError(f"port {port} is already listening or unavailable") from exc


def ready(port, model_name):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        connection.request("GET", "/v1/models")
        response = connection.getresponse()
        body = response.read()
        if response.status != 200:
            return False
        return [row["id"] for row in json.loads(body).get("data", [])] == [model_name]
    except (OSError, http.client.HTTPException, ValueError):
        return False
    finally:
        connection.close()


def launch(args):
    check_port(args.port)
    if not args.gpu or "," in args.gpu or args.gpu == "-1":
        raise ValueError("--gpu must select exactly one GPU index or UUID")
    if args.max_context_tokens < 257 or args.ready_timeout <= 0:
        raise ValueError("invalid context limit or readiness timeout")
    version = importlib.metadata.version("vllm")
    if version != "0.27.1":
        raise ValueError("run this launcher with envs/vllm-serve/.venv/bin/python (vLLM 0.27.1)")
    model = args.model.resolve(strict=True)
    tokenizer = (args.tokenizer or args.model).resolve(strict=True)
    state = args.state_dir.absolute()
    state.mkdir(parents=True, exist_ok=False)
    # Set before importing torch through bfas. The probe and child use the same
    # interpreter, physical device visibility, and software environment.
    # rai holds mixed cards, so CUDA's default FASTEST_FIRST order does not match
    # nvidia-smi: without PCI_BUS_ID, --gpu 4 selected a labmate's card instead.
    os.environ.update(CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=args.gpu, VLLM_USE_FLASHINFER_SAMPLER="0",
        PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        VLLM_CACHE_ROOT=str(state / "cache/vllm"),
        TORCHINDUCTOR_CACHE_DIR=str(state / "cache/inductor"),
        TRITON_CACHE_DIR=str(state / "cache/triton"))
    os.environ.pop("VLLM_ATTENTION_BACKEND", None)
    from bfas.rtd.benchmarks.alfworld_identity import model_identity, tokenizer_identity
    from bfas.rtd.benchmarks.alfworld_server import SERVER_VERSION, checked_server_identity
    from bfas.rtd.hardware import hardware_identity
    from bfas.rtd.persistence import atomic_json, digest

    hardware = hardware_identity(optional_packages=("peft",))
    name = "alfworld-" + uuid.uuid4().hex
    identity = dict(version=SERVER_VERSION, vllm_version=version,
        host="127.0.0.1", port=args.port, served_model_name=name,
        hardware=hardware, hardware_hash=digest(hardware["hard"]),
        model=model_identity(model), tokenizer=tokenizer_identity(tokenizer),
        max_context_tokens=args.max_context_tokens)
    identity["identity_hash"] = digest(identity)
    checked_server_identity(identity)
    command = [sys.executable, "-B", "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model), "--tokenizer", str(tokenizer),
        "--host", "127.0.0.1", "--port", str(args.port), "--served-model-name", name,
        "--dtype", "bfloat16", "--tensor-parallel-size", "1",
        "--max-model-len", str(args.max_context_tokens), "--generation-config", "vllm",
        "--no-enable-log-requests"]
    def stopped(signum, frame):
        raise KeyboardInterrupt
    previous = {sig: signal.signal(sig, stopped) for sig in (signal.SIGINT, signal.SIGTERM)}
    process = None
    try:
        with (state / "server.log").open("x") as log:
            process = subprocess.Popen(command, cwd=state, env=dict(os.environ),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + args.ready_timeout
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"vLLM exited before readiness; see {state / 'server.log'}")
                if ready(args.port, name) and process.poll() is None:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"vLLM readiness timeout; see {state / 'server.log'}")
                time.sleep(1)
            atomic_json(state / "hardware.json", hardware)
            (state / "port").write_text(f"{args.port}\n")
            # Publish last: presence means the exact launch was ready.
            atomic_json(state / "server.json", identity)
            print(f"Ready: {state / 'server.json'}", flush=True)
            return process.wait()
    except KeyboardInterrupt:
        return 130
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--state-dir", type=Path, required=True, help="new private directory outside model/env trees")
    parser.add_argument("--max-context-tokens", type=int, default=32768)
    parser.add_argument("--ready-timeout", type=float, default=900)
    return launch(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
