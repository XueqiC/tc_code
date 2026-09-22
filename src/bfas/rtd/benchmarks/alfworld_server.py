"""Read-only binding to the generator's launch identity, never client CUDA."""
from copy import deepcopy

from ..hardware import checked_hardware
from ..persistence import digest


SERVER_VERSION = "alfworld-vllm-server-v1"


def checked_server_identity(server, *, manifest=None):
    if (server.get("version") != SERVER_VERSION or
            server.get("vllm_version") != "0.27.1" or
            server.get("host") != "127.0.0.1" or
            type(server.get("port")) is not int or not 1 <= server["port"] <= 65535 or
            not isinstance(server.get("served_model_name"), str) or
            not server["served_model_name"].startswith("alfworld-") or
            type(server.get("max_context_tokens")) is not int or server["max_context_tokens"] < 257 or
            server.get("identity_hash") != digest({k: v for k, v in server.items() if k != "identity_hash"})):
        raise ValueError("invalid vLLM server launch identity")
    hardware = checked_hardware(server["hardware"])
    if server.get("hardware_hash") != digest(hardware["hard"]):
        raise ValueError("server hardware class hash mismatch")
    if manifest is not None:
        if (hardware["hard"] != checked_hardware(manifest["hardware"])["hard"] or
                server["hardware_hash"] != manifest["hardware_hash"]):
            raise ValueError("server hardware class differs from campaign")
        if (manifest["paths"].get("checkpoint") or
                server.get("model") != manifest["checkpoint"] or
                server.get("tokenizer") != manifest["evaluation_harness"]["tokenizer"] or
                server["max_context_tokens"] != manifest["config"]["max_context_tokens"]):
            raise ValueError("server model/tokenizer/context differs; require the bound merged snapshot")
    return deepcopy(server)
