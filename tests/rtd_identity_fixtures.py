"""Real source selectors in small synthetic BFCL checkouts; no code execution."""
from pathlib import Path

from bfas.rtd.identity import EVALUATION_TOOLS
from bfas.rtd.scoring_scope import PYTHON_SCOPES, SHELL


ROOT = Path(__file__).resolve().parents[1]


def tool_content(name):
    return (ROOT/name).read_text() if name in {*PYTHON_SCOPES, SHELL} else 'fixture'


def put_tools(root):
    for name in EVALUATION_TOOLS:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tool_content(name))


def hardware_fixture():
    from bfas.rtd.hardware import VERSION
    return dict(version=VERSION, hard=dict(
        gpu='NVIDIA B200', capability=[10, 0], memory=192_000_000_000, cuda='13.0',
        driver='580.95.05', versions={n: 'fixture' for n in ('torch', 'transformers', 'peft', 'numpy')},
        python='3.12.13', machine='x86_64', host_class='hpg-b200'),
        metadata=dict(hostname='c1100a-s25', uuid='4200c43f-original',
                      pci_bus_id='0000:41:00.0', cuda_device_order='PCI_BUS_ID'))
