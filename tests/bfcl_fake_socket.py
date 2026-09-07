"""Test-only socket boundary: the managed sandbox forbids even local sockets.

flock, descriptors, shell processes and BFCL fake runs remain real. Only the
kernel free/occupied-port probe is simulated; no production bypass is added.
"""
import errno
import os
from pathlib import Path
import shutil
import socket
from types import SimpleNamespace


class FakeSocket:
    bound = set()
    next_port = 50000 + os.getpid() % 10000

    def __init__(self):
        self.port = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.port is not None:
            self.bound.remove(self.port)

    def bind(self, address):
        port = address[1]
        if not port:
            while self.next_port in self.bound:
                type(self).next_port += 1
            port = self.next_port
            type(self).next_port += 1
        if port in self.bound or str(port) == os.environ.get('TEST_BUSY_PORT'):
            raise OSError(errno.EADDRINUSE, 'Address already in use')
        self.bound.add(port)
        self.port = port

    def getsockname(self):
        return '0.0.0.0', self.port

    def listen(self):
        pass


def install_fake_socket(monkeypatch=None):
    from bfas.rtd import evaluation_lock
    fake = SimpleNamespace(socket=FakeSocket, gethostname=socket.gethostname)
    if monkeypatch:
        monkeypatch.setattr(evaluation_lock, 'socket', fake)
    else:
        evaluation_lock.socket = fake


def fake_network_pythonpath(root):
    support = root / 'fake_network'
    support.mkdir()
    shutil.copy(__file__, support / 'sitecustomize.py')
    return f'{support}:{root / "src"}'


if __name__ == 'sitecustomize':
    install_fake_socket()
