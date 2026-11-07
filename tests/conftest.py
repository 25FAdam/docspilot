import socket

import pytest


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    """Tests are offline: any real socket connection is a bug."""

    def guard(self, *args, **kwargs):
        raise RuntimeError("network access is blocked in tests")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket.socket, "connect_ex", guard)
