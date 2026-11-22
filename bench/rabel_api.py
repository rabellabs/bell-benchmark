"""The Rabel API, as the benchmark reaches it: Bell's guesses come from the API itself, never from anything in this repository.

The key comes from the RABEL_API_KEY environment variable and is never printed or written; the address from RABEL_API_URL
(default https://api.rabellabs.com).

  - `http(method, path, body)`: one HTTPS request (agents, teaching, usage).
  - `Conversation(hello)`: one conversation on /v1/conversation (contract version 2) over a plain blocking socket. It uses websockets'
    sans-I/O protocol, with no thread and no timer of its own, so it can run inside the replay clock (`hosts/vclock.py`): while it
    waits for the API, the replay's clock stands still, and the real time the answer took is measured on the real clock and handed
    back, so the replay delivers the answer that much later on its own clock.
  - `Conversation.ask(message)`: sends one message and returns every answer to it, each with its real round trip. The API answers
    a `heard` with a guess or with nothing, so after the `heard` the benchmark sends a `ping`: the API answers it with `pong` once
    every answer to the messages before it has gone out, so once the `pong` is back, every answer to the `heard` is in. The `pong`
    is never handed to the hookup. An `end_of_turn` always has its own answer and needs no `ping`.
"""
from __future__ import annotations

import collections
import json
import os
import socket
import ssl
import time
import urllib.error
import urllib.request

PING = {"type": "ping"}          # answered with pong after every answer to the messages before it; it teaches nothing


def base_url() -> str:
    return os.environ.get("RABEL_API_URL", "https://api.rabellabs.com").rstrip("/")


def api_key() -> str:
    k = os.environ.get("RABEL_API_KEY", "").strip()
    if not k:
        raise SystemExit("RABEL_API_KEY is not set (a Rabel API key; it is read from the environment only)")
    return k


def http(method: str, path: str, body=None, timeout: float = 300.0):
    """-> (status, the answer's JSON or None)"""
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(base_url() + path, data=data, method=method,
                                 headers={"Authorization": "Bearer " + api_key(), "Content-Type": "application/json",
                                          "User-Agent": "rabel-bell-benchmark"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            status = r.status
    except urllib.error.HTTPError as e:
        raw, status = e.read(), e.code
    try:
        return status, json.loads(raw.decode("utf-8")) if raw else None
    except ValueError:
        return status, None


class Closed(ConnectionError):
    pass


def _sleep_real(s):
    """a wait on the operating system's clock (time.sleep, untouched by the replay clock)"""
    time.sleep(s)


class Conversation:
    """one conversation on /v1/conversation, blocking. clock: the real clock the round trips are read on (inside the replay, the
    wall's perf_counter kept by vclock.REAL, since time.perf_counter then reads the replay's clock)."""

    OPEN_TRIES = 3          # a conversation that cannot be opened (the network or the front stalled) is tried again, 1 s and 3 s later

    def __init__(self, hello: dict, timeout: float = 30.0, clock=time.perf_counter):
        self.open_retries = 0
        for i in range(self.OPEN_TRIES):
            try:
                self._open(hello, timeout, clock)
                return
            except (OSError, Closed):
                if i == self.OPEN_TRIES - 1:
                    raise
                self.open_retries += 1
                _sleep_real((1.0, 3.0)[min(i, 1)])

    def _open(self, hello, timeout, clock):
        from websockets.client import ClientProtocol
        from websockets.protocol import State
        from websockets.uri import parse_uri
        self.clock = clock
        self._State = State
        url = base_url().replace("https://", "wss://").replace("http://", "ws://") + "/v1/conversation"
        u = parse_uri(url)
        self.p = ClientProtocol(u, max_size=None)
        sock = socket.create_connection((u.host, u.port), timeout=timeout)
        if u.secure:
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=u.host)
        sock.settimeout(timeout)
        self.sock = sock
        req = self.p.connect()
        req.headers["Authorization"] = "Bearer " + api_key()
        req.headers["User-Agent"] = "rabel-bell-benchmark"
        self.p.send_request(req)
        self._flush()
        self.inbox: collections.deque = collections.deque()
        self._frag = b""
        while self.p.state is not State.OPEN:
            self._read()
            if self.p.handshake_exc is not None:
                raise Closed(f"the API refused the conversation: {self.p.handshake_exc}")
        hello = {k: v for k, v in hello.items() if k != "key"}
        hello["type"] = "hello"
        self.send(hello)
        self.ready = self.recv()[1]
        if self.ready.get("type") != "ready":
            raise Closed(f"the API did not answer ready: {self.ready.get('type')} {self.ready.get('error')}")
        self.pongs = 0

    def _flush(self):
        for chunk in self.p.data_to_send():
            if chunk:
                self.sock.sendall(chunk)

    def _read(self):
        from websockets.frames import Opcode
        data = self.sock.recv(65536)
        if not data:
            self.p.receive_eof()
        else:
            self.p.receive_data(data)
        for ev in self.p.events_received():
            op = getattr(ev, "opcode", None)
            if op is Opcode.TEXT or op is Opcode.CONT:
                self._frag += ev.data
                if ev.fin:
                    self.inbox.append((self.clock(), json.loads(self._frag.decode("utf-8"))))
                    self._frag = b""
            elif op is Opcode.CLOSE:
                self._flush()
                raise Closed(f"the API closed the conversation ({self.p.close_code})")
        self._flush()
        if not data:
            raise Closed("the API's socket ended")

    def send(self, msg: dict):
        self.p.send_text(json.dumps(msg).encode("utf-8"))
        self._flush()

    def recv(self):
        """-> (its arrival on the real clock, the message)"""
        while not self.inbox:
            self._read()
        return self.inbox.popleft()

    def ask(self, msg: dict):
        """-> [(round trip s, answer)]: every answer to this message, each with the real time from its sending to its arrival"""
        t0 = self.clock()
        self.send(msg)
        kind = msg.get("type")
        if kind == "heard":
            self.send(PING)
        out = []
        while True:
            at, m = self.recv()
            if kind == "heard" and m.get("type") == "pong":
                self.pongs += 1
                return out
            out.append((at - t0, m))
            if kind == "end_of_turn" and m.get("type") == "end_of_turn":
                return out
            if m.get("type") == "error":
                raise Closed(f"the API refused: {m.get('status')} {m.get('error')}")

    def close(self):
        try:
            if self.p.state is self._State.OPEN:
                self.p.send_close(1000)
                self._flush()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.sock.close()
        except Exception:  # noqa: BLE001
            pass
