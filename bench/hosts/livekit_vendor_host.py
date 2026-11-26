"""The library `livekit_default_host.py` builds on: the recorded socket, the watch of LiveKit's own early replies, the call's run
state and the per-turn read.

How the recording is fed: LiveKit's own Flux plugin runs as it ships (its own send, receive and event code); only its `_connect_ws`
is replaced, returning a local socket that hands the plugin's receive loop the recorded messages at their recorded arrival times,
answers its pings, and closes when the plugin sends its close. Nothing in the plugin decides differently from a live call: the same
messages arrive in the same order and spacing.

One JSON line per turn: key, conv, calls, early_made, kept ("livekit", "bell" or None), head (ms), guesses Bell sent.
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys
import time

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

GAP_S = 3.0
LEAD_S = 2.0


# ---------------------------------------------------------------- the recorded socket

class FakeWS:
    """what the plugin's receive loop reads: the recorded messages at their times, pongs to its pings, a close after its close"""

    def __init__(self, call, vendor):
        import aiohttp
        self._aiohttp = aiohttp
        self.call, self.vendor = call, vendor
        self.q: asyncio.Queue = asyncio.Queue()
        self.closed = False
        self.close_code = None
        self.sent_text, self.sent_bytes = [], 0
        self.feeder = asyncio.ensure_future(self._feed())

    def _msg(self, kind, data=None):
        return self._aiohttp.WSMessage(kind, data, None)

    async def _feed(self):
        c = self.call
        c.origin = time.time() + LEAD_S
        for tr in c.turns:
            for (t_ms, m) in tr["sched"]:
                wait = c.origin + t_ms / 1000.0 - time.time()
                if wait > 0:
                    await asyncio.sleep(wait)
                now = time.time()
                c.lag.append((now - (c.origin + t_ms / 1000.0)) * 1000.0)
                if m["_k"] == 0:
                    tr["first_wall"] = now
                if m.get("_is_end"):
                    tr["eot_wall"] = now
                c.cur = tr["key"]
                payload = {k: v for k, v in m.items() if not k.startswith("_")}
                await self.q.put(self._msg(self._aiohttp.WSMsgType.TEXT, json.dumps(payload)))
                await asyncio.sleep(0)
        c.done_wall = time.time()
        c.done.set()

    async def receive(self, timeout=None):
        return await self.q.get()

    async def send_str(self, data, compress=None):
        """the plugin's text messages are kept (their start); the close is the subclass's (Flux's CloseStream)"""
        self.sent_text.append(data[:100])

    async def send_bytes(self, data, compress=None):
        self.sent_bytes += len(data)

    async def ping(self, message=b""):
        await self.q.put(self._msg(self._aiohttp.WSMsgType.PONG, message))

    async def pong(self, message=b""):
        return None

    async def close(self, *, code=1000, message=b""):
        if not self.closed:
            self.closed = True
            self.close_code = code
            self.feeder.cancel()
            await self.q.put(self._msg(self._aiohttp.WSMsgType.CLOSED))
        return True

    def exception(self):
        return None

    def __aiter__(self):
        return self

    async def __anext__(self):
        m = await self.receive()
        if m.type in (self._aiohttp.WSMsgType.CLOSE, self._aiohttp.WSMsgType.CLOSED):
            raise StopAsyncIteration
        return m


CALLS_BY_STT: dict = {}
BELL_BY_STT: dict = {}


def watch_bell_emits():
    """the hookup writes its guess in the final's form, so which early reply is Bell's is read from the preflights the hookup itself
    emitted, not from the API's words. Observed where the hookup makes them; nothing in it is changed."""
    from livekit.plugins.rabel import stt as _RS
    orig = _RS.SpeechStream._take_guess

    def _take(self, m, *a, **kw):   # the hookup also takes a guess again (`released=`: rule 8's held guess, 1.9.0's waiting one)
        ev = orig(self, m, *a, **kw)
        c = BELL_BY_STT.get(id(self._rstt))
        if ev is not None and c is not None:
            c.bell_emits.append((time.time(), ev.alternatives[0].text))
        return ev
    _RS.SpeechStream._take_guess = _take


def hold_vendor_early(stt_obj):
    """arms nothing and bell: the vendor's PREFLIGHT_TRANSCRIPT passed on as an INTERIM_TRANSCRIPT (the event is held back, its words
    are not); wraps the plugin's stream() so each stream's event channel converts it"""
    from livekit.agents import stt as lkstt
    orig = stt_obj.stream

    def stream(*a, **kw):
        s = orig(*a, **kw)
        ch = s._event_ch
        send = ch.send_nowait

        def conv(ev):
            if ev.type == lkstt.SpeechEventType.PREFLIGHT_TRANSCRIPT:
                ev = lkstt.SpeechEvent(type=lkstt.SpeechEventType.INTERIM_TRANSCRIPT, request_id=ev.request_id,
                                       alternatives=ev.alternatives)
            return send(ev)
        ch.send_nowait = conv
        return s
    stt_obj.stream = stream
    return stt_obj


# ---------------------------------------------------------------- one call

class CallRun:
    def __init__(self, clip, turns, http, continuous=False):
        """continuous (the livekit mode): the whole call at its recorded times, the audio beside it; else each turn alone, GAP_S
        apart"""
        self.clip, self.http = clip, http
        self.turns = turns
        t_off = 0.0
        for tr in turns:
            raw = tr["raw"]
            t0 = 0.0 if continuous else raw[0][0]
            sched = []
            for k, (t, m) in enumerate(raw):
                m = dict(m)
                m["_k"] = k
                m["_is_end"] = k == len(raw) - 1
                sched.append((t_off + (t - t0), m))
            tr["sched"] = sched
            if not continuous:
                t_off += (tr["eot_ms"] - t0) + GAP_S * 1000.0
        self.sched_times = []
        self.ws = None
        self.cur = None
        self.done = asyncio.Event()
        self.bell_ready = asyncio.Event()
        self.origin = None
        self.lag = []
        self.heard = 0
        self.guesses = []
        self.bell_emits = []  # Bell's preflights as the hookup emitted them (time, text): which early reply is Bell's is read from these
        self.pre_calls, self.created, self.scheduled, self.held = [], [], set(), []


SESSIONS: dict = {}


def patch_livekit(arm):
    """a watch of LiveKit's own record of each early reply; under `held`, a repeated early start with equivalent words is held back"""
    from livekit.agents.voice import agent_activity as _AA
    orig_pre = _AA.AgentActivity.on_preemptive_generation
    orig_sched = _AA.AgentActivity._schedule_speech
    equiv = _AA._transcripts_equivalent

    def _pre(self, info):
        c = SESSIONS.get(id(self._session))
        run = self._preemptive_generation
        if arm == "held" and run is not None and equiv(run.info.new_transcript, info.new_transcript):
            if c is not None:
                c.held.append((time.time(), info.new_transcript))
            return
        before = run
        orig_pre(self, info)
        after = self._preemptive_generation
        if c is not None:
            c.pre_calls.append((time.time(), info.new_transcript))
            if after is not None and after is not before:
                c.created.append((after.created_at, info.new_transcript, id(after.speech_handle)))

    def _sched(self, speech, *a, **kw):
        c = SESSIONS.get(id(self._session))
        if c is not None:
            c.scheduled.add(id(speech))
            c.sched_times.append((time.time(), id(speech)))
        return orig_sched(self, speech, *a, **kw)

    _AA.AgentActivity.on_preemptive_generation = _pre
    _AA.AgentActivity._schedule_speech = _sched


def stub_classes():
    from livekit import rtc
    from livekit.agents import utils
    from livekit.agents.llm import LLM, ChatChunk, ChoiceDelta, LLMStream
    from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
    from livekit.agents.voice.io import AudioInput

    class StubStream(LLMStream):
        async def _run(self) -> None:
            self._llm_owner.log.append((time.time(), self._words, id(self)))
            self._event_ch.send_nowait(ChatChunk(id=str(id(self)), delta=ChoiceDelta(role="assistant", content="Okay.")))

    class StubLLM(LLM):
        def __init__(self):
            super().__init__()
            self.log = []

        @property
        def model(self):
            return "stand-in"

        @property
        def provider(self):
            return "stand-in"

        def chat(self, *, chat_ctx, tools=None, conn_options=DEFAULT_API_CONNECT_OPTIONS, **kw):
            s = StubStream(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)
            s._llm_owner = self
            last = [i for i in chat_ctx.items if i.type == "message" and i.role == "user"]
            s._words = last[-1].text_content if last else ""
            return s

    class OneFrameInput(AudioInput):
        def __init__(self):
            super().__init__(label="replay")
            self._ch = utils.aio.Chan[rtc.AudioFrame]()
            n = 160
            self._ch.send_nowait(rtc.AudioFrame(data=b"\x00\x00" * n, sample_rate=8000, num_channels=1, samples_per_channel=n))

        async def __anext__(self):
            return await self._ch.__anext__()

    return StubLLM, OneFrameInput


def per_turn(c, llm, livekit_mode=False):
    out = []
    for i, tr in enumerate(c.turns):
        lo = tr.get("first_wall")
        if lo is None or "eot_wall" not in tr:
            out.append({"key": tr["key"], "conv": tr["conv"], "fault": "not fed"})
            continue
        hi = c.turns[i + 1].get("first_wall", c.done_wall + GAP_S) if i + 1 < len(c.turns) else c.done_wall + GAP_S
        eot = tr["eot_wall"]
        commit = None
        if livekit_mode:  # LiveKit's own end of turn: the moment it scheduled the reply in this turn's window
            st = [t for t, _ in c.sched_times if lo <= t < hi]
            commit = st[0] if st else None
            eot = commit if commit is not None else eot
        calls = [x for x in llm.log if lo <= x[0] < hi]
        made = [x for x in c.created if lo <= x[0] < hi]
        kept_made = [x for x in made if x[2] in c.scheduled]
        guesses = {g[2] for g in c.guesses if g[1] == tr["key"]}
        emits = [(t, x) for t, x in c.bell_emits if lo <= t < hi]
        kept, head, head_first_commit, kept_commit = None, None, None, None
        if kept_made:
            head_first_commit = (eot - kept_made[-1][0]) * 1000.0  # the head from the first commit in the window, kept beside
            k = kept_made[-1]
            # Bell's when LiveKit made it from a preflight the hookup emitted as Bell's: its text, within 100 ms after it
            kept = "bell" if any(x == k[1] or k[1].endswith(" " + x) for t, x in emits if 0 <= k[0] - t <= 0.1) else (
                "livekit" if livekit_mode else "vendor")   # livekit mode: LiveKit's own start on the final words
            if livekit_mode:
                # LiveKit's own commit of THIS reply: when LiveKit commits more than once in the vendor's turn (a pause
                # mid-turn), the first commit in the window is a different reply, and measuring from it gave heads of minus tens of
                # seconds; the head is the kept reply's start before the commit that scheduled it
                ks = [t for t, sid in c.sched_times if sid == k[2]]
                head = ((ks[0] if ks else eot) - k[0]) * 1000.0
                kept_commit = (ks[0] - tr["eot_wall"]) * 1000.0 if ks else None   # the kept reply's commit after the speech-to-text's end
            else:
                head = (eot - k[0]) * 1000.0
        out.append({"key": tr["key"], "conv": tr["conv"], "calls": len(calls), "early_made": len(made), "kept": kept, "head": head,
                    "kept_count": len(kept_made), "made_after_eot": sum(1 for x in made if x[0] > eot),
                    "vendor_early_in_recording": len(tr["early"]), "bell_guesses": len(guesses), "bell_emits": len(emits),
                    "kept_bell_no_restart": bool(emits) and len(made) == 1 and kept == "bell",
                    "preemptive_events": sum(1 for x in c.pre_calls if lo <= x[0] < hi),
                    "repeat_held": sum(1 for x in c.held if lo <= x[0] < hi),
                    **({"commit_after_vendor_end_ms": None if commit is None else (commit - tr["eot_wall"]) * 1000.0,
                        "commits_in_window": sum(1 for t, _ in c.sched_times if lo <= t < hi),
                        "head_first_commit": head_first_commit, "kept_commit_after_vendor_end_ms": kept_commit}
                       if livekit_mode else {})})
    return out


def _has(md, p):
    try:
        md.version(p)
        return True
    except Exception:  # noqa: BLE001
        return False
