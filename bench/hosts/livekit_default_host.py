#!/usr/bin/env python3
"""LiveKit's default ending on the replay clock: recorded speech-to-text messages and the call's audio replayed into a real
LiveKit 1.8.13 AgentSession at its defaults (its VAD, its audio turn detector `turn-detector-v1-mini` run locally, its endpointing
0.3 s / 2.5 s, preemptive generation on), with the stand-in LLM, and Bell (the hookup, `livekit-plugins-rabel`) on or off.

It builds on `livekit_vendor_host.py` (the recorded socket, the watch and the per-turn read):
  - `--clock replay` (the default): every call is replayed on the replay clock of `vclock.py`, alone in its own event loop, the clock
    at the same reading at its start. LiveKit's timers, its VAD and its turn detector all follow that clock, and their inference runs
    in the order LiveKit asks for it; so the same call replayed twice gets the same answers at the same moments. `--clock wall` runs
    calls at once on the wall clock with inference on threads, kept to show the spread this removes.
  - Flux's recorded words (`--vendor flux`, the 28 test callers' calls recorded with Flux's eager end of turn at 0.5; eager is OFF
    here: its EagerEndOfTurn and TurnResumed messages are not fed, and Flux sends neither with eager off), through LiveKit's own
    Flux plugin (`deepgram.STTv2` at its defaults, 8 kHz) with only its socket replaced.
  - Bell's guesses come from the Rabel API itself (RABEL_API_URL, RABEL_API_KEY; `--agent`: demo-bank, or an agent of your own
    account). The hookup's WebSocket goes to a relay in the process (`ApiBell`), which hands each message to the API and waits for its
    answer with the replay's clock standing still; each answer reaches the hookup its real round trip later on the replay's clock.
    So a guess arrives as late as the API really answered it from this machine, and the replay still never waits on the wall.
  - `--fault-detector-ms N`: the planted fault, the turn detector's answer handed back N ms late (its audio window is taken when it is
    asked, as LiveKit does; only the answer waits).

Arms: lk (LiveKit's default alone), lk_bell (with Bell). One JSON line per turn, the vendor host's fields (livekit mode) plus
`kept_commit_after_vendor_end_ms` and the LiveKit commits in the turn's window.

  python bench/hosts/livekit_default_host.py --vendor flux --events EVENTS_28.jsonl --sched SCHED_0.5.json --audio DIR --arm lk \
      --out OUT.jsonl [--agent demo-bank] [--procs 8] [--clock replay|wall] [--conc 10] [--calls N] [--fault-detector-ms 100]
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import multiprocessing as mp
import os
import pathlib
import sys
import time

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import vclock  # noqa: E402

vclock.install()      # before LiveKit, aiohttp or the hookup are imported; reads the wall clock until a replay starts
import livekit_vendor_host as VH  # noqa: E402

os.environ.setdefault("DEEPGRAM_API_KEY", "replay-only")
GAP_S = VH.GAP_S
ARMS = {"livekit": ("lk", "lk_bell"), "stt": ("nothing", "bell", "passed", "both")}
BELL_ARMS = ("lk_bell", "bell", "both")
EAGER_ARMS = ("passed", "both")
FLUX_DROPPED = ("EagerEndOfTurn", "TurnResumed")


# ---------------------------------------------------------------- Flux's recorded words, eager off

def load_flux(a):
    """the 28's turns as the schedule file reads them (2,193), each with every recorded message after the previous read turn's end up to
    its own first EndOfTurn (the eager events left out), and the call's audio file"""
    sched = json.load(open(a.sched, encoding="utf-8"))
    want = collections.defaultdict(dict)
    for tr in sched["turns"]:
        want[tr["clip"]][tr["ti"]] = tr
    keep = None
    if a.clips:
        keep = {ln.strip() for ln in open(a.clips, encoding="utf-8") if ln.strip()}
    by, cnt = {}, collections.Counter()
    for ln in open(a.events, encoding="utf-8"):
        if not ln.strip():
            continue
        rec = json.loads(ln)
        clip = rec.get("clip")
        if "error" in rec or clip not in want:
            cnt["records_skipped"] += 1
            continue
        if keep is not None and clip not in keep and rec.get("conversation") not in keep:
            continue
        msgs = []
        for t, m in rec.get("msgs", []):
            if not isinstance(m, dict):
                continue
            if m.get("type") == "TurnInfo" and m.get("event") in FLUX_DROPPED and not (a.turn == "stt" and a.arm in EAGER_ARMS):
                cnt["eager_messages_not_fed"] += 1
                continue
            msgs.append((float(t), m))
        ends = {}
        for i, (t, m) in enumerate(msgs):
            if m.get("type") == "TurnInfo" and m.get("event") == "EndOfTurn" and m.get("turn_index") not in ends:
                ends[m["turn_index"]] = i
        turns, prev = [], -1
        for ti in sorted(want[clip]):
            tr = want[clip][ti]
            i = ends.get(ti)
            if i is None or i <= prev:
                cnt["turn_not_found"] += 1
                continue
            raw = msgs[prev + 1: i + 1]
            assert abs(raw[-1][0] - tr["eot_ms"]) < 1e-6, tr["key"]
            turns.append({"key": tr["key"], "clip": clip, "ti": ti, "conv": tr["conv"], "raw": raw, "eot_ms": tr["eot_ms"],
                          "early": [], "path": str(pathlib.Path(a.audio) / clip)})
            prev = i
        if turns:
            by[clip] = turns
            cnt["turns"] += len(turns)
    return by, cnt


class FluxWS(VH.FakeWS):
    """the recorded Flux socket: the recorded socket with Flux's close (CloseStream -> the socket closes)"""

    async def send_str(self, data, compress=None):
        self.sent_text.append(data[:100])
        try:
            ty = json.loads(data or "{}").get("type")
        except ValueError:
            ty = None
        if ty == "CloseStream":
            await self.q.put(self._msg(self._aiohttp.WSMsgType.CLOSE, 1000))


def patch_flux():
    from livekit.plugins.deepgram.stt_v2 import SpeechStreamv2

    async def fake_connect(self):
        c = VH.CALLS_BY_STT[id(self._stt)]
        if c.ws is not None:
            raise RuntimeError("the plugin reconnected: a replay fault, counted")
        c.ws = FluxWS(c, "flux")
        return c.ws
    SpeechStreamv2._connect_ws = fake_connect


def make_inner(vendor, http, eager=False, threshold=0.5):
    if vendor == "flux":
        from livekit.plugins import deepgram
        if eager:   # Flux's own end of turn with eager on: at the recording's threshold (0.5)
            return deepgram.STTv2(sample_rate=8000, http_session=http, eager_eot_threshold=threshold)
        return deepgram.STTv2(sample_rate=8000, http_session=http)     # eager off: no eager_eot_threshold
    raise SystemExit(f"--vendor {vendor}: only flux is replayed")


# ---------------------------------------------------------------- the relay to the Rabel API

class MemWS:
    """one end of an in-process WebSocket (what the hookup and the relay each read and write)"""

    def __init__(self):
        import aiohttp
        self._A = aiohttp
        self.q: asyncio.Queue = asyncio.Queue()
        self.peer: MemWS | None = None
        self.closed = False
        self.close_code = None

    def _msg(self, kind, data=None):
        return self._A.WSMessage(kind, data, None)

    async def send_str(self, data, compress=None):
        if self.closed:
            raise ConnectionResetError("closed")
        self.peer.q.put_nowait(self._msg(self._A.WSMsgType.TEXT, data))

    async def receive(self, timeout=None):
        return await self.q.get()

    async def close(self, *, code=1000, message=b""):
        if not self.closed:
            self.closed, self.close_code = True, code
            if not self.peer.closed:
                self.peer.closed, self.peer.close_code = True, code
                self.peer.q.put_nowait(self._msg(self._A.WSMsgType.CLOSE, code))
            self.q.put_nowait(self._msg(self._A.WSMsgType.CLOSED))
        return True

    def exception(self):
        return None

    def __aiter__(self):
        return self

    async def __anext__(self):
        m = await self.receive()
        if m.type in (self._A.WSMsgType.CLOSE, self._A.WSMsgType.CLOSED):
            raise StopAsyncIteration
        return m


API_STATS = collections.Counter()     # per process: conversations, heard, guesses, pongs, faults
CALL_TRIES = 3                          # a call is replayed again, from its start, when its conversation with the API broke
API_RT_MS: list = []                    # per process: each guess's real round trip through the network, ms


class ApiBell:
    """the Rabel API itself, reached from inside the replay: the hookup's conversation goes over an in-process socket to this relay,
    which hands each of the hookup's messages to the API (rabel_api.Conversation, blocking: the replay's clock stands still while
    the API answers) and hands each answer back to the hookup that answer's real round trip later on the replay's clock. The hello is
    the hookup's own (its agent, min_chance and learn), with the key put in by the relay from RABEL_API_KEY."""

    def __init__(self):
        self.calls: dict = {}

    async def serve(self, ws):
        import aiohttp
        import rabel_api as RA
        hello = json.loads((await ws.receive()).data)
        c = self.calls[hello.get("key")]
        conv = None
        try:
            conv = RA.Conversation(hello, clock=vclock.REAL["perf_counter"])
            API_STATS["conversations"] += 1
            await ws.send_str(json.dumps(conv.ready))
            c.bell_ready.set()
            loop = asyncio.get_running_loop()
            guessed = False                   # the API sends at most one guess a turn: once it has, the rest of the turn's heard
            async for msg in ws:              # messages are passed on without waiting for an answer
                if msg.type != aiohttp.WSMsgType.TEXT:
                    break
                m = json.loads(msg.data)
                kind = m.get("type")
                if kind == "heard":
                    c.heard += 1
                    API_STATS["heard"] += 1
                    if guessed:
                        conv.send(m)
                        continue
                    for rt, ans in conv.ask(m):
                        if ans.get("type") == "guess":
                            guessed = True
                            API_STATS["guesses"] += 1
                            API_RT_MS.append(round(rt * 1000.0, 3))
                        loop.call_later(rt, self._deliver, ws, c, ans)
                elif kind == "end_of_turn":
                    guessed = False
                    for rt, ans in conv.ask(m):
                        loop.call_later(rt, self._deliver, ws, c, ans)
                else:
                    conv.send(m)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001  counted: a call whose conversation failed is a fault of the run, never a reading
            API_STATS["faults"] += 1
            API_STATS[f"fault:{type(e).__name__}"] += 1
            c.api_fault = f"{type(e).__name__}: {str(e)[:120]}"
        finally:
            if conv is not None:
                API_STATS["pongs"] += conv.pongs
                API_STATS["open_retries"] += conv.open_retries
                conv.close()

    def _deliver(self, ws, c, ans):
        if ws.closed:
            return
        if ans.get("type") == "guess":
            c.guesses.append((time.time(), c.cur, ans.get("text")))
        asyncio.ensure_future(ws.send_str(json.dumps(ans)))


class MemSession:
    """what the hookup takes as its http session: ws_connect opens an in-process socket to the relay (ApiBell)"""
    closed = False

    def __init__(self, bell):
        self.bell = bell
        self.tasks = []

    async def ws_connect(self, url, **kw):
        a, b = MemWS(), MemWS()
        a.peer, b.peer = b, a
        self.tasks.append(asyncio.ensure_future(self.bell.serve(b)))
        return a

    async def close(self):
        return None


# ---------------------------------------------------------------- one call

COMMITS: dict = {}


def patch_end_of_turn():
    """LiveKit's own end-of-turn commits, with their words, watched (not changed)"""
    from livekit.agents.voice import agent_activity as _AA
    orig = _AA.AgentActivity.on_end_of_turn

    def _eot(self, info):
        r = orig(self, info)
        c = VH.SESSIONS.get(id(self._session))
        if c is not None:
            t = time.time()
            c.eot_commits.append((t, bool(r)))
            # LiveKit's own end-of-turn metrics for this commit (its last speaking time, how long it waited after it), and the
            # words the reply is made on; read, never changed
            m = getattr(info, "metrics", None)
            c.eot_info.append({"t": t, "ok": bool(r), "words": info.new_transcript,
                               "stopped": getattr(m, "stopped_speaking_at", None), "eou_delay": getattr(m, "end_of_turn_delay", None),
                               "tr_delay": getattr(m, "transcription_delay", None)})
        return r
    _AA.AgentActivity.on_end_of_turn = _eot


def hookup_offset_check():
    """A CHECK of a cause, never a reading (`--hookup-offset-check`): the hookup (livekit-plugins-rabel, stt.py SpeechStream._run)
    makes the inner speech-to-text's stream itself and never hands it the `start_time_offset` LiveKit sets on the hookup's own stream
    (livekit/agents/voice/agent.py: `stream.start_time_offset = time.time() - _audio_input_started_at`), so the vendor's word times
    reach LiveKit counted from the inner stream's first audio instead of the session's start. This hands the inner stream the same
    offset LiveKit gave the hookup's stream, at the moment the inner stream is made; nothing else."""
    from livekit.agents import stt as lkstt
    from livekit.plugins.rabel import stt as RS
    base = lkstt.RecognizeStream.start_time_offset

    def _set(self, value):
        base.fset(self, value)
        self._rstt._audio_started_at = time.time() - value
    RS.SpeechStream.start_time_offset = property(base.fget, _set)


def wrap_inner_offset(inner, rstt):
    orig = inner.stream

    def stream(*a, **kw):
        s = orig(*a, **kw)
        anchor = getattr(rstt, "_audio_started_at", None)
        if anchor is not None:
            s.start_time_offset = max(0.0, time.time() - anchor)
        return s
    inner.stream = stream


def plant_detector_fault(ms):
    from livekit.agents.inference.eot import transports as TR
    orig = TR._LocalTransport._predict

    async def _late(self, request_id, pcm_snapshot, turn_samples):
        await asyncio.sleep(ms / 1000.0)
        return await orig(self, request_id, pcm_snapshot, turn_samples)
    TR._LocalTransport._predict = _late


async def run_call(clip, turns, a, classes, bell):
    import aiohttp
    from livekit.agents import Agent
    import livekit_default as LD
    StubLLM, _ = classes
    async with aiohttp.ClientSession() as http:
        c = VH.CallRun(clip, turns, http, continuous=True)
        c.eot_commits = []
        c.eot_info = []
        inner = make_inner(a.vendor, http, eager=a.turn == "stt" and a.arm in EAGER_ARMS, threshold=a.eager)
        VH.CALLS_BY_STT[id(inner)] = c
        if a.arm not in EAGER_ARMS:
            VH.hold_vendor_early(inner)
        mem = None
        if a.arm in BELL_ARMS:
            from livekit.plugins import rabel
            mem = MemSession(bell)
            stt_obj = rabel.STT(inner, agent=a.agent, api_key=f"call:{clip}", base_url="http://bell.replay", http_session=mem,
                                learn=False)
            if a.hookup_offset_check:
                wrap_inner_offset(inner, stt_obj)
            bell.calls[f"call:{clip}"] = c
            VH.BELL_BY_STT[id(stt_obj)] = c
        else:
            stt_obj = inner
            c.bell_ready.set()
        llm = StubLLM()
        if a.turn == "stt":   # Flux's own end of turn, no VAD, no audio
            from livekit.agents import AgentSession, TurnHandlingOptions
            session = AgentSession(stt=stt_obj, vad=None, llm=llm, turn_handling=TurnHandlingOptions(turn_detection="stt"))
            VH.SESSIONS[id(session)] = c
            session.input.audio = classes[1]()
        else:
            session = LD.session_for(stt_obj, llm, c)
            VH.SESSIONS[id(session)] = c
            session.input.audio = LD.audio_input(turns[0]["path"], c)
        await session.start(agent=Agent(instructions="You answer a bank's callers."))
        await c.done.wait()
        await asyncio.sleep(GAP_S)
        await asyncio.gather(session.aclose(), return_exceptions=True)
        VH.SESSIONS.pop(id(session), None)
        VH.CALLS_BY_STT.pop(id(inner), None)
        VH.BELL_BY_STT.pop(id(stt_obj), None)
        if mem is not None:
            for t in mem.tasks:
                t.cancel()
        return c, llm


def rows_of(c, llm, livekit_mode=True):
    rows = VH.per_turn(c, llm, livekit_mode)
    by = {tr["key"]: tr for tr in c.turns}
    for i, r in enumerate(rows):
        tr = by.get(r["key"])
        if tr is None or "fault" in r:
            continue
        lo = tr["first_wall"]
        hi = c.turns[i + 1].get("first_wall", c.done_wall + GAP_S) if i + 1 < len(c.turns) else c.done_wall + GAP_S
        r["lk_commits_ms"] = [round((t - tr["eot_wall"]) * 1000.0, 3) for t, ok in c.eot_commits if lo <= t < hi and ok]
        r["vendor_last_word_ms"] = vendor_last_word_ms(tr)
    # the answering reply of each turn, by the caller's last word (not by the turn's message window: Flux's next turn's messages
    # start right after its EndOfTurn, so LiveKit's commit 0.3 s later falls in the next turn's window)
    lw = []
    for tr in c.turns:
        v = vendor_last_word_ms(tr) if "eot_wall" in tr else None
        lw.append(None if "eot_wall" not in tr else tr["eot_wall"] + (v if v is not None else 0.0) / 1000.0)
    for i, r in enumerate(rows):
        if "fault" in r or lw[i] is None:
            continue
        nxt = next((x for x in lw[i + 1:] if x is not None), None)
        r["answer"] = answer_after(c, c.turns[i], lw[i], nxt)
    return rows


def answer_after(c, tr, lw, lw_next):
    """end to end on the replay clock: the reply that answers this turn of the caller's: the first reply LiveKit scheduled after
    the end of the turn's last word and before the end of the next turn's last word. None when LiveKit scheduled none there (its turn
    ran on into the next one: the next turn's reply answers both, counted once, on the next turn)."""
    sched = sorted((t, sid) for t, sid in c.sched_times if t >= lw and (lw_next is None or t < lw_next))
    if not sched:
        return None
    return answer_of(c, tr, sched[0])

def vendor_last_word_ms(tr):
    """the end of the caller's last word in this turn, by the speech-to-text's own word times (the EndOfTurn's words' `end`,
    seconds of audio; a recording in tokens: the final tokens' `end_ms` before the turn's end), in ms from the vendor's end of turn on the
    replay's clock (the recording's message times and its audio times share the first audio chunk as their zero). None when the
    recording carries no word time for the turn."""
    best = None
    for _t, m in tr["raw"]:
        if not isinstance(m, dict):
            continue
        if m.get("type") == "TurnInfo" and m.get("event") == "EndOfTurn":
            ends = [w.get("end") for w in (m.get("words") or []) if isinstance(w, dict) and w.get("end") is not None]
            if ends:
                best = max(ends) * 1000.0
        elif m.get("tokens"):
            for x in m["tokens"]:
                if x.get("text") in ("<end>", "<fin>"):
                    break
                if x.get("is_final") and x.get("end_ms") is not None:
                    best = max(best or 0.0, float(x["end_ms"]))
    return None if best is None else round(best - tr["eot_ms"], 3)


def answer_of(c, tr, first):
    """the answering reply's fields: when LiveKit scheduled it (its commit), when its LLM call started (its preemptive start if it was
    one, else the commit), LiveKit's own last speaking time for the turn it committed (its end-of-turn metrics' `stopped_speaking_at`,
    beside the speech-to-text's own), how long LiveKit waited after it, and the words the reply was made on. Times in ms from the
    vendor's end of turn."""
    ts, sid = first
    pre = [x for x in c.created if x[2] == sid and x[0] <= ts]
    start = pre[-1][0] if pre else ts
    eots = [e for e in c.eot_info if e["ok"] and e["t"] <= ts + 1e-6]
    e = eots[-1] if eots else None
    rel = lambda t: None if t is None else round((t - tr["eot_wall"]) * 1000.0, 3)  # noqa: E731
    return {"commit_ms": rel(ts), "start_ms": rel(start), "preemptive": bool(pre),
            "last_word_ms": rel(e["stopped"]) if e else None,
            "eou_delay_ms": None if not e or e["eou_delay"] is None else round(e["eou_delay"] * 1000.0, 3),
            "tr_delay_ms": None if not e or e["tr_delay"] is None else round(e["tr_delay"] * 1000.0, 3),
            "commit_to_sched_ms": None if not e else round((ts - e["t"]) * 1000.0, 3),
            "words": e["words"] if e else None}


def shard_main(args):
    calls, a, out = args
    import logging
    logging.basicConfig(level=logging.WARNING)
    VH.patch_livekit(a.arm)
    patch_end_of_turn()
    patch_flux()
    if a.fault_detector_ms:
        plant_detector_fault(a.fault_detector_ms)
    classes = VH.stub_classes()
    bell = None
    if a.arm in BELL_ARMS:
        bell = ApiBell()
        VH.watch_bell_emits()
        if a.hookup_offset_check:
            hookup_offset_check()
    rows, lags, faults = [], [], collections.Counter()
    t0 = time.time()

    def one_done(clip, turns, res):
        if isinstance(res, BaseException):
            faults[type(res).__name__] += 1
            rows.extend({"key": t["key"], "conv": t["conv"], "fault": type(res).__name__} for t in turns)
            return
        c, llm = res
        if getattr(c, "api_fault", None):   # the call's conversation with the API failed: its turns are a fault, never a reading
            faults["api"] += 1
            rows.extend({"key": t["key"], "conv": t["conv"], "fault": "api: " + c.api_fault} for t in turns)
            return
        rows.extend(rows_of(c, llm, a.turn == "livekit"))
        lags.extend(c.lag)

    if a.clock == "replay":
        for clip, turns in calls:
            for attempt in range(CALL_TRIES):   # a call whose conversation with the API broke (the network) is replayed again
                try:
                    res = vclock.run(run_call, clip, turns, a, classes, bell)
                except Exception as e:  # noqa: BLE001  a call that fails is counted, never dropped silently
                    res = e
                if isinstance(res, BaseException) or not getattr(res[0], "api_fault", None) or attempt == CALL_TRIES - 1:
                    break
                API_STATS["calls_replayed_again"] += 1
            one_done(clip, turns, res)
    else:
        async def all_wall():
            sem = asyncio.Semaphore(a.conc)

            async def one(clip, turns):
                async with sem:
                    try:
                        res = await run_call(clip, turns, a, classes, bell)
                    except Exception as e:  # noqa: BLE001
                        res = e
                    one_done(clip, turns, res)
            await asyncio.gather(*(one(cl, ts) for cl, ts in calls))
        asyncio.run(all_wall())
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    lags.sort()
    return {"rows": len(rows), "api": dict(API_STATS), "api_rt_ms": API_RT_MS,
            "lag_median": lags[len(lags) // 2] if lags else None, "lag_max": lags[-1] if lags else None,
            "faults": dict(faults), "real_s": round(time.time() - t0, 1), "real_infer": vclock.real_infer_summary(),
            "replay_stats": dict(vclock.STATS)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor", choices=("flux",), required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--sched", default="", help="flux: the schedule of the 28 (sched_0.5.json), which turns are read")
    ap.add_argument("--audio", default="", help="flux: the caller audio folder")
    ap.add_argument("--turn", choices=tuple(ARMS), default="livekit",
                    help="livekit: LiveKit's default ending (arms lk, lk_bell); stt: Flux's own end of turn (arms nothing, bell, passed, both)")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--agent", default="demo-bank", help="the API's agent Bell guesses for (Bell arms)")
    ap.add_argument("--eager", type=float, default=0.5,
                    help="--turn stt, arms passed/both: Flux's eager threshold, the one the --events/--sched recording was made at "
                         "(0.5)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--conc", type=int, default=10, help="--clock wall: calls at once in each process")
    ap.add_argument("--clock", choices=("replay", "wall"), default="replay")
    ap.add_argument("--calls", type=int, default=0, help="only the N shortest calls (a trial)")
    ap.add_argument("--clips", default="")
    ap.add_argument("--fault-detector-ms", type=float, default=0.0)
    ap.add_argument("--fresh", action="store_true", help="each call in a new process of its own")
    ap.add_argument("--hookup-offset-check", action="store_true", help="a check of a cause, never a reading: see hookup_offset_check")
    ap.add_argument("--hookup-label", default="", help="which livekit-plugins-rabel commit is installed (recorded in the meta)")
    a = ap.parse_args()
    if a.arm not in ARMS[a.turn]:
        raise SystemExit(f"--turn {a.turn} has the arms {ARMS[a.turn]}")
    import rabel_api as RA
    if a.arm in BELL_ARMS:
        RA.api_key()     # set, or stop here
        if a.clock != "replay":
            raise SystemExit("Bell's arms run on the replay clock (the relay waits for the API with the replay's clock standing still)")
    by, cnt = load_flux(a)
    dur = {cl: ts[-1]["eot_ms"] for cl, ts in by.items()}
    clips = sorted(by, key=lambda c: (-dur[c], c))
    if a.calls:
        clips = sorted(by, key=lambda c: (dur[c], c))[: a.calls]
    procs = max(1, min(a.procs, len(clips)))
    if a.fresh:   # every call in a process of its own, made for it (nothing from an earlier call in the process)
        shards = [[(cl, by[cl])] for cl in clips]
    else:
        shards = [[] for _ in range(procs)]
        load = [0.0] * procs
        for cl in clips:
            j = min(range(procs), key=lambda i: load[i])
            shards[j].append((cl, by[cl]))
            load[j] += dur[cl]
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    parts = [str(out) + f".part{i}" for i in range(len(shards))]
    t0 = time.time()
    with mp.get_context("spawn").Pool(procs, maxtasksperchild=1 if a.fresh else None) as pool:
        res = pool.map(shard_main, [(s, a, p) for s, p in zip(shards, parts) if s])
    with open(out, "w", encoding="utf-8") as f:
        for p in parts:
            if os.path.exists(p):
                f.write(open(p, encoding="utf-8").read())
                os.remove(p)
    faults = collections.Counter()
    for r in res:
        faults.update(r["faults"])
    meta = {"host": "livekit_default", "turn": a.turn, "eager": a.eager if a.arm in EAGER_ARMS else None, "clock": a.clock, "vendor": a.vendor, "arm": a.arm, "agent": a.agent if a.arm in BELL_ARMS else None, "events": a.events,
            "fault_detector_ms": a.fault_detector_ms, "hookup_offset_check": a.hookup_offset_check, "fresh": a.fresh,
            "hookup_label": a.hookup_label, "calls": len(clips), "turns": sum(r["rows"] for r in res),
            "load_counts": dict(cnt), "faults": dict(faults), "wall_s": round(time.time() - t0, 1), "procs": procs,
            "conc": a.conc if a.clock == "wall" else 1, "by_process": res}
    if a.arm in BELL_ARMS:
        api, rts = collections.Counter(), sorted(x for r in res for x in r.get("api_rt_ms", []))
        for r in res:
            api.update(r.get("api", {}))
            r.pop("api_rt_ms", None)
        q = lambda f: round(rts[min(len(rts) - 1, int(f * len(rts)))], 1) if rts else None  # noqa: E731
        meta["api"] = {"url": RA.base_url(), "stats": dict(api), "guess_round_trip_ms": {"n": len(rts), "p50": q(0.5), "p90": q(0.9),
                                                                                         "p99": q(0.99), "max": rts[-1] if rts else None},
                       "rt_ms_all": rts}
    try:
        import importlib.metadata as md
        meta["versions"] = {p: md.version(p) for p in ("livekit-agents", "livekit-plugins-deepgram", f"livekit-plugins-{a.vendor}",
                                                       "livekit-plugins-rabel", "livekit-local-inference") if VH._has(md, p)}
    except Exception as e:  # noqa: BLE001
        meta["versions"] = str(e)
    pathlib.Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "by_process"}))
    print("LIVEKIT_DEFAULT_HOST_DONE")


if __name__ == "__main__":
    main()
