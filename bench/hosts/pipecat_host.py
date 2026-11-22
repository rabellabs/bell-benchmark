"""The benchmark's Pipecat host: recorded Flux streams replayed into a real Pipecat pipeline with a stand-in LLM.

Per call, one pipeline (Pipecat, the release installed):
  DeepgramFluxSTTService -> the user aggregator (LLMContextAggregatorPair) -> a stand-in LLM -> the assistant aggregator
  - the speech-to-text is Pipecat's own Flux service with its WebSocket replaced by the recorded Flux messages: the service's own
    receive loop reads them (a "Connected" message first, then the call's messages at their recorded arrival times), so Pipecat's
    own handling of StartOfTurn, Update, EagerEndOfTurn, TurnResumed and EndOfTurn decides everything;
  - eager on: `enable_eager_end_of_turn=True` and the user aggregator's strategies `EagerUserTurnStrategies()` (Pipecat's own way
    to answer Flux's eager end of turn; it keeps the early reply only when the committed transcript matches it, its default
    `NormalizedMatch`); eager off ("nothing"): `enable_eager_end_of_turn=False`, `ExternalUserTurnStrategies()`, and Flux's
    EagerEndOfTurn and TurnResumed messages are not fed (Flux sends neither with eager off);
  - the stand-in LLM records only when an inference was started (each `LLMContextFrame` reaching it: its time, whether Pipecat marked
    it speculative, the words of its last user message) and answers "Okay." at once;
  - Bell: the Pipecat hookup `pipecat-rabel` 0.3.0: `RabelBellUserTurnStrategies(agent=..., learn=False)` at its defaults is
    the user aggregator's `user_turn_strategies`, as its README places it, and nothing stands between the speech-to-text and the
    aggregator. Its WebSocket goes to the Rabel API itself (RABEL_API_URL, RABEL_API_KEY), on the wall clock as the call is
    replayed, so each guess arrives when the API really answered it; the connection is only watched (each guess's arrival
    recorded), nothing of the hookup is changed.
Arms: nothing (eager off), eager (Flux's eager at the schedule's threshold, as Pipecat runs), bell (Bell alone, Flux's eager off),
both (Bell beside Flux's eager). Each call's turns are replayed in order at their recorded times within the turn, GAP_S apart, on
the wall clock (Pipecat runs no local inference here, so nothing but scheduling reads the clock).

Per turn: LLM calls (the inferences started in the turn's window), kept ("eager" when the turn's reply was an early one that Pipecat
did not withdraw: a speculative inference started and no ordinary inference after it in the window; "bell" when that early start
was the hookup's, its words those of a guess the hookup pushed in the turn), head (the EndOfTurn's feed time less that inference's
start); `answer` (the end to end read's fields, `combine.py`): the commit (Pipecat's release of the kept early reply at the
EndOfTurn, or its ordinary inference's start), the reply's start, its words, and `audio_rule` "pipecat_gate": Pipecat's
speculation gate (`pipecat/turns/speculation_gate.py`, hosted by every `LLMService`) holds the early reply's output until the turn's
end and only then passes it to the voice, so first audio = max(commit, start + the first sentence's time) + the voice's time from
the sentence to its first audio; and `vendor_last_word_ms` (Flux's words' `end` in the EndOfTurn, as on LiveKit).

--hookup-as processor (the Bell arms; the default, `strategy`, is the hookup as above): pipecat-rabel up to 0.2.0, whose
`RabelBellProcessor(agent=..., learn=False)` goes between the speech-to-text and the user aggregator, with the aggregator on
Pipecat's `EagerUserTurnStrategies()`. Each early reply the hookup starts on a guess is recorded either way.

  python bench/hosts/pipecat_host.py --sched SCHED.json --arm eager|nothing|bell|both --out OUT.jsonl [--procs 8] [--conc 12]
      [--calls N] [--clips FILE] [--agent demo-bank --hookup-label LABEL]
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

GAP_S = 3.0
LEAD_S = 2.0
ARMS = ("eager", "nothing", "bell", "both")
EAGER_ARMS = ("eager", "both")
BELL_ARMS = ("bell", "both")
BELL = None          # the watch on the hookup's connection to the API (set in each process)
AGENT = "demo-bank"


# ---------------------------------------------------------------- the hookup's connection to the Rabel API, watched

class WatchedConn:
    """the hookup's real conversation with the API, passed through; each guess's arrival is recorded for the per-turn read"""

    def __init__(self, ws, call):
        self.ws, self.call = ws, call

    @property
    def close_code(self):
        return self.ws.close_code

    async def send(self, data):
        return await self.ws.send(data)

    def _seen(self, raw):
        try:
            m = json.loads(raw)
        except (TypeError, ValueError):
            return
        if isinstance(m, dict) and m.get("type") == "guess":
            self.call.guesses.append((time.time(), m.get("text")))
            BELL.guesses += 1

    async def recv(self):
        raw = await self.ws.recv()
        self._seen(raw)
        return raw

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            raw = await self.ws.recv()
        except Exception as e:  # noqa: BLE001
            import websockets
            if isinstance(e, websockets.ConnectionClosed):
                raise StopAsyncIteration from None
            raise
        self._seen(raw)
        return raw

    async def close(self, code=1000, reason=""):
        return await self.ws.close(code, reason)


class ApiWatch:
    def __init__(self):
        self.calls: dict = {}
        self.conversations = self.guesses = self.faults = 0

    async def connect(self, url, additional_headers=None, **kw):
        import websockets
        here = pathlib.Path(__file__).resolve().parent
        sys.path.insert(0, str(here.parent))
        import rabel_api as RA
        key = (additional_headers or {}).get("Authorization", "").split(" ", 1)[-1]
        call = self.calls[key]
        api = RA.base_url().replace("https://", "wss://").replace("http://", "ws://") + "/v1/conversation"
        try:
            ws = await websockets.connect(api, additional_headers={"Authorization": "Bearer " + RA.api_key()}, **kw)
        except Exception:
            self.faults += 1
            raise
        self.conversations += 1
        return WatchedConn(ws, call)


def patch_hookup(watch):
    """the hookup's connection goes to the Rabel API through the watch; nothing else of it changes"""
    import types
    import websockets
    from pipecat_rabel import processor as PR
    PR.websockets = types.SimpleNamespace(connect=watch.connect, ConnectionClosed=websockets.ConnectionClosed)


class FakeWS:
    """stands in for Flux's WebSocket: yields "Connected", then the call's recorded messages at their times"""

    def __init__(self, call):
        from websockets.protocol import State
        self.state = State.OPEN
        self._call = call
        self.response = None
        self._closed = asyncio.Event()

    async def send(self, _data):
        return None

    async def close(self):
        from websockets.protocol import State
        self.state = State.CLOSED
        self._closed.set()

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        c = self._call
        yield json.dumps({"type": "Connected", "request_id": "replay", "sequence_id": 0})
        await c.start.wait()
        c.origin = time.time() + LEAD_S
        for tr in c.turns:
            for (t_ms, m) in tr["sched"]:
                wait = c.origin + t_ms / 1000.0 - time.time()
                if wait > 0:
                    await asyncio.sleep(wait)
                now = time.time()
                c.lag.append((now - (c.origin + t_ms / 1000.0)) * 1000.0)
                if m.get("event") == "EndOfTurn":
                    tr["eot_wall"] = now
                if m["_k"] == 0:
                    tr["first_wall"] = now
                yield json.dumps({k: v for k, v in m.items() if k != "_k"})
        c.done_wall = time.time()
        c.done.set()
        await self._closed.wait()


class CallRun:
    def __init__(self, clip, turns, arm):
        self.clip = clip
        self.turns = turns
        t_off = 0.0
        for tr in turns:
            ms = tr["msgs"]
            if arm not in EAGER_ARMS:
                ms = [x for x in ms if x[1].get("event") not in ("EagerEndOfTurn", "TurnResumed")]
            t0 = ms[0][0]
            sched = []
            for k, (t, m) in enumerate(ms):
                m = dict(m)
                m["_k"] = k
                sched.append((t_off + (t - t0), m))
            tr["sched"] = sched
            t_off += (tr["eot_ms"] - t0) + GAP_S * 1000.0
        self.start = asyncio.Event()
        self.done = asyncio.Event()
        self.origin = None
        self.lag = []
        self.llm = []       # (time, speculative, words)
        self.cancels = []   # UserTurnResponseCancelFrame times seen by the stand-in
        self.guesses = []   # (time, text): the stand-in Bell's guesses sent
        self.pushes = []    # (time, text): the hookup's early starts pushed to Pipecat
        self.heard = 0
        self.hello = None
        self.proc = None


# --hookup-as, read here so the spawned processes see it (main puts it in their environment)
HOOKUP_AS = os.environ.get("PIPECAT_HOST_HOOKUP_AS", "strategy")
_patch_processor = patch_hookup


def patch_hookup(bell):
    """the hookup's connection to the Rabel API goes to `bell`: the processor's (pipecat-rabel up to 0.2) as above, or with
    --hookup-as strategy the stop strategy's (0.3.0 on); nothing else of the hookup changes"""
    if HOOKUP_AS != "strategy":
        return _patch_processor(bell)
    import types
    import websockets
    from pipecat_rabel import strategy as PS
    PS.websockets = types.SimpleNamespace(connect=bell.connect, ConnectionClosed=websockets.ConnectionClosed)


def bell_strategies(call):
    """--hookup-as strategy: Bell's turn strategies for the user aggregator, the early replies they start recorded as pushes"""
    from pipecat_rabel import RabelBellUserTurnStrategies
    strategies = RabelBellUserTurnStrategies(api_key=f"call:{call.clip}", agent=globals().get("AGENT", "demo-bank"), learn=False,
                                             url="http://bell.replay")
    bell = strategies.bell
    BELL.calls[f"call:{call.clip}"] = call
    orig = bell._start_bell

    async def _start_bell(text, chance=None):
        t, n = time.time(), bell.stats.guesses_pushed
        await orig(text, chance)
        if bell.stats.guesses_pushed > n:   # started (it starts none outside Pipecat's turn)
            call.pushes.append((t, text))
    bell._start_bell = _start_bell
    call.proc = bell
    return strategies


def build(call, arm):
    from pipecat.frames.frames import LLMContextFrame, LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame, \
        UserTurnResponseCancelFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
    from pipecat.turns.user_turn_strategies import EagerUserTurnStrategies, ExternalUserTurnStrategies

    class ReplayFlux(DeepgramFluxSTTService):
        async def _connect_websocket(self):
            if self._websocket is not None:
                return
            self._connection_established_event.clear()
            self._user_is_speaking = False
            self._websocket = FakeWS(call)

    class StandInLLM(FrameProcessor):
        async def process_frame(self, frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, LLMContextFrame):
                msgs = [m for m in frame.context.get_messages() if isinstance(m, dict) and m.get("role") == "user"]
                words = msgs[-1].get("content") if msgs else ""
                call.llm.append((time.time(), bool(frame.speculation), words if isinstance(words, str) else json.dumps(words)))
                await self.push_frame(LLMFullResponseStartFrame())
                await self.push_frame(LLMTextFrame("Okay."))
                await self.push_frame(LLMFullResponseEndFrame())
                return
            if isinstance(frame, UserTurnResponseCancelFrame):
                call.cancels.append(time.time())
            await self.push_frame(frame, direction)

    stt = ReplayFlux(api_key="replay-only", sample_rate=8000, enable_eager_end_of_turn=(arm in EAGER_ARMS))
    strategies = ExternalUserTurnStrategies() if arm == "nothing" else EagerUserTurnStrategies()
    if arm in BELL_ARMS and HOOKUP_AS == "strategy":
        strategies = bell_strategies(call)
    context = LLMContext()
    user, assistant = LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams(user_turn_strategies=strategies))
    procs = [stt]
    if arm in BELL_ARMS and HOOKUP_AS != "strategy":
        from pipecat_rabel import RabelBellProcessor
        bell = RabelBellProcessor(api_key=f"call:{call.clip}", agent=AGENT, learn=False, url="http://bell.replay")
        BELL.calls[f"call:{call.clip}"] = call
        orig = bell._speculate

        async def _speculate(text, chance=None):
            call.pushes.append((time.time(), text))
            return await orig(text, chance)
        bell._speculate = _speculate
        call.proc = bell
        procs.append(bell)
    pipeline = Pipeline(procs + [user, StandInLLM(), assistant])
    worker = PipelineWorker(pipeline, params=PipelineParams(audio_in_sample_rate=8000), idle_timeout_secs=None)
    return stt, worker


async def run_call(clip, turns, arm):
    from pipecat.workers.runner import WorkerRunner
    c = CallRun(clip, turns, arm)
    stt, worker = build(c, arm)
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    run = asyncio.create_task(runner.run())
    await asyncio.sleep(0.5)
    if c.proc is not None:   # the hookup opens its conversation in the background at the pipeline's start (it fails open)
        for _ in range(3000):    # the API's conversation opened over the network (up to 30 s) before the call's words start
            if c.proc.ready.is_set():
                break
            await asyncio.sleep(0.01)
        if not c.proc.ready.is_set():   # no conversation with the API for this call: counted, the run's gate refuses it
            BELL.faults += 1
    c.start.set()
    await c.done.wait()
    await asyncio.sleep(GAP_S)
    await worker.cancel()
    try:
        await asyncio.wait_for(run, 20)
    except (asyncio.TimeoutError, TimeoutError):
        pass
    return c


def vendor_last_word_ms(tr):
    """the end of the caller's last word in the turn by Flux's own word times (the EndOfTurn's words' `end`), in ms from the
    EndOfTurn (the recording's message times and audio times share the first audio chunk as their zero); as on LiveKit"""
    best = None
    for _t, m in tr["msgs"]:
        if isinstance(m, dict) and m.get("type") == "TurnInfo" and m.get("event") == "EndOfTurn":
            ends = [w.get("end") for w in (m.get("words") or []) if isinstance(w, dict) and w.get("end") is not None]
            if ends:
                best = max(ends) * 1000.0
    return None if best is None else round(best - tr["eot_ms"], 3)


def per_turn(c):
    try:
        from pipecat_rabel.processor import _same
    except ImportError:   # pipecat-rabel 0.3.0 on: the same check, Pipecat's NormalizedMatch (the strategy's default)
        from pipecat.turns.user_stop import NormalizedMatch
        _m = NormalizedMatch()
        _same = lambda a, b: bool(a) and bool(b) and _m.matches(a, b)  # noqa: E731
    out = []
    for i, tr in enumerate(c.turns):
        lo = tr["first_wall"]
        hi = c.turns[i + 1]["first_wall"] if i + 1 < len(c.turns) else c.done_wall + GAP_S
        eot = tr["eot_wall"]
        calls = [x for x in c.llm if lo <= x[0] < hi]
        spec = [x for x in calls if x[1]]
        ordinary = [x for x in calls if not x[1]]
        kept, head = None, None
        pushes = [x for x in c.pushes if lo <= x[0] < hi]
        answer = None
        rel = lambda t: round((t - eot) * 1000.0, 3)  # noqa: E731
        if spec and not any(x[0] > spec[-1][0] for x in ordinary):
            kept = "eager"
            if _same is not None and any(_same(p[1], spec[-1][2]) and p[0] <= spec[-1][0] + 1e-3 for p in pushes):
                kept = "bell"
            head = (eot - spec[-1][0]) * 1000.0
            # released by the gate at the turn's end (Flux's EndOfTurn fed), or at once if it started after it
            answer = {"commit_ms": max(0.0, rel(spec[-1][0])), "start_ms": rel(spec[-1][0]), "preemptive": True,
                      "words": spec[-1][2], "audio_rule": "pipecat_gate"}
        elif ordinary:
            o = next((x for x in ordinary if x[0] >= eot - 1e-3), ordinary[-1])
            answer = {"commit_ms": rel(o[0]), "start_ms": rel(o[0]), "preemptive": False, "words": o[2], "audio_rule": "pipecat_gate"}
        row = {"key": tr["key"], "conv": tr["conv"], "calls": len(calls), "early_made": len(spec), "kept": kept, "head": head,
               "ordinary": len(ordinary), "spec_after_eot": sum(1 for x in spec if x[0] > eot),
               "withdrawn": sum(1 for t in c.cancels if lo <= t < hi), "repeat_held": 0,
               "vendor_last_word_ms": vendor_last_word_ms(tr), "answer": answer}
        if c.proc is not None:
            row["bell_guesses"] = sum(1 for x in c.guesses if lo <= x[0] < hi)
            row["bell_pushed"] = len(pushes)
        out.append(row)
    if c.proc is not None and out:
        st = c.proc.stats
        out[0]["bell_call_stats"] = {k: getattr(st, k) for k in ("conversations", "turns_sent", "guesses_received", "guesses_pushed",
                                                                 "guesses_dropped", "guesses_right", "eager_held_back")
                                     if hasattr(st, k)}
    return out


async def shard_async(calls, arm, conc):
    sem = asyncio.Semaphore(conc)
    rows, lags = [], []

    async def one(clip, turns):
        async with sem:
            c = await run_call(clip, turns, arm)
            rows.extend(per_turn(c))
            lags.extend(c.lag)

    await asyncio.gather(*(one(cl, ts) for cl, ts in calls))
    return rows, lags


def shard_main(args):
    calls, arm, conc, out, bell_args = args
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    if arm in BELL_ARMS:
        global BELL, AGENT
        AGENT = bell_args
        BELL = ApiWatch()
        patch_hookup(BELL)
    rows, lags = asyncio.run(shard_async(calls, arm, conc))
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    lags.sort()
    api = None if BELL is None else {"conversations": BELL.conversations, "guesses": BELL.guesses, "faults": BELL.faults}
    return len(rows), (lags[len(lags) // 2] if lags else None), (lags[int(len(lags) * 0.99)] if lags else None), (lags[-1] if lags else None), api


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sched", required=True)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--conc", type=int, default=12)
    ap.add_argument("--calls", type=int, default=0)
    ap.add_argument("--clips", default="", help="a file of clip names (or conversation ids): only these calls")
    ap.add_argument("--agent", default="demo-bank", help="the API's agent Bell guesses for (Bell arms)")
    ap.add_argument("--hookup-label", default="", help="which pipecat-rabel commit is installed (recorded in the meta)")
    ap.add_argument("--hookup-as", choices=("processor", "strategy"), default="strategy",
                    help="the Bell arms: pipecat-rabel's processor (up to 0.2) or its user turn stop strategy (0.3.0 on)")
    a = ap.parse_args()
    if a.arm in BELL_ARMS:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
        import rabel_api as RA
        RA.api_key()     # set, or stop here
    data = json.loads(pathlib.Path(a.sched).read_text(encoding="utf-8"))
    by = collections.defaultdict(list)
    for tr in data["turns"]:
        by[tr["clip"]].append(tr)
    for v in by.values():
        v.sort(key=lambda t: t["ti"])
    dur = {cl: sum(t["eot_ms"] - t["msgs"][0][0] for t in ts) for cl, ts in by.items()}
    clips = sorted(by, key=lambda c: -dur[c])
    if a.calls:
        clips = sorted(by, key=lambda c: dur[c])[: a.calls]
    if a.clips:
        keep = {ln.strip() for ln in open(a.clips, encoding="utf-8") if ln.strip()}
        clips = [c for c in clips if c in keep or c.rsplit(".", 1)[0] in keep]
    shards = [[] for _ in range(a.procs)]
    load = [0.0] * a.procs
    for cl in clips:
        j = min(range(a.procs), key=lambda i: load[i])
        shards[j].append((cl, by[cl]))
        load[j] += dur[cl]
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    os.environ["PIPECAT_HOST_HOOKUP_AS"] = a.hookup_as   # the spawned processes read it at import (HOOKUP_AS)
    t0 = time.time()
    parts = [str(out) + f".part{i}" for i in range(a.procs)]
    with mp.get_context("spawn").Pool(a.procs) as pool:
        res = pool.map(shard_main, [(s, a.arm, a.conc, p, a.agent) for s, p in zip(shards, parts) if s])
    with open(out, "w", encoding="utf-8") as f:
        for p in parts:
            if os.path.exists(p):
                f.write(open(p, encoding="utf-8").read())
                os.remove(p)
    meta = {"host": "pipecat", "arm": a.arm, "label": data["label"], "calls": len(clips), "turns": sum(r[0] for r in res),
            "agent": a.agent if a.arm in BELL_ARMS else None, "hookup": a.hookup_label,
            "api": [r[4] for r in res] if a.arm in BELL_ARMS else None,
            "wall_s": round(time.time() - t0, 1), "procs": a.procs, "conc": a.conc,
            "feed_lag_ms_by_process": [{"median": r[1], "p99": r[2], "max": r[3]} for r in res]}
    try:
        import importlib.metadata as md
        meta["versions"] = {"pipecat-ai": md.version("pipecat-ai")}
        if a.arm in BELL_ARMS:
            try:
                meta["versions"]["pipecat-rabel"] = md.version("pipecat-rabel")
            except Exception:  # noqa: BLE001  (run from its source: the label says which commit)
                meta["versions"]["pipecat-rabel"] = "source"
    except Exception as e:  # noqa: BLE001
        meta["versions"] = str(e)
    if a.hookup_as != "processor":
        meta["hookup_as"] = a.hookup_as
    pathlib.Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta))
    print("PIPECAT_HOST_DONE")


if __name__ == "__main__":
    main()
