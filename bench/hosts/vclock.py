"""The replay clock: one clock that LiveKit's timers, its VAD and its turn detector all follow, so the same recorded call
replayed twice gives LiveKit the same inputs at the same moments and LiveKit decides the same way.

Why: under LiveKit's default ending the same arm run twice on the same
recorded messages moved LiveKit's commit by a median 3 to 177 ms. LiveKit 1.8.13 reads the wall clock throughout its turn ending
(`time.time()` for the last speaking time and the endpointing sleep, `loop.time()` for every timer), and it runs its VAD
(`run_in_executor`, a one-worker thread a stream) and its turn detector (`asyncio.to_thread`) on threads. So, on a loaded machine,
(1) how far the VAD and the detector have read the call's audio when a recorded message arrives, (2) which 20 ms of audio the
detector's window holds when it is asked, and (3) how long each inference takes, all move with the machine's timing; and the
detector's answer (commit 0.3 s or 2.5 s after the last speech) can flip on (2).

What this does, and nothing else:
  - `time.time`, `time.monotonic`, `time.perf_counter` (and their `_ns`) read one virtual clock while a replay runs (installed before
    LiveKit is imported, so `from time import time` binds it too); outside a replay they read the wall clock;
  - the event loop's `time()` is the same clock, and when nothing is ready it does not sleep: it moves the clock to the next timer
    (a replay therefore runs as fast as the machine computes it, and its speed never changes what happens in it);
  - `run_in_executor` (and so `asyncio.to_thread`) runs the function at once on the loop's own thread and hands back its result:
    the VAD's and the detector's inference take no time on the replay clock, and run in the order the code asks for them. Their
    real time on this machine is kept beside (`REAL_INFER`), so a reader can see what that leaves out.
Nothing in LiveKit's code is changed: the same functions run on the same inputs; only the clock they read and the thread they run
on are the replay's.
"""
from __future__ import annotations

import asyncio
import collections
import selectors
import time as _time

EPOCH = 1_800_000_000.0     # the replay's wall-clock reading at its start (a fixed value: every replay starts at the same reading)
MONO0 = 1_000.0             # the replay's monotonic reading at its start

REAL = {n: getattr(_time, n) for n in ("time", "monotonic", "perf_counter", "time_ns", "monotonic_ns", "perf_counter_ns")}
REAL_INFER: dict = collections.defaultdict(list)    # function name -> real seconds each inline call took
STATS = collections.Counter()


class Clock:
    def __init__(self):
        self.on = False
        self.t = 0.0        # seconds since the replay began

    def reset(self):
        self.t = 0.0

    def advance(self, dt):
        if dt > 0:
            self.t += dt


CLOCK = Clock()


def _time_():
    return EPOCH + CLOCK.t if CLOCK.on else REAL["time"]()


def _mono():
    return MONO0 + CLOCK.t if CLOCK.on else REAL["monotonic"]()


def _perf():
    return MONO0 + CLOCK.t if CLOCK.on else REAL["perf_counter"]()


def install():
    """patch the time module's readings (before LiveKit, aiohttp or the hookup are imported)"""
    _time.time = _time_
    _time.monotonic = _mono
    _time.perf_counter = _perf
    _time.time_ns = lambda: int(round(_time_() * 1e9))
    _time.monotonic_ns = lambda: int(round(_mono() * 1e9))
    _time.perf_counter_ns = lambda: int(round(_perf() * 1e9))


class ReplaySelector(selectors.BaseSelector):
    """the loop's selector: any I/O already there is returned at once; otherwise the clock moves to the loop's next timer"""
    STALL_S = 10.0

    def __init__(self):
        self._sel = selectors.DefaultSelector()

    def register(self, fileobj, events, data=None):
        return self._sel.register(fileobj, events, data)

    def unregister(self, fileobj):
        return self._sel.unregister(fileobj)

    def modify(self, fileobj, events, data=None):
        return self._sel.modify(fileobj, events, data)

    def get_key(self, fileobj):
        return self._sel.get_key(fileobj)

    def get_map(self):
        return self._sel.get_map()

    def close(self):
        self._sel.close()

    def select(self, timeout=None):
        ev = self._sel.select(0)
        if ev:
            STATS["io_events"] += 1
            return ev
        if timeout is None:
            # nothing ready and no timer: only another thread could wake the loop (none should, everything runs inline)
            STATS["stalls"] += 1
            return self._sel.select(self.STALL_S)
        if timeout > 0:
            CLOCK.advance(timeout)
            STATS["advances"] += 1
        return []


class ReplayLoop(asyncio.SelectorEventLoop):
    def __init__(self):
        super().__init__(ReplaySelector())

    def time(self):
        return MONO0 + CLOCK.t

    def run_in_executor(self, executor, func, *args):
        fut = self.create_future()
        name = getattr(func, "__qualname__", None) or getattr(getattr(func, "func", None), "__qualname__", None) or repr(func)[:60]
        if hasattr(func, "args") and func.args and callable(func.args[0]):   # asyncio.to_thread: partial(ctx.run, f, ...)
            name = getattr(func.args[0], "__qualname__", name)
        t0 = REAL["perf_counter"]()
        try:
            res = func(*args)
        except BaseException as e:  # noqa: BLE001  handed to the awaiting code as the thread would
            fut.set_exception(e)
        else:
            fut.set_result(res)
        REAL_INFER[name].append(REAL["perf_counter"]() - t0)
        return fut


def run(coro_fn, *a):
    """one replay: a fresh loop, the clock at its start, the coroutine run to its end; the clock back to the wall after"""
    CLOCK.reset()
    CLOCK.on = True
    try:
        with asyncio.Runner(loop_factory=ReplayLoop) as r:
            return r.run(coro_fn(*a))
    finally:
        CLOCK.on = False


def real_infer_summary():
    out = {}
    for k, v in REAL_INFER.items():
        s = sorted(v)
        out[k] = {"n": len(s), "median_ms": round(s[len(s) // 2] * 1000, 3), "p99_ms": round(s[int(len(s) * 0.99)] * 1000, 3),
                  "max_ms": round(s[-1] * 1000, 3)}
    return out
