#!/usr/bin/env python3
"""The live part: the time from a reply's LLM call to its first audio, measured live
with Claude Haiku 5.5 and Deepgram Aura-2, once for each distinct reply text, REPS times. The commit and reply-start times come from
the replay clock (`bench/hosts/livekit_default_host.py`'s `answer` field); `combine.py` puts the two together.

One measurement, as LiveKit 1.8.13 runs a reply (read in its source: `voice/agent_activity.py` starts the LLM and the TTS inference
before the reply is authorised and plays the audio at the commit; `plugins/deepgram/tts.py` streams the words to Aura-2 over its
WebSocket and sends Flush at the end of the segment):
  - the Aura-2 socket is opened beside the LLM call (LiveKit keeps the voice's socket pooled, so its opening is not on the path; the
    time it took is kept beside);
  - the LLM call: Claude Haiku 5.5 through the Anthropic SDK, streaming, one system prompt for every reply, the turn's words as the
    user message, effort low, thinking off, max_tokens 80; the output is cut after its first sentence;
  - the first sentence's words are sent to Aura-2 as they come (Speak), then Flush; the first audio bytes end the measurement.
  - Fields (ms from the LLM request): ttft (first text), sentence (the first sentence complete), first_audio (L). Beside: the
    socket's opening, the sentence's characters, the input and output tokens when the API reported them, and the row's UTC time.
Aura-2 at LiveKit's plugin defaults (model aura-2-andromeda-en, linear16, 24 kHz, container none).

Keys come from the environment only (ANTHROPIC_API_KEY, DEEPGRAM_API_KEY) and are never printed or written.

  python bench/e2e/live.py check [--deepgram]        GET /v1/models (one request) -> KEYCHECK_ANTHROPIC_OK / _REFUSED <status>;
                                                     --deepgram: Aura-2's socket opened and closed with nothing sent (101) -> ..._OK
  python bench/e2e/live.py run --texts TEXTS.json --out LIVE.jsonl [--reps 2] [--conc 8] [--max-requests N] [--max-usd X]
         [--max-s 7200] [--pace-s 1800] [--mock] [--mock-rate SPEC] [--backoff-scale 1]
  --mock: the stand-in reply and voice (no network, no key): the same scheduler, caps and fields, with times drawn from a fixed
          seed (a dry run of the chain; never a reading).
  --mock-rate SPEC: the stand-in's planted refusals, a check of the back-off and the rate stop: `i:status[:retry-after s]`, comma
          separated, i the stand-in's i-th Claude request (0-based, retries counted), status 429 or 529.

Anthropic's organization may be shared with live services, so this part must never crowd them out:
  - 8 requests at once by default (--conc 8);
  - a 429 or 529 from Anthropic (the HTTP status, or a stream's `rate_limit_error` / `overloaded_error`) is retried after the
    `retry-after` it gives, else after 2, 4, 8 and 16 s (then the request fails, counted); the request keeps its slot while it waits,
    so fewer go at once; every 429 and 529 is counted in the run's json;
  - the rate stop: when the 429s and 529s pass 1% of the Claude requests in any rolling 200 (3 or more in the last 200 sent) or
    reach 20 in all, the run stops cleanly (LIVE_RATE_STOP): nothing new starts, what is in flight ends or is written as failed, and
    the json says why. The chain then falls back to the 10 callers drawn by seed 23, as for a missed pace, and states the size.
Stops: the request cap (each text-and-rep is one Claude request and one Aura-2 request), the dollar cap (prices given; each request
reserves its worst case before it starts, 80 output tokens and 200 characters, and is settled to its input tokens and the characters
sent once it ends), the time cap, and the pace check
(after 5% of the work, if the whole is projected over --pace-s, it stops with LIVE_PACE_MISSED and writes what it has).
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import os
import pathlib
import random
import re
import sys
import time

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 80
SYSTEM = ("You are the phone agent of a bank, answering a caller. Reply in one short spoken sentence, as a person on the phone would. "
          "No lists, no formatting.")
AURA_URL = "wss://api.deepgram.com/v1/speak"
AURA = {"model": "aura-2-andromeda-en", "encoding": "linear16", "sample_rate": "24000", "container": "none"}
SENTENCE_END = re.compile(r"[.!?](\s|$)")
# prices (read on the day and passed in; these are the defaults as the claude-api skill and Deepgram's page gave them on 19 Nov)
HAIKU_IN, HAIKU_OUT = 0.10e-6, 0.50e-6   # $ a token, prompts under 100K
AURA_CHAR = 0.030 / 1000                  # $ a character
RATE_CODES = (429, 529)
BACKOFF_S = (2.0, 4.0, 8.0, 16.0)          # without a retry-after; after the fourth wait the request fails
RATE_WINDOW, RATE_WINDOW_MAX, RATE_ALL_MAX = 200, 2, 20   # stop: over 1% of the last 200 (3 or more), or 20 in all


def rate_status(e):
    """(429 or 529, retry-after seconds or None) when the exception is Anthropic's rate limit or overload, else (None, None).
    The SDK raises the HTTP status as `status_code`; an error inside a stream carries its type in the body."""
    st = getattr(e, "status_code", None)
    if st not in RATE_CODES:
        body = getattr(e, "body", None)
        err = body.get("error") if isinstance(body, dict) else None
        typ = err.get("type") if isinstance(err, dict) else None
        st = {"rate_limit_error": 429, "overloaded_error": 529}.get(typ)
    if st not in RATE_CODES:
        return None, None
    ra = None
    try:
        h = getattr(getattr(e, "response", None), "headers", None)
        if h is not None and h.get("retry-after") is not None:
            ra = max(0.0, float(h.get("retry-after")))
    except (TypeError, ValueError):
        ra = None
    return st, ra


class PlantedStatus(Exception):
    """the stand-in's planted refusal (--mock-rate), shaped as the SDK's: status_code, response.headers"""

    def __init__(self, status, retry_after=None):
        super().__init__(f"planted {status}")
        self.status_code = status
        self.response = type("R", (), {"headers": {} if retry_after is None else {"retry-after": str(retry_after)}})()


def tid(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- the key checks

async def check_anthropic() -> int:
    from anthropic import AsyncAnthropic, APIStatusError
    c = AsyncAnthropic(max_retries=0, timeout=20.0)
    try:
        await c.models.list(limit=1)
        return 200
    except APIStatusError as e:
        return int(e.status_code)
    except Exception:  # noqa: BLE001  (no network, etc.)
        return -1


async def check_deepgram() -> int:
    import websockets
    from urllib.parse import urlencode
    try:
        ws = await websockets.connect(AURA_URL + "?" + urlencode(AURA),
                                      additional_headers={"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}"},
                                      open_timeout=20, close_timeout=2)
    except websockets.exceptions.InvalidStatus as e:
        return int(e.response.status_code)
    except Exception:  # noqa: BLE001
        return -1
    await ws.close()
    return 101


# ---------------------------------------------------------------- one measurement

class Real:
    def __init__(self, thinking):
        from anthropic import AsyncAnthropic
        self.claude = AsyncAnthropic(max_retries=0, timeout=30.0)
        self.thinking = thinking

    async def voice_socket(self):
        import websockets
        from urllib.parse import urlencode
        return await websockets.connect(AURA_URL + "?" + urlencode(AURA),
                                        additional_headers={"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}"},
                                        max_size=None, ping_interval=None, open_timeout=20, close_timeout=2)

    async def reply(self, words):
        """yields ("usage", (input tokens, None)) at the message's start, then ("text", piece) as Claude streams"""
        extra = {"thinking": {"type": "disabled"}} if self.thinking == "disabled" else {}
        async with self.claude.messages.stream(model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM,
                                               messages=[{"role": "user", "content": words}],
                                               output_config={"effort": "low"}, **extra) as s:
            async for ev in s:
                if ev.type == "message_start":
                    yield ("usage", (ev.message.usage.input_tokens, None))
                elif ev.type == "text" and ev.text:
                    yield ("text", ev.text)

    async def speak(self, ws, text):
        """the sentence's words to Aura-2 (Speak each, as LiveKit's plugin does), Flush; returns at the first audio bytes"""
        for w in text.split():
            await ws.send(json.dumps({"type": "Speak", "text": w + " "}))
        await ws.send(json.dumps({"type": "Flush"}))
        while True:
            m = await ws.recv()
            if isinstance(m, (bytes, bytearray)) and len(m):
                return
            try:
                o = json.loads(m)
            except (TypeError, ValueError):
                continue
            if o.get("type") == "Error" or o.get("err_code"):
                raise RuntimeError(f"aura error {str(o.get('err_code') or o.get('description'))[:80]}")

    async def close(self, ws):
        try:
            await ws.send(json.dumps({"type": "Close"}))
            await asyncio.wait_for(ws.close(), 3)
        except Exception:  # noqa: BLE001
            pass


class Mock:
    """the stand-in: times from a fixed seed per (text, rep); a 9-word sentence; --mock-rate plants refusals at given requests"""

    def __init__(self, scale, rate_spec=""):
        self.scale = scale
        self.n_claude = 0
        self.plant = {}
        for part in filter(None, (rate_spec or "").split(",")):
            bits = part.split(":")
            self.plant[int(bits[0])] = (int(bits[1]), float(bits[2]) if len(bits) > 2 else None)

    def _r(self, words, tag):
        return random.Random(f"{tid(words)}:{tag}:{self._rep}")

    async def voice_socket(self):
        await asyncio.sleep(0.02 * self.scale)
        return object()

    async def reply(self, words):
        i = self.n_claude
        self.n_claude += 1
        if i in self.plant:
            await asyncio.sleep(0.01 * self.scale)
            raise PlantedStatus(*self.plant[i])
        r = self._r(words, "llm")
        yield ("usage", (40, None))
        await asyncio.sleep(r.lognormvariate(-1.1, 0.25) * self.scale)        # ttft, median about 330 ms
        for w in ["Sure,", "I", "can", "help", "you", "with", "that", "right", "now."]:
            await asyncio.sleep(0.012 * self.scale)
            yield ("text", w + " ")

    async def speak(self, ws, text):
        await asyncio.sleep(self._r(text, "tts").lognormvariate(-2.3, 0.3) * self.scale)  # first audio, median about 100 ms

    async def close(self, ws):
        return None


async def one(eng, words, rep):
    if isinstance(eng, Mock):
        eng._rep = rep
    t0 = time.perf_counter()
    ms = lambda: round((time.perf_counter() - t0) * 1000.0, 2)  # noqa: E731
    out = {"ttft": None, "sentence": None, "first_audio": None, "socket_open": None, "chars": 0, "in_tok": None}

    async def opened():
        ws = await eng.voice_socket()
        out["socket_open"] = ms()
        return ws
    vtask = asyncio.ensure_future(opened())
    text = ""
    gen = eng.reply(words)
    try:
        try:
            async for kind, x in gen:
                if kind == "usage":
                    out["in_tok"] = x[0]
                    continue
                if out["ttft"] is None:
                    out["ttft"] = ms()
                text += x
                mm = SENTENCE_END.search(text)
                if mm:                       # cut after the first sentence: the stream is closed here
                    text = text[: mm.end()]
                    break
        finally:
            await gen.aclose()
        out["sentence"] = ms()
        text = text.strip()
        out["chars"] = len(text)
        ws = await vtask
        if text:
            await eng.speak(ws, text)
            out["first_audio"] = ms()
        await eng.close(ws)
    finally:
        if not vtask.done():
            vtask.cancel()
    return out


# ---------------------------------------------------------------- the run

async def run(a):
    texts = json.load(open(a.texts, encoding="utf-8"))
    work = [(t["id"], t["text"], r) for r in range(a.reps) for t in texts]
    done = set()
    outp = pathlib.Path(a.out)
    if outp.exists():   # a stopped run continues where it stopped
        for ln in open(outp, encoding="utf-8"):
            if ln.strip():
                x = json.loads(ln)
                if x.get("ok"):
                    done.add((x["id"], x["rep"]))
    work = [w for w in work if (w[0], w[2]) not in done]
    eng = Mock(a.mock_scale, a.mock_rate) if a.mock else Real(a.thinking)
    sem = asyncio.Semaphore(a.conc)
    state = {"requests": 0, "usd": 0.0, "ok": 0, "err": 0, "stop": None}
    # every Claude request sent (retries counted), and the 429s and 529s among them
    rate = {"claude_requests": 0, "429": 0, "529": 0, "retried": 0, "gave_up": 0, "waited_s": 0.0, "retry_after_used": 0,
            "max_in_window": 0, "stopped_in_flight": 0, "events": []}
    window = collections.deque(maxlen=RATE_WINDOW)

    def note(limited, st=None):
        rate["claude_requests"] += 1
        window.append(1 if limited else 0)
        if limited:
            rate[str(st)] += 1
            n_all = rate["429"] + rate["529"]
            rate["max_in_window"] = max(rate["max_in_window"], sum(window))
            rate["events"].append({"at": round(time.time() - t_start, 2), "status": st, "claude_request": rate["claude_requests"],
                                   "in_last_200": sum(window), "in_all": n_all})
            if not state["stop"] and (sum(window) > RATE_WINDOW_MAX or n_all >= RATE_ALL_MAX):
                state["stop"] = "LIVE_RATE_STOP"
                print(f"rate stop: {sum(window)} of the last {len(window)} Claude requests refused (429/529), {n_all} in all", flush=True)
    t_start = time.time()
    f = open(outp, "a", encoding="utf-8")
    warm = max(20, len(work) // 20)

    def worst(words):
        return len(words) / 4 * HAIKU_IN * a.price_scale + MAX_TOKENS * HAIKU_OUT * a.price_scale + 200 * AURA_CHAR * a.price_scale

    async def go(i, w):
        id_, words, rep = w
        async with sem:
            if state["stop"]:
                return
            if state["requests"] >= a.max_requests:
                state["stop"] = "LIVE_REQUEST_CAP"
                return
            if state["usd"] + worst(words) > a.max_usd:
                state["stop"] = "LIVE_DOLLAR_CAP"
                return
            if time.time() - t_start > a.max_s:
                state["stop"] = "LIVE_TIME_CAP"
                return
            state["requests"] += 1
            state["usd"] += worst(words)
            row = {"id": id_, "rep": rep, "mock": bool(a.mock), "at": round(time.time() - t_start, 2),
                   "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            err = None
            attempt, waits = 0, 0
            while True:
                try:
                    row.update(await asyncio.wait_for(one(eng, words, rep), 30))
                    note(False)
                    err = None
                    break
                except Exception as e:  # noqa: BLE001  a failure is counted and written, never dropped
                    err = f"{type(e).__name__}: {str(e)[:80]}"
                    st, ra = rate_status(e)
                    if st is None:
                        note(False)
                        attempt += 1
                        if attempt >= 2:
                            break
                        await asyncio.sleep(1.0)
                        continue
                    note(True, st)
                    row.setdefault("rate_limited", []).append(st)
                    if state["stop"] == "LIVE_RATE_STOP":
                        rate["stopped_in_flight"] += 1
                        err = f"rate_stop after {st}"
                        break
                    if waits >= len(BACKOFF_S):
                        rate["gave_up"] += 1
                        break
                    wait = ra if ra is not None else BACKOFF_S[waits]
                    rate["retry_after_used"] += int(ra is not None)
                    waits += 1
                    rate["retried"] += 1
                    rate["waited_s"] += wait
                    await asyncio.sleep(wait * a.backoff_scale)
                    if state["stop"] == "LIVE_RATE_STOP":   # stopped while this one waited: written as failed, retried by the fallback
                        rate["stopped_in_flight"] += 1
                        err = f"rate_stop while waiting after {st}"
                        break
            # the reservation is settled to what was spent: the input tokens Claude reported, 80 output tokens (the cut leaves fewer),
            # the sentence's characters sent to Aura-2 (none if the voice was never reached)
            spent = ((row.get("in_tok") or len(words) / 4) * HAIKU_IN + MAX_TOKENS * HAIKU_OUT + row.get("chars", 0) * AURA_CHAR) * a.price_scale
            state["usd"] += spent - worst(words)
            row["usd"] = round(spent, 6)
            row["ok"] = err is None and row.get("first_audio") is not None
            if err:
                row["err"] = err
            state["ok" if row["ok"] else "err"] += 1
            f.write(json.dumps(row) + "\n")
            f.flush()
            n = state["ok"] + state["err"]
            if n == warm and len(work) > warm:
                proj = (time.time() - t_start) / n * len(work)
                print(f"pace: {n} of {len(work)} in {time.time() - t_start:.0f} s, the whole projected {proj:.0f} s", flush=True)
                if proj > a.pace_s:
                    state["stop"] = "LIVE_PACE_MISSED"

    await asyncio.gather(*(go(i, w) for i, w in enumerate(work)))
    f.close()
    res = {"texts": len(texts), "reps": a.reps, "work": len(work), "already_done": len(done), "requests": state["requests"],
           "ok": state["ok"], "errors": state["err"], "usd_counted": round(state["usd"], 4), "seconds": round(time.time() - t_start, 1),
           "stop": state["stop"], "mock": bool(a.mock), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)),
           "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": MODEL, "voice": AURA, "thinking": a.thinking, "conc": a.conc,
           "rate_limits": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in rate.items()},
           "rate_rule": {"backoff_s": BACKOFF_S, "window": RATE_WINDOW, "stop_over_in_window": RATE_WINDOW_MAX,
                         "stop_at_in_all": RATE_ALL_MAX},
           "mock_rate": a.mock_rate or None, "backoff_scale": a.backoff_scale}
    pathlib.Path(str(outp) + ".run.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res))
    print(state["stop"] or "LIVE_DONE")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--deepgram", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--texts", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--reps", type=int, default=2)
    r.add_argument("--conc", type=int, default=8, help="requests at once (8: the organization may be shared with live services)")
    r.add_argument("--max-requests", type=int, default=20000)
    r.add_argument("--max-usd", type=float, default=30.0)
    r.add_argument("--max-s", type=float, default=7200.0)
    r.add_argument("--pace-s", type=float, default=1800.0)
    r.add_argument("--price-scale", type=float, default=1.0, help="multiplies the prices above, if the day's pages differ")
    r.add_argument("--thinking", choices=("disabled", "default"), default="disabled")
    r.add_argument("--mock", action="store_true")
    r.add_argument("--mock-scale", type=float, default=1.0)
    r.add_argument("--mock-rate", default="", help="the stand-in's planted 429/529s: i:status[:retry-after],... (a check only)")
    r.add_argument("--backoff-scale", type=float, default=1.0, help="multiplies the waits (a check of the stand-in only; 1 on the day)")
    a = ap.parse_args()
    if a.cmd == "check":
        code = asyncio.run(check_anthropic())
        print(f"KEYCHECK_ANTHROPIC_{'OK' if code == 200 else 'REFUSED'} {code}")
        ok = code == 200
        if a.deepgram:
            dc = asyncio.run(check_deepgram())
            print(f"KEYCHECK_DEEPGRAM_{'OK' if dc == 101 else 'REFUSED'} {dc}")
            ok = ok and dc == 101
        sys.exit(0 if ok else 3)
    if not a.mock and (a.mock_rate or a.backoff_scale != 1.0):
        raise SystemExit("LIVE_REFUSED: --mock-rate and --backoff-scale are for the stand-in only")
    if not a.mock:
        for k in ("ANTHROPIC_API_KEY", "DEEPGRAM_API_KEY"):
            if not os.environ.get(k):
                raise SystemExit(f"LIVE_REFUSED: {k} is not in the environment")
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
