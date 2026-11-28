#!/usr/bin/env python3
"""See what Bell would do on your own calls, with your own Rabel API key: a text read.

Your transcripts go only to the Rabel API, in your own API calls with your own key. Nothing is written to disk except the
report, and only when you ask for it with --out.

    python try_bell.py calls.jsonl --key "$RABEL_API_KEY"
    python try_bell.py calls_folder/ --teach-share 0.7 --out report.txt

1. Reads your calls (JSONL, JSON or CSV; see README.md). Only the caller's turns are kept. The agent's lines are dropped
   before anything is sent.
2. Splits the calls by their order: the first part (70% by default) teaches a new agent through
   POST /v1/agents/{agent}/calls (free). The rest are held out and are not taught before they are played.
3. Plays each held-out call through the live conversation endpoint, as a hookup does: each caller turn's words are sent
   as growing interim text, one word at a time, then the final words. Every guess Bell sends is recorded, and compared
   with the final words by the hookups' rule (case and punctuation ignored).
4. Prints a short report. It counts words, never milliseconds: a text read is not a timing.

Python 3.11 or later, and the `websockets` package (pip install websockets).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import secrets
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

try:
    from websockets.asyncio.client import connect
    from websockets.exceptions import ConnectionClosed, InvalidStatus
except ImportError:  # pragma: no cover
    sys.exit("try_bell.py needs the websockets package: pip install websockets")

__version__ = "0.1.0"
DEFAULT_URL = "https://api.rabellabs.com"

# The API's limits on teaching (its README, "Your agents"). The script stays a little under the byte limit.
TEACH_CALLS_A_REQUEST = 1_000
TEACH_TURNS_A_REQUEST = 50_000
TEACH_BYTES_A_REQUEST = 7_500_000  # the API's limit is 8,000,000
TEACH_HEARD_A_CALL = 20_000  # a turn taught from its final words counts one for each word
TEACH_HEARD_A_REQUEST = 2_000_000
TEXT_MAX = 4_096  # characters in one text
SILENT_CALLS = 2  # a new agent sends no guess until it has learned from this many calls

CALLER = {"caller", "user", "customer", "client", "human", "patient", "speaker_caller"}
AGENT = {"agent", "assistant", "bot", "system", "ai", "operator", "rep", "representative", "speaker_agent"}

_AGENT_NAME = re.compile(r"^[a-z0-9_-]{1,64}$")


# ---------------------------------------------------------------------------------------------------- the match rule

# The words as the hookups and the API compare them (LiveKit's split_words(ignore_punctuation=True,
# split_character=True), as pipecat-rabel's _form.words_of ports it): split on spaces, a character of a script written
# without spaces is a word on its own, punctuation dropped, each word case-folded. Two texts match when their words are
# equal.
_PUNCTUATION = frozenset("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~±—‘’“”…")
_UNSPACED = re.compile(r"[぀-ヿ㐀-䶿一-鿿฀-๿]")


def _is_punctuation(c: str) -> bool:
    return c in _PUNCTUATION or unicodedata.category(c).startswith("P")


def _word_spans(text: str) -> list[tuple[str, int]]:
    """Each word of ``text`` as the API counts it, with the position in ``text`` where it ends."""
    out: list[tuple[str, int]] = []

    def add(a: int, b: int):
        w = "".join(c for c in text[a:b] if not _is_punctuation(c))
        if w:
            out.append((w.casefold(), b))

    start = 0
    for pos, ch in enumerate(text):
        if ch.isspace():
            add(start, pos)
            start = pos + 1
        elif _UNSPACED.match(ch):
            if start < pos:
                add(start, pos)
            add(pos, pos + 1)
            start = pos + 1
    add(start, len(text))
    return out


def words_of(text: str | None) -> tuple[str, ...]:
    return tuple(w for w, _ in _word_spans(text or ""))


def growing(text: str) -> list[str]:
    """The interim texts a speech-to-text would show for ``text``, one word more each time, in the text's own spelling."""
    out: list[str] = []
    for _, end in _word_spans(text):
        h = text[:end].strip()
        if h and (not out or out[-1] != h):
            out.append(h)
    return out


# ---------------------------------------------------------------------------------------------------- reading calls


@dataclass
class Call:
    call_id: str
    turns: list[str]  # the caller's turns, in order
    agent_lines: int = 0


class InputError(Exception):
    pass


def _speaker(value, where: str) -> str:
    s = str(value or "").strip().lower()
    if s in CALLER:
        return "caller"
    if s in AGENT:
        return "agent"
    raise InputError(f"{where}: speaker {value!r} is neither a caller ({', '.join(sorted(CALLER))}) "
                     f"nor an agent ({', '.join(sorted(AGENT))})")


def _call_of(obj, where: str, n: int) -> Call:
    if isinstance(obj, list):
        turns, cid = obj, f"call-{n}"
    elif isinstance(obj, dict) and isinstance(obj.get("turns"), list):
        turns, cid = obj["turns"], str(obj.get("call_id") or obj.get("id") or f"call-{n}")
    else:
        raise InputError(f'{where}: a call is {{"call_id": "...", "turns": [{{"speaker": "caller", "text": "..."}}, ...]}}')
    call = Call(cid, [])
    for i, t in enumerate(turns):
        if not isinstance(t, dict) or not isinstance(t.get("text"), str):
            raise InputError(f'{where}, turn {i + 1}: a turn is {{"speaker": "caller" or "agent", "text": "..."}}')
        if _speaker(t.get("speaker", t.get("role")), f"{where}, turn {i + 1}") == "caller":
            if t["text"].strip():
                call.turns.append(t["text"].strip())
        else:
            call.agent_lines += 1
    return call


def _read_json(path: Path) -> list[Call]:
    text = path.read_text(encoding="utf-8-sig")
    calls: list[Call] = []
    if path.suffix.lower() == ".jsonl":
        for ln, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise InputError(f"{path}, line {ln}: not JSON ({e.msg})") from None
                calls.append(_call_of(obj, f"{path}, line {ln}", len(calls) + 1))
        return calls
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise InputError(f"{path}: not JSON ({e.msg})") from None
    if isinstance(obj, dict) and isinstance(obj.get("calls"), list):
        obj = obj["calls"]
    if isinstance(obj, dict) and isinstance(obj.get("turns"), list):
        obj = [obj]
    if not isinstance(obj, list):
        raise InputError(f'{path}: expected a list of calls, or {{"calls": [...]}}')
    if obj and all(isinstance(t, dict) and "text" in t and "turns" not in t for t in obj):
        obj = [obj]  # one call written as a bare list of turns
    for i, c in enumerate(obj):
        calls.append(_call_of(c, f"{path}, call {i + 1}", len(calls) + 1))
    return calls


def _read_csv(path: Path) -> list[Call]:
    by_id: dict[str, Call] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = csv.DictReader(f)
        cols = {c.strip().lower(): c for c in (rows.fieldnames or [])}
        if not {"call_id", "speaker", "text"} <= set(cols):
            raise InputError(f"{path}: a CSV needs the columns call_id, speaker, text (it has {rows.fieldnames})")
        for ln, row in enumerate(rows, 2):
            cid = (row[cols["call_id"]] or "").strip()
            if not cid:
                raise InputError(f"{path}, line {ln}: no call_id")
            call = by_id.setdefault(cid, Call(cid, []))
            text = (row[cols["text"]] or "").strip()
            if _speaker(row[cols["speaker"]], f"{path}, line {ln}") == "caller":
                if text:
                    call.turns.append(text)
            else:
                call.agent_lines += 1
    return list(by_id.values())


def read_calls(src: Path) -> list[Call]:
    """The calls in a file, or in a folder's .jsonl, .json and .csv files taken in the order of their names."""
    if src.is_dir():
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in (".jsonl", ".json", ".csv"))
        if not files:
            raise InputError(f"{src}: no .jsonl, .json or .csv files")
    elif src.is_file():
        files = [src]
    else:
        raise InputError(f"{src}: no such file or folder")
    calls: list[Call] = []
    for p in files:
        calls += _read_csv(p) if p.suffix.lower() == ".csv" else _read_json(p)
    return calls


def fits(text: str) -> bool:
    """Whether the API takes this turn: a text and its words, joined, at most 4,096 characters."""
    w = words_of(text)
    return bool(w) and len(text) <= TEXT_MAX and sum(map(len, w)) + len(w) - 1 <= TEXT_MAX


# ---------------------------------------------------------------------------------------------------- the API


class ApiError(Exception):
    def __init__(self, status: int, error: str, message: str, retry_after_s: float = 0):
        super().__init__(f"{status} {error}: {message}")
        self.status, self.error, self.message, self.retry_after_s = status, error, message, retry_after_s


def _urls(url: str) -> tuple[str, str]:
    """The API's base address (https) and its conversation endpoint (wss)."""
    base = url.rstrip("/")
    if base.endswith("/v1/conversation"):
        base = base[: -len("/v1/conversation")]
    if base.startswith("wss://"):
        base = "https://" + base[6:]
    elif base.startswith("ws://"):
        base = "http://" + base[5:]
    elif not base.startswith(("http://", "https://")):
        base = "https://" + base
    ws = ("wss://" + base[8:] if base.startswith("https://") else "ws://" + base[7:]) + "/v1/conversation"
    return base, ws


def http(base: str, key: str, method: str, path: str, body=None, timeout: float = 120):
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(base + path, data=data, method=method, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": f"try-bell/{__version__}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        try:
            j = json.loads(e.read())
        except Exception:  # noqa: BLE001
            j = {}
        raise ApiError(e.code, j.get("error", "http_error"), j.get("message", str(e)), j.get("retry_after_s") or 0) from None


def batches(calls: list[list[str]]):
    """The calls in requests within the API's limits."""
    part: list[list[str]] = []
    size = turns = heard = 0
    for turns_of in calls:
        doc = {"turns": [{"final": t} for t in turns_of]}
        b = len(json.dumps(doc, ensure_ascii=False).encode()) + 2
        h = sum(len(words_of(t)) for t in turns_of)
        if part and (len(part) >= TEACH_CALLS_A_REQUEST or size + b > TEACH_BYTES_A_REQUEST
                     or turns + len(turns_of) > TEACH_TURNS_A_REQUEST or heard + h > TEACH_HEARD_A_REQUEST):
            yield part
            part, size, turns, heard = [], 0, 0, 0
        part.append(turns_of)
        size, turns, heard = size + b, turns + len(turns_of), heard + h
    if part:
        yield part


def split_long(turns: list[str]) -> list[list[str]]:
    """A call cut where it would pass the API's 20,000 heard messages a call (each word of a final counts one)."""
    out, cur, n = [], [], 0
    for t in turns:
        w = len(words_of(t))
        if cur and n + w > TEACH_HEARD_A_CALL:
            out.append(cur)
            cur, n = [], 0
        cur.append(t)
        n += w
    if cur:
        out.append(cur)
    return out


def teach(base: str, key: str, agent: str, calls: list[list[str]], log) -> int:
    """Teaches the calls (final words only) to the agent; returns the agent's calls_heard."""
    heard = 0
    pieces = [p for c in calls for p in split_long(c)]
    done = 0
    for part in batches(pieces):
        for attempt in range(3):
            try:
                r = http(base, key, "POST", f"/v1/agents/{agent}/calls", {"calls": [{"turns": [{"final": t} for t in c]}
                                                                                  for c in part]})
                break
            except ApiError as e:
                if e.status == 429 and e.retry_after_s and e.retry_after_s <= 120 and attempt < 2:
                    log(f"  the API asks to wait {e.retry_after_s:.0f} s ({e.error}); waiting")
                    time.sleep(e.retry_after_s)
                    continue
                if e.error == "teach_limit":
                    raise ApiError(e.status, e.error, e.message + " The script teaches at most 10,000 calls an hour; use "
                                   "a smaller file or a smaller --teach-share.", e.retry_after_s) from None
                raise
        done += len(part)
        heard = r.get("calls_heard", heard)
        log(f"  taught {done} of {len(pieces)} calls")
    return heard


# ---------------------------------------------------------------------------------------------------- playing


@dataclass
class TurnResult:
    words: int
    guesses: int = 0
    wrong: int = 0
    right: bool = False  # the last guess before the final words equalled them (the guess a hookup keeps)
    early: int = 0  # words of the final still to come when that guess was sent (0: at the last word)


@dataclass
class Play:
    turns: list[TurnResult] = field(default_factory=list)
    silent_calls: int = 0  # conversations the API opened silent (the agent had learned from too few calls)
    calls_heard_first: int | None = None


async def _recv_until_pong(ws, ping_id: str, guesses: list):
    while True:
        m = json.loads(await asyncio.wait_for(ws.recv(), 30))
        kind = m.get("type") if isinstance(m, dict) else None
        if kind == "pong" and m.get("id") == ping_id:
            return
        if kind == "guess":
            guesses.append(m)
        elif kind == "error":
            raise ApiError(int(m.get("status") or 0), str(m.get("error")), str(m.get("message")),
                           m.get("retry_after_s") or 0)


async def play_call(ws_url: str, key: str, agent: str, turns: list[str], learn: bool, play: Play):
    for attempt in range(4):
        start, silent = len(play.turns), play.silent_calls
        try:
            async with connect(ws_url, additional_headers={"Authorization": f"Bearer {key}"},
                               user_agent_header=f"try-bell/{__version__}", open_timeout=30, close_timeout=5,
                               max_size=2**20) as ws:
                await ws.send(json.dumps({"type": "hello", "v": 3, "agent": agent, "learn": learn,
                                          "host": "try-bell", "hookup": __version__}))
                ready = json.loads(await asyncio.wait_for(ws.recv(), 30))
                if ready.get("type") == "error":
                    raise ApiError(int(ready.get("status") or 0), str(ready.get("error")), str(ready.get("message")),
                                   ready.get("retry_after_s") or 0)
                if ready.get("type") != "ready":
                    raise ApiError(0, "unexpected", f"the API answered the hello with {ready.get('type')!r}")
                if ready.get("v") != 3:
                    raise ApiError(0, "old_api", "this API does not serve version 3 of the conversation (the new Bell)")
                ch = ready.get("calls_heard")
                if play.calls_heard_first is None:
                    play.calls_heard_first = ch
                if isinstance(ch, int) and ch < SILENT_CALLS:
                    play.silent_calls += 1
                for n, text in enumerate(turns):
                    final = words_of(text)
                    res = TurnResult(words=len(final))
                    got: list[dict] = []
                    for h in growing(text):
                        await ws.send(json.dumps({"type": "heard", "text": h}))
                    # A ping is answered once every answer to the messages before it has gone out: after it, Bell has
                    # nothing more to say to these words. The turn's words arrive faster than speech, which changes
                    # nothing Bell guesses: it answers each message in order, from its words alone.
                    pid = f"t{n}"
                    await ws.send(json.dumps({"type": "ping", "id": pid}))
                    await _recv_until_pong(ws, pid, got)
                    await ws.send(json.dumps({"type": "end_of_turn", "text": text}))
                    while True:  # the API's answer to the end of the turn
                        m = json.loads(await asyncio.wait_for(ws.recv(), 30))
                        if m.get("type") == "end_of_turn":
                            break
                        if m.get("type") == "error":
                            raise ApiError(int(m.get("status") or 0), str(m.get("error")), str(m.get("message")))
                    seen = {}
                    for g in got:
                        seen.setdefault(g.get("id"), g)
                    gs = list(seen.values())
                    res.guesses = len(gs)
                    ok = [words_of(g.get("text")) == final for g in gs]
                    res.wrong = ok.count(False)
                    if gs and ok[-1]:
                        res.right = True
                        i = len(ok) - 1
                        while i > 0 and ok[i - 1]:
                            i -= 1  # the first of the run of right guesses the hookup would have kept
                        res.early = max(0, len(final) - len(words_of(gs[i].get("heard"))))
                    play.turns.append(res)
                await ws.close()  # a clean close: with learn on, Bell learns from this call now, as it would live
                return
        except InvalidStatus as e:
            status = e.response.status_code
            try:
                j = json.loads(e.response.body or b"{}")
            except Exception:  # noqa: BLE001
                j = {}
            err = ApiError(status, j.get("error", "refused"), j.get("message", ""), j.get("retry_after_s") or 0)
        except ApiError as e:
            err = e
        except (OSError, ConnectionClosed, asyncio.TimeoutError, TimeoutError) as e:
            err = ApiError(0, "connection", f"{type(e).__name__}: {e}")
        # A refusal or a failure before the call's first turn is tried again. One later is not: with learn on, the
        # API has learned the call's turns before the cut, and playing them again would read them against themselves.
        retry = err.status in (408, 429, 503, 529) or err.error == "connection"
        if retry and attempt < 3 and len(play.turns) == start:
            play.silent_calls = silent
            await asyncio.sleep(max(2.0 * (attempt + 1), min(float(err.retry_after_s or 0), 60.0)))
            continue
        raise err


# ---------------------------------------------------------------------------------------------------- the report

NOTE = (
    "This is a text read. Each caller turn's final words were sent to Bell one word at a time, as a speech-to-text's\n"
    "interim words, then the final words. Live speech-to-text sends fewer, later and less clean updates, so a live\n"
    "share is lower: on our bank calls Bell guessed 57% of turns in text against about a third through LiveKit.\n"
    "This read measures no time. Milliseconds come only from a live install: its hookup reports the head start of\n"
    "each kept guess in your agent's status (GET /v1/agents/{agent})."
)


def report(agent: str, n_calls: int, n_taught: int, n_held: int, share: float, skipped: int, dropped_agent: int,
           play: Play, learn: bool) -> str:
    T = play.turns
    n = len(T)
    right = [t for t in T if t.right]
    word = [t for t in right if t.early > 0]
    pct = (lambda k: f"{100 * k / n:.0f}%") if n else (lambda k: "-")
    avg_early = sum(t.early for t in right) / len(right) if right else 0.0
    lines = [
        "Bell on your calls: a text read",
        "",
        f"  agent                 {agent}",
        f"  calls                 {n_calls}: {n_taught} taught (the first {share:.0%}), {n_held} held out and played"
        + ("" if learn else " (not learned from)"),
        f"  caller turns played   {n}" + (f"  ({skipped} left out: over the API's 4,096 characters, or no words)"
                                         if skipped else ""),
        f"  agent lines dropped   {dropped_agent} (never sent)",
        "",
        f"  turns Bell guessed before the caller finished   {len(right)} of {n} ({pct(len(right))})",
        f"    with words still to come                      {len(word)} ({pct(len(word))})",
        f"    at the last word, before the final words      {len(right) - len(word)} ({pct(len(right) - len(word))})",
        f"  on those turns, words early on average          {avg_early:.1f}",
        f"  wrong guesses a turn                            {(sum(t.wrong for t in T) / n if n else 0):.2f}"
        "  (each one an extra call to your agent's LLM, never heard by the caller)",
        f"  guesses a turn                                  {(sum(t.guesses for t in T) / n if n else 0):.2f}",
        "",
    ]
    if play.silent_calls:
        lines += [f"  Bell was silent on {play.silent_calls} held-out calls: the agent had learned from fewer than "
                  f"{SILENT_CALLS} calls when they started. Teach more calls (--teach-share).", ""]
    lines += [NOTE]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="See what Bell would do on your own calls: a text read with your Rabel API key.")
    ap.add_argument("calls", type=Path, help="a .jsonl, .json or .csv file of calls, or a folder of them")
    ap.add_argument("--key", default=None, help="your Rabel API key (default: the RABEL_API_KEY environment variable)")
    ap.add_argument("--agent", default=None, help="the agent to teach and play: a new one by default, try-xxxx")
    ap.add_argument("--teach-share", type=float, default=0.7, help="the share of calls, from the first, taught (0.7)")
    ap.add_argument("--url", default=None, help=f"the API (default: RABEL_API_URL, else {DEFAULT_URL})")
    ap.add_argument("--no-learn", action="store_true",
                    help="the held-out calls teach nothing after they are played (a live agent learns from each)")
    ap.add_argument("--forget", action="store_true",
                    help="after the report, delete what Bell learned for the agent (DELETE /v1/agents/{agent})")
    ap.add_argument("--out", type=Path, default=None, help="also write the report to this file")
    a = ap.parse_args(argv)

    key = a.key or os.environ.get("RABEL_API_KEY", "")
    if not key:
        ap.error("no key: pass --key or set RABEL_API_KEY")
    if not 0 < a.teach_share < 1:
        ap.error("--teach-share is a share between 0 and 1, such as 0.7")
    agent = a.agent or f"try-{secrets.token_hex(2)}"
    if not _AGENT_NAME.match(agent):
        ap.error("--agent is 1 to 64 lower-case letters, digits, '-' or '_'")
    if agent == "demo-bank":
        ap.error("demo-bank is read-only; give another agent name")
    base, ws_url = _urls(a.url or os.environ.get("RABEL_API_URL") or DEFAULT_URL)
    log = lambda s: print(s, file=sys.stderr, flush=True)  # noqa: E731

    try:
        calls = read_calls(a.calls)
    except InputError as e:
        log(f"error: {e}")
        return 2
    except UnicodeDecodeError as e:
        log(f"error: a file is not UTF-8 text ({e})")
        return 2
    dropped_agent = sum(c.agent_lines for c in calls)
    skipped = sum(1 for c in calls for t in c.turns if not fits(t))
    caller = [[t for t in c.turns if fits(t)] for c in calls]
    caller = [c for c in caller if c]
    if len(caller) < SILENT_CALLS + 1:
        log(f"error: {len(caller)} calls with caller turns; at least {SILENT_CALLS + 1} are needed "
            "(a few hundred give a steadier read)")
        return 2
    n_teach = min(max(round(len(caller) * a.teach_share), SILENT_CALLS), len(caller) - 1)
    taught, held = caller[:n_teach], caller[n_teach:]
    log(f"{len(calls)} calls read: {sum(map(len, caller))} caller turns kept, {dropped_agent} agent lines dropped")

    try:
        try:
            st = http(base, key, "GET", f"/v1/agents/{agent}")
            if a.agent:
                log(f"note: the agent {agent} has already learned from {st.get('calls_heard', '?')} calls; the read is "
                    "not of a new agent")
            else:
                log(f"error: the agent {agent} already exists; run again for another name")
                return 2
        except ApiError as e:
            if e.status != 404:
                raise
        log(f"teaching {len(taught)} calls to {agent} (teaching is free)")
        teach(base, key, agent, taught, log)
        log(f"playing {len(held)} held-out calls through the conversation endpoint (this uses conversation seconds)")
        play = Play()

        async def run():
            for i, turns in enumerate(held, 1):
                await play_call(ws_url, key, agent, turns, not a.no_learn, play)
                if i % 10 == 0 or i == len(held):
                    log(f"  played {i} of {len(held)} calls")

        asyncio.run(run())
    except ApiError as e:
        log(f"error from the API: {e}")
        return 1
    except urllib.error.URLError as e:
        log(f"error: the API cannot be reached ({e.reason})")
        return 1

    text = report(agent, len(caller), len(taught), len(held), n_teach / len(caller), skipped, dropped_agent, play,
                  not a.no_learn)
    print(text)
    if a.out:
        a.out.write_text(text + "\n", encoding="utf-8")
        log(f"report written to {a.out}")
    if a.forget:
        try:
            http(base, key, "DELETE", f"/v1/agents/{agent}")
            log(f"what Bell learned for {agent} is deleted")
        except ApiError as e:
            log(f"the agent could not be deleted: {e}")
    else:
        log(f"the agent {agent} keeps what it learned; DELETE /v1/agents/{agent} (or --forget) removes it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
