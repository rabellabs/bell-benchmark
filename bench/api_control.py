#!/usr/bin/env python3
"""The control through the Rabel API, before anything else runs: the 28 test callers' 376 calls sent to an agent as their heard texts
and final words (data/streams_28.json), "learn": false, several conversations at once; every guess and every end_of_turn read.

For `demo-bank` the recorded result is 741 guesses, 586 of them equal to the turn's final words (the API's own `right`). The same
read on an agent this run taught all 1,070 of the 27 known callers' calls must give the same guesses (`--same-as` a first read).

`--v 3` reads the same calls with a version 3 hello, as the hookup 1.9.0 announces itself, which gets the new Bell: a turn can get
more than one guess, and `right` says whether any of them equalled the final words. Its two numbers are every guess sent and the
turns right.

  python bench/api_control.py --streams data/streams_28.json --agent demo-bank --out out/api_control_demo-bank.json \
      [--v 2|3] [--want 741,586] [--same-as out/api_control_demo-bank.json] [--at-once 8]
Prints API_CONTROL_PASSES or API_CONTROL_FAILS (and exits 1).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rabel_api as RA  # noqa: E402


def one_call(agent, call, tries=3, v=2):
    """the call, read again from its start if its conversation broke (the network), at most `tries` times"""
    for i in range(tries):
        try:
            return _one_call(agent, call) if v == 2 else _one_call_v3(agent, call)
        except (OSError, RA.Closed):
            if i == tries - 1:
                raise
            time.sleep(2.0)


def _one_call(agent, call):
    """-> [(turn, guess text or None, right, round trip ms of the guess or None)]"""
    conv = RA.Conversation({"v": 2, "agent": agent, "learn": False, "host": "rabel-bell-benchmark"})
    out = []
    try:
        for tn in call["turns"]:
            g = rt = None
            for h in tn["heard"]:
                for t, m in conv.ask({"type": "heard", "text": h}):
                    if m.get("type") == "guess":
                        if g is not None:
                            raise RuntimeError("two guesses in one turn")
                        g, rt = m, t
            ans = conv.ask({"type": "end_of_turn", "text": tn["final"]})
            e = next(m for _t, m in ans if m.get("type") == "end_of_turn")
            if g is None and e.get("right") is not None:
                raise RuntimeError("a right or wrong without a guess")
            out.append((tn["turn"], None if g is None else g["text"], e.get("right"), None if rt is None else round(rt * 1000.0, 2)))
    finally:
        conv.close()
    return out


def _one_call_v3(agent, call):
    """version 3 -> [(turn, [guess texts in the order sent], right, round trip ms of the turn's first guess or None)]"""
    conv = RA.Conversation({"v": 3, "agent": agent, "learn": False, "host": "rabel-bell-benchmark"})
    if conv.ready.get("v") != 3:
        conv.close()
        raise RuntimeError(f"a version 3 hello was answered by a version {conv.ready.get('v')} ready")
    out = []
    try:
        for tn in call["turns"]:
            gs, rt = [], None
            for h in tn["heard"]:
                for t, m in conv.ask({"type": "heard", "text": h}):
                    if m.get("type") == "guess":
                        gs.append(m["text"])
                        rt = t if rt is None else rt
            ans = conv.ask({"type": "end_of_turn", "text": tn["final"]})
            e = next(m for _t, m in ans if m.get("type") == "end_of_turn")
            if not gs and e.get("right") is not None:
                raise RuntimeError("a right or wrong without a guess")
            out.append((tn["turn"], gs, e.get("right"), None if rt is None else round(rt * 1000.0, 2)))
    finally:
        conv.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", required=True)
    ap.add_argument("--agent", default="demo-bank")
    ap.add_argument("--v", type=int, default=2, choices=(2, 3), help="the hello's version (3: the new Bell, as the hookup 1.9.0 asks)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--want", default="", help="guesses,right that must be read (demo-bank: 741,586)")
    ap.add_argument("--same-as", default="", help="a first read whose guesses this one must equal turn by turn")
    ap.add_argument("--at-once", type=int, default=8)
    ap.add_argument("--calls", type=int, default=0, help="only the first N calls (a dry run; no gate)")
    a = ap.parse_args()
    calls = json.load(open(a.streams, encoding="utf-8"))["calls"]
    if a.calls:
        calls = calls[: a.calls]
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(a.at_once) as ex:
        res = list(ex.map(lambda c: one_call(a.agent, c, v=a.v), calls))
    rows = [{"turn": t, "guess": g, "right": r, "rt_ms": rt} for call in res for t, g, r, rt in call]
    if a.v == 2:
        sends = sum(1 for r in rows if r["guess"] is not None)
    else:   # every guess sent; a turn without one has an empty list
        sends = sum(len(r["guess"]) for r in rows)
    right = sum(1 for r in rows if r["right"])
    rts = sorted(r["rt_ms"] for r in rows if r["rt_ms"] is not None)
    q = lambda f: rts[min(len(rts) - 1, int(f * len(rts)))] if rts else None  # noqa: E731
    ok, why = True, []
    if a.want and not a.calls:
        w = [int(x) for x in a.want.split(",")]
        if [sends, right] != w:
            ok = False
            why.append(f"read {sends}, {right}; want {w[0]}, {w[1]}")
    if a.same_as:
        first = {r["turn"]: r["guess"] for r in json.load(open(a.same_as, encoding="utf-8"))["rows"]}
        diff = [r["turn"] for r in rows if first.get(r["turn"], "missing") != r["guess"]]
        if diff:
            ok = False
            why.append(f"{len(diff)} turns' guesses differ from {a.same_as} (first: {diff[:3]})")
    res_ = {"agent": a.agent, "v": a.v, "url": RA.base_url(), "calls": len(calls), "turns": len(rows), "guesses": sends, "right": right,
            "turns_guessed": sum(1 for r in rows if r["guess"]),
            "guess_round_trip_ms": {"n": len(rts), "p50": q(0.5), "p90": q(0.9), "p99": q(0.99)}, "seconds": round(time.time() - t0, 1),
            "passes": ok, "why": why, "rows": rows}
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(res_), encoding="utf-8")
    hv = "" if a.v == 2 else f" (hello version {a.v})"
    print(f"the API's control on {a.agent}{hv}: {len(calls)} calls, {len(rows)} turns, {sends} guesses, {right} right; guess round trip "
          f"median {q(0.5)} ms (90th {q(0.9)}, 99th {q(0.99)}); {res_['seconds']} s" + ("" if ok else "; " + "; ".join(why)))
    print("API_CONTROL_PASSES" if ok else "API_CONTROL_FAILS")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
