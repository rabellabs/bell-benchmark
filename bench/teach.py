#!/usr/bin/env python3
"""The curve's agents, taught through the Rabel API: for each N, a new agent of this account (`<prefix>-c<N>`) taught the first N of
the 27 known callers' 1,070 calls (data/teach_27.json, seed 23's order) through `POST /v1/agents/{agent}/calls`, in requests of at
most 200 calls; then its status read back (`GET /v1/agents/{agent}`: calls_heard must be N). N = 0 is the first call: an agent that
was never taught. Nothing of the 28 test callers is taught (the data holds the 27's calls only).

An agent that already exists with calls taught is refused (a run is never taught twice into the same agent): give each run its own
prefix.

  python bench/teach.py --teach data/teach_27.json --prefix bench-1700000000 --ns 0,12,24,48,100,300,1070 --out out/teach.json
Prints TEACH_OK (and the agents' names) or TEACH_FAILED.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rabel_api as RA  # noqa: E402

PER_REQUEST = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teach", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--ns", default="0,12,24,48,100,300,1070")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not re.fullmatch(r"[a-z0-9_-]{1,50}", a.prefix):
        raise SystemExit("TEACH_FAILED: the prefix is 1 to 50 of a-z, 0-9, - and _")
    calls = json.load(open(a.teach, encoding="utf-8"))["calls"]
    res, ok = {"url": RA.base_url(), "prefix": a.prefix, "agents": {}}, True
    t0 = time.time()
    for n in [int(x) for x in a.ns.split(",")]:
        agent = f"{a.prefix}-c{n}"
        st, cur = RA.http("GET", f"/v1/agents/{agent}")
        before = (cur or {}).get("calls_heard", 0) if st == 200 else 0
        if before:
            print(f"{agent}: already taught {before} calls; refused (give this run its own --prefix)")
            ok = False
            break
        learned, t1 = 0, time.time()
        for i in range(0, n, PER_REQUEST):
            part = [{"turns": [{"heard": tn["heard"], "final": tn["final"]} for tn in c["turns"]]} for c in calls[i: min(i + PER_REQUEST, n)]]
            st, ans = RA.http("POST", f"/v1/agents/{agent}/calls", {"calls": part})
            if st != 200 or not ans or ans.get("calls_learned") != len(part):
                print(f"{agent}: a request of {len(part)} calls answered {st} {str(ans)[:200]}")
                ok = False
                break
            learned += ans["calls_learned"]
        st, cur = RA.http("GET", f"/v1/agents/{agent}") if n else (200, {"calls_heard": 0})
        heard = (cur or {}).get("calls_heard")
        good = ok and learned == n and heard == n
        ok &= good
        res["agents"][str(n)] = {"agent": agent, "calls_taught": learned, "calls_heard_read_back": heard, "seconds": round(time.time() - t1, 1),
                                 "ok": good}
        print(f"{agent}: taught {learned} of {n} calls, the agent reads {heard} calls heard; {time.time() - t1:.1f} s"
              + ("" if good else " FAILED"))
        if not good:
            break
    res["seconds"] = round(time.time() - t0, 1)
    res["ok"] = ok
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(("TEACH_OK " if ok else "TEACH_FAILED ") + " ".join(f"{k}={v['agent']}" for k, v in res["agents"].items()))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
