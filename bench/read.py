#!/usr/bin/env python3
"""The benchmark's read of a host pass: its fields and their intervals by whole calls, beside the same turns with nothing
switched on (one LLM call and no early start on every turn).

  ms sooner a turn   the kept early reply's head start before the end of turn, capped at 1,000 ms (and 500), 0 on every other
                     turn, the mean over all turns
  LLM calls a turn   every LLM start the host made in the turn's window (early starts, plus the one at the end of the turn when no
                     early reply was kept)
  last word to first audio   end to end with a real LLM and voice: read by e2e/combine.py, empty here

Intervals: 95%, 2,000 resamples of whole calls with numpy default_rng(20261117), over the calls sorted by id, so every read of
the same schedule uses the same draws (bench/repeat.py and bench/e2e/combine.py use these draws too).

  python bench/read.py --sched SCHED.json --host HOST.jsonl [--partial] [--json OUT.json]
"""
import argparse
import json
import pathlib
import sys

sys.dont_write_bytecode = True

CAPS = (1000.0, 500.0)
SEED = 20261117
B = 2000


def resample_weights(n_conv, B_=B, seed=SEED):
    import numpy as np
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_conv, size=(B_, n_conv))
    W = np.zeros((B_, n_conv))
    np.add.at(W, (np.repeat(np.arange(B_), n_conv), idx.ravel()), 1.0)
    return W


def saving(head, cap):
    return 0.0 if head is None else min(head, cap)


def fields(rows, convs, W):
    """rows: [{"conv", "calls", "head"}] -> the fields with their intervals by whole calls"""
    import numpy as np
    ci = {c: i for i, c in enumerate(convs)}
    S = np.zeros((len(convs), 4))
    for r in rows:
        S[ci[r["conv"]]] += (1.0, r["calls"], saving(r["head"], CAPS[0]), saving(r["head"], CAPS[1]))
    n = S[:, 0].sum()
    WS = W @ S
    out = {"turns": int(n), "calls": len(convs)}
    for j, name in ((1, "llm_calls_a_turn"), (2, "ms_sooner_a_turn_cap1000"), (3, "ms_sooner_a_turn_cap500")):
        v = WS[:, j] / WS[:, 0]
        out[name] = float(S[:, j].sum() / n)
        out[name + "_interval"] = [float(x) for x in np.percentile(v, [2.5, 97.5])]
    out["kept"] = sum(1 for r in rows if r["head"] is not None)
    return out


def read(sched_path, host_path, partial=False):
    sched = json.loads(pathlib.Path(sched_path).read_text(encoding="utf-8"))
    host = {}
    for ln in open(host_path, encoding="utf-8"):
        if ln.strip():
            r = json.loads(ln)
            host[r["key"]] = r
    turns = sched["turns"]
    if partial:  # a trial on some calls: read only the calls the host pass holds
        have = {r["conv"] for r in host.values()}
        turns = [t for t in turns if t["conv"] in have]
    convs = sorted({t["conv"] for t in turns})
    W = resample_weights(len(convs))
    hrows = [host[t["key"]] for t in turns if t["key"] in host]
    if len(hrows) != len(turns):
        raise SystemExit(f"READ_REFUSED: the host pass has {len(hrows)} of the schedule's {len(turns)} turns")
    return {
        "label": sched["label"],
        "host": fields(hrows, convs, W),
        "nothing": fields([{"conv": t["conv"], "calls": 1, "head": None} for t in turns], convs, W),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sched", required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--json")
    ap.add_argument("--partial", action="store_true", help="a trial: read only the calls the host pass holds")
    a = ap.parse_args()
    out = read(a.sched, a.host, a.partial)
    h = out["host"]
    print(f"schedule {out['label']}: {h['turns']} turns of {h['calls']} calls; {h['llm_calls_a_turn']:.4f} LLM calls a turn "
          f"[{h['llm_calls_a_turn_interval'][0]:.4f}, {h['llm_calls_a_turn_interval'][1]:.4f}], "
          f"{h['ms_sooner_a_turn_cap1000']:.2f} ms sooner a turn [{h['ms_sooner_a_turn_cap1000_interval'][0]:.2f}, "
          f"{h['ms_sooner_a_turn_cap1000_interval'][1]:.2f}], {h['kept']} early replies kept")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
