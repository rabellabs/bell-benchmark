#!/usr/bin/env python3
"""Does the same arm, run twice, agree? Two host passes read turn by turn on the turns both read:
  - ms sooner a turn (the head, capped at 1,000 ms, 0 when no early reply was kept): the paired mean difference with its 95% interval
    by whole calls (bench/read.py's 2,000 draws, seed 20261117), and how many turns differ at all (by more than 0.5 ms);
  - LiveKit's commit: the first commit in the turn's window after the vendor's end (`commit_after_vendor_end_ms`), the same;
  - LLM calls a turn, the same; and which reply was kept (turns where it differs).
With --quiet PASS: also on the turns where Bell sent nothing in PASS (its `bell_guesses` is 0), where the two arms should agree.

  python bench/repeat.py --a A.jsonl --b B.jsonl [--quiet B.jsonl] [--label TEXT] [--json OUT.json]
"""
import argparse
import json
import pathlib
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import read as R  # noqa: E402

TOL = 0.5


def load(p):
    return {r["key"]: r for r in (json.loads(ln) for ln in open(p, encoding="utf-8") if ln.strip())}


def compare(A, B, keys):
    import numpy as np
    convs = sorted({A[k]["conv"] for k in keys})
    ci = {c: i for i, c in enumerate(convs)}
    W = R.resample_weights(len(convs)) if convs else None
    out = {"turns": len(keys), "calls": len(convs)}
    fields = {
        "ms_sooner": lambda r: R.saving(r.get("head"), R.CAPS[0]),
        "commit_ms": lambda r: r.get("commit_after_vendor_end_ms"),
        "llm_calls": lambda r: r.get("calls"),
    }
    for name, f in fields.items():
        S = np.zeros((len(convs), 2))
        d_all, missing = [], 0
        for k in keys:
            x, y = f(A[k]), f(B[k])
            if x is None or y is None:
                if (x is None) != (y is None):
                    missing += 1
                continue
            S[ci[A[k]["conv"]]] += (1.0, y - x)
            d_all.append(y - x)
        n = S[:, 0].sum()
        if n == 0:
            out[name] = {"turns": 0, "one_side_missing": missing}
            continue
        WS = W @ S
        ad = sorted(abs(v) for v in d_all)
        out[name] = {"turns": int(n), "mean_diff": float(S[:, 1].sum() / n),
                     "mean_diff_interval": [float(v) for v in np.percentile(WS[:, 1] / np.maximum(WS[:, 0], 1e-9), [2.5, 97.5])],
                     "turns_differing": sum(1 for v in d_all if abs(v) > TOL), "abs_median": ad[len(ad) // 2],
                     "abs_p90": ad[int(len(ad) * 0.9)], "abs_max": ad[-1], "one_side_missing": missing}
    out["kept_differs"] = sum(1 for k in keys if A[k].get("kept") != B[k].get("kept"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--quiet", default="", help="a Bell pass: also read on the turns where Bell sent nothing in it")
    ap.add_argument("--label", default="")
    ap.add_argument("--json")
    a = ap.parse_args()
    A, B = load(a.a), load(a.b)
    both = sorted(k for k in set(A) & set(B))
    faulted = [k for k in both if "fault" in A[k] or "fault" in B[k]]
    keys = [k for k in both if k not in set(faulted)]
    res = {"label": a.label, "a": a.a, "b": a.b, "turns_in_both": len(both), "turns_faulted": len(faulted),
           "turns_in_one_only": len(set(A) ^ set(B)), "all": compare(A, B, keys)}
    if a.quiet:
        Q = load(a.quiet)
        qk = [k for k in keys if k in Q and "fault" not in Q[k] and Q[k].get("bell_guesses", 0) == 0]
        res["bell_sent_nothing"] = compare(A, B, qk)
        # under LiveKit's own ending, LiveKit's turn can stay open into the vendor's next turn (the VAD heard speech again before
        # LiveKit committed), and a guess Bell sends there replaces the early reply made for this turn: this turn is untouched by
        # Bell only when Bell sent nothing in it AND in the call's next turn
        nxt = lambda k: k.rsplit("#", 1)[0] + "#" + str(int(k.rsplit("#", 1)[1]) + 1)  # noqa: E731
        qk2 = [k for k in qk if Q.get(nxt(k), {}).get("bell_guesses", 0) == 0]
        res["bell_sent_nothing_here_or_next"] = compare(A, B, qk2)
    for part in ("all", "bell_sent_nothing", "bell_sent_nothing_here_or_next"):
        if part not in res:
            continue
        r = res[part]
        line = [f"{a.label} [{part}] {r['turns']} turns of {r['calls']} calls; kept differs on {r['kept_differs']}"]
        for name in ("ms_sooner", "commit_ms", "llm_calls"):
            f = r[name]
            if not f.get("turns"):
                line.append(f"  {name}: no turns")
                continue
            line.append(f"  {name}: mean diff {f['mean_diff']:+.3f} [{f['mean_diff_interval'][0]:+.3f}, {f['mean_diff_interval'][1]:+.3f}]"
                        f"; {f['turns_differing']} of {f['turns']} turns differ; |diff| median {f['abs_median']:.3f}, "
                        f"p90 {f['abs_p90']:.3f}, max {f['abs_max']:.3f}; one side missing {f['one_side_missing']}")
        print("\n".join(line))
    print(f"turns faulted {len(faulted)}, in one pass only {res['turns_in_one_only']}")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
