#!/usr/bin/env python3
"""The read of the lead table: LiveKit's default ending alone (`lk`) against with Bell (`lk_bell`), each arm's passes averaged
by turn (each arm run twice), with the end to end time put together from the replay clock and the live
measurement (`live.py`).

  python bench/e2e/combine.py texts --pass lk=A1.jsonl,A2.jsonl --pass lk_bell=B1.jsonl,B2.jsonl --out TEXTS.json
      the distinct words every arm's answering reply was made on (what live.py measures)
  python bench/e2e/combine.py read --pass lk=A1.jsonl,A2.jsonl --pass lk_bell=B1.jsonl,B2.jsonl [--live LIVE.jsonl] --label TEXT
      --json OUT.json

On the turns every pass of every arm read (a turn any pass failed to read leaves every arm, counted):
  - ms sooner a turn (the kept early reply's head start before LiveKit's commit, capped at 1,000 ms; 0 on every other turn) and LLM
    calls a turn, each the mean of the arm's passes for the turn; the mean over turns with its 95% interval by whole calls (bench/read.py's
    2,000 draws, seed 20261117); what Bell adds, paired by turn.
  - How LiveKit's default behaves on these calls (alone): how long it waits after the last word (its end-of-turn metrics'
    `end_of_turn_delay`: median, 90th, and the share near its 0.3 s and its 2.5 s), the head start its preemptive generation gives
    (on the turns where LiveKit's own early start was kept), LLM calls a turn.
  - Last word to first audio (with --live): per pass, turn and live repetition, first audio = max(LiveKit's commit, the reply's start +
    L), L the live time from the LLM request to Aura-2's first audio for the words the reply was made on (both arms read the same
    measurements for the same words), less the end of the caller's last word: by default the speech-to-text's own last word time
    (`vendor_last_word_ms`: the same for every arm and both ways of ending the turn; --last-word livekit takes LiveKit's own last speaking
    time instead, which LiveKit has only with its VAD, i.e. under its default ending); the mean of these for the turn; then the median and the 95th
    over turns, with intervals by whole calls, and paired by turn. A turn's answering reply is the first LiveKit scheduled after its last
    word and before the next turn's (the host's `answer`); turns with none of their own (LiveKit's turn ran on into the next one,
    whose reply is counted there) or no last word time are counted and left out of this field only.
  - Pipecat's rows (`answer.audio_rule` "pipecat_gate"): Pipecat's speculation gate holds an early reply's output until the
    turn's end and only then hands it to the voice, so first audio = max(commit, start + S) + (L - S), S the live time from the LLM
    request to the first sentence complete; an ordinary reply (start = commit) reads commit + L as on LiveKit.
  - `--convs FILE` (read and texts): only these calls' turns (the pace or rate fallback; the live pass's cross-check on its calls).
  - `--e2e-convs FILE` (read): last word to first audio only on these calls' turns (after the live part's fallback to the 10 callers
    drawn by seed 23: the other fields stay on every call).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import read as R  # noqa: E402


def tid(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def load_passes(specs):
    arms = {}
    for sp in specs:
        arm, paths = sp.split("=", 1)
        arms[arm] = [{r["key"]: r for r in (json.loads(ln) for ln in open(p, encoding="utf-8") if ln.strip())}
                     for p in paths.split(",")]
    return arms


def common_keys(arms):
    allp = [p for ps in arms.values() for p in ps]
    keys = set.intersection(*(set(p) for p in allp))
    faults = collections.Counter()
    good = []
    for k in sorted(keys):
        bad = [p[k]["fault"] for p in allp if "fault" in p[k]]
        if bad:
            faults.update(set(bad))     # a turn counted once, by each kind of fault it met
        else:
            good.append(k)
    return good, dict(faults), len(set.union(*(set(p) for p in allp)) - keys), len(keys) - len(good)


def wq(vals, wts, q):
    """weighted quantile (vals sorted ascending)"""
    import numpy as np
    c = np.cumsum(wts)
    if c[-1] <= 0:
        return float("nan")
    return float(vals[min(len(vals) - 1, int(np.searchsorted(c, q * c[-1])))])


def dist(vals_by_turn, conv_of, convs, W, qs=(0.5, 0.95)):
    """median and 95th over turns, with intervals by whole calls"""
    import numpy as np
    keys = sorted(vals_by_turn, key=lambda k: vals_by_turn[k])
    v = np.array([vals_by_turn[k] for k in keys])
    ci = {c: i for i, c in enumerate(convs)}
    idx = np.array([ci[conv_of[k]] for k in keys])
    out = {"turns": len(keys)}
    for q in qs:
        point = wq(v, np.ones(len(v)), q)
        draws = [wq(v, W[d, idx], q) for d in range(W.shape[0])]
        out[f"q{int(q * 100)}"] = point
        out[f"q{int(q * 100)}_interval"] = [float(x) for x in np.percentile(draws, [2.5, 97.5])]
    out["mean"] = float(v.mean()) if len(v) else None
    return out


def mean_iv(vals_by_turn, conv_of, convs, W):
    import numpy as np
    ci = {c: i for i, c in enumerate(convs)}
    S = np.zeros((len(convs), 2))
    for k, x in vals_by_turn.items():
        S[ci[conv_of[k]]] += (1.0, x)
    n = S[:, 0].sum()
    if not n:
        return {"turns": 0}
    WS = W @ S
    return {"turns": int(n), "mean": float(S[:, 1].sum() / n),
            "interval": [float(x) for x in np.percentile(WS[:, 1] / np.maximum(WS[:, 0], 1e-9), [2.5, 97.5])]}


def cmd_texts(a):
    arms = load_passes(a.passes)
    keep = None
    if a.convs:   # only these calls' turns (the pace fallback: the 10 callers drawn by seed 23)
        keep = {ln.strip() for ln in open(a.convs, encoding="utf-8") if ln.strip()}
    seen = {}
    for ps in arms.values():
        for p in ps:
            for r in p.values():
                if keep is not None and r.get("conv") not in keep:
                    continue
                w = (r.get("answer") or {}).get("words")
                if w:
                    seen[tid(w)] = w
    out = [{"id": k, "text": v} for k, v in sorted(seen.items())]
    pathlib.Path(a.out).write_text(json.dumps(out), encoding="utf-8")
    print(f"TEXTS {len(out)} distinct reply words from {sum(len(p) for ps in arms.values() for p in ps)} turn rows")


def first_audio_of(an, m):
    """the first audio's time (ms from the vendor's end of turn) for this reply and one live measurement, by the host's rule"""
    if an.get("audio_rule") == "pipecat_gate":
        s = m.get("sentence")
        if s is None:
            return None
        return max(an["commit_ms"], an["start_ms"] + s) + (m["first_audio"] - s)
    return max(an["commit_ms"], an["start_ms"] + m["first_audio"])


def cmd_read(a):
    import statistics
    arms = load_passes(a.passes)
    if a.convs:
        keep = {ln.strip() for ln in open(a.convs, encoding="utf-8") if ln.strip()}
        arms = {arm: [{k: r for k, r in p.items() if r.get("conv") in keep} for p in ps] for arm, ps in arms.items()}
        if not any(p for ps in arms.values() for p in ps):
            raise SystemExit("COMBINE_EMPTY: no turn of these calls")
    keys, faults, not_all, n_faulted = common_keys(arms)
    first = next(iter(arms.values()))[0]
    conv_of = {k: first[k]["conv"] for k in keys}
    convs = sorted(set(conv_of.values()))
    W = R.resample_weights(len(convs))
    res = {"label": a.label, "turns_read": len(keys), "calls": len(convs), "turns_faulted": n_faulted, "turn_faults": faults,
           "turns_not_in_every_pass": not_all,
           "passes": {arm: len(ps) for arm, ps in arms.items()}, "arms": {}, "paired": {}}
    per = {}
    for arm, ps in arms.items():
        ms = {k: sum(R.saving(p[k].get("head"), R.CAPS[0]) for p in ps) / len(ps) for k in keys}
        calls = {k: sum(p[k]["calls"] for p in ps) / len(ps) for k in keys}
        per[arm] = {"ms": ms, "calls": calls}
        kept = collections.Counter()
        for p in ps:
            kept.update(str(p[k].get("kept")) for k in keys)
        f = {"ms_sooner_a_turn": mean_iv(ms, conv_of, convs, W), "llm_calls_a_turn": mean_iv(calls, conv_of, convs, W),
             "kept_by_over_passes": dict(kept),
             "kept_bell_no_restart_per_pass": [sum(1 for k in keys if p[k].get("kept_bell_no_restart")) for p in ps],
             "bell_guesses_per_pass": [sum(p[k].get("bell_guesses", 0) for k in keys) for p in ps]}
        # how LiveKit's default behaves: its wait after the last word, its own early start's head
        eou = [p[k]["answer"]["eou_delay_ms"] for p in ps for k in keys
               if p[k].get("answer") and p[k]["answer"].get("eou_delay_ms") is not None]
        if eou:
            eou.sort()
            f["livekit_wait_after_last_word_ms"] = {"n": len(eou), "median": eou[len(eou) // 2], "p90": eou[int(len(eou) * 0.9)],
                                                    "share_under_500": sum(1 for x in eou if x < 500) / len(eou),
                                                    "share_over_2000": sum(1 for x in eou if x > 2000) / len(eou)}
        vw = sorted(p[k]["answer"]["commit_ms"] - p[k]["vendor_last_word_ms"] for p in ps for k in keys
                    if p[k].get("answer") and p[k].get("vendor_last_word_ms") is not None)
        if vw:
            f["commit_after_vendor_last_word_ms"] = {"n": len(vw), "median": vw[len(vw) // 2], "p90": vw[int(len(vw) * 0.9)]}
        lkh = [p[k]["head"] for p in ps for k in keys if p[k].get("kept") == "livekit" and p[k].get("head") is not None]
        if lkh:
            f["livekit_own_early_start_head_ms"] = {"kept_turns_over_passes": len(lkh), "median": statistics.median(lkh),
                                                    "mean": sum(lkh) / len(lkh)}
        res["arms"][arm] = f
    names = list(arms)
    if len(names) == 2:
        A, B = names
        res["paired"]["ms_sooner_a_turn"] = mean_iv({k: per[B]["ms"][k] - per[A]["ms"][k] for k in keys}, conv_of, convs, W)
        res["paired"]["llm_calls_a_turn"] = mean_iv({k: per[B]["calls"][k] - per[A]["calls"][k] for k in keys}, conv_of, convs, W)
    if a.live:
        L = collections.defaultdict(dict)
        nlive = collections.Counter()
        for ln in open(a.live, encoding="utf-8"):
            if ln.strip():
                x = json.loads(ln)
                nlive["ok" if x.get("ok") else "failed"] += 1
                if x.get("ok"):
                    L[x["id"]][x["rep"]] = x
        res["live_rows"] = dict(nlive)
        e2e, left = {}, collections.Counter()
        for arm, ps in arms.items():
            v = {}
            for k in keys:
                xs = []
                for p in ps:
                    an = p[k].get("answer")
                    lw = an.get("last_word_ms") if a.last_word == "livekit" else p[k].get("vendor_last_word_ms") if an else None
                    if not an or lw is None or not an.get("words"):
                        left["no answer or last word"] += 1
                        continue
                    reps = L.get(tid(an["words"]))
                    if not reps:
                        left["no live measurement"] += 1
                        continue
                    for rep, m in reps.items():
                        fa = first_audio_of(an, m)
                        if fa is not None:
                            xs.append(fa - lw)
                if xs:
                    v[k] = sum(xs) / len(xs)
            e2e[arm] = v
        both = set.intersection(*(set(v) for v in e2e.values()))
        if a.e2e_convs:
            ek = {ln.strip() for ln in open(a.e2e_convs, encoding="utf-8") if ln.strip()}
            both = {k for k in both if conv_of[k] in ek}
            res["e2e_only_calls_of"] = a.e2e_convs
        res["e2e_turns_in_every_arm"] = len(both)
        res["e2e_left_out"] = dict(left)
        econvs = sorted({conv_of[k] for k in both})
        EW = R.resample_weights(max(1, len(econvs)))
        for arm in (arms if both else []):
            res["arms"][arm]["last_word_to_first_audio_ms"] = dist({k: e2e[arm][k] for k in both}, conv_of, econvs, EW)
        if len(names) == 2 and both:
            A, B = names
            res["paired"]["last_word_to_first_audio_ms"] = mean_iv({k: e2e[B][k] - e2e[A][k] for k in both}, conv_of, econvs, EW)
        lv = [m for reps in L.values() for m in reps.values()]
        if lv:
            for fld in ("ttft", "sentence", "first_audio", "socket_open"):
                xs = sorted(m[fld] for m in lv if m.get(fld) is not None)
                if xs:
                    res.setdefault("live_ms", {})[fld] = {"n": len(xs), "median": xs[len(xs) // 2], "p95": xs[int(len(xs) * 0.95)]}
    pathlib.Path(a.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
    for arm, f in res["arms"].items():
        ln = (f"{a.label} {arm}: {f['llm_calls_a_turn']['mean']:.4f} calls a turn, {f['ms_sooner_a_turn']['mean']:.2f} ms sooner a turn "
              f"[{f['ms_sooner_a_turn']['interval'][0]:.2f}, {f['ms_sooner_a_turn']['interval'][1]:.2f}]")
        if "last_word_to_first_audio_ms" in f:
            e = f["last_word_to_first_audio_ms"]
            ln += f"; last word to first audio median {e['q50']:.0f} ms [{e['q50_interval'][0]:.0f}, {e['q50_interval'][1]:.0f}], 95th {e['q95']:.0f}"
        print(ln)
    for name, f in res["paired"].items():
        if f.get("turns"):
            print(f"{a.label} paired {name}: {f['mean']:+.3f} [{f['interval'][0]:+.3f}, {f['interval'][1]:+.3f}] on {f['turns']} turns")
    print(f"turns read {len(keys)} of {len(keys) + n_faulted}; faults {faults}; not in every pass {not_all}")
    print("COMBINE_DONE")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("texts")
    t.add_argument("--pass", dest="passes", action="append", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--convs", default="", help="a file of conversation ids: only their turns")
    r = sub.add_parser("read")
    r.add_argument("--pass", dest="passes", action="append", required=True)
    r.add_argument("--live", default="")
    r.add_argument("--last-word", choices=("vendor", "livekit"), default="vendor")
    r.add_argument("--label", default="")
    r.add_argument("--json", required=True)
    r.add_argument("--convs", default="", help="a file of conversation ids: only their turns")
    r.add_argument("--e2e-convs", default="", help="a file of conversation ids: last word to first audio on their turns only")
    a = ap.parse_args()
    cmd_texts(a) if a.cmd == "texts" else cmd_read(a)


if __name__ == "__main__":
    main()
