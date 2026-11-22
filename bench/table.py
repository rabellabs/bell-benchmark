#!/usr/bin/env python3
"""The benchmark's table from the chain's reads: out/TABLE.md (LiveKit's default ending alone against with Bell on Flux's words,
the curve and the first call, Bell alone against nothing under Flux's own end of turn, and on Pipecat; with the checks under
it) and out/TABLE.json (every number, flat, for comparing two runs: bench/compare.py).

  python bench/table.py --out OUT_DIR --stamps STAMPS_DIR --curve-ns 0,12,24,48,100,300,1070 --calls 0 --passes 2 [--pipecat 1]
"""
from __future__ import annotations

import argparse
import json
import pathlib

STTS = ("flux",)   # the speech-to-text whose recorded words are replayed: Deepgram Flux


def rd(p):
    p = pathlib.Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def txt(p):
    p = pathlib.Path(p)
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


def f2(x, nd=2, sign=False):
    if x is None:
        return "–"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def iv(d, nd=2, sign=False):
    return f"{f2(d['mean'], nd, sign)} [{f2(d['interval'][0], nd, sign)}, {f2(d['interval'][1], nd, sign)}]" if d and d.get("turns") else "–"


def e2e(d):
    if not d or not d.get("turns"):
        return "–", "–"
    return (f"{d['q50']:.0f} [{d['q50_interval'][0]:.0f}, {d['q50_interval'][1]:.0f}]",
            f"{d['q95']:.0f} [{d['q95_interval'][0]:.0f}, {d['q95_interval'][1]:.0f}]")


def live_summary(out, mock):
    """when the live part ran and how fast the reply model answered: the UTC hours of the rows used and the median time to the first
    text (ttft), so a reader can tell an hour when the model was slow from a change in what is measured"""
    p = pathlib.Path(out) / "live.jsonl"
    rows = []
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                x = json.loads(ln)
                if x.get("ok") and x.get("ttft") is not None:
                    rows.append(x)
    tt = sorted(x["ttft"] for x in rows)
    hours = sorted({x["utc"][:13] for x in rows if x.get("utc")})
    return {"stand_in": mock, "rows": len(rows), "ttft_median_ms": round(tt[len(tt) // 2], 1) if tt else None,
            "utc_hours": hours, "rows_without_hour": sum(1 for x in rows if not x.get("utc"))}


def api_summary(out):
    """the API's guess round trips over every LiveKit Bell arm's meta (each process's samples were summarised there)"""
    rts, url = [], None
    for p in sorted(pathlib.Path(out).glob("*.jsonl.meta.json")):
        m = rd(p) or {}
        if isinstance(m.get("api"), dict):      # LiveKit's arms (Pipecat's carry a list of per-process counts)
            url = url or m["api"].get("url")
            rts.extend(m["api"].get("rt_ms_all", []))
    rts.sort()
    q = lambda f: round(rts[min(len(rts) - 1, int(f * len(rts)))], 1) if rts else None  # noqa: E731
    return {"url": url or "?", "n": len(rts), "p50": q(0.5), "p90": q(0.9), "p99": q(0.99)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stamps", required=True)
    ap.add_argument("--curve-ns", default="0,12,24,48,100,300,1070")
    ap.add_argument("--flux-corners", default="1")
    ap.add_argument("--calls", default="0")
    ap.add_argument("--passes", default="2")
    ap.add_argument("--pipecat", default="1")
    ap.add_argument("--live-pass", default="1")
    a = ap.parse_args()
    pipecat_on = a.pipecat == "1"
    out, st = pathlib.Path(a.out), pathlib.Path(a.stamps)
    R = out / "reads"
    live_run = rd(out / "live.jsonl.run.json") or {}
    mock = bool(live_run.get("mock"))
    flat = {}
    rate_info = {}
    L = []
    w = L.append
    sizes = {}
    w("# What a builder gets by switching Bell on: the benchmark's table")
    w("")
    if mock:
        w("> **DRY RUN.** The reply and the voice are the stand-in (`live.py --mock`): no Claude, no Aura-2. "
          "Every \"last word to first audio\" here is made up by the stand-in and is NOT a reading. "
          + (f"Only the {a.calls} shortest calls were replayed." if a.calls != "0" else ""))
        w("")
    api = api_summary(out)
    w(f"- Configuration: LiveKit Agents 1.8.13 at its defaults; Pipecat 1.16.0; Bell's guesses from the Rabel API at {api['url']} "
      f"(the ready-made agent `demo-bank`, and for the curve new agents of this run's account taught the 27 known callers' calls through "
      f"`POST /v1/agents/{{agent}}/calls`), every conversation with \"learn\": false; the hookups `livekit-plugins-rabel` "
      f"{(txt(out.parent / 'lkr' / 'HOOKUP_COMMIT') or '?').split(' ')[0]} and `pipecat-rabel` "
      f"{(txt(out.parent / 'pcr' / 'HOOKUP_COMMIT') or '?').split(' ')[0]}; nothing taught from the 28 test callers, nothing tuned on them.")
    inst = [ln for nm in ("freeze.txt", "freeze_pc.txt") for ln in txt(out.parent / "logs" / nm).splitlines() if ln.strip()]
    if inst:
        w(f"- Installed: {', '.join(sorted(set(inst)))}.")
    w(f"- Bell's answer time is the API's real round trip from this machine: each guess reaches the hookup as long after the words it "
      f"was made from as the API took to answer it through the network (LiveKit's arms, on the replay clock: median {api['p50']} ms, "
      f"90th {api['p90']}, 99th {api['p99']}, over {api['n']:,} guesses; Pipecat's arms run on the wall clock against the API as a call would).")
    w("- Last word to first audio runs from the end of the caller's last word by the speech-to-text's own word times (Flux's words' "
      "`end`), the same for every arm and both ways of ending the turn.")
    w(f"- Each LiveKit arm replayed {a.passes} times on the replay clock and averaged by turn; intervals 95% by whole calls (2,000 draws, "
      "seed 20261117). The reply: Claude Haiku 5.5, one prompt, cut after its first sentence; the voice: Aura-2 (LiveKit's plugin's "
      "defaults), first audio only. LiveKit's commit and the reply's start come from the replay clock; the time from the reply's "
      f"LLM call to its first audio is measured live ({live_run.get('reps', '?')} times for each reply's words).")
    if mock:
        w(f"- The live part is the stand-in: {live_run.get('ok', '?')} stand-in replies, {live_run.get('errors', '?')} failed; no request "
          f"went to Claude or Aura-2 and nothing was spent; {txt(st / 'LIVE')}")
    else:
        w(f"- The live part: {live_run.get('ok', '?')} measured, {live_run.get('errors', '?')} failed, stop: "
          f"{live_run.get('stop') or 'none'}, spend counted ${live_run.get('usd_counted', '?')}, {live_run.get('conc', '?')} at once; "
          f"{txt(st / 'LIVE')}")
    for nm in ("live_first.run.json", "live.jsonl.run.json"):
        lr_ = rd(out / nm)
        if lr_ and lr_.get("rate_limits") is not None and not lr_.get("mock"):
            rl = lr_["rate_limits"]
            w(f"  - {'first run' if nm.startswith('live_first') else 'last run'}: Anthropic answered 429 {rl.get('429')} times and 529 "
              f"{rl.get('529')} times in {rl.get('claude_requests')} Claude requests (retried {rl.get('retried')}, waited "
              f"{rl.get('waited_s')} s, most in any 200: {rl.get('max_in_window')}); stop: {lr_.get('stop') or 'none'}"
              + (" (planted on the stand-in: a check, not Anthropic)" if lr_.get("mock_rate") else ""))
            rate_info[nm] = {"429": rl.get("429"), "529": rl.get("529"), "requests": rl.get("claude_requests"), "stop": lr_.get("stop")}
    w("")
    w("## The results")
    w("")
    w("### The lead table: LiveKit's default ending alone against with Bell, eager off")
    w("")
    w("| words | | LLM calls a turn | ms sooner a turn (cap 1,000) | last word to first audio, median (ms) | 95th (ms) |")
    w("|---|---|---|---|---|---|")
    lines = []
    for v in STTS:
        d = rd(R / f"lead_{v}.json")
        if not d:
            w(f"| {v} | not read | | | | |")
            continue
        sizes[v] = (d["calls"], d["turns_read"])
        for arm, name in (("lk", "LiveKit's default alone"), ("lk_bell", "with Bell")):
            f = d["arms"][arm]
            m, p95 = e2e(f.get("last_word_to_first_audio_ms"))
            w(f"| {v} | {name} | {iv(f['llm_calls_a_turn'], 4)} | {iv(f['ms_sooner_a_turn'])} | {m} | {p95} |")
            flat[f"lead.{v}.{arm}.calls"] = f["llm_calls_a_turn"]
            flat[f"lead.{v}.{arm}.ms"] = f["ms_sooner_a_turn"]
            if f.get("last_word_to_first_audio_ms"):
                flat[f"lead.{v}.{arm}.e2e"] = f["last_word_to_first_audio_ms"]
        pd = d["paired"]
        w(f"| {v} | **Bell adds** | {iv(pd['llm_calls_a_turn'], 4, True)} | **{iv(pd['ms_sooner_a_turn'], 2, True)}** | "
          f"{iv(pd.get('last_word_to_first_audio_ms'), 1, True)} (with Bell less alone, paired mean; below 0 = first audio sooner) | |")
        flat[f"lead.{v}.adds.calls"] = pd["llm_calls_a_turn"]
        flat[f"lead.{v}.adds.ms"] = pd["ms_sooner_a_turn"]
        if pd.get("last_word_to_first_audio_ms"):
            flat[f"lead.{v}.adds.e2e"] = pd["last_word_to_first_audio_ms"]
        taught = "demo-bank, taught from Flux's words of the 27"
        lines.append(f"- **{v}, tuned ({taught}), on the bank's 28 test callers ({d['calls']} calls, {d['turns_read']} turns), "
                     f"LiveKit's default ending, eager off:** Bell adds {iv(pd['ms_sooner_a_turn'], 0, True)} ms a turn at "
                     f"{iv(pd['llm_calls_a_turn'], 3, True)} LLM calls a turn"
                     + (f"; last word to first audio changes by {iv(pd['last_word_to_first_audio_ms'], 0, True)} ms a turn (paired mean; below 0 = sooner)"
                        if pd.get("last_word_to_first_audio_ms") else "") + (" (DRY RUN, not a reading)" if mock else "") + ".")
    for v in STTS:
        d = rd(R / f"lead_{v}.json")
        if d and "e2e_turns_in_every_arm" in d:
            w(f"- {v}: last word to first audio read on {d['e2e_turns_in_every_arm']} of {d['turns_read']} turns (in every arm); left out, "
              f"over arms and passes: {d.get('e2e_left_out')}; live rows {d.get('live_rows')}.")
    ls = live_summary(out, mock)
    if ls["stand_in"]:
        w("- Last word to first audio above is the stand-in's (DRY RUN): no reply model answered.")
    else:
        w(f"- Last word to first audio depends on the hour it was measured: the live part ran in the UTC hours "
          f"{', '.join(h.replace('T', ' ') + ':00' for h in ls['utc_hours']) or 'not recorded'}"
          + (f" (rows given from earlier runs, which carry no hour: {ls['rows_without_hour']})" if ls["rows_without_hour"] else "")
          + f", and the reply model's median time to its first text was {ls['ttft_median_ms']} ms over {ls['rows']} measurements. "
          "Read it as one reading of a range, not as a fixed number; the paired change Bell makes moves far less from hour to hour.")
    w("")
    w("One line each:")
    L.extend(lines)
    w("")
    w("### How LiveKit's default ending behaves on these calls (LiveKit alone)")
    w("")
    for v in STTS:
        d = rd(R / f"lead_{v}.json")
        if not d:
            continue
        f = d["arms"]["lk"]
        wt = f.get("livekit_wait_after_last_word_ms") or {}
        hd = f.get("livekit_own_early_start_head_ms") or {}
        vw = f.get("commit_after_vendor_last_word_ms") or {}
        w(f"- {v}: LiveKit waits after the last word a median {f2(wt.get('median'), 0)} ms (90th {f2(wt.get('p90'), 0)}; under 0.5 s on "
          f"{f2(100 * wt['share_under_500'], 0) if wt else '–'}% of commits, over 2 s on {f2(100 * wt['share_over_2000'], 0) if wt else '–'}%); "
          f"its preemptive generation's kept head start a median {f2(hd.get('median'), 0)} ms on {hd.get('kept_turns_over_passes', '–')} "
          f"kept turns over the passes; {f2(f['llm_calls_a_turn']['mean'], 4)} LLM calls a turn. Kept: {f['kept_by_over_passes']}. "
          f"Its commit comes a median {f2(vw.get('median'), 0)} ms (90th {f2(vw.get('p90'), 0)}) after the last word by {v}'s own word times "
          "(the wait above is LiveKit's own metric, from its VAD).")
    fc = a.flux_corners == "1"
    if fc:
        d = rd(R / "fc_bell.json")
        w("")
        w("### Bell alone against nothing, under Flux's own end of turn (eager off)")
        w("")
        if d:
            pd = d["paired"]
            fb, fn = d["arms"]["bell"], d["arms"]["nothing"]
            w(f"- nothing: {iv(fn['llm_calls_a_turn'], 4)} calls, {iv(fn['ms_sooner_a_turn'])} ms; Bell alone: "
              f"{iv(fb['llm_calls_a_turn'], 4)} calls, {iv(fb['ms_sooner_a_turn'])} ms; **Bell adds {iv(pd['ms_sooner_a_turn'], 2, True)} ms "
              f"a turn at {iv(pd['llm_calls_a_turn'], 4, True)} calls**"
              + (f"; last word to first audio changes by {iv(pd['last_word_to_first_audio_ms'], 0, True)} ms (paired mean; below 0 = sooner)" if pd.get("last_word_to_first_audio_ms") else "")
              + f" ({d['calls']} calls, {d['turns_read']} turns).")
            flat["fc.bell_vs_nothing.ms"] = pd["ms_sooner_a_turn"]
            flat["fc.bell_vs_nothing.calls"] = pd["llm_calls_a_turn"]
    if pipecat_on:
        d = rd(R / "pc_bell.json")
        w("")
        w("### Pipecat 1.16.0: Bell alone against nothing, under Flux's own end of turn (eager off)")
        w("")
        if d:
            pd = d["paired"]
            fb, fn = d["arms"]["bell"], d["arms"]["nothing"]
            m0, _ = e2e(fn.get("last_word_to_first_audio_ms"))
            m1, _ = e2e(fb.get("last_word_to_first_audio_ms"))
            w(f"- nothing: {iv(fn['llm_calls_a_turn'], 4)} calls, {iv(fn['ms_sooner_a_turn'])} ms, last word to first audio median {m0}; "
              f"Bell alone: {iv(fb['llm_calls_a_turn'], 4)} calls, {iv(fb['ms_sooner_a_turn'])} ms, median {m1}; **Bell adds "
              f"{iv(pd['ms_sooner_a_turn'], 2, True)} ms a turn at {iv(pd['llm_calls_a_turn'], 4, True)} calls**"
              + (f"; last word to first audio changes by {iv(pd['last_word_to_first_audio_ms'], 0, True)} ms (paired mean; below 0 = sooner)"
                 if pd.get("last_word_to_first_audio_ms") else "")
              + f" ({d['calls']} calls, {d['turns_read']} turns; kept {fb['kept_by_over_passes']}; Bell's guesses per pass "
              f"{fb['bell_guesses_per_pass']}).")
            flat["pc.bell_vs_nothing.ms"] = pd["ms_sooner_a_turn"]
            flat["pc.bell_vs_nothing.calls"] = pd["llm_calls_a_turn"]
            if pd.get("last_word_to_first_audio_ms"):
                flat["pc.bell_vs_nothing.e2e"] = pd["last_word_to_first_audio_ms"]
        else:
            w("- not read")
        w("- Pipecat runs on the wall clock here: no local inference reads the clock, but its passes are not "
          "replayed on the replay clock; each arm's two passes are compared under the checks.")
    w("")
    w("### The curve: Bell against LiveKit's default alone, by how many of the 27 known callers' calls the agent was taught (Flux's words)")
    w("")
    w("| calls taught (the 27's, seed 23's order) | Bell's guesses sent (per pass) | LLM calls added a turn | ms sooner added a turn |")
    w("|---|---|---|---|")
    for n in [int(x) for x in a.curve_ns.split(",")]:
        d = rd(R / f"curve_{n}.json")
        if not d:
            w(f"| {n} | not read | | |")
            continue
        pd = d["paired"]
        g = d["arms"]["lk_bell"]["bell_guesses_per_pass"]
        w(f"| {n}{' (the first call)' if n == 0 else ''}{' (all)' if n == 1070 else ''} | {g} | {iv(pd['llm_calls_a_turn'], 4, True)} | "
          f"{iv(pd['ms_sooner_a_turn'], 2, True)} |")
        flat[f"curve.{n}.ms"] = pd["ms_sooner_a_turn"]
        flat[f"curve.{n}.calls"] = pd["llm_calls_a_turn"]
    w("")
    w("### Notes")
    w("")
    w("- Rows that measure a speech-to-text's own early start are not in this table.")
    w("")
    w("")
    w("## The checks")
    w("")
    w("```")
    w(txt(st / "CONTROL"))
    w(txt(st / "TEACH"))
    w(txt(st / "API_FAULTS"))
    w("```")
    w("")
    w("The same arm twice, and alone against with Bell where Bell sent nothing (this turn and the next), ms sooner a turn:")
    w("")
    for p in sorted(R.glob("rep_*.json")):
        d = rd(p)
        for part in ("all", "bell_sent_nothing_here_or_next"):
            if part in d and d[part].get("ms_sooner", {}).get("turns"):
                x = d[part]["ms_sooner"]
                w(f"- {d['label']} [{part}]: {x['mean_diff']:+.2f} [{x['mean_diff_interval'][0]:+.2f}, {x['mean_diff_interval'][1]:+.2f}] ms, "
                  f"{x['turns_differing']} of {x['turns']} turns differ")
                flat[f"rep.{p.stem}.{part}"] = {"mean": x["mean_diff"], "interval": x["mean_diff_interval"], "turns": x["turns"]}
    w("")
    w(f"Sizes: {sizes}; calls replayed: {'all' if a.calls == '0' else a.calls + ' shortest'}; passes {a.passes}.")
    (out / "TABLE.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "TABLE.json").write_text(json.dumps({"mock": mock, "calls": a.calls, "numbers": flat, "live_rate_limits": rate_info, "api": api,
                                                "live": ls},
                                               indent=1), encoding="utf-8")
    print(f"TABLE {len(flat)} numbers -> {out / 'TABLE.md'}")
    print("TABLE_OK")


if __name__ == "__main__":
    main()
