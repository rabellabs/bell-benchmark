# Bell benchmark

What a voice-agent builder gets by switching Bell on, measured inside the hosts as they ship: LiveKit Agents 1.8.13 at its defaults and
Pipecat 1.16.0. Bell guesses the words a caller has still to say, so the reply can start before the caller stops. This benchmark
replays recorded bank calls into a real LiveKit and a real Pipecat, with and without Bell, and reads how much sooner the reply
starts, how many more LLM calls that costs, and the time from the caller's last word to the agent's first audio.

Bell's guesses come from the Rabel API itself, with your key. Nothing in this repository guesses.

## The results

Rabel's published numbers, and a rerun of this repository's scripts and data on 22 November 2026.

| | Rabel's published run | The rerun of 22 November 2026 |
|---|---|---|
| Bell's answer time | about 32 ms, from a stand-in for the API inside the process | the API's real round trip, median 2.4 ms, from a machine in the API's own cloud region |
| LiveKit's default alone, then with Bell | 118 ms a turn sooner [108, 128], 0.11 more LLM calls a turn | 122 [111, 133], 0.11 |
| Bell alone against no early start, in LiveKit | 161 [150, 172], 0.09 | 167 [156, 178], 0.09 |
| Bell alone against no early start, in Pipecat 1.16.0 | 170 [159, 182], 0.07 | 176 [164, 187], 0.07 |
| The curve, after 0, 12, 24, 48, 100, 300 and 1,070 calls taught | silent, 9, 44, 72, 94, 102, 118 | silent, 10, 46, 73, 96, 105, 122 |

- **ms a turn sooner** is how much sooner the reply's LLM call starts, averaged over every turn of the 376 test calls, with a 95%
  interval by whole calls. Both hosts hold an early reply until the turn ends, so the caller hears less than that. In one
  reading with Claude Haiku 5.5 and Deepgram Aura-2, the first audio came 115 ms a turn sooner under LiveKit's default, about 142
  with Bell alone in LiveKit, and 133 in Pipecat. That figure moves with the reply model's speed from hour to hour.
- Both runs passed the control first: `demo-bank` answered 741 guesses, 586 of them right.
- The rerun reads a few ms higher because the API answered sooner than the stand-in did. A run from farther away reads lower.
- The rerun was made before the API's public launch, against the same code on Rabel's test address, with both packages built
  from their public commits. Its reply and voice were the stand-in, so it gives no first-audio reading.

## The limits

- **One kind of call.** Scripted, simulated calls to one bank, in English, on Deepgram Flux's words. On freer speech Rabel has
  only looked informally, and Bell mostly stayed quiet. No live deployment's callers and no other language have been measured.
- **A trained agent.** Bell had learned 1,070 calls by 27 of the bank's callers. The 28 test callers were never taught and
  nothing was tuned on them. The curve shows what fewer calls give.
- **No speech-to-text's own early start.** Flux's eager end of turn was off in every row.
- **Where you run it matters.** Bell's answer reaches the replay after the API's real round trip from your machine.
- **Two passes are not identical with Bell.** Its answers come through the network, so the same arm run twice differed by 3 ms
  a turn in the rerun, on 164 of 2,193 turns.

## Get the data folder

A run needs the data folder: the recorded speech-to-text messages of the test calls, the turns read, and the calls the curve's
agents are taught (listed with their sizes and checksums in its `MANIFEST.json`). It is one file on this repository's releases,
5 MB to fetch and 75 MB unpacked. Fetch it and unpack it beside `run.sh`, which makes `data/`:

```
curl -L -O https://github.com/rabellabs/bell-benchmark/releases/download/data-v1/bell-benchmark-data.tar.gz
tar -xzf bell-benchmark-data.tar.gz
```

Its SHA-256 is `2a2d16096b8175ca03b3945669ec4d913fb65ca7cd5945975edb8c9796d4ecea`. A run checks every file in it against
`MANIFEST.json` before it starts. Its texts are Deepgram Flux's machine transcripts of the callers' audio; the audio itself is not
in it, and a run fetches that from the dataset's own repository.

## Run it

On a Linux machine (Ubuntu 24.04, 8 cores or more, 30 GB of disk), with the data folder in `data/`, as root:

```
RABEL_API_KEY=<your key> bash run.sh
```

Not root? Run `sudo apt-get install -y python3-venv git` first (a fresh Ubuntu lacks both), then the line above without `sudo`:
`sudo` would drop `RABEL_API_KEY` from the environment.

The table is written to `work/out/TABLE.md` (and every number to `work/out/TABLE.json`). Every package a run installs is pinned in
`run.sh`, and the table lists the versions that were installed.

- With `LIVE=1` (and `ANTHROPIC_API_KEY`, `DEEPGRAM_API_KEY`) the end to end column is measured with Claude Haiku 5.5 and Deepgram
  Aura-2. Without it, that column comes from a stand-in and the table says DRY RUN.
- `CALLS=3` replays only the three shortest calls: a check that everything runs.
- The other settings are listed at the top of `run.sh`. Keys are read from the environment only and are never printed or written.

## What a run costs

Estimates from Rabel's own runs in November 2026; yours depend on your machine and your network.

- **The small check** (`CALLS=3`, `LIVE=0`): about 20 minutes of machine time and about 95 minutes of your account's conversation
  time, most of it the two controls (812 conversations).
- **A full run:** about four hours on one 8-core machine (a run split over three such machines took 54, 117 and 74 minutes). One
  full run, with its dry runs and a stopped first attempt, used about 2,600 minutes of a test account's conversation time, most of
  it Pipecat's calls, which hold a conversation open for the whole call. `PIPECAT=0` leaves them out.
- **`LIVE=1`:** about $10 to $12 of Claude Haiku 5.5 and Aura-2 when every reply's words are measured. `LIVE_MAX_USD` (30 by
  default) stops it.
- **The curve's agents stay in your account,** named `<AGENT_PREFIX>-c<N>`. Delete each with `DELETE /v1/agents/{agent}` when you
  are done. A run never teaches an agent that already has calls.

## What it measures

- **The lead table:** LiveKit's default turn ending (its own turn detector, with preemptive generation on) alone, against the same with
  Bell, on Deepgram Flux's words. The calls were recorded with Flux's eager end of turn at 0.5; its eager messages are not fed, which
  is what an agent with eager off receives.
- **The curve:** the same, with Bell guessing for a new agent of your account taught 0, 12, 24, 48, 100, 300 or all of the 27 known
  callers' 1,070 calls. 0 is the first call: Bell sends nothing until it has heard the agent's calls.
- **Bell alone against nothing** under Flux's own end of turn, in LiveKit and in Pipecat.
- **Last word to first audio** includes the reply model's and the voice's own times, which change from hour to hour. The table
  prints the UTC hours the live part ran in and the reply model's median time to its first text beside it. Read it as one reading
  of a range, not as a fixed number.
- **The calls:** the Gridspace-Stanford Harper Valley dataset of simulated bank calls (CC BY 4.0). The 28 test callers' 376 calls are
  only read: nothing is taught from them and nothing is tuned on them. The recorded speech-to-text messages of those calls are in the
  data folder, so every run reads the same words at the same times.

## How it works

- **Replay clock.** LiveKit's timers, its VAD and its turn detector follow one clock that the replay drives (`bench/hosts/vclock.py`),
  so the same call replayed twice gets the same answers at the same moments from LiveKit itself. Each arm is replayed twice
  and the two are compared.
- **Bell's answer time is the API's real round trip.** In LiveKit's replay, the hookup's messages go to the API from the machine the
  benchmark runs on. While the API answers, the replay's clock stands still; each answer then reaches the hookup as long after the
  words it was made from as the API really took through the network. The table states the median and the tail of those round trips.
  Pipecat runs on the wall clock against the API, as a live call would.
- **Knowing that the API has no guess.** The API answers a `heard` message with a guess or with nothing. After each `heard` the replay
  sends a `ping`; the API answers it with `pong` once every answer to the messages before it has gone out, so once the `pong` is
  back every answer to the `heard` is in. The `pong` is never shown to the hookup.
- **The control, first.** Before any replay, the 28 test callers' calls are sent to the ready-made agent `demo-bank`: it must answer
  741 guesses, 586 of them right. The agent this run teaches all 1,070 calls must then guess exactly as `demo-bank` does. If the
  control fails (`CONTROL_FAILS`), the API's `demo-bank` does not answer as it did when this benchmark was made, or the address in
  `RABEL_API_URL` does not serve the API's version 2; nothing is replayed and no table is written.
- **Intervals** are 95%, from 2,000 resamples of whole calls.

## What is here

| file | what it does |
|---|---|
| `run.sh` | the whole run, from nothing to the table |
| `bench/rabel_api.py` | the API: HTTPS requests and one conversation over a plain socket |
| `bench/api_control.py` | the control through the API |
| `bench/teach.py` | the curve's agents, taught through `POST /v1/agents/{agent}/calls` |
| `bench/hosts/livekit_default_host.py` | LiveKit with the recorded messages and the call's audio, on the replay clock, Bell through the API |
| `bench/hosts/livekit_vendor_host.py`, `livekit_default.py`, `vclock.py` | the recorded socket, LiveKit's default ending, the replay clock |
| `bench/hosts/pipecat_host.py` | Pipecat with the recorded Flux messages, Bell's turn strategies in its user aggregator, Bell through the API |
| `bench/e2e/live.py`, `combine.py` | the live LLM-to-first-audio times, and the reads that put them together with the replays |
| `bench/read.py`, `repeat.py`, `table.py`, `compare.py` | the reads, the run-twice checks, the table, and two tables compared |

## Credit and licence

The calls are from the Gridspace-Stanford Harper Valley dataset of simulated bank calls, made by Gridspace and Stanford's CS224S
(https://github.com/cricketclub/gridspace-stanford-harper-valley), under CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). A run fetches the caller audio from that repository. The data folder's texts are
Deepgram Flux's machine transcripts of that audio, changed in form for the replay.

LiveKit, Pipecat and Deepgram are named only to say what the benchmark runs. Rabel Labs is not affiliated with them.

The code here is under the Apache License 2.0 (`LICENSE`).
