#!/bin/bash
# The benchmark, from nothing to its table, on one Linux machine (Ubuntu 24.04, 8 cores or more). Run it as root, or first run
# `sudo apt-get install -y python3-venv git` (a fresh Ubuntu lacks both, and sudo would drop RABEL_API_KEY from the environment).
#
#   RABEL_API_KEY=<your key> bash run.sh
#
# What it does, in order (every step writes a stamp in $WORK/stamps; a stop names its step in $WORK/stamps/END):
#   1. setup: two Python venvs (LiveKit Agents 1.8.13 with its Deepgram plugin and livekit-plugins-rabel; Pipecat 1.16.0 with its
#      Deepgram service and pipecat-rabel; every version pinned below); the 28 test callers' caller audio from the public repository
#      of the Gridspace-Stanford Harper Valley dataset (CC BY 4.0).
#   2. the control through the Rabel API, first, and it gates everything: the 28 test callers' 376 calls sent to the ready-made agent
#      demo-bank ("learn": false) must read 741 guesses, 586 of them right (bench/api_control.py). Then the same calls with a version 3
#      hello, as livekit-plugins-rabel 1.9.0 asks, which gets the new Bell: its two numbers (every guess sent, the turns right) must
#      read DEMO_V3_WANT (set below); while DEMO_V3_WANT is empty they are read and written down, not checked.
#   3. the curve's agents: new agents of your account taught the first 0, 12, 24, 48, 100, 300 and all 1,070 of the 27 known callers'
#      calls through POST /v1/agents/{agent}/calls (bench/teach.py); the agent taught all of them must guess exactly as demo-bank does
#      on the 28 (the control again, turn by turn).
#   4. the replays (each arm PASSES times), LiveKit on the replay clock (bench/hosts/vclock.py), Bell's guesses from the API with their
#      real round trips:
#        the lead table: LiveKit's default ending (its turn detector, preemptive generation on) alone and with Bell, on Flux's words
#          (eager off);
#        the curve and the first call: with Bell on Flux's words, the agent taught N calls (all 1,070 is demo-bank: the lead table's);
#        Flux's own end of turn: nothing, and Bell alone;
#        Pipecat 1.16.0, Flux's own end of turn: nothing, and Bell alone (on the wall clock, against the API as a call would).
#   5. the live part: the time from each reply's LLM call to its first audio, with Claude Haiku 5.5 and Deepgram Aura-2, for every
#      distinct reply text (LIVE=1: ANTHROPIC_API_KEY and DEEPGRAM_API_KEY; LIVE=0: a stand-in, and the table says DRY RUN). Rows
#      already measured can be given (LIVE_CACHE: live.jsonl files, the first file's row kept for a text); the rest are measured.
#   6. the reads and the table: $WORK/out/TABLE.md and TABLE.json (bench/table.py).
#
# Settings (environment): RABEL_API_KEY (required), RABEL_API_URL (default https://api.rabellabs.com), DATA (the data folder,
#   default ./data; data/MANIFEST.json lists its files), WORK (default ./work), PASSES (2), PROCS (the machine's cores), CALLS (0 = all
#   376; N = the N shortest, a dry run), CURVE_NS (0,12,24,48,100,300), AGENT_PREFIX (bench-<time>: the curve's agents' names),
#   PIPECAT (1), PC_AT_ONCE (40: Pipecat's calls at once, each a conversation with the API; your key's limit is
#   50), LIVE (0), LIVE_CACHE, REPS (2), LIVE_CONC (8), LIVE_MAX_USD (30), LIVE_MAX_REQ (20000), PACE_S (3600),
#   HOOKUP_LK (pip spec of livekit-plugins-rabel, default the release pinned below) and HOOKUP_LK_LABEL, HOOKUP_PC and HOOKUP_PC_LABEL
#   likewise for pipecat-rabel, PART (all | setup | table), ARMS (lead,curve,ends: which replays this machine runs; a run split over
#   machines gives each its part, every machine runs the control first, only the one with curve teaches).
# Keys are read from the environment only and never printed or written.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE" || exit 1
DATA=$(cd "${DATA:-$HERE/data}" 2>/dev/null && pwd) || { echo "no data folder: fetch it as the README says, or set DATA"; exit 1; }
WORK=${WORK:-$HERE/work}
mkdir -p $WORK/stamps $WORK/logs $WORK/out/reads $WORK/lkr $WORK/pcr
ST=$WORK/stamps; LG=$WORK/logs; OUT=$WORK/out
PART=${PART:-all}
ARMS=${ARMS:-lead,curve,ends}
has() { case ",$ARMS," in *,$1,*) return 0;; esac; return 1; }
NP=${PASSES:-2}
P=${PROCS:-$(nproc)}
CURVE_NS=${CURVE_NS:-0,12,24,48,100,300}
PIPECAT=${PIPECAT:-1}
PC_AT_ONCE=${PC_AT_ONCE:-40}
LIVE=${LIVE:-0}
VPY=$WORK/lkv/bin/python
PPY=$WORK/pcv/bin/python
EVF=$DATA/flux28_events.jsonl
SCH=$DATA/sched_0.5.json
HVB=$WORK/hvb
AUD=$HVB/data/audio/caller
HVB_URL=https://github.com/cricketclub/gridspace-stanford-harper-valley.git
HVB_COMMIT=0bd721e877c4a85d8c13ff837e68661ea6200a98
# the versions every run installs (a later release of a host or a hookup can change what is measured)
LK_AGENTS='livekit-agents[deepgram]==1.8.13'
PC_AI='pipecat-ai[deepgram]==1.16.0'
HOOKUP_LK_PIN='livekit-plugins-rabel==1.9.0'
# demo-bank's version 3 check (the new Bell): every guess sent and the turns right, once read on the API at its release
DEMO_V3_WANT=${DEMO_V3_WANT:-2341,1239}
HOOKUP_PC_PIN='pipecat-rabel==0.3.0'
PY_PINS='numpy==2.5.5 anthropic==2.11.0 websockets==17.4'
PC_PINS='numpy==2.5.5 websockets==17.4'
export PYTHONDONTWRITEBYTECODE=1
say() { echo "$(date -u +%FT%TZ) $*" | tee -a $LG/run.log; }
stop() { say "STOP $1"; echo "$1" > $ST/END; exit 1; }
[ -n "${RABEL_API_KEY:-}" ] || stop "NO_RABEL_API_KEY (set RABEL_API_KEY)"
export RABEL_API_URL=${RABEL_API_URL:-https://api.rabellabs.com}
date -u +%FT%TZ > $ST/START
rm -f $ST/END
say "the benchmark: PART=$PART ARMS=$ARMS API=$RABEL_API_URL DATA=$DATA PASSES=$NP PROCS=$P CALLS=${CALLS:-0} CURVE_NS=$CURVE_NS PIPECAT=$PIPECAT LIVE=$LIVE"

if [ "$PART" = all ] || [ "$PART" = setup ]; then
  # 0. the data, checked against its manifest
  python3 - "$DATA" <<'EOF' > $ST/DATA 2>&1 || { cat $ST/DATA; stop DATA_FAILED; }
import hashlib, json, pathlib, sys
d = pathlib.Path(sys.argv[1]); m = json.load(open(d / "MANIFEST.json"))
bad = [n for n, v in m.items() if not (d / n).exists() or hashlib.sha256((d / n).read_bytes()).hexdigest() != v["sha256"]]
print("DATA_OK" if not bad else f"DATA_BAD {bad}")
sys.exit(1 if bad else 0)
EOF
  # 1. the venvs and the audio
  if [ ! -x $VPY ]; then
    if ! python3 -c "import ensurepip" > /dev/null 2>&1 || ! command -v git > /dev/null; then
      export DEBIAN_FRONTEND=noninteractive
      apt-get update -qq > /dev/null && apt-get install -y -qq python3-venv git > /dev/null || stop APT_FAILED
    fi
    python3 -m venv $WORK/lkv && $VPY -m pip install -q "$LK_AGENTS" "${HOOKUP_LK:-$HOOKUP_LK_PIN}" $PY_PINS \
      > $LG/pip.log 2>&1 || stop "PIP_FAILED (the log: $LG/pip.log)"
  fi
  if [ "$PIPECAT" = 1 ] && [ ! -x $PPY ]; then
    python3 -m venv $WORK/pcv && $PPY -m pip install -q "$PC_AI" "${HOOKUP_PC:-$HOOKUP_PC_PIN}" $PC_PINS \
      > $LG/pip_pc.log 2>&1 || stop "PIP_PC_FAILED (the log: $LG/pip_pc.log)"
  fi
  echo "${HOOKUP_LK_LABEL:-${HOOKUP_LK:-$HOOKUP_LK_PIN}}" > $WORK/lkr/HOOKUP_COMMIT
  echo "${HOOKUP_PC_LABEL:-${HOOKUP_PC:-$HOOKUP_PC_PIN}}" > $WORK/pcr/HOOKUP_COMMIT
  $VPY -m pip freeze | grep -i -E "^livekit|^numpy|^anthropic|^websockets" > $LG/freeze.txt
  [ "$PIPECAT" = 1 ] && $PPY -m pip freeze | grep -i -E "^pipecat|^websockets|^numpy" > $LG/freeze_pc.txt
  if [ ! -s $ST/AUDIO ]; then
    python3 -c "import json; print('\n'.join(s for c in json.load(open('$DATA/test_callers.json'))['callers'] for s in c['sids']))" > $WORK/sids_28.txt
    [ -d $HVB/.git ] || git clone -q --filter=blob:none --no-checkout $HVB_URL $HVB || stop AUDIO_CLONE_FAILED
    git -C $HVB sparse-checkout set --no-cone $(sed 's#.*#data/audio/caller/&.wav#' $WORK/sids_28.txt) LICENSE README.md > $LG/audio.log 2>&1
    git -C $HVB checkout -q $HVB_COMMIT >> $LG/audio.log 2>&1
    NA=$(ls $AUD 2>/dev/null | wc -l); say "audio: $NA caller files (the Gridspace-Stanford Harper Valley dataset, $HVB_COMMIT)"
    [ "$NA" -ge 376 ] && echo "$NA" > $ST/AUDIO || stop AUDIO_FAILED
  fi
  # 2. the control through the API (gates everything)
  $VPY bench/api_control.py --streams $DATA/streams_28.json --agent demo-bank --want 741,586 --out $OUT/api_control_demo-bank.json \
    > $LG/api_control.log 2>&1
  tail -n 2 $LG/api_control.log > $ST/CONTROL; cat $ST/CONTROL
  grep -q API_CONTROL_PASSES $ST/CONTROL || stop CONTROL_FAILS
  # the same with a version 3 hello (the new Bell, as the hookup 1.9.0 asks): read, and checked against DEMO_V3_WANT when it is set
  VW=""; [ -n "$DEMO_V3_WANT" ] && VW="--want $DEMO_V3_WANT"
  $VPY bench/api_control.py --streams $DATA/streams_28.json --agent demo-bank --v 3 $VW --out $OUT/api_control_demo-bank_v3.json \
    > $LG/api_control_v3.log 2>&1
  { tail -n 2 $LG/api_control_v3.log; [ -n "$DEMO_V3_WANT" ] || echo "(version 3: read, not checked: DEMO_V3_WANT is not set)"; } \
    > $ST/CONTROL_V3; cat $ST/CONTROL_V3
  grep -q API_CONTROL_PASSES $ST/CONTROL_V3 || stop CONTROL_V3_FAILS
  # 3. the curve's agents, and the one taught everything read against demo-bank
  PFX=${AGENT_PREFIX:-bench-$(date +%s)}
  echo "$PFX" > $ST/AGENT_PREFIX
  if has curve; then
  $VPY bench/teach.py --teach $DATA/teach_27.json --prefix $PFX --ns ${CURVE_NS},1070 --out $OUT/teach.json > $LG/teach.log 2>&1
  cat $LG/teach.log
  grep -q TEACH_OK $LG/teach.log || { tail -n 3 $LG/teach.log > $ST/TEACH; stop TEACH_FAILED; }
  $VPY bench/api_control.py --streams $DATA/streams_28.json --agent $PFX-c1070 --same-as $OUT/api_control_demo-bank.json \
    --out $OUT/api_control_taught.json > $LG/api_control_taught.log 2>&1
  { tail -n 1 $LG/teach.log; tail -n 2 $LG/api_control_taught.log; } > $ST/TEACH; cat $ST/TEACH
  grep -q API_CONTROL_PASSES $ST/TEACH || stop TAUGHT_AGENT_DIFFERS
  fi
  echo SETUP_DONE > $ST/SETUP
  [ "$PART" = setup ] && { echo SETUP_DONE > $ST/END; exit 0; }
fi

grep -q SETUP_DONE $ST/SETUP 2>/dev/null || stop NO_SETUP
grep -q API_CONTROL_PASSES $ST/CONTROL || stop NO_CONTROL_PASS
PFX=$(cat $ST/AGENT_PREFIX)
CL=""; [ "${CALLS:-0}" != 0 ] && CL="--calls $CALLS"
HK=$(cat $WORK/lkr/HOOKUP_COMMIT); PHK=$(cat $WORK/pcr/HOOKUP_COMMIT)
host() {  # host <turn> <vendor> <arm> <out> [extra]
  turn=$1; v=$2; arm=$3; o=$4; shift 4
  $VPY bench/hosts/livekit_default_host.py --turn $turn --vendor $v --events $EVF --sched $SCH --audio $AUD --arm $arm \
    --out $OUT/$o.jsonl --procs $P --hookup-label "$HK" $CL "$@" > $LG/$o.log 2>&1
  say "$o: $(tail -n 2 $LG/$o.log | head -n 1 | cut -c1-300)"
  grep -q LIVEKIT_DEFAULT_HOST_DONE $LG/$o.log || stop "HOST_FAILED $o"
}
pch() {  # pch <arm> <out>
  arm=$1; o=$2; at=$(( PC_AT_ONCE / P )); [ $at -ge 1 ] || at=1
  [ "$arm" = nothing ] && at=12
  $PPY bench/hosts/pipecat_host.py --sched $SCH --arm $arm --out $OUT/$o.jsonl --procs $P --conc $at --hookup-label "$PHK" --hookup-as strategy $CL \
    > $LG/$o.log 2>&1
  say "$o: $(tail -n 2 $LG/$o.log | head -n 1 | cut -c1-300)"
  grep -q PIPECAT_HOST_DONE $LG/$o.log || stop "HOST_FAILED $o"
}
rep() {  # rep <a> <b> <label> [quiet pass]
  q=""; [ -n "${4:-}" ] && q="--quiet $OUT/$4.jsonl"
  $VPY bench/repeat.py --a $OUT/$1.jsonl --b $OUT/$2.jsonl $q --label "$3" --json $OUT/reads/rep_$1__$2.json >> $LG/reads.txt 2>&1
}
passes() { seq -s, -f "$OUT/$1_%g.jsonl" 1 $NP; }

# 4. the replays
if has lead; then
  for r in $(seq 1 $NP); do host livekit flux lk flux_lk_$r; host livekit flux lk_bell flux_bell_$r; done
  rep flux_lk_1 flux_lk_2 "flux: LiveKit's default alone, run twice"
  rep flux_bell_1 flux_bell_2 "flux: with Bell, run twice"
  rep flux_lk_1 flux_bell_1 "flux: alone against with Bell" flux_bell_1
fi
if has curve; then
  for n in ${CURVE_NS//,/ }; do
    for r in $(seq 1 $NP); do host livekit flux lk_bell curve_${n}_$r --agent $PFX-c$n; done
  done
  has lead || for r in $(seq 1 $NP); do host livekit flux lk flux_lk_$r; done    # the curve's own baseline when the lead runs elsewhere
fi
if has ends; then
  for arm in nothing bell; do for r in $(seq 1 $NP); do host stt flux $arm fc_${arm}_$r; done; done
  rep fc_nothing_1 fc_nothing_2 "Flux's end: nothing, run twice"
  rep fc_bell_1 fc_bell_2 "Flux's end: Bell alone, run twice"
fi
if has ends && [ "$PIPECAT" = 1 ]; then
  for arm in nothing bell; do for r in $(seq 1 $NP); do pch $arm pc_${arm}_$r; done; done
  rep pc_nothing_1 pc_nothing_2 "Pipecat: nothing, run twice"
  rep pc_bell_1 pc_bell_2 "Pipecat: Bell alone, run twice"
fi
# every call's conversation with the API reached ready and answered to the end: any fault stops the table
$VPY - "$OUT" <<'EOF' > $ST/API_FAULTS 2>&1
import json, pathlib, sys
bad = {}
for p in sorted(pathlib.Path(sys.argv[1]).glob("*.jsonl.meta.json")):
    m = json.loads(p.read_text())
    n = sum((m.get("faults") or {}).values()) if m.get("host") == "livekit_default" else sum((x or {}).get("faults", 0) for x in (m.get("api") or []))
    if n:
        bad[p.name] = n
print(f"calls that failed (a replay fault, or a conversation with the API that failed): {sum(bad.values())} {bad}")
print("API_FAULTS_NONE" if not bad else "API_FAULTS")
EOF
cat $ST/API_FAULTS
grep -q API_FAULTS_NONE $ST/API_FAULTS || stop API_FAULTS

# 5. the live part: the distinct words every answering reply was made on
TP=""
if has lead; then
  TP="--pass lead=$(passes flux_lk),$(passes flux_bell)"
fi
has curve && for n in ${CURVE_NS//,/ }; do TP="$TP --pass c$n=$(passes flux_lk),$(passes curve_$n)"; done
has ends && TP="$TP --pass fc=$(passes fc_nothing),$(passes fc_bell)"
has ends && [ "$PIPECAT" = 1 ] && TP="$TP --pass pc=$(passes pc_nothing),$(passes pc_bell)"
$VPY bench/e2e/combine.py texts $TP --out $OUT/texts.json > $LG/texts.log 2>&1 || stop TEXTS_FAILED
cat $LG/texts.log
if [ -n "${LIVE_CACHE:-}" ] && [ ! -s $OUT/live.jsonl ]; then
  $VPY - $OUT/texts.json $OUT/live.jsonl ${LIVE_CACHE//,/ } <<'EOF' > $ST/LIVE_CACHE 2>&1
import json, sys
want = {t["id"] for t in json.load(open(sys.argv[1]))}
seen, rows = set(), []
for f in sys.argv[3:]:
    have = set()
    for ln in open(f):
        if ln.strip():
            x = json.loads(ln)
            if x.get("ok") and x["id"] in want and x["id"] not in seen:
                rows.append(ln if ln.endswith("\n") else ln + "\n"); have.add(x["id"])
    seen |= have
open(sys.argv[2], "w").writelines(rows)
print(f"LIVE_CACHE: {len(seen)} of {len(want)} reply texts already measured ({len(rows)} rows from {len(sys.argv) - 3} files); the rest measured now")
EOF
  cat $ST/LIVE_CACHE
fi
LV="--reps ${REPS:-2} --conc ${LIVE_CONC:-8} --max-usd ${LIVE_MAX_USD:-30} --max-requests ${LIVE_MAX_REQ:-20000}"
if [ "$LIVE" = 1 ]; then
  $VPY bench/e2e/live.py check --deepgram > $ST/KEYCHECK 2>&1 || { cat $ST/KEYCHECK; stop KEY_REFUSED; }
  $VPY bench/e2e/live.py run --texts $OUT/texts.json --out $OUT/live.jsonl $LV --pace-s ${PACE_S:-3600} > $LG/live.log 2>&1
  tail -n 1 $LG/live.log > $ST/LIVE
elif [ -n "${LIVE_CACHE:-}" ]; then   # the given rows only: a reply text not among them leaves its turn out of the end to end read
  echo "LIVE_DONE (the given rows only, no live call: $(cat $ST/LIVE_CACHE))" > $ST/LIVE
else
  $VPY bench/e2e/live.py run --texts $OUT/texts.json --out $OUT/live.jsonl $LV --mock --pace-s ${PACE_S:-3600} > $LG/live.log 2>&1
  tail -n 1 $LG/live.log > $ST/LIVE
fi
grep -q LIVE_DONE $ST/LIVE || stop "LIVE_STOPPED $(cat $ST/LIVE | tr '\n' ' ')"

# 6. the reads and the table
cmb() {  # cmb <name> <label> <armA>=<prefix> <armB>=<prefix>
  n=$1; lab=$2; a=${3%%=*}; ap=${3#*=}; b=${4%%=*}; bp=${4#*=}
  $VPY bench/e2e/combine.py read --pass $a=$(passes $ap) --pass $b=$(passes $bp) --live $OUT/live.jsonl --label "$lab" \
    --json $OUT/reads/$n.json > $LG/read_$n.log 2>&1 || stop "READ_FAILED $n"
  cat $LG/read_$n.log >> $LG/reads.txt
}
has lead && cmb lead_flux "flux words, LiveKit's default ending, eager off" lk=flux_lk lk_bell=flux_bell
has ends && cmb fc_bell "Flux's end of turn: Bell alone against nothing" nothing=fc_nothing bell=fc_bell
has ends && [ "$PIPECAT" = 1 ] && cmb pc_bell "Pipecat, Flux's end of turn: Bell alone against nothing" nothing=pc_nothing bell=pc_bell
has curve && for n in ${CURVE_NS//,/ }; do cmb curve_$n "the curve: the agent taught $n calls, LiveKit's default ending, Flux's words" lk=flux_lk lk_bell=curve_$n; done
has lead && cmb curve_1070 "the curve: all 1,070 calls (demo-bank, the lead table's own)" lk=flux_lk lk_bell=flux_bell
$VPY bench/table.py --out $OUT --stamps $ST --curve-ns ${CURVE_NS},1070 --calls ${CALLS:-0} --passes $NP \
  --pipecat $PIPECAT > $LG/table.log 2>&1 || { tail -n 5 $LG/table.log; stop TABLE_FAILED; }
tail -n 2 $LG/table.log
say "the table: $OUT/TABLE.md"
echo TABLE_DONE > $ST/END
