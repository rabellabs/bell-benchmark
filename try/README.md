# Try Bell on your own calls

**Your transcripts go only to the Rabel API, in your own API calls, with your own key. The script writes nothing to disk
except the report, and only when you ask for it with `--out`. Your agent's own lines are dropped before anything is sent.**

`try_bell.py` shows what Bell would have done on your calls. It teaches a new agent your earlier calls, plays your later
calls through the live API the way a hookup does, and counts how often Bell's guess at a caller's turn equalled the final
words before the caller finished.

## Run it

Python 3.11 or later, and `websockets` (the package the hookups use for the connection):

```sh
pip install websockets
export RABEL_API_KEY=...          # your key from rabellabs.com
python try_bell.py calls.jsonl
```

| flag | |
|---|---|
| `--key` | your Rabel API key (default `RABEL_API_KEY`) |
| `--agent` | the agent to teach and play (default a new name, `try-xxxx`) |
| `--teach-share` | the share of calls, from the first, that teach the agent (default `0.7`) |
| `--url` | the API (default `RABEL_API_URL`, else `https://api.rabellabs.com`) |
| `--no-learn` | the played calls teach nothing after they are played (a live agent learns from every call) |
| `--forget` | after the report, delete what Bell learned for the agent |
| `--out` | also write the report to this file |

## Your calls

A file, or a folder of files taken in the order of their names. Calls are taken in the order they come, so put them in
the order they happened.

JSONL, one call a line (a `.json` file holding a list of calls, or `{"calls": [...]}`, works too):

```json
{"call_id": "c-001", "turns": [{"speaker": "agent", "text": "Thanks for calling, how can I help?"}, {"speaker": "caller", "text": "Hi, I'd like to move my appointment to Friday."}]}
```

CSV, one turn a row, in order, with the columns `call_id`, `speaker`, `text`.

A caller is `caller`, `user`, `customer`, `client`, `human` or `patient`; an agent is `agent`, `assistant`, `bot`,
`system`, `ai`, `operator`, `rep` or `representative`. Any other speaker stops the script and names the line. A few hundred
calls give a steadier read than a few dozen. Strip names, numbers and card details first if your calls carry them.

## What it does

1. Keeps the caller's turns. A turn over the API's 4,096 characters is left out and counted.
2. Splits the calls by order. The first 70% teach a new agent through `POST /v1/agents/{agent}/calls`, from their final
   words, within the API's limits (1,000 calls and 8 MB a request, 10,000 calls an hour). Teaching is free. The rest are
   held out, and none is taught before it is played.
3. Plays each held-out call in its own conversation (`/v1/conversation`, version 3): each turn's words sent as growing
   interim text, one word at a time, then the final words. It records every guess and compares it with the final words
   as the hookups do (case and punctuation ignored). A turn counts as guessed when the last guess before the final words
   equalled them: that is the reply a hookup keeps. With learning on (the default), Bell learns from each played call when
   it closes, as it would live; a call is always played before it is learned from.
4. Prints the report: turns played; the share of turns Bell guessed before the caller finished; on those, how many words
   early on average; wrong guesses a turn (each one an extra call to your agent's language model, never heard by the
   caller).

Playing uses conversation seconds from your account (a conversation is metered by the second it is open; a call here
takes a second or two). The agent keeps what it learned, so you can install a hookup on it and it starts already trained.
`--forget`, or `DELETE /v1/agents/{agent}`, removes it.

## What the numbers are

A text read, not a timing. Words arrive here cleaner and more often than from a live speech-to-text, so a live share is
lower: on our bank calls Bell guessed 57% of turns in text, and about a third through LiveKit. The script prints no
milliseconds. Those come from a live install: its hookup reports the head start of each kept guess in your agent's
status, `GET /v1/agents/{agent}`.
