"""LiveKit's default turn ending, as `livekit_default_host.py` runs it on Flux's recorded words.

What LiveKit Agents 1.8.13 does when an AgentSession is made with only an STT and an LLM (read in its source, 19 Nov 2026):
  - VAD: `inference.VAD(model="silero")`, run locally (livekit-local-inference's native Silero), on the call's audio.
  - Turn detector: `inference.TurnDetector()`, an AUDIO end-of-turn model: `turn-detector-v1-mini`, run locally from the
    livekit-local-inference wheel (no download), unless LiveKit Cloud's inference keys are set (then `turn-detector-v1`, in their
    cloud). It is not the text model on huggingface.co/livekit/turn-detector (that is livekit-plugins-turn-detector's
    MultilingualModel, which 1.8.13 no longer uses by default).
  - Endpointing: the streaming detector's defaults, min_delay 0.3 s and max_delay 2.5 s. After the VAD's end of speech (its
    silence must be at least 200 ms for the detector to run), the detector scores the turn's audio; the turn is committed
    min_delay after the last speech when the score is at or over its "unlikely" threshold (per language), max_delay after it when
    under.
  - Preemptive generation, on by default (at most 3 starts a turn, none after 10 s of speech): the LLM is started on each final
    transcript that arrives once the turn is ended by the VAD, and on each PREFLIGHT_TRANSCRIPT (a vendor's early start, or Bell's
    guess) at any time; at the commit the reply is kept when its words are the final words. So with the vendor's early start off,
    the head start LiveKit gives is the commit time less the final transcript's arrival: at most about min_delay (0.3 s) less the
    vendor's final latency, and up to max_delay on turns the detector finds unlikely to be over.
So this mode needs the call's audio, fed in real time beside the recorded messages on the same clock (the recording's t0 is the
first audio chunk the streamer sent). The vendor's early start is held back (its PREFLIGHT passed on as an interim); Bell, when on,
is the only early start. A turn's commit is LiveKit's own: the moment it schedules the reply.
"""
from __future__ import annotations

import asyncio
import time
import wave


def session_for(stt_obj, llm, call):
    from livekit.agents import AgentSession
    # LiveKit's defaults: nothing but the STT and the LLM given (the default VAD and audio turn detector, preemptive generation on)
    return AgentSession(stt=stt_obj, llm=llm)


def audio_input(path, call):
    from livekit import rtc
    from livekit.agents.voice.io import AudioInput

    class AudioReplay(AudioInput):
        """the caller's file in 20 ms frames at the recording's clock: frame k at origin + 20k ms"""

        def __init__(self):
            super().__init__(label="replay-audio")
            with wave.open(str(path), "rb") as w:
                self.rate = w.getframerate()
                self.pcm = w.readframes(w.getnframes())
            self.n = self.rate // 50
            self.k = 0

        async def __anext__(self):
            while call.origin is None:
                await asyncio.sleep(0.005)
            i = self.k * self.n * 2
            if i >= len(self.pcm) + self.rate * 2 * 3:   # the file, then 3 s of silence (the streamer's tail)
                await call.done.wait()
                await asyncio.sleep(3600)
            due = call.origin + self.k * 0.020
            wait = due - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
            chunk = self.pcm[i:i + self.n * 2]
            if len(chunk) < self.n * 2:
                chunk = chunk + b"\x00\x00" * (self.n - len(chunk) // 2)
            self.k += 1
            return rtc.AudioFrame(data=chunk, sample_rate=self.rate, num_channels=1, samples_per_channel=self.n)

    return AudioReplay()
