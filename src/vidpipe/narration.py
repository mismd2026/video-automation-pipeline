"""Narration audio via VOICEVOX and subtitle timing derived from it.

Subtitle timestamps come from the *actual* synthesized WAV durations, so captions
never drift from the voice — the usual failure of estimating from character count.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx


class TTS(Protocol):
    def synthesize(self, text: str) -> bytes: ...


@dataclass
class Voicevox:
    base_url: str = "http://localhost:50021"
    speaker: int = 3
    speed: float = 1.15

    def synthesize(self, text: str) -> bytes:
        with httpx.Client(base_url=self.base_url, timeout=60) as c:
            q = c.post("/audio_query", params={"text": text, "speaker": self.speaker})
            q.raise_for_status()
            query = q.json()
            query.update(speedScale=self.speed, postPhonemeLength=0.3)
            s = c.post("/synthesis", params={"speaker": self.speaker}, json=query)
            s.raise_for_status()
            return s.content


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


def synthesize_scenes(tts: TTS, narrations: list[str], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, text in enumerate(narrations):
        p = out_dir / f"scene_{i:02d}.wav"
        if not p.exists():  # resumable: skip scenes already synthesized
            p.write_bytes(tts.synthesize(text))
        paths.append(p)
    return paths


def build_cues(texts: list[str], durations: list[float], max_chars: int = 28) -> list[Cue]:
    """Split each scene's text into readable chunks and spread them over its duration
    proportionally to length."""
    cues, t = [], 0.0
    for text, dur in zip(texts, durations, strict=True):
        chunks = _split(text, max_chars)
        total = sum(len(c) for c in chunks) or 1
        cursor = t
        for c in chunks:
            span = dur * len(c) / total
            cues.append(Cue(round(cursor, 3), round(cursor + span, 3), c))
            cursor += span
        t += dur
    return cues


def _split(text: str, max_chars: int) -> list[str]:
    parts, buf = [], ""
    for ch in text.strip():
        buf += ch
        if (
            ch in "。！？!?"
            or (len(buf) >= max_chars and ch in "、, ")
            or len(buf) >= max_chars + 8
        ):
            parts.append(buf.strip())
            buf = ""
    if buf.strip():
        parts.append(buf.strip())
    return parts


def to_srt(cues: list[Cue]) -> str:
    def ts(sec: float) -> str:
        ms = round(sec * 1000)
        h, ms = divmod(ms, 3_600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    return "\n".join(
        f"{i}\n{ts(c.start)} --> {ts(c.end)}\n{c.text}\n" for i, c in enumerate(cues, 1)
    )
