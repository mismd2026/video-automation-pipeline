from __future__ import annotations

import io
import json
import math
import shutil
import struct
import wave

import pytest

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def make_wav(seconds: float, rate: int = 24000, freq: float = 440.0) -> bytes:
    """Mono 16-bit sine tone — a stand-in for VOICEVOX output."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", int(8000 * math.sin(2 * math.pi * freq * i / rate)))
            for i in range(int(seconds * rate))
        )
        w.writeframes(frames)
    return buf.getvalue()


class FakeTTS:
    """Duration proportional to text length, like a real TTS."""

    def __init__(self) -> None:
        self.calls = 0

    def synthesize(self, text: str) -> bytes:
        self.calls += 1
        return make_wav(min(0.05 * len(text), 2.0))


SAMPLE_SCRIPT = {
    "title": "ローカルLLM入門：Ollamaで社内AIを動かす",
    "description": "クラウドに出さずにLLMを動かす方法を3分で解説します。",
    "tags": ["Ollama", "ローカルLLM", "生成AI"],
    "scenes": [
        {
            "narration": "今日はローカルLLMについて解説します。",
            "caption": "ローカルLLMとは",
            "visual": "",
        },
        {
            "narration": "Ollamaを使えば、コマンド一つでモデルを動かせます。",
            "caption": "Ollamaで簡単導入",
            "visual": "",
        },
        {
            "narration": "データが社外に出ないのが最大の利点です。",
            "caption": "データは社内に",
            "visual": "",
        },
    ],
}


class ScriptedLLM:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def chat_json(self, system: str, user: str) -> str:
        self.prompts.append(user)
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


@pytest.fixture
def good_llm() -> ScriptedLLM:
    return ScriptedLLM(json.dumps(SAMPLE_SCRIPT, ensure_ascii=False))
