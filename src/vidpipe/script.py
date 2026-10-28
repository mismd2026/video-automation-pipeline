"""AI script generation.

The LLM must return strict JSON matching ``VideoScript``. Output is validated and,
on failure, the model is asked once more with the validation error attached —
cheaper and more reliable than hoping the first answer parses.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Protocol

import httpx

MAX_TITLE = 100  # YouTube hard limit
MAX_DESCRIPTION = 5000
MAX_TAGS_CHARS = 500


class ChatModel(Protocol):
    def chat_json(self, system: str, user: str) -> str: ...


@dataclass
class OllamaJSON:
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:7b-instruct"
    timeout: float = 180

    def chat_json(self, system: str, user: str) -> str:
        r = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.7},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]


@dataclass
class Scene:
    narration: str
    caption: str
    visual: str = ""  # hint for the background image/clip, e.g. "server room"


@dataclass
class VideoScript:
    title: str
    description: str
    tags: list[str]
    scenes: list[Scene] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> VideoScript:
        return cls(
            title=d["title"],
            description=d["description"],
            tags=list(d.get("tags", [])),
            scenes=[Scene(**s) for s in d["scenes"]],
        )


class ScriptError(ValueError):
    pass


SYSTEM = """あなたはYouTube解説動画の構成作家です。次のJSONだけを出力してください。
{"title": str, "description": str, "tags": [str],
 "scenes": [{"narration": str, "caption": str, "visual": str}]}
- title は32文字以内で、検索されやすい語を前半に置く
- scenes は {n_scenes} 個。narration は読み上げ用の自然な話し言葉で1シーン60〜120文字
- caption は画面に出す要点（20文字以内）
- 事実が不確かな数値・固有名詞は書かない
"""


def validate(d: dict, n_scenes: int) -> VideoScript:
    try:
        script = VideoScript.from_dict(d)
    except (KeyError, TypeError) as e:
        raise ScriptError(f"missing or malformed field: {e}") from e
    problems = []
    if not 0 < len(script.title) <= MAX_TITLE:
        problems.append(f"title must be 1..{MAX_TITLE} chars")
    if len(script.description) > MAX_DESCRIPTION:
        problems.append("description too long")
    if any(c in script.title + script.description for c in "<>"):
        problems.append("'<' and '>' are not allowed by YouTube")
    if len(script.scenes) != n_scenes:
        problems.append(f"expected {n_scenes} scenes, got {len(script.scenes)}")
    if any(not s.narration.strip() for s in script.scenes):
        problems.append("every scene needs narration")
    if problems:
        raise ScriptError("; ".join(problems))
    # trim tags to YouTube's total-length limit instead of failing
    kept, total = [], 0
    for t in script.tags:
        if total + len(t) + 1 > MAX_TAGS_CHARS:
            break
        kept.append(t)
        total += len(t) + 1
    script.tags = kept
    return script


def generate_script(
    llm: ChatModel, topic: str, n_scenes: int = 5, context: str = "", retries: int = 1
) -> VideoScript:
    system = SYSTEM.replace("{n_scenes}", str(n_scenes))
    user = f"テーマ: {topic}\n参考（伸びている競合動画のタイトル）:\n{context}".strip()
    last_error = ""
    for _ in range(retries + 1):
        prompt = (
            user if not last_error else f"{user}\n\n前回の出力の問題: {last_error}\n修正して再出力"
        )
        raw = llm.chat_json(system, prompt)
        try:
            return validate(json.loads(raw), n_scenes)
        except (json.JSONDecodeError, ScriptError) as e:
            last_error = str(e)
    raise ScriptError(f"LLM output invalid after {retries + 1} attempts: {last_error}")
