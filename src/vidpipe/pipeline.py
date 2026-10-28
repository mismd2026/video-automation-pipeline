"""End-to-end job orchestration with a resumable manifest.

Each job lives in its own directory::

    jobs/<job-id>/
      manifest.json      step status + outputs (the source of truth for resume)
      script.json        generated script (edit by hand before rendering if you like)
      audio/scene_*.wav  narration
      clips/scene_*.mp4  per-scene clips
      subs.srt           subtitles
      final.mp4          upload-ready video
      thumb.jpg          thumbnail

Re-running a job skips completed steps, so a failure at upload time does not
re-spend LLM/TTS/encode time, and a human can edit script.json and re-render
only the downstream steps with ``--from render``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from . import editor
from .narration import TTS, build_cues, synthesize_scenes, to_srt, wav_duration
from .script import ChatModel, VideoScript, generate_script

log = logging.getLogger("vidpipe")

STEPS = ["script", "narration", "render", "upload"]


@dataclass
class Manifest:
    path: Path
    data: dict = field(default_factory=dict)

    @classmethod
    def load(cls, job_dir: Path) -> Manifest:
        p = job_dir / "manifest.json"
        data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"steps": {}}
        return cls(p, data)

    def done(self, step: str) -> bool:
        return self.data["steps"].get(step, {}).get("status") == "done"

    def mark(self, step: str, **outputs) -> None:
        self.data["steps"][step] = {
            "status": "done",
            "at": datetime.now(UTC).isoformat(),
            **outputs,
        }
        self.save()

    def reset_from(self, step: str) -> None:
        for s in STEPS[STEPS.index(step) :]:
            self.data["steps"].pop(s, None)
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)  # atomic: a crash never leaves a half-written manifest


@dataclass
class Job:
    dir: Path
    topic: str
    n_scenes: int = 5
    context: str = ""
    background: Path | None = None
    bgm: Path | None = None
    style: editor.Style = field(default_factory=editor.Style)

    def run(
        self,
        llm: ChatModel,
        tts: TTS,
        uploader=None,
        upload_meta_overrides: dict | None = None,
        start_from: str | None = None,
        dry_run: bool = False,
    ) -> Manifest:
        self.dir.mkdir(parents=True, exist_ok=True)
        m = Manifest.load(self.dir)
        m.data.setdefault("topic", self.topic)
        if start_from:
            m.reset_from(start_from)

        script_path = self.dir / "script.json"
        if not m.done("script"):
            log.info("[%s] generating script", self.dir.name)
            script = generate_script(llm, self.topic, self.n_scenes, self.context)
            script_path.write_text(script.to_json(), encoding="utf-8")
            m.mark("script", file=script_path.name, title=script.title)
        script = VideoScript.from_dict(json.loads(script_path.read_text(encoding="utf-8")))

        audio_dir = self.dir / "audio"
        if not m.done("narration"):
            log.info("[%s] synthesizing %d scenes", self.dir.name, len(script.scenes))
            wavs = synthesize_scenes(tts, [s.narration for s in script.scenes], audio_dir)
            m.mark("narration", durations=[round(wav_duration(w), 3) for w in wavs])
        durations = m.data["steps"]["narration"]["durations"]

        final, thumb = self.dir / "final.mp4", self.dir / "thumb.jpg"
        if not m.done("render"):
            log.info("[%s] rendering", self.dir.name)
            self._render(script, audio_dir, durations, final, thumb, dry_run)
            if dry_run:
                return m
            m.mark("render", video=final.name, seconds=round(editor.probe_duration(final), 2))

        if uploader and not m.done("upload"):
            from .uploader import UploadMeta

            meta = UploadMeta(
                script.title, script.description, script.tags, **(upload_meta_overrides or {})
            )
            log.info("[%s] uploading (%s)", self.dir.name, meta.privacy)
            video_id = uploader.upload(final, meta, thumb)
            m.mark("upload", video_id=video_id, url=f"https://youtu.be/{video_id}")
        return m

    def _render(self, script, audio_dir, durations, final, thumb, dry_run) -> None:
        clips_dir = self.dir / "clips"
        clips_dir.mkdir(exist_ok=True)
        clips = []
        for i, (scene, dur) in enumerate(zip(script.scenes, durations, strict=True)):
            clip = clips_dir / f"scene_{i:02d}.mp4"
            editor.run(
                editor.scene_clip_cmd(
                    audio_dir / f"scene_{i:02d}.wav",
                    clip,
                    dur,
                    scene.caption,
                    self.background,
                    self.style,
                ),
                dry_run=dry_run,
            )
            clips.append(clip)

        cues = build_cues([s.narration for s in script.scenes], durations)
        (self.dir / "subs.srt").write_text(to_srt(cues), encoding="utf-8")

        joined = self.dir / "joined.mp4"
        editor.run(editor.concat_cmd(clips, self.dir / "clips.txt", joined), dry_run=dry_run)
        editor.run(
            editor.finish_cmd(
                joined.resolve(),
                "subs.srt",
                final.resolve(),
                self.bgm.resolve() if self.bgm else None,
                self.style,
            ),
            cwd=self.dir,
            dry_run=dry_run,
        )
        # first scene clip has the caption but no subtitles -> cleaner thumbnail
        editor.run(
            editor.thumbnail_cmd(
                clips[0], script.title, thumb, at=min(1.0, durations[0] / 2), style=self.style
            ),
            dry_run=dry_run,
        )
        if not dry_run:
            joined.unlink(missing_ok=True)
