"""End-to-end: real ffmpeg renders a real MP4 from fake LLM/TTS output."""

import json
import subprocess

import pytest
from conftest import FakeTTS, make_wav, needs_ffmpeg

from vidpipe import editor
from vidpipe.pipeline import Job, Manifest


class FakeUploader:
    def __init__(self):
        self.calls = []

    def upload(self, video, meta, thumbnail=None):
        self.calls.append((video.name, meta.title, meta.privacy, thumbnail.name))
        return "VID999"


@needs_ffmpeg
def test_full_job_renders_and_resumes(tmp_path, good_llm):
    bgm = tmp_path / "bgm.wav"
    bgm.write_bytes(make_wav(1.0, freq=220))
    tts = FakeTTS()
    job = Job(dir=tmp_path / "job", topic="ローカルLLM", n_scenes=3, bgm=bgm)

    m = job.run(good_llm, tts)
    final = job.dir / "final.mp4"
    assert final.exists() and (job.dir / "thumb.jpg").exists()

    durations = m.data["steps"]["narration"]["durations"]
    assert editor.probe_duration(final) == pytest.approx(sum(durations), abs=0.3)
    streams = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,width,height",
            "-of",
            "json",
            str(final),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(streams.stdout)["streams"]
    assert {s["codec_type"] for s in info} == {"video", "audio"}
    assert any(s.get("width") == 1920 and s.get("height") == 1080 for s in info)
    assert "00:00:00,000" in (job.dir / "subs.srt").read_text(encoding="utf-8")

    # Second run with an uploader: nothing is regenerated, only the upload happens.
    uploader = FakeUploader()
    m2 = job.run(good_llm, tts, uploader=uploader)
    assert tts.calls == 3 and len(good_llm.prompts) == 1
    assert uploader.calls == [
        ("final.mp4", m2.data["steps"]["script"]["title"], "private", "thumb.jpg")
    ]
    assert Manifest.load(job.dir).data["steps"]["upload"]["url"] == "https://youtu.be/VID999"


def test_dry_run_prints_commands_without_rendering(tmp_path, good_llm, capsys):
    job = Job(dir=tmp_path / "job", topic="t", n_scenes=3)
    job.run(good_llm, FakeTTS(), dry_run=True)
    out = capsys.readouterr().out
    assert out.count("ffmpeg ") == 3 + 3  # 3 scene clips + concat + finish + thumbnail
    assert "loudnorm=I=-14.0" in out
    assert not (job.dir / "final.mp4").exists()


def test_drawtext_escaping():
    assert editor._esc("50%: it's") == "50\\%\\: it\\'s"


@needs_ffmpeg
def test_run_raises_when_ffmpeg_writes_nothing(tmp_path):
    clip = tmp_path / "c.mp4"
    editor.run(editor.scene_clip_cmd(_wav(tmp_path), clip, 0.5, "x"))
    with pytest.raises(editor.FFmpegError, match="no output"):
        editor.run(editor.thumbnail_cmd(clip, "t", tmp_path / "t.jpg", at=30))


def _wav(tmp_path):
    p = tmp_path / "a.wav"
    p.write_bytes(make_wav(0.5))
    return p
