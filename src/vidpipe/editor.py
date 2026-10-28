"""Video assembly with ffmpeg.

Every step is a pure function that returns an ffmpeg argument list, and a
single ``run`` executes it. That keeps the filter graphs unit-testable and
lets ``--dry-run`` print the exact commands for review.

Stages:
  1. scene clips   background (image or solid colour) + scene caption + narration
  2. concat        stream-copy the clips into one file (no re-encode)
  3. finish        burn subtitles, duck BGM under the voice, normalise to -14 LUFS
  4. thumbnail     frame grab + title text
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

W, H, FPS = 1920, 1080, 30
DEFAULT_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


@dataclass(frozen=True)
class Style:
    font: str = DEFAULT_FONT
    font_name: str = "Noto Sans CJK JP"
    background: str = "0x1e293b"
    caption_size: int = 64
    subtitle_size: int = 22  # libass units, scaled to video height
    target_lufs: float = -14.0  # YouTube's loudness reference
    bgm_volume: float = 0.18


DEFAULT_STYLE = Style()  # frozen, so sharing one instance as a default is safe


def _esc(text: str) -> str:
    """Escape text for drawtext's text= option."""
    for ch in ("\\", ":", "'", "%", ","):
        text = text.replace(ch, "\\" + ch)
    return text


def scene_clip_cmd(
    narration: Path,
    out: Path,
    duration: float,
    caption: str,
    background: Path | None = None,
    style: Style = DEFAULT_STYLE,
) -> list[str]:
    if background:
        bg_input = ["-loop", "1", "-framerate", str(FPS), "-i", str(background)]
        fit = (
            f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
            # slow Ken Burns zoom so still images do not look frozen
            f"zoompan=z='min(zoom+0.0006,1.08)':d=1:s={W}x{H}:fps={FPS},"
        )
    else:
        bg_input = ["-f", "lavfi", "-i", f"color=c={style.background}:s={W}x{H}:r={FPS}"]
        fit = ""
    vf = (
        f"[0:v]{fit}format=yuv420p,"
        f"drawbox=x=0:y=60:w=iw:h=130:color=black@0.45:t=fill,"
        f"drawtext=fontfile='{style.font}':text='{_esc(caption)}':"
        f"fontsize={style.caption_size}:fontcolor=white:x=(w-text_w)/2:y=90[v]"
    )
    return [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        *bg_input,
        "-i",
        str(narration),
        "-filter_complex",
        vf,
        "-map",
        "[v]",
        "-map",
        "1:a",
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-r",
        str(FPS),
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-ac",
        "2",
        str(out),
    ]


def concat_cmd(clips: list[Path], list_file: Path, out: Path) -> list[str]:
    list_file.write_text("".join(f"file '{c.resolve()}'\n" for c in clips), encoding="utf-8")
    return [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c",
        "copy",
        str(out),
    ]


def finish_cmd(
    video: Path, srt_name: str, out: Path, bgm: Path | None = None, style: Style = DEFAULT_STYLE
) -> list[str]:
    """srt_name must be a bare filename in the working directory (avoids libass path escaping)."""
    sub = (
        f"subtitles={srt_name}:force_style='FontName={style.font_name},"
        f"Fontsize={style.subtitle_size},Outline=2,MarginV=40'"
    )
    loud = f"loudnorm=I={style.target_lufs}:TP=-1.5:LRA=11"
    if bgm:
        # sidechaincompress: BGM volume dips automatically whenever the narrator speaks
        graph = (
            f"[0:v]{sub}[v];"
            f"[0:a]asplit=2[voice][key];"
            f"[1:a]volume={style.bgm_volume},aloop=loop=-1:size=2e9[bgm];"
            f"[bgm][key]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=400[duck];"
            f"[voice][duck]amix=inputs=2:duration=first:normalize=0,{loud}[a]"
        )
        inputs = ["-i", str(video), "-i", str(bgm)]
    else:
        graph = f"[0:v]{sub}[v];[0:a]{loud}[a]"
        inputs = ["-i", str(video)]
    return [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        *inputs,
        "-filter_complex",
        graph,
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        str(out),
    ]


def wrap_title(title: str, max_line: int = 14) -> list[str]:
    """Split a title into at most two lines, preferring a natural break (：、 space)."""
    title = title.strip()
    if len(title) <= max_line:
        return [title]
    mid = len(title) // 2

    def ok(i: int) -> bool:  # never split inside an ASCII word or number ("1|0", "LL|M")
        a, b = title[i - 1], title[i]
        return not (a.isascii() and a.isalnum() and b.isascii() and b.isalnum())

    # 1st choice: punctuation; 2nd: right after a particle; else nearest safe point
    for chars in ("：:、 ｜|-", "をでのとにはがへ"):
        breaks = [i + 1 for i, ch in enumerate(title[:-1]) if ch in chars]
        breaks = [i for i in breaks if abs(i - mid) <= len(title) // 3]
        if breaks:
            cut = min(breaks, key=lambda i: abs(i - mid))
            break
    else:
        safe = [i for i in range(1, len(title)) if ok(i)] or [mid]  # all-ASCII: hard split
        cut = min(safe, key=lambda i: abs(i - mid))
    first, second = title[:cut].strip(" ：:｜|-"), title[cut:].strip()
    return [first, second[: max_line + 4]]


def thumbnail_cmd(
    video: Path, title: str, out: Path, at: float = 1.0, style: Style = DEFAULT_STYLE
) -> list[str]:
    lines = wrap_title(title)
    longest = max(len(line) for line in lines)
    size = max(48, min(96, int(1180 / max(longest, 1))))  # ~1 em per CJK glyph
    y0 = 720 * 0.60 if len(lines) == 2 else 720 * 0.68
    draws = ",".join(
        f"drawtext=fontfile='{style.font}':text='{_esc(line)}':fontsize={size}:"
        f"fontcolor=yellow:borderw=5:bordercolor=black:x=(w-text_w)/2:y={y0 + i * size * 1.15:.0f}"
        for i, line in enumerate(lines)
    )
    vf = f"scale=1280:720,drawbox=x=0:y=ih*0.55:w=iw:h=ih*0.45:color=black@0.6:t=fill,{draws}"
    return [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{at}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-vf",
        vf,
        str(out),
    ]


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout.strip())


class FFmpegError(RuntimeError):
    pass


def run(cmd: list[str], cwd: Path | None = None, dry_run: bool = False) -> None:
    if dry_run:
        print(shlex.join(cmd))
        return
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(f"ffmpeg failed ({proc.returncode}): {proc.stderr.strip()[-2000:]}")
    # ffmpeg can exit 0 yet write nothing (e.g. seeking past the end): treat as failure
    out = Path(cmd[-1])
    if cwd and not out.is_absolute():
        out = cwd / out
    if not out.exists() or out.stat().st_size == 0:
        raise FFmpegError(f"ffmpeg produced no output: {out}")
