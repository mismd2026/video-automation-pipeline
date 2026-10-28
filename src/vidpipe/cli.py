"""Command line entrypoint.

    vidpipe research "生成AI"                     # trending sub-topics + top videos
    vidpipe make  --topic "ローカルLLM入門"        # script -> narration -> render
    vidpipe make  --job jobs/xxx --from render    # re-render after editing script.json
    vidpipe run   --seed "生成AI" --upload        # research -> pick top topic -> make -> upload

Secrets come from environment variables only (see .env.example).
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from .narration import Voicevox
from .pipeline import STEPS, Job
from .research import YouTubeResearch, extract_topics, rank_videos
from .script import OllamaJSON


def _env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        sys.exit(f"environment variable {name} is required")
    return val


def _slug(text: str) -> str:
    return re.sub(r"[^\w-]+", "-", text).strip("-")[:40] or "job"


def _uploader():
    from .uploader import OAuthCredentials, YouTubeUploader

    return YouTubeUploader(
        OAuthCredentials(_env("YT_CLIENT_ID"), _env("YT_CLIENT_SECRET"), _env("YT_REFRESH_TOKEN"))
    )


def cmd_research(args) -> list:
    yt = YouTubeResearch(_env("YOUTUBE_API_KEY"))
    videos = rank_videos(yt.recent_videos(args.seed, days=args.days))
    now = datetime.now(UTC)
    print(f"== top videos for '{args.seed}' (last {args.days} days) ==")
    for v in videos[:10]:
        print(f"{v.score(now):6.2f}  {v.velocity(now):8.0f} views/h  {v.title}  [{v.channel}]")
    topics = extract_topics(videos, args.seed)
    print("\n== trending terms ==")
    for t in topics:
        print(f"{t.score:7.2f}  {t.term}  ({t.videos} videos)")
    return [videos, topics]


def _make(topic: str, job_dir: Path, args, context: str = "") -> None:
    job = Job(
        dir=job_dir,
        topic=topic,
        n_scenes=args.scenes,
        context=context,
        background=Path(args.background) if args.background else None,
        bgm=Path(args.bgm) if args.bgm else None,
    )
    publish_at = datetime.fromisoformat(args.publish_at) if args.publish_at else None
    m = job.run(
        llm=OllamaJSON(
            os.environ.get("OLLAMA_URL", "http://localhost:11434"),
            os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct"),
        ),
        tts=Voicevox(
            os.environ.get("VOICEVOX_URL", "http://localhost:50021"),
            int(os.environ.get("VOICEVOX_SPEAKER", "3")),
        ),
        uploader=_uploader() if args.upload else None,
        upload_meta_overrides={"privacy": args.privacy, "publish_at": publish_at},
        start_from=args.start_from,
        dry_run=args.dry_run,
    )
    print(f"job: {job_dir}")
    for step, info in m.data["steps"].items():
        print(f"  {step:10s} {info['status']}  {info.get('url', '')}")


def cmd_make(args) -> None:
    if args.job:
        job_dir = Path(args.job)
        topic = args.topic or ""
    else:
        if not args.topic:
            sys.exit("--topic or --job is required")
        topic = args.topic
        job_dir = Path(args.jobs_dir) / f"{datetime.now():%Y%m%d-%H%M}-{_slug(topic)}"
    _make(topic, job_dir, args)


def cmd_run(args) -> None:
    videos, topics = cmd_research(args)
    if not topics:
        sys.exit("no trending topic found")
    topic = f"{args.seed} {topics[0].term}"
    context = "\n".join(v.title for v in videos[:5])
    job_dir = Path(args.jobs_dir) / f"{datetime.now():%Y%m%d-%H%M}-{_slug(topic)}"
    _make(topic, job_dir, args, context)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="vidpipe")
    sub = p.add_subparsers(dest="cmd", required=True)

    def make_opts(sp):
        sp.add_argument("--scenes", type=int, default=5)
        sp.add_argument("--background", help="background image for all scenes")
        sp.add_argument("--bgm", help="background music file (auto-ducked)")
        sp.add_argument("--upload", action="store_true")
        sp.add_argument("--privacy", default="private", choices=["private", "unlisted", "public"])
        sp.add_argument("--publish-at", help="ISO8601, e.g. 2026-11-01T19:00:00+09:00")
        sp.add_argument("--from", dest="start_from", choices=STEPS, help="redo from this step")
        sp.add_argument("--jobs-dir", default="jobs")
        sp.add_argument("--dry-run", action="store_true", help="print ffmpeg commands only")

    r = sub.add_parser("research")
    r.add_argument("seed")
    r.add_argument("--days", type=int, default=14)
    r.set_defaults(func=cmd_research)

    m = sub.add_parser("make")
    m.add_argument("--topic")
    m.add_argument("--job", help="existing job directory to resume")
    make_opts(m)
    m.set_defaults(func=cmd_make)

    a = sub.add_parser("run")
    a.add_argument("--seed", required=True)
    a.add_argument("--days", type=int, default=14)
    make_opts(a)
    a.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
