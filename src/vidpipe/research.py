"""Trend and competitor research via the YouTube Data API v3.

For a seed keyword, fetch recent videos, score them by *velocity*
(views per hour since publish, weighted by engagement) rather than raw views,
so that a 2-day-old video with 50k views outranks a 3-year-old one with 500k.
Title terms that recur across the fastest-growing videos become topic candidates.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx

API = "https://www.googleapis.com/youtube/v3"

# Japanese/English filler that should never become a "topic"
STOPWORDS = {
    "の",
    "は",
    "が",
    "を",
    "に",
    "で",
    "と",
    "も",
    "する",
    "した",
    "こと",
    "これ",
    "それ",
    "方法",
    "解説",
    "紹介",
    "まとめ",
    "最新",
    "徹底",
    "完全",
    "初心者",
    "おすすめ",
    "版",
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "for",
    "how",
    "with",
    "is",
    "you",
}
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#.\-]{1,}|[ァ-ヴー]{2,}|[一-龠々]{2,}")


@dataclass(frozen=True)
class VideoStat:
    id: str
    title: str
    channel: str
    published_at: datetime
    views: int
    likes: int
    comments: int

    def velocity(self, now: datetime) -> float:
        hours = max((now - self.published_at).total_seconds() / 3600, 1.0)
        return self.views / hours

    def engagement(self) -> float:
        return (self.likes + 3 * self.comments) / max(self.views, 1)

    def score(self, now: datetime) -> float:
        # log keeps a single viral outlier from dominating the ranking
        return math.log1p(self.velocity(now)) * (1 + 10 * self.engagement())


@dataclass(frozen=True)
class Topic:
    term: str
    videos: int
    score: float


class YouTubeResearch:
    def __init__(self, api_key: str, client: httpx.Client | None = None) -> None:
        self._key = api_key
        self._client = client or httpx.Client(timeout=30)

    def _get(self, path: str, **params) -> dict:
        resp = self._client.get(f"{API}/{path}", params={**params, "key": self._key})
        resp.raise_for_status()
        return resp.json()

    def recent_videos(
        self, query: str, days: int = 14, max_results: int = 50, region: str = "JP"
    ) -> list[VideoStat]:
        since = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        search = self._get(
            "search",
            part="id",
            q=query,
            type="video",
            order="viewCount",
            publishedAfter=since,
            maxResults=min(max_results, 50),
            regionCode=region,
            relevanceLanguage="ja",
        )
        ids = [item["id"]["videoId"] for item in search.get("items", [])]
        if not ids:
            return []
        # search.list costs 100 quota units; videos.list costs 1 per call of up to 50 ids
        detail = self._get("videos", part="snippet,statistics", id=",".join(ids))
        return [_to_stat(v) for v in detail.get("items", [])]


def _to_stat(v: dict) -> VideoStat:
    s = v.get("statistics", {})
    return VideoStat(
        id=v["id"],
        title=v["snippet"]["title"],
        channel=v["snippet"]["channelTitle"],
        published_at=datetime.fromisoformat(v["snippet"]["publishedAt"].replace("Z", "+00:00")),
        views=int(s.get("viewCount", 0)),
        likes=int(s.get("likeCount", 0)),  # hidden like counts -> 0
        comments=int(s.get("commentCount", 0)),
    )


def rank_videos(videos: list[VideoStat], now: datetime | None = None) -> list[VideoStat]:
    now = now or datetime.now(UTC)
    return sorted(videos, key=lambda v: v.score(now), reverse=True)


def extract_topics(
    videos: list[VideoStat], seed: str, top_n: int = 10, now: datetime | None = None
) -> list[Topic]:
    """Terms that recur in titles of fast-growing videos, weighted by video score."""
    now = now or datetime.now(UTC)
    seed_terms = {t.lower() for t in TOKEN_RE.findall(seed)}
    weight: Counter[str] = Counter()
    count: Counter[str] = Counter()
    for v in videos:
        terms = {t.lower() for t in TOKEN_RE.findall(v.title)} - STOPWORDS - seed_terms
        for t in terms:
            weight[t] += v.score(now)
            count[t] += 1
    # a term must appear in at least 2 videos to count as a trend, not a one-off
    ranked = [Topic(t, count[t], round(w, 2)) for t, w in weight.items() if count[t] >= 2]
    return sorted(ranked, key=lambda t: t.score, reverse=True)[:top_n]
