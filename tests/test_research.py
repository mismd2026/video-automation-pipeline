from datetime import UTC, datetime, timedelta

import httpx

from vidpipe.research import VideoStat, YouTubeResearch, extract_topics, rank_videos

NOW = datetime(2026, 10, 28, 12, tzinfo=UTC)


def _v(i, title, hours_ago, views, likes=0, comments=0):
    return VideoStat(str(i), title, "ch", NOW - timedelta(hours=hours_ago), views, likes, comments)


def test_velocity_beats_raw_views():
    fresh = _v(1, "new", hours_ago=10, views=20_000)  # 2000 views/h
    old = _v(2, "old", hours_ago=24 * 300, views=500_000)  # ~69 views/h
    assert rank_videos([old, fresh], NOW)[0].id == "1"


def test_engagement_breaks_ties():
    a = _v(1, "a", 10, 10_000, likes=100)
    b = _v(2, "b", 10, 10_000, likes=900, comments=50)
    assert rank_videos([a, b], NOW)[0].id == "2"


def test_extract_topics_requires_recurrence_and_drops_seed_and_stopwords():
    vids = [
        _v(1, "生成AI ローカルLLM 徹底解説", 5, 50_000),
        _v(2, "ローカルLLM を Ollama で動かす", 8, 30_000),
        _v(3, "生成AI Ollama 入門", 20, 10_000),
        _v(4, "生成AI 料理レシピ", 20, 100),
    ]
    terms = [t.term for t in extract_topics(vids, seed="生成AI", now=NOW)]
    assert "ローカルllm" in terms or "ローカル" in terms or "llm" in terms
    assert "ollama" in terms
    assert "料理" not in terms  # appears once only
    assert "徹底" not in terms and "生成" not in terms


def test_api_client_uses_two_calls_and_parses():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.url.path)
        assert req.url.params["key"] == "k"
        if req.url.path.endswith("/search"):
            return httpx.Response(200, json={"items": [{"id": {"videoId": "abc"}}]})
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": "abc",
                        "snippet": {
                            "title": "t",
                            "channelTitle": "c",
                            "publishedAt": "2026-10-27T00:00:00Z",
                        },
                        "statistics": {"viewCount": "1200", "commentCount": "4"},  # likes hidden
                    }
                ]
            },
        )

    yt = YouTubeResearch("k", httpx.Client(transport=httpx.MockTransport(handler)))
    vids = yt.recent_videos("生成AI")
    assert seen == ["/youtube/v3/search", "/youtube/v3/videos"]
    assert vids[0].views == 1200 and vids[0].likes == 0
