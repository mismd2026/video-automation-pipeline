from datetime import datetime, timedelta, timezone

import httpx
import pytest

import vidpipe.uploader as up
from vidpipe.uploader import OAuthCredentials, UploadError, UploadMeta, YouTubeUploader

SESSION = "https://upload.example/session/1"
JST = timezone(timedelta(hours=9))


class FakeYouTube:
    """Simulates the resumable-upload server, including a transient 503."""

    def __init__(self, size: int, fail_once_at: int | None = None) -> None:
        self.size = size
        self.received = bytearray()
        self.fail_once_at = fail_once_at
        self.init_body = None
        self.log: list[str] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        if req.url.host == "oauth2.googleapis.com":
            return httpx.Response(200, json={"access_token": "tok"})
        assert req.headers["Authorization"] == "Bearer tok"
        if req.method == "POST" and "thumbnails" in req.url.path:
            self.log.append("thumb")
            return httpx.Response(200, json={})
        if req.method == "POST":
            self.init_body = req.read()
            return httpx.Response(200, headers={"Location": SESSION})

        rng = req.headers["Content-Range"]
        if rng.startswith("bytes */"):  # status query after a failure
            self.log.append("query")
            return httpx.Response(308, headers={"Range": f"bytes=0-{len(self.received) - 1}"})
        start = int(rng.split()[1].split("-")[0])
        if self.fail_once_at is not None and start == self.fail_once_at:
            self.fail_once_at = None
            self.log.append("503")
            return httpx.Response(503)
        assert start == len(self.received), "client must resume from server offset"
        self.received += req.read()
        self.log.append(f"chunk@{start}")
        if len(self.received) == self.size:
            return httpx.Response(200, json={"id": "VID123"})
        return httpx.Response(308, headers={"Range": f"bytes=0-{len(self.received) - 1}"})


@pytest.fixture
def video(tmp_path, monkeypatch):
    monkeypatch.setattr(up, "CHUNK", 256 * 1024)
    p = tmp_path / "v.mp4"
    p.write_bytes(bytes(range(256)) * 4096)  # 1 MiB -> 4 chunks
    return p


def _uploader(server):
    return YouTubeUploader(
        OAuthCredentials("id", "secret", "refresh"),
        client=httpx.Client(transport=httpx.MockTransport(server)),
        sleep=lambda _s: None,
    )


def test_chunked_upload_resumes_after_503(video, tmp_path):
    server = FakeYouTube(video.stat().st_size, fail_once_at=512 * 1024)
    thumb = tmp_path / "t.jpg"
    thumb.write_bytes(b"\xff\xd8jpeg")
    vid = _uploader(server).upload(video, UploadMeta("t", "d", ["a"]), thumb)
    assert vid == "VID123"
    assert bytes(server.received) == video.read_bytes()
    assert server.log[:4] == ["chunk@0", "chunk@262144", "503", "query"]
    assert server.log[-1] == "thumb"


def test_default_is_private_and_schedule_forces_private():
    assert UploadMeta("t", "d", []).body()["status"]["privacyStatus"] == "private"
    when = datetime(2026, 11, 1, 19, tzinfo=JST)
    body = UploadMeta("t", "d", [], privacy="public", publish_at=when).body()
    assert body["status"] == {
        "privacyStatus": "private",
        "selfDeclaredMadeForKids": False,
        "publishAt": "2026-11-01T19:00:00+09:00",
    }


def test_non_retryable_error_raises(video):
    def server(req):
        if req.url.host == "oauth2.googleapis.com":
            return httpx.Response(200, json={"access_token": "tok"})
        if req.method == "POST":
            return httpx.Response(200, headers={"Location": SESSION})
        return httpx.Response(403, text="quotaExceeded")

    with pytest.raises(UploadError, match="403"):
        _uploader(server).upload(video, UploadMeta("t", "d", []))
