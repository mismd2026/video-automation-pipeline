"""YouTube upload via the Data API v3 resumable protocol.

Implemented directly on httpx (no google-api-python-client) to keep the
dependency surface small and to make retry behaviour explicit:

  1. Exchange the stored OAuth refresh token for an access token.
  2. POST metadata -> receive a session URI.
  3. PUT the file in chunks; on 5xx / network errors, ask the server how many
     bytes it has (``Content-Range: bytes */N``) and resume from there.

Videos are uploaded as ``private`` by default (optionally with ``publishAt``),
so nothing goes public without a human check.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx

TOKEN_URL = "https://oauth2.googleapis.com/token"
UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
THUMB_URL = "https://www.googleapis.com/upload/youtube/v3/thumbnails/set"
CHUNK = 8 * 1024 * 1024  # must be a multiple of 256 KiB
RETRYABLE = {500, 502, 503, 504}


@dataclass(frozen=True)
class OAuthCredentials:
    client_id: str
    client_secret: str
    refresh_token: str


@dataclass(frozen=True)
class UploadMeta:
    title: str
    description: str
    tags: list[str]
    category_id: str = "28"  # Science & Technology
    privacy: str = "private"
    publish_at: datetime | None = None  # requires privacy=private
    language: str = "ja"

    def body(self) -> dict:
        status: dict = {"privacyStatus": self.privacy, "selfDeclaredMadeForKids": False}
        if self.publish_at:
            status["privacyStatus"] = "private"
            status["publishAt"] = self.publish_at.isoformat()
        return {
            "snippet": {
                "title": self.title,
                "description": self.description,
                "tags": self.tags,
                "categoryId": self.category_id,
                "defaultLanguage": self.language,
                "defaultAudioLanguage": self.language,
            },
            "status": status,
        }


class UploadError(RuntimeError):
    pass


class YouTubeUploader:
    def __init__(
        self,
        creds: OAuthCredentials,
        client: httpx.Client | None = None,
        max_retries: int = 5,
        sleep=time.sleep,
    ) -> None:
        self._creds = creds
        self._client = client or httpx.Client(timeout=120)
        self._max_retries = max_retries
        self._sleep = sleep

    def _access_token(self) -> str:
        r = self._client.post(
            TOKEN_URL,
            data={
                "client_id": self._creds.client_id,
                "client_secret": self._creds.client_secret,
                "refresh_token": self._creds.refresh_token,
                "grant_type": "refresh_token",
            },
        )
        r.raise_for_status()
        return r.json()["access_token"]

    def upload(self, video: Path, meta: UploadMeta, thumbnail: Path | None = None) -> str:
        token = self._access_token()
        auth = {"Authorization": f"Bearer {token}"}
        size = video.stat().st_size

        init = self._client.post(
            UPLOAD_URL,
            params={"uploadType": "resumable", "part": "snippet,status"},
            headers={
                **auth,
                "X-Upload-Content-Type": "video/mp4",
                "X-Upload-Content-Length": str(size),
            },
            json=meta.body(),
        )
        if init.status_code != 200:
            raise UploadError(f"session init failed: {init.status_code} {init.text[:300]}")
        session = init.headers["Location"]

        video_id = self._send_chunks(session, video, size, auth)
        if thumbnail:
            r = self._client.post(
                THUMB_URL,
                params={"videoId": video_id},
                headers={**auth, "Content-Type": "image/jpeg"},
                content=thumbnail.read_bytes(),
            )
            r.raise_for_status()
        return video_id

    def _send_chunks(self, session: str, video: Path, size: int, auth: dict) -> str:
        offset, failures = 0, 0
        with video.open("rb") as f:
            while True:
                f.seek(offset)
                data = f.read(CHUNK)
                end = offset + len(data) - 1
                try:
                    r = self._client.put(
                        session,
                        content=data,
                        headers={
                            **auth,
                            "Content-Range": f"bytes {offset}-{end}/{size}",
                        },
                    )
                except httpx.TransportError:
                    r = None

                if r is not None and r.status_code in (200, 201):
                    return r.json()["id"]
                if r is not None and r.status_code == 308:  # chunk accepted, continue
                    offset = _next_offset(r)
                    failures = 0
                    continue
                if r is not None and r.status_code not in RETRYABLE:
                    raise UploadError(f"upload failed: {r.status_code} {r.text[:300]}")

                failures += 1
                if failures > self._max_retries:
                    raise UploadError("upload failed: retries exhausted")
                self._sleep(min(2**failures, 60))  # exponential backoff
                offset = self._query_offset(session, size, auth)

    def _query_offset(self, session: str, size: int, auth: dict) -> int:
        r = self._client.put(session, headers={**auth, "Content-Range": f"bytes */{size}"})
        return _next_offset(r) if r.status_code == 308 else 0


def _next_offset(resp: httpx.Response) -> int:
    rng = resp.headers.get("Range")  # e.g. "bytes=0-8388607"
    return int(rng.split("-")[1]) + 1 if rng else 0
