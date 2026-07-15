"""Frigate NVR event clip downloader."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import json
from json import JSONDecodeError
from pathlib import Path
import re
import shutil
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urljoin
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class FrigateDownloadStats:
    """Summary of a Frigate event download run."""

    events_seen: int = 0
    downloaded: int = 0
    skipped_existing: int = 0
    skipped_without_clip: int = 0
    failed: int = 0


class FrigateEventDownloader:
    """Download MP4 clips for events discovered from the Frigate events API."""

    def __init__(
        self,
        base_url: str,
        output_dir: str | Path,
        *,
        headers: Mapping[str, str] | None = None,
        page_size: int = 100,
        timeout: float = 30.0,
    ) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be greater than zero")
        self.base_url = base_url.rstrip("/") + "/"
        self.output_dir = Path(output_dir)
        self.headers = dict(headers or {})
        self.page_size = page_size
        self.timeout = timeout

    def download_all(
        self,
        *,
        max_pages: int | None = None,
        include_events_without_clip: bool = False,
        on_event_count: Callable[[int], None] | None = None,
        on_download: Callable[[str, Path], None] | None = None,
    ) -> FrigateDownloadStats:
        """Download every paged event clip, skipping clips already on disk."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        events_seen = 0
        downloaded = 0
        skipped_existing = 0
        skipped_without_clip = 0
        failed = 0

        events = list(self.iter_events(max_pages=max_pages))
        if on_event_count is not None:
            on_event_count(len(events))

        for event in events:
            events_seen += 1
            event_id = str(event.get("id", "")).strip()
            if not event_id:
                failed += 1
                continue
            if event.get("has_clip") is False and not include_events_without_clip:
                skipped_without_clip += 1
                continue
            destination = self.output_dir / f"{_safe_filename(event_id)}.mp4"
            if destination.exists() and destination.stat().st_size > 0:
                skipped_existing += 1
                continue
            try:
                self.download_clip(event_id, destination)
            except (HTTPError, URLError, OSError):
                failed += 1
            else:
                if on_download is not None:
                    on_download(event_id, destination)
                downloaded += 1

        return FrigateDownloadStats(
            events_seen=events_seen,
            downloaded=downloaded,
            skipped_existing=skipped_existing,
            skipped_without_clip=skipped_without_clip,
            failed=failed,
        )

    def iter_events(self, *, max_pages: int | None = None) -> Iterable[dict[str, Any]]:
        """Yield events from Frigate newest-to-oldest using start-time pagination."""

        before: float | None = None
        pages_read = 0
        seen_ids: set[str] = set()
        while max_pages is None or pages_read < max_pages:
            params: dict[str, str | int | float] = {"limit": self.page_size}
            if before is not None:
                params["before"] = before
            events = self._get_json("/api/events", params=params)
            if not isinstance(events, list) or not events:
                return

            pages_read += 1
            next_before = before
            yielded_this_page = False
            for event in events:
                if not isinstance(event, dict):
                    continue
                event_id = str(event.get("id", "")).strip()
                if event_id in seen_ids:
                    continue
                if event_id:
                    seen_ids.add(event_id)
                yielded_this_page = True
                yield event
                start_time = event.get("start_time")
                if isinstance(start_time, int | float):
                    next_before = start_time if next_before is None else min(next_before, start_time)

            if next_before is None or next_before == before or not yielded_this_page:
                return
            before = next_before

    def download_clip(self, event_id: str, destination: Path) -> None:
        """Download one event clip to a destination path using an atomic replace."""

        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_destination = destination.with_suffix(destination.suffix + ".part")
        url = self._url(f"/api/events/{quote(event_id, safe='')}/clip.mp4")
        request = Request(url, headers=self.headers | {"Accept": "video/mp4"})
        with urlopen(request, timeout=self.timeout) as response, tmp_destination.open("wb") as output:
            shutil.copyfileobj(response, output)
        tmp_destination.replace(destination)

    def _get_json(self, path: str, *, params: Mapping[str, str | int | float]) -> Any:
        query = urlencode(params)
        url = self._url(path)
        if query:
            url = f"{url}?{query}"
        request = Request(url, headers=self.headers | {"Accept": "application/json"})
        with urlopen(request, timeout=self.timeout) as response:
            body = response.read()
        try:
            return json.loads(body.decode("utf-8"))
        except (JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"Frigate events endpoint did not return JSON: {url}. "
                "Use the Frigate server root for --base-url, not a UI path like /settings."
            ) from exc

    def _url(self, path: str) -> str:
        return urljoin(self.base_url, path.lstrip("/"))


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "event"
