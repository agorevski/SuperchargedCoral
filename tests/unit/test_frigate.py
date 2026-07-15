from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from threading import Lock, Thread
from time import sleep
from urllib.parse import parse_qs, urlparse

import pytest

from supercharged_coral.integrations.frigate import FrigateEventDownloader


class FrigateHandler(BaseHTTPRequestHandler):
    downloaded_paths: list[str] = []

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/events":
            query = parse_qs(parsed.query)
            before = query.get("before", [None])[0]
            if before is None:
                self._send_json(
                    [
                        {"id": "camera/three", "start_time": 3.0, "has_clip": True, "has_snapshot": True},
                        {"id": "camera-two", "start_time": 2.0, "has_clip": False, "has_snapshot": False},
                    ]
                )
            elif float(before) == 2.0:
                self._send_json([{"id": "camera-one", "start_time": 1.0, "has_clip": True, "has_snapshot": True}])
            else:
                self._send_json([])
            return

        if parsed.path == "/settings/api/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html>not json</html>")
            return

        if parsed.path.startswith("/api/events/") and parsed.path.endswith("/clip.mp4"):
            self.downloaded_paths.append(parsed.path)
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.end_headers()
            self.wfile.write(b"mp4-bytes")
            return

        if parsed.path.startswith("/api/events/") and parsed.path.endswith("/snapshot.jpg"):
            self.downloaded_paths.append(parsed.path)
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.end_headers()
            self.wfile.write(b"jpg-bytes")
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, payload: object) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ConcurrentFrigateHandler(BaseHTTPRequestHandler):
    active_downloads = 0
    max_active_downloads = 0
    lock = Lock()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/events":
            self._send_json([{"id": f"event-{index}", "start_time": 20 - index, "has_clip": True} for index in range(20)])
            return

        if parsed.path.startswith("/api/events/") and parsed.path.endswith("/clip.mp4"):
            with self.lock:
                type(self).active_downloads += 1
                type(self).max_active_downloads = max(type(self).max_active_downloads, type(self).active_downloads)
            sleep(0.05)
            try:
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.end_headers()
                self.wfile.write(b"mp4-bytes")
            finally:
                with self.lock:
                    type(self).active_downloads -= 1
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, payload: object) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_frigate_downloader_pages_events_and_skips_existing_files(tmp_path):
    server = HTTPServer(("127.0.0.1", 0), FrigateHandler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    FrigateHandler.downloaded_paths = []
    output_dir = tmp_path / "events"
    output_dir.mkdir()
    existing = output_dir / "camera-one.mp4"
    existing.write_bytes(b"already-here")
    event_counts: list[int] = []
    downloaded: list[tuple[str, str]] = []

    try:
        downloader = FrigateEventDownloader(
            f"http://127.0.0.1:{server.server_port}",
            output_dir,
            page_size=2,
        )
        stats = downloader.download_all(
            on_event_count=event_counts.append,
            on_download=lambda event_id, path: downloaded.append((event_id, path.name)),
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    assert stats.events_seen == 3
    assert stats.downloaded == 1
    assert stats.skipped_existing == 1
    assert stats.skipped_without_media == 1
    assert stats.failed == 0
    assert (output_dir / "camera_three.mp4").read_bytes() == b"mp4-bytes"
    assert existing.read_bytes() == b"already-here"
    assert FrigateHandler.downloaded_paths == ["/api/events/camera%2Fthree/clip.mp4"]
    assert event_counts == [3]
    assert downloaded == [("camera/three", "camera_three.mp4")]


def test_frigate_downloader_downloads_best_event_snapshots(tmp_path):
    server = HTTPServer(("127.0.0.1", 0), FrigateHandler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    FrigateHandler.downloaded_paths = []
    output_dir = tmp_path / "snapshots"
    output_dir.mkdir()
    existing = output_dir / "camera-one.jpg"
    existing.write_bytes(b"already-here")

    try:
        downloader = FrigateEventDownloader(
            f"http://127.0.0.1:{server.server_port}",
            output_dir,
            page_size=2,
        )
        stats = downloader.download_all(media_type="snapshot")
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    assert stats.events_seen == 3
    assert stats.downloaded == 1
    assert stats.skipped_existing == 1
    assert stats.skipped_without_media == 1
    assert stats.failed == 0
    assert (output_dir / "camera_three.jpg").read_bytes() == b"jpg-bytes"
    assert existing.read_bytes() == b"already-here"
    assert FrigateHandler.downloaded_paths == ["/api/events/camera%2Fthree/snapshot.jpg"]


def test_frigate_downloader_downloads_with_worker_limit(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), ConcurrentFrigateHandler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    ConcurrentFrigateHandler.active_downloads = 0
    ConcurrentFrigateHandler.max_active_downloads = 0

    try:
        downloader = FrigateEventDownloader(
            f"http://127.0.0.1:{server.server_port}",
            tmp_path,
            page_size=100,
        )
        stats = downloader.download_all(download_workers=10)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    assert stats.events_seen == 20
    assert stats.downloaded == 20
    assert stats.failed == 0
    assert ConcurrentFrigateHandler.max_active_downloads <= 10
    assert ConcurrentFrigateHandler.max_active_downloads > 1


def test_frigate_downloader_reports_non_json_events_response(tmp_path):
    server = HTTPServer(("127.0.0.1", 0), FrigateHandler)
    thread = Thread(target=server.serve_forever)
    thread.start()

    try:
        downloader = FrigateEventDownloader(
            f"http://127.0.0.1:{server.server_port}/settings",
            tmp_path,
        )
        with pytest.raises(ValueError, match="server root"):
            list(downloader.iter_events())
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
