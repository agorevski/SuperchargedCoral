from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from urllib.parse import parse_qs, urlparse

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
                        {"id": "camera/three", "start_time": 3.0, "has_clip": True},
                        {"id": "camera-two", "start_time": 2.0, "has_clip": False},
                    ]
                )
            elif float(before) == 2.0:
                self._send_json([{"id": "camera-one", "start_time": 1.0, "has_clip": True}])
            else:
                self._send_json([])
            return

        if parsed.path.startswith("/api/events/") and parsed.path.endswith("/clip.mp4"):
            self.downloaded_paths.append(parsed.path)
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.end_headers()
            self.wfile.write(b"mp4-bytes")
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

    try:
        downloader = FrigateEventDownloader(
            f"http://127.0.0.1:{server.server_port}",
            output_dir,
            page_size=2,
        )
        stats = downloader.download_all()
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    assert stats.events_seen == 3
    assert stats.downloaded == 1
    assert stats.skipped_existing == 1
    assert stats.skipped_without_clip == 1
    assert stats.failed == 0
    assert (output_dir / "camera_three.mp4").read_bytes() == b"mp4-bytes"
    assert existing.read_bytes() == b"already-here"
    assert FrigateHandler.downloaded_paths == ["/api/events/camera%2Fthree/clip.mp4"]
