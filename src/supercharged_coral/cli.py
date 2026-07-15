"""Command-line entry points for local Supercharged Coral workflows."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

from supercharged_coral.camera_service.providers import SyntheticCameraSource
from supercharged_coral.common.config import CameraConfig, MotionConfig
from supercharged_coral.dataset.database import SecurityDatabase
from supercharged_coral.integrations.frigate import FrigateEventDownloader
from supercharged_coral.motion_service.algorithms import FrameDifferencingMotionDetector
from supercharged_coral.pipeline.async_pipeline import AsyncFramePipeline, RegionEchoDetector
from supercharged_coral.tracker.sort_like import SortLikeTracker

__all__ = ["build_parser", "main"]


def _init_db(args: argparse.Namespace) -> int:
    database = SecurityDatabase(args.database)
    database.initialize()
    database.close()
    print(f"Initialized database at {args.database}")
    return 0


async def _smoke_test_async(args: argparse.Namespace) -> int:
    camera_config = CameraConfig(
        id="synthetic-driveway",
        source_type="synthetic",
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    camera = SyntheticCameraSource(camera_config, frame_limit=args.frames)
    database = SecurityDatabase(args.database)
    database.initialize()
    pipeline = AsyncFramePipeline(
        cameras=[camera],
        motion_detectors={
            camera.camera_id: FrameDifferencingMotionDetector(
                MotionConfig(threshold=20, min_area=64, blur_kernel=3, morphology_kernel=3)
            )
        },
        object_detector=RegionEchoDetector(),
        database=database,
        tracker=SortLikeTracker(),
        max_frames_per_camera=args.frames,
    )
    stats = await pipeline.run()
    payload = {
        "frames_seen": stats.frames_seen,
        "motion_frames": stats.motion_frames,
        "detections": stats.detections,
        "tracks": stats.tracks,
        "database": str(args.database),
    }
    database.close()
    print(json.dumps(payload, sort_keys=True))
    return 0


def _smoke_test(args: argparse.Namespace) -> int:
    return asyncio.run(_smoke_test_async(args))


def _parse_headers(values: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for value in values:
        if ":" not in value:
            raise argparse.ArgumentTypeError(f"header must be in 'Name: value' format: {value}")
        name, header_value = value.split(":", 1)
        name = name.strip()
        if not name:
            raise argparse.ArgumentTypeError(f"header name cannot be empty: {value}")
        headers[name] = header_value.strip()
    return headers


def _download_frigate_events(args: argparse.Namespace) -> int:
    headers = _parse_headers(args.header)
    api_key = args.api_key or os.environ.get("FRIGATE_API_KEY")
    if api_key and "Authorization" not in headers:
        headers["Authorization"] = " ".join(("Bearer", api_key))

    downloader = FrigateEventDownloader(
        args.base_url,
        args.output_dir,
        headers=headers,
        page_size=args.page_size,
        timeout=args.timeout,
    )
    try:
        stats = downloader.download_all(
            max_pages=args.max_pages,
            include_events_without_clip=args.include_events_without_clip,
            on_event_count=lambda count: print(f"Found {count} Frigate events"),
            on_download=lambda event_id, path: print(f"Downloaded {event_id} -> {path}"),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(stats.__dict__, sort_keys=True))
    return 1 if stats.failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="supercharged-coral")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_db = subparsers.add_parser("init-db", help="Create or migrate the SQLite database schema")
    init_db.add_argument("--database", default="data/supercharged-coral.db")
    init_db.set_defaults(func=_init_db)

    smoke = subparsers.add_parser("smoke-test", help="Run a local synthetic camera pipeline")
    smoke.add_argument("--frames", type=int, default=10)
    smoke.add_argument("--fps", type=float, default=120.0)
    smoke.add_argument("--width", type=int, default=320)
    smoke.add_argument("--height", type=int, default=180)
    smoke.add_argument("--database", default=":memory:")
    smoke.set_defaults(func=_smoke_test)

    frigate = subparsers.add_parser(
        "download-frigate-events",
        aliases=["download-friday-events"],
        help="Download Frigate event MP4 clips, skipping clips already on disk",
    )
    frigate.add_argument("--base-url", required=True, help="Base Frigate URL, e.g. http://frigate:5000")
    frigate.add_argument("--output-dir", default="data/frigate-events")
    frigate.add_argument("--page-size", type=int, default=100)
    frigate.add_argument("--max-pages", type=int, default=None)
    frigate.add_argument("--timeout", type=float, default=30.0)
    frigate.add_argument(
        "--api-key",
        default=None,
        help="Frigate API credential; defaults to FRIGATE_API_KEY when set",
    )
    frigate.add_argument(
        "--header",
        action="append",
        default=[],
        help="Additional request header in 'Name: value' format; repeatable",
    )
    frigate.add_argument(
        "--include-events-without-clip",
        action="store_true",
        help="Attempt clip downloads even when the event reports has_clip=false",
    )
    frigate.set_defaults(func=_download_frigate_events)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "database", None) not in (None, ":memory:"):
        Path(args.database).parent.mkdir(parents=True, exist_ok=True)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
