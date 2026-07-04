"""Command-line entry points for local Supercharged Coral workflows."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from supercharged_coral.camera_service.providers import SyntheticCameraSource
from supercharged_coral.common.config import CameraConfig, MotionConfig
from supercharged_coral.dataset.database import SecurityDatabase
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.database != ":memory:":
        Path(args.database).parent.mkdir(parents=True, exist_ok=True)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
