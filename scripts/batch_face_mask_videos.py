#!/usr/bin/env python3
"""Batch face-masks all MP4s in ../videos (skipping *output*.mp4).

Outputs: <name>_masked.mp4 next to the input.
Uses MediaPipe face detection when available (falls back to Haar).
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

import cv2
import yaml

# Ensure local imports work when run from anywhere
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from face_masking import FaceMasker, FaceMaskingConfig  # noqa: E402


def _load_face_masking_config(config_path: Path) -> FaceMaskingConfig:
    if not config_path.exists():
        return FaceMaskingConfig(async_enabled=False, detector_type="mediapipe")

    with config_path.open("r") as f:
        config_dict: Dict[str, Any] = yaml.safe_load(f) or {}

    fm_dict = (config_dict.get("face_masking") or {}) if isinstance(config_dict, dict) else {}
    fm_config = FaceMaskingConfig.from_dict(fm_dict)

    # For offline batch processing, prefer sync to avoid any cross-thread race.
    fm_config.async_enabled = False

    # Guardrails
    if fm_config.detection_interval_frames < 1:
        fm_config.detection_interval_frames = 1
    if fm_config.persistence_frames < 0:
        fm_config.persistence_frames = 0

    return fm_config


def _iter_input_videos(videos_dir: Path):
    for path in sorted(videos_dir.glob("*.mp4")):
        name_lower = path.name.lower()
        if name_lower.endswith("output.mp4") or "_output" in name_lower:
            continue
        if name_lower.endswith("_masked.mp4"):
            continue
        yield path


def mask_video(input_path: Path, output_path: Path, fm_config: FaceMaskingConfig) -> None:
    tmp_output_path = output_path.with_name(f"{output_path.stem}.__tmp__.mp4")
    if tmp_output_path.exists():
        try:
            tmp_output_path.unlink()
        except Exception:
            pass

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video dimensions for {input_path}: {width}x{height}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_output_path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output writer: {tmp_output_path}")

    masker = FaceMasker(config=fm_config)

    try:
        frame_num = 0
        last_print = -1

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            masked = masker.mask_faces(frame, frame_num=frame_num)
            writer.write(masked)

            if total_frames > 0:
                pct = int((frame_num + 1) * 100 / total_frames)
                if pct // 5 != last_print:
                    last_print = pct // 5
                    print(f"  {input_path.name}: {pct}% ({frame_num + 1}/{total_frames})")
            else:
                if frame_num % 150 == 0:
                    print(f"  {input_path.name}: frame {frame_num}")

            frame_num += 1

    finally:
        try:
            masker.stop()
        except Exception:
            pass
        writer.release()
        cap.release()

    # Only publish output once the whole video is written
    os.replace(tmp_output_path, output_path)


def main():
    repo_root = SCRIPT_DIR.parent
    videos_dir = repo_root / "videos"
    config_path = repo_root / "config.yaml"

    if not videos_dir.exists():
        raise SystemExit(f"videos dir not found: {videos_dir}")

    fm_config = _load_face_masking_config(config_path)

    print("[BatchFaceMask] Using config:")
    print(f"  detector_type={fm_config.detector_type}  async_enabled={fm_config.async_enabled}")
    print(f"  detection_interval_frames={fm_config.detection_interval_frames}  persistence_frames={fm_config.persistence_frames}")
    print(f"  mask_type={fm_config.mask_type}  blur_strength={fm_config.blur_strength}")

    inputs = list(_iter_input_videos(videos_dir))
    if not inputs:
        print("[BatchFaceMask] No input videos found (or all were filtered).")
        return

    print(f"[BatchFaceMask] Found {len(inputs)} videos to process")

    for input_path in inputs:
        output_path = input_path.with_name(f"{input_path.stem}_masked.mp4")
        if output_path.exists():
            print(f"[BatchFaceMask] SKIP (exists): {output_path.name}")
            continue

        print(f"[BatchFaceMask] Processing: {input_path.name} -> {output_path.name}")
        mask_video(input_path, output_path, fm_config)
        print(f"[BatchFaceMask] Wrote: {output_path.name}")


if __name__ == "__main__":
    main()
