import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import imageio
import numpy as np
import torch
import utils3d.torch
from PIL import Image

from trellis.pipelines import TrellisVideoTo4DPipeline
from trellis.utils import render_utils
from trellis.utils.general_utils import save_images_as_video


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_view_indices(raw: Optional[str], available: Sequence[int]) -> List[int]:
    if raw is None or raw.strip() == "":
        return list(available)
    values: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    values = list(dict.fromkeys(values))
    unsupported = sorted(set(values) - set(available))
    if unsupported:
        raise ValueError(f"requested view idxs not found in dataset: {unsupported}")
    return values


def discover_sample_root(data_root: Path, sample_name: Optional[str]) -> Path:
    if sample_name is None or sample_name.strip() == "":
        return data_root
    return data_root / sample_name


def list_available_views(sample_root: Path) -> List[int]:
    views = []
    for path in sample_root.glob("view_*"):
        if not path.is_dir():
            continue
        suffix = path.name.replace("view_", "")
        if suffix.isdigit():
            views.append(int(suffix))
    views = sorted(set(views))
    if len(views) == 0:
        raise FileNotFoundError(f"no view folders found under {sample_root}")
    return views


def _frame_idx_from_name(name: str) -> Optional[int]:
    marker = "_frame_"
    if marker not in name:
        return None
    token = name.split(marker)[-1].split(".")[0]
    if token.isdigit():
        return int(token)
    return None


def load_view_sequence(view_root: Path, max_frames: Optional[int]) -> Tuple[List[Image.Image], List[int], List[str], dict]:
    transforms_path = view_root / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"missing transforms json: {transforms_path}")
    with transforms_path.open("r") as f:
        transforms = json.load(f)

    frame_metas = transforms.get("frames", [])
    if len(frame_metas) == 0:
        raise ValueError(f"no frames in {transforms_path}")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("--max_frames must be > 0 when provided")

    images: List[Image.Image] = []
    frame_indices: List[int] = []
    file_names: List[str] = []

    for i, meta in enumerate(frame_metas):
        if max_frames is not None and len(images) >= max_frames:
            break
        rel = meta.get("file_path", "")
        rel = rel.replace("./", "")
        img_path = view_root / rel
        if not img_path.exists():
            continue
        img = Image.open(img_path)
        if img.mode != "RGBA":
            rgb = np.array(img.convert("RGB"))
            alpha = (np.max(rgb, axis=-1) > 0).astype(np.uint8) * 255
            rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
            img = Image.fromarray(rgba, mode="RGBA")
        images.append(img)

        frame_idx = meta.get("frame_idx", None)
        if frame_idx is None:
            frame_idx = _frame_idx_from_name(img_path.name)
        if frame_idx is None:
            frame_idx = i
        frame_indices.append(int(frame_idx))
        file_names.append(img_path.name)

    if len(images) == 0:
        raise ValueError(f"no images loaded from {view_root}")
    return images, frame_indices, file_names, transforms


def load_camera_from_view(
    sample_root: Path, view_idx: int, resolution_override: Optional[int], camera_distance_scale: float
) -> Tuple[torch.Tensor, torch.Tensor, str, int]:
    view_root = sample_root / f"view_{view_idx:03d}"
    transforms_path = view_root / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"missing transforms json for view {view_idx}: {transforms_path}")

    with transforms_path.open("r") as f:
        meta = json.load(f)
    frames = meta.get("frames", [])
    if len(frames) == 0:
        raise ValueError(f"no frames in camera file: {transforms_path}")

    frame0 = frames[0]
    fov = frame0.get("camera_angle_x", meta.get("camera_angle_x", None))
    if fov is None:
        raise KeyError(f"camera_angle_x missing in {transforms_path}")

    c2w = torch.tensor(frame0["transform_matrix"]).float().cuda()
    # Move camera farther away from origin while preserving viewing direction.
    c2w[:3, 3] *= float(camera_distance_scale)
    c2w[:3, 1:3] *= -1
    extr = torch.inverse(c2w)
    intr = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov).float().cuda(), torch.tensor(fov).float().cuda())

    default_h = meta.get("h", 512)
    default_w = meta.get("w", 512)
    resolution = int(min(int(default_h), int(default_w)))
    if resolution_override is not None:
        resolution = int(resolution_override)

    return extr, intr, f"view_{view_idx:03d}", resolution


def render_rgba_from_black_white(sample, extrinsics, intrinsics, resolution: int):
    render_black = render_utils.render_frames(
        sample, extrinsics, intrinsics, {"resolution": resolution, "bg_color": (0, 0, 0)}, verbose=False
    )["color"]
    render_white = render_utils.render_frames(
        sample, extrinsics, intrinsics, {"resolution": resolution, "bg_color": (1, 1, 1)}, verbose=False
    )["color"]

    rgba_views = []
    for img_black_u8, img_white_u8 in zip(render_black, render_white):
        img_black = img_black_u8.astype(np.float32) / 255.0
        img_white = img_white_u8.astype(np.float32) / 255.0

        alpha = 1.0 - np.mean(img_white - img_black, axis=-1)
        alpha = np.clip(alpha, 0.0, 1.0)
        alpha_safe = np.clip(alpha, 1e-6, 1.0)
        fg = np.clip(img_black / alpha_safe[..., None], 0.0, 1.0)
        fg[alpha <= 1e-4] = 0.0

        rgba = np.concatenate([fg, alpha[..., None]], axis=-1)
        rgba_u8 = np.clip(rgba * 255.0, 0, 255).astype(np.uint8)
        rgba_views.append(rgba_u8)

    return render_black, render_white, rgba_views


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    sample_root = discover_sample_root(Path(args.data_root), args.sample_name)
    if not sample_root.is_dir():
        raise FileNotFoundError(f"sample root does not exist: {sample_root}")

    available_views = list_available_views(sample_root)
    if args.input_view_idx not in available_views:
        raise ValueError(f"input view {args.input_view_idx} not found in {available_views}")
    render_view_idxs = parse_view_indices(args.render_view_idxs, available_views)

    input_view_root = sample_root / f"view_{args.input_view_idx:03d}"
    input_images, frame_indices, source_names, _ = load_view_sequence(input_view_root, args.max_frames)

    print("=== Configuration ===")
    print(f"sample_root      : {sample_root}")
    print(f"input_view_idx   : {args.input_view_idx}")
    print(f"render_view_idxs : {render_view_idxs}")
    print(f"num_input_frames : {len(input_images)}")
    print(f"pipeline_path    : {args.pipeline_path}")
    print(f"output_dir       : {args.output_dir}")
    print(f"camera_dist_scale: {args.camera_distance_scale}")

    pipeline = TrellisVideoTo4DPipeline.from_pretrained(args.pipeline_path)
    pipeline.cuda()
    _, samples, images_cond = pipeline.run(input_images, seed=args.seed, return_images=True, formats=["gaussian"])
    cond_rgb = np.array(images_cond)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    save_images_as_video(cond_rgb, str(output_root / "conditioning_sequence.mp4"))

    exts = []
    ints = []
    view_names = []
    res_candidates = []
    for view_idx in render_view_idxs:
        ext, intr, name, resolution = load_camera_from_view(
            sample_root, view_idx, args.resolution, args.camera_distance_scale
        )
        exts.append(ext)
        ints.append(intr)
        view_names.append(name)
        res_candidates.append(resolution)
    render_resolution = int(min(res_candidates))

    per_view_rgb_video = {name: [] for name in view_names}
    for seq_idx, sample in enumerate(samples["gaussian"]):
        img_black, img_white, img_rgba = render_rgba_from_black_white(sample, exts, ints, render_resolution)
        frame_idx = frame_indices[seq_idx] if seq_idx < len(frame_indices) else seq_idx
        src_name = source_names[seq_idx] if seq_idx < len(source_names) else f"seq_{seq_idx:03d}.png"
        for view_id, view_name in enumerate(view_names):
            out_dir = output_root / view_name
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = f"seq_{seq_idx:03d}_frame_{frame_idx:03d}"
            imageio.imwrite(str(out_dir / f"{stem}_pred_black.png"), img_black[view_id])
            imageio.imwrite(str(out_dir / f"{stem}_pred_white.png"), img_white[view_id])
            imageio.imwrite(str(out_dir / f"{stem}_pred_rgba.png"), img_rgba[view_id])
            per_view_rgb_video[view_name].append(img_rgba[view_id][..., :3])

        if args.save_metadata:
            frame_meta = {
                "seq_idx": int(seq_idx),
                "frame_idx": int(frame_idx),
                "source_name": src_name,
            }
            with (output_root / f"frame_meta_{seq_idx:03d}.json").open("w") as f:
                json.dump(frame_meta, f, indent=2)

    for view_name, rgb_frames in per_view_rgb_video.items():
        save_images_as_video(np.stack(rgb_frames), str(output_root / f"{view_name}_pred_rgb.mp4"))

    print(f"Done. Saved outputs to: {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SS4D on prepared GrowFlow-format dataset and render RGBA outputs.")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to dataset root or sample root. Expected sample structure with view_XXX folders.",
    )
    parser.add_argument(
        "--sample_name",
        type=str,
        default=None,
        help="Optional sample folder name under --data_root. If omitted, --data_root is treated as sample root.",
    )
    parser.add_argument("--input_view_idx", type=int, default=0, help="View index for input temporal sequence.")
    parser.add_argument(
        "--render_view_idxs",
        type=str,
        default=None,
        help="Comma-separated view idxs for rendering (default: all available views).",
    )
    parser.add_argument(
        "--pipeline_path",
        type=str,
        default="lizb6626/SS4D",
        help="Pretrained SS4D path or HuggingFace repo.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_frames", type=int, default=None, help="Optional max number of input frames.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Optional render resolution override. By default, uses per-view transforms.json h/w.",
    )
    parser.add_argument(
        "--camera_distance_scale",
        type=float,
        default=1.3,
        help="Scale factor for camera distance to origin. >1.0 moves camera farther away.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for predicted renders.")
    parser.add_argument("--save_metadata", action="store_true", help="Save per-frame metadata json files.")

    main(parser.parse_args())
