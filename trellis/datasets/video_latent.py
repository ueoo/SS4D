import json
import math
import os
import random

from typing import *

import numpy as np
import torch
import utils3d
import utils3d.torch

from PIL import Image

from .. import models
from ..modules import sparse as sp
from ..renderers import OctreeRenderer
from ..representations.octree import DfsOctree as Octree
from ..utils import render_utils
from ..utils.data_utils import load_balanced_group_indices
from .base import StandardVideoDatasetBase


def _read_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _make_frame_montage(frames: torch.Tensor, num_cols: int = 5) -> torch.Tensor:
    """
    frames: [T, C, H, W]
    returns: [C, H_grid, W_grid]
    """
    num_frames = frames.shape[0]
    num_cols = min(num_cols, num_frames)
    num_rows = int(math.ceil(num_frames / num_cols))
    _, channels, height, width = frames.shape
    montage = torch.zeros(channels, num_rows * height, num_cols * width, device=frames.device, dtype=frames.dtype)
    for frame_idx in range(num_frames):
        row = frame_idx // num_cols
        col = frame_idx % num_cols
        montage[:, row * height : (row + 1) * height, col * width : (col + 1) * width] = frames[frame_idx]
    return montage


def _sample_bbox_from_rgba(images: list[np.ndarray]) -> tuple[int, int, int, int]:
    alpha_masks = [img[..., 3] > 0 for img in images]
    mask = np.logical_or.reduce(alpha_masks)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        h, w = images[0].shape[:2]
        return (0, 0, w, h)
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max())
    y1 = int(ys.max())
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    half_size = max(x1 - x0, y1 - y0) / 2 * 1.2
    return (
        int(center_x - half_size),
        int(center_y - half_size),
        int(center_x + half_size),
        int(center_y + half_size),
    )


def _crop_and_resize_rgba(image: np.ndarray, bbox: tuple[int, int, int, int], image_size: int) -> torch.Tensor:
    x0, y0, x1, y1 = bbox
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - image.shape[1])
    pad_bottom = max(0, y1 - image.shape[0])
    if pad_left or pad_top or pad_right or pad_bottom:
        image = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        x0 += pad_left
        x1 += pad_left
        y0 += pad_top
        y1 += pad_top
    image = image[y0:y1, x0:x1]
    pil = Image.fromarray(image, mode="RGBA")
    pil = pil.resize((image_size, image_size), Image.Resampling.LANCZOS)
    rgba = np.array(pil).astype(np.float32) / 255.0
    rgb = rgba[..., :3] * rgba[..., 3:4]
    return torch.from_numpy(rgb).permute(2, 0, 1).float()


def maybe_apply_cond_aug(image: torch.Tensor, aug_cfg: dict | None) -> torch.Tensor:
    if aug_cfg is None:
        return image
    p = float(aug_cfg.get("p", 0.0))
    if p <= 0 or float(np.random.rand()) > p:
        return image

    x = image
    b = float(aug_cfg.get("brightness", 0.0))
    if b > 0:
        x = x * float(1.0 + np.random.uniform(-b, b))

    c = float(aug_cfg.get("contrast", 0.0))
    if c > 0:
        factor = float(1.0 + np.random.uniform(-c, c))
        mean = x.mean(dim=(1, 2), keepdim=True)
        x = (x - mean) * factor + mean

    s = float(aug_cfg.get("saturation", 0.0))
    if s > 0:
        factor = float(1.0 + np.random.uniform(-s, s))
        gray = x.mean(dim=0, keepdim=True)
        x = (x - gray) * factor + gray

    nstd = float(aug_cfg.get("gaussian_noise_std", 0.0))
    if nstd > 0:
        x = x + torch.randn_like(x) * nstd

    dp = float(aug_cfg.get("dropout_p", 0.0))
    if dp > 0:
        mask = (torch.rand_like(x[:1]) > dp).float()
        x = x * mask

    return x.clamp(0.0, 1.0)


def check_compressed_availability(roots: str, use_renders: bool = False) -> bool:
    use_compressed = True
    for root in roots.split(","):
        if not os.path.exists(os.path.join(root, "renders_cond_compressed")):
            use_compressed = False
            break
        if use_renders and not os.path.exists(os.path.join(root, "renders_compressed")):
            use_compressed = False
            break
    return use_compressed


class VideoLatentDatasetBase(StandardVideoDatasetBase):
    """
    Metadata-driven video dataset base.

    Frame-wise metadata rows (sha256 like prefix_sampleidx_frameidx) are grouped into
    sample-wise instances (prefix_sampleidx). Each __getitem__ samples a temporal sequence
    from that grouped frame list.
    """

    def __init__(
        self,
        roots: str,
        *,
        latent_model: str,
        latent_data_root: Optional[str] = None,
        sample_prefixes: Optional[Sequence[str]] = None,
        num_frames: Optional[int] = None,
        max_frames: Optional[int] = None,
        frame_step: int = 1,
        random_start: bool = True,
        **kwargs,
    ):
        self.latent_model = latent_model
        self.latent_data_root = latent_data_root
        self.sample_prefixes = [str(x) for x in sample_prefixes] if sample_prefixes is not None else None
        self.num_frames = num_frames if num_frames is not None else max_frames
        self.frame_step = int(frame_step)
        self.random_start = bool(random_start)
        self.value_range = (0, 1)

        if self.frame_step <= 0:
            raise ValueError(f"frame_step must be positive, got {self.frame_step}")

        super().__init__(roots, **kwargs)

        if self.sample_prefixes is not None:
            self.instances = [
                (root, sample)
                for root, sample in self.instances
                if any(sample.startswith(prefix) for prefix in self.sample_prefixes)
            ]

        if len(self.instances) == 0:
            raise ValueError(f"No grouped video instances found in roots={roots}")

        if self.num_frames is None:
            self.num_frames = min(len(self.get_sample_frames(root, sample)) for root, sample in self.instances)

        min_required = 1 + (self.num_frames - 1) * self.frame_step
        self.instances = [
            (root, sample)
            for root, sample in self.instances
            if len(self.get_sample_frames(root, sample)) >= min_required
        ]
        if len(self.instances) == 0:
            raise ValueError(
                f"No sample has enough frames for num_frames={self.num_frames}, frame_step={self.frame_step}."
            )

    def filter_metadata(self, metadata):
        return metadata, {}

    def _get_latent_root(self, root: str) -> str:
        return self.latent_data_root if self.latent_data_root is not None else root

    def _select_sequence_shas(self, root: str, sample: str) -> list[str]:
        frame_shas = self.get_sample_frames(root, sample)
        needed = 1 + (self.num_frames - 1) * self.frame_step
        if len(frame_shas) < needed:
            raise ValueError(
                f"Sample {sample} has {len(frame_shas)} frames, but needs {needed} "
                f"(num_frames={self.num_frames}, frame_step={self.frame_step})."
            )
        start = random.randint(0, len(frame_shas) - needed) if self.random_start else 0
        return [frame_shas[start + i * self.frame_step] for i in range(self.num_frames)]

    def _resolve_latent_instance(self, root: str, frame_sha: str) -> str:
        row = self.get_metadata_row(root, frame_sha)
        source_instance = row.get("source_instance", None)
        if source_instance is not None and str(source_instance) != "" and str(source_instance) != "nan":
            return str(source_instance)
        return frame_sha

    def visualize_condition(self, cond: torch.Tensor) -> torch.Tensor:
        if cond.ndim == 4:
            cond = cond.unsqueeze(0)
        return torch.stack([_make_frame_montage(seq) for seq in cond], dim=0)

    def visualize_sample(self, sample):
        if isinstance(sample, dict):
            cond = sample["cond"]
            x_0 = sample["x_0"]
            num_frames = sample["num_frames"]
        else:
            raise ValueError(f"Unsupported sample payload for visualization: {type(sample)}")
        return {
            "cond": self.visualize_condition(cond),
            "target": self.visualize_latent(x_0, num_frames),
        }

    def visualize_latent(self, x_0, num_frames: int) -> torch.Tensor:
        raise NotImplementedError


class VideoImageConditionedMixin:
    """
    Video counterpart of TRELLIS ImageConditionedMixin:
    - random one view per sequence
    - same view index for all frames in the chosen sequence
    """

    def __init__(self, roots, *, image_size=518, use_renders=False, cond_aug: dict | None = None, **kwargs):
        self.image_size = image_size
        self.cond_aug = cond_aug
        self.use_compressed = False
        self.has_renders = self._check_folder_availability(roots, folder="renders")
        if not self.has_renders:
            raise ValueError("`renders` folder is required in every dataset root for video conditioning.")
        super().__init__(roots, image_size=image_size, use_renders=use_renders, cond_aug=cond_aug, **kwargs)

    @staticmethod
    def _check_folder_availability(roots: str, folder: str) -> bool:
        for root in roots.split(","):
            if not os.path.isdir(os.path.join(root, folder)):
                return False
        return True

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        if "rendered" in metadata.columns:
            metadata = metadata[metadata["rendered"]]
            stats["Rendered"] = len(metadata)
        return metadata, stats

    def _resolve_render_folder(self):
        return "renders"

    def _load_rgba_by_view(self, root: str, frame_sha: str, render_folder: str, view_idx: int) -> np.ndarray:
        image_root = os.path.join(root, render_folder, frame_sha)
        transforms = _read_json(os.path.join(image_root, "transforms.json"))
        frames = transforms["frames"]
        if view_idx >= len(frames):
            raise ValueError(f"view_idx={view_idx} out of range for {image_root}")
        image_path = os.path.join(image_root, frames[view_idx]["file_path"])
        image = Image.open(image_path)
        if image.mode != "RGBA":
            rgb = np.array(image.convert("RGB"))
            alpha = (np.max(rgb, axis=-1) > 0).astype(np.uint8) * 255
            rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
            image = Image.fromarray(rgba, mode="RGBA")
        return np.array(image)

    def _load_video_condition(self, root: str, frame_shas: list[str]) -> torch.Tensor:
        render_folder = self._resolve_render_folder()
        n_views = []
        for frame_sha in frame_shas:
            transforms_path = os.path.join(root, render_folder, frame_sha, "transforms.json")
            transforms = _read_json(transforms_path)
            n_views.append(len(transforms["frames"]))
        max_common_views = min(n_views)
        if max_common_views <= 0:
            raise ValueError(f"No shared rendered views across sequence {frame_shas[0]} ... {frame_shas[-1]}")
        view_idx = np.random.randint(max_common_views)

        images = [self._load_rgba_by_view(root, frame_sha, render_folder, view_idx) for frame_sha in frame_shas]
        bbox = _sample_bbox_from_rgba(images)
        seq = [_crop_and_resize_rgba(image, bbox, self.image_size) for image in images]
        seq = [maybe_apply_cond_aug(image, self.cond_aug) for image in seq]
        return torch.stack(seq, dim=0)


class VideoConditionedSparseStructureLatent(VideoImageConditionedMixin, VideoLatentDatasetBase):
    def __init__(
        self,
        roots: str,
        *,
        min_aesthetic_score: float = 5.0,
        pretrained_ss_dec: str = "lizb6626/SS4D/ckpts/ss_dec_conv3d_16l8_fp16",
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
        normalization: Optional[dict] = None,
        **kwargs,
    ):
        self.min_aesthetic_score = min_aesthetic_score
        self.normalization = normalization
        super().__init__(roots, normalization=normalization, **kwargs)
        self.pretrained_ss_dec = pretrained_ss_dec
        self.ss_dec_path = ss_dec_path
        self.ss_dec_ckpt = ss_dec_ckpt
        self._ss_decoder = None

        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization["mean"]).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(self.normalization["std"]).reshape(-1, 1, 1, 1)

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        latent_col = f"ss_latent_{self.latent_model}"
        if latent_col in metadata.columns:
            metadata = metadata[metadata[latent_col]]
            stats["With sparse structure latents"] = len(metadata)
        if "aesthetic_score" in metadata.columns:
            metadata = metadata[metadata["aesthetic_score"] >= self.min_aesthetic_score]
            stats[f"Aesthetic score >= {self.min_aesthetic_score}"] = len(metadata)
        return metadata, stats

    def _load_frame_latent(self, root: str, frame_sha: str) -> torch.Tensor:
        latent_instance = self._resolve_latent_instance(root, frame_sha)
        latent_root = self._get_latent_root(root)
        latent_path = os.path.join(latent_root, "ss_latents", self.latent_model, f"{latent_instance}.npz")
        latent = np.load(latent_path)
        z = torch.tensor(latent["mean"]).float()
        if self.normalization is not None:
            z = (z - self.mean) / self.std
        return z

    def get_instance(self, root: str, sample: str):
        frame_shas = self._select_sequence_shas(root, sample)
        x_0 = torch.stack([self._load_frame_latent(root, frame_sha) for frame_sha in frame_shas], dim=0)
        cond = self._load_video_condition(root, frame_shas)
        return {
            "x_0": x_0,
            "cond": cond,
            "num_frames": self.num_frames,
            "sample_id": sample,
            "sha256s": frame_shas,
        }

    @staticmethod
    def collate_fn(batch, split_size=None):
        return {
            "x_0": torch.stack([b["x_0"] for b in batch], dim=0),
            "cond": torch.stack([b["cond"] for b in batch], dim=0),
            "num_frames": batch[0]["num_frames"],
            "sample_id": [b["sample_id"] for b in batch],
            "sha256s": [b["sha256s"] for b in batch],
        }

    def _load_ss_decoder(self):
        if self._ss_decoder is not None:
            return
        if self.ss_dec_path is not None:
            cfg = _read_json(os.path.join(self.ss_dec_path, "config.json"))
            decoder = getattr(models, cfg["models"]["decoder"]["name"])(**cfg["models"]["decoder"]["args"])
            ckpt_path = os.path.join(self.ss_dec_path, "ckpts", f"decoder_{self.ss_dec_ckpt}.pt")
            decoder.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        else:
            decoder = models.from_pretrained(self.pretrained_ss_dec)
        self._ss_decoder = decoder.cuda().eval()

    @torch.no_grad()
    def _decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        self._load_ss_decoder()
        if self.normalization is not None:
            z = z * self.std.to(z.device) + self.mean.to(z.device)
        return self._ss_decoder(z)

    @torch.no_grad()
    def visualize_latent(self, x_0, num_frames: int) -> torch.Tensor:
        if isinstance(num_frames, torch.Tensor):
            num_frames = int(num_frames.item())
        if x_0.ndim == 6:
            x_0 = x_0.reshape(x_0.shape[0] * x_0.shape[1], *x_0.shape[2:])
        elif x_0.ndim == 4:
            x_0 = x_0.unsqueeze(0)
        occ = self._decode_latent(x_0.cuda())
        batch_size = occ.shape[0] // num_frames

        yaw = np.pi / 6
        pitch = np.pi / 9
        orig = (
            torch.tensor([np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch), np.sin(pitch)]).float().cuda()
            * 2.0
        )
        fov = torch.deg2rad(torch.tensor(30.0)).cuda()
        extr = utils3d.torch.extrinsics_look_at(
            orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda()
        )
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)

        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = 256
        renderer.rendering_options.near = 0.8
        renderer.rendering_options.far = 1.6
        renderer.rendering_options.bg_color = (0, 0, 0)
        renderer.rendering_options.ssaa = 2
        renderer.pipe.primitive = "voxel"

        seq_images = []
        for b in range(batch_size):
            frames = []
            for t in range(num_frames):
                idx = b * num_frames + t
                representation = Octree(
                    depth=10,
                    aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
                    device="cuda",
                    primitive="voxel",
                    sh_degree=0,
                    primitive_config={"solid": True},
                )
                coords = torch.nonzero(occ[idx, 0] > 0, as_tuple=False)
                resolution = occ.shape[-1]
                representation.position = coords.float() / resolution
                representation.depth = torch.full(
                    (representation.position.shape[0], 1),
                    int(np.log2(resolution)),
                    dtype=torch.uint8,
                    device="cuda",
                )
                image = renderer.render(representation, extr, intr, colors_overwrite=representation.position)["color"]
                frames.append(image)
            seq_images.append(_make_frame_montage(torch.stack(frames, dim=0)))
        return torch.stack(seq_images, dim=0)


class VideoConditionedSLat(VideoImageConditionedMixin, VideoLatentDatasetBase):
    def __init__(
        self,
        roots: str,
        *,
        min_aesthetic_score: float = 5.0,
        max_num_voxels: int = 32768,
        pretrained_slat_dec: str = "lizb6626/SS4D/ckpts/slat_dec_gs_4d",
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
        normalization: Optional[dict] = None,
        **kwargs,
    ):
        self.min_aesthetic_score = min_aesthetic_score
        self.max_num_voxels = max_num_voxels
        self.normalization = normalization
        super().__init__(roots, normalization=normalization, **kwargs)
        self.pretrained_slat_dec = pretrained_slat_dec
        self.slat_dec_path = slat_dec_path
        self.slat_dec_ckpt = slat_dec_ckpt
        self._slat_decoder = None

        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization["mean"]).reshape(1, -1)
            self.std = torch.tensor(self.normalization["std"]).reshape(1, -1)

        self.loads = [self._estimate_sequence_load(root, sample) for root, sample in self.instances]

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        latent_col = f"latent_{self.latent_model}"
        if latent_col in metadata.columns:
            metadata = metadata[metadata[latent_col]]
            stats["With latent"] = len(metadata)
        if "aesthetic_score" in metadata.columns:
            metadata = metadata[metadata["aesthetic_score"] >= self.min_aesthetic_score]
            stats[f"Aesthetic score >= {self.min_aesthetic_score}"] = len(metadata)
        if "num_voxels" in metadata.columns:
            metadata = metadata[metadata["num_voxels"] <= self.max_num_voxels]
            stats[f"Num voxels <= {self.max_num_voxels}"] = len(metadata)
        return metadata, stats

    def _estimate_sequence_load(self, root: str, sample: str) -> int:
        frame_shas = self.get_sample_frames(root, sample)
        total = 0
        for frame_sha in frame_shas[: self.num_frames]:
            row = self.get_metadata_row(root, frame_sha)
            total += int(row.get("num_voxels", 1))
        return total

    def _load_frame_latent(self, root: str, frame_sha: str) -> dict[str, torch.Tensor]:
        latent_instance = self._resolve_latent_instance(root, frame_sha)
        latent_root = self._get_latent_root(root)
        latent_path = os.path.join(latent_root, "latents", self.latent_model, f"{latent_instance}.npz")
        data = np.load(latent_path)
        coords = torch.tensor(data["coords"]).int()
        feats = torch.tensor(data["feats"]).float()
        if self.normalization is not None:
            feats = (feats - self.mean) / self.std
        return {"coords": coords, "feats": feats}

    def get_instance(self, root: str, sample: str):
        frame_shas = self._select_sequence_shas(root, sample)
        latents = [self._load_frame_latent(root, frame_sha) for frame_sha in frame_shas]
        cond = self._load_video_condition(root, frame_shas)
        return {
            "coords_list": [latent["coords"] for latent in latents],
            "feats_list": [latent["feats"] for latent in latents],
            "cond": cond,
            "num_frames": self.num_frames,
            "sample_id": sample,
            "sha256s": frame_shas,
        }

    @staticmethod
    def collate_fn(batch, split_size=None):
        loads = [sum(coords.shape[0] for coords in b["coords_list"]) for b in batch]
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices(loads, split_size)

        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            num_frames = sub_batch[0]["num_frames"]
            coords = []
            feats = []
            layout = []
            start = 0
            for batch_idx, sample in enumerate(sub_batch):
                for frame_idx in range(num_frames):
                    batch_frame_idx = batch_idx * num_frames + frame_idx
                    frame_coords = sample["coords_list"][frame_idx]
                    frame_feats = sample["feats_list"][frame_idx]
                    coords.append(
                        torch.cat(
                            [torch.full((frame_coords.shape[0], 1), batch_frame_idx, dtype=torch.int32), frame_coords],
                            dim=-1,
                        )
                    )
                    feats.append(frame_feats)
                    layout.append(slice(start, start + frame_coords.shape[0]))
                    start += frame_coords.shape[0]
            coords = torch.cat(coords, dim=0)
            feats = torch.cat(feats, dim=0)
            x_0 = sp.SparseTensor(
                coords=coords,
                feats=feats,
                shape=torch.Size([len(sub_batch) * num_frames, feats.shape[-1]]),
                layout=layout,
            )
            pack = {
                "x_0": x_0,
                "cond": torch.stack([b["cond"] for b in sub_batch], dim=0),
                "num_frames": num_frames,
                "sample_id": [b["sample_id"] for b in sub_batch],
                "sha256s": [b["sha256s"] for b in sub_batch],
            }
            packs.append(pack)
        if split_size is None:
            return packs[0]
        return packs

    def _load_slat_decoder(self):
        if self._slat_decoder is not None:
            return
        if self.slat_dec_path is not None:
            cfg = _read_json(os.path.join(self.slat_dec_path, "config.json"))
            decoder = getattr(models, cfg["models"]["decoder"]["name"])(**cfg["models"]["decoder"]["args"])
            ckpt_path = os.path.join(self.slat_dec_path, "ckpts", f"decoder_{self.slat_dec_ckpt}.pt")
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(self.slat_dec_path, "ckpts_persist", f"decoder_{self.slat_dec_ckpt}.pt")
            decoder.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        else:
            decoder = models.from_pretrained(self.pretrained_slat_dec)
        self._slat_decoder = decoder.cuda().eval()

    @torch.no_grad()
    def _decode_latent(self, z: sp.SparseTensor, num_frames: int):
        self._load_slat_decoder()
        if self.normalization is not None:
            z = z.replace(z.feats * self.std.to(z.device) + self.mean.to(z.device))
        return self._slat_decoder(z.cuda(), num_frames)

    @torch.no_grad()
    def visualize_latent(self, x_0, num_frames: int) -> torch.Tensor:
        if isinstance(num_frames, torch.Tensor):
            num_frames = int(num_frames.item())
        reps = self._decode_latent(x_0, num_frames)
        batch_size = len(reps) // num_frames

        yaw = np.pi / 6
        pitch = np.pi / 9
        exts, ints = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, 2.0, 40)

        seq_images = []
        for b in range(batch_size):
            frames = []
            for t in range(num_frames):
                representation = reps[b * num_frames + t]
                renderer = render_utils.get_renderer(representation, resolution=256, bg_color=(0, 0, 0), ssaa=1)
                image = renderer.render(representation, exts, ints)["color"]
                frames.append(image)
            seq_images.append(_make_frame_montage(torch.stack(frames, dim=0)))
        return torch.stack(seq_images, dim=0)
