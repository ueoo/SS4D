import copy

from typing import *

import numpy as np
import torch
import torch.nn.functional as F

from easydict import EasyDict as edict
from torch.utils.data import DataLoader

from ...pipelines import samplers
from ...utils.general_utils import dict_reduce
from ..basic import BasicTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.video_conditioned import VideoConditionedMixin


class FlowMatchingTrainer(BasicTrainer):
    _NON_MODEL_KWARGS = {"sample_id", "sha256s"}

    def __init__(
        self,
        *args,
        t_schedule: dict = {
            "name": "logitNormal",
            "args": {
                "mean": 0.0,
                "std": 1.0,
            },
        },
        sigma_min: float = 1e-5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.t_schedule = t_schedule
        self.sigma_min = sigma_min

    def diffuse(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_0)
        assert noise.shape == x_0.shape
        t = t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
        return (1 - t) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t) * noise

    def reverse_diffuse(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        assert noise.shape == x_t.shape
        t = t.view(-1, *[1 for _ in range(len(x_t.shape) - 1)])
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * noise) / (1 - t)

    def get_v(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (1 - self.sigma_min) * noise - x_0

    def get_cond(self, cond, **kwargs):
        return cond

    def get_inference_cond(self, cond, **kwargs):
        return {"cond": cond, **kwargs}

    def get_sampler(self, **kwargs) -> samplers.FlowEulerSampler:
        return samplers.FlowEulerSampler(self.sigma_min)

    def vis_cond(self, **kwargs):
        return {}

    def _filter_model_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in kwargs.items() if k not in self._NON_MODEL_KWARGS}

    def sample_t(self, batch_size: int) -> torch.Tensor:
        if self.t_schedule["name"] == "uniform":
            return torch.rand(batch_size)
        if self.t_schedule["name"] == "logitNormal":
            mean = self.t_schedule["args"]["mean"]
            std = self.t_schedule["args"]["std"]
            return torch.sigmoid(torch.randn(batch_size) * std + mean)
        raise ValueError(f"Unknown t_schedule: {self.t_schedule['name']}")

    def training_losses(self, x_0: torch.Tensor, cond=None, **kwargs) -> Tuple[Dict, Dict]:
        model_kwargs = self._filter_model_kwargs(kwargs)
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **model_kwargs)

        pred = self.training_models["denoiser"](x_t, t * 1000, cond, **model_kwargs)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)
        terms = edict()
        terms["mse"] = F.mse_loss(pred, target)
        terms["loss"] = terms["mse"]

        mse_per_instance = np.array([F.mse_loss(pred[i], target[i]).item() for i in range(x_0.shape[0])])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}
        return terms, {}

    @torch.no_grad()
    def run_snapshot(self, num_samples: int, batch_size: int, verbose: bool = False) -> Dict:
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None,
        )

        sampler = self.get_sampler()
        sample_gt = []
        sample = []
        cond_vis = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            data = {k: v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch] for k, v in data.items()}
            noise = torch.randn_like(data["x_0"])
            sample_gt.append(data["x_0"])
            cond_vis.append(self.vis_cond(**data))
            del data["x_0"]
            args = self.get_inference_cond(**data)
            res = sampler.sample(
                self.models["denoiser"],
                noise=noise,
                **args,
                steps=50,
                cfg_strength=3.0,
                verbose=verbose,
            )
            sample.append(res.samples)

        sample_gt = torch.cat(sample_gt, dim=0)
        sample = torch.cat(sample, dim=0)
        sample_dict = {
            "sample_gt": {"value": sample_gt, "type": "sample"},
            "sample": {"value": sample, "type": "sample"},
        }
        sample_dict.update(
            dict_reduce(
                cond_vis,
                None,
                {
                    "value": lambda x: torch.cat(x, dim=0),
                    "type": lambda x: x[0],
                },
            )
        )
        return sample_dict


class FlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, FlowMatchingTrainer):
    pass


class VideoConditionedFlowMatchingCFGTrainer(VideoConditionedMixin, FlowMatchingCFGTrainer):
    def _resolve_num_frames(self, num_frames) -> int:
        if isinstance(num_frames, int):
            return int(num_frames)
        if isinstance(num_frames, torch.Tensor):
            if num_frames.numel() == 1:
                return int(num_frames.item())
            unique = torch.unique(num_frames)
            if unique.numel() != 1:
                raise ValueError(f"Mixed num_frames in a batch is not supported: {unique.tolist()}")
            return int(unique[0].item())
        if isinstance(num_frames, (list, tuple)):
            unique = set([int(x) for x in num_frames])
            if len(unique) != 1:
                raise ValueError(f"Mixed num_frames in a batch is not supported: {sorted(unique)}")
            return int(next(iter(unique)))
        raise ValueError(f"Unsupported num_frames type: {type(num_frames)}")

    def _flatten_sequence_batch(self, x_0, cond, num_frames):
        num_frames = self._resolve_num_frames(num_frames)
        if x_0.ndim < 6:
            raise ValueError(f"Expected dense video latent tensor [B,T,C,D,H,W], got {x_0.shape}")
        batch_size = x_0.shape[0]
        x_0 = x_0.reshape(batch_size * num_frames, *x_0.shape[2:])
        if isinstance(cond, torch.Tensor):
            cond = cond.reshape(batch_size * num_frames, *cond.shape[2:])
        return x_0, cond, num_frames, batch_size

    def training_losses(self, x_0: torch.Tensor, cond=None, num_frames=None, **kwargs) -> Tuple[Dict, Dict]:
        model_kwargs = self._filter_model_kwargs(kwargs)
        x_0, cond, num_frames, _ = self._flatten_sequence_batch(x_0, cond, num_frames)
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **model_kwargs)

        pred = self.training_models["denoiser"](x_t, t * 1000, cond, num_frames=num_frames, **model_kwargs)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)
        terms = edict()
        terms["mse"] = F.mse_loss(pred, target)
        terms["loss"] = terms["mse"]

        mse_per_instance = np.array([F.mse_loss(pred[i], target[i]).item() for i in range(x_0.shape[0])])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}
        return terms, {"num_frames": float(num_frames)}

    @torch.no_grad()
    def run_snapshot(self, num_samples: int, batch_size: int, verbose: bool = False) -> Dict:
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None,
        )

        sampler = self.get_sampler()
        sample_gt_vis = []
        sample_vis = []
        cond_vis = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            data = {
                k: v[:batch].cuda() if isinstance(v, torch.Tensor) and v.shape[0] >= batch else v
                for k, v in data.items()
            }
            x_0, cond, num_frames, _ = self._flatten_sequence_batch(data["x_0"][:batch], data["cond"][:batch], data["num_frames"])
            noise = torch.randn_like(x_0)
            sample_gt_vis.append(self.dataset.visualize_latent(x_0, num_frames))
            cond_vis.append(self.dataset.visualize_condition(data["cond"][:batch]))
            args = self.get_inference_cond(cond=cond)
            res = sampler.sample(
                self.models["denoiser"],
                noise=noise,
                **args,
                steps=25,
                cfg_strength=5.0,
                rescale_t=3.0,
                num_frames=num_frames,
                verbose=verbose,
            )
            sample_vis.append(self.dataset.visualize_latent(res.samples, num_frames))

        return {
            "sample_gt": {"value": torch.cat(sample_gt_vis, dim=0), "type": "image"},
            "sample": {"value": torch.cat(sample_vis, dim=0), "type": "image"},
            "cond": {"value": torch.cat(cond_vis, dim=0), "type": "image"},
        }
