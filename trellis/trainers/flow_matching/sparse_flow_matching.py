import copy
import functools

from typing import *

import numpy as np
import torch
import torch.nn.functional as F

from easydict import EasyDict as edict
from torch.utils.data import DataLoader

from ...modules import sparse as sp
from ...utils.data_utils import BalancedResumableSampler, cycle, recursive_to_device
from .flow_matching import FlowMatchingTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.video_conditioned import VideoConditionedMixin


class SparseFlowMatchingTrainer(FlowMatchingTrainer):
    def prepare_dataloader(self, **kwargs):
        self.data_sampler = BalancedResumableSampler(
            self.dataset,
            shuffle=True,
            batch_size=self.batch_size_per_gpu,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=functools.partial(self.dataset.collate_fn, split_size=self.batch_split),
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    def training_losses(self, x_0: sp.SparseTensor, cond=None, **kwargs) -> Tuple[Dict, Dict]:
        model_kwargs = self._filter_model_kwargs(kwargs)
        noise = x_0.replace(torch.randn_like(x_0.feats))
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **model_kwargs)

        pred = self.training_models["denoiser"](x_t, t * 1000, cond, **model_kwargs)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)
        terms = edict()
        terms["mse"] = F.mse_loss(pred.feats, target.feats)
        terms["loss"] = terms["mse"]

        mse_per_instance = np.array(
            [F.mse_loss(pred.feats[x_0.layout[i]], target.feats[x_0.layout[i]]).item() for i in range(x_0.shape[0])]
        )
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}
        return terms, {}


class SparseFlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, SparseFlowMatchingTrainer):
    pass


class VideoConditionedSparseFlowMatchingCFGTrainer(VideoConditionedMixin, SparseFlowMatchingCFGTrainer):
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

    def training_losses(self, x_0: sp.SparseTensor, cond=None, num_frames=None, **kwargs) -> Tuple[Dict, Dict]:
        model_kwargs = self._filter_model_kwargs(kwargs)
        num_frames = self._resolve_num_frames(num_frames)
        noise = x_0.replace(torch.randn_like(x_0.feats))
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **model_kwargs)

        pred = self.training_models["denoiser"](x_t, t * 1000, cond, num_frames=num_frames, **model_kwargs)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)
        terms = edict()
        terms["mse"] = F.mse_loss(pred.feats, target.feats)
        terms["loss"] = terms["mse"]

        mse_per_instance = np.array(
            [F.mse_loss(pred.feats[x_0.layout[i]], target.feats[x_0.layout[i]]).item() for i in range(x_0.shape[0])]
        )
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
            num_frames = self._resolve_num_frames(data["num_frames"])
            # Keep snapshot tensors/sparse tensors on the trainer device.
            data = {
                k: (
                    v[: batch * num_frames]
                    if (k == "x_0" and isinstance(v, sp.SparseTensor) and v.shape[0] >= batch * num_frames)
                    else (v[:batch] if (isinstance(v, torch.Tensor) and v.shape[0] >= batch) else v)
                )
                for k, v in data.items()
            }
            data = recursive_to_device(data, self.device)
            x_0 = data["x_0"]
            noise = x_0.replace(torch.randn_like(x_0.feats))
            sample_gt_vis.append(self.dataset.visualize_latent(x_0, num_frames))
            cond_vis.append(self.dataset.visualize_condition(data["cond"][:batch]))
            args = self.get_inference_cond(cond=data["cond"][:batch])
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
