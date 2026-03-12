import copy
import glob
import json
import os
import threading
import time

from abc import abstractmethod
from contextlib import nullcontext
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import wandb  # type: ignore

from safetensors.torch import load_file
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils

from ..utils import elastic_utils, grad_clip_utils
from ..utils.data_utils import ResumableSampler, cycle, recursive_to_device
from ..utils.dist_utils import *
from ..utils.general_utils import *
from .utils import *


class BasicTrainer:
    def __init__(
        self,
        models,
        dataset,
        *,
        output_dir,
        load_dir,
        step,
        max_steps,
        batch_size=None,
        batch_size_per_gpu=None,
        batch_split=None,
        optimizer={},
        lr_scheduler=None,
        elastic=None,
        grad_clip=None,
        ema_rate=0.9999,
        mix_precision_mode="inflat_all",
        mix_precision_dtype="float16",
        fp16_mode=None,
        fp16_scale_growth=1e-3,
        parallel_mode="ddp",
        finetune_ckpt=None,
        log_param_stats=False,
        prefetch_data=True,
        snapshot_batch_size=4,
        i_print=1000,
        i_log=500,
        i_sample=10000,
        i_save=10000,
        i_save_persist=100000,
        i_ddpcheck=1000,
        max_keep_ckpts=None,
        num_workers=None,
        dataloader_prefetch_factor=2,
        dataloader_persistent_workers=True,
        **kwargs,
    ):
        assert batch_size is not None or batch_size_per_gpu is not None

        self.models = models
        self.dataset = dataset
        self.batch_split = batch_split if batch_split is not None else 1
        self.max_steps = max_steps
        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler
        self.elastic_controller_config = elastic
        self.grad_clip = grad_clip
        self.ema_rate = [ema_rate] if isinstance(ema_rate, float) else ema_rate
        if fp16_mode is not None:
            mix_precision_dtype = "float16"
            mix_precision_mode = fp16_mode
        self.mix_precision_mode = mix_precision_mode
        self.mix_precision_dtype = str_to_dtype(mix_precision_dtype)
        self.fp16_scale_growth = fp16_scale_growth
        self.parallel_mode = parallel_mode
        self.log_param_stats = log_param_stats
        self.prefetch_data = prefetch_data
        self.snapshot_batch_size = snapshot_batch_size
        self.log = []
        if self.prefetch_data:
            self._data_prefetched = None

        self.output_dir = output_dir
        self.i_print = i_print
        self.i_log = i_log
        self.i_sample = i_sample
        self.i_save = i_save
        self.i_save_persist = i_save_persist
        self.i_ddpcheck = i_ddpcheck
        self.max_keep_ckpts = max_keep_ckpts
        self.num_workers = (
            num_workers if num_workers is not None else int(np.ceil(os.cpu_count() / torch.cuda.device_count()))
        )
        self.dataloader_prefetch_factor = dataloader_prefetch_factor
        self.dataloader_persistent_workers = dataloader_persistent_workers

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = dist.get_rank() % torch.cuda.device_count()
            self.is_master = self.rank == 0
        else:
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.is_master = True

        self.batch_size = batch_size if batch_size_per_gpu is None else batch_size_per_gpu * self.world_size
        self.batch_size_per_gpu = (
            batch_size_per_gpu if batch_size_per_gpu is not None else batch_size // self.world_size
        )
        assert self.batch_size % self.world_size == 0
        assert self.batch_size_per_gpu % self.batch_split == 0

        self.init_models_and_more(**kwargs)
        self.prepare_dataloader(**kwargs)

        self.step = 0
        if load_dir is not None and step is not None:
            self.load(load_dir, step)
        elif finetune_ckpt is not None:
            self.finetune_from(finetune_ckpt)

        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, "ckpts"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "samples"), exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.output_dir, "tb_logs"))
            self.use_wandb = getattr(wandb, "run", None) is not None

        if self.world_size > 1:
            if hasattr(self, "sync_ddp_params"):
                try:
                    self.sync_ddp_params()  # type: ignore[attr-defined]
                except Exception as e:
                    if self.is_master:
                        print(f"Warning: sync_ddp_params() failed: {e}")
            self.check_ddp()

        if self.is_master:
            print("\n\nTrainer initialized.")
            print(self)

    def __str__(self):
        lines = [self.__class__.__name__]
        lines.append("  - Models:")
        for name, model in self.models.items():
            lines.append(f"    - {name}: {model.__class__.__name__}")
        lines.append(f"  - Dataset: {indent(str(self.dataset), 2)}")
        lines.append("  - Dataloader:")
        sampler_name = "None" if self.data_sampler is None else self.data_sampler.__class__.__name__
        lines.append(f"    - Sampler: {sampler_name}")
        lines.append(f"    - Num workers: {self.dataloader.num_workers}")
        lines.append(f"  - Number of steps: {self.max_steps}")
        lines.append(f"  - Number of GPUs: {self.world_size}")
        lines.append(f"  - Batch size: {self.batch_size}")
        lines.append(f"  - Batch size per GPU: {self.batch_size_per_gpu}")
        lines.append(f"  - Batch split: {self.batch_split}")
        lines.append(f"  - Optimizer: {self.optimizer.__class__.__name__}")
        lines.append(f'  - Learning rate: {self.optimizer.param_groups[0]["lr"]}')
        if self.lr_scheduler_config is not None:
            lines.append(f"  - LR scheduler: {self.lr_scheduler.__class__.__name__}")
        if self.elastic_controller_config is not None:
            lines.append(f"  - Elastic memory: {indent(str(self.elastic_controller), 2)}")
        if self.grad_clip is not None:
            lines.append(f"  - Gradient clip: {indent(str(self.grad_clip), 2)}")
        lines.append(f"  - EMA rate: {self.ema_rate}")
        lines.append(f"  - Mixed precision dtype: {self.mix_precision_dtype}")
        lines.append(f"  - Mixed precision mode: {self.mix_precision_mode}")
        if self.mix_precision_mode == "amp" and self.mix_precision_dtype == torch.float16:
            lines.append(f"  - FP16 scale growth: {self.fp16_scale_growth}")
        lines.append(f"  - Parallel mode: {self.parallel_mode}")
        return "\n".join(lines)

    @property
    def device(self):
        for _, model in self.models.items():
            if hasattr(model, "device"):
                return model.device
        return next(list(self.models.values())[0].parameters()).device

    def init_models_and_more(self, **kwargs):
        self.save_lora_only = False

        trainable_param_names = kwargs.get("trainable_param_names", None)
        frozen_model_names = kwargs.get("frozen_model_names", [])
        if isinstance(frozen_model_names, str):
            frozen_model_names = [frozen_model_names]
        self.frozen_model_names = set(frozen_model_names)

        if self.is_master:
            print(f"Trainable parameter names: {trainable_param_names}")
            if len(self.frozen_model_names) > 0:
                print(f"Frozen models (excluded from optimization): {self.frozen_model_names}")

        self.trainable_param_names = None
        if trainable_param_names is not None and len(trainable_param_names) > 0:
            self.trainable_param_names = set([str(x) for x in trainable_param_names])
            num_trainable = 0
            num_total = 0
            for model_name, model in self.models.items():
                for param_name, param in model.named_parameters():
                    full_name = f"{model_name}.{param_name}"
                    is_trainable = False
                    for trainable_name in self.trainable_param_names:
                        if trainable_name in param_name or trainable_name in full_name:
                            is_trainable = True
                            break
                    param.requires_grad = bool(is_trainable)
                    num_total += 1
                    if param.requires_grad:
                        num_trainable += 1
            if self.is_master:
                print(
                    f"Trainable parameter filtering enabled: {num_trainable}/{num_total} namedparams will be optimized."
                )

        if self.world_size > 1:
            self.training_models = {}
            for name, model in self.models.items():
                has_trainable = any(p.requires_grad for p in model.parameters())
                if not has_trainable:
                    if self.is_master:
                        print(f"Skipping DDP for frozen model {name} (no trainable params)")
                    continue
                ddp_kwargs = dict(
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    bucket_cap_mb=128,
                    find_unused_parameters=False,
                )
                self.training_models[name] = DDP(model, **ddp_kwargs)
        else:
            self.training_models = self.models

        self.model_params = []
        self.model_param_names = []
        for model_name, model in self.models.items():
            if model_name in self.frozen_model_names:
                continue
            for param_name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                self.model_params.append(param)
                self.model_param_names.append(f"{model_name}.{param_name}")

        if self.mix_precision_mode == "amp":
            self.master_params = self.model_params
            if self.mix_precision_dtype == torch.float16:
                self.scaler = torch.GradScaler()
        elif self.mix_precision_mode == "inflat_all":
            self.master_params = make_master_params(self.model_params)
            if self.mix_precision_dtype == torch.float16:
                self.log_scale = 20.0
        elif self.mix_precision_mode is None:
            self.master_params = self.model_params
        else:
            raise NotImplementedError(f"Mix precision mode {self.mix_precision_mode} is not implemented.")

        if self.is_master:
            self.ema_params = [copy.deepcopy(self.master_params) for _ in self.ema_rate]

        if hasattr(torch.optim, self.optimizer_config["name"]):
            self.optimizer = getattr(torch.optim, self.optimizer_config["name"])(
                self.master_params, **self.optimizer_config["args"]
            )
        else:
            self.optimizer = globals()[self.optimizer_config["name"]](
                self.master_params, **self.optimizer_config["args"]
            )

        if self.lr_scheduler_config is not None:
            if hasattr(torch.optim.lr_scheduler, self.lr_scheduler_config["name"]):
                self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_config["name"])(
                    self.optimizer, **self.lr_scheduler_config["args"]
                )
            else:
                self.lr_scheduler = globals()[self.lr_scheduler_config["name"]](
                    self.optimizer, **self.lr_scheduler_config["args"]
                )

        if self.elastic_controller_config is not None:
            assert any(
                [
                    isinstance(model, (elastic_utils.ElasticModule, elastic_utils.ElasticModuleMixin))
                    for model in self.models.values()
                ]
            ), "No elastic module found in models, please inherit from ElasticModule or ElasticModuleMixin"
            self.elastic_controller = getattr(elastic_utils, self.elastic_controller_config["name"])(
                **self.elastic_controller_config["args"]
            )
            for model in self.models.values():
                if isinstance(model, (elastic_utils.ElasticModule, elastic_utils.ElasticModuleMixin)):
                    model.register_memory_controller(self.elastic_controller)

        if self.grad_clip is not None:
            if isinstance(self.grad_clip, (float, int)):
                self.grad_clip = float(self.grad_clip)
            else:
                self.grad_clip = getattr(grad_clip_utils, self.grad_clip["name"])(**self.grad_clip["args"])

    def prepare_dataloader(self, **kwargs):
        base_loader_kwargs = {
            "batch_size": self.batch_size_per_gpu,
            "num_workers": self.num_workers,
            "pin_memory": True,
            "drop_last": True,
            "collate_fn": self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None,
        }
        if self.num_workers > 0:
            base_loader_kwargs["persistent_workers"] = bool(self.dataloader_persistent_workers)
            if self.dataloader_prefetch_factor is not None:
                base_loader_kwargs["prefetch_factor"] = int(self.dataloader_prefetch_factor)

        if isinstance(self.dataset, IterableDataset):
            self.data_sampler = None
            self.dataloader = DataLoader(self.dataset, **base_loader_kwargs)
        else:
            self.data_sampler = ResumableSampler(self.dataset, shuffle=True)
            self.dataloader = DataLoader(self.dataset, **base_loader_kwargs, sampler=self.data_sampler)
        self.data_iterator = cycle(self.dataloader)

    @abstractmethod
    def run_snapshot(self, num_samples, batch_size=4, verbose=False, **kwargs):
        pass

    @torch.no_grad()
    def visualize_sample(self, sample):
        if hasattr(self.dataset, "visualize_sample"):
            return self.dataset.visualize_sample(sample)
        return sample

    @torch.no_grad()
    def snapshot_dataset(self, num_samples=100, batch_size=None):
        if batch_size is not None:
            num_samples = int(batch_size)
        if isinstance(self.dataset, IterableDataset):
            dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=num_samples,
                num_workers=0,
                collate_fn=self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None,
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=num_samples,
                num_workers=0,
                shuffle=True,
                collate_fn=self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None,
            )
        data = next(iter(dataloader))
        data = recursive_to_device(data, self.device)
        vis = self.visualize_sample(data)
        if isinstance(vis, dict):
            save_cfg = [(f"dataset_{k}", v) for k, v in vis.items()]
        else:
            save_cfg = [("dataset", vis)]
        images_to_log = {}
        for name, image in save_cfg:
            utils.save_image(
                image,
                os.path.join(self.output_dir, "samples", f"{name}.jpg"),
                nrow=int(np.sqrt(num_samples)),
                normalize=True,
                value_range=self.dataset.value_range,
            )
            if getattr(self, "use_wandb", False):
                grid = utils.make_grid(
                    image,
                    nrow=int(np.sqrt(num_samples)),
                    normalize=True,
                    value_range=self.dataset.value_range,
                )
                images_to_log[f"{name}"] = wandb.Image(grid, caption=name)
        if getattr(self, "use_wandb", False) and len(images_to_log) > 0:
            wandb.log(images_to_log, step=self.step)

    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=4, batch_size=4, verbose=False):
        if self.is_master:
            print(f"\nSampling {num_samples} images...", end="")

        if suffix is None:
            suffix = f"step{self.step:07d}"

        num_samples_per_process = int(np.ceil(num_samples / self.world_size))
        samples = self.run_snapshot(num_samples_per_process, batch_size=batch_size, verbose=verbose)

        if self.world_size > 1:
            for key in samples.keys():
                samples[key]["value"] = samples[key]["value"].contiguous()
                if self.is_master:
                    all_images = [torch.empty_like(samples[key]["value"]) for _ in range(self.world_size)]
                else:
                    all_images = []
                dist.gather(samples[key]["value"], all_images, dst=0)
                if self.is_master:
                    samples[key]["value"] = torch.cat(all_images, dim=0)[:num_samples]

        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, "samples", suffix), exist_ok=True)
            images_to_log = {}
            for key in samples.keys():
                img_value = samples[key]["value"]
                if samples[key]["type"] == "image":
                    utils.save_image(
                        img_value,
                        os.path.join(self.output_dir, "samples", suffix, f"{key}_{suffix}.jpg"),
                        nrow=int(np.sqrt(num_samples)),
                        normalize=True,
                        value_range=self.dataset.value_range,
                    )
                    if getattr(self, "use_wandb", False):
                        grid = utils.make_grid(
                            img_value,
                            nrow=int(np.sqrt(num_samples)),
                            normalize=True,
                            value_range=self.dataset.value_range,
                        )
                        images_to_log[f"samples/{key}"] = wandb.Image(grid, caption=f"{key}_{suffix}")
                elif samples[key]["type"] == "number":
                    minv = img_value.min()
                    maxv = img_value.max()
                    images = (img_value - minv) / (maxv - minv)
                    images = utils.make_grid(images, nrow=int(np.sqrt(num_samples)), normalize=False)
                    save_image_with_notes(
                        images,
                        os.path.join(self.output_dir, "samples", suffix, f"{key}_{suffix}.jpg"),
                        notes=f"{key} min: {minv}, max: {maxv}",
                    )
                    if getattr(self, "use_wandb", False):
                        images_to_log[f"samples/{key}"] = wandb.Image(images, caption=f"{key}_{suffix}")
            if getattr(self, "use_wandb", False) and len(images_to_log) > 0:
                wandb.log(images_to_log, step=self.step)

        if self.is_master:
            print(" Done.")

    @abstractmethod
    def training_losses(**mb_data):
        pass

    def load_data(self):
        if self.prefetch_data:
            if self._data_prefetched is None:
                self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
            data = self._data_prefetched
            self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
        else:
            data = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)

        if isinstance(data, dict):
            if self.batch_split == 1:
                data_list = [data]
            else:
                batch_size = None
                for value in data.values():
                    if isinstance(value, torch.Tensor):
                        batch_size = value.shape[0]
                        break
                if batch_size is None:
                    raise ValueError("Could not infer batch size from minibatch dict.")
                data_list = []
                for i in range(self.batch_split):
                    start = i * batch_size // self.batch_split
                    end = (i + 1) * batch_size // self.batch_split
                    split_pack = {}
                    for k, v in data.items():
                        if isinstance(v, torch.Tensor):
                            split_pack[k] = v[start:end]
                        else:
                            split_pack[k] = v
                    data_list.append(split_pack)
        elif isinstance(data, list):
            data_list = data
        else:
            raise ValueError("Data must be a dict or a list of dicts.")
        return data_list

    def save_logs(self):
        log_str = "\n".join([f"{step}: {json.dumps(dict_foreach(log, lambda x: float(x)))}" for step, log in self.log])
        with open(os.path.join(self.output_dir, "log.txt"), "a") as log_file:
            log_file.write(log_str + "\n")

        log_show = [l for _, l in self.log if not dict_any(l, lambda x: np.isnan(x))]
        log_show = dict_reduce(log_show, lambda x: np.mean(x))
        log_show = dict_flatten(log_show, sep="/")
        for key, value in log_show.items():
            self.writer.add_scalar(key, value, self.step)
        self.log = []

    def run(self):
        if self.is_master:
            print("\nStarting training...")
            self.snapshot_dataset(batch_size=self.snapshot_batch_size)
        if self.step == 0:
            self.snapshot(suffix="init", batch_size=self.snapshot_batch_size)
        else:
            self.snapshot(suffix=f"resume_step{self.step:07d}", batch_size=self.snapshot_batch_size)

        time_last_print = 0.0
        time_elapsed = 0.0
        while self.step < self.max_steps:
            time_start = time.time()
            data_list = self.load_data()
            step_log = self.run_step(data_list)
            time_end = time.time()
            time_elapsed += time_end - time_start

            self.step += 1

            if self.is_master and self.step % self.i_print == 0:
                speed = self.i_print / (time_elapsed - time_last_print) * 3600
                columns = [
                    f"Step: {self.step}/{self.max_steps} ({self.step / self.max_steps * 100:.2f}%)",
                    f"Elapsed: {time_elapsed / 3600:.2f} h",
                    f"Speed: {speed:.2f} steps/h",
                    f"ETA: {(self.max_steps - self.step) / speed:.2f} h",
                ]
                print(" | ".join([c.ljust(25) for c in columns]), flush=True)
                time_last_print = time_elapsed

            if self.world_size > 1 and self.i_ddpcheck is not None and self.step % self.i_ddpcheck == 0:
                self.check_ddp()

            if self.step % self.i_sample == 0:
                self.snapshot()

            if self.is_master:
                self.log.append((self.step, {}))
                self.log[-1][1]["time"] = {"step": time_end - time_start, "elapsed": time_elapsed}
                if step_log is not None:
                    self.log[-1][1].update(step_log)
                if self.mix_precision_dtype == torch.float16:
                    if self.mix_precision_mode == "amp":
                        self.log[-1][1]["scale"] = self.scaler.get_scale()
                    elif self.mix_precision_mode == "inflat_all":
                        self.log[-1][1]["log_scale"] = self.log_scale
                if self.step % self.i_log == 0:
                    self.save_logs()

                save_due = (self.i_save > 0) and (self.step % self.i_save == 0)
                persist_due = (self.i_save_persist > 0) and (self.step % self.i_save_persist == 0)
                if save_due or persist_due:
                    self.save()

            self.check_abort()

        self.snapshot(suffix="final", batch_size=self.snapshot_batch_size)
        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            self.writer.close()
            print("Training finished.")

    def profile(self, wait=2, warmup=3, active=5):
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.output_dir, "profile")),
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(wait + warmup + active):
                self.run_step()
                prof.step()

    def sync_ddp_params(self):
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return
        with torch.no_grad():
            for _, model in self.models.items():
                for p in model.parameters():
                    dist.broadcast(p.data, src=0)
                for b in model.buffers():
                    dist.broadcast(b.data, src=0)
            if hasattr(self, "master_params"):
                for p in self.master_params:
                    dist.broadcast(p.data, src=0)
            if getattr(self, "fp16_mode", None) == "inflat_all" and hasattr(self, "model_params"):
                master_params_to_model_params(self.model_params, self.master_params)

    def _master_params_to_state_dicts(self, master_params):
        if self.mix_precision_mode == "inflat_all":
            master_params = unflatten_master_params(self.model_params, master_params)
        state_dicts = {name: model.state_dict() for name, model in self.models.items()}
        master_params_names = sum(
            [
                [(name, n) for n, p in model.named_parameters() if p.requires_grad]
                for name, model in self.models.items()
                if name not in self.frozen_model_names
            ],
            [],
        )
        for i, (model_name, param_name) in enumerate(master_params_names):
            state_dicts[model_name][param_name] = master_params[i]
        return state_dicts

    def _state_dicts_to_master_params(self, master_params, state_dicts):
        master_params_names = sum(
            [
                [(name, n) for n, p in model.named_parameters() if p.requires_grad]
                for name, model in self.models.items()
                if name not in self.frozen_model_names
            ],
            [],
        )
        params = [state_dicts[name][param_name] for name, param_name in master_params_names]
        if self.mix_precision_mode == "inflat_all":
            model_params_to_master_params(params, master_params)
        else:
            for i, param in enumerate(params):
                master_params[i].data.copy_(param.data)

    def _resolve_pretrained_path(self, path: str) -> str:
        if os.path.exists(path):
            return path
        if os.path.exists(f"{path}.safetensors"):
            return f"{path}.safetensors"
        if os.path.exists(f"{path}.pt"):
            return f"{path}.pt"
        path_parts = path.split("/")
        if len(path_parts) >= 3:
            from huggingface_hub import hf_hub_download

            repo_id = f"{path_parts[0]}/{path_parts[1]}"
            model_name = "/".join(path_parts[2:])
            try:
                return hf_hub_download(repo_id, f"{model_name}.safetensors")
            except Exception:
                return hf_hub_download(repo_id, f"{model_name}.pt")
        return path

    def load(self, load_dir, step=0):
        if self.is_master:
            print(f"\nLoading checkpoint from step {step}...", end="")

        model_ckpts = {}
        for name, model in self.models.items():
            full_path = os.path.join(load_dir, "ckpts", f"{name}_step{step:07d}.pt")
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"No checkpoint found for model {name} at step {step}")
            model_ckpt = torch.load(read_file_dist(full_path), map_location=self.device, weights_only=True)
            model_ckpts[name] = model_ckpt
            model.load_state_dict(model_ckpt)

        self._state_dicts_to_master_params(self.master_params, model_ckpts)
        del model_ckpts

        if self.is_master:
            for i, ema_rate in enumerate(self.ema_rate):
                ema_ckpts = {}
                for name, model in self.models.items():
                    ema_ckpt = torch.load(
                        os.path.join(load_dir, "ckpts", f"{name}_ema{ema_rate}_step{step:07d}.pt"),
                        map_location=self.device,
                        weights_only=True,
                    )
                    ema_ckpts[name] = ema_ckpt
                self._state_dicts_to_master_params(self.ema_params[i], ema_ckpts)
                del ema_ckpts

        misc_ckpt = torch.load(
            read_file_dist(os.path.join(load_dir, "ckpts", f"misc_step{step:07d}.pt")),
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        self.optimizer.load_state_dict(misc_ckpt["optimizer"])
        self.step = misc_ckpt["step"]
        if self.data_sampler is not None and "data_sampler" in misc_ckpt:
            self.data_sampler.load_state_dict(misc_ckpt["data_sampler"])
        if self.mix_precision_mode == "amp" and self.mix_precision_dtype == torch.float16:
            self.scaler.load_state_dict(misc_ckpt["scaler"])
        elif self.mix_precision_mode == "inflat_all" and self.mix_precision_dtype == torch.float16:
            self.log_scale = misc_ckpt["log_scale"]
        if self.lr_scheduler_config is not None:
            self.lr_scheduler.load_state_dict(misc_ckpt["lr_scheduler"])
        if self.elastic_controller_config is not None:
            self.elastic_controller.load_state_dict(misc_ckpt["elastic_controller"])
        if self.grad_clip is not None and not isinstance(self.grad_clip, float):
            self.grad_clip.load_state_dict(misc_ckpt["grad_clip"])
        del misc_ckpt

        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print(" Done.")
        if self.world_size > 1:
            self.check_ddp()

    def save(self, non_blocking=True):
        assert self.is_master
        save_due = (self.i_save is not None) and (int(self.i_save) > 0) and (self.step % int(self.i_save) == 0)
        persist_due = (
            (getattr(self, "i_save_persist", None) is not None)
            and (int(self.i_save_persist) > 0)
            and (self.step % int(self.i_save_persist) == 0)
        )
        if not (save_due or persist_due):
            return

        print(f"\nSaving checkpoint at step {self.step}...", end="")

        model_ckpts = self._master_params_to_state_dicts(self.master_params)
        ema_ckpts_all = [self._master_params_to_state_dicts(self.ema_params[i]) for i in range(len(self.ema_rate))]
        misc_ckpt = {"optimizer": self.optimizer.state_dict(), "step": self.step}
        if self.data_sampler is not None:
            misc_ckpt["data_sampler"] = self.data_sampler.state_dict()
        if self.mix_precision_mode == "amp" and self.mix_precision_dtype == torch.float16:
            misc_ckpt["scaler"] = self.scaler.state_dict()
        elif self.mix_precision_mode == "inflat_all" and self.mix_precision_dtype == torch.float16:
            misc_ckpt["log_scale"] = self.log_scale
        if self.lr_scheduler_config is not None:
            misc_ckpt["lr_scheduler"] = self.lr_scheduler.state_dict()
        if self.elastic_controller_config is not None:
            misc_ckpt["elastic_controller"] = self.elastic_controller.state_dict()
        if self.grad_clip is not None and not isinstance(self.grad_clip, float):
            misc_ckpt["grad_clip"] = self.grad_clip.state_dict()

        def _write_to_dir(ckpt_dir: str):
            os.makedirs(ckpt_dir, exist_ok=True)
            for name, model_ckpt in model_ckpts.items():
                model_ckpt = {k: v.cpu() for k, v in model_ckpt.items()}
                out_path = os.path.join(ckpt_dir, f"{name}_step{self.step:07d}.pt")
                if non_blocking:
                    threading.Thread(target=torch.save, args=(model_ckpt, out_path)).start()
                else:
                    torch.save(model_ckpt, out_path)
            for i, ema_rate in enumerate(self.ema_rate):
                ema_ckpts = ema_ckpts_all[i]
                for name, ema_ckpt in ema_ckpts.items():
                    ema_ckpt = {k: v.cpu() for k, v in ema_ckpt.items()}
                    out_path = os.path.join(ckpt_dir, f"{name}_ema{ema_rate}_step{self.step:07d}.pt")
                    if non_blocking:
                        threading.Thread(target=torch.save, args=(ema_ckpt, out_path)).start()
                    else:
                        torch.save(ema_ckpt, out_path)
            out_path = os.path.join(ckpt_dir, f"misc_step{self.step:07d}.pt")
            if non_blocking:
                threading.Thread(target=torch.save, args=(misc_ckpt, out_path)).start()
            else:
                torch.save(misc_ckpt, out_path)

        if save_due:
            regular_dir = os.path.join(self.output_dir, "ckpts")
            _write_to_dir(regular_dir)
            if getattr(self, "max_keep_ckpts", None):
                try:
                    misc_files = glob.glob(os.path.join(regular_dir, "misc_step*.pt"))
                    steps = sorted([int(os.path.basename(f).split("step")[-1].split(".")[0]) for f in misc_files])
                    if len(steps) > int(self.max_keep_ckpts):
                        to_remove = steps[: -int(self.max_keep_ckpts)]
                        for s in to_remove:
                            pattern = os.path.join(regular_dir, f"*step{int(s):07d}.pt")
                            for f in glob.glob(pattern):
                                try:
                                    os.remove(f)
                                except FileNotFoundError:
                                    pass
                except Exception as e:
                    print(f"\nWarning: checkpoint pruning failed: {e}")

        if persist_due:
            persist_dir = os.path.join(self.output_dir, "ckpts_persist")
            _write_to_dir(persist_dir)

        print(" Done.")

    def finetune_from(self, finetune_ckpt):
        if self.is_master:
            print("\nFinetuning from:")
            for name, path in finetune_ckpt.items():
                print(f"  - {name}: {path}")

        model_ckpts = {}
        for name, model in self.models.items():
            model_state_dict = model.state_dict()
            if name in finetune_ckpt:
                resolved = self._resolve_pretrained_path(finetune_ckpt[name])
                if resolved.endswith(".safetensors"):
                    model_ckpt = load_file(resolved)
                else:
                    model_ckpt = torch.load(read_file_dist(resolved), map_location=self.device, weights_only=True)
                for k, v in list(model_ckpt.items()):
                    if k not in model_state_dict:
                        if self.is_master:
                            print(f"{k} is not found in model_state_dict")
                        continue
                    if v.shape != model_state_dict[k].shape:
                        if self.is_master:
                            print(f"Warning: {k} shape mismatch, {v.shape} vs {model_state_dict[k].shape}, skipped.")
                        model_ckpt[k] = model_state_dict[k]
                for _k, _v in model_state_dict.items():
                    if _k not in model_ckpt:
                        if self.is_master:
                            print(f"Warning: {_k} not found in finetune_ckpt, use model state dict.")
                        model_ckpt[_k] = _v
                for _k, _v in list(model_ckpt.items()):
                    if isinstance(_v, torch.Tensor):
                        model_ckpt[_k] = _v.to(self.device)
                model_ckpts[name] = model_ckpt
                model.load_state_dict(model_ckpt, strict=False)
            else:
                if self.is_master:
                    print(f"Warning: {name} not found in finetune_ckpt, skipped.")
                model_ckpts[name] = model_state_dict
        self._state_dicts_to_master_params(self.master_params, model_ckpts)
        if self.is_master:
            for i, _ in enumerate(self.ema_rate):
                self._state_dicts_to_master_params(self.ema_params[i], model_ckpts)
        del model_ckpts

        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print("Done.")
        if self.world_size > 1:
            self.check_ddp()

    def update_ema(self):
        assert self.is_master
        for i, ema_rate in enumerate(self.ema_rate):
            for master_param, ema_param in zip(self.master_params, self.ema_params[i]):
                ema_param.detach().mul_(ema_rate).add_(master_param, alpha=1.0 - ema_rate)

    def check_ddp(self):
        if not dist.is_initialized():
            return
        if self.is_master:
            print("\nPerforming DDP check...")
            print("Checking if parameters are consistent across processes...")

        dist.barrier()
        try:
            for p_idx, p in enumerate(self.master_params):
                p_name = (
                    self.model_param_names[p_idx]
                    if hasattr(self, "model_param_names") and p_idx < len(self.model_param_names)
                    else f"master_param[{p_idx}]"
                )
                for i in range(0, p.numel(), 10000000):
                    sub_size = min(10000000, p.numel() - i)
                    sub_p = p.detach().view(-1)[i : i + sub_size]
                    sub_p_gather = [torch.empty_like(sub_p) for _ in range(self.world_size)]
                    dist.all_gather(sub_p_gather, sub_p)
                    for r in range(self.world_size):
                        if not torch.equal(sub_p, sub_p_gather[r]):
                            raise AssertionError(
                                f"parameters are not consistent across processes for {p_name} at chunk offset {i}, rank {r}"
                            )
        except AssertionError as e:
            if self.is_master:
                print(f"\n\033[91mError: {e}\033[0m")
                print("DDP check failed.")
            raise e

        dist.barrier()
        if self.is_master:
            print("Done.")

    def run_step(self, data_list):
        step_log = {"loss": {}, "status": {}}
        amp_context = (
            partial(torch.autocast, device_type="cuda", dtype=self.mix_precision_dtype)
            if self.mix_precision_mode == "amp"
            else nullcontext
        )
        elastic_controller_context = (
            self.elastic_controller.record if self.elastic_controller_config is not None else nullcontext
        )

        losses = []
        statuses = []
        elastic_controller_logs = []
        zero_grad(self.model_params)
        for i, mb_data in enumerate(data_list):
            sync_contexts = (
                [self.training_models[name].no_sync for name in self.training_models]
                if i != len(data_list) - 1 and self.world_size > 1
                else [nullcontext]
            )
            with nested_contexts(*sync_contexts), elastic_controller_context():
                with amp_context():
                    loss, status = self.training_losses(**mb_data)
                    l = loss["loss"] / len(data_list)
                if self.mix_precision_mode == "amp" and self.mix_precision_dtype == torch.float16:
                    self.scaler.scale(l).backward()
                elif self.mix_precision_mode == "inflat_all" and self.mix_precision_dtype == torch.float16:
                    scaled_l = l * (2**self.log_scale)
                    scaled_l.backward()
                else:
                    l.backward()
            losses.append(dict_foreach(loss, lambda x: x.item() if isinstance(x, torch.Tensor) else x))
            statuses.append(dict_foreach(status, lambda x: x.item() if isinstance(x, torch.Tensor) else x))
            if self.elastic_controller_config is not None:
                elastic_controller_logs.append(self.elastic_controller.log())

        if self.grad_clip is not None:
            if self.mix_precision_mode == "amp" and self.mix_precision_dtype == torch.float16:
                self.scaler.unscale_(self.optimizer)
            elif self.mix_precision_mode == "inflat_all":
                model_grads_to_master_grads(self.model_params, self.master_params)
                if self.mix_precision_dtype == torch.float16:
                    self.master_params[0].grad.mul_(1.0 / (2**self.log_scale))
            if isinstance(self.grad_clip, float):
                grad_norm = torch.nn.utils.clip_grad_norm_(self.master_params, self.grad_clip)
            else:
                grad_norm = self.grad_clip(self.master_params)
            if torch.isfinite(grad_norm):
                statuses[-1]["grad_norm"] = grad_norm.item()

        if self.mix_precision_mode == "amp" and self.mix_precision_dtype == torch.float16:
            prev_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        elif self.mix_precision_mode == "inflat_all":
            if self.mix_precision_dtype == torch.float16:
                prev_scale = 2**self.log_scale
                if not any(not p.grad.isfinite().all() for p in self.model_params):
                    if self.grad_clip is None:
                        model_grads_to_master_grads(self.model_params, self.master_params)
                        self.master_params[0].grad.mul_(1.0 / (2**self.log_scale))
                    self.optimizer.step()
                    master_params_to_model_params(self.model_params, self.master_params)
                    self.log_scale += self.fp16_scale_growth
                else:
                    self.log_scale -= 1
            else:
                prev_scale = 1.0
                if self.grad_clip is None:
                    model_grads_to_master_grads(self.model_params, self.master_params)
                if not any(not p.grad.isfinite().all() for p in self.master_params):
                    self.optimizer.step()
                    master_params_to_model_params(self.model_params, self.master_params)
                else:
                    print("\n\033[93mWarning: NaN detected in gradients. Skipping update.\033[0m")
        else:
            prev_scale = 1.0
            if not any(not p.grad.isfinite().all() for p in self.model_params):
                self.optimizer.step()
            else:
                print("\n\033[93mWarning: NaN detected in gradients. Skipping update.\033[0m")

        if self.lr_scheduler_config is not None:
            statuses[-1]["lr"] = self.lr_scheduler.get_last_lr()[0]
            self.lr_scheduler.step()

        step_log["loss"] = dict_reduce(losses, lambda x: np.mean(x))
        step_log["status"] = dict_reduce(
            statuses, lambda x: np.mean(x), special_func={"min": lambda x: np.min(x), "max": lambda x: np.max(x)}
        )
        if self.elastic_controller_config is not None:
            step_log["elastic"] = dict_reduce(elastic_controller_logs, lambda x: np.mean(x))
        if self.grad_clip is not None:
            step_log["grad_clip"] = self.grad_clip if isinstance(self.grad_clip, float) else self.grad_clip.log()

        if self.log_param_stats:
            param_norms = {}
            param_grads = {}
            for name, param in self.backbone.named_parameters():
                if param.requires_grad:
                    param_norms[name] = param.norm().item()
                    if param.grad is not None and torch.isfinite(param.grad).all():
                        param_grads[name] = param.grad.norm().item() / prev_scale
            step_log["param_norms"] = param_norms
            step_log["param_grads"] = param_grads

        if self.is_master:
            self.update_ema()
        return step_log

    def check_abort(self):
        if (
            self.mix_precision_dtype == torch.float16
            and self.mix_precision_mode == "inflat_all"
            and self.log_scale < 0
        ):
            if self.is_master:
                print("\n\n\033[91m")
                print(f"ABORT: log_scale in inflat_all mode is less than 0 at step {self.step}.")
                print("This indicates that the model is diverging. You should look into the model and the data.")
                print("\033[0m")
                self.save(non_blocking=False)
                self.save_logs()
            if self.world_size > 1:
                dist.barrier()
            raise ValueError("ABORT: log_scale in inflat_all mode is less than 0.")
