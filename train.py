import argparse
import glob
import json
import os
import random
import sys

import numpy as np
import torch
import torch.multiprocessing as mp
import wandb

from easydict import EasyDict as edict

from trellis import datasets, models, trainers
from trellis.utils.dist_utils import setup_dist


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.dtype):
        return str(obj)
    return str(obj)


def parse_override_value(raw: str):
    try:
        return json.loads(raw)
    except Exception:
        return raw


def apply_dataset_arg_overrides(config: dict, overrides: list[str]):
    if len(overrides) == 0:
        return config
    config.setdefault("dataset", {})
    config["dataset"].setdefault("args", {})
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --dataset_arg override: {item}")
        key, value = item.split("=", 1)
        config["dataset"]["args"][key] = parse_override_value(value)
    return config


def find_ckpt(cfg):
    cfg["load_ckpt"] = None
    if cfg.load_dir != "":
        if cfg.ckpt == "latest":
            files = glob.glob(os.path.join(cfg.load_dir, "ckpts", "misc_*.pt"))
            if len(files) != 0:
                cfg.load_ckpt = max([int(os.path.basename(f).split("step")[-1].split(".")[0]) for f in files])
        elif cfg.ckpt == "none":
            cfg.load_ckpt = None
        else:
            cfg.load_ckpt = int(cfg.ckpt)
    return cfg


def setup_rng(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed_all(rank)
    np.random.seed(rank)
    random.seed(rank)


def get_model_summary(model):
    model_summary = "Parameters:\n"
    model_summary += "=" * 128 + "\n"
    model_summary += f'{"Name":<{72}}{"Shape":<{32}}{"Type":<{16}}{"Grad"}\n'
    num_params = 0
    num_trainable_params = 0
    for name, param in model.named_parameters():
        model_summary += f"{name:<{72}}{str(param.shape):<{32}}{str(param.dtype):<{16}}{param.requires_grad}\n"
        num_params += param.numel()
        if param.requires_grad:
            num_trainable_params += param.numel()
    model_summary += "\n"
    model_summary += f"Number of parameters: {num_params}\n"
    model_summary += f"Number of trainable parameters: {num_trainable_params}\n"
    return model_summary


def main(local_rank, cfg):
    rank = cfg.node_rank * cfg.num_gpus + local_rank
    world_size = cfg.num_nodes * cfg.num_gpus
    if world_size > 1:
        setup_dist(rank, local_rank, world_size, cfg.master_addr, cfg.master_port)

    setup_rng(rank)

    if rank == 0:
        wb_cfg = getattr(cfg, "wandb", {}) if hasattr(cfg, "wandb") else {}
        project = wb_cfg.get("project", os.environ.get("WANDB_PROJECT", "ss4d"))
        entity = wb_cfg.get("entity", os.environ.get("WANDB_ENTITY", None))
        name = wb_cfg.get("name", os.environ.get("WANDB_NAME", os.path.basename(cfg.output_dir.rstrip("/"))))
        mode = wb_cfg.get("mode", os.environ.get("WANDB_MODE", None))
        disabled = os.environ.get("WANDB_DISABLED", "false").lower() in ("1", "true", "yes")
        if not disabled:
            init_kwargs = {
                "project": project,
                "name": name,
                "dir": cfg.output_dir,
                "config": json.loads(json.dumps(cfg.__dict__, default=json_default)),
            }
            if entity is not None:
                init_kwargs["entity"] = entity
            if mode is not None:
                init_kwargs["mode"] = mode
            wandb.init(**init_kwargs)

    dataset_root = cfg.data_dir
    dataset = getattr(datasets, cfg.dataset.name)(dataset_root, **cfg.dataset.args)
    dataset_test = None
    if getattr(cfg, "data_dir_test", ""):
        dataset_test = getattr(datasets, cfg.dataset.name)(cfg.data_dir_test, **cfg.dataset.args)

    model_dict = {name: getattr(models, model.name)(**model.args).cuda() for name, model in cfg.models.items()}

    trainer = getattr(trainers, cfg.trainer.name)(
        model_dict,
        dataset,
        test_dataset=dataset_test,
        **cfg.trainer.args,
        output_dir=cfg.output_dir,
        load_dir=cfg.load_dir,
        step=cfg.load_ckpt,
    )

    if rank == 0:
        for name, backbone in model_dict.items():
            model_summary = get_model_summary(backbone)
            print(f"\n\nBackbone: {name}\n" + model_summary)
            with open(os.path.join(cfg.output_dir, f"{name}_model_summary.txt"), "w") as fp:
                print(model_summary, file=fp)

    if not cfg.tryrun:
        if cfg.profile:
            trainer.profile()
        else:
            trainer.run()
    if rank == 0 and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Experiment config file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--load_dir", type=str, default="", help="Load directory, default to output_dir")
    parser.add_argument("--ckpt", type=str, default="latest", help="Checkpoint step to resume training")
    parser.add_argument("--data_dir", type=str, default="./data/", help="Sequence dataset directory")
    parser.add_argument("--data_dir_test", type=str, default="", help="Optional test dataset directory")
    parser.add_argument(
        "--dataset_arg",
        action="append",
        default=[],
        help="Override dataset.args entries with KEY=VALUE. VALUE may be raw text or JSON.",
    )
    parser.add_argument("--auto_retry", type=int, default=0, help="Number of retries on error")
    parser.add_argument("--tryrun", action="store_true", help="Try run without training")
    parser.add_argument("--profile", action="store_true", help="Profile training")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--node_rank", type=int, default=0, help="Node rank")
    parser.add_argument("--num_gpus", type=int, default=-1, help="Number of GPUs per node, default to all")
    parser.add_argument("--master_addr", type=str, default="localhost", help="Master address for distributed training")
    parser.add_argument("--master_port", type=str, default="12345", help="Port for distributed training")
    opt = parser.parse_args()
    opt.load_dir = opt.load_dir if opt.load_dir != "" else opt.output_dir
    opt.num_gpus = torch.cuda.device_count() if opt.num_gpus == -1 else opt.num_gpus

    config = json.load(open(opt.config, "r"))
    config = apply_dataset_arg_overrides(config, opt.dataset_arg)

    cfg = edict()
    cfg.update(opt.__dict__)
    cfg.update(config)
    print("\n\nConfig:")
    print("=" * 80)
    print(json.dumps(cfg.__dict__, indent=4, default=json_default))

    if cfg.node_rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(os.path.join(cfg.output_dir, "command.txt"), "w") as fp:
            print(" ".join(["python"] + sys.argv), file=fp)
        with open(os.path.join(cfg.output_dir, "config.json"), "w") as fp:
            json.dump(config, fp, indent=4)

    if cfg.auto_retry == 0:
        cfg = find_ckpt(cfg)
        if cfg.num_gpus > 1:
            mp.spawn(main, args=(cfg,), nprocs=cfg.num_gpus, join=True)
        else:
            main(0, cfg)
    else:
        for rty in range(cfg.auto_retry):
            try:
                cfg = find_ckpt(cfg)
                if cfg.num_gpus > 1:
                    mp.spawn(main, args=(cfg,), nprocs=cfg.num_gpus, join=True)
                else:
                    main(0, cfg)
                break
            except Exception as e:
                print(f"Error: {e}")
                print(f"Retrying ({rty + 1}/{cfg.auto_retry})...")
