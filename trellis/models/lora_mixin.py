from typing import *

import torch

from ..modules.lora import (
    apply_lora_to_conv3d_model,
    apply_lora_to_model,
    mark_only_lora_as_trainable,
)


class LoRAMixin:
    """
    Inject LoRA adapters into attention projections and restrict training to LoRA.
    """

    def __init__(
        self,
        *args,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_train_bias: bool = False,
        lora_target_modules: Sequence[str] = ("to_qkv", "to_q", "to_kv", "to_out"),
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        apply_lora_to_model(
            self,
            target_modules=lora_target_modules,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            train_bias="all" if lora_train_bias else "none",
        )
        mark_only_lora_as_trainable(self, train_bias=lora_train_bias)

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor], strict: bool = False):
        return super().load_state_dict(state_dict, strict=False)


class LoRAConv3dMixin:
    """
    Inject LoRA adapters into Conv3D layers and restrict training to LoRA.
    """

    def __init__(
        self,
        *args,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_train_bias: bool = False,
        lora_conv_wrap_all: bool = True,
        lora_conv_target_module_names: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        apply_lora_to_conv3d_model(
            self,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            train_bias=lora_train_bias,
            wrap_all=lora_conv_wrap_all,
            target_module_names=lora_conv_target_module_names,
        )
        mark_only_lora_as_trainable(self, train_bias=lora_train_bias)

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor], strict: bool = False):
        return super().load_state_dict(state_dict, strict=False)
