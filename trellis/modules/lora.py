from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Linear):
    """
    A drop-in replacement for nn.Linear that adds a low-rank residual (LoRA).

    Forward: y = x W^T + b + scale * ((x A) B)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        r: int = 0,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        assert r >= 0, "LoRA rank r must be >= 0"
        self.lora_r = int(r)
        self.lora_alpha = float(alpha)
        self.lora_dropout_p = float(dropout)
        self.lora_scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 0.0
        if self.lora_r > 0:
            self.lora_A = nn.Parameter(torch.zeros(self.in_features, self.lora_r))
            self.lora_B = nn.Parameter(torch.zeros(self.lora_r, self.out_features))
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            nn.init.zeros_(self.lora_B)
            self.lora_dropout = nn.Dropout(self.lora_dropout_p) if self.lora_dropout_p > 0 else nn.Identity()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.lora_dropout = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        if self.lora_r == 0 or self.lora_scaling == 0.0:
            return base_out
        x_d = self.lora_dropout(x)
        lora_out = x_d.matmul(self.lora_A).matmul(self.lora_B)
        return base_out + lora_out * self.lora_scaling


def _wrap_linear_with_lora(
    linear: nn.Linear,
    *,
    r: int,
    alpha: float,
    dropout: float,
    train_bias: Literal["none", "all", "lora_only"] = "none",
) -> LoRALinear:
    new_linear = LoRALinear(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        r=r,
        alpha=alpha,
        dropout=dropout,
    )
    new_linear = new_linear.to(device=linear.weight.device, dtype=linear.weight.dtype)
    with torch.no_grad():
        new_linear.weight.copy_(linear.weight.data)
        if linear.bias is not None:
            new_linear.bias.copy_(linear.bias.data)

    new_linear.weight.requires_grad = False
    if new_linear.bias is not None:
        new_linear.bias.requires_grad = train_bias in ("all",)

    if new_linear.lora_A is not None:
        new_linear.lora_A.requires_grad = True
        new_linear.lora_B.requires_grad = True

    return new_linear


def apply_lora_to_model(
    model: nn.Module,
    *,
    target_modules: Sequence[str] = ("to_qkv", "to_q", "to_kv", "to_out"),
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    train_bias: Literal["none", "all", "lora_only"] = "none",
) -> None:
    targets = set(target_modules)

    def maybe_wrap(module: nn.Module) -> None:
        for _, child in list(module.named_children()):
            maybe_wrap(child)
            for attr in list(targets):
                if hasattr(child, attr):
                    linear = getattr(child, attr)
                    if isinstance(linear, nn.Linear) and not isinstance(linear, LoRALinear):
                        setattr(
                            child,
                            attr,
                            _wrap_linear_with_lora(linear, r=r, alpha=alpha, dropout=dropout, train_bias=train_bias),
                        )

    maybe_wrap(model)


class LoRAConv3d(nn.Conv3d):
    """
    A drop-in replacement for nn.Conv3d that adds a low-rank residual (LoRA).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        *,
        r: int = 0,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        assert r >= 0, "LoRA rank r must be >= 0"
        if groups != 1:
            raise NotImplementedError("LoRAConv3d currently supports groups=1 only.")

        self.lora_r = int(r)
        self.lora_alpha = float(alpha)
        self.lora_dropout_p = float(dropout)
        self.lora_scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 0.0

        if self.lora_r > 0:
            self.lora_A = nn.Conv3d(self.in_channels, self.lora_r, kernel_size=1, bias=False)
            self.lora_B = nn.Conv3d(
                self.lora_r,
                self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=False,
                padding_mode=self.padding_mode,
            )
            nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
            nn.init.zeros_(self.lora_B.weight)
            self.lora_dropout = nn.Dropout(self.lora_dropout_p) if self.lora_dropout_p > 0 else nn.Identity()
        else:
            self.lora_A = nn.Identity()
            self.lora_B = nn.Identity()
            self.lora_dropout = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = super().forward(x)
        if self.lora_r == 0 or self.lora_scaling == 0.0:
            return base_out
        x_d = self.lora_dropout(x)
        lora_out = self.lora_B(self.lora_A(x_d))
        return base_out + lora_out * self.lora_scaling


def _wrap_conv3d_with_lora(
    conv: nn.Conv3d,
    *,
    r: int,
    alpha: float,
    dropout: float,
    train_bias: Literal["none", "all"] = "none",
) -> LoRAConv3d:
    new_conv = LoRAConv3d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
        r=r,
        alpha=alpha,
        dropout=dropout,
    )
    new_conv = new_conv.to(device=conv.weight.device, dtype=conv.weight.dtype)
    with torch.no_grad():
        new_conv.weight.copy_(conv.weight.data)
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias.data)

    new_conv.weight.requires_grad = False
    if new_conv.bias is not None:
        new_conv.bias.requires_grad = train_bias == "all"
    return new_conv


def apply_lora_to_conv3d_model(
    model: nn.Module,
    *,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    train_bias: bool = False,
    wrap_all: bool = True,
    target_module_names: Optional[Sequence[str]] = None,
) -> None:
    targets = set(target_module_names or [])
    bias_mode: Literal["none", "all"] = "all" if train_bias else "none"

    def recurse(module: nn.Module) -> None:
        if isinstance(module, LoRAConv3d):
            return
        for name, child in list(module.named_children()):
            recurse(child)
            if isinstance(child, nn.Conv3d) and not isinstance(child, LoRAConv3d):
                if wrap_all or (name in targets):
                    setattr(
                        module,
                        name,
                        _wrap_conv3d_with_lora(child, r=r, alpha=alpha, dropout=dropout, train_bias=bias_mode),
                    )

    recurse(model)


def mark_only_lora_as_trainable(model: nn.Module, train_bias: bool = False) -> None:
    for name, param in model.named_parameters():
        is_lora = ("lora_A" in name) or ("lora_B" in name)
        is_bias = name.endswith(".bias")
        param.requires_grad = bool(is_lora or (train_bias and is_bias))
