"""Inference-compatible CompressAI 1.2.0 entropy subset for native Windows.

CompressAI 1.2.0 does not publish Windows wheels.  The official NTSCC model
uses only EntropyBottleneck, GaussianConditional and ste_round during forward
inference.  These implementations preserve their parameter/buffer names and
inference equations so the unmodified official model imports and the original
checkpoint can be validated strictly.  Entropy coding/compression APIs are
intentionally not implemented.
"""
from __future__ import annotations

import math
import sys
import types

import torch
from torch import nn
from torch.nn import functional as F


class LowerBound(nn.Module):
    def __init__(self, bound: float):
        super().__init__()
        self.register_buffer("bound", torch.tensor([float(bound)]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.maximum(inputs, self.bound.to(dtype=inputs.dtype))


def ste_round(inputs: torch.Tensor) -> torch.Tensor:
    return torch.round(inputs) - inputs.detach() + inputs


class EntropyBottleneck(nn.Module):
    def __init__(
        self,
        channels: int,
        tail_mass: float = 1e-9,
        init_scale: float = 10.0,
        filters: tuple[int, ...] = (3, 3, 3, 3),
        likelihood_bound: float = 1e-9,
        **_: object,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        filters = (1, *filters, 1)
        scale = init_scale ** (1 / (len(filters) - 1))
        for index in range(len(filters) - 1):
            init = math.log(math.expm1(1 / scale / filters[index + 1]))
            matrix = torch.full((channels, filters[index + 1], filters[index]), init)
            self.register_parameter(f"_matrix{index}", nn.Parameter(matrix))
            bias = torch.empty(channels, filters[index + 1], 1)
            nn.init.uniform_(bias, -0.5, 0.5)
            self.register_parameter(f"_bias{index}", nn.Parameter(bias))
            if index < len(filters) - 2:
                factor = torch.zeros(channels, filters[index + 1], 1)
                self.register_parameter(f"_factor{index}", nn.Parameter(factor))

        quantiles = torch.tensor([-init_scale, 0.0, init_scale]).repeat(channels, 1, 1)
        self.quantiles = nn.Parameter(quantiles)
        target = math.log(2 / tail_mass - 1)
        self.register_buffer("target", torch.tensor([-target, 0.0, target]))
        self.register_buffer("_offset", torch.empty(0, dtype=torch.int32))
        self.register_buffer("_quantized_cdf", torch.empty(0, dtype=torch.int32))
        self.register_buffer("_cdf_length", torch.empty(0, dtype=torch.int32))
        self.likelihood_lower_bound = LowerBound(likelihood_bound)

    def _get_medians(self) -> torch.Tensor:
        return self.quantiles[:, :, 1:2]

    def _logits_cumulative(self, inputs: torch.Tensor, stop_gradient: bool = False) -> torch.Tensor:
        logits = inputs
        for index in range(5):
            matrix = getattr(self, f"_matrix{index}")
            bias = getattr(self, f"_bias{index}")
            if stop_gradient:
                matrix, bias = matrix.detach(), bias.detach()
            logits = torch.matmul(F.softplus(matrix), logits) + bias
            if index < 4:
                factor = getattr(self, f"_factor{index}")
                if stop_gradient:
                    factor = factor.detach()
                logits += torch.tanh(factor) * torch.tanh(logits)
        return logits

    def _likelihood(self, inputs: torch.Tensor) -> torch.Tensor:
        lower = self._logits_cumulative(inputs - 0.5)
        upper = self._logits_cumulative(inputs + 0.5)
        sign = -torch.sign(lower + upper).detach()
        return torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))

    def forward(self, inputs: torch.Tensor, training: bool | None = None):
        if inputs.ndim < 2 or inputs.shape[1] != self.channels:
            raise ValueError(f"Expected channel dimension {self.channels}, got {tuple(inputs.shape)}")
        permutation = (1, 0, *range(2, inputs.ndim))
        inverse = (1, 0, *range(2, inputs.ndim))
        values = inputs.permute(*permutation).contiguous()
        shape = values.shape
        values = values.reshape(self.channels, 1, -1)
        medians = self._get_medians()
        mode_training = self.training if training is None else bool(training)
        outputs = values + torch.empty_like(values).uniform_(-0.5, 0.5) if mode_training else torch.round(values - medians) + medians
        likelihood = self.likelihood_lower_bound(self._likelihood(outputs))
        outputs = outputs.reshape(shape).permute(*inverse).contiguous()
        likelihood = likelihood.reshape(shape).permute(*inverse).contiguous()
        return outputs, likelihood

    def loss(self) -> torch.Tensor:
        logits = self._logits_cumulative(self.quantiles, stop_gradient=True)
        return torch.abs(logits - self.target).sum()


class GaussianConditional(nn.Module):
    def __init__(
        self,
        scale_table=None,
        scale_bound: float = 0.11,
        tail_mass: float = 1e-9,
        likelihood_bound: float = 1e-9,
        **_: object,
    ) -> None:
        super().__init__()
        del tail_mass
        table = torch.empty(0) if scale_table is None else torch.as_tensor(scale_table, dtype=torch.float32)
        self.register_buffer("scale_table", table)
        self.register_buffer("scale_bound", torch.tensor([float(scale_bound)]))
        self.register_buffer("_offset", torch.empty(0, dtype=torch.int32))
        self.register_buffer("_quantized_cdf", torch.empty(0, dtype=torch.int32))
        self.register_buffer("_cdf_length", torch.empty(0, dtype=torch.int32))
        self.likelihood_lower_bound = LowerBound(likelihood_bound)
        self.lower_bound_scale = LowerBound(scale_bound)

    @staticmethod
    def _standardized_cumulative(inputs: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.erfc(-(2 ** -0.5) * inputs)

    def _likelihood(self, inputs: torch.Tensor, scales: torch.Tensor, means=None) -> torch.Tensor:
        values = inputs if means is None else inputs - means
        scales = self.lower_bound_scale(scales)
        values = torch.abs(values)
        upper = self._standardized_cumulative((0.5 - values) / scales)
        lower = self._standardized_cumulative((-0.5 - values) / scales)
        return upper - lower

    def forward(self, inputs: torch.Tensor, scales: torch.Tensor, means=None, training: bool | None = None):
        mode_training = self.training if training is None else bool(training)
        if mode_training:
            outputs = inputs + torch.empty_like(inputs).uniform_(-0.5, 0.5)
        else:
            values = inputs if means is None else inputs - means
            outputs = torch.round(values)
            if means is not None:
                outputs = outputs + means
        likelihood = self.likelihood_lower_bound(self._likelihood(outputs, scales, means))
        return outputs, likelihood


def install_compressai_1_2_compat() -> None:
    """Expose the compatibility subset under the imports used by official NTSCC."""
    if "compressai.entropy_models" in sys.modules:
        return
    package = types.ModuleType("compressai")
    package.__version__ = "1.2.0-windows-inference-compat"
    entropy_models = types.ModuleType("compressai.entropy_models")
    entropy_models.EntropyBottleneck = EntropyBottleneck
    entropy_models.GaussianConditional = GaussianConditional
    ops = types.ModuleType("compressai.ops")
    ops.ste_round = ste_round
    package.entropy_models = entropy_models
    package.ops = ops
    sys.modules["compressai"] = package
    sys.modules["compressai.entropy_models"] = entropy_models
    sys.modules["compressai.ops"] = ops
