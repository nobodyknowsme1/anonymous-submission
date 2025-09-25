# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import svd_lowrank
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import dequantize_module_weight, gather_params_ctx
from peft.utils.other import transpose

from .config import LoraConfig
from .dora import DoraConv2dLayer, DoraLinearLayer


class LoraLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    # adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B", "lora_proj_A", "lora_proj_B", "lora_Y")
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Module, ephemeral_gpu_offload: bool = False, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})

        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.use_dora: dict[str, bool] = {}
        self.lora_magnitude_vector = torch.nn.ModuleDict()  # for DoRA
        self._caches: dict[str, Any] = {}
        self.ephemeral_gpu_offload: bool = ephemeral_gpu_offload
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        elif hasattr(base_layer, "codebooks") and base_layer.__class__.__name__ == "QuantizedLinear":
            # AQLM QuantLinear
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "w_bit") and base_layer.__class__.__name__ == "WQLinear_GEMM":
            # Awq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif base_layer.__class__.__name__ == "EetqLinear":
            # Eetq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "W_q") and base_layer.__class__.__name__ == "HQQLinear":
            # HQQ layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            # possibly support user provided custom layer types using dynamic dispatch
            if hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
                in_features, out_features = base_layer.in_features, base_layer.out_features
            else:
                in_features, out_features = None, None
            warnings.warn(
                f"Unsupported layer type '{type(base_layer)}' encountered, proceed at your own risk.", UserWarning
            )

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora: bool = False
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.pissa_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
            with gather_params_ctx(self.get_base_layer().weight):
                self.olora_init(adapter_name)
        elif init_lora_weights == "loftq":
            with gather_params_ctx(self.get_base_layer().weight):
                self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        # print(f"Resetting LoRA parameters for adapter {adapter_name} with init_lora_weights={init_lora_weights}")
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            if init_lora_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # [ANONYMOUS_REPO]
                nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            elif init_lora_weights.lower() == "gaussian":
                nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        if adapter_name in self.lora_embedding_A.keys():
            # Initialize A to zeros and B the same way as the default for nn.Embedding, see:
            # [ANONYMOUS_REPO]
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])

    def olora_init(self, adapter_name):
        dtype = self.get_base_layer().weight.dtype
        if dtype in [torch.int8, torch.uint8]:
            weight_tensor = dequantize_module_weight(self.get_base_layer())
        elif dtype in [torch.float32, torch.float16, torch.bfloat16]:
            weight_tensor = self.get_base_layer().weight
        else:
            raise TypeError(f"Unsupported data type for the base layer. Got {dtype}.")

        scale_factor = self.scaling[adapter_name]
        r = self.r[adapter_name]
        weight_tensor = weight_tensor.to(torch.float32)
        Q, R = torch.linalg.qr(weight_tensor.data)

        Qr, Rr = Q[:, :r], R[:r]

        self.lora_A[adapter_name].weight.data = Rr.contiguous()
        self.lora_B[adapter_name].weight.data = Qr.contiguous()

        weight_tensor.data -= scale_factor * self.lora_B[adapter_name].weight @ self.lora_A[adapter_name].weight
        weight_tensor = weight_tensor.to(dtype)
        self.get_base_layer().weight.data = weight_tensor

    def pissa_init(self, adapter_name, init_lora_weights):
        weight = self.get_base_layer().weight
        dtype = weight.dtype
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise TypeError(
                "Please initialize PiSSA under float32, float16, or bfloat16. "
                "Subsequently, re-quantize the residual model to help minimize quantization errors."
            )
        weight = weight.to(torch.float32)
        if init_lora_weights == "pissa":
            # USV^T = W <-> VSU^T = W^T, where W^T = weight.data in R^{out_channel, in_channel},
            V, S, Uh = torch.linalg.svd(weight.data, full_matrices=False)
            Vr = V[:, : self.r[adapter_name]]
            Sr = S[: self.r[adapter_name]]
            Sr /= self.scaling[adapter_name]
            Uhr = Uh[: self.r[adapter_name]]
        elif len(init_lora_weights.split("_niter_")) == 2:
            Vr, Sr, Ur = svd_lowrank(
                weight.data, self.r[adapter_name], niter=int(init_lora_weights.split("_niter_")[-1])
            )
            Sr /= self.scaling[adapter_name]
            Uhr = Ur.t()
        else:
            raise ValueError(
                f"init_lora_weights should be 'pissa' or 'pissa_niter_[number of iters]', got {init_lora_weights} instead."
            )

        lora_A = torch.diag(torch.sqrt(Sr)) @ Uhr
        lora_B = Vr @ torch.diag(torch.sqrt(Sr))
        self.lora_A[adapter_name].weight.data = lora_A
        self.lora_B[adapter_name].weight.data = lora_B
        weight = weight.data - self.scaling[adapter_name] * lora_B @ lora_A
        weight = weight.to(dtype)
        self.get_base_layer().weight.data = weight

    def loftq_init(self, adapter_name):
        from peft.utils.loftq_utils import loftq_init

        weight = self.get_base_layer().weight
        kwargs = {
            "num_bits": self.kwargs.get("loftq_bits", 4),
            "reduced_rank": self.r[adapter_name],
            "num_iter": self.kwargs.get("loftq_iter", 1),
        }

        qweight, lora_A, lora_B = loftq_init(weight, **kwargs)
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            self.lora_A[adapter_name].weight.data = lora_A
            self.lora_B[adapter_name].weight.data = lora_B
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            self.lora_embedding_A[adapter_name].weight.data = lora_A
            self.lora_embedding_B[adapter_name].weight.data = lora_B
        self.get_base_layer().weight.data = qweight

    def dora_init(self, adapter_name: str) -> None:
        if not self.lora_magnitude_vector:
            # first dora layer being added, add lora_magnitude_vector to the list of learnable parameters
            self.adapter_layer_names = self.adapter_layer_names[:] + ("lora_magnitude_vector",)

        dora_layer = DoraLinearLayer(fan_in_fan_out=getattr(self, "fan_in_fan_out", False))
        lora_A = self.lora_A[adapter_name].weight
        lora_B = self.lora_B[adapter_name].weight
        place_on_cpu = self.ephemeral_gpu_offload and (lora_A.device.type == "cpu" or lora_B.device.type == "cpu")
        if self.ephemeral_gpu_offload:
            if lora_A.device.type == "cuda":
                lora_B = lora_B.to(lora_A.device)
            else:
                if lora_B.device.type != "cuda":
                    lora_B = lora_B.to("cuda")
                lora_A = lora_A.to(lora_B.device)
        scaling = self.scaling[adapter_name]
        dora_layer.update_layer(
            base_layer=self.get_base_layer(), lora_A=lora_A, lora_B=lora_B, scaling=scaling, place_on_cpu=place_on_cpu
        )
        self.lora_magnitude_vector[adapter_name] = dora_layer

    def _cache_store(self, key: str, value: Any) -> None:
        self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key)
        return value

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            if scale is None:
                self.scaling[active_adapter] = self.lora_alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)

        if self.merged:
            # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
            # adapters. Therefore, it is better to raise an error in this case.
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)

        unique_adapters = set(self.active_adapters)
        for adapter_name in unique_adapters:
            if self.use_dora.get(adapter_name, False):
                msg = "Cannot pass `adapter_names` when DoRA is enabled."
                raise ValueError(msg)

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_A.keys():
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]].to(lora_A.weight.dtype)
            lora_output = lora_B(lora_A(dropout(sub_batch))) * scaling
            result[sub_batch_indices_list[i]] += lora_output.to(torch_result_dtype)

        return result


# Below code is based on [ANONYMOUS_REPO]
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class Linear(nn.Module, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        orig_weights = orig_weights + delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(orig_weights, transpose(delta_weight, self.fan_in_fan_out), scaling=1)
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                        orig_weights = dora_factor * (orig_weights + delta_weight)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        base_layer.weight.data = base_layer.weight.data + delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(
                                base_layer.weight, transpose(delta_weight, self.fan_in_fan_out), scaling=1
                            )
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                        new_weight = dora_factor * (base_layer.weight.data + delta_weight)
                        base_layer.weight.data = new_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    weight_orig = weight.data / dora_factor.view(-1, 1) - delta_weight
                    weight.data = weight_orig

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                # if active_adapter not in self.lora_Y.keys():
                if active_adapter not in self.lora_A.keys():
                    print(f"[DEBUG] Adapter {active_adapter} not in lora_Y.keys(): {self.lora_Y.keys()}")
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    result = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    x = dropout(x)
                    result = result + self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                    )

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


class Embedding(nn.Module, LoraLayer):
    # LoRA implemented in a Embedding layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer)

        if use_dora:
            raise ValueError(f"{self.__class__.__name__} does not support DoRA yet, please set it to False")

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        weight_A = torch.randn((r, self.in_features))
        weight_B = torch.randn((self.out_features, r))
        self.lora_embedding_A[adapter_name] = nn.Parameter(weight_A)
        self.lora_embedding_B[adapter_name] = nn.Parameter(weight_B)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_embedding_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights = orig_weights + self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data = base_layer.weight.data + self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_embedding_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_embedding_B[adapter].device
        dtype = self.lora_embedding_A[adapter].dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_embedding_A[adapter]
        weight_B = self.lora_embedding_B[adapter]

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, True) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_embedding_A[adapter] = weight_A.to(dtype)
            self.lora_embedding_B[adapter] = weight_B.to(dtype)

        return output_tensor

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_embedding_A.keys():
                continue

            embedding_A = self.lora_embedding_A[active_adapter].T
            embedding_B = self.lora_embedding_B[active_adapter].T
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]]
            after_A = self._embed(sub_batch, embedding_A)
            result[sub_batch_indices_list[i]] += (after_A @ embedding_B) * scaling

        return result

    def _embed(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        base_layer = self.get_base_layer()
        return F.embedding(
            input,
            weight,
            padding_idx=base_layer.padding_idx,
            max_norm=base_layer.max_norm,
            norm_type=base_layer.norm_type,
            scale_grad_by_freq=base_layer.scale_grad_by_freq,
            sparse=base_layer.sparse,
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # TODO: no dtype conversion here, unlike in Linear, is that correct?
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_embedding_A:
                    continue
                embedding_A = self.lora_embedding_A[active_adapter].T
                embedding_B = self.lora_embedding_B[active_adapter].T
                scaling = self.scaling[active_adapter]
                after_A = self._embed(x, embedding_A)
                result = result + (after_A @ embedding_B) * scaling
            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


class Conv2d(nn.Module, LoraLayer):
    # Lora implemented in a conv2d layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        base_layer = self.get_base_layer()
        kernel_size = base_layer.kernel_size
        stride = base_layer.stride
        padding = base_layer.padding
        self.lora_A[adapter_name] = nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False)
        self.lora_B[adapter_name] = nn.Conv2d(r, self.out_features, (1, 1), (1, 1), bias=False)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def dora_init(self, adapter_name: str) -> None:
        if self.lora_magnitude_vector is None:
            # first dora layer being added, add lora_magnitude_vector to the list of learnable parameters
            self.adapter_layer_names = self.adapter_layer_names[:] + ("lora_magnitude_vector",)

        dora_layer = DoraConv2dLayer(fan_in_fan_out=False)
        lora_A = self.lora_A[adapter_name].weight
        lora_B = self.lora_B[adapter_name].weight
        scaling = self.scaling[adapter_name]
        dora_layer.update_layer(base_layer=self.get_base_layer(), lora_A=lora_A, lora_B=lora_B, scaling=scaling)
        self.lora_magnitude_vector[adapter_name] = dora_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights inside the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)

                    if not self.use_dora[active_adapter]:
                        orig_weights = orig_weights + delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(orig_weights, delta_weight, scaling=1)
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        orig_weights = dora_factor.view(-1, 1, 1, 1) * (orig_weights + delta_weight)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        base_layer.weight.data = base_layer.weight.data + delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(base_layer.weight, delta_weight, scaling=1)
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        new_weight = dora_factor.view(-1, 1, 1, 1) * (base_layer.weight.data + delta_weight)
                        base_layer.weight.data = new_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    weight_orig = weight.data / dora_factor.view(-1, 1, 1, 1) - delta_weight
                    weight.data = weight_orig

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_A[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        # [ANONYMOUS_REPO]
        if self.get_base_layer().weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            output_tensor = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(
                3
            ) * self.scaling[adapter]
        else:
            # conv2d 3x3
            output_tensor = (
                F.conv2d(
                    weight_A.permute(1, 0, 2, 3),
                    weight_B,
                ).permute(1, 0, 2, 3)
                * self.scaling[adapter]
            )

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    result = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    x = dropout(x)
                    result = result + self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                    )

            result = result.to(torch_result_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: LoraConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    # Check if we should use Sum LoRA implementation
    use_sum_lora = getattr(lora_config, "use_sum_lora", False)

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Embedding):
        embedding_kwargs = kwargs.copy()
        embedding_kwargs.pop("fan_in_fan_out", None)
        embedding_kwargs.update(lora_config.loftq_config)
        if use_sum_lora:
            new_module = CoSAEmbedding(target, adapter_name, **embedding_kwargs)
        else:
            new_module = Embedding(target, adapter_name, **embedding_kwargs)
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        kwargs.update(lora_config.loftq_config)
        if use_sum_lora:
            new_module = CoSAConv2d(target, adapter_name, **kwargs)
        else:
            new_module = Conv2d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        if use_sum_lora:
            new_module = CoSALinear(target, adapter_name, **kwargs)
        else:
            new_module = Linear(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        if not kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to False but the target module is `Conv1D`. " "Setting fan_in_fan_out to True."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
        kwargs.update(lora_config.loftq_config)
        new_module = Linear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)

    return new_module

class CoSALayer(LoraLayer):
    # adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B", "lora_proj_A", "lora_proj_B", "lora_Y")
    # adapter_layer_names = ("lora_A", "lora_B", "lora_Y")
    adapter_layer_names = ("lora_Y",)


    """Base CoSA layer that implements compressed sensing-based adaptation."""
    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        super().__init__(base_layer, **kwargs)
        self.use_compression = {}
        self.compression_a = {}
        self.compression_b = {}
        self.kron = {}

        # Trainable Y matrix (adapter parameter)
        self.lora_Y = nn.ModuleDict({})

        # Frozen L and R matrices (regular parameters, not adapter parameters)
        self.lora_L = nn.ParameterDict({})
        self.lora_R = nn.ParameterDict({})

        # Seed-based compression matrix generation
        self.compression_config = {}  # Store generation metadata
        self._experiment_seed = None  # Will be set during training

    def update_layer(
        self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora,
        use_dora: bool = False, use_compression: bool = False, compression_a: int = 128, compression_b: int = 128,
        compression_seed: int = None
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters

        if use_compression:
            # Create trainable Y matrix
            self.lora_Y[adapter_name] = nn.Linear(compression_b, compression_a, bias=False)
            self.use_compression[adapter_name] = True

            # Set the seed if provided
            if compression_seed is not None:
                self._experiment_seed = compression_seed

            # Store compression configuration
            self.compression_config[adapter_name] = {
                'compression_a': compression_a,
                'compression_b': compression_b,
                'layer_name': getattr(self, '_layer_name', None),
                'in_features': self.in_features,
                'out_features': self.out_features,
                'L_std': 1 / math.sqrt(compression_a),
                'R_std': 1 / math.sqrt(compression_b)
            }

            # Initialize Y matrix
            nn.init.zeros_(self.lora_Y[adapter_name].weight)

            # Generate and store frozen L and R matrices
            device = self.lora_Y[adapter_name].weight.device
            dtype = self.lora_Y[adapter_name].weight.dtype

            L = self._generate_compression_matrix_L(adapter_name, device, dtype)
            R = self._generate_compression_matrix_R(adapter_name, device, dtype)

            # Store as frozen parameters (not adapter parameters)
            self.lora_L[adapter_name] = nn.Parameter(L, requires_grad=False)
            self.lora_R[adapter_name] = nn.Parameter(R, requires_grad=False)
        else:
            self.use_compression[adapter_name] = False

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r


        # super().update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora)

        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.pissa_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
            with gather_params_ctx(self.get_base_layer().weight):
                self.olora_init(adapter_name)
        elif init_lora_weights == "loftq":
            with gather_params_ctx(self.get_base_layer().weight):
                self.loftq_init(adapter_name)
        elif init_lora_weights and use_compression == False:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False     

        self.set_adapter(self.active_adapters)
        # No longer need to set requires_grad=False for L,R since they don't exist as parameters

    def set_experiment_seed(self, seed: int):
        """Set the experiment seed for reproducible matrix generation."""
        self._experiment_seed = seed

    def _get_layer_identifier(self, adapter_name: str) -> str:
        """Generate a unique identifier for this layer based on its position in the model."""
        # Use the layer's class name and module name if available
        layer_name = getattr(self, '_layer_name', self.__class__.__name__)
        base_layer_type = type(self.base_layer).__name__
        return f"{layer_name}_{base_layer_type}_{adapter_name}"

    def _generate_compression_matrix_L(self, adapter_name: str, device: torch.device, dtype: torch.dtype = None) -> torch.Tensor:
        """Generate L matrix deterministically from experiment seed."""
        config = self.compression_config[adapter_name]
        compression_a = config['compression_a']
        out_features = config['out_features']
        std = config['L_std']

        # Create unique seed for this specific L matrix
        layer_id = self._get_layer_identifier(adapter_name)
        matrix_seed = (self._experiment_seed or 42) + hash(f"{layer_id}_L") % 10000

        # Generate matrix deterministically
        generator = torch.Generator(device=device)
        generator.manual_seed(matrix_seed)

        L = torch.randn(compression_a, out_features, generator=generator, device=device, dtype=dtype or torch.float32)
        L = L * std  # Apply initialization scaling

        return L

    def _generate_compression_matrix_R(self, adapter_name: str, device: torch.device, dtype: torch.dtype = None) -> torch.Tensor:
        """Generate R matrix deterministically from experiment seed."""
        config = self.compression_config[adapter_name]
        compression_b = config['compression_b']
        in_features = config['in_features']
        std = config['R_std']

        # Create unique seed for this specific R matrix
        layer_id = self._get_layer_identifier(adapter_name)
        matrix_seed = (self._experiment_seed or 42) + hash(f"{layer_id}_R") % 10000

        # Generate matrix deterministically
        generator = torch.Generator(device=device)
        generator.manual_seed(matrix_seed)

        R = torch.randn(in_features, compression_b, generator=generator, device=device, dtype=dtype or torch.float32)
        R = R * std  # Apply initialization scaling

        return R

    def orthogonal_init(self, A: torch.Tensor, B: torch.Tensor, adapter_name: str):
        with torch.no_grad():
            # A ∈ ℝ^{r × n}, B ∈ ℝ^{m × r}

            # Step 1: Randomly initialize A (no need to be orthogonal)
            A_ = torch.empty_like(A)
            nn.init.normal_(A_, std=1 / self.r[adapter_name])

            # Step 2: Randomly initialize B
            B_ = torch.empty_like(B)
            nn.init.normal_(B_, std=1 / self.r[adapter_name])

            # Step 3: Project rows of B to be outside the row space of A
            # QR on A.T to get row space basis
            Q, _ = torch.linalg.qr(A_.T, mode='reduced')  # Q ∈ ℝ^{n × r}

            # Remove component of B along Q
            B_proj = B_ - (B_ @ Q.T) @ Q  # (m, r) - (m, n) @ (n, r) → (m, r)

            # Optional: normalize rows to keep scale
            # B_proj = torch.nn.functional.normalize(B_proj, dim=1)

            # Assign
            self.lora_A[adapter_name].weight = A_
            self.lora_B[adapter_name].weight = B_proj

            # print(f"A norm: {A.norm()}")
            # print(f"B norm: {B.norm()}")
            # print(f"A non-zero rows: {torch.count_nonzero(A, dim=1).sum()}")
            # print(f"B non-zero rows: {torch.count_nonzero(B, dim=1).sum()}")

    def lora_sparse_one_hot_init(self, A, B):
        """
        Custom LoRA init:
        - Each row of A and B has one 1 (rest are 0), position chosen randomly.
        - Then columns are normalized: non-zero entries in column j are divided by sqrt(nj).
        """
        with torch.no_grad():
            for tensor in [A, B]:
                rows, cols = tensor.shape
                tensor.zero_()
                
                # Randomly place a single 1 in each row
                rand_indices = torch.randint(low=0, high=cols, size=(rows,))
                tensor[torch.arange(rows), rand_indices] = 1.0

                # Normalize column-wise
                column_counts = torch.bincount(rand_indices, minlength=cols).float()
                norm_factors = torch.sqrt(column_counts)
                norm_factors[column_counts == 0] = 1.0  # avoid division by zero

                tensor.div_(norm_factors)  # broadcasting normalization

    def random_init(self, A, B, adapter_name):
        """
        Random initialization for LoRA parameters A and B.
        A is initialized with normal distribution, B is initialized with zeros.
        """
        nn.init.normal_(A, std=1 / self.r[adapter_name])
        nn.init.normal_(B, std=1 / self.r[adapter_name])

    def identity_init(self, adapter_name):
        """
        Initialize LoRA parameters A and B such that A is an identity matrix and B is zero.
        This is useful for initializing LoRA layers without any additional learnable parameters.
        """
        # nn.init.eye_(self.lora_A[adapter_name].weight)
        # nn.init.eye_(self.lora_A[adapter_name].weight)
        n, m = self.in_features, self.out_features
        r = self.r[adapter_name]
        if n % r != 0 or m % r != 0:
            warnings.warn(
                f"Identity initialization requires out_features ({n}) and in_features ({m}) to be divisible by r ({r}). "
                "Skpping add lora for this layer."
            )
            return
        I_r = torch.eye(r, dtype=self.lora_A[adapter_name].weight.dtype, device=self.lora_A[adapter_name].weight.device)
        A = I_r.repeat(n // r, 1)  # (n, r)
        B = I_r.repeat(1, m // r)  # (r, m)
        self.lora_A[adapter_name].weight.data = A.T.contiguous()
        self.lora_B[adapter_name].weight.data = B.T.contiguous()

    def svd_init(self, scaled: bool = False, adapter_name: str = None):
        # A: (r, n), B: (m, r), W: (m, n)
        # P_A: (r, n), P_B: (m, r)
        W = self.get_base_layer().weight.data

        with torch.no_grad():
            d_out, d_in = W.shape
            r = self.r[adapter_name]

            # Compute full SVD
            # USV^T = W <-> VSU^T = W^T, where W^T = weight.data in R^{out_channel, in_channel},
            # U, S, Vh = torch.linalg.svd(W.data, full_matrices=False)
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            # U: (m, m), S: (m,), Vh: (m, n)

            # Top-r components
            U_r = U[:, :r]            # (m, r)
            Vh_r = Vh[:r, :]         # (r, n)
            # print(f"U_r shape: {U_r.shape}, V_r shape: {Vh_r.shape}")
            # print(f"A shape: {A.shape}, B shape: {B.shape}")
            # print(f"P_A shape: {P_A.shape if P_A is not None else None}, P_B shape: {P_B.shape if P_B is not None else None}")

            if scaled == False:
                # Set A = Vh_r, B = U_r
                self.lora_A[adapter_name].weight.data = Vh_r.contiguous()
                self.lora_B[adapter_name].weight.data = U_r.contiguous()
            else:
                S_r = S[:r]              # (r,)
                S_r /= self.scaling[adapter_name]  # Scale S by scaling factor
                # Set A = Vh_r * sqrt(S[:r]), B = U_r * sqrt(S[:r])
                scaled_A = torch.diag(torch.sqrt(S_r)) @ Vh_r
                scaled_B = U_r @ torch.diag(torch.sqrt(S_r))
                # print(f"scaled_A norm: {scaled_A.norm()}, scaled_B norm: {scaled_B.norm()}")
                print(f"W shape: {W.shape}")
                self.lora_A[adapter_name].weight.data = scaled_A.contiguous()
                self.lora_B[adapter_name].weight.data = scaled_B.contiguous()

                # W = W - self.scaling[adapter_name] * ((P_A @ A) + (B @ P_B))
                # weight = W.to(dtype)
                # self.get_base_layer().weight.data = weight

    def lm_svd_init(self, adapter_name=None, rank_large=6):
        # A: (r, n), B: (m, r), W: (m, n)
        # P_A: (r, n), P_B: (m, r)
        W = self.get_base_layer().weight.data
        dtype = self.get_base_layer().weight.dtype

        with torch.no_grad():
            r = self.r[adapter_name]
            r1 = rank_large
            r2= r - rank_large

            # Compute full SVD
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            # U: (m, m), S: (m,), Vh: (n, n)

            U1, S1, Vh1 = U[:, :r1], S[:r1], Vh[:r1, :]
            S1 /= self.scaling[adapter_name]
            U2, S2, Vh2 = U[:, r1:r1 + r2], S[r1:r1 + r2], Vh[r1:r1 + r2, :]
            S2 /= self.scaling[adapter_name]

            # For A P_A (dominant directions)
            lora_A = torch.diag(torch.sqrt(S1)) @ Vh1 
            lora_proj_A = U1 @ torch.diag(torch.sqrt(S1))
            self.lora_A[adapter_name].weight.data[:r1] = torch.diag(torch.sqrt(S1)) @ Vh1          # (r1, n)
            self.lora_proj_A[adapter_name].weight.data[:, :r1] = U1 @ torch.diag(torch.sqrt(S1))        # (m, r1)

            # For P_B B (residual directions)
            lora_B = U2 @ torch.diag(torch.sqrt(S2))
            lora_proj_B = torch.diag(torch.sqrt(S2)) @ Vh2
            self.lora_B[adapter_name].weight.data[:, r1:].copy_(U2 @ torch.diag(torch.sqrt(S2)))           # (m, r2)
            self.lora_proj_B[adapter_name].weight.data[r1:].copy_(torch.diag(torch.sqrt(S2)) @ Vh2)        # (r2, n)

            # W = W - self.scaling[adapter_name] * ((P_A @ A) + (B @ P_B))
            W = W - self.scaling[adapter_name] * ((lora_proj_A @ lora_A) + (lora_B @ lora_proj_B))
            print(f"lora_proj_A @ lora_A norm: {(lora_proj_A @ lora_A).norm()}")
            print(f"lora_B @ lora_proj_B norm: {(lora_B @ lora_proj_B).norm()}")

            weight = W.to(dtype)
            self.get_base_layer().weight.data = weight


    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        # Reset CoSA parameters for the given adapter
        if not init_lora_weights:
            return

        if adapter_name in self.lora_A.keys():
            if init_lora_weights is True:
                nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
                nn.init.normal_(self.lora_B[adapter_name].weight, std=1 / self.r[adapter_name])

                # nn.init.kaiming_uniform_(P_A, a=math.sqrt(5))
                # nn.init.zeros_(B)
            elif init_lora_weights.lower() in ["gaussian", "kaiming"]:
                if init_lora_weights.lower() == "gaussian":
                    # nn.init.normal_(A, std=1 / self.r[adapter_name])
                    nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
                    nn.init.normal_(self.lora_B[adapter_name].weight, std=1 / self.r[adapter_name])
                else:
                    # nn.init.kaiming_uniform_(A, a=math.sqrt(5))
                    # nn.init.normal_(P_A, std=1 / self.r[adapter_name])
                    nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
                    nn.init.kaiming_uniform_(self.lora_B[adapter_name].weight, a=math.sqrt(5))

                # with torch.no_grad():
                    # A_proj = A @ P_A
                    # B_init = -torch.linalg.pinv(P_B) @ A_proj
                    # B.copy_(B_init.T)
            elif init_lora_weights.lower() == "onehot":
                self.lora_sparse_one_hot_init(A, B)
            elif init_lora_weights.lower() == "identity":
                # nn.init.eye_(self.lora_A[adapter_name].weight)
                # nn.init.eye_(self.lora_B[adapter_name].weight)
                self.identity_init(adapter_name)
            elif init_lora_weights.lower() == "orthogonal":
                self.orthogonal_init(A, B, adapter_name)
            elif init_lora_weights.lower() == "svd":
                self.svd_init(scaled=False, adapter_name=adapter_name)
            elif init_lora_weights.lower() == "scaled_svd":
                self.svd_init(scaled=True, adapter_name=adapter_name)
            nn.init.zeros_(self.lora_proj_A[adapter_name].weight)
            nn.init.zeros_(self.lora_proj_B[adapter_name].weight)
            if init_lora_weights is not True and init_lora_weights.lower() == "lm_svd":
                self.lm_svd_init(adapter_name, rank_large=6)

            print(f"P_A norm after reset: {self.lora_proj_A[adapter_name].weight.data.norm()}")
            print(f"P_B norm after reset: {self.lora_proj_B[adapter_name].weight.data.norm()}")
            print(f"A norm after reset: {self.lora_A[adapter_name].weight.data.norm()}")
            print(f"B norm after reset: {self.lora_B[adapter_name].weight.data.norm()}")

    def get_y_matrix_memory_usage(self) -> float:
        """
        Calculate GPU memory usage of Y matrices only (in bytes)

        Returns:
            float: Memory usage in bytes
        """
        y_memory_bytes = 0.0
        for adapter_name in self.lora_Y:
            y_tensor = self.lora_Y[adapter_name]
            if hasattr(y_tensor, 'weight'):
                tensor = y_tensor.weight
            else:
                tensor = y_tensor
            y_memory_bytes += tensor.numel() * tensor.element_size()
        return y_memory_bytes

    def get_y_matrix_parameter_count(self) -> int:
        """
        Count trainable Y matrix parameters only

        Returns:
            int: Number of Y matrix parameters
        """
        y_param_count = 0
        for adapter_name in self.lora_Y:
            y_tensor = self.lora_Y[adapter_name]
            if hasattr(y_tensor, 'weight'):
                tensor = y_tensor.weight
            else:
                tensor = y_tensor
            y_param_count += tensor.numel()
        return y_param_count

    def get_y_matrix_statistics(self) -> Dict:
        """
        Get comprehensive Y-matrix statistics

        Returns:
            Dict: Dictionary containing Y-matrix statistics
        """
        y_memory_bytes = self.get_y_matrix_memory_usage()
        y_param_count = self.get_y_matrix_parameter_count()

        return {
            'y_matrix_params': y_param_count,
            'y_matrix_memory_bytes': y_memory_bytes,
            'y_matrix_memory_mb': y_memory_bytes / (1024 * 1024),
            'y_matrix_adapters_count': len(self.lora_Y),
            'y_matrix_rank': self.r.get(list(self.lora_Y.keys())[0], 0) if self.lora_Y else 0,
        }




class CoSALinear(nn.Module, CoSALayer):
    def __init__(
        self,
        # in_features,
        # out_features,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        use_compression: bool = False,
        compression_a: int = 128,
        compression_b: int = 128,
        compression_seed: int = None,
        **kwargs,
    ):
        # base_layer = nn.Linear(in_features, out_features, bias=False).to(device)
        super().__init__()
        CoSALayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            use_compression=use_compression,
            compression_a=compression_a,
            compression_b=compression_b,
            compression_seed=compression_seed,
        )
        # self.set_adapter([adapter_name])

    def get_delta_weight_compression(self, adapter) -> torch.Tensor:
        # Direct access to frozen L and R parameters
        Y = self.lora_Y[adapter].weight
        L = self.lora_L[adapter]
        R = self.lora_R[adapter]

        device = Y.device
        dtype = Y.dtype
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        if cast_to_fp32:
            Y = Y.float()
            L = L.float()
            R = R.float()

        # Efficient computation: L @ Y @ R
        Y_R = Y @ R.t()  # Y:(compression_a, compression_b) @ R.t():(compression_b, in_features)
        delta = self.scaling[adapter] * (L.t() @ Y_R)  # L.t():(out_features, compression_a) @ Y_R

        output_tensor = transpose(delta, self.fan_in_fan_out)

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

        return output_tensor
        

    def get_delta_weight(self, adapter) -> torch.Tensor:
        # print(f"self.use_compression[adapter]: {self.use_compression[adapter]}")
        if self.use_compression[adapter]:
            return self.get_delta_weight_compression(adapter)
        device = self.lora_proj_A[adapter].weight.device
        dtype = self.lora_proj_A[adapter].weight.dtype
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        A = self.lora_A[adapter].weight
        B = self.lora_B[adapter].weight
        P_A = self.lora_proj_A[adapter].weight
        P_B = self.lora_proj_B[adapter].weight    
        if cast_to_fp32:
            A = A.float()
            P_A = P_A.float()
            B = B.float()
            P_B = P_B.float()
        # P_B = self.identity_B[adapter].to(B.device).to(B.dtype)
        # delta = A @ P_A + P_B @ B + alpha_ab * A @ B
        delta = P_A @ A + P_B @ B + A @ B
        output_tensor = transpose(delta, self.fan_in_fan_out) * self.scaling[adapter]
        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)
            # cast back the weights
            self.lora_A[adapter].weight.data = A.to(dtype)
            self.lora_proj_A[adapter].weight.data = P_A.to(dtype)
            self.lora_B[adapter].weight.data = B.to(dtype)
            self.lora_proj_B[adapter].weight.data = P_B.to(dtype)

        return output_tensor

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:                
            # if active_adapter in self.lora_A.keys():
            if active_adapter in self.lora_A.keys() and active_adapter in self.lora_B.keys() or active_adapter in self.lora_Y.keys():
                # print(f"Merging adapter: {active_adapter}")
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    
                    # Calculate delta weight using CoSA approach
                    delta_weight = self.get_delta_weight(active_adapter)
                    
                    if not self.use_dora[active_adapter]:
                        orig_weights = orig_weights + delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(orig_weights, transpose(delta_weight, self.fan_in_fan_out), scaling=1)
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                        orig_weights = dora_factor * (orig_weights + delta_weight)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    # Calculate delta weight using CoSA approach
                    delta_weight = self.get_delta_weight(active_adapter)
                    # print(f"delta_weight: {delta_weight.shape}")
                    
                    if not self.use_dora[active_adapter]:
                        # print(f"base_layer.weight: {base_layer.weight.shape}")
                        base_layer.weight.data = base_layer.weight.data + delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(
                                base_layer.weight, transpose(delta_weight, self.fan_in_fan_out), scaling=1
                            )
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                        new_weight = dora_factor * (base_layer.weight.data + delta_weight)
                        base_layer.weight.data = new_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        print(f"Unmerging: {self.merged_adapters}")
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    weight_orig = weight.data / dora_factor.view(-1, 1) - delta_weight
                    weight.data = weight_orig

    def kron_l_y_r(self, active_adapter):
        """
        Compute vec(L * Y * R) using (R^T ⊗ L) @ vec(Y)

        L: [m, a]
        Y: [a, b]
        R: [b, n]
        Return: [m, n] = (R^T ⊗ L) @ vec(Y) reshaped to (m, n)
        """
        Yw = self.lora_Y[active_adapter].weight   # [a, b]

        m, a = self.out_features, self.compression_a[active_adapter]
        b, n = self.compression_b[active_adapter], self.in_features
        assert Yw.shape == (a, b), f"Expected Y: ({a}, {b}), got {Yw.shape}"

        vec_Y = Yw.reshape(-1, 1)  # [a * b, 1]

        # print(f"kron shape: {kron.shape}, vec_Y shape: {vec_Y.shape}")
        vec_delta_W = self.kron[active_adapter] @ vec_Y  # [m * n, 1]
        delta_W = vec_delta_W.reshape(m, n)  # use reshape instead of view
        print(f"vec_delta_W shape: {delta_W.shape}, norm: {delta_W.norm()}")
        return delta_W

    
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        # self.base_layer.to(x.device)  # Ensure base layer is on the same device as input

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                if self.use_compression[active_adapter]:
                    # Direct access to frozen L and R parameters
                    Y = self.lora_Y[active_adapter]
                    L = self.lora_L[active_adapter]
                    R = self.lora_R[active_adapter]
                    dtype = Y.weight.dtype

                    x = x.to(dtype)
                    drop_x = dropout(x)

                    # Efficient L @ Y @ R computation
                    r_output = torch.matmul(drop_x, R)  # [batch, seq, compression_b]
                    y_output = Y(r_output)  # [batch, seq, compression_a]
                    l_output = torch.matmul(y_output, L)  # [batch, seq, out_features]

                    result = result + l_output * scaling
                    return result.to(torch_result_dtype)

                if active_adapter not in self.lora_A.keys():
                    continue
                
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                lora_proj_A = self.lora_proj_A[active_adapter]
                lora_proj_B = self.lora_proj_B[active_adapter]
                x = x.to(lora_A.weight.dtype) # [1, 512, 4096]]
                drop_x = dropout(x)

                # for param in lora_A.parameters():
                #     param.requires_grad = False

                if not self.use_dora[active_adapter]:
                    # Standard CoSA forward pass
                    # Instead of A(B(x)), we'll implement our own logic                    

                    # A_output = lora_A(drop_x) @ lora_proj_A   # A_output: torch.Size([1, 512, 4096])
                    A_output = lora_proj_A(lora_A(drop_x))  
                    B_output = lora_B(lora_proj_B(drop_x))  # B_output: torch.Size([1, 512, 4096])
                    lora_AB = lora_B(lora_A(drop_x))
                    # print(f"A_output: {A_output.norm()}")  # AB: torch.Size([1, 512, 4096])
                    # print(f"lora_proj_A: {lora_proj_A.weight.norm()}")  # lora_proj_A: torch.Size([r, out_features])
                    # print(f"lora_AB: {lora_AB.norm()}")  # lora_AB: torch.Size([1, 512, 4096])

                    # B_output = lora_B(drop_x @ identity_B)  # [1, 512, 4096]

                    # Combine A*P_A + P_B*B
                    # Add standard AB term
                    # AB_output = lora_B(lora_A(drop_x))
                    # AB_output = lora_A(drop_x) @ lora_B.weight.T
                    # result = result + (A_output + B_output + alpha_ab * AB_output) * scaling
                    # print(f"result.norm(): {result.norm()}")
                    # result = result + (A_output + alpha_ab * AB_output) * scaling
                    result = result + (A_output + B_output) * scaling
                    # print(f"result.norm(): {result.norm()}")

                else:
                    # DoRA with CoSA
                    x = dropout(x)
                    result = result + self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        # lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                    )

            result = result.to(torch_result_dtype)

        return result

    def state_dict(self, *args, **kwargs):
        """Override state_dict to exclude L and R matrices from being saved."""
        state = super().state_dict(*args, **kwargs)
        # Filter out L and R matrix keys to reduce checkpoint size
        filtered_state = {}
        for key, value in state.items():
            if '.lora_L.' not in key and '.lora_R.' not in key:
                filtered_state[key] = value
        return filtered_state

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Override to handle missing L and R matrices during loading."""
        # First call parent's implementation
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

        # After loading Y matrices, regenerate L and R if they're missing
        for adapter_name in self.lora_Y.keys():
            if adapter_name not in self.lora_L or adapter_name not in self.lora_R:
                if self.use_compression.get(adapter_name, False):
                    device = self.lora_Y[adapter_name].weight.device
                    dtype = self.lora_Y[adapter_name].weight.dtype

                    L = self._generate_compression_matrix_L(adapter_name, device, dtype)
                    R = self._generate_compression_matrix_R(adapter_name, device, dtype)

                    self.lora_L[adapter_name] = nn.Parameter(L, requires_grad=False)
                    self.lora_R[adapter_name] = nn.Parameter(R, requires_grad=False)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "cosa." + rep


class CoSAEmbedding(nn.Module, CoSALayer):
    """CoSA implementation for Embedding layers."""
    
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        CoSALayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=False,  # DoRA is not supported for Embedding
        )

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora=False):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        weight_A = torch.randn((self.in_features, r))
        weight_B = torch.randn((r, self.out_features))
        self.lora_embedding_A[adapter_name] = nn.Parameter(weight_A)
        self.lora_embedding_B[adapter_name] = nn.Parameter(weight_B)
        
        # # Create identity matrices for CoSA
        # self.lora_proj_A[adapter_name] = nn.Parameter(torch.eye(r, self.out_features), requires_grad=False)
        # self.identity_B[adapter_name] = nn.Parameter(torch.eye(self.in_features, r), requires_grad=False)
        
        
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter using CoSA approach: A*P_A + P_B*B
        """
        device = self.lora_embedding_B[adapter].device
        dtype = self.lora_embedding_A[adapter].dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_embedding_A[adapter]
        weight_B = self.lora_embedding_B[adapter]
        lora_proj_A = self.lora_proj_A[adapter].to(device).to(dtype)
        identity_B = self.identity_B[adapter].to(device).to(dtype)

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()
            lora_proj_A = lora_proj_A.float()
            identity_B = identity_B.float()

        # Implement CoSA: A*P_A + P_B*B instead of A*B
        A_part = weight_A @ lora_proj_A
        B_part = identity_B @ weight_B
        
        # For Embedding layers
        output_tensor = transpose(A_part + B_part, True) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)
            self.lora_embedding_A[adapter] = weight_A.to(dtype)
            self.lora_embedding_B[adapter] = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # TODO: no dtype conversion here, unlike in Linear, is that correct?
        # Forward pass for CoSA Embedding layer
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_embedding_A:
                    continue
                    
                # Get the parameters
                # embedding_A = self.lora_embedding_A[active_adapter].T
                # embedding_B = self.lora_embedding_B[active_adapter].T
                embedding_A = self.lora_embedding_A[active_adapter]
                print(f"embedding_A: {embedding_A.shape}")
                embedding_B = self.lora_embedding_B[active_adapter]
                print(f"embedding_B: {embedding_B.shape}")
                lora_proj_A = self.lora_proj_A[active_adapter].to(embedding_A.device).to(embedding_A.dtype)
                print(f"lora_proj_A: {lora_proj_A.shape}")
                identity_B = self.identity_B[active_adapter].to(embedding_B.device).to(embedding_B.dtype)
                print(f"identity_B: {identity_B.shape}")
                scaling = self.scaling[active_adapter]
                print(f"scaling: {scaling}")
                print(f"x: {x.shape}")
                
                # CoSA approach: A*P_A + P_B*B
                A_part = embedding_A @ lora_proj_A
                print(f"A_part: {A_part.shape}")
                B_part = identity_B @ embedding_B
                print(f"B_part: {B_part.shape}")
                
                # Apply the embedding
                after_A = self._embed(x, A_part)
                after_B = self._embed(x, B_part)
                print(f"after_A: {after_A.shape}")
                print(f"after_B: {after_B.shape}")
                
                # Sum the two parts
                result = result + (after_A + after_B) * scaling
                print(f"result: {result.shape}")
                
            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "cosa." + rep


class CoSAConv2d(nn.Module, CoSALayer):
    """CoSA implementation for Conv2d layers."""
    
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        CoSALayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        base_layer = self.get_base_layer()
        kernel_size = base_layer.kernel_size
        stride = base_layer.stride
        padding = base_layer.padding
        self.lora_A[adapter_name] = nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False)
        self.lora_B[adapter_name] = nn.Conv2d(r, self.out_features, (1, 1), (1, 1), bias=False)
        
        # # Create identity matrices for CoSA - for Conv2d, identity will be applied differently
        # self.lora_proj_A[adapter_name] = nn.Parameter(torch.ones(r, self.out_features), requires_grad=False)
        # self.identity_B[adapter_name] = nn.Parameter(torch.ones(self.in_features, r), requires_grad=False)
        
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter using CoSA approach: A*P_A + P_B*B
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_A[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        # For Conv2d, implementing CoSA is more complex
        # We need to handle the spatial dimensions carefully
        
        # For 1x1 convolutions, it's similar to the Linear case
        if self.get_base_layer().weight.size()[2:4] == (1, 1):
            # Conv2d 1x1
            A_squeezed = weight_A.squeeze(3).squeeze(2)
            B_squeezed = weight_B.squeeze(3).squeeze(2)
            
            # Get identity matrices
            lora_proj_A = self.lora_proj_A[adapter].to(device).to(dtype)
            identity_B = self.identity_B[adapter].to(device).to(dtype)
            
            if cast_to_fp32:
                lora_proj_A = lora_proj_A.float()
                identity_B = identity_B.float()
            
            # Apply CoSA: A*P_A + P_B*B
            A_part = A_squeezed @ lora_proj_A
            B_part = identity_B @ B_squeezed
            
            # Reshape back to 4D
            output_tensor = (A_part + B_part).unsqueeze(2).unsqueeze(3) * self.scaling[adapter]
        else:
            # For Conv2d 3x3 or other sizes
            # We'll use the standard LoRA approach but apply the CoSA concept
            # This is a simplified approximation and might not be perfect
            
            # Regular LoRA term: F.conv2d(weight_A.permute(1, 0, 2, 3), weight_B).permute(1, 0, 2, 3)
            
            # For CoSA in Conv2d, we would ideally want to apply identity matrices
            # But the direct application is complex, so we'll separate the operations
            
            # A term
            A_output = F.conv2d(
                torch.eye(weight_A.size(1)).reshape(weight_A.size(1), weight_A.size(1), 1, 1).to(device).to(dtype),
                weight_A,
            ).permute(1, 0, 2, 3)
            
            # B term
            B_output = F.conv2d(
                weight_B.permute(1, 0, 2, 3),
                torch.eye(weight_B.size(0)).reshape(weight_B.size(0), weight_B.size(0), 1, 1).to(device).to(dtype),
            )
            
            # Combine the two terms
            output_tensor = (A_output + B_output) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    # Standard CoSA approach for Conv2d
                    x_dropout = dropout(x)
                    
                    # A path
                    A_output = lora_A(x_dropout)
                    
                    # B path
                    B_output = lora_B(x_dropout)
                    
                    # Sum them up
                    result = result + (A_output + B_output) * scaling
                else:
                    # DoRA with CoSA
                    x = dropout(x)
                    result = result + self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                    )

            result = result.to(torch_result_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "cosa." + rep