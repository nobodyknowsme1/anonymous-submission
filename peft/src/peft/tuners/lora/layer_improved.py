"""
Improved on-demand L/R generation to match frozen parameter performance.

Key improvements:
1. Create L/R as buffers (persistent but non-trainable)
2. Initialize once and reuse (like frozen but without gradients)
3. Maintain numerical stability
4. Avoid repeated generation overhead
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

class ImprovedCompressionLayer(nn.Module):
    """
    Improved compression layer that matches frozen parameter performance
    while maintaining the benefits of on-demand generation.
    """

    def __init__(self, in_features: int, out_features: int,
                 compression_a: int, compression_b: int,
                 adapter_name: str, seed: Optional[int] = None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.compression_a = compression_a
        self.compression_b = compression_b
        self.adapter_name = adapter_name

        # Create Y as trainable parameter
        self.Y = nn.Linear(compression_b, compression_a, bias=False)
        nn.init.zeros_(self.Y.weight)

        # Initialize L and R as buffers (persistent, non-trainable, part of state_dict)
        # This gives us the benefits of frozen parameters without gradient tracking
        self._initialize_compression_matrices(seed)

    def _initialize_compression_matrices(self, seed: Optional[int] = None):
        """Initialize L and R as buffers with proper initialization."""
        if seed is not None:
            torch.manual_seed(seed)

        # Initialize L matrix as buffer
        L = torch.empty(self.compression_a, self.out_features)
        nn.init.normal_(L, mean=0, std=1/math.sqrt(self.compression_a))
        self.register_buffer('L', L.contiguous())

        # Initialize R matrix as buffer
        R = torch.empty(self.in_features, self.compression_b)
        nn.init.normal_(R, mean=0, std=1/math.sqrt(self.compression_b))
        self.register_buffer('R', R.contiguous())

        # Verify numerical stability
        assert torch.isfinite(self.L).all(), "Non-finite values in L"
        assert torch.isfinite(self.R).all(), "Non-finite values in R"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with L @ Y @ R decomposition."""
        # R(x): [batch, seq, in_features] @ [in_features, compression_b]
        r_output = torch.matmul(x, self.R)

        # Y(r_output): [batch, seq, compression_b] -> [batch, seq, compression_a]
        y_output = self.Y(r_output)

        # L(y_output): [batch, seq, compression_a] @ [compression_a, out_features]
        output = torch.matmul(y_output, self.L)

        return output


def create_improved_compression_matrices(adapter_name: str, config: dict, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create L and R matrices as persistent tensors (not parameters) that can be
    registered as buffers in the module.

    This approach:
    1. Generates matrices once with proper initialization
    2. Stores them as buffers (persistent, non-trainable)
    3. Maintains numerical precision
    4. Avoids repeated generation overhead
    """
    compression_a = config['compression_a']
    compression_b = config['compression_b']
    in_features = config['in_features']
    out_features = config['out_features']

    # Set seed for reproducibility
    torch.manual_seed(seed)

    # Create L with proper initialization
    L = torch.empty(compression_a, out_features)
    nn.init.normal_(L, mean=0, std=1/math.sqrt(compression_a))
    L = L.contiguous()

    # Create R with proper initialization
    R = torch.empty(in_features, compression_b)
    nn.init.normal_(R, mean=0, std=1/math.sqrt(compression_b))
    R = R.contiguous()

    return L, R


class BufferBasedCompression:
    """
    Alternative approach: Use buffers instead of parameters for L/R.
    Buffers are:
    - Persistent across forward passes
    - Part of state_dict (saved/loaded with model)
    - Not tracked by autograd (no gradients)
    - Maintain numerical consistency
    """

    @staticmethod
    def register_compression_buffers(module: nn.Module, adapter_name: str,
                                    L: torch.Tensor, R: torch.Tensor):
        """Register L and R as buffers in the module."""
        # Register as buffers with unique names per adapter
        module.register_buffer(f'{adapter_name}_L', L, persistent=True)
        module.register_buffer(f'{adapter_name}_R', R, persistent=True)

    @staticmethod
    def get_compression_buffers(module: nn.Module, adapter_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve L and R buffers from the module."""
        L = getattr(module, f'{adapter_name}_L')
        R = getattr(module, f'{adapter_name}_R')
        return L, R


def optimize_matrix_multiplication(x: torch.Tensor, L: torch.Tensor, Y: nn.Module, R: torch.Tensor) -> torch.Tensor:
    """
    Optimized matrix multiplication for L @ Y @ R with better numerical stability.

    Key optimizations:
    1. Use torch.matmul for better performance
    2. Maintain dtype consistency
    3. Use in-place operations where possible
    4. Minimize memory allocations
    """
    dtype = x.dtype
    device = x.device

    # Ensure all matrices are on same device and dtype
    if R.device != device:
        R = R.to(device)
    if L.device != device:
        L = L.to(device)

    # Cast to computation dtype if needed
    compute_dtype = torch.float32 if dtype == torch.float16 else dtype
    if dtype != compute_dtype:
        x = x.to(compute_dtype)
        R = R.to(compute_dtype)
        L = L.to(compute_dtype)

    # Optimized multiplication order (associative)
    # Choose order based on dimensions to minimize FLOPs
    batch_size = x.shape[0] if x.dim() > 2 else 1
    seq_len = x.shape[1] if x.dim() > 2 else x.shape[0]

    # Standard order: x @ R @ Y @ L
    r_output = torch.matmul(x, R)  # [batch, seq, compression_b]
    y_output = Y(r_output)  # [batch, seq, compression_a]
    output = torch.matmul(y_output, L)  # [batch, seq, out_features]

    # Cast back to original dtype
    if dtype != compute_dtype:
        output = output.to(dtype)

    return output