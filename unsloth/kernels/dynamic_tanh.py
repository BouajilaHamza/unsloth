import triton
import triton.language as tl
import torch
import torch.nn as nn
# from .utils import calculate_settings, torch_cuda_device

class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, elementwise_affine=True, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.elementwise_affine = elementwise_affine
        self.alpha_init_value = alpha_init_value
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        return dynamic_tanh_forward(x, self.alpha, self.weight if self.elementwise_affine else None, self.bias if self.elementwise_affine else None)

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, elementwise_affine={self.elementwise_affine}, alpha_init_value={self.alpha_init_value}"

@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1

@triton.jit
def dynamic_tanh_kernel(
    x_ptr, alpha_ptr, weight_ptr, bias_ptr, y_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    alpha = tl.load(alpha_ptr)
    y = tl.tanh(alpha * x)
    if weight_ptr is not None and bias_ptr is not None:
        weight = tl.load(weight_ptr + offsets, mask=mask)
        bias = tl.load(bias_ptr + offsets, mask=mask)
        y = y * weight + bias
    tl.store(y_ptr + offsets, y, mask=mask)

def dynamic_tanh_forward(x, alpha, weight=None, bias=None):
    batch_size, seq_len, hidden_dim = x.shape
    n_elements = batch_size * seq_len * hidden_dim
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    dynamic_tanh_kernel[grid](
        x, alpha, weight, bias, y,
        n_elements=n_elements,
        BLOCK_SIZE=1024,
    )
    return y