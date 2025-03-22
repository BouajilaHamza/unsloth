import torch
import torch.nn as nn
from unsloth.kernels.dynamic_tanh import DynamicTanh
import time

def test_dynamic_tanh():
    # Test parameters
    batch_size = 32
    seq_len = 128
    hidden_dim = 768
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, hidden_dim).cuda()
    
    # Initialize modules
    layer_norm = nn.LayerNorm(hidden_dim).cuda()
    dynamic_tanh = DynamicTanh(hidden_dim).cuda()
    
    # Warmup
    for _ in range(10):
        _ = layer_norm(x)
        _ = dynamic_tanh(x)
    
    # Speed test
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(100):
        _ = layer_norm(x)
    torch.cuda.synchronize()
    ln_time = time.time() - start_time
    
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(100):
        _ = dynamic_tanh(x)
    torch.cuda.synchronize()
    dt_time = time.time() - start_time
    
    print(f"LayerNorm time: {ln_time:.4f}s")
    print(f"DynamicTanh time: {dt_time:.4f}s")
    print(f"Speedup: {ln_time/dt_time:.2f}x")

# Simple transformer block for testing
class SimpleTransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, use_dynamic_tanh=False):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_attention_heads)
        self.norm1 = DynamicTanh(hidden_size) if use_dynamic_tanh else nn.LayerNorm(hidden_size)
        self.norm2 = DynamicTanh(hidden_size) if use_dynamic_tanh else nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
    
    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        return x

if __name__ == "__main__":
    test_dynamic_tanh()
    
    # Test in transformer block
    hidden_size = 768
    num_heads = 12
    batch_size = 8
    seq_len = 64
    
    x = torch.randn(seq_len, batch_size, hidden_size).cuda()
    
    model_ln = SimpleTransformerBlock(hidden_size, num_heads, use_dynamic_tanh=False).cuda()
    model_dt = SimpleTransformerBlock(hidden_size, num_heads, use_dynamic_tanh=True).cuda()
    
    out_ln = model_ln(x)
    out_dt = model_dt(x)
    
    print("\nTransformer output shapes:")
    print(f"LayerNorm: {out_ln.shape}")
    print(f"DynamicTanh: {out_dt.shape}")
