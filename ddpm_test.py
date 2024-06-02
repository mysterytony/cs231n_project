import ddpm
import torch

B, C, H, W = 8, 64, 32, 32
torch.manual_seed(123)
input = torch.randn((B, C, H, W))
torch.manual_seed(123)
a = ddpm.AttentionBlock(n_channels=64, n_heads=2, d_k=16)
out_a = a.forward(input)
torch.manual_seed(123)
b = ddpm.AttentionBlock2(n_channels=64, n_heads=2, d_k=16)
out_b = b.forward(input)
torch.testing.assert_close(out_b, out_a)