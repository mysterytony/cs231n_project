from typing import Tuple, Optional

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

import math
from typing import Optional, Tuple, Union, List

ENABLE_CLASS_EMBEDDING = True


def gather(consts: torch.Tensor, t: torch.Tensor):
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)


class TimeEmbedding(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        return emb


class ClassEmbedding(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        in_channels = # of classes
        out_channels = embedding dimension
        """
        super().__init__()
        self.embedding_layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.in_channels = in_channels

    def forward(self, class_: torch.Tensor):
        assert class_.shape[1] == self.in_channels
        return self.embedding_layer(class_)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int,
                 n_groups: int = 32, dropout: float = 0.1):
        super().__init__()
        # Group normalization and the first convolution layer
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=(3, 3), padding=(1, 1))

        # Group normalization and the second convolution layer
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=(3, 3), padding=(1, 1))

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        # Linear layer for time embeddings
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = nn.SiLU()

        if ENABLE_CLASS_EMBEDDING:
            self.class_emb = nn.Linear(time_channels, out_channels)
            self.class_act = nn.SiLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c):
        h = self.conv1(self.act1(self.norm1(x)))

        if ENABLE_CLASS_EMBEDDING:
            h *= self.class_emb(self.class_act(c))[:, :, None, None]

        # Add time embeddings
        h += self.time_emb(self.time_act(t))[:, :, None, None]
        # Second convolution layer
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))

        # Add the shortcut connection and return
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        super().__init__()
        if d_k is None:
            d_k = n_channels
        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k ** -0.5
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        _ = t
        batch_size, n_channels, height, width = x.shape
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        qkv = self.projection(x).view(
            batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attn = attn.softmax(dim=2)
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        res = res.reshape(batch_size, -1, self.n_heads * self.d_k)
        res = self.output(res)
        res += x
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return res


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor, c):
        x = self.res(x, t, c)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        # The input has in_channels + out_channels because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(
            in_channels + out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor, c):
        x = self.res(x, t, c)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c):
        x = self.res1(x, t, c)
        x = self.attn(x)
        x = self.res2(x, t, c)
        return x


class Upsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, c):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels,
                              kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c):
        # return F.max_pool2d(x, kernel_size=2)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, image_channels: int = 3, n_channels: int = 64,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 is_attn: Union[Tuple[bool, ...], List[bool]] = (
                     False, False, True, True),
                 n_blocks: int = 2,
                 n_class: int = 10):
        """
        * `image_channels` is the number of channels in the image. $3$ for RGB.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        super().__init__()
        n_resolutions = len(ch_mults)
        self.image_proj = nn.Conv2d(
            image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))
        self.time_emb = TimeEmbedding(n_channels * 4)
        down = []
        out_channels = in_channels = n_channels
        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels,
                            n_channels * 4, is_attn[i]))
                in_channels = out_channels
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))
        self.down = nn.ModuleList(down)
        self.middle = MiddleBlock(out_channels, n_channels * 4, )
        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels,
                          n_channels * 4, is_attn[i]))
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels,
                      n_channels * 4, is_attn[i]))
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels))
        self.up = nn.ModuleList(up)
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = nn.SiLU()
        self.final = nn.Conv2d(in_channels, image_channels,
                               kernel_size=(3, 3), padding=(1, 1))

        if ENABLE_CLASS_EMBEDDING:
            self.n_class = n_class
            self.class_embedding_layer = ClassEmbedding(
                self.n_class, n_channels * 4)

    def forward(self, x: torch.Tensor, time_step: torch.Tensor, class_: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        class_ has shape (batch_size, )
        """
        time_step_embedding = self.time_emb(time_step)
        x = self.image_proj(x)
        class_embedding = torch.tensor([])
        if ENABLE_CLASS_EMBEDDING:
            class_one_hot = nn.functional.one_hot(
                class_, num_classes=self.n_class).type(torch.float)
            class_embedding = self.class_embedding_layer(class_one_hot)

        h = [x]
        for m in self.down:
            x = m(x, time_step_embedding, class_embedding)
            h.append(x)

        x = self.middle(x, time_step_embedding, class_embedding)

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, time_step_embedding, class_embedding)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, time_step_embedding, class_embedding)

        return self.final(self.act(self.norm(x)))


class DenoiseDiffusion:
    def __init__(self, eps_model: nn.Module, n_diffusion_timestep: int, device: torch.device):
        """
        * `eps_model` is $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$ model
        * `n_diffusion_timestep` is $t$
        * `device` is the device to place constants on
        """
        super().__init__()
        self.eps_model = eps_model
        def beta_at_t(t): return math.cos(
            (t + 0.008) / 1.008 * math.pi / 2) ** 2
        betas = []
        for i in range(n_diffusion_timestep):
            t1 = i / n_diffusion_timestep
            t2 = (i+1) / n_diffusion_timestep
            betas.append(min(1 - beta_at_t(t2) / beta_at_t(t1), 0.02))
        self.beta = torch.tensor(betas).to(device)
        # self.beta = torch.linspace(
        #     0.0001, 0.02, n_diffusion_timestep).to(device)

        self.device = device

        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_diffusion_timestep = n_diffusion_timestep
        self.sigma2 = self.beta

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, class_: torch.Tensor):
        eps_theta = self.eps_model(xt, t, class_)
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = gather(self.sigma2, t)
        eps = torch.randn(xt.shape, device=xt.device)
        return mean + (var ** .5) * eps

    def q(self, x0, t, eps=None):
        """
        Add noise from x0 to xt
        """
        if eps is None:
            eps = torch.randn_like(x0)
        noise_mean = torch.sqrt(gather(self.alpha_bar, t)) * x0
        noise_sd = torch.sqrt(1 - gather(self.alpha_bar, t))
        return noise_mean + noise_sd * eps

    def loss(self, x0: torch.Tensor, class_: torch.Tensor):
        batch_size = x0.shape[0]
        t = torch.randint(low=0, high=self.n_diffusion_timestep, size=(batch_size,),
                          device=x0.device, dtype=torch.long)

        eps = torch.randn_like(x0)

        xt = self.q(x0, t, eps)

        eps_theta = self.eps_model(xt, t, class_)

        return F.mse_loss(eps, eps_theta)
