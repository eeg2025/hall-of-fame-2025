import copy
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from typing import Optional, Union, Callable
from einops import rearrange
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from pathlib import Path
from einops.layers.torch import Rearrange


class BaseTransform(ABC):
    """Base class for all transforms."""

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply transform to input array."""
        pass


class ComposedTransform(BaseTransform):
    """Compose multiple transforms into a pipeline."""

    def __init__(self, transforms: List[BaseTransform]):
        self.transforms = transforms

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            x = t(x)
        return x


class RobustChannelNorm(BaseTransform):
    """Channel-wise normalization using median and MAD."""

    def __init__(
        self,
        median: np.ndarray,
        mad: np.ndarray,
        eps: float = 1e-6,
        use_sigma: bool = True,
    ):
        scale = 1.4826 if use_sigma else 1.0
        self.median = np.asarray(median, dtype=np.float32)
        self.scale = np.asarray(mad, dtype=np.float32) * scale
        self.eps = eps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x_np = np.asarray(x, dtype=np.float32)
        x_norm = (x_np - self.median[:, None]) / (self.scale[:, None] + self.eps)
        return x_norm.astype(x.dtype if isinstance(x, np.ndarray) else np.float32)


def _init_final_linear(layer, init_xavier=True):
    if init_xavier:
        torch.nn.init.xavier_uniform_(layer.weight)
    else:
        torch.nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        torch.nn.init.zeros_(layer.bias)


class ResidualGLUBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),  # GLU needs 2x for gate
            nn.GLU(dim=-1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class ImprovedResidualFlattenMLPHead(nn.Module):
    """Improved flatten projection with residual GLU blocks and better gradient flow."""

    def __init__(
        self,
        num_channels: int,
        seq_len: int,
        embed_dim: int,
        dropout: float,
        init_xavier: bool,
        d_proj: int = 1024,
        depth: int = 4,
        mlp_ratio: float = 2.0,
        final_bias: bool = False,
        residual_scale: float = 0.1,  # Scale residual connections
        use_input_skip: bool = True,  # Skip from input to pre-head
    ) -> None:
        super().__init__()
        flattened_dim = num_channels * seq_len * embed_dim
        self.use_input_skip = use_input_skip
        self.residual_scale = residual_scale

        self.proj = nn.Sequential(
            Rearrange("b c s d -> b (c s d)"),
            nn.Linear(flattened_dim, d_proj),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Pre-norm for better stability
        self.input_norm = nn.LayerNorm(d_proj)
        self.blocks = nn.ModuleList(
            [
                ResidualGLUBlock(d_proj, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(depth)
            ]
        )

        # Optional skip projection if using input skip
        if self.use_input_skip:
            self.skip_proj = nn.Linear(d_proj, d_proj)

        self.norm = nn.LayerNorm(d_proj)
        self.head = nn.Linear(d_proj, 1, bias=final_bias)

        # Better initialization
        nn.init.xavier_uniform_(self.proj[1].weight)
        _init_final_linear(self.head, init_xavier)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = self.proj(feats)

        # Store input for potential skip connection
        if self.use_input_skip:
            skip = self.skip_proj(x)

        x = self.input_norm(x)

        # Process through residual blocks with scaling
        for block in self.blocks:
            x = x + self.residual_scale * block(x)

        # Add input skip connection
        if self.use_input_skip:
            x = x + skip

        x = self.norm(x)
        return self.head(x)


class LayerScale(nn.Module):
    """Per-channel learnable residual scaling."""

    def __init__(self, dim: int, init_value: float = 1e-4):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class DropPath(nn.Module):
    """Stochastic depth per sample (when applied in residual branches)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor.floor()
        return x.div(keep_prob) * random_tensor


def _build_fourier_positional_encoding(
    n_positions: int, dim: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Generate deterministic Fourier features for positional information."""

    if dim <= 0:
        raise ValueError("Positional encoding dimension must be positive")

    half_dim = dim // 2
    if half_dim == 0:
        return torch.zeros(n_positions, dim, device=device, dtype=dtype)

    pos = torch.arange(n_positions, device=device, dtype=dtype).unsqueeze(-1)
    freq_bands = torch.logspace(
        0.0, math.log10(1e4), steps=half_dim, device=device, dtype=dtype
    )
    angles = pos / freq_bands.unsqueeze(0)
    pe = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if pe.shape[-1] < dim:
        pad = torch.zeros(n_positions, dim - pe.shape[-1], device=device, dtype=dtype)
        pe = torch.cat((pe, pad), dim=-1)
    return pe


class CBraModSeqFirst(nn.Module):
    """
    Sequence-first CBraMod: embed (B, C, T) -> tokens (B, C, S, D) -> Transformer -> proj.
    """

    def __init__(
        self,
        patch_size: int,
        out_dim: int = 200,
        d_model: int = 200,
        dim_feedforward: int = 600,
        n_layer: int = 6,
        nhead: int = 8,
        drop_path_rate: float = 0.05,
        drop_path: float = 0.0,
        patch_dropout: float = 0.1,
        transformer_dropout: float = 0.1,
        multi_scale_kernels=(15, 31, 63),
        patchify: str = "lppool",
        flash_attn: bool = False,
        rope: bool = False,
        full_attn: bool = False,
    ):
        super().__init__()
        self.patch_size = int(patch_size)
        self.d_model = d_model

        self.seq_embed = SeqFirstEmbedding(
            d_model=d_model,
            patch_size=self.patch_size,
            multi_scale_kernels=multi_scale_kernels,
            patch_dropout=patch_dropout,
            patchify=patchify,
        )

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,
            activation=F.gelu,
            drop_path=drop_path,
            dropout=transformer_dropout,
            flash_attn=flash_attn,
            rope=rope,
            full_attn=full_attn,
        )
        drop_path_rates = (
            torch.linspace(0, drop_path_rate, steps=n_layer).tolist()
            if n_layer > 0
            else []
        )
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_layers=n_layer,
            drop_path_rates=drop_path_rates,
        )
        self.proj_out = nn.Linear(d_model, out_dim)
        self.apply(_weights_init)
        self.full_attn = full_attn

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, C, T)
        returns: (B, C, S, out_dim)
        """
        tokens = self.seq_embed(x)  # (B, C, S, D)
        feats = self.encoder(tokens)  # (B, C, S, D)
        return self.proj_out(feats)  # (B, C, S, out_dim)


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        d_model: int,
        seq_len: int,
        multi_scale_kernels=(15, 31, 63),
        patch_dropout: float = 0.05,
        rope: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.freq_bins = in_dim // 2 + 1
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=True)
        self.rope = rope
        num_branches = len(multi_scale_kernels)
        base_dim = d_model // num_branches
        branch_dims = [base_dim] * num_branches
        branch_dims[-1] += d_model - sum(branch_dims)

        branches = []
        for kernel, branch_dim in zip(multi_scale_kernels, branch_dims):
            padding = kernel // 2
            groups = math.gcd(4, branch_dim)
            branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        1, branch_dim, kernel_size=kernel, padding=padding, bias=False
                    ),
                    nn.GroupNorm(groups, branch_dim),
                    nn.GELU(),
                    nn.Conv1d(branch_dim, branch_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(groups, branch_dim),
                    nn.GELU(),
                    nn.AdaptiveAvgPool1d(1),
                )
            )
        self.temporal_branches = nn.ModuleList(branches)

        self.proj_norm = nn.LayerNorm(d_model)
        self.spectral_proj = nn.Sequential(
            nn.Linear(self.freq_bins, d_model, bias=False),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(patch_dropout),
        )
        self.spectral_scale = nn.Parameter(torch.zeros(1))
        self.embed_dropout = nn.Dropout(patch_dropout)
        self.channel_pos = None
        self.temporal_pos = None

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        bz, ch_num, patch_num, patch_size = x.shape

        if mask is None:
            mask_x = x
        else:
            mask_x = x.clone()
            fill_value = self.mask_encoding[:patch_size]
            mask_x[mask == 1] = fill_value

        seq = mask_x.contiguous().view(bz * ch_num * patch_num, 1, patch_size)

        branch_outputs = []
        for branch in self.temporal_branches:
            out = branch(seq).squeeze(-1)
            branch_outputs.append(out)
        token_emb = torch.cat(branch_outputs, dim=1)
        token_emb = token_emb.view(bz, ch_num, patch_num, self.d_model)

        spectral = torch.fft.rfft(seq.squeeze(1), dim=-1, norm="forward")
        spectral = torch.abs(spectral).view(bz * ch_num * patch_num, self.freq_bins)
        spectral_emb = self.spectral_proj(spectral)
        spectral_emb = spectral_emb.view(bz, ch_num, patch_num, self.d_model)

        spectral_weight = torch.tanh(self.spectral_scale)
        patch_emb = token_emb + spectral_weight * spectral_emb
        if self.rope:
            if self.channel_pos is None or self.channel_pos.shape[1] != ch_num:
                channel_pos_init = _build_fourier_positional_encoding(
                    ch_num, self.d_model, patch_emb.device, patch_emb.dtype
                ).view(1, ch_num, 1, self.d_model)
                self.channel_pos = nn.Parameter(channel_pos_init)
            channel_pos = self.channel_pos

            if self.temporal_pos is None or self.temporal_pos.shape[2] != patch_num:
                temporal_pos_init = _build_fourier_positional_encoding(
                    patch_num, self.d_model, patch_emb.device, patch_emb.dtype
                ).view(1, 1, patch_num, self.d_model)
                self.temporal_pos = nn.Parameter(temporal_pos_init)
            temporal_pos = self.temporal_pos

            patch_emb = patch_emb + channel_pos + temporal_pos

        patch_emb = self.proj_norm(patch_emb)
        patch_emb = self.embed_dropout(patch_emb)

        return patch_emb


class SeqFirstEmbedding(nn.Module):
    """
    Full-sequence multi-scale temporal embedding, then patchify by pooling.

    Input : x (B, C, T)
    Output: tokens (B, C, S, D) where S = floor(T / patch_size)
    """

    def __init__(
        self,
        d_model: int,
        patch_size: int,
        multi_scale_kernels=(15, 31, 63),
        patch_dropout: float = 0.05,
        patchify: str = "lppool",
    ):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(
                "d_model must be even to split dims for (channel|time) PEs."
            )
        self.d_model = d_model
        self.patch_size = int(patch_size)
        self.patchify = (patchify or "lppool").lower()
        if self.patchify not in {"lppool", "mlp"}:
            raise ValueError("patchify must be 'lppool' or 'mlp'")

        # -------- Residual temporal branches (no temporal pooling here!) --------
        num_branches = len(multi_scale_kernels)
        base_dim = d_model // num_branches
        branch_dims = [base_dim] * num_branches
        branch_dims[-1] += d_model - sum(branch_dims)

        class ResidualBranch(nn.Module):
            def __init__(self, kernel: int, branch_dim: int):
                super().__init__()
                padding = kernel // 2
                groups = math.gcd(4, branch_dim)
                self.conv1 = nn.Conv1d(
                    1, branch_dim, kernel_size=kernel, padding=padding, bias=False
                )
                self.gn1 = nn.GroupNorm(groups, branch_dim)
                self.act1 = nn.GELU()

            def forward(self, x: Tensor) -> Tensor:
                y = self.act1(self.gn1(self.conv1(x)))  # (N, branch_dim, T)
                return y

        self.temporal_branches = nn.ModuleList(
            [
                ResidualBranch(kernel, branch_dim)
                for kernel, branch_dim in zip(multi_scale_kernels, branch_dims)
            ]
        )

        # -------- Positional encodings (fixed Fourier) applied AFTER patchify --------
        self.proj_norm = nn.LayerNorm(d_model)
        self.embed_dropout = nn.Dropout(patch_dropout)

        # Patch projection: choose between LP pooling and MLP
        if self.patchify == "lppool":
            self.patch_pool = nn.LPPool1d(
                norm_type=2, kernel_size=self.patch_size, stride=self.patch_size
            )
        else:
            self.patch_mlp = nn.Sequential(
                nn.Linear(self.d_model * self.patch_size, 2 * self.d_model),
                nn.GELU(),
                nn.Dropout(patch_dropout),
                nn.Linear(2 * self.d_model, self.d_model),
                nn.Dropout(patch_dropout),
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, C, T) raw sequence
        return: (B, C, S, D) tokens after sequence-first embedding then patchify
        """
        B, C, T = x.shape
        S = T // self.patch_size
        if S == 0:
            raise ValueError(
                f"Sequence length T={T} shorter than patch_size={self.patch_size}."
            )
        T_trim = S * self.patch_size
        x = x[..., :T_trim]  # (B, C, T_trim)

        # --- Temporal multi-branch embedding on full sequence ---
        seq = x.reshape(B * C, 1, T_trim)  # (B*C, 1, T_trim)
        branch_outs = [
            branch(seq) for branch in self.temporal_branches
        ]  # each: (B*C, d_i, T_trim)
        h = torch.cat(branch_outs, dim=1)  # (B*C, D, T_trim)

        # --- Patchify AFTER embedding ---
        if self.patchify == "lppool":
            h_tokens = self.patch_pool(h)  # (B*C, D, S)
            h_tokens = h_tokens.view(B, C, self.d_model, S).permute(0, 1, 3, 2)
        else:
            # MLP over flattened (D x P) patches
            h = h.view(B, C, self.d_model, S, self.patch_size).permute(0, 1, 3, 2, 4)
            patches = h.reshape(B * C * S, self.d_model * self.patch_size)
            h_tokens = self.patch_mlp(patches).view(B, C, S, self.d_model)

        # --- Fixed split positional encodings (channel half | temporal half) ---
        d_half = self.d_model // 2
        ch_pe = _build_fourier_positional_encoding(
            n_positions=C, dim=d_half, device=h_tokens.device, dtype=h_tokens.dtype
        ).view(1, C, 1, d_half)
        tm_pe = _build_fourier_positional_encoding(
            n_positions=S, dim=d_half, device=h_tokens.device, dtype=h_tokens.dtype
        ).view(1, 1, S, d_half)

        h_tokens = torch.cat(
            (h_tokens[..., :d_half] + ch_pe, h_tokens[..., d_half:] + tm_pe), dim=-1
        )

        # --- Normalize + dropout ---
        h_tokens = self.proj_norm(h_tokens)
        h_tokens = self.embed_dropout(h_tokens)
        return h_tokens  # (B, C, S, D)


def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        norm=None,
        enable_nested_tensor=True,
        mask_check=True,
        rope: bool = False,
        drop_path_rates: Optional[list] = None,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        if drop_path_rates is not None:
            if len(drop_path_rates) != num_layers:
                raise ValueError("Length of drop_path_rates must match num_layers")
            for layer, drop_prob in zip(self.layers, drop_path_rates):
                if hasattr(layer, "set_drop_path"):
                    layer.set_drop_path(drop_prob)

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = None,
    ) -> Tensor:
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d: int, base: int = 2000):
        super().__init__()
        self.base = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x: torch.Tensor):
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return

        seq_len = x.shape[0]
        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(
            x.device
        )  # THETA = 10,000^(-2*i/d) or 1/10,000^(2i/d)
        seq_idx = (
            torch.arange(seq_len, device=x.device).float().to(x.device)
        )  # Position Index -> [0,1,2...seq-1]
        idx_theta = torch.einsum(
            "n,d->nd", seq_idx, theta
        )  # Calculates m*(THETA) = [ [0, 0...], [THETA_1, THETA_2...THETA_d/2], ... [seq-1*(THETA_1), seq-1*(THETA_2)...] ]
        idx_theta2 = torch.cat(
            [idx_theta, idx_theta], dim=1
        )  # [THETA_1, THETA_2...THETA_d/2] -> [THETA_1, THETA_2...THETA_d]

        self.cos_cached = idx_theta2.cos()[
            :, None, None, :
        ]  # Cache [cosTHETA_1, cosTHETA_2...cosTHETA_d]
        self.sin_cached = idx_theta2.sin()[
            :, None, None, :
        ]  # cache [sinTHETA_1, sinTHETA_2...sinTHETA_d]

    def _neg_half(self, x: torch.Tensor):
        d_2 = self.d // 2  #
        return torch.cat(
            [-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1
        )  # [x_1, x_2,...x_d] -> [-x_d/2, ... -x_d, x_1, ... x_d/2]

    def forward(self, x: torch.Tensor):
        self._build_cache(x)
        neg_half_x = self._neg_half(x)
        x_rope = (x * self.cos_cached[: x.shape[0]]) + (
            neg_half_x * self.sin_cached[: x.shape[0]]
        )  # [x_1*cosTHETA_1 - x_d/2*sinTHETA_d/2, ....]
        return x_rope


class FlashMultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        dropout=0.0,
        bias=True,
        batch_first=True,
        rope=False,
        base=10000,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout
        self.to_qkv = nn.Linear(d_model, d_model * 3, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        self.sqrt_dim = self.d_k**0.5
        self.batch_first = batch_first

        self.rope = rope
        self.base = base
        if self.rope:
            if self.d_k % 2 != 0:
                raise ValueError("d_k must be even for RoPE")
            self.rope_emb = RotaryPositionalEmbeddings(d=self.d_k, base=self.base)

    def forward(
        self,
        x: torch.tensor,
        x2: torch.tensor,
        x3: torch.tensor,
        attn_mask: Optional[torch.tensor] = None,
        key_padding_mask: Optional[torch.tensor] = None,
        need_weights: Optional[bool] = False,
    ):
        """
        x2 and x3 are useless here, just to match nn.MultiheadAttention signature
        Args:
            x: (B, T, D) input tensor
            attention_mask: (B, T) or (B, 1, T, T) boolean mask where True = ignore
                          or additive float mask where -inf = ignore
        Returns:
            out: (B, T, D) output tensor
        """
        # Project to Q, K, V
        qkv = self.to_qkv(x)  # (B, T, 3*D)
        q, k, v = rearrange(qkv, "b t (n h d) -> n b h t d", n=3, h=self.num_heads)
        if self.rope:
            q = self.rope_emb(q)
            k = self.rope_emb(k)
        # Prepare attention mask for FSDPA
        # FSDPA expects: None, (B, T), (B, H, T, T), or (B, 1, T, T)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                orig_mask = attn_mask
                attn_mask = torch.zeros(orig_mask.shape, dtype=x.dtype, device=x.device)
                attn_mask.masked_fill_(orig_mask, float("-inf"))

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout, is_causal=False
        )  # (B, H, T, D)

        # Merge heads
        attn_output = rearrange(attn_output, "b h t d -> b t (h d)")

        # Output projection
        out = self.W_o(attn_output)

        return out, None


class TransformerEncoderLayer(nn.Module):
    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
        drop_path: float = 0.0,
        flash_attn: bool = False,
        rope: bool = False,
        full_attn: bool = False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.full_attn = full_attn
        if nhead % 2 != 0:
            raise ValueError("nhead must be even for axis-split attention")
        if flash_attn or rope:
            self.self_attn_s = FlashMultiheadAttention(
                d_model // 2,
                nhead // 2,
                dropout=dropout,
                bias=bias,
                batch_first=batch_first,
                rope=rope,
                **factory_kwargs,
            )
            self.self_attn_t = FlashMultiheadAttention(
                d_model // 2,
                nhead // 2,
                dropout=dropout,
                bias=bias,
                batch_first=batch_first,
                rope=rope,
                **factory_kwargs,
            )
        else:
            if self.full_attn:
                self.self_attn = nn.MultiheadAttention(
                    d_model,
                    nhead,
                    dropout=dropout,
                    bias=bias,
                    batch_first=batch_first,
                    **factory_kwargs,
                )
            else:
                self.self_attn_s = nn.MultiheadAttention(
                    d_model // 2,
                    nhead // 2,
                    dropout=dropout,
                    bias=bias,
                    batch_first=batch_first,
                    **factory_kwargs,
                )
                self.self_attn_t = nn.MultiheadAttention(
                    d_model // 2,
                    nhead // 2,
                    dropout=dropout,
                    bias=bias,
                    batch_first=batch_first,
                    **factory_kwargs,
                )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(
            d_model, dim_feedforward * 2, bias=bias, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ls1 = LayerScale(d_model, init_value=1e-4)
        self.ls2 = LayerScale(d_model, init_value=1e-4)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = F.relu

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        x = src
        if self.full_attn:
            sa_output = self._sa_block_full(
                self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
            )
        else:
            sa_output = self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
            )
        x = x + self.drop_path(self.ls1(sa_output))
        ff_output = self._ff_block(self.norm2(x))
        x = x + self.drop_path(self.ls2(ff_output))
        return x

    def _sa_block_full(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        bz, ch_num, patch_num, patch_size = x.shape

        # Reshape to [batch_size, sequence_length, embed_dim]
        # where sequence_length = ch_num * patch_num and embed_dim = patch_size
        x_reshaped = (
            x.permute(0, 2, 1, 3).contiguous().view(bz, patch_num * ch_num, patch_size)
        )

        # Apply full self-attention across all channels and patches
        attn_output = self.self_attn(
            x_reshaped,
            x_reshaped,
            x_reshaped,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]

        # Reshape back to original dimensions
        output = (
            attn_output.view(bz, patch_num, ch_num, patch_size)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        return self.dropout1(output)

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        bz, ch_num, patch_num, patch_size = x.shape
        xs = x[:, :, :, : patch_size // 2]
        xt = x[:, :, :, patch_size // 2 :]
        xs = (
            xs.transpose(1, 2)
            .contiguous()
            .view(bz * patch_num, ch_num, patch_size // 2)
        )
        xt = xt.contiguous().view(bz * ch_num, patch_num, patch_size // 2)
        xs = self.self_attn_s(
            xs,
            xs,
            xs,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        xs = (
            xs.contiguous().view(bz, patch_num, ch_num, patch_size // 2).transpose(1, 2)
        )
        xt = self.self_attn_t(
            xt,
            xt,
            xt,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        xt = xt.contiguous().view(bz, ch_num, patch_num, patch_size // 2)
        local = torch.cat((xs, xt), dim=3)
        return self.dropout1(local)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x_proj = self.linear1(x)
        x_u, x_v = x_proj.chunk(2, dim=-1)
        x = self.linear2(self.dropout(F.silu(x_v) * x_u))
        return self.dropout2(x)

    def set_drop_path(self, drop_prob: float) -> None:
        self.drop_path = DropPath(drop_prob) if drop_prob > 0.0 else nn.Identity()


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class ModelChall1_seqfirst(nn.Module):
    """Challenge 1 regressor built on CBraMod with conv-before-patch and flatten MLP.
    Equivalent to ModelChall1, but embedding happens on the full sequence first.
    """

    def __init__(
        self,
        norm_factor: float = 1.0,
        init_xavier: bool = False,
        regressor_dropout: float = 0.2,
        patch_dropout: float = 0.2,
        transformer_dropout: float = 0.2,
        patch_size: int = 25,  # same default as ModelChall1
        seq_len: int = 8,  # must satisfy seq_len == T // patch_size
        embed_dim: int = 512,
        n_layer: int = 12,
        nhead: int = 8,
        drop_path_rate: float = 0.05,
        drop_path: float = 0.0,
        multi_scale_kernels=(3, 7, 15, 31, 47, 55, 63, 77, 87, 91, 99, 127),
        flash_attn: bool = False,
        full_attn: bool = True,
    ):
        super().__init__()

        # Data/normalization parameters (same behavior as ModelChall1)
        self.regressor_dropout = regressor_dropout
        self.num_channels = 128
        self.seq_len = int(seq_len)
        self.register_buffer("train_loc", torch.as_tensor(1.6).float())
        self.register_buffer("train_spread", torch.as_tensor(0.406).float())
        self.norm_factor = norm_factor
        self.init_xavier = init_xavier

        # Backbone: sequence-first (full-sequence convs -> patchify) -> (B, C, S, D)
        self.patch_size = int(patch_size)
        self.backbone = CBraModSeqFirst(
            patch_size=self.patch_size,
            out_dim=embed_dim,
            d_model=embed_dim,
            dim_feedforward=embed_dim * 4,
            n_layer=n_layer,
            nhead=nhead,
            drop_path_rate=drop_path_rate,
            drop_path=drop_path,
            patch_dropout=patch_dropout,
            transformer_dropout=transformer_dropout,
            multi_scale_kernels=multi_scale_kernels,
            flash_attn=flash_attn,
            full_attn=full_attn,
        )

        # Normalize and expose raw d_model features (no projection)
        self.norm = nn.LayerNorm(embed_dim)
        self.backbone.proj_out = nn.Identity()

        # Flatten MLP head: identical structure to ModelChall1
        self.classifier = ImprovedResidualFlattenMLPHead(
            num_channels=self.num_channels,
            seq_len=self.seq_len,
            embed_dim=embed_dim,
            dropout=self.regressor_dropout,
            init_xavier=False,
            d_proj=1024,
            depth=5,
            mlp_ratio=4.0,
            final_bias=False,
            residual_scale=0.2,  # Scale residual connections
            use_input_skip=True
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, C, T), with fixed T such that T % patch_size == 0 and
           S = T // patch_size == self.seq_len.
        returns: (B, 1)
        """
        # Scale input and enforce 128 channels if a reference channel is present.
        x = x * self.norm_factor
        if x.size(1) == 129:
            x = x[:, :128, :]

        # Sequence-first backbone -> tokens (B, C, S, D)
        feats = self.backbone(x)  # (B, C, S, D)
        feats = self.norm(feats)  # (B, C, S, D)

        # Assert fixed S matches expected seq_len
        B, C, S, D = feats.shape
        assert S == self.seq_len, (
            f"S (got {S}) must equal seq_len ({self.seq_len}). "
            f"Check T and patch_size: expected T = seq_len*patch_size."
        )

        # Flatten MLP head (same as ModelChall1)
        out = self.classifier(feats)  # (B, 1)

        # Un-normalize to target space (keep if your targets were standardized)
        out = out * self.train_spread + self.train_loc

        return {"pred": out}


class ModelChall1_seqfirst_submission(nn.Module):
    """Challenge 1 submission model with preprocessing in forward pass."""

    def __init__(
        self,
        model: nn.Module,
        median: np.ndarray,
        mad: np.ndarray,
        eps: float = 1e-6,
        use_sigma: bool = True,
    ):
        super().__init__()
        self.model = model
        self.robust_norm = RobustChannelNorm(
            median=median, mad=mad, eps=eps, use_sigma=use_sigma
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that applies preprocessing to raw EEG data."""
        x_preprocessed = self.preprocess(x)
        output_dict = self.model(x_preprocessed)
        outputs = output_dict["pred"]
        return outputs

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Apply robust channel normalization using RobustChannelNorm."""
        # Convert to numpy, apply RobustChannelNorm, convert back to torch
        x_np = x.detach().cpu().numpy()
        x_preprocessed_np = self.robust_norm(x_np)
        return torch.from_numpy(x_preprocessed_np).to(x.device, dtype=x.dtype)

    @classmethod
    def load_complete_model(cls, weights_path: str, device: str = "cpu"):
        """Load complete model with preprocessing from saved weights."""
        # checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        checkpoint = torch.hub.load_state_dict_from_url('https://huggingface.co/eeg2025/Sigma-Nova/resolve/main/'+weights_path, map_location=device, weights_only=False)

        # Extract data
        median = checkpoint["median"]
        mad = checkpoint["mad"]
        eps = checkpoint.get("eps", 1e-6)
        use_sigma = checkpoint.get("use_sigma", True)

        # Create base model
        base_model = ModelChall1_seqfirst()
        base_model.load_state_dict(checkpoint["model_state_dict"])

        # Create submission model
        submission_model = cls(base_model, median, mad, eps, use_sigma)
        return submission_model


class ModelChall2_dummy_submission(nn.Module):
    """Dummy Challenge 2 submission model that always outputs 0."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Always return zeros with same batch size as input."""
        batch_size = x.shape[0]
        return torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype) - 0.09


def resolve_path(name="model_file_name"):
    if Path(f"/app/input/res/{name}").exists():
        return f"/app/input/res/{name}"
    elif Path(f"/app/input/{name}").exists():
        return f"/app/input/{name}"
    elif Path(f"{name}").exists():
        return f"{name}"
    elif Path(__file__).parent.joinpath(f"{name}").exists():
        return str(Path(__file__).parent.joinpath(f"{name}"))
    else:
        raise FileNotFoundError(
            f"Could not find {name} in /app/input/res/ or /app/input/ or current directory"
        )


class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE

    def get_model_challenge_1(self):
        model = ModelChall1_seqfirst_submission.load_complete_model(
            "weights_challenge_1.pt", device=self.device
        )
        return model.to(self.device)

    def get_model_challenge_2(self):
        """Load dummy Challenge 2 model."""
        return ModelChall2_dummy_submission().to(self.device)
    
if __name__ == "__main__":
    # Example usage
    s = Submission(SFREQ=100, DEVICE="cpu")
    model_challenge_1 = s.get_model_challenge_1()
    model_challenge_2 = s.get_model_challenge_2()
    print("Models for both challenges are loaded.")