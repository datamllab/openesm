"""The paper-aligned ESM model.

The repository intentionally exposes one training architecture:

* NLP with time embedding;
* one free-embedding MCMC update (K=1);
* a direct unembedding TF head;
* the Muon+AdamW optimizer and linear warmdown schedule are configured by the
  trainer, not by alternate model branches.

Old ablations and alternative heads are not supported. Keeping one model shape
is important because checkpoints are reconstructed from their saved
hyper-parameters before the state dict is loaded.
"""

import math
import os
import pickle
import shutil
import types
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

try:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    from transformers.modeling_outputs import MaskedLMOutput
except ImportError:
    PreTrainedModel = None
    PreTrainedTokenizer = None
    MaskedLMOutput = None

try:
    from .configuration_esm import ESMConfig
except ImportError:
    try:
        from configuration_esm import ESMConfig
    except ImportError:
        ESMConfig = None


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Sample from a batch of probability distributions with nucleus sampling."""

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    cumulative = torch.cumsum(probs_sort, dim=-1)
    probs_sort[cumulative - probs_sort > p] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    return torch.gather(probs_idx, -1, torch.multinomial(probs_sort, 1))


def call_model_forward_decode(hparams, model, input_tokens, start_pos=0, bsz=None):
    """Run ESM on the supplied prefix and return vocabulary logits.

    Callers that pass the full prefix on each decoding iteration must keep
    ``start_pos=0`` because this model does not cache prior key/value states.
    """

    del hparams, bsz
    _, _, pred_hiddens = model.forward(
        input_tokens,
        start_pos=start_pos,
        learning=False,
        return_pred_hiddens=True,
    )
    pred_hidden = pred_hiddens[-1]
    if pred_hidden is None:
        raise RuntimeError("ESM returned no post-update hidden state.")
    return model.tf_head(pred_hidden, model.embeddings(input_tokens))


@dataclass
class ESMModelArgs:
    """Architecture arguments persisted indirectly through checkpoint hparams."""

    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32768
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 64
    max_seq_len: int = 16
    weight_initialization: str = "xavier"
    esm_norm: str = "rms"
    esm_act_func: str = "silu"
    weight_initialization_gain: float = 1.0
    use_sdpa_attention: bool = False
    float_precision: str = "32-true"


ESM_HEAD_DIM = 128


def esm_depth_scaling(depth, aspect_ratio=64):
    base_dim = depth * aspect_ratio
    head_dim = ESM_HEAD_DIM
    embedding_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = embedding_dim // head_dim
    return {
        "num_transformer_blocks": depth,
        "embedding_dim": embedding_dim,
        "multiheaded_attention_heads": num_heads,
    }


model_sizes = {f"d{depth}": esm_depth_scaling(depth) for depth in range(4, 29, 2)}


def init_whole_model_weights(
    model,
    weight_initialization_method,
    nonlinearity="linear",
    weight_initialization_gain=1.0,
):
    """Initialize all linear layers using the checkpoint-compatible policy."""

    def init_weights(module):
        if not isinstance(module, nn.Linear):
            return
        if weight_initialization_method == "he":
            valid_nonlinearities = [
                "linear",
                "relu",
                "leaky_relu",
                "selu",
                "tanh",
            ]
            if nonlinearity not in valid_nonlinearities:
                raise ValueError(
                    f"Unsupported nonlinearity: {nonlinearity}. "
                    f"Must be one of {valid_nonlinearities}"
                )
            nn.init.kaiming_normal_(module.weight, nonlinearity=nonlinearity)
        elif weight_initialization_method == "xavier":
            nn.init.xavier_normal_(module.weight)
        else:
            raise ValueError(
                f"Unknown weight init method: {weight_initialization_method}"
            )
        if weight_initialization_gain != 1.0:
            module.weight.data *= weight_initialization_gain
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)

    model.apply(init_weights)


def setup_esm(hparams):
    """Construct the canonical time-embedding trunk from saved hparams."""

    max_seq_len = hparams.context_length + 1
    if not hasattr(hparams, "vocab_size"):
        raise AttributeError(
            "hparams must define explicit vocab_size before setup_esm()"
        )
    transformer_args = ESMModelArgs(
        dim=hparams.embedding_dim,
        n_layers=hparams.num_transformer_blocks,
        n_heads=hparams.multiheaded_attention_heads,
        max_batch_size=hparams.batch_size_per_device,
        max_seq_len=max_seq_len,
        weight_initialization=hparams.weight_initialization_method,
        esm_norm=hparams.esm_norm,
        ffn_dim_multiplier=hparams.ffn_dim_multiplier,
        esm_act_func=hparams.esm_act_func,
        weight_initialization_gain=hparams.weight_initialization_gain,
        vocab_size=hparams.vocab_size,
        use_sdpa_attention=getattr(hparams, "use_sdpa_attention", False),
        float_precision=getattr(hparams, "float_precision", "32-true"),
    )
    return ESMTimeConcat(
        params=transformer_args,
        gradient_checkpointing=getattr(hparams, "gradient_checkpointing", False),
    )


def has_layer_norm(model):
    return any(isinstance(module, nn.LayerNorm) for _, module in model.named_modules())


def init_wandb_watch(
    wandb_logger,
    model_trainer,
    wandb_watch_log_freq,
    wandb_watch_level="parameters",
):
    """Configure optional W&B watches without importing training utilities."""

    log_mode = wandb_watch_level
    if not has_layer_norm(model_trainer.model):
        wandb_logger.watch(
            model_trainer.model,
            log=log_mode,
            log_freq=wandb_watch_log_freq,
        )
        return

    non_layernorm_container = nn.Module()
    layernorm_container = nn.Module()
    non_ln_modules = {}
    ln_modules = {}

    for name, module in model_trainer.model.named_modules():
        if name == "":
            continue
        safe_name = name.replace(".", "_")
        if isinstance(module, nn.LayerNorm):
            ln_modules[safe_name] = module
        else:
            has_ln_child = any(
                isinstance(child, nn.LayerNorm) for child in module.modules()
            )
            if not has_ln_child:
                non_ln_modules[safe_name] = module

    for name, module in non_ln_modules.items():
        non_layernorm_container.add_module(name, module)
    for name, module in ln_modules.items():
        layernorm_container.add_module(name, module)

    wandb_logger.watch(
        non_layernorm_container,
        log=log_mode,
        log_freq=wandb_watch_log_freq,
    )
    ln_log_mode = "parameters" if log_mode in ("gradients", "all") else log_mode
    wandb_logger.watch(
        layernorm_container,
        log=ln_log_mode,
        log_freq=wandb_watch_log_freq,
    )


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, use_fp32_norm: bool = True):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
            use_fp32_norm (bool): If True, compute normalization in FP32 then cast back.
                Should be False for bf16-true to avoid FP32 graph nodes.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.use_fp32_norm = use_fp32_norm

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        if self.use_fp32_norm:
            output = self._norm(x.float()).type_as(x)
        else:
            output = self._norm(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the cosine and sine frequency tensors for rotary embeddings.

    Returns a tuple of (cos, sin) tensors with shape (end, dim//2), stored as
    float32 but applied via real-valued arithmetic to avoid dtype casting in
    the forward/backward graph.

    Args:
        dim (int): Head dimension.
        end (int): Maximum sequence length.
        theta (float, optional): RoPE base. Defaults to 10000.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (cos, sin) each of shape (end, dim//2).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor of shape (seq, head_dim//2) complex.
        x (torch.Tensor): Target tensor of shape (bs, seq, n_heads, head_dim//2) complex.

    Returns:
        torch.Tensor: Reshaped frequency tensor for broadcast.
    """
    ndim = x.ndim

    assert ndim >= 2, (
        "reshape_for_broadcast expects x to have at least two dimensions "
        f"with a sequence axis at dim=1; got x.ndim={ndim}, shape={tuple(x.shape)}"
    )
    expected_freqs_shape = (x.shape[1], x.shape[-1])
    assert freqs_cis.shape == expected_freqs_shape, (
        "freqs_cis shape must match x sequence and rotary dimensions: "
        f"expected {expected_freqs_shape}, got {tuple(freqs_cis.shape)}"
    )
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    rope_use_complex_fp32: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors.

    Supports two paths:
    - rope_use_complex_fp32=True: complex-float32 arithmetic (for bf16-mixed / fp32 modes).
    - rope_use_complex_fp32=False: real-valued arithmetic in native dtype (for bf16-true mode).

    The two paths implement the same rotary transform. The complex path treats
    each even/odd hidden-dim pair as the real/imaginary part of a complex value
    and multiplies by cos + i sin; the real path expands the same multiplication
    into the equivalent 2D rotation formula. Numeric differences should only
    come from dtype and rounding behavior.

    Args:
        xq (torch.Tensor): Query tensor, shape (..., seq, n_heads, head_dim).
        xk (torch.Tensor): Key tensor, shape (..., seq, n_kv_heads, head_dim).
        freqs_cis (Tuple[torch.Tensor, torch.Tensor]): (cos, sin) each of
            shape (seq, head_dim//2), precomputed by precompute_freqs_cis.
        rope_use_complex_fp32 (bool): If True, use complex-float32 arithmetic.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Rotated xq and xk in original dtype.
    """
    cos, sin = freqs_cis

    if rope_use_complex_fp32:
        freqs_cis_complex = cos + 1j * sin
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis_complex = reshape_for_broadcast(freqs_cis_complex, xq_)
        xq_out = torch.view_as_real(xq_ * freqs_cis_complex).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis_complex).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)
    else:
        cos = cos.to(dtype=xq.dtype)
        sin = sin.to(dtype=xq.dtype)
        xq_r = xq.reshape(*xq.shape[:-1], -1, 2)
        xk_r = xk.reshape(*xk.shape[:-1], -1, 2)
        xq0, xq1 = xq_r[..., 0], xq_r[..., 1]
        xk0, xk1 = xk_r[..., 0], xk_r[..., 1]
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        xq_out = torch.stack(
            [xq0 * cos - xq1 * sin, xq0 * sin + xq1 * cos], dim=-1
        ).flatten(-2)
        xk_out = torch.stack(
            [xk0 * cos - xk1 * sin, xk0 * sin + xk1 * cos], dim=-1
        ).flatten(-2)
        return xq_out, xk_out


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """Multi-head attention module."""

    def __init__(self, layer_id: int, args: ESMModelArgs):
        """
        Initialize the Attention module.

        Args:
            args (ESMModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        init_whole_model_weights(
            self.wq,
            args.weight_initialization,
            weight_initialization_gain=args.weight_initialization_gain,
        )

        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        init_whole_model_weights(
            self.wk,
            args.weight_initialization,
            weight_initialization_gain=args.weight_initialization_gain,
        )

        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        init_whole_model_weights(
            self.wv,
            args.weight_initialization,
            weight_initialization_gain=args.weight_initialization_gain,
        )

        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        init_whole_model_weights(
            self.wo,
            args.weight_initialization,
            weight_initialization_gain=args.weight_initialization_gain,
        )

        self.time_offset = 1
        self.rope_use_complex_fp32 = getattr(args, "float_precision", "") != "bf16-true"
        self.use_sdpa_attention = args.use_sdpa_attention

        self.register_buffer("superdiag_rows", torch.arange(args.max_seq_len - 1))
        self.register_buffer(
            "superdiag_cols",
            torch.arange(self.time_offset, args.max_seq_len + self.time_offset - 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """

        bsz, full_seqlen, _ = x.shape
        original_seqlen = (full_seqlen + 1) // 2
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, full_seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, full_seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, full_seqlen, self.n_local_kv_heads, self.head_dim)

        xq_o = xq[:, :original_seqlen, :, :]
        xk_o = xk[:, :original_seqlen, :, :]
        xv_o = xv[:, :original_seqlen, :, :]

        xq_p = xq[:, original_seqlen:, :, :]
        xk_p = xk[:, original_seqlen:, :, :]
        xv_p = xv[:, original_seqlen:, :, :]

        cos, sin = freqs_cis
        xq_o, xk_o = apply_rotary_emb(
            xq_o,
            xk_o,
            freqs_cis=(cos[:original_seqlen], sin[:original_seqlen]),
            rope_use_complex_fp32=self.rope_use_complex_fp32,
        )

        xq_p, xk_p = apply_rotary_emb(
            xq_p,
            xk_p,
            freqs_cis=(
                cos[self.time_offset : original_seqlen + 1],
                sin[self.time_offset : original_seqlen + 1],
            ),
            rope_use_complex_fp32=self.rope_use_complex_fp32,
        )

        xq_o = xq_o.transpose(1, 2)
        keys_o = xk_o.transpose(1, 2)
        values_o = xv_o.transpose(1, 2)

        if self.use_sdpa_attention:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False,
                enable_cudnn=False,
            ):
                output_o = F.scaled_dot_product_attention(
                    xq_o, keys_o, values_o, is_causal=True
                )
            output_o = (
                output_o.transpose(1, 2).contiguous().view(bsz, original_seqlen, -1)
            )
        else:
            scores_o = torch.matmul(xq_o, keys_o.transpose(2, 3)) / math.sqrt(
                self.head_dim
            )
            if mask is not None:
                o_mask = mask[:-1, :-1]
                scores_o = scores_o + o_mask
            scores_o = F.softmax(scores_o.float(), dim=-1).type_as(xq_o)
            output_o = torch.matmul(scores_o, values_o)
            output_o = (
                output_o.transpose(1, 2).contiguous().view(bsz, original_seqlen, -1)
            )

        xq_p = xq_p.transpose(1, 2)
        keys_p = xk_p.transpose(1, 2)
        values_p = xv_p.transpose(1, 2)
        n_pred = full_seqlen - original_seqlen

        if self.use_sdpa_attention:
            K = original_seqlen
            pred_seqlen = n_pred

            keys_all = torch.cat([keys_o, keys_p], dim=2)
            values_all = torch.cat([values_o, values_p], dim=2)
            orig_rows = torch.arange(pred_seqlen, device=x.device).unsqueeze(1)
            orig_cols = torch.arange(K, device=x.device).unsqueeze(0)
            orig_mask_part = torch.where(
                orig_cols <= orig_rows + (self.time_offset - 1),
                x.new_zeros(1),
                x.new_full((1,), float("-inf")),
            )

            self_mask_part = x.new_full((pred_seqlen, pred_seqlen), float("-inf"))
            self_mask_part.fill_diagonal_(0.0)

            mask_p = torch.cat([orig_mask_part, self_mask_part], dim=1)

            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False,
                enable_cudnn=False,
            ):
                output_p = F.scaled_dot_product_attention(
                    xq_p, keys_all, values_all, attn_mask=mask_p
                )
            output_p = output_p.transpose(1, 2).contiguous().view(bsz, n_pred, -1)
        else:
            scores_p = torch.matmul(xq_p, keys_o.transpose(2, 3)) / math.sqrt(
                self.head_dim
            )

            temp_append = torch.zeros(
                (scores_p.shape[0], scores_p.shape[1], scores_p.shape[2], 1),
                dtype=scores_p.dtype,
                device=scores_p.device,
            )
            scores_p = torch.cat((scores_p, temp_append), dim=-1)

            insertion_superdiagonal = (xq_p * keys_p).sum(dim=3) / math.sqrt(
                self.head_dim
            )
            insertion_superdiagonal = insertion_superdiagonal.to(scores_p.dtype)

            seq_len_minus_1 = scores_p.shape[2]
            if seq_len_minus_1 <= self.superdiag_rows.numel():
                superdiag_rows = self.superdiag_rows[:seq_len_minus_1]
                superdiag_cols = self.superdiag_cols[:seq_len_minus_1]
            else:
                superdiag_rows = torch.arange(seq_len_minus_1, device=scores_p.device)
                superdiag_cols = torch.arange(
                    self.time_offset,
                    self.time_offset + seq_len_minus_1,
                    device=scores_p.device,
                )

            zero_superdiag = torch.zeros_like(
                insertion_superdiagonal, dtype=scores_p.dtype, device=scores_p.device
            )
            diagonal_removal_mask = torch.ones_like(
                scores_p, dtype=scores_p.dtype, device=scores_p.device
            )
            diagonal_removal_mask[:, :, superdiag_rows, superdiag_cols] = zero_superdiag
            scores_p = scores_p * diagonal_removal_mask

            diagonal_addition_mask = torch.zeros_like(
                scores_p, dtype=scores_p.dtype, device=scores_p.device
            )
            diagonal_addition_mask[:, :, superdiag_rows, superdiag_cols] = (
                insertion_superdiagonal
            )
            scores_p = scores_p + diagonal_addition_mask

            if mask is not None:
                p_mask = mask[self.time_offset :, :]
                scores_p = scores_p + p_mask
            if scores_p.dtype != torch.float32:
                scores_p = scores_p.float()
            scores_p = F.softmax(scores_p, dim=-1)
            if scores_p.dtype != xq_p.dtype:
                scores_p = scores_p.to(xq_p.dtype)

            scores_p_superdiagonal = scores_p.diagonal(
                offset=self.time_offset, dim1=2, dim2=3
            )

            scores_p = scores_p * diagonal_removal_mask

            scores_p = scores_p[:, :, :, :-1]
            output_p = torch.matmul(scores_p, values_o)

            next_pred_self_attention = values_p * scores_p_superdiagonal.unsqueeze(
                dim=-1
            )

            output_p = output_p + next_pred_self_attention
            output_p = output_p.transpose(1, 2).contiguous().view(bsz, n_pred, -1)

        output = torch.cat((output_o, output_p), dim=1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim_multiplier: Optional[float],
        weight_initialization: str,
        esm_act_func: str = "silu",
        weight_initialization_gain: float = 1.0,
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = (
            dim if ffn_dim_multiplier is None else int(dim * ffn_dim_multiplier)
        )

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        init_whole_model_weights(
            self.w1,
            weight_initialization,
            weight_initialization_gain=weight_initialization_gain,
        )

        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        init_whole_model_weights(
            self.w2,
            weight_initialization,
            weight_initialization_gain=weight_initialization_gain,
        )

        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        init_whole_model_weights(
            self.w3,
            weight_initialization,
            weight_initialization_gain=weight_initialization_gain,
        )

        self.act_func = {"silu": F.silu, "relu": F.relu, "gelu": F.gelu, "elu": F.elu}[
            esm_act_func
        ]

    def forward(self, x):
        return self.w2(self.act_func(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ESMModelArgs):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ESMModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(layer_id, args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            weight_initialization=args.weight_initialization,
            esm_act_func=args.esm_act_func,
            weight_initialization_gain=args.weight_initialization_gain,
        )
        self.layer_id = layer_id
        use_fp32 = getattr(args, "float_precision", "") != "bf16-true"
        if args.esm_norm != "rms":
            raise ValueError(f"Invalid esm_norm value: {args.esm_norm}")
        self.attention_norm = RMSNorm(
            args.dim, eps=args.norm_eps, use_fp32_norm=use_fp32
        )
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps, use_fp32_norm=use_fp32)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """

        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class ESMTimeConcat(nn.Module):
    def __init__(self, params: ESMModelArgs, gradient_checkpointing=False):
        """
        Initialize a Transformer model.

        Args:
            params (ESMModelArgs): Model configuration parameters.
        Attributes:
            params (ESMModelArgs): Model configuration parameters.
            n_layers (int): Number of layers in the model.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        """
        super().__init__()
        self.params = params
        self.n_layers = params.n_layers
        self.gradient_checkpointing = gradient_checkpointing
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        use_fp32 = getattr(params, "float_precision", "") != "bf16-true"
        if params.esm_norm != "rms":
            raise ValueError(f"Invalid esm_norm value: {params.esm_norm}")
        self.norm = RMSNorm(params.dim, eps=params.norm_eps, use_fp32_norm=use_fp32)

        freqs_cos, freqs_sin = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len + 1
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.final_layer = nn.Linear(params.dim, 1, bias=False)
        init_whole_model_weights(self.final_layer, self.params.weight_initialization)

    def forward(
        self,
        embeddings: torch.Tensor,
        start_pos: int,
        mcmc_step=0,
        real_token_ids: Optional[torch.Tensor] = None,
        predicted_tokens: Optional[torch.Tensor] = None,
        return_pred_hidden: bool = False,
    ):
        """
        Perform a forward pass through the Transformer model.

        Args:
            embeds (torch.Tensor): Embeddings (instead of tokens since is for vision).
            start_pos (int): Starting position for attention caching.
            mcmc_step (int): Unused argument; the trunk is shared across MCMC steps.
            return_pred_hidden (bool): When True, also return the post-norm hidden
                state at candidate positions ([B, S, D]) used by the TF head.

        Returns:
            torch.Tensor: Output energies after applying the Transformer model.
            (energies, pred_hidden) tuple when return_pred_hidden=True.

        """
        _bsz, seqlen = embeddings.shape[:2]
        seqlen = (seqlen + 2) // 2

        required_length = start_pos + seqlen
        if required_length > self.freqs_cos.shape[0]:
            new_cos, new_sin = precompute_freqs_cis(
                self.params.dim // self.params.n_heads, required_length
            )
            self.freqs_cos = new_cos.to(embeddings.device)
            self.freqs_sin = new_sin.to(embeddings.device)

        freqs_cis = (
            self.freqs_cos[start_pos : start_pos + seqlen],
            self.freqs_sin[start_pos : start_pos + seqlen],
        )

        mask = None
        if seqlen > 1:
            mask = torch.empty((seqlen, seqlen), device=embeddings.device).fill_(
                float("-inf")
            )

            mask.triu_(diagonal=1)

            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=embeddings.device), mask]
            ).type_as(embeddings)

            for layer in self.layers:
                if (
                    self.gradient_checkpointing
                    and self.training
                    and torch.is_grad_enabled()
                ):
                    embeddings = checkpoint(
                        layer,
                        embeddings,
                        start_pos,
                        freqs_cis,
                        mask,
                        use_reentrant=False,
                    )
                else:
                    embeddings = layer(embeddings, start_pos, freqs_cis, mask)
            embeddings = self.norm(embeddings)
            split_idx = embeddings.shape[1] // 2
            energies = self.final_layer(embeddings)
            energies = energies[:, split_idx:]
            if return_pred_hidden:
                pred_hidden = embeddings[:, split_idx:]
                return energies, pred_hidden
            return energies


class _HParams(dict):
    """Small attribute-accessible mapping used by the ESM module."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class _ESMModule(nn.Module):
    """Minimal ESM base module; training lifecycle belongs to ``ModelTrainer``."""

    def __init__(self):
        super().__init__()
        self.hparams = _HParams()

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")


PRECISION_TO_MCMC_DTYPE = {
    "bf16-true": torch.bfloat16,
    "bf16-mixed": torch.float32,
    "16-mixed": torch.float32,
    "32-true": torch.float32,
}


class ESM_NLP(_ESMModule):
    """ESM's single paper path for NLP training and validation."""

    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))

        self._validate_paper_path()

        self.tokenizer = (
            self.hparams.tokenizer_obj
            if hasattr(self.hparams, "tokenizer_obj")
            else self.hparams.tokenizer
        )
        self.tokenizer_pad_token_id = None
        if hasattr(self.tokenizer, "get_vocab_size"):
            self.vocab_size = self.tokenizer.get_vocab_size()
        else:
            self.vocab_size = len(self.tokenizer)
        self.hparams.vocab_size = self.vocab_size

        self.alpha = nn.Parameter(
            torch.tensor(float(self.hparams.mcmc_step_size), dtype=torch.float32),
            requires_grad=bool(self.hparams.mcmc_step_size_learnable),
        )

        self.embeddings = nn.Embedding(self.vocab_size, self.hparams.embedding_dim)
        init_whole_model_weights(
            self.embeddings,
            self.hparams.weight_initialization_method,
            weight_initialization_gain=self.hparams.weight_initialization_gain,
        )

        self.vocab_to_embed = nn.Linear(
            self.vocab_size,
            self.hparams.embedding_dim,
            bias=False,
            device=self.device,
        )
        init_whole_model_weights(
            self.vocab_to_embed,
            self.hparams.weight_initialization_method,
            weight_initialization_gain=self.hparams.weight_initialization_gain,
        )

        self.transformer = setup_esm(self.hparams)

        self.use_tf_head = True
        self.use_free_embedding_mcmc = True
        self.tf_head = build_tf_head(self.hparams)
        init_whole_model_weights(
            self.tf_head,
            self.hparams.weight_initialization_method,
            weight_initialization_gain=self.hparams.weight_initialization_gain,
        )
        self.finished_warming_up = False

    def _validate_paper_path(self):
        """Reject removed architectures instead of silently creating a mismatch."""

        required = {
            "modality": "NLP",
            "time_embedding": True,
            "mcmc_num_steps": 1,
            "tf_head_type": "direct_unembed",
            "use_tf_head": True,
            "free_embedding_mcmc": True,
        }
        for name, expected in required.items():
            actual = getattr(self.hparams, name, None)
            if actual != expected:
                raise ValueError(
                    f"ESM only supports the paper path: {name}={expected!r}; "
                    f"received {actual!r}."
                )

        model_size = getattr(self.hparams, "model_size", None)
        if model_size is not None and model_size not in model_sizes:
            valid_sizes = ", ".join(model_sizes)
            raise ValueError(
                f"unsupported ESM model_size={model_size!r}; use one of: {valid_sizes}"
            )

    def _apply(self, fn):
        """Keep the learned MCMC step size in fp32 during dtype conversion."""

        result = super()._apply(fn)
        result.alpha.data = result.alpha.data.to(dtype=torch.float32)
        return result

    @torch.compiler.disable
    def _mcmc_step_excluded(
        self,
        predicted_tokens,
        real_embeddings_input,
        mcmc_step,
        start_pos,
        learning,
        compute_pred_hidden=False,
    ):
        """Run the paper's one gradient update outside a compiled graph."""

        predicted_tokens = predicted_tokens.detach().requires_grad_()
        all_embeddings = torch.cat(
            (real_embeddings_input.detach(), predicted_tokens), dim=1
        )

        transformer = getattr(self, "transformer_eager", self.transformer)
        energy_preds = transformer(
            all_embeddings,
            start_pos=start_pos,
            mcmc_step=mcmc_step,
            real_token_ids=None,
            predicted_tokens=predicted_tokens,
        ).reshape(-1, 1)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            predicted_tokens_grad = torch.autograd.grad(
                energy_preds.float().sum(),
                predicted_tokens,
                create_graph=learning,
            )[0]

        if (
            torch.isnan(predicted_tokens_grad).any()
            or torch.isinf(predicted_tokens_grad).any()
        ):
            raise ValueError("NaN or Inf gradients detected during ESM MCMC.")

        alpha = torch.clamp(self.alpha, min=0.0001).float()
        predicted_tokens = (predicted_tokens - alpha * predicted_tokens_grad).to(
            dtype=all_embeddings.dtype
        )

        pred_hidden = None
        if compute_pred_hidden:
            post_embeddings = torch.cat(
                (real_embeddings_input.detach(), predicted_tokens), dim=1
            )
            _, pred_hidden = transformer(
                post_embeddings,
                start_pos=start_pos,
                mcmc_step=mcmc_step,
                real_token_ids=None,
                predicted_tokens=predicted_tokens,
                return_pred_hidden=True,
            )

        return predicted_tokens, energy_preds.detach(), pred_hidden

    def forward(
        self,
        x,
        start_pos=0,
        learning=True,
        return_raw_logits=False,
        no_randomness=True,
        return_pred_hiddens=False,
    ):
        """Return one updated free-embedding state and its energy."""

        del return_raw_logits, no_randomness
        real_embeddings_input = self.embeddings(x)
        predicted_tokens = self.corrupt_embeddings(real_embeddings_input)

        with torch.set_grad_enabled(True):
            _, energy_preds, pred_hidden = self._mcmc_step_excluded(
                predicted_tokens,
                real_embeddings_input,
                mcmc_step=0,
                start_pos=start_pos,
                learning=learning,
                compute_pred_hidden=return_pred_hiddens,
            )

        if return_pred_hiddens:
            return [None], [energy_preds], [pred_hidden]
        return [None], [energy_preds]

    def forward_loss_wrapper(
        self, x, phase="train", token_bytes=None, global_step=None
    ):
        """Compute the paper's direct-unembedding next-token CE loss."""

        del global_step
        input_ids = x[0].squeeze(dim=0)
        next_token_indices = x[1].squeeze(dim=0).reshape(-1)
        _, predicted_energies, predicted_hiddens = self(
            input_ids,
            learning=phase == "train",
            return_pred_hiddens=True,
        )
        pred_hidden = predicted_hiddens[-1]
        if pred_hidden is None:
            raise RuntimeError("ESM did not return the post-update hidden state.")

        logits = self.tf_head(pred_hidden, self.embeddings(input_ids))
        logits_2d = logits.reshape(-1, self.vocab_size)
        supervised_tokens = (next_token_indices != -1).sum()
        supervised_tokens_denom = supervised_tokens.clamp_min(1)
        per_token_ce = F.cross_entropy(
            logits_2d,
            next_token_indices,
            ignore_index=-1,
            reduction="none",
        )
        loss = per_token_ce.sum() / supervised_tokens_denom
        ppl = torch.exp(loss.detach())

        if token_bytes is not None:
            bpb, bpb_nats, bpb_bytes = calculate_bpb_score(
                next_token_indices,
                per_token_ce.detach(),
                token_bytes,
            )
        else:
            bpb = bpb_nats = bpb_bytes = 0

        initial_energy = predicted_energies[0].mean().detach()
        return {
            "loss": loss,
            "initial_loss": loss.detach(),
            "final_step_loss": loss.detach(),
            "initial_final_pred_energies_gap": torch.zeros_like(initial_energy),
            "perplexity": ppl,
            "bpb": bpb,
            "bpb_nats": bpb_nats,
            "bpb_bytes": bpb_bytes,
            "supervised_tokens": supervised_tokens.detach(),
            "empty_supervision_batch": (supervised_tokens == 0).to(dtype=torch.int64),
        }

    def corrupt_embeddings(self, embeddings):
        """Sample the paper's random D-dimensional free-embedding initial state."""

        float_precision = getattr(self.hparams, "float_precision", "")
        mcmc_dtype = PRECISION_TO_MCMC_DTYPE.get(float_precision, torch.float32)
        noise_scale = self.hparams.gaussian_random_noise_scaling * float(
            getattr(self.hparams, "free_embed_noise_scale", 1.0)
        )
        if self.hparams.denoising_initial_condition != "random_noise":
            raise ValueError(
                "The paper path uses denoising_initial_condition='random_noise'."
            )
        return (
            torch.randn(
                embeddings.shape[0],
                embeddings.shape[1],
                self.hparams.embedding_dim,
                dtype=mcmc_dtype,
                device=self.device,
            )
            * noise_scale
        ).to(dtype=embeddings.dtype)

    def warm_up_finished(self):
        self.finished_warming_up = True


class TFDirectUnembedHead(nn.Module):
    """The single supported TF head: post-update hidden state to vocabulary."""

    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, pred_hidden: torch.Tensor, prev_token_embed: torch.Tensor):
        del prev_token_embed
        return self.proj(pred_hidden)


class StandaloneTokenizer:
    """Load the serialized tokenizer without importing the training package."""

    def __init__(self, encoding):
        self.encoding = encoding
        self.bos_token_id = encoding.encode_single_token("<|bos|>")
        self.eos_token_id = self.bos_token_id
        self.pad_token_id = self.bos_token_id

    @classmethod
    def from_directory(cls, tokenizer_dir):
        path = os.path.join(os.fspath(tokenizer_dir), "tokenizer.pkl")
        with open(path, "rb") as handle:
            return cls(pickle.load(handle))

    def get_vocab_size(self):
        return self.encoding.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode_special(self, token):
        return self.encoding.encode_single_token(token)

    def encode(self, text, prepend=None, append=None, num_threads=8):
        prepend_id = None
        append_id = None
        if prepend is not None:
            prepend_id = (
                prepend if isinstance(prepend, int) else self.encode_special(prepend)
            )
        if append is not None:
            append_id = (
                append if isinstance(append, int) else self.encode_special(append)
            )

        if isinstance(text, str):
            rows = [self.encoding.encode_ordinary(text)]
        elif isinstance(text, list):
            rows = self.encoding.encode_ordinary_batch(text, num_threads=num_threads)
        else:
            raise TypeError(f"expected str or list[str], got {type(text).__name__}")

        for row in rows:
            if prepend_id is not None:
                row.insert(0, prepend_id)
            if append_id is not None:
                row.append(append_id)
        return rows[0] if isinstance(text, str) else rows

    def __call__(self, text, prepend=None, append=None, num_threads=8):
        return self.encode(
            text,
            prepend=prepend,
            append=append,
            num_threads=num_threads,
        )

    def decode(self, token_ids):
        return self.encoding.decode(token_ids)


class ESMTokenizerWrapper:
    """Minimal tokenizer adapter required while reconstructing the model."""

    def __init__(self, tokenizer_obj):
        self.tokenizer = tokenizer_obj
        self.eos_token_id = tokenizer_obj.get_bos_token_id()
        self.bos_token_id = self.eos_token_id
        self.pad_token_id = self.eos_token_id
        self.unk_token_id = 0

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def encode(self, text, **kwargs):
        del kwargs
        return self.tokenizer.encode(text)

    def decode(self, token_ids, **kwargs):
        del kwargs
        return self.tokenizer.decode(token_ids)


class _SerializedObject:
    pass


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("nanochat."):
            return _SerializedObject
        return super().find_class(module, name)


_CHECKPOINT_PICKLE = types.ModuleType("pickle")
_CHECKPOINT_PICKLE.Unpickler = _CheckpointUnpickler
_CHECKPOINT_PICKLE.Pickler = pickle.Pickler
_CHECKPOINT_PICKLE.load = pickle.load
_CHECKPOINT_PICKLE.dump = pickle.dump


def load_checkpoint_file(checkpoint_path, map_location="cpu"):
    """Load a Lightning checkpoint without importing the training repository."""

    return torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
        pickle_module=_CHECKPOINT_PICKLE,
    )


def get_tokenizer(tokenizer_dir):
    """Load ``tokenizer.pkl`` from a model repository directory."""

    if tokenizer_dir is None:
        raise ValueError("tokenizer_dir is required when loading an ESM checkpoint")
    return StandaloneTokenizer.from_directory(tokenizer_dir)


def get_token_bytes(device="cpu", tokenizer_dir=None):
    """Load the per-token UTF-8 byte counts used for BPB evaluation."""

    if tokenizer_dir is None:
        raise ValueError("tokenizer_dir is required when loading token bytes")
    path = os.path.join(os.fspath(tokenizer_dir), "token_bytes.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"token byte table does not exist: {path}")
    return torch.load(path, map_location=device, weights_only=False)


@torch.no_grad()
def calculate_bpb_score(next_token_indices, per_token_loss, token_bytes):
    """Return bits-per-byte, total nats, and total token bytes."""

    if (next_token_indices < 0).any():
        valid = next_token_indices >= 0
        safe_indices = torch.where(
            valid, next_token_indices, torch.zeros_like(next_token_indices)
        )
        num_bytes = torch.where(
            valid,
            token_bytes[safe_indices],
            torch.zeros_like(next_token_indices, dtype=token_bytes.dtype),
        )
    else:
        num_bytes = token_bytes[next_token_indices]

    total_nats = (per_token_loss * (num_bytes > 0)).sum()
    total_bytes = num_bytes.sum().to(torch.int64)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(total_nats, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_bytes, op=torch.distributed.ReduceOp.SUM)
    total_nats_value = total_nats.item()
    total_bytes_value = total_bytes.item()
    bpb = (
        total_nats_value / (math.log(2) * total_bytes_value)
        if total_bytes_value > 0
        else float("inf")
    )
    return bpb, total_nats_value, total_bytes_value


def build_tf_head(hparams) -> nn.Module:
    """Build the paper's direct-unembedding head and reject old variants."""

    head_type = getattr(hparams, "tf_head_type", "direct_unembed")
    if head_type != "direct_unembed":
        raise ValueError(
            f"ESM only supports tf_head_type='direct_unembed'; received {head_type!r}."
        )
    return TFDirectUnembedHead(
        dim=int(hparams.embedding_dim),
        vocab_size=int(hparams.vocab_size),
    )


class ESMInferenceWrapper:
    """Small inference adapter shared by BPB, CORE and generation code."""

    def __init__(self, model, tokenizer, device, max_seq_len: int = 256):
        import torch

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_seq_len = max_seq_len
        self.model_dtype = model.embeddings.weight.dtype
        self.use_autocast = device.type == "cuda" and self.model_dtype in (
            torch.float16,
            torch.bfloat16,
        )
        self._precision_state_logged = False

    def __call__(self, input_ids, targets=None, loss_reduction="mean"):
        import torch
        import torch.nn.functional as F

        autocast = (
            torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.model_dtype,
                enabled=self.use_autocast,
            )
            if self.device.type in {"cuda", "cpu"}
            else nullcontext()
        )
        with torch.no_grad(), autocast:
            if not self._precision_state_logged:
                print(
                    "[checkpoint] "
                    f"model_dtype={self.model_dtype}, autocast={self.use_autocast}"
                )
                self._precision_state_logged = True

            if getattr(self.model, "use_tf_head", False):
                _, _, hidden_states = self.model.forward(
                    input_ids,
                    start_pos=0,
                    learning=False,
                    return_raw_logits=True,
                    return_pred_hiddens=True,
                )
                prev_embed = self.model.embeddings(input_ids)
                distributions = [
                    None if hidden is None else self.model.tf_head(hidden, prev_embed)
                    for hidden in hidden_states
                ]
            else:
                distributions, _ = self.model.forward(
                    input_ids,
                    start_pos=0,
                    learning=False,
                    return_raw_logits=True,
                )

            if not distributions or distributions[-1] is None:
                raise RuntimeError(
                    "ESM returned no final logits; check TF-head/free-embedding settings."
                )
            logits = distributions[-1]
            if targets is None:
                return logits
            return F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )

    def get_device(self):
        return self.device


def resolve_tokenizer_dir(checkpoint_path: str, tokenizer_path: str | None = None):
    """Resolve an explicit or checkpoint-adjacent tokenizer directory."""

    if tokenizer_path:
        tokenizer_dir = Path(tokenizer_path).expanduser().resolve()
        if not tokenizer_dir.is_dir():
            raise FileNotFoundError(
                f"tokenizer directory does not exist: {tokenizer_dir}"
            )
        return tokenizer_dir

    sibling_dir = Path(checkpoint_path).expanduser().resolve().parent / "tokenizer"
    if sibling_dir.is_dir():
        return sibling_dir
    return None


def _resolve_device(torch, device: str | Any):
    if isinstance(device, str):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)
    return device


def _resolve_dtype(torch, dtype: str | Any, device) -> Any:
    if not isinstance(dtype, str):
        return dtype
    if dtype == "float32":
        return torch.float32
    if dtype == "bfloat16":
        return torch.bfloat16
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def _is_huggingface_source(source: str | os.PathLike[str]) -> bool:
    """Return whether a source points to a Transformers model or Hub ID."""

    source_path = Path(os.fspath(source)).expanduser()
    if source_path.is_dir():
        return (source_path / "config.json").is_file()
    if source_path.exists():
        return False
    return not str(source).lower().endswith((".ckpt", ".pt", ".pth"))


def _load_huggingface_checkpoint(
    source: str | os.PathLike[str],
    *,
    device,
    dtype,
    tokenizer_path: str | os.PathLike[str] | None = None,
):
    """Load a Transformers model and expose the legacy inference interface."""

    hf_model_class = globals().get("ESMForMaskedLM")
    hf_tokenizer_class = globals().get("ESMTokenizer")
    if hf_model_class is None or hf_tokenizer_class is None:
        raise ImportError(
            "Loading a Hugging Face ESM model requires the optional dependencies. "
            "Install them with `uv sync --extra hf --extra gpu` or `--extra cpu`."
        )

    model = hf_model_class.from_pretrained(os.fspath(source))
    tokenizer_source = tokenizer_path or source
    tokenizer = hf_tokenizer_class.from_pretrained(os.fspath(tokenizer_source))
    legacy_tokenizer = tokenizer._encoding

    model = model.to(device=device, dtype=dtype)
    model.eval()
    model.esm.eval()

    hparams = model.config.to_hparams()
    hparams["tokenizer_dir"] = os.fspath(tokenizer_source)
    wrapper = ESMInferenceWrapper(
        model.esm,
        ESMTokenizerWrapper(tokenizer_obj=legacy_tokenizer),
        device,
        max_seq_len=int(hparams.get("context_length", 256)),
    )
    print(
        f"[checkpoint] loaded Hugging Face model={hparams.get('model_name', 'esm')} "
        f"size={hparams.get('model_size', 'unknown')} "
        f"context={hparams.get('context_length', 256)}"
    )
    return wrapper, legacy_tokenizer, hparams, device


def _normalize_checkpoint_hparams(hparams, tokenizer):
    hparams = dict(hparams)
    if hparams.get("time_embedding") is None:
        hparams["time_embedding"] = True
    if not hparams.get("esm_norm"):
        hparams["esm_norm"] = hparams.get("ebt_norm") or "rms"
    if not hparams.get("esm_act_func"):
        hparams["esm_act_func"] = hparams.get("ebt_act_func") or "silu"
    if not hparams.get("vocab_size"):
        hparams["vocab_size"] = tokenizer.get_vocab_size()
    hparams.setdefault("float_precision", "32-true")
    hparams.setdefault("weight_initialization_method", "xavier")
    hparams.setdefault("weight_initialization_gain", 1.0)
    hparams.setdefault("gradient_checkpointing", False)
    hparams.setdefault("use_sdpa_attention", False)
    return hparams


def _normalize_legacy_state_dict(state_dict):
    """Convert Lightning state-dict keys to canonical ESM module keys."""

    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            key = key[len("model.") :]
        key = key.replace("._orig_mod.", ".")
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        if key == "langevin_dynamics_noise_std":
            continue
        normalized[key] = value

    eager_prefix = "transformer_eager."
    transformer_prefix = "transformer."
    eager_keys = [key for key in normalized if key.startswith(eager_prefix)]
    if eager_keys:
        missing = [
            key
            for key in eager_keys
            if transformer_prefix + key[len(eager_prefix) :] not in normalized
        ]
        if missing:
            raise RuntimeError(
                "Cannot discard transformer_eager keys without canonical counterparts: "
                + ", ".join(missing[:5])
            )
        normalized = {
            key: value
            for key, value in normalized.items()
            if not key.startswith(eager_prefix)
        }
    return normalized


def load_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    *,
    device: str | Any = "auto",
    dtype: str | Any = "auto",
    tokenizer_path: str | None = None,
):
    """Load a Lightning checkpoint or Transformers model.

    The return value is ``(model, tokenizer, hparams, device)``. ``model`` is
    always an :class:`ESMInferenceWrapper`, so chat and evaluation code can use
    the same interface for all supported model sources. ``checkpoint_path``
    may be a Lightning ``.ckpt`` file, a local Transformers directory, or a
    Hugging Face Hub model ID.

    For a Lightning checkpoint, ``tokenizer_path`` points to the directory
    containing ``tokenizer.pkl`` and ``token_bytes.pt``. If omitted, a
    ``tokenizer`` directory next to the checkpoint is used. For a Transformers
    source, the tokenizer is loaded from the model source unless an explicit
    ``tokenizer_path`` is provided.
    """

    import torch

    resolved_device = _resolve_device(torch, device)
    resolved_dtype = _resolve_dtype(torch, dtype, resolved_device)
    if _is_huggingface_source(checkpoint_path):
        return _load_huggingface_checkpoint(
            checkpoint_path,
            device=resolved_device,
            dtype=resolved_dtype,
            tokenizer_path=tokenizer_path,
        )

    print(
        f"[checkpoint] loading {checkpoint_path} on {resolved_device} ({resolved_dtype})"
    )
    checkpoint = load_checkpoint_file(checkpoint_path, map_location="cpu")
    hparams = dict(checkpoint.get("hyper_parameters", {}))
    tokenizer_dir = resolve_tokenizer_dir(checkpoint_path, tokenizer_path)
    if tokenizer_dir is not None:
        hparams["tokenizer_dir"] = str(tokenizer_dir)
    else:
        configured_tokenizer_dir = hparams.get("tokenizer_dir")
        if configured_tokenizer_dir:
            configured_tokenizer_dir = Path(configured_tokenizer_dir).expanduser()
            if configured_tokenizer_dir.is_dir():
                tokenizer_dir = configured_tokenizer_dir
        if tokenizer_dir is None:
            raise FileNotFoundError(
                "No tokenizer assets found. Pass tokenizer_path or place a "
                "tokenizer directory next to the checkpoint."
            )
    hparams["no_wandb"] = True
    hparams["compile_model"] = False

    tokenizer = get_tokenizer(tokenizer_dir=tokenizer_dir)
    hparams = _normalize_checkpoint_hparams(hparams, tokenizer)
    hparams["tokenizer_obj"] = ESMTokenizerWrapper(tokenizer_obj=tokenizer)
    if resolved_dtype == torch.bfloat16:
        hparams["float_precision"] = "bf16-true"
    elif resolved_dtype == torch.float16:
        hparams["float_precision"] = "16-mixed"
    else:
        hparams["float_precision"] = "32-true"
    model = ESM_NLP(hparams)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state_dict = _normalize_legacy_state_dict(state_dict)

    model.load_state_dict(model_state_dict, strict=True)
    model.eval()
    model = model.to(device=resolved_device, dtype=resolved_dtype)
    model.eval()
    wrapped = ESMInferenceWrapper(
        model,
        ESMTokenizerWrapper(tokenizer_obj=tokenizer),
        resolved_device,
        max_seq_len=int(hparams.get("context_length", 256)),
    )
    print(
        f"[checkpoint] loaded model={hparams.get('model_name', 'esm')} "
        f"size={hparams.get('model_size', 'unknown')} "
        f"context={hparams.get('context_length', 256)}"
    )
    return wrapped, tokenizer, hparams, resolved_device


class _VocabularyOnlyTokenizer:
    """Tokenizer substitute used when the HF model receives integer IDs."""

    def __init__(self, vocab_size: int):
        self.vocab_size = int(vocab_size)

    def get_vocab_size(self):
        return self.vocab_size


if ESMConfig is not None and PreTrainedModel is not None and PreTrainedTokenizer is not None:

    class ESMPreTrainedModel(PreTrainedModel):
        """Shared Transformers base class for ESM models."""

        config_class = ESMConfig
        base_model_prefix = "esm"
        main_input_name = "input_ids"

        def _init_weights(self, module):
            del module


    class ESMForMaskedLM(ESMPreTrainedModel):
        """Hugging Face wrapper exposing ESM token logits."""

        def __init__(self, config):
            super().__init__(config)
            hparams = config.to_hparams()
            hparams["tokenizer_obj"] = _VocabularyOnlyTokenizer(config.vocab_size)
            self.esm = ESM_NLP(hparams)
            self.post_init()

        def get_input_embeddings(self):
            return self.esm.embeddings

        def set_input_embeddings(self, value):
            self.esm.embeddings = value

        def get_output_embeddings(self):
            return self.esm.tf_head.proj

        def set_output_embeddings(self, value):
            self.esm.tf_head.proj = value

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            output_hidden_states=None,
            output_attentions=None,
            return_dict=None,
            **kwargs,
        ):
            del attention_mask, output_attentions, kwargs
            if input_ids is None:
                raise ValueError("input_ids must be provided")
            if input_ids.ndim != 2:
                raise ValueError(
                    f"input_ids must have shape [batch, sequence], got {input_ids.shape}"
                )
            if return_dict is None:
                return_dict = self.config.use_return_dict

            parameter = next(self.parameters())
            use_autocast = parameter.device.type == "cuda" and parameter.dtype in {
                torch.float16,
                torch.bfloat16,
            }
            autocast_context = (
                torch.autocast(
                    device_type=parameter.device.type,
                    dtype=parameter.dtype,
                )
                if use_autocast
                else nullcontext()
            )
            with autocast_context:
                _, _, hidden_states = self.esm(
                    input_ids,
                    learning=self.training,
                    return_pred_hiddens=True,
                )
                hidden = hidden_states[-1]
                if hidden is None:
                    raise RuntimeError("ESM did not return the post-update hidden state")
                logits = self.esm.tf_head(hidden, self.esm.embeddings(input_ids))

                loss = None
                if labels is not None:
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                        ignore_index=-100,
                    )

            if not return_dict:
                output = (logits,)
                if output_hidden_states:
                    output = output + ((hidden,),)
                return ((loss,) + output) if loss is not None else output

            return MaskedLMOutput(
                loss=loss,
                logits=logits,
                hidden_states=(hidden,) if output_hidden_states else None,
                attentions=None,
            )

        @classmethod
        def from_legacy_checkpoint(cls, checkpoint_path, tokenizer_path=None):
            """Load a Lightning checkpoint into the Transformers wrapper."""

            checkpoint = load_checkpoint_file(checkpoint_path, map_location="cpu")
            hparams = dict(checkpoint.get("hyper_parameters", {}))
            tokenizer_dir = resolve_tokenizer_dir(checkpoint_path, tokenizer_path)
            if tokenizer_dir is None:
                raise FileNotFoundError(
                    "No tokenizer directory found next to the checkpoint; "
                    "pass tokenizer_path explicitly."
                )
            tokenizer = get_tokenizer(tokenizer_dir)
            hparams = _normalize_checkpoint_hparams(hparams, tokenizer)
            config = ESMConfig.from_legacy_hparams(
                hparams, vocab_size=tokenizer.get_vocab_size()
            )
            model = cls(config)
            canonical_state = _normalize_legacy_state_dict(
                checkpoint.get("state_dict", checkpoint)
            )
            hf_state = {f"esm.{key}": value for key, value in canonical_state.items()}
            incompatible = model.load_state_dict(hf_state, strict=False)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    "Legacy checkpoint does not match the ESM Transformers wrapper: "
                    f"missing={incompatible.missing_keys[:5]}, "
                    f"unexpected={incompatible.unexpected_keys[:5]}"
                )
            return model, tokenizer


    class ESMTokenizer(PreTrainedTokenizer):
        """Transformers tokenizer backed by ESM's serialized RustBPE data."""

        model_input_names = ["input_ids", "attention_mask"]
        vocab_files_names = {"tokenizer_file": "tokenizer.pkl"}

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
            model_path = Path(str(pretrained_model_name_or_path)).expanduser()
            if model_path.is_dir() and "tokenizer_file" not in kwargs:
                kwargs["tokenizer_file"] = str(model_path / "tokenizer.pkl")
            return super().from_pretrained(
                pretrained_model_name_or_path, *init_inputs, **kwargs
            )

        def __init__(self, tokenizer_file=None, **kwargs):
            if tokenizer_file is None:
                tokenizer_file = "tokenizer.pkl"
            tokenizer_file = Path(tokenizer_file).expanduser().resolve()
            self._tokenizer_file = tokenizer_file
            self._encoding = StandaloneTokenizer.from_directory(tokenizer_file.parent)
            self._special_ids = {
                "<|bos|>": self._encoding.bos_token_id,
                "<|eos|>": self._encoding.eos_token_id,
                "<|pad|>": self._encoding.pad_token_id,
                "<|unk|>": 0,
            }
            special_tokens = {
                "bos_token": kwargs.pop("bos_token", "<|bos|>"),
                "eos_token": kwargs.pop("eos_token", "<|eos|>"),
                "pad_token": kwargs.pop("pad_token", "<|pad|>"),
                "unk_token": kwargs.pop("unk_token", "<|unk|>"),
            }
            super().__init__(
                **special_tokens,
                **kwargs,
            )

        @property
        def vocab_size(self):
            return self._encoding.get_vocab_size()

        def get_vocab(self):
            vocabulary = {str(index): index for index in range(self.vocab_size)}
            vocabulary.update(self._special_ids)
            return vocabulary

        def _tokenize(self, text, **kwargs):
            del kwargs
            return [str(index) for index in self._encoding.encode(text)]

        def _convert_token_to_id(self, token):
            if token in self._special_ids:
                return self._special_ids[token]
            return int(token)

        def _convert_id_to_token(self, index):
            return str(index)

        def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
            del token_ids_1
            return [self.bos_token_id] + list(token_ids_0)

        def _decode(
            self,
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
            **kwargs,
        ):
            del clean_up_tokenization_spaces, kwargs
            token_ids = [int(token_id) for token_id in token_ids]
            if skip_special_tokens:
                token_ids = [
                    token_id
                    for token_id in token_ids
                    if token_id not in self._special_ids.values()
                ]
            return self._encoding.decode(token_ids)

        def save_vocabulary(self, save_directory, filename_prefix=None):
            del filename_prefix
            save_directory = Path(save_directory)
            save_directory.mkdir(parents=True, exist_ok=True)
            target = save_directory / "tokenizer.pkl"
            shutil.copyfile(self._tokenizer_file, target)
            return (str(target),)

else:
    ESMConfig = None
    ESMPreTrainedModel = None
    ESMForMaskedLM = None
    ESMTokenizer = None


__all__ = [
    "ESMConfig",
    "ESMPreTrainedModel",
    "ESMForMaskedLM",
    "ESMTokenizer",
    "ESMModelArgs",
    "ESMInferenceWrapper",
    "load_checkpoint",
    "resolve_tokenizer_dir",
    "load_checkpoint_file",
    "get_tokenizer",
    "get_token_bytes",
    "calculate_bpb_score",
    "StandaloneTokenizer",
    "ESMTokenizerWrapper",
    "ESM_NLP",
    "ESMTimeConcat",
    "PRECISION_TO_MCMC_DTYPE",
    "Attention",
    "FeedForward",
    "TransformerBlock",
    "RMSNorm",
    "precompute_freqs_cis",
    "reshape_for_broadcast",
    "apply_rotary_emb",
    "repeat_kv",
    "build_tf_head",
    "TFDirectUnembedHead",
    "init_wandb_watch",
    "init_whole_model_weights",
    "setup_esm",
]
