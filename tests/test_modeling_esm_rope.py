import torch

from esm.modeling_esm import apply_rotary_emb, precompute_freqs_cis


def test_rotary_complex_fp32_and_real_paths_match_for_bf16_inputs():
    torch.manual_seed(1234)
    xq = (torch.randn(2, 7, 4, 16) * 0.05).to(torch.bfloat16)
    xk = (torch.randn(2, 7, 4, 16) * 0.05).to(torch.bfloat16)
    freqs_cis = precompute_freqs_cis(dim=16, end=7)

    xq_complex, xk_complex = apply_rotary_emb(
        xq, xk, freqs_cis=freqs_cis, rope_use_complex_fp32=True
    )
    xq_real, xk_real = apply_rotary_emb(
        xq, xk, freqs_cis=freqs_cis, rope_use_complex_fp32=False
    )

    assert xq_complex.dtype == torch.bfloat16
    assert xk_complex.dtype == torch.bfloat16
    assert xq_real.dtype == torch.bfloat16
    assert xk_real.dtype == torch.bfloat16

    max_abs_diff = max(
        (xq_complex.float() - xq_real.float()).abs().max().item(),
        (xk_complex.float() - xk_real.float()).abs().max().item(),
    )
    assert max_abs_diff < 1e-3
