"""Small generation helpers shared by the ESM chat application."""

import torch


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Sample from a batch of probability distributions with nucleus sampling."""

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    cumulative = torch.cumsum(probs_sort, dim=-1)
    probs_sort[cumulative - probs_sort > p] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    return torch.gather(probs_idx, -1, torch.multinomial(probs_sort, 1))


def call_model_forward_decode(hparams, model, input_tokens, start_pos=0, bsz=None):
    """Run the single ESM paper path and return vocabulary logits."""

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
