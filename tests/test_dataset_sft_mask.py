import torch

from esm.dataset_sft import _build_sft_inputs_and_targets


def test_sft_mask_correctness_keeps_only_assistant_targets():
    batch_tensor = torch.tensor(
        [
            [101, 11, 12, 201, 21, 22, 102],
            [101, 31, 301, 41, 42, 43, 102],
        ],
        dtype=torch.long,
    )
    mask_tensor = torch.tensor(
        [
            [False, False, False, False, True, True, False],
            [False, False, False, True, True, True, False],
        ],
        dtype=torch.bool,
    )

    inputs, targets = _build_sft_inputs_and_targets(
        batch_tensor, mask_tensor, device="cpu", use_cuda=False
    )

    expected_inputs = torch.tensor(
        [
            [101, 11, 12, 201, 21, 22],
            [101, 31, 301, 41, 42, 43],
        ],
        dtype=torch.long,
    )
    expected_targets = torch.tensor(
        [
            [-1, -1, -1, 21, 22, -1],
            [-1, -1, 41, 42, 43, -1],
        ],
        dtype=torch.long,
    )

    assert torch.equal(inputs, expected_inputs)
    assert torch.equal(targets, expected_targets)

    shifted_mask = mask_tensor[:, 1:]
    assert torch.equal(targets[shifted_mask], batch_tensor[:, 1:][shifted_mask])
    assert torch.all(targets[~shifted_mask] == -1)
