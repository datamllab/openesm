import pytest


transformers = pytest.importorskip("transformers")

from esm.configuration_esm import ESMConfig


def test_esm_config_round_trip(tmp_path):
    config = ESMConfig(model_size="d4", vocab_size=128, context_length=32)
    config.save_pretrained(tmp_path)
    restored = ESMConfig.from_pretrained(tmp_path)

    assert restored.model_type == "esm"
    assert restored.model_size == "d4"
    assert restored.vocab_size == 128
    assert restored.context_length == 32
    assert restored.to_hparams()["mcmc_num_steps"] == 1
