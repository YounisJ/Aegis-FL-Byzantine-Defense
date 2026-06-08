import torch
from src.model.sentinel_mock import EdgeAutoencoder, GenAISentinel
from src.adversary.attacks import SignFlipAttack


def test_autoencoder_shapes():
    model = EdgeAutoencoder(input_dim=41, latent_dim=16)
    x = torch.rand(5, 41)
    out = model(x)
    assert out.shape == (5, 41), "Output shape should match input shape"


def test_sign_flip_attack():
    model = EdgeAutoencoder(input_dim=10)
    original_weight = model.encoder[0].weight.data.clone()

    attack = SignFlipAttack(factor=-5.0)
    attack.apply(model)

    poisoned_weight = model.encoder[0].weight.data
    assert torch.allclose(
        poisoned_weight, original_weight * -5.0
    ), "Attack failed to flip and scale weights properly"


def test_sentinel_process_packet():
    sentinel = GenAISentinel(input_dim=41)
    dummy_input = torch.rand(1, 41)
    is_anomaly, explanation = sentinel.process_packet(dummy_input)
    assert isinstance(is_anomaly, bool)
    assert isinstance(explanation, str)
