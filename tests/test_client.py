from src.fl_client.client import AegisClient
from src.model.sentinel_mock import GenAISentinel
from src.data.dataset_generator import get_train_dataloader, get_val_dataloader


def test_client_evaluate():
    model = GenAISentinel()
    train_loader = get_train_dataloader(num_samples=20)
    val_loader = get_val_dataloader(num_samples=20)

    client = AegisClient(model, train_loader, val_loader, client_id=0)
    parameters = client.get_parameters({})
    loss, num_examples, metrics = client.evaluate(parameters, {})
    assert num_examples == 20
    assert "accuracy" in metrics
    assert metrics["accuracy"] >= 0.0 and metrics["accuracy"] <= 1.0
