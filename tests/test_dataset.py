from src.data.dataset_generator import MockIoTDataset
import torch


def test_dataset_normal_only():
    dataset = MockIoTDataset(num_samples=100, only_normal=True)
    assert len(dataset) == 100
    assert torch.sum(torch.tensor(dataset.labels)) == 0.0


def test_dataset_mixed():
    dataset = MockIoTDataset(num_samples=100, only_normal=False)
    assert len(dataset) == 100
    assert torch.sum(torch.tensor(dataset.labels)) > 0.0
