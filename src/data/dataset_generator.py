import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MockIoTDataset(Dataset):
    """
    Mock dataset simulating IIoT (Industrial IoT) telemetry and network traffic.
    Features are scaled between 0 and 1.
    We create structured clusters to represent 'normal' traffic,
    so the Autoencoder has a pattern to learn.
    """

    def __init__(self, num_samples=1000, num_features=41, only_normal=False):
        super().__init__()
        self.num_samples = num_samples
        self.num_features = num_features

        if only_normal:
            num_normal = num_samples
            num_anomaly = 0
        else:
            # 80% normal traffic, 20% anomaly traffic
            num_normal = int(num_samples * 0.8)
            num_anomaly = num_samples - num_normal

        t = np.linspace(0, 2 * np.pi, num_features)

        # Normal traffic: Structured sine waves with fixed frequencies but random phase
        # This forms a simple 2D manifold that the Autoencoder can easily learn.
        normal_data = np.zeros((num_normal, num_features))
        for i in range(num_normal):
            phase1, phase2 = np.random.uniform(0, 2 * np.pi, 2)
            wave = 0.25 * np.sin(2.0 * t + phase1) + 0.25 * np.sin(
                3.0 * t + phase2
            )
            normal_data[i] = (
                0.5 + wave + np.random.normal(0, 0.02, num_features)
            )

        # Anomaly traffic: Unstructured random noise with the same mean (0.5)
        # This ensures the model cannot simply predict a constant to achieve separation.
        anomaly_data = np.random.uniform(
            low=0.1, high=0.9, size=(num_anomaly, num_features)
        )

        self.data = np.vstack([normal_data, anomaly_data])
        # Clip to [0, 1]
        self.data = np.clip(self.data, 0.0, 1.0).astype(np.float32)

        # Labels: 0 for benign, 1 for anomaly
        self.labels = np.hstack(
            [np.zeros(num_normal), np.ones(num_anomaly)]
        ).astype(np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])


def get_dataloader(
    num_samples=1000, num_features=41, batch_size=32, shuffle=True
):
    dataset = MockIoTDataset(num_samples, num_features)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_train_dataloader(
    num_samples=200, num_features=41, batch_size=32, shuffle=True
):
    dataset = MockIoTDataset(num_samples, num_features, only_normal=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_val_dataloader(
    num_samples=50, num_features=41, batch_size=32, shuffle=False
):
    dataset = MockIoTDataset(num_samples, num_features, only_normal=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    loader = get_dataloader(num_samples=100)
    for x, y in loader:
        print(f"Batch shape: {x.shape}, Labels shape: {y.shape}")
        break
