import flwr as fl
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")

from src.model.sentinel_mock import GenAISentinel
from src.data.dataset_generator import get_train_dataloader, get_val_dataloader
from src.adversary.attacks import SignFlipAttack


class AegisClient(fl.client.NumPyClient):
    def __init__(
        self, sentinel, train_loader, val_loader, client_id, is_malicious=False
    ):
        self.sentinel = sentinel
        self.model = sentinel.autoencoder
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.client_id = client_id
        self.is_malicious = is_malicious
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def get_parameters(self, config):
        return [
            val.cpu().numpy() for _, val in self.model.state_dict().items()
        ]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        self.model.train()
        # Train for 3 local epochs
        for epoch in range(3):
            for x, _ in self.train_loader:
                x = x.to(self.device)
                optimizer.zero_grad()
                reconstructed = self.model(x)
                loss = criterion(reconstructed, x)
                loss.backward()
                optimizer.step()

        if self.is_malicious:
            logging.info(
                f"[Client {self.client_id}] Malicious! Injecting poisoned weights."
            )
            attack = SignFlipAttack(factor=-10.0)
            attack.apply(self.model)

        return (
            self.get_parameters(config={}),
            len(self.train_loader.dataset),
            {},
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        criterion = nn.MSELoss()
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        self.logged_anomaly = False

        with torch.no_grad():
            # First pass: calculate dynamic threshold using only normal data
            normal_mses = []
            for x, y in self.val_loader:
                x = x.to(self.device)
                reconstructed = self.model(x)
                mse_per_sample = torch.mean((reconstructed - x) ** 2, dim=1)
                normal_mses.extend(mse_per_sample[y == 0].cpu().tolist())

            if len(normal_mses) > 0:
                mean_mse = np.mean(normal_mses)
                std_mse = np.std(normal_mses)
                dynamic_threshold = mean_mse + 2 * std_mse
            else:
                dynamic_threshold = 0.05  # fallback

            # Second pass: evaluate
            for x, y in self.val_loader:
                x = x.to(self.device)
                reconstructed = self.model(x)
                loss = criterion(reconstructed, x)
                total_loss += loss.item() * x.size(0)

                # Calculate MSE per sample
                mse_per_sample = torch.mean((reconstructed - x) ** 2, dim=1)

                # Dynamic threshold
                predictions = (mse_per_sample > dynamic_threshold).float()
                correct_predictions += (predictions.cpu() == y).sum().item()

                # GenAI Integration: Trigger LLM explanation on first anomaly detected
                if predictions.sum() > 0 and not self.logged_anomaly:
                    idx = torch.argmax(predictions).item()
                    is_anom, explanation = self.sentinel.process_packet(
                        x[idx : idx + 1], dynamic_threshold
                    )
                    logging.info(
                        f"[Client {self.client_id}] GenAI Sentinel triggered: {explanation}"
                    )
                    self.logged_anomaly = True

        avg_loss = total_loss / len(self.val_loader.dataset)
        accuracy = correct_predictions / len(self.val_loader.dataset)
        logging.info(
            f"[Client {self.client_id}] Evaluation MSE Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
        )
        return (
            float(avg_loss),
            len(self.val_loader.dataset),
            {"mse": float(avg_loss), "accuracy": float(accuracy)},
        )


def start_client(client_id=0, is_malicious=False, port=8080):
    sentinel = GenAISentinel()
    train_loader = get_train_dataloader(num_samples=200)
    val_loader = get_val_dataloader(num_samples=50)

    client = AegisClient(
        sentinel, train_loader, val_loader, client_id, is_malicious
    )
    logging.info(
        f"Starting Aegis-FL Client {client_id} (Malicious: {is_malicious}) on port {port}"
    )
    fl.client.start_numpy_client(
        server_address=f"127.0.0.1:{port}",
        client=client,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=0, help="Client ID")
    parser.add_argument(
        "--malicious",
        action="store_true",
        help="Set flag to make client malicious",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    start_client(args.id, args.malicious, args.port)
