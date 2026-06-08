import flwr as fl
import json
import os
import time
import uuid
import torch
import numpy as np


def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    return {"server_round": server_round}


class MetricsLogger:
    def __init__(self, run_id, log_dir="dashboard", clear=False):
        self.run_id = run_id
        self.log_file = os.path.join(log_dir, "metrics.jsonl")
        os.makedirs(log_dir, exist_ok=True)
        # Clear log on startup only if requested
        if clear:
            with open(self.log_file, "w"):
                pass

    def log(
        self,
        round_num: int,
        strategy_name: str,
        loss: float,
        accuracy: float,
        attack_success: float = 0.0,
    ):
        metric = {
            "timestamp": time.time(),
            "run_id": self.run_id,
            "round": round_num,
            "strategy": strategy_name,
            "loss": loss,
            "accuracy": accuracy,
            "attack_success": attack_success,
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(metric) + "\n")


class LoggingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, logger=None, strategy_name="FedAvg", **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logger
        self.strategy_name = strategy_name
        self.prev_loss = None

    def aggregate_fit(self, server_round, results, failures):
        parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if parameters is not None and server_round == 10:
            weights = fl.common.parameters_to_ndarrays(parameters)
            os.makedirs("dashboard", exist_ok=True)
            np.savez(
                f"dashboard/global_model_{self.strategy_name}.npz", *weights
            )
            print(
                f"[{self.strategy_name}] Saved global model to dashboard/global_model_{self.strategy_name}.npz"
            )
        return parameters, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        accuracy = 0.0
        attack_success = 0.0
        if results:
            total_examples = sum([res.num_examples for _, res in results])
            if total_examples > 0:
                accuracy = (
                    sum(
                        [
                            res.metrics.get("accuracy", 0.0) * res.num_examples
                            for _, res in results
                        ]
                    )
                    / total_examples
                )
        if loss is not None and self.logger is not None:
            # Baseline accuracy is ~0.90+, if it drops below, it's an attack success
            attack_success = max(0.0, 0.90 - accuracy)
            self.prev_loss = loss
            self.logger.log(
                server_round,
                self.strategy_name,
                loss,
                accuracy,
                attack_success,
            )
        return loss, metrics


class LoggingFedKrum(fl.server.strategy.Krum):
    def __init__(self, *args, logger=None, strategy_name="FedKrum", **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logger
        self.strategy_name = strategy_name
        self.prev_loss = None

    def aggregate_fit(self, server_round, results, failures):
        parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if parameters is not None and server_round == 10:
            weights = fl.common.parameters_to_ndarrays(parameters)
            os.makedirs("dashboard", exist_ok=True)
            np.savez(
                f"dashboard/global_model_{self.strategy_name}.npz", *weights
            )
            print(
                f"[{self.strategy_name}] Saved global model to dashboard/global_model_{self.strategy_name}.npz"
            )
        return parameters, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        accuracy = 0.0
        attack_success = 0.0
        if results:
            total_examples = sum([res.num_examples for _, res in results])
            if total_examples > 0:
                accuracy = (
                    sum(
                        [
                            res.metrics.get("accuracy", 0.0) * res.num_examples
                            for _, res in results
                        ]
                    )
                    / total_examples
                )
        if loss is not None and self.logger is not None:
            # Baseline accuracy is ~0.90+, if it drops below, it's an attack success
            attack_success = max(0.0, 0.90 - accuracy)
            self.prev_loss = loss
            self.logger.log(
                server_round,
                self.strategy_name,
                loss,
                accuracy,
                attack_success,
            )
        return loss, metrics


class LoggingFedMedian(fl.server.strategy.FedMedian):
    def __init__(
        self, *args, logger=None, strategy_name="FedMedian", **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.logger = logger
        self.strategy_name = strategy_name
        self.prev_loss = None

    def aggregate_fit(self, server_round, results, failures):
        parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if parameters is not None and server_round == 10:
            weights = fl.common.parameters_to_ndarrays(parameters)
            os.makedirs("dashboard", exist_ok=True)
            np.savez(
                f"dashboard/global_model_{self.strategy_name}.npz", *weights
            )
            print(
                f"[{self.strategy_name}] Saved global model to dashboard/global_model_{self.strategy_name}.npz"
            )
        return parameters, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        accuracy = 0.0
        attack_success = 0.0
        if results:
            total_examples = sum([res.num_examples for _, res in results])
            if total_examples > 0:
                accuracy = (
                    sum(
                        [
                            res.metrics.get("accuracy", 0.0) * res.num_examples
                            for _, res in results
                        ]
                    )
                    / total_examples
                )
        if loss is not None and self.logger is not None:
            # Baseline accuracy is ~0.90+, if it drops below, it's an attack success
            attack_success = max(0.0, 0.90 - accuracy)
            self.prev_loss = loss
            self.logger.log(
                server_round,
                self.strategy_name,
                loss,
                accuracy,
                attack_success,
            )
        return loss, metrics


def get_strategy(strategy_name="FedAvg", logger=None, num_malicious_clients=1):
    """
    Returns the appropriate FL strategy.
    Available: FedAvg, FedKrum, FedMedian
    """
    if strategy_name == "FedAvg":
        return LoggingFedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=3,
            min_evaluate_clients=3,
            min_available_clients=3,
            on_fit_config_fn=fit_config,
            logger=logger,
            strategy_name=strategy_name,
        )
    elif strategy_name == "FedKrum":
        return LoggingFedKrum(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=3,
            min_evaluate_clients=3,
            min_available_clients=3,
            num_malicious_clients=num_malicious_clients,
            num_clients_to_keep=2,
            on_fit_config_fn=fit_config,
            logger=logger,
            strategy_name=strategy_name,
        )
    elif strategy_name == "FedMedian":
        return LoggingFedMedian(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=3,
            min_evaluate_clients=3,
            min_available_clients=3,
            on_fit_config_fn=fit_config,
            logger=logger,
            strategy_name=strategy_name,
        )
    # Default
    return LoggingFedAvg(logger=logger, strategy_name="FedAvg")


def start_server(
    strategy_name="FedAvg",
    num_malicious_clients=1,
    clear=False,
    run_id=None,
    seed=None,
    port=8080,
):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    if run_id is None:
        run_id = str(uuid.uuid4())
    logger = MetricsLogger(run_id=run_id, clear=clear)
    strategy = get_strategy(
        strategy_name,
        logger=logger,
        num_malicious_clients=num_malicious_clients,
    )
    print(
        f"Starting Aegis-FL Server with {strategy_name} strategy. Run ID: {run_id}"
    )
    fl.server.start_server(
        server_address=f"0.0.0.0:{port}",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )


if __name__ == "__main__":
    start_server("FedAvg", 1, clear=True)
