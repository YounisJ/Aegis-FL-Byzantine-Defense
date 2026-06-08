import os
import json
from src.fl_server.server import MetricsLogger


def test_metrics_logger(tmp_path):
    logger = MetricsLogger(
        run_id="test-123", log_dir=str(tmp_path), clear=True
    )
    logger.log(
        round_num=1,
        strategy_name="FedAvg",
        loss=0.5,
        accuracy=0.8,
        attack_success=0.1,
    )

    log_file = os.path.join(tmp_path, "metrics.jsonl")
    assert os.path.exists(log_file)
    with open(log_file, "r") as f:
        data = json.loads(f.read().strip())
    assert data["run_id"] == "test-123"
    assert data["round"] == 1
    assert data["accuracy"] == 0.8
