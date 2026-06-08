from src.fl_server.server import get_strategy, LoggingFedAvg, LoggingFedKrum


def test_get_strategy_factory():
    strategy_avg = get_strategy("FedAvg")
    assert isinstance(strategy_avg, LoggingFedAvg)

    strategy_krum = get_strategy("FedKrum")
    assert isinstance(strategy_krum, LoggingFedKrum)
