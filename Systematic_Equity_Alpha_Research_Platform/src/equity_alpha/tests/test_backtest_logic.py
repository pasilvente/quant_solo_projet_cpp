from equity_alpha.backtest.engine import BacktestConfig, run_backtest


def test_backtest_shapes_and_stats():
    """Run a tiny backtest and check basic properties."""
    universe = ["AAPL", "MSFT", "SPY"]

    cfg = BacktestConfig(
        tickers=universe,
        start="2018-01-01",
        end="2020-01-01",
        horizon=1,
        long_quantile=0.4,
        short_quantile=0.4,
        gross_target=1.0,
    )

    res = run_backtest(cfg)

    scores = res["scores"]
    weights = res["weights"]
    port_rets = res["port_rets"]
    stats = res["stats"]

    # dimensions de base
    assert set(scores.columns) == set(universe)
    assert set(weights.columns) == set(universe)
    assert len(port_rets) > 0

    # stats présentes
    for key in ["ann_return", "ann_vol", "sharpe", "max_drawdown", "turnover"]:
        assert key in stats

    # poids raisonnablement neutres en moyenne
    net = weights.sum(axis=1).dropna()
    assert abs(net.mean()) < 1e-2  # ~neutre
