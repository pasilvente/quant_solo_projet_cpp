from equity_alpha.data.factors import build_factors


def test_build_factors_basic():
    """Check that factor construction runs and produces non-empty panels."""
    universe = ["AAPL", "MSFT", "SPY"]
    factors = build_factors(
        tickers=universe,
        start="2018-01-01",
        end="2020-01-01",
        force_download=False,
    )

    # quelques facteurs clés doivent exister
    for name in ["mom_12m_excl_1m", "mom_1m", "vol_20d", "liq_20d_dv"]:
        assert name in factors
        df = factors[name]
        # index non vide, bonnes colonnes
        assert len(df.index) > 0
        assert set(df.columns) == set(universe)
