"""Portfolio construction, final holdout backtest, CPCV backtest, and metrics."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import vectorbt as vbt

from equity_project.src.utils import (
    PROJECT_PATH,
    average_universe_ipc,
    calmar_ratio,
    max_drawdown,
    max_drawdown_duration,
    rolling_pairwise_ipc,
    save_dict,
    sharpe_ratio,
)
from equity_project.src.utils import load_config


def load_feature_columns() -> list[str]:
    """Load the feature column order used during model training."""
    with open(PROJECT_PATH / "models" / "feature_columns.json", "r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload["feature_columns"]


def predict_model_scores(model, X: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Run model inference and return class probabilities indexed by Date/Ticker."""
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.reindex(columns=feature_columns)
    proba = model.predict_proba(X)
    return pd.DataFrame(proba, index=X.index, columns=[0, 1, 2])


def get_rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    """Return last available trading date in every rebalance period."""
    idx = pd.DatetimeIndex(index).sort_values().unique()
    last_dates = idx.to_series(index=idx).groupby(idx.to_period(frequency)).max()
    return pd.DatetimeIndex(last_dates.values)


def generate_weights(
    preds: pd.DataFrame,
    membership: pd.DataFrame,
    top_n: int = 50,
    max_position_weight: float = 0.03,
    max_gross_exposure: float = 0.95,
    rebalance_frequency: str = "W-FRI",
) -> pd.DataFrame:
    """Convert class probabilities into long-only target portfolio weights.

    The score is ``P(outperform) - P(underperform)``. At each rebalance date the
    strategy buys only the top-ranked positive-score S&P500 members. Weights are
    score-proportional, capped by name, and scaled so gross exposure never
    exceeds ``max_gross_exposure``.
    """
    score = (preds[2] - preds[0]).unstack("Ticker").sort_index()
    membership = membership.reindex(index=score.index, columns=score.columns).fillna(False).astype(bool)
    score = score.where(membership)

    rebalance_dates = get_rebalance_dates(score.index, rebalance_frequency)
    score_rebalanced = score.loc[score.index.intersection(rebalance_dates)]

    ranks = score_rebalanced.rank(axis=1, ascending=False, method="first")
    selected = (ranks <= top_n) & (score_rebalanced > 0)
    raw_weights = score_rebalanced.clip(lower=0).where(selected, 0.0).fillna(0.0)

    row_sums = raw_weights.sum(axis=1).replace(0.0, np.nan)
    weights = raw_weights.div(row_sums, axis=0).fillna(0.0)
    weights = weights.clip(upper=max_position_weight)

    gross = weights.sum(axis=1).replace(0.0, np.nan)
    scale = (max_gross_exposure / gross).clip(upper=1.0).fillna(0.0)
    weights = weights.mul(scale, axis=0)

    weights = weights.reindex(score.index).ffill().fillna(0.0)
    return weights


def get_twap_price(data: pd.DataFrame) -> pd.DataFrame:
    """Approximate TWAP execution price by the OHLC arithmetic average."""
    return (data["Open"] + data["High"] + data["Low"] + data["Close"]) / 4.0


def extract_benchmark_close(benchmark: pd.DataFrame) -> pd.Series:
    """Extract a single benchmark close series from a yfinance dataframe."""
    if isinstance(benchmark.columns, pd.MultiIndex):
        close = benchmark.xs("Close", axis=1, level=0)
        if isinstance(close, pd.DataFrame):
            return close.iloc[:, 0].rename("benchmark_close")
        return close.rename("benchmark_close")
    if "Close" in benchmark.columns:
        return benchmark["Close"].rename("benchmark_close")
    if "Adj Close" in benchmark.columns:
        return benchmark["Adj Close"].rename("benchmark_close")
    raise KeyError("Benchmark dataframe must contain Close or Adj Close column")


def run_vectorbt_portfolio(
    close: pd.DataFrame,
    execution_price: pd.DataFrame,
    weights: pd.DataFrame,
    init_cash: float,
    fees: float,
) -> vbt.Portfolio:
    """Create a vectorbt target-percent portfolio with shared cash."""
    common_cols = close.columns.intersection(weights.columns).intersection(execution_price.columns)
    close = close.loc[:, common_cols].dropna(axis=1, how="all")
    execution_price = execution_price.loc[:, close.columns]
    weights = weights.reindex(index=close.index).ffill().reindex(columns=close.columns).fillna(0.0)

    return vbt.Portfolio.from_orders(
        close=close,
        price=execution_price,
        size=weights,
        size_type="targetpercent",
        group_by=True,
        cash_sharing=True,
        freq="1d",
        init_cash=init_cash,
        fees=fees,
    )


def calculate_metrics(
    strategy_value: pd.Series,
    benchmark_close: pd.Series,
    weights: pd.DataFrame,
    cfg: Dict,
) -> Dict[str, float | str | bool]:
    """Calculate strategy, benchmark, and constraint-check metrics."""
    strategy_value = strategy_value.dropna()
    strategy_returns = strategy_value.pct_change().fillna(0.0)

    benchmark_close = benchmark_close.reindex(strategy_value.index).ffill().dropna()
    benchmark_value = benchmark_close / benchmark_close.iloc[0] * float(cfg["init_cash"])
    benchmark_returns = benchmark_value.pct_change().fillna(0.0)

    strategy_total_return = float(strategy_value.iloc[-1] / strategy_value.iloc[0] - 1.0)
    benchmark_total_return = float(benchmark_value.iloc[-1] / benchmark_value.iloc[0] - 1.0)
    strategy_mdd = max_drawdown(strategy_value)
    benchmark_mdd = max_drawdown(benchmark_value)
    mdd_duration = max_drawdown_duration(strategy_value)
    max_gross = float(weights.abs().sum(axis=1).max()) if not weights.empty else np.nan
    min_weight = float(weights.min().min()) if not weights.empty else np.nan

    metrics = {
        "strategy_total_return": strategy_total_return,
        "benchmark_total_return": benchmark_total_return,
        "strategy_sharpe": sharpe_ratio(strategy_returns),
        "benchmark_sharpe": sharpe_ratio(benchmark_returns),
        "strategy_calmar": calmar_ratio(strategy_returns, strategy_value),
        "benchmark_calmar": calmar_ratio(benchmark_returns, benchmark_value),
        "strategy_max_drawdown": strategy_mdd,
        "benchmark_max_drawdown": benchmark_mdd,
        "strategy_max_drawdown_duration_trading_days": mdd_duration,
        "max_gross_exposure": max_gross,
        "min_weight": min_weight,
        "transaction_cost_per_trade": float(cfg["fees"]),
        "execution_model": "TWAP proxy: execution at daily OHLC average",
        "constraint_total_return_positive": bool(strategy_total_return > 0.0),
        "constraint_max_drawdown_lt_20pct": bool(abs(strategy_mdd) < 0.20),
        "constraint_max_drawdown_period_lt_6m": bool(mdd_duration < 126),
        "constraint_no_leverage": bool(max_gross <= 1.000001 and min_weight >= -1e-12),
        "beats_benchmark_by_sharpe": bool(sharpe_ratio(strategy_returns) > sharpe_ratio(benchmark_returns)),
    }
    return metrics


def save_backtest_plots(
    strategy_value: pd.Series,
    benchmark_close: pd.Series,
    ipc: pd.DataFrame,
    output_dir: Path,
    init_cash: float,
    artifact_prefix: str = "backtest",
) -> None:
    """Save PnL, drawdown, and IPC plots as PNG files."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    file_prefix = "" if artifact_prefix == "backtest" else f"{artifact_prefix}_"
    benchmark_close = benchmark_close.reindex(strategy_value.index).ffill().dropna()
    benchmark_value = benchmark_close / benchmark_close.iloc[0] * init_cash

    ax = strategy_value.rename("Strategy").plot(figsize=(12, 6), title="Strategy vs S&P500 benchmark")
    benchmark_value.rename("Benchmark").plot(ax=ax)
    ax.set_ylabel("Portfolio value, USD")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output_dir / f"{file_prefix}pnl.png", dpi=160)
    plt.close(fig)

    drawdown = strategy_value / strategy_value.cummax() - 1.0
    ax = drawdown.rename("Strategy drawdown").plot(figsize=(12, 4), title="Strategy drawdown")
    ax.set_ylabel("Drawdown")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output_dir / f"{file_prefix}drawdown.png", dpi=160)
    plt.close(fig)

    if not ipc.empty:
        ax = ipc.plot(figsize=(12, 5), title="Intra-portfolio correlation dynamics")
        ax.set_ylabel("Rolling correlation")
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(output_dir / f"{file_prefix}ipc.png", dpi=160)
        plt.close(fig)


def run_single_backtest(
    preds: pd.DataFrame,
    price_data: pd.DataFrame,
    benchmark_close: pd.Series,
    membership: pd.DataFrame,
    cfg: Dict,
    artifact_prefix: str = "backtest",
) -> Dict:
    """Run one portfolio backtest and save metrics/artifacts."""
    close = price_data["Close"].dropna(axis=1, how="all")
    twap_price = get_twap_price(price_data).reindex(columns=close.columns)

    weights = generate_weights(
        preds=preds,
        membership=membership,
        top_n=int(cfg.get("top_n", 50)),
        max_position_weight=float(cfg.get("max_position_weight", 0.03)),
        max_gross_exposure=float(cfg.get("max_gross_exposure", 0.95)),
        rebalance_frequency=str(cfg.get("rebalance_frequency", "W-FRI")),
    )
    weights = weights.reindex(index=close.index).ffill().reindex(columns=close.columns).fillna(0.0)

    portfolio = run_vectorbt_portfolio(
        close=close,
        execution_price=twap_price,
        weights=weights,
        init_cash=float(cfg["init_cash"]),
        fees=float(cfg["fees"]),
    )
    strategy_value = portfolio.value()
    if isinstance(strategy_value, pd.DataFrame):
        strategy_value = strategy_value.iloc[:, 0]
    strategy_value.name = "strategy_value"

    asset_returns = close.pct_change()
    strategy_ipc = rolling_pairwise_ipc(asset_returns, weights, window=63)
    universe_ipc = average_universe_ipc(
        asset_returns,
        membership.reindex(index=asset_returns.index, columns=asset_returns.columns).fillna(False),
        window=63,
        evaluation_index=strategy_ipc.dropna().index,
    )
    ipc = pd.concat([strategy_ipc.rename("strategy_ipc"), universe_ipc.rename("sp500_universe_ipc")], axis=1)

    metrics = calculate_metrics(strategy_value, benchmark_close, weights, cfg)

    metrics_dir = PROJECT_PATH / "artifacts" / "metrics"
    plots_dir = PROJECT_PATH / "artifacts" / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    strategy_value.to_frame().to_parquet(metrics_dir / f"{artifact_prefix}_equity_curve.parquet")
    weights.to_parquet(metrics_dir / f"{artifact_prefix}_weights.parquet")
    ipc.to_parquet(metrics_dir / f"{artifact_prefix}_ipc.parquet")
    save_dict(metrics, metrics_dir / f"{artifact_prefix}_metrics.json")
    save_backtest_plots(strategy_value, benchmark_close, ipc, plots_dir, float(cfg["init_cash"]), artifact_prefix=artifact_prefix)
    return metrics


def run_cpcv_backtest_if_available(cfg: Dict) -> None:
    """Run a CPCV portfolio backtest on train/validation OOS predictions."""
    preds_path = PROJECT_PATH / "data" / "processed" / "cpcv_oos_predictions.parquet"
    train_price_path = PROJECT_PATH / "data" / "raw" / "train_data.parquet"
    membership_path = PROJECT_PATH / "data" / "processed" / "membership.parquet"
    benchmark_path = PROJECT_PATH / "data" / "raw" / "benchmark.parquet"
    if not preds_path.exists() or not train_price_path.exists():
        return

    preds = pd.read_parquet(preds_path)
    train_data = pd.read_parquet(train_price_path)
    membership = pd.read_parquet(membership_path)
    benchmark_close = extract_benchmark_close(pd.read_parquet(benchmark_path))

    train_dates = train_data.index
    membership = membership.reindex(train_dates).fillna(False).astype(bool)
    run_single_backtest(
        preds=preds,
        price_data=train_data,
        benchmark_close=benchmark_close,
        membership=membership,
        cfg=cfg,
        artifact_prefix="cpcv_backtest",
    )


def run_backtest() -> None:
    """Run final 2023-2025 holdout backtest and save all required artifacts."""
    os.makedirs(PROJECT_PATH / "artifacts" / "plots", exist_ok=True)
    os.makedirs(PROJECT_PATH / "artifacts" / "metrics", exist_ok=True)

    cfg = load_config()
    X_backtest = pd.read_parquet(PROJECT_PATH / "data" / "processed" / "X_backtest.parquet")
    backtest_data = pd.read_parquet(PROJECT_PATH / "data" / "raw" / "backtest_data.parquet")
    membership = pd.read_parquet(PROJECT_PATH / "data" / "processed" / "membership.parquet")
    benchmark = pd.read_parquet(PROJECT_PATH / "data" / "raw" / "benchmark.parquet")
    benchmark_close = extract_benchmark_close(benchmark)

    model = joblib.load(PROJECT_PATH / "models" / "model.joblib")
    feature_columns = load_feature_columns()
    preds = predict_model_scores(model, X_backtest, feature_columns)
    preds.to_parquet(PROJECT_PATH / "data" / "processed" / "backtest_predictions.parquet")

    membership = membership.reindex(backtest_data.index).fillna(False).astype(bool)
    run_single_backtest(
        preds=preds,
        price_data=backtest_data,
        benchmark_close=benchmark_close,
        membership=membership,
        cfg=cfg,
        artifact_prefix="backtest",
    )

    if bool(cfg.get("run_cpcv", True)):
        run_cpcv_backtest_if_available(cfg)


if __name__ == "__main__":
    run_backtest()
