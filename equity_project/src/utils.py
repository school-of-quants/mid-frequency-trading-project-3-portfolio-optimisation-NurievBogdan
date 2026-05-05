from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from yaml import safe_load


PROJECT_PATH = Path(__file__).resolve().parents[1]
REPO_PATH = PROJECT_PATH.parent


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Optional path to ``config.yaml``. If omitted, the config in
            the repository root is used.

    Returns:
        A dictionary with configuration parameters.
    """
    path = Path(config_path) if config_path is not None else REPO_PATH / "config.yaml"
    with open(path, "r", encoding="utf-8") as file:
        return safe_load(file)


def save_dict(payload: Dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as a pretty JSON file.

    Args:
        payload: JSON-serializable dictionary.
        path: Destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=4, default=str)


def clean_ticker(ticker: str) -> str:
    """Convert index ticker notation to yfinance-compatible notation.

    Yahoo Finance uses dashes instead of dots for share classes such as BRK.B.
    Extra whitespace is removed as well.
    """
    return str(ticker).strip().replace(".", "-")


def extract_dates(index: pd.Index | pd.MultiIndex) -> pd.DatetimeIndex:
    """Extract the date level from a regular or MultiIndex index.

    The project stores ML samples under a ``(Date, Ticker)`` MultiIndex. Some
    helper functions also work with a plain DatetimeIndex.
    """
    if isinstance(index, pd.MultiIndex):
        if "Date" in index.names:
            values = index.get_level_values("Date")
        else:
            values = index.get_level_values(0)
    else:
        values = index
    return pd.to_datetime(values)


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate geometric annualized return from periodic returns."""
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    total_return = (1.0 + returns).prod() - 1.0
    years = len(returns) / periods_per_year
    if years <= 0:
        return np.nan
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate annualized Sharpe ratio with zero risk-free rate."""
    returns = returns.dropna()
    std = returns.std(ddof=1)
    if returns.empty or std == 0 or np.isnan(std):
        return np.nan
    return float(np.sqrt(periods_per_year) * returns.mean() / std)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Return maximum drawdown as a negative fraction."""
    equity_curve = equity_curve.dropna()
    if equity_curve.empty:
        return np.nan
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def max_drawdown_duration(equity_curve: pd.Series) -> int:
    """Calculate the longest drawdown duration in trading days.

    Duration is the number of consecutive observations below the previous high
    watermark. The project limit is six months, which is roughly 126 trading
    days.
    """
    equity_curve = equity_curve.dropna()
    if equity_curve.empty:
        return 0
    underwater = equity_curve < equity_curve.cummax()
    durations = []
    current = 0
    for flag in underwater:
        if flag:
            current += 1
        else:
            durations.append(current)
            current = 0
    durations.append(current)
    return int(max(durations) if durations else 0)


def calmar_ratio(returns: pd.Series, equity_curve: Optional[pd.Series] = None) -> float:
    """Calculate Calmar ratio as annualized return divided by max drawdown."""
    if equity_curve is None:
        equity_curve = (1.0 + returns.fillna(0.0)).cumprod()
    ann_ret = annualized_return(returns)
    mdd = abs(max_drawdown(equity_curve))
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return float(ann_ret / mdd)


def safe_dataframe_to_parquet(frame: pd.DataFrame, path: str | Path) -> None:
    """Save a dataframe to parquet and create the parent directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow")


def rolling_pairwise_ipc(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    window: int = 63,
    min_assets: int = 3,
    rebalance_only: bool = True,
) -> pd.Series:
    """Calculate rolling intra-portfolio correlation (IPC).

    For each date, the function selects assets with positive portfolio weights,
    estimates a rolling correlation matrix from historical returns, and then
    computes a weighted average of pairwise correlations. The loops are over
    rebalance dates only; the expensive return/correlation operations remain
    vectorized in pandas/numpy.

    Args:
        returns: Daily asset returns with dates as rows and tickers as columns.
        weights: Portfolio weights aligned to the same date/ticker grid.
        window: Rolling lookback window for correlations.
        min_assets: Minimum number of selected assets needed to compute IPC.
        rebalance_only: If True, compute IPC only when weights change.

    Returns:
        A series indexed by date with IPC values.
    """
    common_index = returns.index.intersection(weights.index)
    returns = returns.loc[common_index]
    weights = weights.loc[common_index, returns.columns]

    if rebalance_only:
        changed = weights.diff().abs().sum(axis=1).fillna(1.0) > 1e-12
        evaluation_dates = weights.index[changed]
    else:
        evaluation_dates = weights.index

    ipc_values: Dict[pd.Timestamp, float] = {}
    for dt in evaluation_dates:
        pos = weights.loc[dt]
        selected = pos[pos > 0].index.tolist()
        if len(selected) < min_assets:
            ipc_values[dt] = np.nan
            continue

        hist = returns.loc[:dt, selected].tail(window).dropna(axis=1, how="any")
        if hist.shape[0] < max(20, window // 3) or hist.shape[1] < min_assets:
            ipc_values[dt] = np.nan
            continue

        corr = hist.corr().values
        selected_after_drop = hist.columns
        w = pos.loc[selected_after_drop].values.astype(float)
        w_sum = w.sum()
        if w_sum <= 0:
            ipc_values[dt] = np.nan
            continue
        w = w / w_sum

        pair_weights = np.outer(w, w)
        mask = ~np.eye(len(w), dtype=bool)
        denom = pair_weights[mask].sum()
        ipc_values[dt] = float((corr[mask] * pair_weights[mask]).sum() / denom) if denom > 0 else np.nan

    result = pd.Series(ipc_values, name="ipc")
    return result.reindex(weights.index).ffill()


def average_universe_ipc(
    returns: pd.DataFrame,
    membership: pd.DataFrame,
    window: int = 63,
    min_assets: int = 20,
    evaluation_index: Optional[Iterable[pd.Timestamp]] = None,
) -> pd.Series:
    """Calculate equal-weight IPC for the investable S&P500 universe.

    This benchmark IPC is the average off-diagonal correlation among historical
    S&P500 constituents available on each evaluation date.
    """
    membership = membership.reindex(returns.index).fillna(False).astype(bool)
    eval_dates = pd.Index(evaluation_index if evaluation_index is not None else returns.index)
    ipc_values: Dict[pd.Timestamp, float] = {}

    for dt in eval_dates:
        if dt not in returns.index:
            continue
        selected = membership.loc[dt]
        tickers = selected[selected].index.intersection(returns.columns)
        if len(tickers) < min_assets:
            ipc_values[dt] = np.nan
            continue
        hist = returns.loc[:dt, tickers].tail(window).dropna(axis=1, how="any")
        if hist.shape[0] < max(20, window // 3) or hist.shape[1] < min_assets:
            ipc_values[dt] = np.nan
            continue
        corr = hist.corr().values
        mask = ~np.eye(corr.shape[0], dtype=bool)
        ipc_values[dt] = float(np.nanmean(corr[mask]))

    result = pd.Series(ipc_values, name="sp500_universe_ipc")
    return result.reindex(returns.index).ffill()
