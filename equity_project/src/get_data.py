"""Data ingestion, S&P500 membership handling, feature engineering, and labeling"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from equity_project.src.utils import (
    PROJECT_PATH,
    clean_ticker,
    load_config,
    safe_dataframe_to_parquet,
    save_dict,
)

warnings.filterwarnings("ignore")


TRASH_TICKERS = {"DEC", "USBC", "CPWR", "TNB", "APP", "BMC", "SBNY"}


def parse_component_list(value: str) -> list[str]:
    """Parse a comma-separated S&P500 component list from the historical CSV"""
    if pd.isna(value):
        return []
    return [clean_ticker(ticker) for ticker in str(value).split(",") if str(ticker).strip()]


def load_historical_components(path: str | Path) -> pd.Series:
    """Load historical S&P500 constituents from the provided CSV file.

    Args:
        path: Path to ``S&P_500_Historical_Components.csv``.

    Returns:
        A date-indexed series where each value is a list of tickers active in
        the index on that historical snapshot date.
    """
    components = pd.read_csv(path, index_col=0)
    components.index = pd.to_datetime(components.index)
    # The baseline file stores the comma-separated constituents in the first
    # data column. The exact column name is not important.
    constituents = components.iloc[:, 0].apply(parse_component_list)
    constituents = constituents.sort_index()
    return constituents


def get_all_historical_tickers(components: pd.Series) -> list[str]:
    """Return a sorted list of all tickers that ever appeared in the index."""
    tickers = sorted({ticker for ticker_list in components for ticker in ticker_list})
    return [ticker for ticker in tickers if ticker not in TRASH_TICKERS]


def build_membership_matrix(components: pd.Series, trading_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build a daily S&P500 membership matrix aligned to trading dates"""
    tickers = get_all_historical_tickers(components)
    membership_snapshots = pd.DataFrame(False, index=components.index, columns=tickers)
    for dt, ticker_list in components.items():
        valid = [ticker for ticker in ticker_list if ticker in membership_snapshots.columns]
        membership_snapshots.loc[dt, valid] = True

    membership = membership_snapshots.reindex(pd.DatetimeIndex(trading_index), method="ffill")
    membership = membership.fillna(False).astype(bool)
    membership.index.name = "Date"
    return membership


def normalize_yfinance_columns(data: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
    """Normalize yfinance output to a two-level ``(Field, Ticker)`` column index"""
    if not isinstance(data.columns, pd.MultiIndex):
        # yfinance returns a flat frame for a single ticker
        ticker = next(iter(tickers))
        data.columns = pd.MultiIndex.from_product([data.columns, [ticker]])
    if "Adj Close" in data.columns.get_level_values(0):
        data = data.drop(columns="Adj Close", level=0)
    data = data.sort_index(axis=1)
    data.index = pd.to_datetime(data.index)
    data.index.name = "Date"
    return data.astype(float)


def download_ohlcv(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted OHLCV data from Yahoo Finance"""
    data = yf.download(
        tickers=tickers,
        start=start,
        end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        group_by="column",
        auto_adjust=True,
        threads=True,
        progress=True,
    )
    data = normalize_yfinance_columns(data, tickers)
    return data


def download_benchmark(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download a benchmark ETF/index series for comparison with the strategy."""
    benchmark = yf.download(
        ticker,
        start=start,
        end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    benchmark.index = pd.to_datetime(benchmark.index)
    benchmark.index.name = "Date"
    return benchmark.astype(float)


def generate_features(data: pd.DataFrame) -> pd.DataFrame:
    """Generate vectorized price, volatility, volume, and cross-sectional features.

    Args:
        data: Wide OHLCV frame with columns ``(Field, Ticker)``.

    Returns:
        Wide feature matrix with columns ``(Feature, Ticker)``. Every feature is
        shifted by one trading day, so the row for date ``t`` only uses
        information that was available no later than the close of ``t-1``.
    """
    close = data["Close"].replace(0, np.nan)
    open_ = data["Open"].replace(0, np.nan)
    high = data["High"].replace(0, np.nan)
    low = data["Low"].replace(0, np.nan)
    volume = data["Volume"].replace(0, np.nan)
    returns = close.pct_change()

    features: Dict[str, pd.DataFrame] = {}

    for window in (1, 5, 10, 21, 63, 126, 252):
        ret = close.pct_change(window)
        features[f"ret_{window}"] = ret
        features[f"ret_rank_{window}"] = ret.rank(axis=1, pct=True)

    for window in (5, 21, 63, 126, 252):
        ma = close.rolling(window).mean()
        features[f"ma_dev_{window}"] = close / ma - 1.0
        vol = returns.rolling(window).std() * np.sqrt(252)
        features[f"vol_{window}"] = vol
        features[f"vol_rank_{window}"] = vol.rank(axis=1, pct=True)

    features["ma_50_200"] = close.rolling(50).mean() / close.rolling(200).mean() - 1.0
    features["high_low_range"] = (high - low) / close
    features["intraday_ret"] = close / open_ - 1.0
    features["overnight_ret"] = open_ / close.shift(1) - 1.0

    log_volume = np.log1p(volume)
    features["volume_z_21"] = (log_volume - log_volume.rolling(21).mean()) / log_volume.rolling(21).std()
    dollar_volume = close * volume
    features["dollar_volume_rank_21"] = dollar_volume.rolling(21).mean().rank(axis=1, pct=True)

    rolling_high = high.rolling(63).max()
    rolling_low = low.rolling(63).min()
    features["donchian_pos_63"] = (close - rolling_low) / (rolling_high - rolling_low)

    # Cross-sectional de-meaned momentum helps the model learn relative strength
    # instead of only broad market direction.
    for window in (5, 21, 63):
        ret = close.pct_change(window)
        features[f"ret_cs_z_{window}"] = ret.sub(ret.mean(axis=1), axis=0).div(ret.std(axis=1), axis=0)

    X = pd.concat(features, axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.shift(1)
    X = X.iloc[260:]
    X.index.name = "Date"
    return X


def generate_target(close: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    """Create a 3-class cross-sectional target from forward excess returns.

    Class 2 marks the top 30% of stocks by forward return relative to the daily
    universe mean; class 0 marks the bottom 30%; class 1 is neutral. This target
    fits the portfolio-construction problem because the final strategy ranks
    stocks cross-sectionally instead of forecasting absolute market direction.
    """
    forward_return = close.shift(-horizon) / close - 1.0
    excess_return = forward_return.sub(forward_return.mean(axis=1), axis=0)
    ranks = excess_return.rank(axis=1, pct=True)

    target = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    target = target.mask(ranks <= 0.30, 0)
    target = target.mask((ranks > 0.30) & (ranks < 0.70), 1)
    target = target.mask(ranks >= 0.70, 2)
    target[forward_return.isna()] = np.nan
    target.index.name = "Date"
    return target


def stack_feature_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Convert wide ``(Feature, Ticker)`` features to long ``(Date, Ticker)`` rows."""
    X_long = X.stack(level=1)
    X_long.index.names = ["Date", "Ticker"]
    X_long = X_long.sort_index()
    return X_long


def stack_target_frame(y: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide target frame to a long dataframe with one ``target`` column"""
    y_long = y.stack().rename("target").to_frame()
    y_long.index.names = ["Date", "Ticker"]
    return y_long.sort_index()


def get_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download raw OHLCV data, benchmark data, and daily membership matrix."""
    cfg = load_config()
    start = cfg["train_start_date"]
    end = cfg["backtest_end_date"]
    components_path = PROJECT_PATH / "data" / "pony" / "S&P_500_Historical_Components.csv"

    components = load_historical_components(components_path)
    tickers = get_all_historical_tickers(components)
    if cfg.get("debug", False):
        tickers = tickers[: int(cfg.get("debug_n_tickers", 80))]

    data = download_ohlcv(tickers, start, end)
    close = data["Close"]
    available_tickers = close.columns[close.notna().sum() > 100].tolist()
    missing_tickers = sorted(set(tickers) - set(available_tickers))
    data = data.loc[:, pd.IndexSlice[:, available_tickers]]

    membership = build_membership_matrix(components, data.index)
    membership = membership.reindex(columns=available_tickers).fillna(False).astype(bool)

    benchmark = download_benchmark(cfg.get("benchmark_ticker", "SPY"), start, end)

    save_dict({"missing_tickers": missing_tickers}, PROJECT_PATH / "artifacts" / "metrics" / "missing_tickers.json")
    return data, benchmark, membership


def get_data() -> None:
    """Create and persist all raw and processed datasets used by the strategy."""
    cfg = load_config()
    data, benchmark, membership = get_raw_data()

    X = generate_features(data)
    y = generate_target(data["Close"], horizon=int(cfg.get("prediction_horizon", 10)))

    X_long = stack_feature_frame(X)
    y_long = stack_target_frame(y)
    membership_long = membership.stack().rename("is_member").to_frame()
    membership_long.index.names = ["Date", "Ticker"]

    common_index = X_long.index.intersection(y_long.index).intersection(membership_long.index)
    X_long = X_long.loc[common_index]
    y_long = y_long.loc[common_index]
    membership_long = membership_long.loc[common_index]

    # The ML sample exists only when the stock was actually in the S&P500 on
    # that date. This removes both pre-entry and post-exit survivorship bias.
    investable_index = membership_long.index[membership_long["is_member"]]
    X_long = X_long.loc[investable_index]
    y_long = y_long.loc[investable_index]

    X_long = X_long.dropna(how="all")
    y_long = y_long.loc[X_long.index]

    train_start = pd.Timestamp(cfg["train_start_date"])
    train_end = pd.Timestamp(cfg["train_end_date"])
    bt_start = pd.Timestamp(cfg["backtest_start_date"])
    bt_end = pd.Timestamp(cfg["backtest_end_date"])

    dates = X_long.index.get_level_values("Date")
    train_mask = (dates >= train_start) & (dates <= train_end)
    backtest_mask = (dates >= bt_start) & (dates <= bt_end)

    safe_dataframe_to_parquet(data.loc[train_start:train_end], PROJECT_PATH / "data" / "raw" / "train_data.parquet")
    safe_dataframe_to_parquet(data.loc[bt_start:bt_end], PROJECT_PATH / "data" / "raw" / "backtest_data.parquet")
    safe_dataframe_to_parquet(benchmark.loc[train_start:bt_end], PROJECT_PATH / "data" / "raw" / "benchmark.parquet")
    safe_dataframe_to_parquet(membership.loc[train_start:bt_end], PROJECT_PATH / "data" / "processed" / "membership.parquet")

    safe_dataframe_to_parquet(X_long.loc[train_mask], PROJECT_PATH / "data" / "processed" / "X_train.parquet")
    safe_dataframe_to_parquet(y_long.loc[train_mask], PROJECT_PATH / "data" / "processed" / "y_train.parquet")
    safe_dataframe_to_parquet(X_long.loc[backtest_mask], PROJECT_PATH / "data" / "processed" / "X_backtest.parquet")
    safe_dataframe_to_parquet(y_long.loc[backtest_mask], PROJECT_PATH / "data" / "processed" / "y_backtest.parquet")


if __name__ == "__main__":
    get_data()
