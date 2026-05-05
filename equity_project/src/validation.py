
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterator, Tuple

import numpy as np
import pandas as pd

from equity_project.src.utils import extract_dates


@dataclass
class PurgedKFold:
    """K-fold cross-validation with purging and embargo.

    Args:
        n_splits: Number of chronological validation folds.
        purge_days: Calendar days to remove before each validation block. This
            should be at least as large as the prediction horizon.
        embargo_pct: Fraction of all unique dates to embargo after each
            validation block.
    """

    n_splits: int = 5
    purge_days: int = 10
    embargo_pct: float = 0.01

    def split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield train/test positional indices for each purged fold."""
        sample_dates = extract_dates(X.index)
        unique_dates = pd.DatetimeIndex(sorted(sample_dates.unique()))
        date_folds = np.array_split(unique_dates, self.n_splits)
        embargo_n = max(1, int(len(unique_dates) * self.embargo_pct))

        for test_dates in date_folds:
            if len(test_dates) == 0:
                continue
            test_start = test_dates[0]
            test_end = test_dates[-1]
            test_end_pos = unique_dates.searchsorted(test_end)
            embargo_end_pos = min(len(unique_dates) - 1, test_end_pos + embargo_n)
            embargo_end = unique_dates[embargo_end_pos]
            purge_start = test_start - pd.Timedelta(days=self.purge_days)

            test_mask = sample_dates.isin(test_dates)
            train_mask = ~((sample_dates >= purge_start) & (sample_dates <= embargo_end))
            yield np.flatnonzero(train_mask), np.flatnonzero(test_mask)


@dataclass
class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation splitter.

    CPCV creates many out-of-sample paths by testing on combinations of
    chronological groups. It provides a more robust estimate of strategy quality
    than a single chronological split.

    Args:
        n_groups: Number of chronological groups.
        n_test_groups: Number of groups used as validation/test in each split.
        purge_days: Calendar days to purge around test groups.
        embargo_pct: Fraction of all dates to embargo after test groups.
        max_combinations: Optional cap on the number of combinations to keep
            runtime manageable on Colab/free CPUs.
    """

    n_groups: int = 6
    n_test_groups: int = 2
    purge_days: int = 10
    embargo_pct: float = 0.01
    max_combinations: int | None = None

    def split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield train/test positional indices for CPCV folds."""
        sample_dates = extract_dates(X.index)
        unique_dates = pd.DatetimeIndex(sorted(sample_dates.unique()))
        date_groups = np.array_split(unique_dates, self.n_groups)
        embargo_n = max(1, int(len(unique_dates) * self.embargo_pct))

        combos = list(combinations(range(self.n_groups), self.n_test_groups))
        if self.max_combinations is not None:
            combos = combos[: self.max_combinations]

        for combo in combos:
            test_dates = pd.DatetimeIndex([])
            blocked_mask = np.zeros(len(sample_dates), dtype=bool)
            for group_id in combo:
                group_dates = pd.DatetimeIndex(date_groups[group_id])
                if len(group_dates) == 0:
                    continue
                test_dates = test_dates.union(group_dates)

                test_start = group_dates[0]
                test_end = group_dates[-1]
                test_end_pos = unique_dates.searchsorted(test_end)
                embargo_end_pos = min(len(unique_dates) - 1, test_end_pos + embargo_n)
                embargo_end = unique_dates[embargo_end_pos]
                purge_start = test_start - pd.Timedelta(days=self.purge_days)
                blocked_mask |= (sample_dates >= purge_start) & (sample_dates <= embargo_end)

            test_mask = sample_dates.isin(test_dates)
            train_mask = ~blocked_mask
            train_mask &= ~test_mask
            yield np.flatnonzero(train_mask), np.flatnonzero(test_mask)
