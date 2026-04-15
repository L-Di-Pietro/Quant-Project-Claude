"""
optimizer.py — Parameter optimisation with overfitting prevention.

Provides Grid Search and Random Search optimisers that split data into
In-Sample and Out-of-Sample periods so that the best parameters can
be validated on unseen data.
"""

from __future__ import annotations

import itertools
import random
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import pandas as pd

from backtester.engine import BacktestEngine


class _BaseOptimizer:
    """
    Shared logic for Grid / Random Search.

    Parameters
    ----------
    strategy_cls : type
        The Strategy sub-class to optimise.
    param_grid : dict[str, list]
        Mapping of parameter names to lists of candidate values.
    data_handler_factory : callable
        A zero-argument callable that returns a *fresh* DataHandler each
        time it is called (important so each run starts from bar 0).
    portfolio_kwargs : dict
        Extra keyword arguments forwarded to the Portfolio constructor.
    execution_kwargs : dict
        Extra keyword arguments forwarded to the ExecutionHandler
        constructor.
    optimise_metric : str
        Name of the metric to maximise (from ``PerformanceManager.summary()``).
    oos_fraction : float
        Fraction of data reserved for Out-of-Sample validation
        (default 0.3 = 30 %).
    """

    def __init__(
        self,
        strategy_cls: Type,
        param_grid: Dict[str, list],
        data_handler_factory: Callable,
        portfolio_kwargs: Optional[Dict[str, Any]] = None,
        execution_kwargs: Optional[Dict[str, Any]] = None,
        optimise_metric: str = "Sharpe Ratio",
        oos_fraction: float = 0.3,
    ) -> None:
        self.strategy_cls = strategy_cls
        self.param_grid = param_grid
        self.data_handler_factory = data_handler_factory
        self.portfolio_kwargs = portfolio_kwargs or {}
        self.execution_kwargs = execution_kwargs or {}
        self.optimise_metric = optimise_metric
        self.oos_fraction = oos_fraction

    # ------------------------------------------------------------------
    # Run a single parameter combination
    # ------------------------------------------------------------------
    def _run_single(
        self,
        params: Dict[str, Any],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute one backtest and return the parameter dict together
        with all performance metrics.
        """
        data_handler = self.data_handler_factory(
            start_date=start_date, end_date=end_date
        )
        strategy = self.strategy_cls(data_handler=data_handler, **params)

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio_kwargs=self.portfolio_kwargs,
            execution_kwargs=self.execution_kwargs,
        )
        engine.run()
        metrics = engine.get_performance_summary()
        return {**params, **metrics}

    # ------------------------------------------------------------------
    # In-Sample / Out-of-Sample split dates
    # ------------------------------------------------------------------
    def _split_dates(self) -> Tuple[str, str, str, str]:
        """
        Determine IS and OOS date boundaries from the full data.
        Returns (is_start, is_end, oos_start, oos_end).
        """
        dh = self.data_handler_factory()
        # Grab one symbol's full date range
        sym = dh.symbol_list[0]
        full_idx = dh._symbol_data[sym].index
        n = len(full_idx)
        split = int(n * (1 - self.oos_fraction))
        is_start = str(full_idx[0].date())
        is_end = str(full_idx[split - 1].date())
        oos_start = str(full_idx[split].date())
        oos_end = str(full_idx[-1].date())
        return is_start, is_end, oos_start, oos_end

    # ------------------------------------------------------------------
    # Public interface (overridden by subclasses)
    # ------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        raise NotImplementedError


class GridSearchOptimizer(_BaseOptimizer):
    """Exhaustive search over every combination in *param_grid*."""

    def _generate_combinations(self) -> List[Dict[str, Any]]:
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def run(self) -> pd.DataFrame:
        """
        Run the grid search.  Returns a DataFrame with one row per
        combination, sorted by the target metric (descending).
        Includes both IS (In-Sample) and OOS (Out-of-Sample) results.
        """
        combos = self._generate_combinations()
        is_start, is_end, oos_start, oos_end = self._split_dates()
        results = []

        for params in combos:
            # In-sample run
            is_metrics = self._run_single(params, is_start, is_end)
            is_metrics = {f"IS_{k}": v for k, v in is_metrics.items()
                          if k not in self.param_grid}
            # Out-of-sample run
            oos_metrics = self._run_single(params, oos_start, oos_end)
            oos_metrics = {f"OOS_{k}": v for k, v in oos_metrics.items()
                           if k not in self.param_grid}
            results.append({**params, **is_metrics, **oos_metrics})

        df = pd.DataFrame(results)
        sort_col = f"IS_{self.optimise_metric}"
        if sort_col in df.columns:
            df.sort_values(sort_col, ascending=False, inplace=True)
        return df.reset_index(drop=True)


class RandomSearchOptimizer(_BaseOptimizer):
    """
    Random sampling of the parameter space.

    Parameters
    ----------
    n_iter : int
        Number of random combinations to evaluate (default 20).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, *args, n_iter: int = 20, seed: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_iter = n_iter
        self.seed = seed

    def _generate_combinations(self) -> List[Dict[str, Any]]:
        rng = random.Random(self.seed)
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combos = []
        for _ in range(self.n_iter):
            combo = {k: rng.choice(v) for k, v in zip(keys, values)}
            combos.append(combo)
        return combos

    def run(self) -> pd.DataFrame:
        """
        Run random search.  Returns a DataFrame sorted by the target
        metric (descending), with IS and OOS columns.
        """
        combos = self._generate_combinations()
        is_start, is_end, oos_start, oos_end = self._split_dates()
        results = []

        for params in combos:
            is_metrics = self._run_single(params, is_start, is_end)
            is_metrics = {f"IS_{k}": v for k, v in is_metrics.items()
                          if k not in self.param_grid}
            oos_metrics = self._run_single(params, oos_start, oos_end)
            oos_metrics = {f"OOS_{k}": v for k, v in oos_metrics.items()
                           if k not in self.param_grid}
            results.append({**params, **is_metrics, **oos_metrics})

        df = pd.DataFrame(results)
        sort_col = f"IS_{self.optimise_metric}"
        if sort_col in df.columns:
            df.sort_values(sort_col, ascending=False, inplace=True)
        return df.reset_index(drop=True)
