# Backtesting Framework for Quantitative Strategies

> **Project 2.8** — Programming for Finance II (2026)  
> An event-driven backtesting engine for simulating and evaluating algorithmic trading strategies on historical financial data.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Module Reference](#module-reference)
6. [Included Strategies](#included-strategies)
7. [Performance Metrics](#performance-metrics)
8. [Parameter Optimisation](#parameter-optimisation)
9. [Generating a Tear Sheet](#generating-a-tear-sheet)
10. [Project Structure](#project-structure)
11. [Testing](#testing)
12. [Key Pitfalls Addressed](#key-pitfalls-addressed)

---

## Overview

This framework allows users to:

- **Ingest** historical OHLCV data from Yahoo Finance (or any Parquet-formatted source).
- **Simulate** broker execution with realistic commission and slippage models.
- **Implement** custom trading strategies via a simple `Strategy` base class.
- **Track** portfolio state (positions, cash, equity) bar-by-bar.
- **Evaluate** strategy performance through comprehensive risk/return metrics and visual tear sheets.
- **Optimise** strategy parameters using Grid or Random Search with In-Sample / Out-of-Sample validation.

---

## Architecture

The system follows an **Event-Driven Architecture**, which eliminates look-ahead bias by processing data strictly chronologically:

```
┌─────────────┐     MarketEvent      ┌────────────┐
│ DataHandler  │ ──────────────────▶  │  Strategy   │
└─────────────┘                       └──────┬─────┘
                                             │ SignalEvent
                                             ▼
                                      ┌────────────┐
                                      │  Portfolio  │
                                      └──────┬─────┘
                                             │ OrderEvent
                                             ▼
                                    ┌──────────────────┐
                                    │ ExecutionHandler  │
                                    └──────────┬───────┘
                                               │ FillEvent
                                               ▼
                                        ┌────────────┐
                                        │  Portfolio  │
                                        │ (update)    │
                                        └────────────┘
```

All communication flows through a central **event queue** (`queue.Queue`), ensuring strict temporal ordering.

---

## Installation

### Prerequisites

- Python 3.10 or later
- pip

### Setup

```bash
# 1. Navigate to the project directory
cd "Project code"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install the package in development mode (optional)
pip install -e .
```

---

## Quick Start

### Step 1: Download Data

```bash
python data/download_data.py --symbols AAPL MSFT --start 2018-01-01 --end 2024-12-31
```

This creates `data/AAPL.parquet` and `data/MSFT.parquet`.

### Step 2: Run a Backtest

```python
from backtester import BacktestEngine, HistoricParquetDataHandler
from strategies.ma_crossover import MovingAverageCrossoverStrategy

# Load data
data_handler = HistoricParquetDataHandler(
    data_dir="data",
    symbol_list=["AAPL"],
)

# Create strategy
strategy = MovingAverageCrossoverStrategy(
    data_handler=data_handler,
    symbol="AAPL",
    short_window=50,
    long_window=200,
)

# Run the engine
engine = BacktestEngine(
    data_handler=data_handler,
    strategy=strategy,
    portfolio_kwargs={"initial_capital": 100_000, "order_quantity": 100},
    execution_kwargs={
        "commission_type": "fixed",
        "commission_value": 1.0,
        "slippage_type": "spread_pct",
        "slippage_value": 0.05,
    },
)
engine.run()

# View results
print(engine.get_performance_summary())
engine.generate_tearsheet("results/tearsheet.html", strategy_name="MA Crossover 50/200")
```

---

## Module Reference

| Module | Class | Description |
|--------|-------|-------------|
| `events.py` | `MarketEvent`, `SignalEvent`, `OrderEvent`, `FillEvent` | The four event types flowing through the system |
| `data_handler.py` | `HistoricParquetDataHandler` | Loads Parquet data, feeds bars chronologically, prevents look-ahead |
| `strategy.py` | `Strategy` (ABC) | Base class for all strategies |
| `portfolio.py` | `NaivePortfolio` | Tracks positions/cash, converts signals → orders, builds equity curve |
| `execution_handler.py` | `SimulatedExecutionHandler` | Simulates fills with slippage and commission models |
| `performance.py` | `PerformanceManager` | Computes Sharpe, Sortino, drawdowns, trade stats |
| `tearsheet.py` | `TearSheetGenerator` | Generates rich HTML reports with charts and metric tables |
| `optimizer.py` | `GridSearchOptimizer`, `RandomSearchOptimizer` | Parameter search with IS/OOS split |
| `engine.py` | `BacktestEngine` | Central event-loop orchestrator |

---

## Included Strategies

### 1. Moving Average Crossover (`strategies/ma_crossover.py`)

A trend-following strategy:
- **LONG** when the fast MA crosses above the slow MA (golden cross).
- **EXIT** when the fast MA crosses below the slow MA (death cross).
- Parameters: `short_window` (default 50), `long_window` (default 200).

### 2. Mean Reversion — Bollinger Bands (`strategies/mean_reversion.py`)

A mean-reversion strategy:
- **LONG** when the price drops below the lower Bollinger Band (oversold).
- **EXIT** when the price rises above the upper Bollinger Band (overbought).
- Parameters: `lookback` (default 20), `num_std` (default 2.0).

---

## Performance Metrics

| Category | Metric |
|----------|--------|
| **Returns** | Cumulative Return, Annualised Return (CAGR) |
| **Risk-Adjusted** | Sharpe Ratio, Sortino Ratio, Information Ratio |
| **Drawdowns** | Maximum Drawdown (%), Max Drawdown Duration (bars) |
| **Trade Stats** | Win Rate, Profit Factor, Average Win, Average Loss, Total Trades |

---

## Parameter Optimisation

```python
from backtester.optimizer import GridSearchOptimizer
from strategies.ma_crossover import MovingAverageCrossoverStrategy

optimizer = GridSearchOptimizer(
    strategy_cls=MovingAverageCrossoverStrategy,
    param_grid={
        "short_window": [20, 50, 100],
        "long_window": [100, 150, 200],
    },
    data_handler_factory=lambda start_date=None, end_date=None: HistoricParquetDataHandler(
        data_dir="data", symbol_list=["AAPL"],
        start_date=start_date, end_date=end_date,
    ),
    portfolio_kwargs={"initial_capital": 100_000},
    optimise_metric="Sharpe Ratio",
    oos_fraction=0.3,  # 30% out-of-sample
)

results = optimizer.run()
print(results.head())
```

---

## Generating a Tear Sheet

```python
engine.generate_tearsheet("output/tearsheet.html", strategy_name="My Strategy")
```

The tear sheet includes:
- **Equity Curve** — portfolio value over time
- **Underwater Plot** — drawdown depth over time
- **Monthly Returns Heatmap** — colour-coded by month/year
- **Detailed Metrics Table** — all computed statistics

---

## Project Structure

```
Project code/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.py                  # Package configuration
├── backtester/               # Core framework
│   ├── __init__.py
│   ├── events.py             # Event class hierarchy
│   ├── data_handler.py       # No-look-ahead data feeder
│   ├── strategy.py           # Abstract Strategy base
│   ├── portfolio.py          # Position & equity tracking
│   ├── execution_handler.py  # Broker simulation
│   ├── performance.py        # Risk/return metrics
│   ├── tearsheet.py          # HTML report generator
│   ├── optimizer.py          # Grid/Random Search + OOS
│   └── engine.py             # Main event loop
├── data/
│   └── download_data.py      # Yahoo Finance downloader
├── strategies/
│   ├── __init__.py
│   ├── ma_crossover.py       # Moving Average Crossover
│   └── mean_reversion.py     # Bollinger Band Mean Reversion
├── tests/
│   ├── __init__.py
│   ├── test_events.py
│   ├── test_data_handler.py
│   ├── test_portfolio.py
│   ├── test_execution.py
│   └── test_performance.py
└── notebooks/
    └── tutorial.ipynb        # Interactive tutorial
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run a specific test module
pytest tests/test_events.py -v
```

---

## Key Pitfalls Addressed

1. **Look-Ahead Bias** — The `DataHandler` strictly enforces chronological data access. At any point during the simulation, only data up to the current bar is available.

2. **Survivorship Bias** — While historical data from Yahoo Finance inherently suffers from this bias (delisted stocks are excluded), this limitation is explicitly documented. Users can mitigate this by sourcing delisted stock data from premium providers.

3. **Ignoring Slippage and Market Impact** — The `SimulatedExecutionHandler` applies configurable slippage (spread-based or fixed) and commission (fixed or percentage) models to every simulated fill, preventing unrealistically optimistic results.

---

*Built for Programming for Finance II — Spring 2026*
