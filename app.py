"""
Flask app for the Quant Project backtester.

Provides a REST API to:
1. Get available strategies
2. Get available data files
3. Run a backtest with specified parameters
4. Return performance results
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path
from threading import Thread

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from backtester.engine import BacktestEngine
from backtester.data_handler import DataHandler
from backtester.strategy import Strategy
from strategies.ma_crossover import MovingAverageCrossoverStrategy
from strategies.mean_reversion import MeanReversionBollingerStrategy
from data.download_data import download_and_clean


app = Flask(__name__)
CORS(app)

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

STRATEGIES = {
    "moving_average_crossover": {
        "name": "Moving Average Crossover",
        "class": MovingAverageCrossoverStrategy,
        "params": {
            "short_window": {"type": "int", "default": 50, "min": 5, "max": 200},
            "long_window": {"type": "int", "default": 200, "min": 50, "max": 500},
        },
    },
    "mean_reversion": {
        "name": "Mean Reversion (Bollinger Bands)",
        "class": MeanReversionBollingerStrategy,
        "params": {
            "lookback": {"type": "int", "default": 20, "min": 5, "max": 100},
            "num_std": {"type": "float", "default": 2.0, "min": 0.5, "max": 4.0},
        },
    },
}

backtest_results = {}


@app.route("/api/strategies", methods=["GET"])
def get_strategies():
    """Return available strategies and their parameters."""
    result = {}
    for key, strategy_info in STRATEGIES.items():
        result[key] = {
            "name": strategy_info["name"],
            "params": strategy_info["params"],
        }
    return jsonify(result)


@app.route("/api/data", methods=["GET"])
def get_available_data():
    """Return list of available data files in the data directory."""
    if not DATA_DIR.exists():
        return jsonify({"files": [], "error": "Data directory not found"})

    files = []
    for file in DATA_DIR.glob("*.parquet"):
        symbol = file.stem
        try:
            df = pd.read_parquet(file)
            files.append(
                {
                    "symbol": symbol,
                    "filename": file.name,
                    "rows": len(df),
                    "start_date": str(df.index[0]).split()[0],
                    "end_date": str(df.index[-1]).split()[0],
                }
            )
        except Exception as e:
            files.append({"symbol": symbol, "error": str(e)})

    return jsonify({"files": files})


@app.route("/api/download-data", methods=["POST"])
def download_data():
    """Download data for specified symbol(s)."""
    data = request.get_json()
    symbols = data.get("symbols", [])
    start = data.get("start", "2015-01-01")
    end = data.get("end", "2024-12-31")

    if not symbols:
        return jsonify({"error": "No symbols provided"}), 400

    results = {}
    for symbol in symbols:
        try:
            path = download_and_clean(
                symbol=symbol.upper(),
                start=start,
                end=end,
                output_dir=str(DATA_DIR),
            )
            results[symbol] = {"status": "success", "path": path}
        except Exception as e:
            results[symbol] = {"status": "error", "error": str(e)}

    return jsonify(results)


@app.route("/api/backtest", methods=["POST"])
def run_backtest():
    """Run a backtest with specified parameters."""
    data = request.get_json()

    try:
        symbol = data.get("symbol", "").upper()
        strategy_key = data.get("strategy")
        start_date = data.get("start_date", "2015-01-01")
        end_date = data.get("end_date", "2024-12-31")
        initial_capital = float(data.get("initial_capital", 100000))
        order_quantity = int(data.get("order_quantity", 100))

        if not symbol:
            return jsonify({"error": "Symbol is required"}), 400
        if strategy_key not in STRATEGIES:
            return jsonify({"error": f"Strategy {strategy_key} not found"}), 400

        # Check if data file exists
        data_file = DATA_DIR / f"{symbol}.parquet"
        if not data_file.exists():
            return (
                jsonify(
                    {"error": f"Data file for {symbol} not found. Please download it first."}
                ),
                404,
            )

        # Load data
        df = pd.read_parquet(data_file)
        df = df.loc[start_date:end_date]

        if df.empty:
            return jsonify({"error": "No data available for the specified date range"}), 400

        # Initialize components
        data_handler = DataHandler(symbol=symbol, data=df)

        strategy_info = STRATEGIES[strategy_key]
        strategy_class = strategy_info["class"]

        # Build strategy parameters
        strategy_params = {"data_handler": data_handler, "symbol": symbol}
        for param_key, param_value in data.get("strategy_params", {}).items():
            if param_key in strategy_info["params"]:
                param_type = strategy_info["params"][param_key]["type"]
                if param_type == "int":
                    strategy_params[param_key] = int(param_value)
                elif param_type == "float":
                    strategy_params[param_key] = float(param_value)
        strategy = strategy_class(**strategy_params)

        # Run backtest
        portfolio_kwargs = {"initial_capital": initial_capital, "order_quantity": order_quantity}
        execution_kwargs = {"commission_type": "fixed", "commission_value": 5.0}

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio_kwargs=portfolio_kwargs,
            execution_kwargs=execution_kwargs,
        )
        engine.run()

        # Collect results
        equity_curve = engine.get_equity_curve()
        trade_log = engine.get_trade_log()
        performance = engine.get_performance_summary()

        # Store results
        result_id = f"{symbol}_{strategy_key}_{datetime.now().timestamp()}"
        backtest_results[result_id] = {
            "symbol": symbol,
            "strategy": strategy_key,
            "start_date": start_date,
            "end_date": end_date,
            "equity_curve": equity_curve.to_dict("index"),
            "trade_log": trade_log.to_dict("records") if not trade_log.empty else [],
            "performance": performance,
            "timestamp": datetime.now().isoformat(),
        }

        # Prepare response
        response = {
            "result_id": result_id,
            "symbol": symbol,
            "strategy": strategy_key,
            "start_date": start_date,
            "end_date": end_date,
            "equity_curve": equity_curve.to_dict("index"),
            "trade_log": trade_log.to_dict("records") if not trade_log.empty else [],
            "performance": performance,
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/results/<result_id>", methods=["GET"])
def get_result(result_id):
    """Retrieve a previous backtest result."""
    if result_id not in backtest_results:
        return jsonify({"error": "Result not found"}), 404

    return jsonify(backtest_results[result_id])


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "data_dir": str(DATA_DIR), "data_exists": DATA_DIR.exists()})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
