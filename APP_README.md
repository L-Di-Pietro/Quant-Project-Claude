# Quant Backtester Web App

A sleek web dashboard for testing trading strategies on historical market data.

## Features

✨ **Interactive Interface**
- Select securities (symbols)
- Choose from multiple strategies
- Adjust strategy parameters
- Set custom date ranges and capital allocation

🎯 **Multiple Strategies**
- Moving Average Crossover (classic trend-following)
- Mean Reversion Bollinger Bands (oversold/overbought)

📊 **Real-time Results**
- Equity curve visualization
- Performance metrics (Sharpe, Drawdown, Win Rate, etc.)
- Detailed trade log
- Data auto-download from Yahoo Finance

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Flask app:
```bash
python app.py
```

3. Open your browser and go to:
```
http://localhost:5000
```

## Usage

1. **Download Data** - Click "📥 Download Data" to fetch historical prices for your desired security
2. **Configure** - Select a strategy and adjust parameters (moving averages, Bollinger Band settings, etc.)
3. **Set Period** - Choose start/end dates and initial capital
4. **Run Backtest** - Click "🚀 Run Backtest" to simulate the strategy
5. **Review Results** - Check the equity curve, metrics, and trade log

## API Endpoints

- `GET /api/strategies` - List available strategies and their parameters
- `GET /api/data` - List downloaded data files
- `POST /api/download-data` - Download data for specified symbols
- `POST /api/backtest` - Run a backtest with given parameters
- `GET /api/results/<result_id>` - Retrieve previous results

## Project Structure

```
.
├── app.py                 # Flask server & API
├── index.html            # Web interface
├── backtester/           # Core backtesting engine
│   ├── engine.py         # Main event loop
│   ├── data_handler.py   # Price data management
│   ├── strategy.py       # Strategy base class
│   ├── portfolio.py      # Portfolio management
│   ├── execution_handler.py
│   └── events.py
├── strategies/           # Trading strategies
│   ├── ma_crossover.py
│   └── mean_reversion.py
└── data/                # Historical price data (parquet files)
```

## Performance Metrics

- **Total Return** - Overall profit/loss percentage
- **Sharpe Ratio** - Risk-adjusted return
- **Max Drawdown** - Largest peak-to-trough decline
- **Win Rate** - Percentage of winning trades
- **Profit Factor** - Gross profit / gross loss
- **Number of Trades** - Total trades executed

## Tips

- Start with default parameters to understand the strategy
- Adjust date ranges to test different market regimes
- Compare results across different symbols
- Note: Past performance doesn't guarantee future results!
