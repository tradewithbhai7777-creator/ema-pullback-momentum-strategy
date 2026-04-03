# 9 & 20 EMA Pullback with Wick Rejection Strategy

## Overview
A comprehensive backtesting implementation for the 9 & 20 EMA Pullback with Wick Rejection trading strategy designed for 1-hour cryptocurrency markets.

## Strategy Description

### Market & Timeframe
- **Market**: Cryptocurrency (BTCUSDT, ETHUSDT, SOLUSDT, etc.)
- **Timeframe**: 1 Hour

### Indicators Used
- **EMA 9**: Fast exponential moving average
- **EMA 20**: Slow exponential moving average
- **ATR (14)**: Average True Range for volatility measurement

### Buy Conditions
1. Price is above both EMA 9 and EMA 20
2. EMA 9 is above EMA 20 (uptrend)
3. EMAs are sloping upward (positive slope)
4. Price pulls back and touches either EMA 9 or EMA 20
5. The candle touching EMA has a LOWER WICK (wick ratio > threshold)
6. Candle closes bullish after touching EMA

### Sell Conditions
1. Price is below both EMA 9 and EMA 20
2. EMA 9 is below EMA 20 (downtrend)
3. EMAs are sloping downward (negative slope)
4. Price pulls back to EMA 9 or EMA 20
5. Candle has UPPER WICK (wick ratio > threshold)
6. Candle closes bearish

### Trade Management
- **Stop Loss**: Below wick low/nearest swing low (for buys) or above wick high/nearest swing high (for sells)
- **Take Profit**: Minimum 1:2 risk:reward ratio
- **Breakeven**: Move stop loss to breakeven at 1:1 RR
- **Partial Close**: Close 50% position at 1:2 RR
- **Trailing Stop**: Trail remaining position using EMA 20 or swing highs/lows

## Files Structure

### Core Files
- **`ema_pullback_strategy.py`**: Main backtesting engine and strategy implementation
- **`parameter_optimizer.py`**: Parameter optimization and grid search utilities
- **`requirements_trading.txt`**: Required Python packages

### Key Classes

#### EMAPullbackStrategy
Main strategy implementation with methods:
- `backtest()`: Run complete backtest on OHLCV data
- `check_buy_conditions()` / `check_sell_conditions()`: Entry signal logic
- `calculate_wick_ratio()`: Candlestick wick analysis
- `update_trades()`: Trade management and position tracking
- `plot_results()`: Comprehensive visualization
- `print_results()`: Detailed performance metrics

#### ParameterOptimizer
Parameter optimization tools:
- `grid_search()`: Exhaustive parameter testing
- `find_best_parameters()`: Identify optimal settings
- `analyze_parameter_sensitivity()`: Parameter impact analysis

## Installation

```bash
pip install -r requirements_trading.txt
```

## Usage

### Basic Backtest
```python
from ema_pullback_strategy import EMAPullbackStrategy, generate_sample_data

# Strategy configuration
config = {
    'ema_fast': 9,
    'ema_slow': 20,
    'wick_ratio_threshold': 0.6,
    'min_risk_reward': 2.0,
    'slope_threshold': 0.001,
    'ema_distance_filter': 0.005
}

# Initialize and run
strategy = EMAPullbackStrategy(config)
data = generate_sample_data("BTCUSDT", days=365)
results = strategy.backtest(data, initial_capital=10000)

# View results
strategy.print_results(results)
strategy.plot_results(results, data, "BTCUSDT")
```

### Parameter Optimization
```python
from parameter_optimizer import run_optimization

# Run grid search optimization
run_optimization()
```

### Run Complete Analysis
```bash
# Run basic backtest
python ema_pullback_strategy.py

# Run parameter optimization
python parameter_optimizer.py
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ema_fast` | 9 | Fast EMA period |
| `ema_slow` | 20 | Slow EMA period |
| `wick_ratio_threshold` | 0.6 | Minimum wick size relative to total range |
| `min_risk_reward` | 2.0 | Minimum risk:reward ratio |
| `slope_threshold` | 0.001 | Minimum EMA slope for trend detection |
| `ema_distance_filter` | 0.005 | Minimum distance between EMAs (avoid flat markets) |
| `atr_period` | 14 | ATR period for volatility measurement |
| `swing_lookback` | 10 | Lookback period for swing highs/lows |
| `move_to_breakeven_at_rr` | 1.0 | RR ratio to move stop to breakeven |
| `close_partial_at_rr` | 2.0 | RR ratio for partial position close |
| `partial_close_percentage` | 0.5 | Percentage of position to close partially |

## Performance Metrics

The backtest provides comprehensive metrics:

### Basic Metrics
- **Total Trades**: Number of completed trades
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Total wins / Total losses
- **Total Return**: Overall portfolio return
- **Max Drawdown**: Maximum equity drawdown
- **Sharpe Ratio**: Risk-adjusted returns

### Trade Statistics
- **Average Win/Loss**: Mean profit/loss per trade
- **Average R:R**: Average risk:reward ratio achieved
- **Largest Win/Loss**: Best and worst trade outcomes

## Visualization Features

### Price Chart
- Price action with EMAs
- Entry/exit points marked
- Trade direction indicators

### Equity Curve
- Portfolio value over time
- Drawdown visualization
- Performance trends

### Trade Distribution
- P&L histogram
- Win/loss distribution
- Risk analysis

## Data Requirements

### Input Format
OHLCV DataFrame with datetime index:
```python
data = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
}, index=pd.date_range(...))
```

### Sample Data
The script includes `generate_sample_data()` function for testing with realistic crypto price data.

## Advanced Features

### Multi-Asset Testing
- Test across multiple cryptocurrencies
- Comparative analysis
- Portfolio-level metrics

### Risk Management
- Position sizing (configurable)
- Maximum drawdown limits
- Volatility-based adjustments

### Optimization Tools
- Grid search for parameter tuning
- Sensitivity analysis
- Performance heatmaps

## Expected Performance

Based on historical crypto market characteristics:
- **Win Rate**: 40-60%
- **Profit Factor**: 1.2-2.0
- **Max Drawdown**: 10-25%
- **Sharpe Ratio**: 0.5-1.5

*Note: Actual performance varies by market conditions and parameter settings.*

## Customization

### Adding New Filters
```python
def custom_filter(self, data, i):
    # Add your custom logic here
    return True/False
```

### Modifying Exit Conditions
```python
def custom_exit_logic(self, trade, current_price):
    # Add custom exit conditions
    return should_exit
```

### Position Sizing
```python
def calculate_position_size(self, account_balance, risk_per_trade):
    # Implement your position sizing logic
    return position_size
```

## Best Practices

1. **Parameter Optimization**: Always optimize parameters on out-of-sample data
2. **Risk Management**: Use appropriate position sizing (1-2% risk per trade)
3. **Market Regimes**: Consider different performance in trending vs ranging markets
4. **Slippage & Fees**: Account for trading costs in live trading
5. **Multiple Timeframes**: Consider higher timeframe trends for additional confirmation

## Troubleshooting

### Common Issues
- **No Trades Generated**: Check parameter values and market conditions
- **Poor Performance**: Optimize parameters for specific market conditions
- **Overfitting**: Use walk-forward optimization and out-of-sample testing

### Debug Mode
```python
# Enable detailed logging
strategy.debug_mode = True
results = strategy.backtest(data)
```

## Disclaimer

This backtesting system is for educational and research purposes. Past performance does not guarantee future results. Always conduct thorough testing before implementing with real capital.

## Support

For questions, issues, or contributions:
1. Review the code comments and documentation
2. Check parameter optimization results
3. Test with different market conditions
4. Consider risk management implications
