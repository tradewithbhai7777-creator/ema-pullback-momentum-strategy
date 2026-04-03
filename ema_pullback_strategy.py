"""
9 & 20 EMA Pullback with Wick Rejection Strategy
Complete backtesting implementation for 1H crypto trading
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class EMAPullbackStrategy:
    """
    9 & 20 EMA Pullback with Wick Rejection Strategy Implementation
    """
    
    def __init__(self, config: Dict):
        """
        Initialize strategy with configuration parameters
        
        Args:
            config: Dictionary containing strategy parameters
        """
        self.ema_fast = config.get('ema_fast', 9)
        self.ema_slow = config.get('ema_slow', 20)
        self.wick_ratio_threshold = config.get('wick_ratio_threshold', 0.6)
        self.min_risk_reward = config.get('min_risk_reward', 2.0)
        self.slope_threshold = config.get('slope_threshold', 0.001)
        self.ema_distance_filter = config.get('ema_distance_filter', 0.005)
        self.atr_period = config.get('atr_period', 14)
        self.swing_lookback = config.get('swing_lookback', 10)
        
        # Trade management
        self.move_to_breakeven_at_rr = config.get('move_to_breakeven_at_rr', 1.0)
        self.close_partial_at_rr = config.get('close_partial_at_rr', 2.0)
        self.partial_close_percentage = config.get('partial_close_percentage', 0.5)
        
        # Data storage
        self.trades = []
        self.equity_curve = []
        self.positions = {}
        
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_ema_slope(self, ema: pd.Series) -> pd.Series:
        """Calculate EMA slope (rate of change)"""
        return ema.diff() / ema.shift(1)
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def find_swing_highs_lows(self, data: pd.DataFrame, lookback: int) -> Tuple[pd.Series, pd.Series]:
        """Find swing highs and lows"""
        highs = data['high'].rolling(window=lookback*2+1, center=True).max() == data['high']
        lows = data['low'].rolling(window=lookback*2+1, center=True).min() == data['low']
        return highs, lows
    
    def calculate_wick_ratio(self, candle: pd.Series) -> Tuple[float, float]:
        """
        Calculate upper and lower wick ratios
        
        Returns:
            Tuple of (upper_wick_ratio, lower_wick_ratio)
        """
        body_size = abs(candle['close'] - candle['open'])
        upper_wick = candle['high'] - max(candle['close'], candle['open'])
        lower_wick = min(candle['close'], candle['open']) - candle['low']
        total_range = candle['high'] - candle['low']
        
        if total_range == 0:
            return 0, 0
            
        upper_wick_ratio = upper_wick / total_range
        lower_wick_ratio = lower_wick / total_range
        
        return upper_wick_ratio, lower_wick_ratio
    
    def check_buy_conditions(self, data: pd.DataFrame, i: int) -> bool:
        """Check if buy conditions are met at index i"""
        if i < max(self.ema_slow, self.swing_lookback):
            return False
            
        current = data.iloc[i]
        prev = data.iloc[i-1]
        
        # Condition 1: Price is above both EMAs
        if current['close'] <= current[f'ema_{self.ema_fast}'] or current['close'] <= current[f'ema_{self.ema_slow}']:
            return False
            
        # Condition 2: EMA 9 is above EMA 20
        if current[f'ema_{self.ema_fast}'] <= current[f'ema_{self.ema_slow}']:
            return False
            
        # Condition 3: EMAs are sloping upward
        if current[f'ema_{self.ema_fast}_slope'] <= self.slope_threshold or current[f'ema_{self.ema_slow}_slope'] <= self.slope_threshold:
            return False
            
        # Condition 4: Price pulls back and touches EMA
        touches_ema_fast = (current['low'] <= current[f'ema_{self.ema_fast}'] <= current['high'])
        touches_ema_slow = (current['low'] <= current[f'ema_{self.ema_slow}'] <= current['high'])
        
        if not (touches_ema_fast or touches_ema_slow):
            return False
            
        # Condition 5: Candle has lower wick
        upper_wick_ratio, lower_wick_ratio = self.calculate_wick_ratio(current)
        if lower_wick_ratio < self.wick_ratio_threshold:
            return False
            
        # Condition 6: Candle closes bullish
        if current['close'] <= current['open']:
            return False
            
        # Filter: EMAs not too close (avoid flat market)
        ema_distance = abs(current[f'ema_{self.ema_fast}'] - current[f'ema_{self.ema_slow}']) / current['close']
        if ema_distance < self.ema_distance_filter:
            return False
            
        return True
    
    def check_sell_conditions(self, data: pd.DataFrame, i: int) -> bool:
        """Check if sell conditions are met at index i"""
        if i < max(self.ema_slow, self.swing_lookback):
            return False
            
        current = data.iloc[i]
        prev = data.iloc[i-1]
        
        # Condition 1: Price is below both EMAs
        if current['close'] >= current[f'ema_{self.ema_fast}'] or current['close'] >= current[f'ema_{self.ema_slow}']:
            return False
            
        # Condition 2: EMA 9 is below EMA 20
        if current[f'ema_{self.ema_fast}'] >= current[f'ema_{self.ema_slow}']:
            return False
            
        # Condition 3: EMAs are sloping downward
        if current[f'ema_{self.ema_fast}_slope'] >= -self.slope_threshold or current[f'ema_{self.ema_slow}_slope'] >= -self.slope_threshold:
            return False
            
        # Condition 4: Price pulls back to EMA
        touches_ema_fast = (current['low'] <= current[f'ema_{self.ema_fast}'] <= current['high'])
        touches_ema_slow = (current['low'] <= current[f'ema_{self.ema_slow}'] <= current['high'])
        
        if not (touches_ema_fast or touches_ema_slow):
            return False
            
        # Condition 5: Candle has upper wick
        upper_wick_ratio, lower_wick_ratio = self.calculate_wick_ratio(current)
        if upper_wick_ratio < self.wick_ratio_threshold:
            return False
            
        # Condition 6: Candle closes bearish
        if current['close'] >= current['open']:
            return False
            
        # Filter: EMAs not too close
        ema_distance = abs(current[f'ema_{self.ema_fast}'] - current[f'ema_{self.ema_slow}']) / current['close']
        if ema_distance < self.ema_distance_filter:
            return False
            
        return True
    
    def find_stop_loss_level(self, data: pd.DataFrame, i: int, trade_type: str) -> float:
        """Find stop loss level based on wick low/high or swing low/high"""
        current = data.iloc[i]
        
        if trade_type == 'BUY':
            # Stop below wick low
            stop_loss = current['low']
            
            # Find nearest swing low
            swing_lows = data['swing_low'].iloc[max(0, i-self.swing_lookback):i]
            if swing_lows.any():
                nearest_swing_low = data.loc[swing_lows[swing_lows].index[-1], 'low']
                stop_loss = max(stop_loss, nearest_swing_low)
                
        else:  # SELL
            # Stop above wick high
            stop_loss = current['high']
            
            # Find nearest swing high
            swing_highs = data['swing_high'].iloc[max(0, i-self.swing_lookback):i]
            if swing_highs.any():
                nearest_swing_high = data.loc[swing_highs[swing_highs].index[-1], 'high']
                stop_loss = min(stop_loss, nearest_swing_high)
                
        return stop_loss
    
    def calculate_take_profit_level(self, entry_price: float, stop_loss: float, trade_type: str) -> float:
        """Calculate take profit level based on risk:reward ratio"""
        risk = abs(entry_price - stop_loss)
        reward = risk * self.min_risk_reward
        
        if trade_type == 'BUY':
            return entry_price + reward
        else:
            return entry_price - reward
    
    def update_trades(self, data: pd.DataFrame, i: int):
        """Update open trades with current bar data"""
        current_time = data.index[i]
        current_price = data.iloc[i]['close']
        
        trades_to_close = []
        
        for trade_id, trade in self.positions.items():
            if trade['status'] != 'open':
                continue
                
            entry_price = trade['entry_price']
            stop_loss = trade['stop_loss']
            take_profit = trade['take_profit']
            trade_type = trade['type']
            
            # Check stop loss hit
            if trade_type == 'BUY':
                if current_price <= stop_loss:
                    trade['exit_price'] = stop_loss
                    trade['exit_time'] = current_time
                    trade['exit_reason'] = 'stop_loss'
                    trade['status'] = 'closed'
                    trades_to_close.append(trade_id)
                    
                # Check take profit hit
                elif current_price >= take_profit:
                    trade['exit_price'] = take_profit
                    trade['exit_time'] = current_time
                    trade['exit_reason'] = 'take_profit'
                    trade['status'] = 'closed'
                    trades_to_close.append(trade_id)
                    
                # Check partial close at 1:2 RR
                elif not trade.get('partial_closed', False):
                    rr_ratio = (current_price - entry_price) / (entry_price - stop_loss)
                    if rr_ratio >= self.close_partial_at_rr:
                        trade['partial_closed'] = True
                        trade['partial_close_price'] = current_price
                        trade['partial_close_time'] = current_time
                        
                # Move to breakeven at 1:1 RR
                elif not trade.get('breakeven_moved', False):
                    rr_ratio = (current_price - entry_price) / (entry_price - stop_loss)
                    if rr_ratio >= self.move_to_breakeven_at_rr:
                        trade['stop_loss'] = entry_price
                        trade['breakeven_moved'] = True
                        
                # Trail stop loss using EMA 20 or swing lows
                else:
                    ema_20 = data.iloc[i][f'ema_{self.ema_slow}']
                    if ema_20 > trade['stop_loss']:
                        trade['stop_loss'] = ema_20
                        
            else:  # SELL
                if current_price >= stop_loss:
                    trade['exit_price'] = stop_loss
                    trade['exit_time'] = current_time
                    trade['exit_reason'] = 'stop_loss'
                    trade['status'] = 'closed'
                    trades_to_close.append(trade_id)
                    
                elif current_price <= take_profit:
                    trade['exit_price'] = take_profit
                    trade['exit_time'] = current_time
                    trade['exit_reason'] = 'take_profit'
                    trade['status'] = 'closed'
                    trades_to_close.append(trade_id)
                    
                # Check partial close at 1:2 RR
                elif not trade.get('partial_closed', False):
                    rr_ratio = (entry_price - current_price) / (stop_loss - entry_price)
                    if rr_ratio >= self.close_partial_at_rr:
                        trade['partial_closed'] = True
                        trade['partial_close_price'] = current_price
                        trade['partial_close_time'] = current_time
                        
                # Move to breakeven at 1:1 RR
                elif not trade.get('breakeven_moved', False):
                    rr_ratio = (entry_price - current_price) / (stop_loss - entry_price)
                    if rr_ratio >= self.move_to_breakeven_at_rr:
                        trade['stop_loss'] = entry_price
                        trade['breakeven_moved'] = True
                        
                # Trail stop loss using EMA 20 or swing highs
                else:
                    ema_20 = data.iloc[i][f'ema_{self.ema_slow}']
                    if ema_20 < trade['stop_loss']:
                        trade['stop_loss'] = ema_20
        
        # Close trades
        for trade_id in trades_to_close:
            trade = self.positions[trade_id]
            self.trades.append(trade)
            del self.positions[trade_id]
    
    def calculate_pnl(self, trade: Dict) -> float:
        """Calculate profit/loss for a trade"""
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        trade_type = trade['type']
        
        if trade_type == 'BUY':
            pnl = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) / entry_price
            
        return pnl
    
    def backtest(self, data: pd.DataFrame, initial_capital: float = 10000) -> Dict:
        """
        Run backtest on the given data
        
        Args:
            data: OHLCV DataFrame with datetime index
            initial_capital: Starting capital for the backtest
            
        Returns:
            Dictionary containing backtest results
        """
        print(f"Starting backtest with {len(data)} candles...")
        
        # Reset state
        self.trades = []
        self.positions = {}
        self.equity_curve = []
        
        # Calculate indicators
        data[f'ema_{self.ema_fast}'] = self.calculate_ema(data['close'], self.ema_fast)
        data[f'ema_{self.ema_slow}'] = self.calculate_ema(data['close'], self.ema_slow)
        data[f'ema_{self.ema_fast}_slope'] = self.calculate_ema_slope(data[f'ema_{self.ema_fast}'])
        data[f'ema_{self.ema_slow}_slope'] = self.calculate_ema_slope(data[f'ema_{self.ema_slow}'])
        data['atr'] = self.calculate_atr(data['high'], data['low'], data['close'], self.atr_period)
        data['swing_high'], data['swing_low'] = self.find_swing_highs_lows(data, self.swing_lookback)
        
        # Initialize equity
        current_equity = initial_capital
        trade_id_counter = 0
        
        # Run through each candle
        for i in range(len(data)):
            current_time = data.index[i]
            
            # Update existing trades
            self.update_trades(data, i)
            
            # Check for new trade entries
            if len(self.positions) == 0:  # Only one trade at a time
                # Check buy conditions
                if self.check_buy_conditions(data, i):
                    current = data.iloc[i]
                    entry_price = current['close']
                    stop_loss = self.find_stop_loss_level(data, i, 'BUY')
                    take_profit = self.calculate_take_profit_level(entry_price, stop_loss, 'BUY')
                    
                    # Open buy trade
                    trade = {
                        'trade_id': trade_id_counter,
                        'type': 'BUY',
                        'entry_time': current_time,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'status': 'open',
                        'size': 1.0  # Position size (can be optimized)
                    }
                    
                    self.positions[trade_id_counter] = trade
                    trade_id_counter += 1
                    
                # Check sell conditions
                elif self.check_sell_conditions(data, i):
                    current = data.iloc[i]
                    entry_price = current['close']
                    stop_loss = self.find_stop_loss_level(data, i, 'SELL')
                    take_profit = self.calculate_take_profit_level(entry_price, stop_loss, 'SELL')
                    
                    # Open sell trade
                    trade = {
                        'trade_id': trade_id_counter,
                        'type': 'SELL',
                        'entry_time': current_time,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'status': 'open',
                        'size': 1.0
                    }
                    
                    self.positions[trade_id_counter] = trade
                    trade_id_counter += 1
            
            # Calculate current equity
            open_trades_pnl = 0
            for trade in self.positions.values():
                if trade['status'] == 'open':
                    current_price = data.iloc[i]['close']
                    if trade['type'] == 'BUY':
                        open_trades_pnl += (current_price - trade['entry_price']) / trade['entry_price']
                    else:
                        open_trades_pnl += (trade['entry_price'] - current_price) / trade['entry_price']
            
            # Add closed trades PnL
            closed_trades_pnl = sum(self.calculate_pnl(trade) for trade in self.trades)
            
            current_equity = initial_capital * (1 + closed_trades_pnl + open_trades_pnl)
            
            self.equity_curve.append({
                'time': current_time,
                'equity': current_equity,
                'open_trades': len(self.positions),
                'total_trades': len(self.trades)
            })
        
        # Close any remaining open trades at the end
        final_price = data.iloc[-1]['close']
        for trade_id, trade in list(self.positions.items()):
            trade['exit_price'] = final_price
            trade['exit_time'] = data.index[-1]
            trade['exit_reason'] = 'end_of_data'
            trade['status'] = 'closed'
            self.trades.append(trade)
        
        # Calculate final results
        results = self.calculate_results(initial_capital)
        
        return results
    
    def calculate_results(self, initial_capital: float) -> Dict:
        """Calculate comprehensive backtest results"""
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        if trades_df.empty:
            # Return empty results with proper structure
            equity_df = pd.DataFrame(self.equity_curve)
            if not equity_df.empty:
                equity_df.set_index('time', inplace=True)
                peak = equity_df['equity'].expanding().max()
                drawdown = (equity_df['equity'] - peak) / peak
                max_drawdown = drawdown.min()
                returns = equity_df['equity'].pct_change().dropna()
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                total_return = (equity_df['equity'].iloc[-1] / initial_capital) - 1
            else:
                max_drawdown = 0
                sharpe_ratio = 0
                total_return = 0
            
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_return': total_return,
                'total_pnl': 0,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'avg_win': 0,
                'avg_loss': 0,
                'avg_rr': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'trades': trades_df,
                'equity_curve': equity_df
            }
        
        trades_df['pnl'] = trades_df.apply(self.calculate_pnl, axis=1)
        trades_df['rr_ratio'] = trades_df.apply(
            lambda x: abs(x['pnl']) / (abs(x['entry_price'] - x['stop_loss']) / x['entry_price']), 
            axis=1
        )
        
        # Separate wins and losses
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(wins)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit factor
        total_wins = wins['pnl'].sum() if not wins.empty else 0
        total_losses = abs(losses['pnl'].sum()) if not losses.empty else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Return metrics
        total_pnl = trades_df['pnl'].sum()
        total_return = total_pnl
        
        # Equity curve calculations
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('time', inplace=True)
        
        # Maximum drawdown
        peak = equity_df['equity'].expanding().max()
        drawdown = (equity_df['equity'] - peak) / peak
        max_drawdown = drawdown.min()
        
        # Sharpe ratio (assuming 1% risk-free rate)
        returns = equity_df['equity'].pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Additional metrics
        avg_win = wins['pnl'].mean() if not wins.empty else 0
        avg_loss = losses['pnl'].mean() if not losses.empty else 0
        avg_rr = trades_df['rr_ratio'].mean()
        
        results = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_rr': avg_rr,
            'largest_win': wins['pnl'].max() if not wins.empty else 0,
            'largest_loss': losses['pnl'].min() if not losses.empty else 0,
            'trades': trades_df,
            'equity_curve': equity_df
        }
        
        return results
    
    def plot_results(self, results: Dict, data: pd.DataFrame, symbol: str = "BTCUSDT"):
        """Plot comprehensive backtest results"""
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        # Plot 1: Price chart with trades
        ax1 = axes[0]
        ax1.plot(data.index, data['close'], label='Price', linewidth=1, alpha=0.7)
        ax1.plot(data.index, data[f'ema_{self.ema_fast}'], label=f'EMA {self.ema_fast}', linewidth=1)
        ax1.plot(data.index, data[f'ema_{self.ema_slow}'], label=f'EMA {self.ema_slow}', linewidth=1)
        
        # Plot trades
        trades_df = results['trades']
        if not trades_df.empty:
            # Buy entries
            buy_trades = trades_df[trades_df['type'] == 'BUY']
            if not buy_trades.empty:
                ax1.scatter(buy_trades['entry_time'], buy_trades['entry_price'], 
                           color='green', marker='^', s=100, label='Buy Entry', zorder=5)
                ax1.scatter(buy_trades['exit_time'], buy_trades['exit_price'], 
                           color='red', marker='v', s=100, label='Buy Exit', zorder=5)
            
            # Sell entries
            sell_trades = trades_df[trades_df['type'] == 'SELL']
            if not sell_trades.empty:
                ax1.scatter(sell_trades['entry_time'], sell_trades['entry_price'], 
                           color='red', marker='v', s=100, label='Sell Entry', zorder=5)
                ax1.scatter(sell_trades['exit_time'], sell_trades['exit_price'], 
                           color='green', marker='^', s=100, label='Sell Exit', zorder=5)
        
        ax1.set_title(f'{symbol} - 9 & 20 EMA Pullback Strategy', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Equity Curve
        ax2 = axes[1]
        equity_df = results['equity_curve']
        ax2.plot(equity_df.index, equity_df['equity'], label='Equity Curve', linewidth=2, color='blue')
        
        # Add drawdown
        peak = equity_df['equity'].expanding().max()
        drawdown = (equity_df['equity'] - peak) / peak * 100
        ax2.fill_between(equity_df.index, drawdown, 0, alpha=0.3, color='red', label='Drawdown %')
        
        ax2.set_title('Equity Curve & Drawdown', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Equity ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Trade Distribution
        ax3 = axes[2]
        if not trades_df.empty:
            wins = trades_df[trades_df['pnl'] > 0]['pnl']
            losses = trades_df[trades_df['pnl'] <= 0]['pnl']
            
            ax3.hist([wins * 100, losses * 100], bins=30, alpha=0.7, 
                    label=['Wins (%)', 'Losses (%)'], color=['green', 'red'])
            ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        ax3.set_title('Trade P&L Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('P&L (%)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{symbol}_backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_results(self, results: Dict, symbol: str = "BTCUSDT"):
        """Print detailed backtest results"""
        print("\n" + "="*60)
        print(f"BACKTEST RESULTS - {symbol}")
        print("="*60)
        
        print(f"Total Trades: {results['total_trades']}")
        print(f"Winning Trades: {results['winning_trades']}")
        print(f"Losing Trades: {results['losing_trades']}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Total P&L: {results['total_pnl']:.2%}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        
        print(f"\nTrade Statistics:")
        print(f"Average Win: {results['avg_win']:.2%}")
        print(f"Average Loss: {results['avg_loss']:.2%}")
        print(f"Average R:R: {results['avg_rr']:.2f}")
        print(f"Largest Win: {results['largest_win']:.2%}")
        print(f"Largest Loss: {results['largest_loss']:.2%}")
        
        print("\n" + "="*60)


def generate_sample_data(symbol: str = "BTCUSDT", days: int = 365) -> pd.DataFrame:
    """Generate sample crypto data for testing"""
    print(f"Generating sample data for {symbol} ({days} days)...")
    
    # Create date range
    start_date = datetime.now() - timedelta(days=days)
    dates = pd.date_range(start=start_date, periods=days*24, freq='1H')  # 1-hour candles
    
    # Generate realistic price data with trend and volatility
    np.random.seed(42)
    
    # Base price with trend
    base_price = 50000 if symbol == "BTCUSDT" else 3000
    trend = np.linspace(0, 0.3, len(dates))  # 30% uptrend over the period
    
    # Add random walk component
    random_walk = np.cumsum(np.random.normal(0, 0.02, len(dates)))
    
    # Combine trend and random walk
    price_multiplier = 1 + trend + random_walk * 0.1
    prices = base_price * price_multiplier
    
    # Generate OHLC data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Add some volatility
        volatility = np.random.normal(0, 0.02)
        
        # Calculate OHLC
        high = close * (1 + abs(volatility) * 0.5)
        low = close * (1 - abs(volatility) * 0.5)
        
        if i > 0:
            prev_close = data[-1]['close']
            # Ensure continuity
            open_price = prev_close * (1 + np.random.normal(0, 0.005))
            open_price = max(low, min(high, open_price))
        else:
            open_price = close
        
        # Generate volume
        volume = np.random.lognormal(15, 1)  # Realistic crypto volume
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    return df


def main():
    """Main function to run the backtest"""
    print("="*60)
    print("9 & 20 EMA Pullback with Wick Rejection Strategy")
    print("Backtesting Engine")
    print("="*60)
    
    # Strategy configuration
    config = {
        'ema_fast': 9,
        'ema_slow': 20,
        'wick_ratio_threshold': 0.4,  # Reduced from 0.6 to generate more trades
        'min_risk_reward': 1.5,      # Reduced from 2.0
        'slope_threshold': 0.0005,    # Reduced from 0.001
        'ema_distance_filter': 0.002, # Reduced from 0.005
        'atr_period': 14,
        'swing_lookback': 10,
        'move_to_breakeven_at_rr': 1.0,
        'close_partial_at_rr': 2.0,
        'partial_close_percentage': 0.5
    }
    
    # Initialize strategy
    strategy = EMAPullbackStrategy(config)
    
    # Test multiple symbols
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    all_results = {}
    
    for symbol in symbols:
        print(f"\n{'='*40}")
        print(f"Testing {symbol}")
        print(f"{'='*40}")
        
        # Generate or load data
        data = generate_sample_data(symbol, days=365)
        
        # Run backtest
        results = strategy.backtest(data, initial_capital=10000)
        
        # Store results
        all_results[symbol] = results
        
        # Print results
        strategy.print_results(results, symbol)
        
        # Plot results
        strategy.plot_results(results, data, symbol)
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    
    summary_data = []
    for symbol, results in all_results.items():
        summary_data.append({
            'Symbol': symbol,
            'Total Trades': results['total_trades'],
            'Win Rate': f"{results['win_rate']:.1%}",
            'Profit Factor': f"{results['profit_factor']:.2f}",
            'Total Return': f"{results['total_return']:.1%}",
            'Max DD': f"{results['max_drawdown']:.1%}",
            'Sharpe': f"{results['sharpe_ratio']:.2f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    print(f"\n{'='*60}")
    print("Backtesting Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
