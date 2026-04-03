"""
9 & 20 EMA Pullback with Strong Wick Rejection Strategy
Strict rule-based implementation for 1H crypto trading
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

class EMAPullbackStrict:
    """
    Strict 9 & 20 EMA Pullback with Strong Wick Rejection Strategy
    All conditions must be met precisely as specified
    """
    
    def __init__(self, config: Dict):
        """
        Initialize strategy with configuration parameters
        
        Args:
            config: Dictionary containing strategy parameters
        """
        # Core parameters
        self.ema_fast = config.get('ema_fast', 9)
        self.ema_slow = config.get('ema_slow', 20)
        self.adx_period = config.get('adx_period', 14)
        self.adx_threshold = config.get('adx_threshold', 20)
        
        # Wick rejection parameters
        self.lower_wick_ratio = config.get('lower_wick_ratio', 0.5)  # 50% of range
        self.upper_wick_ratio = config.get('upper_wick_ratio', 0.5)  # 50% of range
        self.wick_body_multiplier = config.get('wick_body_multiplier', 2.0)  # wick >= 2x body
        
        # EMA separation filter
        self.ema_distance_threshold = config.get('ema_distance_threshold', 0.001)
        
        # Risk management
        self.risk_reward_ratio = config.get('risk_reward_ratio', 2.0)
        self.move_to_breakeven_at_rr = config.get('move_to_breakeven_at_rr', 1.0)
        self.close_partial_at_rr = config.get('close_partial_at_rr', 2.0)
        self.partial_close_percentage = config.get('partial_close_percentage', 0.5)
        
        # Trend analysis
        self.ema_spread_lookback = config.get('ema_spread_lookback', 3)
        self.ema_flat_lookback = config.get('ema_flat_lookback', 5)
        self.ema_flat_threshold = config.get('ema_flat_threshold', 0.0001)
        
        # Position management
        self.max_trades_per_day = config.get('max_trades_per_day', 2)
        self.swing_lookback = config.get('swing_lookback', 10)
        
        # Data storage
        self.trades = []
        self.equity_curve = []
        self.positions = {}
        
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate ADX (Average Directional Index)"""
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth TR, +DM, -DM
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean()
        plus_di = pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr * 100
        minus_di = pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr * 100
        
        # Calculate ADX
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return adx
    
    def calculate_ema_slope(self, ema: pd.Series, lookback: int) -> pd.Series:
        """Calculate EMA slope over specified lookback period"""
        return (ema - ema.shift(lookback)) / ema.shift(lookback) if lookback > 0 else pd.Series(0, index=ema.index)
    
    def check_ema_spread_increasing(self, data: pd.DataFrame, i: int) -> bool:
        """Check if EMA spread is increasing over last N candles"""
        if i < self.ema_spread_lookback:
            return False
            
        current_spread = abs(data.iloc[i][f'ema_{self.ema_fast}'] - data.iloc[i][f'ema_{self.ema_slow}'])
        
        for j in range(1, self.ema_spread_lookback + 1):
            prev_spread = abs(data.iloc[i-j][f'ema_{self.ema_fast}'] - data.iloc[i-j][f'ema_{self.ema_slow}'])
            if current_spread <= prev_spread:
                return False
        
        return True
    
    def check_ema_flat(self, data: pd.DataFrame, i: int) -> bool:
        """Check if EMAs are too flat (sideways market)"""
        if i < self.ema_flat_lookback:
            return False
            
        fast_ema = data[f'ema_{self.ema_fast}'].iloc[i-self.ema_flat_lookback:i+1]
        slow_ema = data[f'ema_{self.ema_slow}'].iloc[i-self.ema_flat_lookback:i+1]
        
        # Calculate slope over lookback period
        fast_slope = (fast_ema.iloc[-1] - fast_ema.iloc[0]) / fast_ema.iloc[0]
        slow_slope = (slow_ema.iloc[-1] - slow_ema.iloc[0]) / slow_ema.iloc[0]
        
        return abs(fast_slope) < self.ema_flat_threshold and abs(slow_slope) < self.ema_flat_threshold
    
    def calculate_wick_metrics(self, candle: pd.Series) -> Dict:
        """
        Calculate detailed wick metrics for a candle
        
        Returns:
            Dictionary with wick measurements
        """
        open_price = candle['open']
        close_price = candle['close']
        high_price = candle['high']
        low_price = candle['low']
        
        # Basic measurements
        candle_range = high_price - low_price
        body = abs(close_price - open_price)
        upper_wick = high_price - max(open_price, close_price)
        lower_wick = min(open_price, close_price) - low_price
        
        # Ratios
        upper_wick_ratio = upper_wick / candle_range if candle_range > 0 else 0
        lower_wick_ratio = lower_wick / candle_range if candle_range > 0 else 0
        
        # Body to wick ratios
        upper_wick_body_ratio = upper_wick / body if body > 0 else 0
        lower_wick_body_ratio = lower_wick / body if body > 0 else 0
        
        return {
            'range': candle_range,
            'body': body,
            'upper_wick': upper_wick,
            'lower_wick': lower_wick,
            'upper_wick_ratio': upper_wick_ratio,
            'lower_wick_ratio': lower_wick_ratio,
            'upper_wick_body_ratio': upper_wick_body_ratio,
            'lower_wick_body_ratio': lower_wick_body_ratio
        }
    
    def check_ema_touch(self, candle: pd.Series, ema_fast: float, ema_slow: float) -> Tuple[bool, Optional[float]]:
        """
        Check if candle touches/dips into either EMA
        
        Returns:
            Tuple of (touches_ema, touched_ema_value)
        """
        low_price = candle['low']
        high_price = candle['high']
        
        # Check if low touches or dips into EMAs
        if low_price <= ema_fast <= high_price:
            return True, ema_fast
        elif low_price <= ema_slow <= high_price:
            return True, ema_slow
        
        return False, None
    
    def check_buy_conditions(self, data: pd.DataFrame, i: int) -> bool:
        """Check if ALL buy conditions are met at index i"""
        if i < max(self.ema_slow, self.adx_period, self.ema_spread_lookback, self.ema_flat_lookback):
            return False
            
        current = data.iloc[i]
        prev = data.iloc[i-1]
        
        # 1. Trend Direction
        if not (current['close'] > current[f'ema_{self.ema_fast}'] and 
                current['close'] > current[f'ema_{self.ema_slow}'] and
                current[f'ema_{self.ema_fast}'] > current[f'ema_{self.ema_slow}']):
            return False
        
        # 2. Trend Strength (at least one condition)
        adx_strong = current['adx'] > self.adx_threshold
        ema_spread_increasing = self.check_ema_spread_increasing(data, i)
        
        if not (adx_strong or ema_spread_increasing):
            return False
        
        # 3. EMA Separation Filter
        ema_distance = abs(current[f'ema_{self.ema_fast}'] - current[f'ema_{self.ema_slow}']) / current['close']
        if ema_distance <= self.ema_distance_threshold:
            return False
        
        # 4. Pullback Condition (current or previous candle)
        current_touch, current_touched_ema = self.check_ema_touch(
            current, current[f'ema_{self.ema_fast}'], current[f'ema_{self.ema_slow}']
        )
        prev_touch, prev_touched_ema = self.check_ema_touch(
            prev, prev[f'ema_{self.ema_fast}'], prev[f'ema_{self.ema_slow}']
        )
        
        if not (current_touch or prev_touch):
            return False
        
        # Use the candle that touched EMA for wick analysis
        analysis_candle = current if current_touch else prev
        touched_ema = current_touched_ema if current_touch else prev_touched_ema
        
        # 5. Strong Wick Rejection (VERY IMPORTANT)
        wick_metrics = self.calculate_wick_metrics(analysis_candle)
        
        if not (wick_metrics['lower_wick_ratio'] >= self.lower_wick_ratio and
                wick_metrics['lower_wick_body_ratio'] >= self.wick_body_multiplier):
            return False
        
        # 6. Confirmation
        if not (current['close'] > current['open'] and  # Bullish close
                current['close'] > touched_ema):        # Close above touched EMA
            return False
        
        # Global Filters
        if self.check_ema_flat(data, i):
            return False
        
        # Check for extreme candle ranges (avoid late entries)
        avg_range = data['high'].rolling(20).mean() - data['low'].rolling(20).mean()
        if wick_metrics['range'] > avg_range.iloc[i] * 2:  # More than 2x average range
            return False
        
        # Check max trades per day
        current_date = current.name.date()
        trades_today = len([t for t in self.trades if t['entry_time'].date() == current_date])
        if trades_today >= self.max_trades_per_day:
            return False
        
        return True
    
    def check_sell_conditions(self, data: pd.DataFrame, i: int) -> bool:
        """Check if ALL sell conditions are met at index i"""
        if i < max(self.ema_slow, self.adx_period, self.ema_spread_lookback, self.ema_flat_lookback):
            return False
            
        current = data.iloc[i]
        prev = data.iloc[i-1]
        
        # 1. Trend Direction
        if not (current['close'] < current[f'ema_{self.ema_fast}'] and 
                current['close'] < current[f'ema_{self.ema_slow}'] and
                current[f'ema_{self.ema_fast}'] < current[f'ema_{self.ema_slow}']):
            return False
        
        # 2. Trend Strength
        adx_strong = current['adx'] > self.adx_threshold
        ema_spread_increasing = self.check_ema_spread_increasing(data, i)
        
        if not (adx_strong or ema_spread_increasing):
            return False
        
        # 3. EMA Separation Filter
        ema_distance = abs(current[f'ema_{self.ema_fast}'] - current[f'ema_{self.ema_slow}']) / current['close']
        if ema_distance <= self.ema_distance_threshold:
            return False
        
        # 4. Pullback Condition
        current_touch, current_touched_ema = self.check_ema_touch(
            current, current[f'ema_{self.ema_fast}'], current[f'ema_{self.ema_slow}']
        )
        prev_touch, prev_touched_ema = self.check_ema_touch(
            prev, prev[f'ema_{self.ema_fast}'], prev[f'ema_{self.ema_slow}']
        )
        
        if not (current_touch or prev_touch):
            return False
        
        # Use the candle that touched EMA for wick analysis
        analysis_candle = current if current_touch else prev
        touched_ema = current_touched_ema if current_touch else prev_touched_ema
        
        # 5. Strong Wick Rejection
        wick_metrics = self.calculate_wick_metrics(analysis_candle)
        
        if not (wick_metrics['upper_wick_ratio'] >= self.upper_wick_ratio and
                wick_metrics['upper_wick_body_ratio'] >= self.wick_body_multiplier):
            return False
        
        # 6. Confirmation
        if not (current['close'] < current['open'] and  # Bearish close
                current['close'] < touched_ema):        # Close below touched EMA
            return False
        
        # Global Filters
        if self.check_ema_flat(data, i):
            return False
        
        # Check for extreme candle ranges
        avg_range = data['high'].rolling(20).mean() - data['low'].rolling(20).mean()
        if wick_metrics['range'] > avg_range.iloc[i] * 2:
            return False
        
        # Check max trades per day
        current_date = current.name.date()
        trades_today = len([t for t in self.trades if t['entry_time'].date() == current_date])
        if trades_today >= self.max_trades_per_day:
            return False
        
        return True
    
    def find_swing_highs_lows(self, data: pd.DataFrame, lookback: int) -> Tuple[pd.Series, pd.Series]:
        """Find swing highs and lows"""
        highs = data['high'].rolling(window=lookback*2+1, center=True).max() == data['high']
        lows = data['low'].rolling(window=lookback*2+1, center=True).min() == data['low']
        return highs, lows
    
    def update_trades(self, data: pd.DataFrame, i: int):
        """Update open trades with current bar data"""
        current_time = data.index[i]
        current_price = data.iloc[i]['close']
        current_ema_20 = data.iloc[i][f'ema_{self.ema_slow}']
        
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
                    # Use EMA 20 if it's higher than current stop
                    if current_ema_20 > trade['stop_loss']:
                        trade['stop_loss'] = current_ema_20
                        
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
                    if current_ema_20 < trade['stop_loss']:
                        trade['stop_loss'] = current_ema_20
        
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
        data['adx'] = self.calculate_adx(data['high'], data['low'], data['close'], self.adx_period)
        data['swing_high'], data['swing_low'] = self.find_swing_highs_lows(data, self.swing_lookback)
        
        # Initialize equity
        current_equity = initial_capital
        trade_id_counter = 0
        
        # Run through each candle
        for i in range(len(data)):
            current_time = data.index[i]
            
            # Update existing trades
            self.update_trades(data, i)
            
            # Check for new trade entries (only one trade at a time)
            if len(self.positions) == 0:
                # Check buy conditions
                if self.check_buy_conditions(data, i):
                    current = data.iloc[i]
                    entry_price = current['close']
                    
                    # Find rejection candle for stop loss
                    prev = data.iloc[i-1]
                    current_touch, _ = self.check_ema_touch(
                        current, current[f'ema_{self.ema_fast}'], current[f'ema_{self.ema_slow}']
                    )
                    prev_touch, _ = self.check_ema_touch(
                        prev, prev[f'ema_{self.ema_fast}'], prev[f'ema_{self.ema_slow}']
                    )
                    
                    rejection_candle = current if current_touch else prev
                    stop_loss = rejection_candle['low']
                    
                    # Calculate take profit
                    risk = entry_price - stop_loss
                    take_profit = entry_price + (risk * self.risk_reward_ratio)
                    
                    # Open buy trade
                    trade = {
                        'trade_id': trade_id_counter,
                        'type': 'BUY',
                        'entry_time': current_time,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'status': 'open',
                        'size': 1.0,
                        'risk': risk
                    }
                    
                    self.positions[trade_id_counter] = trade
                    trade_id_counter += 1
                    
                # Check sell conditions
                elif self.check_sell_conditions(data, i):
                    current = data.iloc[i]
                    entry_price = current['close']
                    
                    # Find rejection candle for stop loss
                    prev = data.iloc[i-1]
                    current_touch, _ = self.check_ema_touch(
                        current, current[f'ema_{self.ema_fast}'], current[f'ema_{self.ema_slow}']
                    )
                    prev_touch, _ = self.check_ema_touch(
                        prev, prev[f'ema_{self.ema_fast}'], prev[f'ema_{self.ema_slow}']
                    )
                    
                    rejection_candle = current if current_touch else prev
                    stop_loss = rejection_candle['high']
                    
                    # Calculate take profit
                    risk = stop_loss - entry_price
                    take_profit = entry_price - (risk * self.risk_reward_ratio)
                    
                    # Open sell trade
                    trade = {
                        'trade_id': trade_id_counter,
                        'type': 'SELL',
                        'entry_time': current_time,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'status': 'open',
                        'size': 1.0,
                        'risk': risk
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
                'avg_trade_duration': 0,
                'trades': trades_df,
                'equity_curve': equity_df
            }
        
        trades_df['pnl'] = trades_df.apply(self.calculate_pnl, axis=1)
        trades_df['rr_ratio'] = trades_df.apply(
            lambda x: abs(x['pnl']) / (abs(x['entry_price'] - x['stop_loss']) / x['entry_price']), 
            axis=1
        )
        trades_df['duration'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 3600  # hours
        
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
        avg_duration = trades_df['duration'].mean()
        
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
            'avg_trade_duration': avg_duration,
            'trades': trades_df,
            'equity_curve': equity_df
        }
        
        return results
    
    def plot_results(self, results: Dict, data: pd.DataFrame, symbol: str = "BTCUSDT"):
        """Plot comprehensive backtest results"""
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        # Plot 1: Price chart with trades
        ax1 = axes[0]
        ax1.plot(data.index, data['close'], label='Price', linewidth=1, alpha=0.7, color='black')
        ax1.plot(data.index, data[f'ema_{self.ema_fast}'], label=f'EMA {self.ema_fast}', linewidth=1.5, color='blue')
        ax1.plot(data.index, data[f'ema_{self.ema_slow}'], label=f'EMA {self.ema_slow}', linewidth=1.5, color='red')
        
        # Plot ADX on secondary axis
        ax1_adx = ax1.twinx()
        ax1_adx.plot(data.index, data['adx'], label='ADX', linewidth=1, alpha=0.5, color='green')
        ax1_adx.axhline(y=self.adx_threshold, color='green', linestyle='--', alpha=0.5)
        ax1_adx.set_ylabel('ADX', color='green')
        ax1_adx.tick_params(axis='y', labelcolor='green')
        
        # Plot trades
        trades_df = results['trades']
        if not trades_df.empty:
            # Buy entries and exits
            buy_trades = trades_df[trades_df['type'] == 'BUY']
            if not buy_trades.empty:
                ax1.scatter(buy_trades['entry_time'], buy_trades['entry_price'], 
                           color='green', marker='^', s=100, label='Buy Entry', zorder=5, alpha=0.8)
                ax1.scatter(buy_trades['exit_time'], buy_trades['exit_price'], 
                           color='red', marker='v', s=100, label='Buy Exit', zorder=5, alpha=0.8)
                
                # Plot stop loss and take profit lines for buys
                for _, trade in buy_trades.iterrows():
                    ax1.plot([trade['entry_time'], trade['exit_time']], 
                            [trade['stop_loss'], trade['stop_loss']], 
                            'r--', alpha=0.3, linewidth=0.5)
                    ax1.plot([trade['entry_time'], trade['exit_time']], 
                            [trade['take_profit'], trade['take_profit']], 
                            'g--', alpha=0.3, linewidth=0.5)
            
            # Sell entries and exits
            sell_trades = trades_df[trades_df['type'] == 'SELL']
            if not sell_trades.empty:
                ax1.scatter(sell_trades['entry_time'], sell_trades['entry_price'], 
                           color='red', marker='v', s=100, label='Sell Entry', zorder=5, alpha=0.8)
                ax1.scatter(sell_trades['exit_time'], sell_trades['exit_price'], 
                           color='green', marker='^', s=100, label='Sell Exit', zorder=5, alpha=0.8)
                
                # Plot stop loss and take profit lines for sells
                for _, trade in sell_trades.iterrows():
                    ax1.plot([trade['entry_time'], trade['exit_time']], 
                            [trade['stop_loss'], trade['stop_loss']], 
                            'r--', alpha=0.3, linewidth=0.5)
                    ax1.plot([trade['entry_time'], trade['exit_time']], 
                            [trade['take_profit'], trade['take_profit']], 
                            'g--', alpha=0.3, linewidth=0.5)
        
        ax1.set_title(f'{symbol} - 9 & 20 EMA Pullback with Strong Wick Rejection', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
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
            
            # Add statistics
            ax3.text(0.02, 0.98, f"Avg Win: {results['avg_win']*100:.2f}%\nAvg Loss: {results['avg_loss']*100:.2f}%",
                     transform=ax3.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax3.set_title('Trade P&L Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('P&L (%)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{symbol}_strict_backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_results(self, results: Dict, symbol: str = "BTCUSDT"):
        """Print detailed backtest results"""
        print("\n" + "="*60)
        print(f"STRICT BACKTEST RESULTS - {symbol}")
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
        print(f"Avg Trade Duration: {results['avg_trade_duration']:.1f} hours")
        
        print(f"\nTrade Statistics:")
        print(f"Average Win: {results['avg_win']:.2%}")
        print(f"Average Loss: {results['avg_loss']:.2%}")
        print(f"Average R:R: {results['avg_rr']:.2f}")
        print(f"Largest Win: {results['largest_win']:.2%}")
        print(f"Largest Loss: {results['largest_loss']:.2%}")
        
        # Print detailed trade log
        if not results['trades'].empty:
            print(f"\nDetailed Trade Log:")
            print("-" * 100)
            trades_df = results['trades'].copy()
            trades_df['pnl_pct'] = trades_df['pnl'] * 100
            trades_df['rr_ratio'] = trades_df['rr_ratio'].round(2)
            
            display_cols = ['entry_time', 'type', 'entry_price', 'stop_loss', 'take_profit', 
                          'exit_time', 'exit_price', 'exit_reason', 'pnl_pct', 'rr_ratio']
            
            print(trades_df[display_cols].to_string(index=False, float_format='%.4f'))
        
        print("\n" + "="*60)


def generate_realistic_crypto_data(symbol: str = "BTCUSDT", days: int = 365) -> pd.DataFrame:
    """Generate realistic crypto data with proper trend and volatility characteristics"""
    print(f"Generating realistic crypto data for {symbol} ({days} days)...")
    
    # Create date range
    start_date = datetime.now() - timedelta(days=days)
    dates = pd.date_range(start=start_date, periods=days*24, freq='1H')  # 1-hour candles
    
    # Base price with realistic crypto characteristics
    base_prices = {
        "BTCUSDT": 45000,
        "ETHUSDT": 2500,
        "SOLUSDT": 100
    }
    base_price = base_prices.get(symbol, 50000)
    
    np.random.seed(42)
    
    # Generate price with trend, cycles, and volatility
    time_steps = len(dates)
    
    # Long-term trend (crypto tends to have upward bias with corrections)
    trend = np.linspace(0, 0.15, time_steps)  # 15% uptrend over period
    
    # Add market cycles (bull/bear phases)
    cycle1 = 0.1 * np.sin(2 * np.pi * np.arange(time_steps) / (time_steps * 0.3))  # 30% cycle
    cycle2 = 0.05 * np.sin(2 * np.pi * np.arange(time_steps) / (time_steps * 0.1))  # 10% cycle
    
    # Add random walk with volatility clustering
    volatility = np.random.lognormal(0, 0.5, time_steps)
    volatility = np.convolve(volatility, np.ones(24)/24, mode='same')  # Smooth volatility
    volatility = volatility / np.mean(volatility)  # Normalize
    
    random_walk = np.cumsum(np.random.normal(0, 0.015, time_steps) * volatility)
    
    # Combine all components
    price_multiplier = 1 + trend + cycle1 + cycle2 + random_walk * 0.1
    prices = base_price * price_multiplier
    
    # Ensure no negative prices
    prices = np.maximum(prices, base_price * 0.1)
    
    # Generate OHLC data with realistic crypto characteristics
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Crypto has higher volatility during certain hours
        hour_volatility = 1 + 0.3 * np.sin(2 * np.pi * i / 24)  # Intraday volatility pattern
        
        # Calculate OHLC with realistic spreads
        volatility_factor = 0.02 * hour_volatility * volatility[i]
        
        high = close * (1 + abs(np.random.normal(0, volatility_factor)))
        low = close * (1 - abs(np.random.normal(0, volatility_factor)))
        
        if i > 0:
            prev_close = data[-1]['close']
            # Ensure continuity with previous close
            gap = np.random.normal(0, 0.005)  # Small gaps possible in crypto
            open_price = prev_close * (1 + gap)
            open_price = max(low, min(high, open_price))
        else:
            open_price = close
        
        # Generate volume (crypto has distinct volume patterns)
        base_volume = 1000000 if symbol == "BTCUSDT" else 500000
        volume_multiplier = 1 + abs(np.random.normal(0, 1))  # High volume variance
        volume = base_volume * volume_multiplier * volatility[i]
        
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
    """Main function to run the strict backtest"""
    print("="*60)
    print("9 & 20 EMA Pullback with Strong Wick Rejection - STRICT")
    print("Rule-Based Backtesting Engine")
    print("="*60)
    
    # Strategy configuration (strict adherence to rules)
    config = {
        'ema_fast': 9,
        'ema_slow': 20,
        'adx_period': 14,
        'adx_threshold': 20,
        'lower_wick_ratio': 0.5,      # 50% of candle range
        'upper_wick_ratio': 0.5,      # 50% of candle range
        'wick_body_multiplier': 2.0,   # wick >= 2x body
        'ema_distance_threshold': 0.001,
        'risk_reward_ratio': 2.0,
        'move_to_breakeven_at_rr': 1.0,
        'close_partial_at_rr': 2.0,
        'partial_close_percentage': 0.5,
        'ema_spread_lookback': 3,
        'ema_flat_lookback': 5,
        'ema_flat_threshold': 0.0001,
        'max_trades_per_day': 2,
        'swing_lookback': 10
    }
    
    # Initialize strategy
    strategy = EMAPullbackStrict(config)
    
    # Test multiple symbols
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    all_results = {}
    
    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"Testing {symbol}")
        print(f"{'='*50}")
        
        # Generate realistic data
        data = generate_realistic_crypto_data(symbol, days=365)
        
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
    print("STRICT STRATEGY SUMMARY COMPARISON")
    print(f"{'='*60}")
    
    summary_data = []
    for symbol, results in all_results.items():
        summary_data.append({
            'Symbol': symbol,
            'Trades': results['total_trades'],
            'Win Rate': f"{results['win_rate']:.1%}",
            'Profit Factor': f"{results['profit_factor']:.2f}",
            'Return': f"{results['total_return']:.1%}",
            'Max DD': f"{results['max_drawdown']:.1%}",
            'Sharpe': f"{results['sharpe_ratio']:.2f}",
            'Avg RR': f"{results['avg_rr']:.2f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    print(f"\n{'='*60}")
    print("Strict Rule-Based Backtesting Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
