"""
9 & 20 EMA Pullback with Wick Rejection + Momentum Break Strategy
Strict rule-based implementation with next-candle breakout entries
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

class EMAPullbackMomentum:
    """
    9 & 20 EMA Pullback with Wick Rejection + Momentum Break Strategy
    Strict implementation with next-candle breakout entries
    """
    
    def __init__(self, config: Dict):
        """
        Initialize strategy with configuration parameters
        
        Args:
            config: Dictionary containing strategy parameters
        """
        # Core indicators
        self.ema_fast = config.get('ema_fast', 9)
        self.ema_slow = config.get('ema_slow', 20)
        self.adx_period = config.get('adx_period', 14)
        self.adx_threshold = config.get('adx_threshold', 25)
        
        # Wick rejection parameters
        self.lower_wick_ratio = config.get('lower_wick_ratio', 0.5)  # 50% of range
        self.upper_wick_ratio = config.get('upper_wick_ratio', 0.5)  # 50% of range
        self.wick_body_multiplier = config.get('wick_body_multiplier', 2.0)  # wick >= 2x body
        
        # Momentum confirmation
        self.momentum_close_threshold = config.get('momentum_close_threshold', 0.3)  # Top/bottom 30%
        
        # EMA separation filter
        self.ema_distance_threshold = config.get('ema_distance_threshold', 0.0015)
        
        # Risk management
        self.risk_reward_ratio = config.get('risk_reward_ratio', 2.5)  # Default 1:2.5
        self.partial_close_enabled = config.get('partial_close_enabled', False)
        self.partial_close_rr = config.get('partial_close_rr', 2.0)
        self.partial_close_percentage = config.get('partial_close_percentage', 0.5)
        
        # Trend analysis
        self.ema_spread_lookback = config.get('ema_spread_lookback', 3)
        self.ema_flat_lookback = config.get('ema_flat_lookback', 5)
        self.ema_flat_threshold = config.get('ema_flat_threshold', 0.0001)
        
        # Position management
        self.max_trades_per_day = config.get('max_trades_per_day', 2)
        self.max_consecutive_losses = config.get('max_consecutive_losses', 2)
        self.swing_lookback = config.get('swing_lookback', 10)
        
        # Session filter
        self.london_session_start = config.get('london_session_start', 8)  # 8:00 UTC
        self.london_session_end = config.get('london_session_end', 17)     # 17:00 UTC
        self.ny_session_start = config.get('ny_session_start', 13)         # 13:00 UTC
        self.ny_session_end = config.get('ny_session_end', 22)             # 22:00 UTC
        
        # Data storage
        self.trades = []
        self.equity_curve = []
        self.positions = {}
        self.pending_signals = []  # Store rejection candles for breakout entries
        
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
        
        # Close position in range (for momentum confirmation)
        if candle_range > 0:
            close_position = (close_price - low_price) / candle_range
        else:
            close_position = 0.5
        
        return {
            'range': candle_range,
            'body': body,
            'upper_wick': upper_wick,
            'lower_wick': lower_wick,
            'upper_wick_ratio': upper_wick_ratio,
            'lower_wick_ratio': lower_wick_ratio,
            'upper_wick_body_ratio': upper_wick_body_ratio,
            'lower_wick_body_ratio': lower_wick_body_ratio,
            'close_position': close_position
        }
    
    def check_ema_touch(self, candle: pd.Series, ema_fast: float, ema_slow: float) -> Tuple[bool, Optional[float]]:
        """
        Check if candle touches/dips into either EMA
        
        Returns:
            Tuple of (touches_ema, touched_ema_value)
        """
        low_price = candle['low']
        high_price = candle['high']
        
        # Check if low touches or dips into EMAs (for buy setup)
        if low_price <= ema_fast <= high_price:
            return True, ema_fast
        elif low_price <= ema_slow <= high_price:
            return True, ema_slow
        
        return False, None
    
    def check_ema_touch_sell(self, candle: pd.Series, ema_fast: float, ema_slow: float) -> Tuple[bool, Optional[float]]:
        """
        Check if candle touches/rises into either EMA (for sell setup)
        
        Returns:
            Tuple of (touches_ema, touched_ema_value)
        """
        low_price = candle['low']
        high_price = candle['high']
        
        # Check if high touches or rises into EMAs (for sell setup)
        if low_price <= ema_fast <= high_price:
            return True, ema_fast
        elif low_price <= ema_slow <= high_price:
            return True, ema_slow
        
        return False, None
    
    def is_trading_session(self, timestamp: datetime) -> bool:
        """Check if timestamp is within allowed trading sessions"""
        hour = timestamp.hour
        
        # London session: 8:00 - 17:00 UTC
        london_active = self.london_session_start <= hour < self.london_session_end
        
        # New York session: 13:00 - 22:00 UTC
        ny_active = self.ny_session_start <= hour < self.ny_session_end
        
        return london_active or ny_active
    
    def check_buy_setup(self, data: pd.DataFrame, i: int) -> Optional[Dict]:
        """
        Check if buy setup conditions are met at index i
        Returns signal data if setup is valid, None otherwise
        """
        if i < max(self.ema_slow, self.adx_period, self.ema_spread_lookback, self.ema_flat_lookback):
            return None
            
        current = data.iloc[i]
        prev = data.iloc[i-1]
        current_time = current.name
        
        # Session filter
        if not self.is_trading_session(current_time):
            return None
        
        # 1. Trend Direction
        if not (current['close'] > current[f'ema_{self.ema_fast}'] and 
                current['close'] > current[f'ema_{self.ema_slow}'] and
                current[f'ema_{self.ema_fast}'] > current[f'ema_{self.ema_slow}']):
            return None
        
        # 2. Trend Strength (at least one condition)
        adx_strong = current['adx'] > self.adx_threshold
        ema_spread_increasing = self.check_ema_spread_increasing(data, i)
        
        if not (adx_strong or ema_spread_increasing):
            return None
        
        # 3. EMA Separation Filter
        ema_distance = abs(current[f'ema_{self.ema_fast}'] - current[f'ema_{self.ema_slow}']) / current['close']
        if ema_distance <= self.ema_distance_threshold:
            return None
        
        # 4. Pullback Condition (current or previous candle)
        current_touch, current_touched_ema = self.check_ema_touch(
            current, current[f'ema_{self.ema_fast}'], current[f'ema_{self.ema_slow}']
        )
        prev_touch, prev_touched_ema = self.check_ema_touch(
            prev, prev[f'ema_{self.ema_fast}'], prev[f'ema_{self.ema_slow}']
        )
        
        if not (current_touch or prev_touch):
            return None
        
        # Use the candle that touched EMA for analysis
        analysis_candle = current if current_touch else prev
        touched_ema = current_touched_ema if current_touch else prev_touched_ema
        
        # 5. Strong Wick Rejection
        wick_metrics = self.calculate_wick_metrics(analysis_candle)
        
        if not (wick_metrics['lower_wick_ratio'] >= self.lower_wick_ratio and
                wick_metrics['lower_wick_body_ratio'] >= self.wick_body_multiplier):
            return None
        
        # 6. Momentum Confirmation
        if not (current['close'] > current['open'] and  # Bullish close
                wick_metrics['close_position'] >= (1 - self.momentum_close_threshold)):  # Close in top 30%
            return None
        
        # 7. Check for abnormal candle range (avoid late entries)
        avg_range = data['high'].rolling(20).mean() - data['low'].rolling(20).mean()
        if wick_metrics['range'] > avg_range.iloc[i] * 2.5:  # More than 2.5x average range
            return None
        
        # Global Filters
        if self.check_ema_flat(data, i):
            return None
        
        # Check max trades per day
        current_date = current_time.date()
        trades_today = len([t for t in self.trades if t['entry_time'].date() == current_date])
        if trades_today >= self.max_trades_per_day:
            return None
        
        # Check consecutive losses
        recent_trades = [t for t in self.trades if t['entry_time'].date() == current_date and t['status'] == 'closed']
        if len(recent_trades) >= self.max_consecutive_losses:
            last_trades = recent_trades[-self.max_consecutive_losses:]
            if all(t['pnl'] < 0 for t in last_trades):
                return None
        
        # Return signal data for breakout entry on next candle
        return {
            'type': 'BUY',
            'setup_time': current_time,
            'rejection_candle': analysis_candle.name,
            'rejection_high': analysis_candle['high'],
            'rejection_low': analysis_candle['low'],
            'rejection_close': analysis_candle['close'],
            'touched_ema': touched_ema,
            'setup_price': current['close']
        }
    
    def check_sell_setup(self, data: pd.DataFrame, i: int) -> Optional[Dict]:
        """
        Check if sell setup conditions are met at index i
        Returns signal data if setup is valid, None otherwise
        """
        if i < max(self.ema_slow, self.adx_period, self.ema_spread_lookback, self.ema_flat_lookback):
            return None
            
        current = data.iloc[i]
        prev = data.iloc[i-1]
        current_time = current.name
        
        # Session filter
        if not self.is_trading_session(current_time):
            return None
        
        # 1. Trend Direction
        if not (current['close'] < current[f'ema_{self.ema_fast}'] and 
                current['close'] < current[f'ema_{self.ema_slow}'] and
                current[f'ema_{self.ema_fast}'] < current[f'ema_{self.ema_slow}']):
            return None
        
        # 2. Trend Strength
        adx_strong = current['adx'] > self.adx_threshold
        ema_spread_increasing = self.check_ema_spread_increasing(data, i)
        
        if not (adx_strong or ema_spread_increasing):
            return None
        
        # 3. EMA Separation Filter
        ema_distance = abs(current[f'ema_{self.ema_fast}'] - current[f'ema_{self.ema_slow}']) / current['close']
        if ema_distance <= self.ema_distance_threshold:
            return None
        
        # 4. Pullback Condition
        current_touch, current_touched_ema = self.check_ema_touch_sell(
            current, current[f'ema_{self.ema_fast}'], current[f'ema_{self.ema_slow}']
        )
        prev_touch, prev_touched_ema = self.check_ema_touch_sell(
            prev, prev[f'ema_{self.ema_fast}'], prev[f'ema_{self.ema_slow}']
        )
        
        if not (current_touch or prev_touch):
            return None
        
        # Use the candle that touched EMA for analysis
        analysis_candle = current if current_touch else prev
        touched_ema = current_touched_ema if current_touch else prev_touched_ema
        
        # 5. Strong Wick Rejection
        wick_metrics = self.calculate_wick_metrics(analysis_candle)
        
        if not (wick_metrics['upper_wick_ratio'] >= self.upper_wick_ratio and
                wick_metrics['upper_wick_body_ratio'] >= self.wick_body_multiplier):
            return None
        
        # 6. Momentum Confirmation
        if not (current['close'] < current['open'] and  # Bearish close
                wick_metrics['close_position'] <= self.momentum_close_threshold):  # Close in bottom 30%
            return None
        
        # Check for abnormal candle range
        avg_range = data['high'].rolling(20).mean() - data['low'].rolling(20).mean()
        if wick_metrics['range'] > avg_range.iloc[i] * 2.5:
            return None
        
        # Global Filters
        if self.check_ema_flat(data, i):
            return None
        
        # Check max trades per day
        current_date = current_time.date()
        trades_today = len([t for t in self.trades if t['entry_time'].date() == current_date])
        if trades_today >= self.max_trades_per_day:
            return None
        
        # Check consecutive losses
        recent_trades = [t for t in self.trades if t['entry_time'].date() == current_date and t['status'] == 'closed']
        if len(recent_trades) >= self.max_consecutive_losses:
            last_trades = recent_trades[-self.max_consecutive_losses:]
            if all(t['pnl'] < 0 for t in last_trades):
                return None
        
        # Return signal data for breakout entry on next candle
        return {
            'type': 'SELL',
            'setup_time': current_time,
            'rejection_candle': analysis_candle.name,
            'rejection_high': analysis_candle['high'],
            'rejection_low': analysis_candle['low'],
            'rejection_close': analysis_candle['close'],
            'touched_ema': touched_ema,
            'setup_price': current['close']
        }
    
    def check_breakout_entry(self, data: pd.DataFrame, i: int) -> Optional[Dict]:
        """
        Check if current candle breaks out from pending signal
        Returns trade data if entry is triggered, None otherwise
        """
        current = data.iloc[i]
        current_time = current.name
        
        # Check pending signals
        signals_to_remove = []
        
        for j, signal in enumerate(self.pending_signals):
            # Check if signal is too old (more than 5 candles)
            if (current_time - signal['setup_time']).total_seconds() / 3600 > 5:
                signals_to_remove.append(j)
                continue
            
            # Check breakout condition
            if signal['type'] == 'BUY':
                # Buy: Break above rejection candle high
                if current['high'] > signal['rejection_high']:
                    # Enter at breakout price (or close if we want to be more conservative)
                    entry_price = min(current['open'], signal['rejection_high'] + 0.0001)  # Small buffer
                    stop_loss = signal['rejection_low']
                    
                    # Calculate take profit
                    risk = entry_price - stop_loss
                    take_profit = entry_price + (risk * self.risk_reward_ratio)
                    
                    trade_data = {
                        'type': 'BUY',
                        'entry_time': current_time,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'signal': signal
                    }
                    
                    signals_to_remove.append(j)
                    return trade_data
                    
            else:  # SELL
                # Sell: Break below rejection candle low
                if current['low'] < signal['rejection_low']:
                    entry_price = max(current['open'], signal['rejection_low'] - 0.0001)  # Small buffer
                    stop_loss = signal['rejection_high']
                    
                    # Calculate take profit
                    risk = stop_loss - entry_price
                    take_profit = entry_price - (risk * self.risk_reward_ratio)
                    
                    trade_data = {
                        'type': 'SELL',
                        'entry_time': current_time,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'signal': signal
                    }
                    
                    signals_to_remove.append(j)
                    return trade_data
        
        # Remove expired or triggered signals
        for j in reversed(signals_to_remove):
            self.pending_signals.pop(j)
        
        return None
    
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
                    if self.partial_close_enabled:
                        # Partial close logic
                        if not trade.get('partial_closed', False):
                            rr_ratio = (current_price - entry_price) / (entry_price - stop_loss)
                            if rr_ratio >= self.partial_close_rr:
                                trade['partial_closed'] = True
                                trade['partial_close_price'] = current_price
                                trade['partial_close_time'] = current_time
                                # For simplicity, we'll close the full position
                                trade['exit_price'] = current_price
                                trade['exit_time'] = current_time
                                trade['exit_reason'] = 'partial_take_profit'
                                trade['status'] = 'closed'
                                trades_to_close.append(trade_id)
                    else:
                        # Full position close
                        trade['exit_price'] = take_profit
                        trade['exit_time'] = current_time
                        trade['exit_reason'] = 'take_profit'
                        trade['status'] = 'closed'
                        trades_to_close.append(trade_id)
                        
                # Trail stop loss using EMA 20 or swing lows (for remaining position)
                else:
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
                    if self.partial_close_enabled:
                        if not trade.get('partial_closed', False):
                            rr_ratio = (entry_price - current_price) / (stop_loss - entry_price)
                            if rr_ratio >= self.partial_close_rr:
                                trade['partial_closed'] = True
                                trade['partial_close_price'] = current_price
                                trade['partial_close_time'] = current_time
                                trade['exit_price'] = current_price
                                trade['exit_time'] = current_time
                                trade['exit_reason'] = 'partial_take_profit'
                                trade['status'] = 'closed'
                                trades_to_close.append(trade_id)
                    else:
                        trade['exit_price'] = take_profit
                        trade['exit_time'] = current_time
                        trade['exit_reason'] = 'take_profit'
                        trade['status'] = 'closed'
                        trades_to_close.append(trade_id)
                        
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
        print(f"Starting momentum breakout backtest with {len(data)} candles...")
        
        # Reset state
        self.trades = []
        self.positions = {}
        self.equity_curve = []
        self.pending_signals = []
        
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
            
            # Check for breakout entries from pending signals
            if len(self.positions) == 0:  # Only one trade at a time
                breakout_trade = self.check_breakout_entry(data, i)
                if breakout_trade:
                    # Open the trade
                    trade = {
                        'trade_id': trade_id_counter,
                        'entry_time': breakout_trade['entry_time'],
                        'entry_price': breakout_trade['entry_price'],
                        'stop_loss': breakout_trade['stop_loss'],
                        'take_profit': breakout_trade['take_profit'],
                        'type': breakout_trade['type'],
                        'status': 'open',
                        'size': 1.0,
                        'signal': breakout_trade['signal']
                    }
                    
                    self.positions[trade_id_counter] = trade
                    trade_id_counter += 1
                    continue
            
            # Check for new setup signals (only if no open positions and no pending signals)
            if len(self.positions) == 0 and len(self.pending_signals) == 0:
                # Check buy setup
                buy_signal = self.check_buy_setup(data, i)
                if buy_signal:
                    self.pending_signals.append(buy_signal)
                    continue
                
                # Check sell setup
                sell_signal = self.check_sell_setup(data, i)
                if sell_signal:
                    self.pending_signals.append(sell_signal)
                    continue
            
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
                'total_trades': len(self.trades),
                'pending_signals': len(self.pending_signals)
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
        
        # Calculate PnL and other metrics
        trades_df['pnl'] = trades_df.apply(self.calculate_pnl, axis=1)
        trades_df['rr_ratio'] = trades_df.apply(
            lambda x: abs(x['pnl']) / (abs(x['entry_price'] - x['stop_loss']) / x['entry_price']), 
            axis=1
        )
        trades_df['duration'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 3600  # hours
        
        # Add signal information
        if 'signal' in trades_df.columns:
            trades_df['setup_time'] = trades_df['signal'].apply(lambda x: x['setup_time'] if pd.notna(x) else None)
            trades_df['rejection_high'] = trades_df['signal'].apply(lambda x: x['rejection_high'] if pd.notna(x) else None)
            trades_df['rejection_low'] = trades_df['signal'].apply(lambda x: x['rejection_low'] if pd.notna(x) else None)
        
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
        """Plot comprehensive backtest results with momentum breakout visualization"""
        fig, axes = plt.subplots(4, 1, figsize=(16, 16))
        
        # Plot 1: Price chart with EMAs and signals
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
                           color='green', marker='^', s=150, label='Buy Entry', zorder=5, alpha=0.8)
                ax1.scatter(buy_trades['exit_time'], buy_trades['exit_price'], 
                           color='red', marker='v', s=150, label='Buy Exit', zorder=5, alpha=0.8)
                
                # Plot rejection candles and breakout levels for buys
                for _, trade in buy_trades.iterrows():
                    if 'signal' in trade and pd.notna(trade['signal']):
                        signal = trade['signal']
                        # Mark rejection candle
                        ax1.scatter(signal['rejection_candle'], signal['rejection_close'], 
                                   color='orange', marker='o', s=80, alpha=0.6, zorder=4)
                        # Plot breakout level
                        ax1.axhline(y=signal['rejection_high'], color='green', 
                                   linestyle=':', alpha=0.5, linewidth=1)
            
            # Sell entries and exits
            sell_trades = trades_df[trades_df['type'] == 'SELL']
            if not sell_trades.empty:
                ax1.scatter(sell_trades['entry_time'], sell_trades['entry_price'], 
                           color='red', marker='v', s=150, label='Sell Entry', zorder=5, alpha=0.8)
                ax1.scatter(sell_trades['exit_time'], sell_trades['exit_price'], 
                           color='green', marker='^', s=150, label='Sell Exit', zorder=5, alpha=0.8)
                
                # Plot rejection candles and breakout levels for sells
                for _, trade in sell_trades.iterrows():
                    if 'signal' in trade and pd.notna(trade['signal']):
                        signal = trade['signal']
                        # Mark rejection candle
                        ax1.scatter(signal['rejection_candle'], signal['rejection_close'], 
                                   color='orange', marker='o', s=80, alpha=0.6, zorder=4)
                        # Plot breakout level
                        ax1.axhline(y=signal['rejection_low'], color='red', 
                                   linestyle=':', alpha=0.5, linewidth=1)
        
        ax1.set_title(f'{symbol} - EMA Pullback with Momentum Breakout', fontsize=14, fontweight='bold')
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
            ax3.text(0.02, 0.98, f"Avg Win: {results['avg_win']*100:.2f}%\nAvg Loss: {results['avg_loss']*100:.2f}%\nAvg RR: {results['avg_rr']:.2f}",
                     transform=ax3.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax3.set_title('Trade P&L Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('P&L (%)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Session Analysis and Trade Timing
        ax4 = axes[3]
        if not trades_df.empty:
            # Extract hour from entry time
            trades_df['hour'] = trades_df['entry_time'].dt.hour
            
            # Plot trades by hour
            hourly_trades = trades_df.groupby('hour').size()
            hourly_pnl = trades_df.groupby('hour')['pnl'].mean()
            
            ax4_bar = ax4.bar(hourly_trades.index - 0.2, hourly_trades.values, 0.4, 
                             label='Trade Count', alpha=0.7, color='blue')
            ax4_pnl = ax4.bar(hourly_pnl.index + 0.2, hourly_pnl.values * 100, 0.4, 
                             label='Avg P&L (%)', alpha=0.7, color='green')
            
            # Mark trading sessions
            ax4.axvspan(self.london_session_start, self.london_session_end, 
                       alpha=0.2, color='yellow', label='London Session')
            ax4.axvspan(self.ny_session_start, self.ny_session_end, 
                       alpha=0.2, color='cyan', label='NY Session')
            
            ax4.set_xlabel('Hour (UTC)')
            ax4.set_ylabel('Trade Count / P&L (%)')
            ax4.set_title('Trade Distribution by Hour & Session', fontsize=14, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{symbol}_momentum_backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_results(self, results: Dict, symbol: str = "BTCUSDT"):
        """Print detailed backtest results"""
        print("\n" + "="*60)
        print(f"MOMENTUM BREAKOUT RESULTS - {symbol}")
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
        
        # Print detailed trade log with signal information
        if not results['trades'].empty:
            print(f"\nDetailed Trade Log:")
            print("-" * 120)
            trades_df = results['trades'].copy()
            trades_df['pnl_pct'] = trades_df['pnl'] * 100
            trades_df['rr_ratio'] = trades_df['rr_ratio'].round(2)
            
            display_cols = ['entry_time', 'type', 'entry_price', 'stop_loss', 'take_profit', 
                          'exit_time', 'exit_price', 'exit_reason', 'pnl_pct', 'rr_ratio']
            
            if 'setup_time' in trades_df.columns:
                display_cols.insert(1, 'setup_time')
            if 'rejection_high' in trades_df.columns:
                display_cols.extend(['rejection_high', 'rejection_low'])
            
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
    """Main function to run the momentum breakout backtest"""
    print("="*60)
    print("9 & 20 EMA Pullback with Wick Rejection + Momentum Break")
    print("Strict Rule-Based Backtesting Engine")
    print("="*60)
    
    # Strategy configuration (strict adherence to rules)
    config = {
        'ema_fast': 9,
        'ema_slow': 20,
        'adx_period': 14,
        'adx_threshold': 25,
        'lower_wick_ratio': 0.5,      # 50% of candle range
        'upper_wick_ratio': 0.5,      # 50% of candle range
        'wick_body_multiplier': 2.0,   # wick >= 2x body
        'momentum_close_threshold': 0.3,  # Top/bottom 30% close
        'ema_distance_threshold': 0.0015,
        'risk_reward_ratio': 2.5,      # Default 1:2.5
        'partial_close_enabled': False, # Option B disabled by default
        'partial_close_rr': 2.0,
        'partial_close_percentage': 0.5,
        'ema_spread_lookback': 3,
        'ema_flat_lookback': 5,
        'ema_flat_threshold': 0.0001,
        'max_trades_per_day': 2,
        'max_consecutive_losses': 2,
        'swing_lookback': 10,
        'london_session_start': 8,    # 8:00 UTC
        'london_session_end': 17,     # 17:00 UTC
        'ny_session_start': 13,       # 13:00 UTC
        'ny_session_end': 22          # 22:00 UTC
    }
    
    # Initialize strategy
    strategy = EMAPullbackMomentum(config)
    
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
    print("MOMENTUM BREAKOUT STRATEGY SUMMARY")
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
    print("Momentum Breakout Backtesting Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
