"""
9 & 20 EMA Pullback with Wick Rejection + Breakout Confirmation
Production-level realistic backtesting system with execution costs
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

class EMAPullbackRealistic:
    """
    Production-level realistic backtesting system
    Includes execution costs, slippage, and strict no-lookahead bias
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
        self.adx_threshold = config.get('adx_threshold', 20)
        
        # Wick rejection parameters
        self.lower_wick_ratio = config.get('lower_wick_ratio', 0.5)  # 50% of range
        self.upper_wick_ratio = config.get('upper_wick_ratio', 0.5)  # 50% of range
        self.wick_body_multiplier = config.get('wick_body_multiplier', 1.5)  # wick >= 1.5x body
        
        # Confirmation parameters
        self.momentum_close_threshold = config.get('momentum_close_threshold', 0.4)  # Top/bottom 40%
        
        # EMA separation filter
        self.ema_distance_threshold = config.get('ema_distance_threshold', 0.001)
        
        # Risk management
        self.risk_reward_ratio = config.get('risk_reward_ratio', 2.0)  # Strict 1:2 minimum
        
        # Trend analysis
        self.ema_flat_lookback = config.get('ema_flat_lookback', 5)
        self.ema_flat_threshold = config.get('ema_flat_threshold', 0.0001)
        
        # Position management
        self.max_trades_per_day = config.get('max_trades_per_day', 2)
        self.max_consecutive_losses = config.get('max_consecutive_losses', 2)
        self.swing_lookback = config.get('swing_lookback', 10)
        
        # REALISTIC EXECUTION COSTS
        self.trading_fee = config.get('trading_fee', 0.0005)  # 0.05% per trade
        self.slippage = config.get('slippage', 0.0002)  # 0.02% per trade
        self.total_cost_per_trade = self.trading_fee + self.slippage
        
        # Data storage
        self.trades = []
        self.equity_curve = []
        self.positions = {}
        self.pending_signals = []  # Store confirmed setups for next candle execution
        
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
    
    def check_buy_setup(self, data: pd.DataFrame, i: int) -> Optional[Dict]:
        """
        Check if buy setup conditions are met at index i
        Uses ONLY closed candle data for signal generation (no lookahead bias)
        Returns signal data if setup is valid, None otherwise
        """
        if i < max(self.ema_slow, self.adx_period, self.ema_flat_lookback):
            return None
            
        current = data.iloc[i]
        current_time = current.name
        
        # 1. Trend Direction (using closed candle data only)
        if not (current['close'] > current[f'ema_{self.ema_fast}'] and 
                current['close'] > current[f'ema_{self.ema_slow}'] and
                current[f'ema_{self.ema_fast}'] > current[f'ema_{self.ema_slow}']):
            return None
        
        # 2. Trend Strength
        if current['adx'] <= self.adx_threshold:
            return None
        
        # 3. EMA Separation Filter
        ema_distance = abs(current[f'ema_{self.ema_fast}'] - current[f'ema_{self.ema_slow}']) / current['close']
        if ema_distance <= self.ema_distance_threshold:
            return None
        
        # 4. Pullback Condition - candle LOW touches EMA
        touch, touched_ema = self.check_ema_touch(
            current, current[f'ema_{self.ema_fast}'], current[f'ema_{self.ema_slow}']
        )
        
        if not touch:
            return None
        
        # 5. Strong Wick Rejection
        wick_metrics = self.calculate_wick_metrics(current)
        
        if not (wick_metrics['lower_wick_ratio'] >= self.lower_wick_ratio and
                wick_metrics['lower_wick_body_ratio'] >= self.wick_body_multiplier):
            return None
        
        # 6. Confirmation
        if not (current['close'] > current['open'] and  # Bullish close
                wick_metrics['close_position'] >= (1 - self.momentum_close_threshold)):  # Close in top 40%
            return None
        
        # 7. Check for abnormal candle range (avoid late entries)
        avg_range = data['high'].rolling(20).mean() - data['low'].rolling(20).mean()
        if wick_metrics['range'] > avg_range.iloc[i] * 2:  # More than 2x average range
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
        
        # Return signal data for execution on next candle
        return {
            'type': 'BUY',
            'setup_time': current_time,
            'setup_candle': current.name,
            'setup_high': current['high'],
            'setup_low': current['low'],
            'setup_close': current['close'],
            'touched_ema': touched_ema,
            'entry_price': current['close']  # Will use next candle open or breakout
        }
    
    def check_sell_setup(self, data: pd.DataFrame, i: int) -> Optional[Dict]:
        """
        Check if sell setup conditions are met at index i
        Uses ONLY closed candle data for signal generation
        Returns signal data if setup is valid, None otherwise
        """
        if i < max(self.ema_slow, self.adx_period, self.ema_flat_lookback):
            return None
            
        current = data.iloc[i]
        current_time = current.name
        
        # 1. Trend Direction
        if not (current['close'] < current[f'ema_{self.ema_fast}'] and 
                current['close'] < current[f'ema_{self.ema_slow}'] and
                current[f'ema_{self.ema_fast}'] < current[f'ema_{self.ema_slow}']):
            return None
        
        # 2. Trend Strength
        if current['adx'] <= self.adx_threshold:
            return None
        
        # 3. EMA Separation Filter
        ema_distance = abs(current[f'ema_{self.ema_fast}'] - current[f'ema_{self.ema_slow}']) / current['close']
        if ema_distance <= self.ema_distance_threshold:
            return None
        
        # 4. Pullback Condition - candle HIGH touches EMA
        touch, touched_ema = self.check_ema_touch_sell(
            current, current[f'ema_{self.ema_fast}'], current[f'ema_{self.ema_slow}']
        )
        
        if not touch:
            return None
        
        # 5. Strong Wick Rejection
        wick_metrics = self.calculate_wick_metrics(current)
        
        if not (wick_metrics['upper_wick_ratio'] >= self.upper_wick_ratio and
                wick_metrics['upper_wick_body_ratio'] >= self.wick_body_multiplier):
            return None
        
        # 6. Confirmation
        if not (current['close'] < current['open'] and  # Bearish close
                wick_metrics['close_position'] <= self.momentum_close_threshold):  # Close in bottom 40%
            return None
        
        # Check for abnormal candle range
        avg_range = data['high'].rolling(20).mean() - data['low'].rolling(20).mean()
        if wick_metrics['range'] > avg_range.iloc[i] * 2:
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
        
        # Return signal data for execution on next candle
        return {
            'type': 'SELL',
            'setup_time': current_time,
            'setup_candle': current.name,
            'setup_high': current['high'],
            'setup_low': current['low'],
            'setup_close': current['close'],
            'touched_ema': touched_ema,
            'entry_price': current['close']  # Will use next candle open or breakdown
        }
    
    def execute_trade(self, data: pd.DataFrame, i: int, signal: Dict) -> Optional[Dict]:
        """
        Execute trade based on signal and next candle data
        Implements realistic execution with costs and slippage
        """
        current = data.iloc[i]
        current_time = current.name
        
        # Check if signal is too old (more than 3 candles)
        if (current_time - signal['setup_time']).total_seconds() / 3600 > 3:
            return None
        
        # Determine entry price (next candle open or breakout)
        if signal['type'] == 'BUY':
            # Entry at next candle open OR breakout above setup high (whichever occurs first)
            entry_price = max(current['open'], signal['setup_high'] + 0.0001)  # Small buffer for breakout
            stop_loss = signal['setup_low']
        else:  # SELL
            # Entry at next candle open OR breakdown below setup low
            entry_price = min(current['open'], signal['setup_low'] - 0.0001)  # Small buffer for breakdown
            stop_loss = signal['setup_high']
        
        # Apply realistic execution costs
        if signal['type'] == 'BUY':
            # Buy: price moves up due to slippage
            actual_entry_price = entry_price * (1 + self.slippage)
            # Trading fee applied to position size
            effective_entry_price = actual_entry_price * (1 + self.trading_fee)
        else:  # SELL
            # Sell: price moves down due to slippage
            actual_entry_price = entry_price * (1 - self.slippage)
            # Trading fee applied to position size
            effective_entry_price = actual_entry_price * (1 - self.trading_fee)
        
        # Calculate take profit with strict RR
        if signal['type'] == 'BUY':
            risk = effective_entry_price - stop_loss
            take_profit = effective_entry_price + (risk * self.risk_reward_ratio)
        else:  # SELL
            risk = stop_loss - effective_entry_price
            take_profit = effective_entry_price - (risk * self.risk_reward_ratio)
        
        trade_data = {
            'type': signal['type'],
            'entry_time': current_time,
            'signal_time': signal['setup_time'],
            'signal_price': signal['setup_close'],
            'entry_price': effective_entry_price,
            'raw_entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'signal': signal
        }
        
        return trade_data
    
    def update_trades(self, data: pd.DataFrame, i: int):
        """Update open trades with current bar data and realistic exit execution"""
        current_time = data.index[i]
        current = data.iloc[i]
        
        trades_to_close = []
        
        for trade_id, trade in self.positions.items():
            if trade['status'] != 'open':
                continue
                
            entry_price = trade['entry_price']
            stop_loss = trade['stop_loss']
            take_profit = trade['take_profit']
            trade_type = trade['type']
            
            # Check stop loss hit (with slippage)
            if trade_type == 'BUY':
                sl_hit = current['low'] <= stop_loss
                tp_hit = current['high'] >= take_profit
            else:  # SELL
                sl_hit = current['high'] >= stop_loss
                tp_hit = current['low'] <= take_profit
            
            exit_price = None
            exit_reason = None
            
            if sl_hit:
                # Realistic stop loss execution with slippage
                if trade_type == 'BUY':
                    exit_price = stop_loss * (1 - self.slippage) * (1 - self.trading_fee)
                else:  # SELL
                    exit_price = stop_loss * (1 + self.slippage) * (1 + self.trading_fee)
                exit_reason = 'stop_loss'
                
            elif tp_hit:
                # Realistic take profit execution with slippage
                if trade_type == 'BUY':
                    exit_price = take_profit * (1 - self.slippage) * (1 - self.trading_fee)
                else:  # SELL
                    exit_price = take_profit * (1 + self.slippage) * (1 + self.trading_fee)
                exit_reason = 'take_profit'
            
            if exit_price is not None:
                trade['exit_price'] = exit_price
                trade['exit_time'] = current_time
                trade['exit_reason'] = exit_reason
                trade['status'] = 'closed'
                trades_to_close.append(trade_id)
        
        # Close trades
        for trade_id in trades_to_close:
            trade = self.positions[trade_id]
            self.trades.append(trade)
            del self.positions[trade_id]
    
    def calculate_pnl(self, trade: Dict) -> float:
        """Calculate profit/loss for a trade including all costs"""
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
        Run realistic backtest on given data
        Strict no-lookahead bias implementation
        """
        print(f"Starting realistic backtest with {len(data)} candles...")
        print(f"Trading costs: {self.trading_fee*100:.3f}% fee + {self.slippage*100:.3f}% slippage per trade")
        
        # Reset state
        self.trades = []
        self.positions = {}
        self.equity_curve = []
        self.pending_signals = []
        
        # Calculate indicators
        data[f'ema_{self.ema_fast}'] = self.calculate_ema(data['close'], self.ema_fast)
        data[f'ema_{self.ema_slow}'] = self.calculate_ema(data['close'], self.ema_slow)
        data['adx'] = self.calculate_adx(data['high'], data['low'], data['close'], self.adx_period)
        
        # Initialize equity
        current_equity = initial_capital
        trade_id_counter = 0
        
        # Run through each candle (realistic execution)
        for i in range(len(data)):
            current_time = data.index[i]
            
            # Update existing trades
            self.update_trades(data, i)
            
            # Check for trade executions from pending signals
            if len(self.positions) == 0:  # Only one trade at a time
                signals_to_remove = []
                
                for j, signal in enumerate(self.pending_signals):
                    # Execute trade based on next candle data
                    trade_data = self.execute_trade(data, i, signal)
                    if trade_data:
                        # Open trade
                        trade = {
                            'trade_id': trade_id_counter,
                            **trade_data,
                            'status': 'open',
                            'size': 1.0
                        }
                        
                        self.positions[trade_id_counter] = trade
                        trade_id_counter += 1
                        signals_to_remove.append(j)
                        break
                
                # Remove executed signals
                for j in reversed(signals_to_remove):
                    self.pending_signals.pop(j)
            
            # Check for new setup signals (only if no open positions)
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
            # Realistic final exit with costs
            if trade['type'] == 'BUY':
                exit_price = final_price * (1 - self.slippage) * (1 - self.trading_fee)
            else:  # SELL
                exit_price = final_price * (1 + self.slippage) * (1 + self.trading_fee)
                
            trade['exit_price'] = exit_price
            trade['exit_time'] = data.index[-1]
            trade['exit_reason'] = 'end_of_data'
            trade['status'] = 'closed'
            self.trades.append(trade)
        
        # Calculate final results
        results = self.calculate_results(initial_capital)
        
        return results
    
    def calculate_results(self, initial_capital: float) -> Dict:
        """Calculate comprehensive backtest results with cost analysis"""
        
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
                'total_costs': 0,
                'cost_impact': 0,
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
        
        # Calculate costs impact
        trades_df['entry_cost'] = trades_df.apply(
            lambda x: abs(x['entry_price'] - x['raw_entry_price']) / x['raw_entry_price'], axis=1
        )
        trades_df['total_cost'] = self.total_cost_per_trade * 2  # Entry + exit
        trades_df['cost_impact_pct'] = trades_df['total_cost'] * 100
        
        # Add signal information
        if 'signal' in trades_df.columns:
            trades_df['setup_time'] = trades_df['signal'].apply(lambda x: x['setup_time'] if pd.notna(x) else None)
            trades_df['setup_high'] = trades_df['signal'].apply(lambda x: x['setup_high'] if pd.notna(x) else None)
            trades_df['setup_low'] = trades_df['signal'].apply(lambda x: x['setup_low'] if pd.notna(x) else None)
        
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
        
        # Cost analysis
        total_costs = trades_df['total_cost'].sum()
        cost_impact = total_costs / total_trades if total_trades > 0 else 0
        
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
            'total_costs': total_costs,
            'cost_impact': cost_impact,
            'trades': trades_df,
            'equity_curve': equity_df
        }
        
        return results
    
    def plot_results(self, results: Dict, data: pd.DataFrame, symbol: str = "BTCUSDT"):
        """Plot comprehensive realistic backtest results"""
        fig, axes = plt.subplots(4, 1, figsize=(16, 16))
        
        # Plot 1: Price chart with EMAs and realistic signals
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
        
        # Plot trades with cost visualization
        trades_df = results['trades']
        if not trades_df.empty:
            # Buy entries and exits
            buy_trades = trades_df[trades_df['type'] == 'BUY']
            if not buy_trades.empty:
                ax1.scatter(buy_trades['entry_time'], buy_trades['entry_price'], 
                           color='green', marker='^', s=150, label='Buy Entry', zorder=5, alpha=0.8)
                ax1.scatter(buy_trades['exit_time'], buy_trades['exit_price'], 
                           color='red', marker='v', s=150, label='Buy Exit', zorder=5, alpha=0.8)
                
                # Show setup and execution prices for buys
                for _, trade in buy_trades.iterrows():
                    if 'signal' in trade and pd.notna(trade['signal']):
                        signal = trade['signal']
                        # Mark setup candle
                        ax1.scatter(signal['setup_candle'], signal['setup_close'], 
                                   color='orange', marker='o', s=80, alpha=0.6, zorder=4)
                        # Show cost impact
                        if 'raw_entry_price' in trade:
                            ax1.plot([trade['entry_time'], trade['entry_time']], 
                                    [trade['raw_entry_price'], trade['entry_price']], 
                                    'r-', alpha=0.7, linewidth=2, label='Cost Impact' if _ == buy_trades.index[0] else "")
            
            # Sell entries and exits
            sell_trades = trades_df[trades_df['type'] == 'SELL']
            if not sell_trades.empty:
                ax1.scatter(sell_trades['entry_time'], sell_trades['entry_price'], 
                           color='red', marker='v', s=150, label='Sell Entry', zorder=5, alpha=0.8)
                ax1.scatter(sell_trades['exit_time'], sell_trades['exit_price'], 
                           color='green', marker='^', s=150, label='Sell Exit', zorder=5, alpha=0.8)
                
                # Show setup and execution prices for sells
                for _, trade in sell_trades.iterrows():
                    if 'signal' in trade and pd.notna(trade['signal']):
                        signal = trade['signal']
                        # Mark setup candle
                        ax1.scatter(signal['setup_candle'], signal['setup_close'], 
                                   color='orange', marker='o', s=80, alpha=0.6, zorder=4)
                        # Show cost impact
                        if 'raw_entry_price' in trade:
                            ax1.plot([trade['entry_time'], trade['entry_time']], 
                                    [trade['raw_entry_price'], trade['entry_price']], 
                                    'r-', alpha=0.7, linewidth=2)
        
        ax1.set_title(f'{symbol} - Realistic EMA Pullback with Costs', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Equity Curve with cost impact
        ax2 = axes[1]
        equity_df = results['equity_curve']
        ax2.plot(equity_df.index, equity_df['equity'], label='Equity Curve', linewidth=2, color='blue')
        
        # Add drawdown
        peak = equity_df['equity'].expanding().max()
        drawdown = (equity_df['equity'] - peak) / peak * 100
        ax2.fill_between(equity_df.index, drawdown, 0, alpha=0.3, color='red', label='Drawdown %')
        
        # Add cost impact annotation
        if results['total_costs'] > 0:
            cost_text = f"Total Costs: {results['total_costs']:.2f} ({results['cost_impact']*100:.3f}% per trade)"
            ax2.text(0.02, 0.98, cost_text, transform=ax2.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        ax2.set_title('Equity Curve with Cost Impact', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Equity ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Trade Distribution with cost analysis
        ax3 = axes[2]
        if not trades_df.empty:
            wins = trades_df[trades_df['pnl'] > 0]['pnl']
            losses = trades_df[trades_df['pnl'] <= 0]['pnl']
            
            ax3.hist([wins * 100, losses * 100], bins=30, alpha=0.7, 
                    label=['Wins (%)', 'Losses (%)'], color=['green', 'red'])
            ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            
            # Add statistics including costs
            stats_text = (f"Avg Win: {results['avg_win']*100:.2f}%\n"
                        f"Avg Loss: {results['avg_loss']*100:.2f}%\n"
                        f"Avg RR: {results['avg_rr']:.2f}\n"
                        f"Cost/Trade: {results['cost_impact']*100:.3f}%")
            ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax3.set_title('Trade P&L Distribution (After Costs)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('P&L (%)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Cost Analysis and Performance Metrics
        ax4 = axes[3]
        if not trades_df.empty:
            # Create cost breakdown
            cost_data = {
                'Trading Fees': self.trading_fee * 100,
                'Slippage': self.slippage * 100,
                'Total Cost': self.total_cost_per_trade * 100
            }
            
            bars = ax4.bar(cost_data.keys(), cost_data.values(), 
                          color=['blue', 'orange', 'red'], alpha=0.7)
            
            # Add value labels on bars
            for bar, (key, value) in zip(bars, cost_data.items()):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}%', ha='center', va='bottom')
            
            # Add performance metrics text
            metrics_text = (f"Performance Metrics:\n"
                          f"Win Rate: {results['win_rate']:.1%}\n"
                          f"Profit Factor: {results['profit_factor']:.2f}\n"
                          f"Sharpe: {results['sharpe_ratio']:.2f}\n"
                          f"Max DD: {results['max_drawdown']:.1%}")
            ax4.text(0.98, 0.5, metrics_text, transform=ax4.transAxes, 
                    verticalalignment='center', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        ax4.set_title('Cost Breakdown & Performance', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Cost (%)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{symbol}_realistic_backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_results(self, results: Dict, symbol: str = "BTCUSDT"):
        """Print detailed realistic backtest results"""
        print("\n" + "="*60)
        print(f"REALISTIC BACKTEST RESULTS - {symbol}")
        print("="*60)
        
        print(f"Trading Costs: {self.trading_fee*100:.3f}% fee + {self.slippage*100:.3f}% slippage per trade")
        print(f"Total Cost per Trade: {self.total_cost_per_trade*100:.3f}%")
        print()
        
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
        
        print(f"\nCost Analysis:")
        print(f"Total Trading Costs: {results['total_costs']:.4f}")
        print(f"Cost Impact per Trade: {results['cost_impact']*100:.3f}%")
        
        print(f"\nTrade Statistics (After Costs):")
        print(f"Average Win: {results['avg_win']:.2%}")
        print(f"Average Loss: {results['avg_loss']:.2%}")
        print(f"Average R:R: {results['avg_rr']:.2f}")
        print(f"Largest Win: {results['largest_win']:.2%}")
        print(f"Largest Loss: {results['largest_loss']:.2%}")
        
        # Print detailed trade log with cost information
        if not results['trades'].empty:
            print(f"\nDetailed Trade Log (Including Costs):")
            print("-" * 140)
            trades_df = results['trades'].copy()
            trades_df['pnl_pct'] = trades_df['pnl'] * 100
            trades_df['rr_ratio'] = trades_df['rr_ratio'].round(2)
            trades_df['cost_pct'] = trades_df['cost_impact_pct'].round(3)
            
            display_cols = ['signal_time', 'type', 'raw_entry_price', 'entry_price', 'stop_loss', 
                          'take_profit', 'exit_time', 'exit_price', 'exit_reason', 
                          'pnl_pct', 'rr_ratio', 'cost_pct']
            
            if 'setup_time' in trades_df.columns:
                display_cols[0] = 'setup_time'
            
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
    
    # More realistic trend (less dramatic, more sideways periods)
    trend = np.linspace(0, 0.08, time_steps)  # 8% trend over period
    
    # Add market cycles
    cycle1 = 0.06 * np.sin(2 * np.pi * np.arange(time_steps) / (time_steps * 0.4))  # 40% cycle
    cycle2 = 0.03 * np.sin(2 * np.pi * np.arange(time_steps) / (time_steps * 0.15))  # 15% cycle
    
    # Add random walk with volatility clustering
    volatility = np.random.lognormal(0, 0.4, time_steps)
    volatility = np.convolve(volatility, np.ones(24)/24, mode='same')  # Smooth volatility
    volatility = volatility / np.mean(volatility)  # Normalize
    
    random_walk = np.cumsum(np.random.normal(0, 0.012, time_steps) * volatility)
    
    # Combine all components
    price_multiplier = 1 + trend + cycle1 + cycle2 + random_walk * 0.08
    prices = base_price * price_multiplier
    
    # Ensure no negative prices
    prices = np.maximum(prices, base_price * 0.1)
    
    # Generate OHLC data with realistic crypto characteristics
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Crypto has higher volatility during certain hours
        hour_volatility = 1 + 0.25 * np.sin(2 * np.pi * i / 24)  # Intraday volatility pattern
        
        # Calculate OHLC with realistic spreads
        volatility_factor = 0.015 * hour_volatility * volatility[i]
        
        high = close * (1 + abs(np.random.normal(0, volatility_factor)))
        low = close * (1 - abs(np.random.normal(0, volatility_factor)))
        
        if i > 0:
            prev_close = data[-1]['close']
            # Ensure continuity with previous close
            gap = np.random.normal(0, 0.003)  # Smaller gaps for realism
            open_price = prev_close * (1 + gap)
            open_price = max(low, min(high, open_price))
        else:
            open_price = close
        
        # Generate volume (crypto has distinct volume patterns)
        base_volume = 1000000 if symbol == "BTCUSDT" else 500000
        volume_multiplier = 1 + abs(np.random.normal(0, 0.8))  # Less extreme volume variance
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
    """Main function to run realistic backtest"""
    print("="*60)
    print("9 & 20 EMA Pullback - REALISTIC BACKTESTING SYSTEM")
    print("Production-level with execution costs and no lookahead bias")
    print("="*60)
    
    # Strategy configuration (realistic parameters)
    config = {
        'ema_fast': 9,
        'ema_slow': 20,
        'adx_period': 14,
        'adx_threshold': 20,
        'lower_wick_ratio': 0.5,      # 50% of candle range
        'upper_wick_ratio': 0.5,      # 50% of candle range
        'wick_body_multiplier': 1.5,   # wick >= 1.5x body
        'momentum_close_threshold': 0.4,  # Top/bottom 40% close
        'ema_distance_threshold': 0.001,
        'risk_reward_ratio': 2.0,      # Strict 1:2 minimum
        'ema_flat_lookback': 5,
        'ema_flat_threshold': 0.0001,
        'max_trades_per_day': 2,
        'max_consecutive_losses': 2,
        'swing_lookback': 10,
        'trading_fee': 0.0005,        # 0.05% per trade
        'slippage': 0.0002            # 0.02% per trade
    }
    
    # Initialize strategy
    strategy = EMAPullbackRealistic(config)
    
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
    print("REALISTIC STRATEGY SUMMARY")
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
            'Cost/Trade': f"{results['cost_impact']*100:.3f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    print(f"\n{'='*60}")
    print("Realistic Backtesting Complete!")
    print("Note: Results include all trading costs and slippage")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
