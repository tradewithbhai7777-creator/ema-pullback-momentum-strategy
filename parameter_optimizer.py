"""
Parameter Optimization for EMA Pullback Strategy
Grid search and optimization utilities
"""

import pandas as pd
import numpy as np
from itertools import product
from ema_pullback_strategy import EMAPullbackStrategy, generate_sample_data
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class ParameterOptimizer:
    """Parameter optimization for EMA Pullback Strategy"""
    
    def __init__(self):
        self.optimization_results = []
    
    def grid_search(self, param_grid: Dict, symbols: List[str] = ["BTCUSDT"], 
                   days: int = 365, initial_capital: float = 10000) -> pd.DataFrame:
        """
        Perform grid search optimization
        
        Args:
            param_grid: Dictionary of parameter ranges to test
            symbols: List of symbols to test
            days: Number of days of data to use
            initial_capital: Starting capital for backtest
            
        Returns:
            DataFrame with optimization results
        """
        print("Starting grid search optimization...")
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        print(f"Testing {len(combinations)} parameter combinations...")
        
        results = []
        
        for i, combination in enumerate(combinations):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(combinations)} ({i/len(combinations)*100:.1f}%)")
            
            # Create parameter dictionary
            params = dict(zip(param_names, combination))
            
            # Test across all symbols
            symbol_results = []
            
            for symbol in symbols:
                # Generate data
                data = generate_sample_data(symbol, days)
                
                # Initialize strategy with current parameters
                strategy = EMAPullbackStrategy(params)
                
                # Run backtest
                results_dict = strategy.backtest(data, initial_capital)
                
                # Collect key metrics
                symbol_results.append({
                    'symbol': symbol,
                    'total_return': results_dict['total_return'],
                    'win_rate': results_dict['win_rate'],
                    'profit_factor': results_dict['profit_factor'],
                    'max_drawdown': results_dict['max_drawdown'],
                    'sharpe_ratio': results_dict['sharpe_ratio'],
                    'total_trades': results_dict['total_trades']
                })
            
            # Calculate average metrics across symbols
            avg_metrics = {
                'avg_return': np.mean([r['total_return'] for r in symbol_results]),
                'avg_win_rate': np.mean([r['win_rate'] for r in symbol_results]),
                'avg_profit_factor': np.mean([r['profit_factor'] for r in symbol_results]),
                'avg_max_drawdown': np.mean([r['max_drawdown'] for r in symbol_results]),
                'avg_sharpe': np.mean([r['sharpe_ratio'] for r in symbol_results]),
                'avg_total_trades': np.mean([r['total_trades'] for r in symbol_results])
            }
            
            # Combine results
            result_row = {**params, **avg_metrics}
            result_row.update({f"{s}_{k}": r[k] for r in symbol_results 
                              for s, k in [(r['symbol'], 'total_return'), 
                                          (r['symbol'], 'win_rate'),
                                          (r['symbol'], 'profit_factor')]})
            
            results.append(result_row)
        
        results_df = pd.DataFrame(results)
        self.optimization_results = results_df
        
        print("Grid search completed!")
        return results_df
    
    def find_best_parameters(self, results_df: pd.DataFrame, 
                           optimization_metric: str = 'avg_return',
                           min_trades: int = 10,
                           min_win_rate: float = 0.4) -> Dict:
        """
        Find best parameters based on optimization metric
        
        Args:
            results_df: Results from grid search
            optimization_metric: Metric to optimize for
            min_trades: Minimum number of trades required
            min_win_rate: Minimum win rate required
            
        Returns:
            Dictionary with best parameters and metrics
        """
        # Filter results
        filtered = results_df[
            (results_df['avg_total_trades'] >= min_trades) &
            (results_df['avg_win_rate'] >= min_win_rate)
        ].copy()
        
        if filtered.empty:
            print("No parameters meet the minimum criteria!")
            return {}
        
        # Sort by optimization metric
        if optimization_metric in ['avg_max_drawdown']:
            # For drawdown, lower is better
            best = filtered.sort_values(optimization_metric, ascending=True).iloc[0]
        else:
            # For other metrics, higher is better
            best = filtered.sort_values(optimization_metric, ascending=False).iloc[0]
        
        # Extract parameter columns (non-metric columns)
        param_columns = [col for col in best.index if not any(metric in col for metric in 
                         ['avg_', 'BTCUSDT_', 'ETHUSDT_', 'SOLUSDT_'])]
        
        best_params = {col: best[col] for col in param_columns}
        best_metrics = {col: best[col] for col in best.index if col not in param_columns}
        
        return {
            'best_params': best_params,
            'best_metrics': best_metrics,
            'full_result': best
        }
    
    def plot_optimization_results(self, results_df: pd.DataFrame, 
                                 param_x: str, param_y: str, 
                                 metric: str = 'avg_return'):
        """Plot optimization results as heatmap"""
        
        # Create pivot table
        pivot_table = results_df.pivot_table(
            values=metric, 
            index=param_x, 
            columns=param_y, 
            aggfunc='mean'
        )
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        plt.imshow(pivot_table.values, cmap='RdYlGn', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar()
        cbar.set_label(metric.replace('_', ' ').title(), rotation=270, labelpad=15)
        
        # Set labels and ticks
        plt.xlabel(param_y.replace('_', ' ').title())
        plt.ylabel(param_x.replace('_', ' ').title())
        
        # Set tick labels
        plt.xticks(range(len(pivot_table.columns)), pivot_table.columns)
        plt.yticks(range(len(pivot_table.index)), pivot_table.index)
        
        # Add values to heatmap
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                value = pivot_table.iloc[i, j]
                if not np.isnan(value):
                    plt.text(j, i, f'{value:.3f}', ha='center', va='center',
                            color='black' if abs(value) < 0.5 else 'white')
        
        plt.title(f'Optimization Results: {metric.replace("_", " ").title()}')
        plt.tight_layout()
        plt.savefig(f'optimization_heatmap_{param_x}_{param_y}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_parameter_sensitivity(self, results_df: pd.DataFrame, 
                                   param_name: str, metric: str = 'avg_return'):
        """Analyze sensitivity of a specific parameter"""
        
        # Group by parameter and calculate mean metric
        sensitivity = results_df.groupby(param_name)[metric].agg(['mean', 'std', 'count'])
        
        # Plot sensitivity
        plt.figure(figsize=(12, 6))
        
        # Mean line
        plt.plot(sensitivity.index, sensitivity['mean'], 'o-', linewidth=2, 
                label=f'Average {metric}')
        
        # Add error bars (standard deviation)
        plt.fill_between(sensitivity.index, 
                        sensitivity['mean'] - sensitivity['std'],
                        sensitivity['mean'] + sensitivity['std'],
                        alpha=0.3, label='±1 Std Dev')
        
        plt.xlabel(param_name.replace('_', ' ').title())
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Parameter Sensitivity: {param_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'sensitivity_{param_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return sensitivity


def run_optimization():
    """Run parameter optimization"""
    print("="*60)
    print("EMA Pullback Strategy - Parameter Optimization")
    print("="*60)
    
    # Define parameter grid
    param_grid = {
        'ema_fast': [8, 9, 10, 11, 12],
        'ema_slow': [18, 20, 22, 24, 26],
        'wick_ratio_threshold': [0.5, 0.6, 0.7, 0.8],
        'min_risk_reward': [1.5, 2.0, 2.5, 3.0],
        'slope_threshold': [0.0005, 0.001, 0.002, 0.003],
        'ema_distance_filter': [0.003, 0.005, 0.007, 0.01]
    }
    
    # Initialize optimizer
    optimizer = ParameterOptimizer()
    
    # Run grid search
    results_df = optimizer.grid_search(
        param_grid=param_grid,
        symbols=["BTCUSDT", "ETHUSDT"],
        days=365,
        initial_capital=10000
    )
    
    # Find best parameters for different metrics
    metrics_to_optimize = ['avg_return', 'avg_sharpe', 'avg_profit_factor']
    
    print(f"\n{'='*60}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    
    for metric in metrics_to_optimize:
        best_result = optimizer.find_best_parameters(results_df, metric)
        
        if best_result:
            print(f"\nBest by {metric}:")
            print(f"Parameters: {best_result['best_params']}")
            print(f"Metrics: {best_result['best_metrics']}")
    
    # Save results
    results_df.to_csv('optimization_results.csv', index=False)
    print(f"\nOptimization results saved to 'optimization_results.csv'")
    
    # Create sensitivity analysis for key parameters
    key_params = ['ema_fast', 'wick_ratio_threshold', 'min_risk_reward']
    
    for param in key_params:
        print(f"\nAnalyzing sensitivity for {param}...")
        sensitivity = optimizer.analyze_parameter_sensitivity(results_df, param)
        print(sensitivity)
    
    # Create optimization heatmap
    if len(results_df) > 0:
        optimizer.plot_optimization_results(results_df, 'ema_fast', 'ema_slow', 'avg_return')
        optimizer.plot_optimization_results(results_df, 'wick_ratio_threshold', 'min_risk_reward', 'avg_return')
    
    print(f"\n{'='*60}")
    print("Optimization Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_optimization()
