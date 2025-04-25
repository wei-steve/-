import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from backtest_engine import backtest_engine
from strategy_interface import IStrategy

class MACDStrategy(IStrategy):
    @property
    def display_name(self) -> str:
        return "MACD 策略"

    @property
    def description(self) -> str:
        return "基于移动平均线收敛-发散（MACD）的交易策略"

    @property
    def params(self) -> Dict[str, float]:
        return {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'size': 0.1
        }

    @property
    def param_schema(self) -> Dict[str, Dict[str, Any]]:
        return {
            'fast_period': {'type': 'integer', 'default': 12, 'min': 1, 'max': 50, 'description': '快速周期'},
            'slow_period': {'type': 'integer', 'default': 26, 'min': 1, 'max': 100, 'description': '慢速周期'},
            'signal_period': {'type': 'integer', 'default': 9, 'min': 1, 'max': 50, 'description': '信号线周期'},
            'size': {'type': 'float', 'default': 0.1, 'min': 0.01, 'max': 1.0, 'description': '交易手数'}
        }

    def compute_indicator(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        close = data['close'].values
        fast_period = params.get('fast_period', 12)
        slow_period = params.get('slow_period', 26)
        signal_period = params.get('signal_period', 9)
        if len(close) < slow_period:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        ema_fast = pd.Series(close).ewm(span=fast_period, adjust=False).mean()
        ema_slow = pd.Series(close).ewm(span=slow_period, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        return pd.Series(macd, index=data.index)

    def indicator_config(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'shapes': [
                {'y': 0, 'color': 'gray', 'dash': 'dash'}  # MACD零线
            ]
        }

    def run_backtest(self, data: pd.DataFrame, params: Dict[str, Any]) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
        macd = self.compute_indicator(data, params)
        signal = macd.ewm(span=params.get('signal_period', 9), adjust=False).mean()
        buy_signals = pd.DataFrame(index=data.index)
        sell_signals = pd.DataFrame(index=data.index)
        buy_signals['EntryTime'] = data.index
        sell_signals['EntryTime'] = data.index
        buy_signals['EntryPrice'] = np.where(
            (macd > signal) & (macd.shift(1) <= signal.shift(1)),
            data['close'], np.nan
        )
        sell_signals['EntryPrice'] = np.where(
            (macd < signal) & (macd.shift(1) >= signal.shift(1)),
            data['close'], np.nan
        )
        buy_signals = buy_signals.dropna()
        sell_signals = sell_signals.dropna()

        cerebro = backtest_engine
        cerebro.data = data
        cerebro.buy_signals = buy_signals
        cerebro.sell_signals = sell_signals
        cerebro.size = params.get('size', 0.1)
        stats = cerebro.run()

        return stats, buy_signals, sell_signals