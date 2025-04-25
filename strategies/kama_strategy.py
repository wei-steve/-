import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from backtest_engine import backtest_engine
from strategy_interface import IStrategy

class KAMAStrategy(IStrategy):
    @property
    def display_name(self) -> str:
        return "KAMA 策略"

    @property
    def description(self) -> str:
        return "基于考夫曼自适应移动平均（KAMA）的交易策略"

    @property
    def params(self) -> Dict[str, float]:
        return {
            'kama_period': 10,
            'fast_period': 2,
            'slow_period': 30,
            'size': 0.1
        }

    @property
    def param_schema(self) -> Dict[str, Dict[str, Any]]:
        return {
            'kama_period': {'type': 'integer', 'default': 10, 'min': 1, 'max': 100, 'description': 'KAMA周期'},
            'fast_period': {'type': 'integer', 'default': 2, 'min': 1, 'max': 50, 'description': '快速周期'},
            'slow_period': {'type': 'integer', 'default': 30, 'min': 1, 'max': 100, 'description': '慢速周期'},
            'size': {'type': 'float', 'default': 0.1, 'min': 0.01, 'max': 1.0, 'description': '交易手数'}
        }

    def compute_indicator(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        close = data['close'].values
        kama_period = params.get('kama_period', 10)
        fast_period = params.get('fast_period', 2)
        slow_period = params.get('slow_period', 30)
        if len(close) < kama_period + 1:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        kama = np.zeros(len(data))
        kama[:kama_period] = np.nan
        kama[kama_period] = close[kama_period]
        for i in range(kama_period + 1, len(data)):
            change = abs(close[i] - close[i - kama_period])
            volatility = sum(abs(close[j] - close[j-1]) for j in range(i - kama_period + 1, i + 1))
            er = change / volatility if volatility != 0 else 0
            fast = 2 / (fast_period + 1)
            slow = 2 / (slow_period + 1)
            sc = (er * (fast - slow) + slow) ** 2
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
        return pd.Series(kama, index=data.index)

    def indicator_config(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        return {'shapes': []}

    def run_backtest(self, data: pd.DataFrame, params: Dict[str, Any]) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
        kama = self.compute_indicator(data, params)
        buy_signals = pd.DataFrame(index=data.index)
        sell_signals = pd.DataFrame(index=data.index)
        buy_signals['EntryTime'] = data.index
        sell_signals['EntryTime'] = data.index
        buy_signals['EntryPrice'] = np.where(
            (data['close'] > kama) & (data['close'].shift(1) <= kama.shift(1)),
            data['close'], np.nan
        )
        sell_signals['EntryPrice'] = np.where(
            (data['close'] < kama) & (data['close'].shift(1) >= kama.shift(1)),
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