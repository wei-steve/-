import pandas as pd
import numpy as np
from typing import Dict, Tuple, Type
from backtest_engine import backtest_engine
from strategy_interface import IStrategy

class RSIStrategy(IStrategy):
    @property
    def display_name(self) -> str:
        return "RSI 策略"

    @property
    def description(self) -> str:
        return "基于 RSI 指标的交易策略"

    @property
    def params(self) -> Dict[str, float]:
        return {
            'rsi_period': 14,
            'overbought': 60,
            'oversold': 40,
            'size': 0.1
        }

    @property
    def param_schema(self) -> Dict[str, Dict[str, any]]:
        return {
            'rsi_period': {'type': 'integer', 'default': 14, 'min': 1, 'max': 100, 'description': 'RSI 计算周期'},
            'overbought': {'type': 'float', 'default': 60, 'min': 50, 'max': 100, 'description': '超买阈值'},
            'oversold': {'type': 'float', 'default': 40, 'min': 0, 'max': 50, 'description': '超卖阈值'},
            'size': {'type': 'float', 'default': 0.1, 'min': 0.01, 'max': 1.0, 'description': '交易手数'}
        }

    def compute_indicator(self, data: pd.DataFrame, params: Dict[str, any]) -> pd.Series:
        rsi_period = params.get('rsi_period', 14)
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def indicator_config(self, data: pd.DataFrame, params: Dict[str, any]) -> Dict[str, any]:
        overbought = params.get('overbought', 60)
        oversold = params.get('oversold', 40)
        return {
            'shapes': [
                {'y': overbought, 'color': 'red', 'dash': 'dash'},
                {'y': oversold, 'color': 'green', 'dash': 'dash'}
            ]
        }

    def run_backtest(self, data: pd.DataFrame, params: Dict[str, any]) -> Tuple[Dict[str, any], pd.DataFrame, pd.DataFrame]:
        # 确保 params 参数被正确接收
        rsi_period = params.get('rsi_period', 14)
        overbought = params.get('overbought', 60)
        oversold = params.get('oversold', 40)
        size = params.get('size', 0.1)

        # 计算 RSI
        rsi = self.compute_indicator(data, params)

        # 交易信号
        buy_signals = pd.DataFrame(index=data.index)
        sell_signals = pd.DataFrame(index=data.index)
        
        buy_signals['EntryTime'] = data.index
        sell_signals['EntryTime'] = data.index
        
        buy_signals['EntryPrice'] = np.where(rsi < oversold, data['close'], np.nan)
        sell_signals['EntryPrice'] = np.where(rsi > overbought, data['close'], np.nan)

        buy_signals = buy_signals.dropna()
        sell_signals = sell_signals.dropna()

        # 运行回测，使用 backtest_engine 对象
        cerebro = backtest_engine  # 直接使用对象
        cerebro.data = data
        cerebro.buy_signals = buy_signals
        cerebro.sell_signals = sell_signals
        cerebro.size = size
        stats = cerebro.run()

        return stats, buy_signals, sell_signals