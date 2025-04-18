import pandas as pd
import numpy as np
from typing import Dict, Tuple, Type
from backtest_engine import backtest_engine
from strategy_interface import IStrategy

class BollingerStrategy(IStrategy):
    @property
    def display_name(self) -> str:
        return "布林带策略"

    @property
    def description(self) -> str:
        return "基于布林带的交易策略"

    @property
    def params(self) -> Dict[str, float]:
        return {
            'period': 20,
            'devfactor': 2.0,
            'size': 0.1
        }

    @property
    def param_schema(self) -> Dict[str, Dict[str, any]]:
        return {
            'period': {'type': 'integer', 'default': 20, 'min': 1, 'max': 100, 'description': '布林带周期'},
            'devfactor': {'type': 'float', 'default': 2.0, 'min': 0.1, 'max': 5.0, 'description': '布林带标准差倍数'},
            'size': {'type': 'float', 'default': 0.1, 'min': 0.01, 'max': 1.0, 'description': '交易手数'}
        }

    def compute_indicator(self, data: pd.DataFrame, params: Dict[str, any]) -> pd.Series:
        period = params.get('period', 20)
        devfactor = params.get('devfactor', 2.0)
        sma = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        upper_band = sma + (std * devfactor)
        lower_band = sma - (std * devfactor)
        return upper_band  # 返回上轨作为指标（可以根据需要返回其他指标）

    def indicator_config(self, data: pd.DataFrame, params: Dict[str, any]) -> Dict[str, any]:
        period = params.get('period', 20)
        devfactor = params.get('devfactor', 2.0)
        sma = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        upper_band = sma + (std * devfactor)
        lower_band = sma - (std * devfactor)
        return {
            'shapes': [
                {'y': upper_band.mean(), 'color': 'red', 'dash': 'dash'},
                {'y': sma.mean(), 'color': 'gray', 'dash': 'dash'},
                {'y': lower_band.mean(), 'color': 'green', 'dash': 'dash'}
            ]
        }

    def run_backtest(self, data: pd.DataFrame, params: Dict[str, any]) -> Tuple[Dict[str, any], pd.DataFrame, pd.DataFrame]:
        # 确保 params 参数被正确接收
        period = params.get('period', 20)
        devfactor = params.get('devfactor', 2.0)
        size = params.get('size', 0.1)

        # 计算布林带
        sma = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        upper_band = sma + (std * devfactor)
        lower_band = sma - (std * devfactor)

        # 交易信号
        buy_signals = pd.DataFrame(index=data.index)
        sell_signals = pd.DataFrame(index=data.index)
        
        buy_signals['EntryTime'] = data.index
        sell_signals['EntryTime'] = data.index
        
        buy_signals['EntryPrice'] = np.where(data['close'] < lower_band, data['close'], np.nan)
        sell_signals['EntryPrice'] = np.where(data['close'] > upper_band, data['close'], np.nan)

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