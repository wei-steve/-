
import backtrader as bt
import pandas as pd
import numpy as np

class MACDStrategy(bt.Strategy):
    display_name = "MACD 策略"
    description = "基于移动平均线收敛-发散（MACD）的交易策略"
    params = (
        ('fast_period', 12),
        ('slow_period', 26),
        ('signal_period', 9),
        ('size', 0.1)
    )
    indicator_name = "MACD"
    
    @staticmethod
    def indicator_func(data, fast_period=12, slow_period=26, signal_period=9, **kwargs):
        close = data['close'].values
        if len(close) < slow_period:
            return np.full(len(data), np.nan)
        ema_fast = pd.Series(close).ewm(span=fast_period, adjust=False).mean()
        ema_slow = pd.Series(close).ewm(span=slow_period, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        return macd.values
    
    def __init__(self):
        self.macd = bt.indicators.MACD(
            self.datas[0].close,
            period_me1=self.params.fast_period,
            period_me2=self.params.slow_period,
            period_signal=self.params.signal_period
        )
        self.order = None
        self.trades = []
    
    def next(self):
        if self.order:
            return
        if self.macd.macd[0] > self.macd.signal[0] and self.macd.macd[-1] <= self.macd.signal[-1] and self.position.size == 0:
            self.order = self.buy(size=self.params.size)
        elif self.macd.macd[0] < self.macd.signal[0] and self.macd.macd[-1] >= self.macd.signal[-1] and self.position.size > 0:
            self.order = self.sell(size=self.params.size)
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.trades.append({
                'EntryTime': bt.num2date(order.executed.dt),
                'EntryPrice': order.executed.price,
                'Size': order.executed.size
            })
            self.order = None
