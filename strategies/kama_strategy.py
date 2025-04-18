
import backtrader as bt
import numpy as np

class KAMAStrategy(bt.Strategy):
    display_name = "KAMA 策略"
    description = "基于考夫曼自适应移动平均（KAMA）的交易策略"
    params = (
        ('kama_period', 10),
        ('fast_period', 2),
        ('slow_period', 30),
        ('size', 0.1)
    )
    indicator_name = "KAMA"
    
    @staticmethod
    def indicator_func(data, kama_period=10, fast_period=2, slow_period=30, **kwargs):
        close = data['close'].values
        if len(close) < kama_period + 1:
            return np.full(len(data), np.nan)
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
        return kama
    
    def __init__(self):
        self.kama = bt.indicators.KAMA(self.datas[0].close, period=self.params.kama_period)
        self.order = None
        self.trades = []
    
    def next(self):
        if self.order:
            return
        if self.datas[0].close[0] > self.kama[0] and self.datas[0].close[-1] <= self.kama[-1] and self.position.size == 0:
            self.order = self.buy(size=self.params.size)
        elif self.datas[0].close[0] < self.kama[0] and self.datas[0].close[-1] >= self.kama[-1] and self.position.size > 0:
            self.order = self.sell(size=self.params.size)
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.trades.append({
                'EntryTime': bt.num2date(order.executed.dt),
                'EntryPrice': order.executed.price,
                'Size': order.executed.size
            })
            self.order = None
