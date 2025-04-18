import backtrader as bt
import sqlite3
import pandas as pd
import os
import numpy as np


# 1. 加载数据
def load_kline_data(coin_end_time=None, coin_name="LEAUSDT", db_path="D:\\策略研究\\kline_db_new", time_range='15D',
                    resample_interval='5min'):
    db_file = os.path.join(db_path, f"kline_data_{coin_name}.db")
    if not os.path.exists(db_file):
        print(f"数据库文件 {db_file} 不存在")
        return None

    conn = sqlite3.connect(db_file)
    query = f"SELECT * FROM kline_data WHERE coin_name = '{coin_name}' AND interval = '1m'"
    df = pd.read_sql_query(query, conn)
    conn.close()

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)
    df[['open', 'high', 'low', 'close', 'volume', 'turnover']] = df[
        ['open', 'high', 'low', 'close', 'volume', 'turnover']].astype(float)

    df = df.dropna()

    if df.empty:
        print(f"数据为空或全部为 NaN: {coin_name}")
        return None

    if coin_end_time is None:
        end_time = df.index.max()
    else:
        end_time = pd.to_datetime(coin_end_time)
    start_time = end_time - pd.Timedelta(time_range)
    df = df.loc[start_time:end_time]

    if resample_interval:
        df = df.resample(resample_interval).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'turnover': 'sum',
            'coin_name': 'first',
            'interval': 'first',
            'exchange_id': 'first',
            'instrument_id': 'first',
            'period_id': 'first'
        }).dropna()

    if df.empty:
        print(f"重采样后数据为空: {coin_name}")
        return None

    return df


# 2. 定义 KAMA 指标
class KAMA(bt.Indicator):
    lines = ('kama',)
    params = (
        ('length', 14),
        ('fast_length', 2),
        ('slow_length', 20),
    )

    def __init__(self):
        self.addminperiod(1)
        self.src = self.data
        self.mom = abs(self.src - self.src(-self.params.length))
        self.volatility = bt.indicators.SumN(abs(self.src - self.src(-1)), period=self.params.length)
        self.er = bt.If(bt.And(self.volatility != 0,
                               bt.Or(self.mom != self.mom, 0) == 0,
                               bt.Or(self.volatility != self.volatility, 0) == 0),
                        self.mom / self.volatility, 0)
        self.fast_alpha = 2 / (self.params.fast_length + 1)
        self.slow_alpha = 2 / (self.params.slow_length + 1)
        self.sc = (self.er * (self.fast_alpha - self.slow_alpha) + self.slow_alpha) ** 2

    def next(self):
        print(
            f"mom: {self.mom[0]:.6f}, volatility: {self.volatility[0]:.6f}, er: {self.er[0]:.6f}, sc: {self.sc[0]:.6f}")

        if len(self) == 1:
            self.lines.kama[0] = self.src[0] if not np.isnan(self.src[0]) else 0
        else:
            if np.isnan(self.sc[0]) or np.isnan(self.src[0]) or np.isnan(self.lines.kama[-1]):
                self.lines.kama[0] = self.lines.kama[-1] if not np.isnan(self.lines.kama[-1]) else self.src[0]
            else:
                self.lines.kama[0] = self.sc[0] * self.src[0] + (1 - self.sc[0]) * self.lines.kama[-1]


# 3. 定义 KAMA 策略
class KAMAStrategy(bt.Strategy):
    params = (
        ('length1', 20), ('fastLength1', 2), ('slowLength1', 10),
        ('length2', 15), ('fastLength2', 3), ('slowLength2', 22),
        ('length3', 16), ('fastLength3', 4), ('slowLength3', 24),
        ('length4', 17), ('fastLength4', 5), ('slowLength4', 26),
        ('length5', 18), ('fastLength5', 6), ('slowLength5', 28),
        ('length6', 19), ('fastLength6', 7), ('slowLength6', 30),
        ('length7', 20), ('fastLength7', 8), ('slowLength7', 32),
        ('length8', 21), ('fastLength8', 9), ('slowLength8', 34),
        ('entry_filter', 0.5), ('exit_filter', 0.5),
        ('atr_lookback', 14), ('atr_multiplier', 3.0),
        ('do_long', True), ('do_short', True),
        ('kama1_sl', True), ('atr_sl', True),
    )

    # 定义要绘制的线
    lines = ('kama1_delta', 'entry_maaf', 'neg_entry_maaf',)

    # 自定义绘图属性
    plotlines = dict(
        kama1_delta=dict(color='blue', linewidth=1.5),
        entry_maaf=dict(color='green', linewidth=1.0, linestyle='--'),
        neg_entry_maaf=dict(color='red', linewidth=1.0, linestyle='--'),
    )

    # 将 kama1_delta 和 entry_maaf 放入单独的副图
    plotinfo = dict(subplot=True)

    def __init__(self):
        self.kama1 = KAMA(self.data.close, length=self.p.length1, fast_length=self.p.fastLength1,
                          slow_length=self.p.slowLength1)
        self.kama2 = KAMA(self.data.close, length=self.p.length2, fast_length=self.p.fastLength2,
                          slow_length=self.p.slowLength2)
        self.kama3 = KAMA(self.data.close, length=self.p.length3, fast_length=self.p.fastLength3,
                          slow_length=self.p.slowLength3)
        self.kama4 = KAMA(self.data.close, length=self.p.length4, fast_length=self.p.fastLength4,
                          slow_length=self.p.slowLength4)
        self.kama5 = KAMA(self.data.close, length=self.p.length5, fast_length=self.p.fastLength5,
                          slow_length=self.p.slowLength5)
        self.kama6 = KAMA(self.data.close, length=self.p.length6, fast_length=self.p.fastLength6,
                          slow_length=self.p.slowLength6)
        self.kama7 = KAMA(self.data.close, length=self.p.length7, fast_length=self.p.fastLength7,
                          slow_length=self.p.slowLength7)
        self.kama8 = KAMA(self.data.close, length=self.p.length8, fast_length=self.p.fastLength8,
                          slow_length=self.p.slowLength8)

        self.kama1_delta = self.kama1 - self.kama1(-1)
        self.kama3_delta = self.kama3 - self.kama3(-1)
        self.kama8_delta = self.kama8 - self.kama8(-1)

        self.kama1_delta_std = bt.indicators.StdDev(self.kama1_delta, period=self.p.length1)
        self.entry_maaf = self.p.entry_filter * self.kama1_delta_std
        self.exit_maaf = self.p.exit_filter * self.kama1_delta_std

        # 绑定指标到 lines，以便绘图
        self.lines.kama1_delta = self.kama1_delta
        self.lines.entry_maaf = self.entry_maaf
        self.lines.neg_entry_maaf = -self.entry_maaf

        # ATR 追踪止损初始化
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_lookback)
        self.atr_multiplied = self.atr * self.p.atr_multiplier
        self.atr_low = self.data.low - self.atr_multiplied
        self.atr_high = self.data.high + self.atr_multiplied
        self.trail_atr_low = self.atr_low
        self.trail_atr_high = self.atr_high
        self.support_hit = self.data.low <= self.trail_atr_low
        self.resistance_hit = self.data.high >= self.trail_atr_high

        self.order = None

    def next(self):
        if self.order:
            return

        if len(self) == 1:
            self.trail_atr_low[0] = self.atr_low[0]
            self.trail_atr_high[0] = self.atr_high[0]
        else:
            self.trail_atr_low[0] = self.atr_low[0] if self.atr_low[0] >= self.trail_atr_low[-1] else \
            self.trail_atr_low[-1]
            self.trail_atr_high[0] = self.atr_high[0] if self.atr_high[0] <= self.trail_atr_high[-1] else \
            self.trail_atr_high[-1]

        self.support_hit[0] = self.data.low[0] <= self.trail_atr_low[0]
        self.resistance_hit[0] = self.data.high[0] >= self.trail_atr_high[0]

        print(
            f"Date: {self.datetime.datetime(0)}, kama1: {self.kama1[0]:.6f}, kama1_delta: {self.kama1_delta[0]:.6f}, entry_maaf: {self.entry_maaf[0]:.6f}")

        trend_up = self.kama1[0] > self.kama2[0]
        trend_down = self.kama1[0] < self.kama2[0]

        long_condition = self.p.do_long and trend_up and self.kama1_delta[0] > 0 and self.kama1_delta[0] > \
                         self.entry_maaf[0]
        short_condition = self.p.do_short and trend_down and self.kama1_delta[0] < -self.entry_maaf[0]

        long_close = (
                (self.p.kama1_sl and self.kama1_delta[0] < 0 and abs(self.kama1_delta[0]) > self.exit_maaf[0]) or
                (self.p.atr_sl and self.support_hit[0])
        )
        short_close = (
                (self.p.kama1_sl and self.kama1_delta[0] > self.exit_maaf[0]) or
                (self.p.atr_sl and self.resistance_hit[0])
        )

        if not self.position:
            if long_condition:
                self.order = self.buy(size=self.broker.getcash() / self.data.close[0])
            elif short_condition:
                self.order = self.sell(size=self.broker.getcash() / self.data.close[0])
        else:
            if self.position.size > 0 and long_close:
                self.order = self.close()
            elif self.position.size < 0 and short_close:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


# 4. 回测主函数
def run_backtest(df, coin_name, plot=True):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(KAMAStrategy)

    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)

    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.0005)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    results = cerebro.run()
    strat = results[0]

    final_value = cerebro.broker.getvalue()
    print(f"{coin_name} 最终资金: {final_value:.2f}")
    print(f"夏普比率: {strat.analyzers.sharpe.get_analysis()['sharperatio']}")
    print(f"最大回撤: {strat.analyzers.drawdown.get_analysis()['max']['drawdown']}%")

    if plot:
        cerebro.plot(style='candlestick', volume=False)


# 5. 主函数
def main():
    df = load_kline_data()
    if df is None or df.empty:
        print("LEAUSDT 数据为空")
        return

    print(f"数据预览:\n{df.head()}")
    run_backtest(df, "LEAUSDT", plot=True)


if __name__ == "__main__":
    main()