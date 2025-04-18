import pandas as pd

class BacktraderEngine:
    def __init__(self):
        self.data = None
        self.buy_signals = None
        self.sell_signals = None
        self.size = None

    def run(self):
        if self.data is None or self.buy_signals is None or self.sell_signals is None:
            raise ValueError("数据或信号未设置")

        # 模拟回测逻辑
        trades = 0
        wins = 0
        total_return = 0.0
        max_drawdown = 0.0

        # 简单的回测逻辑：基于买入和卖出信号计算收益
        position = 0
        entry_price = 0.0
        portfolio_value = 10000.0  # 初始资金
        peak = portfolio_value
        for timestamp in self.data.index:
            if timestamp in self.buy_signals['EntryTime'].values and position == 0:
                entry_price = self.buy_signals.loc[self.buy_signals['EntryTime'] == timestamp, 'EntryPrice'].iloc[0]
                position = self.size
                trades += 1
            elif timestamp in self.sell_signals['EntryTime'].values and position > 0:
                exit_price = self.sell_signals.loc[self.sell_signals['EntryTime'] == timestamp, 'EntryPrice'].iloc[0]
                trade_return = (exit_price - entry_price) / entry_price * position * portfolio_value
                total_return += trade_return
                portfolio_value += trade_return
                if trade_return > 0:
                    wins += 1
                position = 0
                # 计算最大回撤
                peak = max(peak, portfolio_value)
                drawdown = (peak - portfolio_value) / peak
                max_drawdown = max(max_drawdown, drawdown)

        # 计算统计数据
        stats = {
            "Return [%]": total_return / 10000.0 * 100,  # 总回报率
            "Max. Drawdown [%]": max_drawdown * 100,     # 最大回撤
            "# Trades": trades,                          # 交易次数
            "Win Rate [%]": (wins / trades * 100) if trades > 0 else 0  # 胜率
        }
        return stats

# 全局实例
backtest_engine = BacktraderEngine()