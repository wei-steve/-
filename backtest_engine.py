import pandas as pd
import numpy as np

class BacktraderEngine:
    def __init__(self):
        self.data = None
        self.buy_signals = None
        self.sell_signals = None
        self.size = None

    def run(self):
        if self.data is None or self.buy_signals is None or self.sell_signals is None:
            raise ValueError("数据或信号未设置")

        trades = 0
        wins = 0
        total_return = 0.0
        max_drawdown = 0.0
        portfolio_values = []
        portfolio_value = 10000.0
        peak = portfolio_value
        position = 0
        entry_price = 0.0

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
                portfolio_values.append(portfolio_value)
                if trade_return > 0:
                    wins += 1
                position = 0
                peak = max(peak, portfolio_value)
                drawdown = (peak - portfolio_value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            else:
                portfolio_values.append(portfolio_value)

        # 计算夏普比率和波动率
        returns = pd.Series(portfolio_values).pct_change().dropna()
        annualized_return = returns.mean() * 252 * 1440  # 1分钟K线
        annualized_std = returns.std() * np.sqrt(252 * 1440)
        sharpe_ratio = annualized_return / annualized_std if annualized_std != 0 else 0

        stats = {
            "Return [%]": total_return / 10000.0 * 100,
            "Max. Drawdown [%]": max_drawdown * 100,
            "# Trades": trades,
            "Win Rate [%]": (wins / trades * 100) if trades > 0 else 0,
            "Sharpe Ratio": sharpe_ratio,
            "Annualized Volatility [%]": annualized_std * 100
        }
        return stats

# 全局实例
backtest_engine = BacktraderEngine()