import pandas as pd
import sqlite3
from backtesting import Backtest, Strategy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np


# 1. 数据加载函数
def load_kline_data(db_path, coin_name='LEAUSDT', interval='1m', limit=200, offset=0):
    """
    从 SQLite 数据库加载 K 线数据
    :param db_path: 数据库路径
    :param coin_name: 币种名称
    :param interval: K 线周期
    :param limit: 加载条数
    :param offset: 偏移量（用于分页加载）
    :return: pandas DataFrame
    """
    try:
        conn = sqlite3.connect(db_path)
        query = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM kline_data
            WHERE coin_name = ? AND interval = ?
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """
        df = pd.read_sql_query(query, conn, params=(coin_name, interval, limit, offset))
        conn.close()

        if df.empty:
            print("警告：数据库返回空数据")
            return pd.DataFrame()

        # 转换时间戳为 datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()  # 确保时间升序
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']  # Backtesting.py 要求的列名

        # 验证数据
        if df[['Open', 'High', 'Low', 'Close']].le(0).any().any() or df.isna().any().any():
            print("警告：数据包含零、负值或 NaN")
            df = df[(df[['Open', 'High', 'Low', 'Close']] > 0).all(axis=1)].dropna()

        print(f"加载数据：{len(df)} 条，时间范围：{df.index.min()} 到 {df.index.max()}")
        print(f"价格范围：Open {df['Open'].min()} - {df['Open'].max()}, Close {df['Close'].min()} - {df['Close'].max()}")

        return df
    except Exception as e:
        print(f"数据加载错误：{e}")
        return pd.DataFrame()


# 2. 趋势跟随策略
class TrendFollowingStrategy(Strategy):
    fast_ma = 10  # 快均线周期
    slow_ma = 20  # 慢均线周期
    size = 0.1  # 每次交易 10% 的账户资金

    def init(self):
        # 定义移动平均函数
        def moving_average(series, window):
            return pd.Series(series).rolling(window=window, min_periods=1).mean().values

        # 注册指标
        self.ma1 = self.I(moving_average, self.data.Close, self.fast_ma)
        self.ma2 = self.I(moving_average, self.data.Close, self.slow_ma)

    def next(self):
        price = self.data.Close[-1]
        if price <= 0 or np.isnan(price):
            print(f"跳过交易：无效价格 {price}")
            return

        if crossover(self.ma1, self.ma2):
            self.buy(size=self.size)
            print(f"买入：时间 {self.data.index[-1]}, 价格 {price}, 仓位 {self.size}")
        elif crossover(self.ma2, self.ma1):
            self.sell(size=self.size)
            print(f"卖出：时间 {self.data.index[-1]}, 价格 {price}, 仓位 {self.size}")


# 辅助函数：检测交叉
def crossover(series1, series2):
    return series1[-2] < series2[-2] and series1[-1] > series2[-1]


# 3. 回测和绘图函数
def run_backtest(data, fast_ma=10, slow_ma=20):
    """
    运行回测并返回结果和信号
    """
    if data.empty or len(data) < slow_ma:
        print("错误：数据不足，无法回测")
        return None, pd.DataFrame(), pd.DataFrame()

    # 设置策略参数
    TrendFollowingStrategy.fast_ma = fast_ma
    TrendFollowingStrategy.slow_ma = slow_ma

    # 运行回测
    try:
        bt = Backtest(data, TrendFollowingStrategy, cash=1000000, commission=0.001)
        stats = bt.run()

        # 获取买卖信号
        trades = getattr(bt, '_trades', [])
        if trades:
            trades = pd.DataFrame(trades)
            buy_signals = trades[trades['Size'] > 0][['EntryTime', 'EntryPrice']]
            sell_signals = trades[trades['Size'] < 0][['EntryTime', 'EntryPrice']]
        else:
            buy_signals = pd.DataFrame(columns=['EntryTime', 'EntryPrice'])
            sell_signals = pd.DataFrame(columns=['EntryTime', 'EntryPrice'])

        print(f"回测完成：交易次数 {len(trades)}")
        return stats, buy_signals, sell_signals
    except Exception as e:
        print(f"回测错误：{e}")
        return None, pd.DataFrame(), pd.DataFrame()


def create_kline_plot(data, buy_signals, sell_signals, fast_ma=10, slow_ma=20):
    """
    创建交互式 K 线图，包含均线和买卖信号
    """
    if data.empty:
        print("警告：无数据，无法绘图")
        return go.Figure()

    # 计算均线
    ma1 = data['Close'].rolling(fast_ma, min_periods=1).mean()
    ma2 = data['Close'].rolling(slow_ma, min_periods=1).mean()

    # 创建子图（K 线 + 成交量）
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, subplot_titles=('K线图', '成交量'),
                        row_heights=[0.7, 0.3])

    # 添加 K 线
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'], high=data['High'],
            low=data['Low'], close=data['Close'],
            name='K线'
        ),
        row=1, col=1
    )

    # 添加均线
    fig.add_trace(
        go.Scatter(x=data.index, y=ma1, name=f'MA{fast_ma}', line=dict(color='orange')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=ma2, name=f'MA{slow_ma}', line=dict(color='purple')),
        row=1, col=1
    )

    # 添加买卖信号
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals['EntryTime'], y=buy_signals['EntryPrice'],
                mode='markers', name='买入',
                marker=dict(symbol='triangle-up', size=10, color='green')
            ),
            row=1, col=1
        )
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals['EntryTime'], y=sell_signals['EntryPrice'],
                mode='markers', name='卖出',
                marker=dict(symbol='triangle-down', size=10, color='red')
            ),
            row=1, col=1
        )

    # 添加成交量
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='成交量', marker_color='blue'),
        row=2, col=1
    )

    # 更新布局，模仿 TradingView
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        dragmode='pan',
        template='plotly_dark',
        hovermode='x unified',
        showlegend=True,
        height=600,
        autosize=True
    )
    fig.update_xaxes(
        rangebreaks=[dict(bounds=['sat', 'mon'])]
    )

    return fig


# 4. Dash 应用
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# 数据库路径
DB_PATH = r'D:\策略研究\kline_db_new\kline_data_LEAUSDT.db'

# 初始数据
initial_data = load_kline_data(DB_PATH, limit=200)
initial_stats, initial_buy_signals, initial_sell_signals = run_backtest(initial_data)

# 初始图表
initial_fig = create_kline_plot(initial_data, initial_buy_signals, initial_sell_signals)

# 布局
app.layout = dbc.Container([
    html.H1('LEAUSDT 趋势跟随策略回测', style={'textAlign': 'center'}),
    dbc.Row([
        dbc.Col([
            html.Label('快均线周期'),
            dcc.Input(id='fast-ma', type='number', value=10, min=1, step=1),
            html.Label('慢均线周期'),
            dcc.Input(id='slow-ma', type='number', value=20, min=1, step=1),
            html.Br(),
            dbc.Button('更新回测', id='update-btn', color='primary', n_clicks=0),
            html.Br(),
            html.Label('加载历史数据'),
            dbc.Button('加载更多', id='load-more', color='secondary', n_clicks=0),
        ], width=3),
        dbc.Col([
            dcc.Graph(id='kline-plot', figure=initial_fig, config={'scrollZoom': True}),
            html.Div(id='stats-output')
        ], width=9)
    ]),
    dcc.Store(id='data-store', data=initial_data.to_json(date_format='iso', orient='split')),
    dcc.Store(id='offset-store', data=0)
], fluid=True)


# 回调：更新图表和统计
@app.callback(
    [Output('kline-plot', 'figure'),
     Output('stats-output', 'children'),
     Output('data-store', 'data')],
    [Input('update-btn', 'n_clicks'),
     Input('load-more', 'n_clicks')],
    [State('fast-ma', 'value'),
     State('slow-ma', 'value'),
     State('data-store', 'data'),
     State('offset-store', 'data')]
)
def update_plot(update_clicks, load_clicks, fast_ma, slow_ma, stored_data, offset):
    # 恢复存储的数据
    data = pd.read_json(stored_data, orient='split')
    data.index = pd.to_datetime(data.index)

    # 处理加载更多
    if load_clicks > 0:
        new_offset = offset + 100
        new_data = load_kline_data(DB_PATH, limit=100, offset=new_offset)
        if not new_data.empty:
            data = pd.concat([new_data, data]).sort_index()
            offset = new_offset

    # 运行回测
    stats, buy_signals, sell_signals = run_backtest(data, fast_ma, slow_ma)

    # 创建图表
    fig = create_kline_plot(data, buy_signals, sell_signals, fast_ma, slow_ma)

    # 统计信息
    if stats is None:
        stats_text = [html.P("无足够数据进行回测")]
    else:
        stats_text = [
            html.P(f"总回报: {stats['Return [%]']:.2f}%"),
            html.P(f"最大回撤: {stats['Max. Drawdown [%]']:.2f}%"),
            html.P(f"交易次数: {stats['# Trades']}"),
            html.P(f"胜率: {stats['Win Rate [%]']:.2f}%")
        ]

    # 存储更新后的数据
    data_json = data.to_json(date_format='iso', orient='split')

    return fig, stats_text, data_json


# 回调：更新偏移量
@app.callback(
    Output('offset-store', 'data'),
    Input('load-more', 'n_clicks'),
    State('offset-store', 'data')
)
def update_offset(load_clicks, offset):
    if load_clicks > 0:
        return offset + 100
    return offset


# 运行应用
if __name__ == '__main__':
    app.run(debug=True)