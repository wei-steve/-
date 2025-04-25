# 导入必要的库
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import Dash, dcc, html, Input, Output, State, no_update, dash_table
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
from io import StringIO
import sys
import json
import logging
import os
import argparse
import requests
from strategy_registry import registry
from backtest_engine import BacktraderEngine
from functools import lru_cache
import time

# 设置日志配置（确保只初始化一次）
logger = logging.getLogger(__name__)
logger.handlers.clear()  # 清除所有已存在的处理器，避免重复
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
file_handler = logging.FileHandler('debug.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 定义文件路径和 API 地址
BASE_DIR = os.path.dirname(__file__)
TEST_DATA_PATH = os.path.join(BASE_DIR, 'test_data', 'kline_test_data.csv')
sys.path.append(os.path.join(BASE_DIR, 'strategies'))
API_URL = "http://47.251.103.179:8000/kline"
HEALTH_URL = "http://47.251.103.179:8000/health"

# 解析命令行参数
parser = argparse.ArgumentParser(description='Backtrader Plotly with Test Data Option')
parser.add_argument('--use-test-data', action='store_true', help='Use test data instead of API')
args = parser.parse_args()

# 检查 API 健康状态
def check_api_health():
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        response.raise_for_status()
        logger.info(f"API 健康检查成功：{response.json()}")
        return True
    except Exception as e:
        logger.error(f"API 健康检查失败：{e}")
        return False

# 获取并初始化所有策略（避免重复注册）
STRATEGIES = {}
strategy_classes = registry.get_all_strategies()
if not strategy_classes:
    registry.discover_strategies("strategies")
    strategy_classes = registry.get_all_strategies()
# 去重策略
unique_strategy_classes = {}
for strategy_name, strategy_class in strategy_classes.items():
    if strategy_name not in unique_strategy_classes:
        unique_strategy_classes[strategy_name] = strategy_class
for strategy_name, strategy_class in unique_strategy_classes.items():
    try:
        strategy_instance = strategy_class()
        STRATEGIES[strategy_name] = {
            'class': strategy_class,
            'display_name': strategy_instance.display_name,
            'description': strategy_instance.description,
            'params': strategy_instance.params,
            'param_schema': strategy_instance.param_schema
        }
        logger.info(f"注册策略: {strategy_name}")
    except Exception as e:
        logger.error(f"初始化策略 {strategy_name} 失败：{e}")

# 查询 API 时间范围
def get_api_time_range(coin_name='LEAUSDT', interval='1m'):
    try:
        # 获取最早时间
        params_earliest = {
            "coin_name": coin_name,
            "interval": interval,
            "limit": 1
        }
        response_earliest = requests.get(API_URL, params=params_earliest, timeout=10)
        response_earliest.raise_for_status()
        data_earliest = response_earliest.json().get("data", [])
        if not data_earliest:
            logger.warning("API 返回空数据（最早时间），尝试更早时间")
            return datetime.now() - timedelta(days=180), datetime.now()

        min_timestamp = pd.to_datetime(data_earliest[0]["timestamp"])

        # 获取最晚时间（从当前时间向前尝试）
        max_attempts = 30
        current_time = datetime.now()
        max_timestamp = None
        for attempt in range(max_attempts):
            params_latest = {
                "coin_name": coin_name,
                "interval": interval,
                "limit": 1,
                "start_time": (current_time - timedelta(days=attempt)).isoformat()
            }
            response_latest = requests.get(API_URL, params=params_latest, timeout=10)
            response_latest.raise_for_status()
            data_latest = response_latest.json().get("data", [])
            if data_latest:
                max_timestamp = pd.to_datetime(data_latest[0]["timestamp"])
                break
        if max_timestamp is None:
            logger.warning("无法获取最新数据，设置默认时间范围")
            return datetime.now() - timedelta(days=180), datetime.now()

        return min_timestamp, max_timestamp
    except Exception as e:
        logger.error(f"查询 API 时间范围错误：{e}")
        return datetime.now() - timedelta(days=180), datetime.now()

# 公共数据加载和重采样函数
def load_and_resample_data(start_time, cycle, coin_name='LEAUSDT', interval='1m'):
    """加载并重采样 K 线数据"""
    try:
        interval_minutes = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1D': 1440, '1W': 10080, '1M': 43200}
        cycle_minutes = interval_minutes.get(cycle, 1)
        required_candles = 2000 if cycle == '1m' else 2000 * (cycle_minutes // interval_minutes[interval])
        data = load_kline_data(coin_name, interval, start_timestamp=start_time, limit=required_candles)
        if data.empty:
            logger.warning(f"加载数据为空：开始时间={start_time}")
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        resampled_data = resample_kline_data(data, cycle)
        if len(resampled_data) < 300:
            logger.warning(f"重采样后数据不足，周期={cycle}，K线数量={len(resampled_data)}")
        return resampled_data
    except Exception as e:
        logger.error(f"加载和重采样数据失败：{e}, 参数={locals()}")
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

# 数据加载函数（带 LRU 缓存）
@lru_cache(maxsize=10)
def load_kline_data(coin_name='LEAUSDT', interval='1m', start_timestamp=None, limit=None, chunk_size=10000):
    if args.use_test_data and os.path.exists(TEST_DATA_PATH):
        logger.info(f"从测试数据加载：{TEST_DATA_PATH}")
        try:
            data = pd.read_csv(TEST_DATA_PATH, index_col='timestamp', parse_dates=True)
            data.index = pd.to_datetime(data.index, unit='s')
            return data
        except Exception as e:
            logger.error(f"加载测试数据失败：{e}, 路径={TEST_DATA_PATH}")
            return pd.DataFrame()
    
    try:
        params = {
            "coin_name": coin_name,
            "interval": interval,
            "limit": limit if limit else chunk_size
        }
        if start_timestamp:
            params["start_time"] = start_timestamp.isoformat()
        
        data_chunks = []
        total_limit = limit if limit else float('inf')
        current_start = start_timestamp
        while len(data_chunks) * chunk_size < total_limit:
            params["limit"] = min(chunk_size, total_limit - len(data_chunks) * chunk_size)
            if current_start:
                params["start_time"] = current_start.isoformat()
            response = requests.get(API_URL, params=params, timeout=10)
            response.raise_for_status()
            kline_data = response.json().get("data", [])
            if not kline_data:
                break
            df_chunk = pd.DataFrame(kline_data)
            df_chunk['timestamp'] = pd.to_datetime(df_chunk['timestamp'])
            df_chunk.set_index('timestamp', inplace=True)
            df_chunk = df_chunk[['open', 'high', 'low', 'close', 'volume']]
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df_chunk[col] = pd.to_numeric(df_chunk[col], errors='coerce')
            data_chunks.append(df_chunk)
            # 更新 start_time 为最后一条记录的时间，获取更早的数据
            current_start = df_chunk.index.min()
        if not data_chunks:
            logger.warning(f"API 返回空数据：参数={params}")
            return pd.DataFrame()
        data = pd.concat(data_chunks).sort_index()
        data = data[~data.index.duplicated(keep='last')]
        data = data.dropna()
        if not data.index.is_monotonic_increasing:
            data = data.sort_index()
        return data
    except Exception as e:
        logger.error(f"API 数据加载错误：{e}, 参数={params}")
        return pd.DataFrame()

# 数据重采样函数
def resample_kline_data(data, target_interval):
    if target_interval == '1m':
        return data
    interval_map = {'5m': '5min', '15m': '15min', '30m': '30min', '1h': '1h', '4h': '4h', '1D': '1D', '1W': '1W', '1M': '1M'}
    if target_interval not in interval_map:
        logger.error(f"不支持的周期：{target_interval}")
        return data
    resample_rule = interval_map[target_interval]
    resampled_data = data.resample(resample_rule).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    if len(resampled_data) < 300:
        logger.warning(f"重采样后数据不足，周期={target_interval}，K线数量={len(resampled_data)}")
    return resampled_data

# 导出测试数据函数
def export_test_data(output_path, coin_name='LEAUSDT', interval='1m', limit=2000):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        start_time = datetime.now() - timedelta(minutes=limit)
        data = load_kline_data(coin_name, interval, start_timestamp=start_time, limit=limit)
        if data.empty:
            logger.error("无法加载测试数据，数据为空")
            return
        data.reset_index().to_csv(output_path, index=False)
        logger.info(f"测试数据已导出到：{output_path}")
    except Exception as e:
        logger.error(f"导出测试数据失败：{e}")
        return

# 绘制 K 线图函数（带缓存）
@lru_cache(maxsize=10)
def create_plotly_kline_plot_cached(
    data_tuple, backtest_data_tuple, buy_signals_tuple, sell_signals_tuple, indicator_data_tuple,
    indicator_config_tuple, xaxis_range_tuple, show_volume, show_indicators, show_signals, cycle
):
    try:
        data = pd.DataFrame(data_tuple[0], columns=data_tuple[1], index=pd.to_datetime(data_tuple[2]))
        backtest_data = pd.DataFrame(backtest_data_tuple[0], columns=backtest_data_tuple[1], index=pd.to_datetime(backtest_data_tuple[2]))
        buy_signals = pd.read_json(StringIO(buy_signals_tuple)) if buy_signals_tuple else None
        sell_signals = pd.read_json(StringIO(sell_signals_tuple)) if sell_signals_tuple else None
        indicator_data = json.loads(indicator_data_tuple) if indicator_data_tuple else None
        indicator_config = json.loads(indicator_config_tuple) if indicator_config_tuple else None
        # 规范化时间范围精度（秒级）
        xaxis_range = [pd.to_datetime(t).floor('s').isoformat() for t in xaxis_range_tuple] if xaxis_range_tuple else None
        logger.debug(f"缓存绘图调用：xaxis_range={xaxis_range}, show_volume={show_volume}, cycle={cycle}")
        return create_plotly_kline_plot(
            data, backtest_data, buy_signals, sell_signals, indicator_data,
            indicator_config, xaxis_range, show_volume, show_indicators, show_signals, cycle
        )
    except Exception as e:
        logger.error(f"缓存绘图失败：{e}")
        return go.Figure()

# 绘制 K 线图函数
def create_plotly_kline_plot(
    data, backtest_data, buy_signals=None, sell_signals=None, indicator_data=None,
    indicator_config=None, xaxis_range=None, show_volume=False, show_indicators=False,
    show_signals=False, cycle='1m'
):
    start_time = time.time()
    if data.empty:
        logger.warning("无数据，无法绘图")
        fig = go.Figure()
        default_start = pd.to_datetime('2025-04-19 00:00:00')
        default_end = pd.to_datetime('2025-04-19 23:59:59')
        fig.update_layout(
            xaxis_rangeslider_visible=True, dragmode='pan', template='plotly_dark',
            hovermode='x unified', showlegend=True, height=600, width=1200,
            xaxis=dict(
                range=[default_start, default_end],
                rangeselector=dict(buttons=[
                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(step="all", label="全部")
                ]),
                type="date", tickformat='%H:%M:%S'
            )
        )
        return fig
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    if not data.index.is_monotonic_increasing:
        data = data.sort_index()
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logger.error(f"数据缺少必要列：{missing_columns}")
        return go.Figure()
    # 设置时间轴范围
    interval_minutes = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1D': 1440, '1W': 10080, '1M': 43200}
    display_candles = 300
    if xaxis_range:
        try:
            range_start = pd.to_datetime(xaxis_range[0]).floor('s')
            range_end = pd.to_datetime(xaxis_range[1]).floor('s')
            if range_start > range_end:
                range_start, range_end = range_end, range_start
                xaxis_range = [range_start.isoformat(), range_end.isoformat()]
            if range_start < data.index.min():
                range_start = data.index.min()
            if range_end > data.index.max():
                range_end = data.index.max()
        except Exception as e:
            logger.error(f"xaxis_range 转换失败：{xaxis_range}, 错误：{e}")
            range_start = data.index.min()
            range_end = data.index.max()
    else:
        range_end = data.index.max()
        range_start = range_end - timedelta(minutes=display_candles * interval_minutes.get(cycle, 1))
        if range_start < data.index.min():
            range_start = data.index.min()
    # 动态创建子图
    rows = 1
    row_heights = [1.0]
    subplot_titles = ['K线图']
    if show_volume:
        rows += 1
        row_heights = [0.7, 0.3] if rows == 2 else [0.5, 0.2, 0.3]
        subplot_titles.append('成交量')
    if show_indicators and indicator_data:
        rows += 1
        row_heights = [0.5, 0.2, 0.3] if rows == 3 else [0.7, 0.3]
        subplot_titles.append('指标')
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        subplot_titles=subplot_titles, row_heights=row_heights
    )
    # 绘制 K 线图
    fig.add_trace(
        go.Candlestick(
            x=data.index, open=data['open'], high=data['high'],
            low=data['low'], close=data['close'], name='K线'
        ),
        row=1, col=1
    )
    # 绘制买卖信号
    if show_signals and buy_signals is not None and not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals['EntryTime'], y=buy_signals['EntryPrice'],
                mode='markers', name='买入', marker=dict(symbol='triangle-up', size=10, color='green'),
                text=[f"买入价格: {price:.2f}" for price in buy_signals['EntryPrice']],
                hoverinfo='text+x'
            ),
            row=1, col=1
        )
    if show_signals and sell_signals is not None and not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals['EntryTime'], y=sell_signals['EntryPrice'],
                mode='markers', name='卖出', marker=dict(symbol='triangle-down', size=10, color='red'),
                text=[f"卖出价格: {price:.2f}" for price in sell_signals['EntryPrice']],
                hoverinfo='text+x'
            ),
            row=1, col=1
        )
    # 绘制成交量
    if show_volume:
        fig.add_trace(
            go.Bar(
                x=data.index, y=data['volume'], name='成交量', marker_color='blue'
            ),
            row=2 if not (show_indicators and indicator_data) else 2,
            col=1
        )
    # 绘制指标（支持多策略）
    if show_indicators and indicator_data:
        for idx, ind_data in enumerate(indicator_data):
            if ind_data and 'values' in ind_data and len(ind_data['values']) > 0:
                indicator_series = pd.Series(ind_data['values'], index=backtest_data.index)
                fig.add_trace(
                    go.Scatter(
                        x=indicator_series.index, y=indicator_series,
                        name=ind_data.get('name', f'Indicator_{idx}'), line=dict(color='purple')
                    ),
                    row=3 if show_volume else 2, col=1
                )
                if indicator_config and idx < len(indicator_config):
                    for shape in indicator_config[idx].get('shapes', []):
                        fig.add_shape(
                            type='line', x0=data.index.min(), x1=data.index.max(),
                            y0=shape['y'], y1=shape['y'],
                            line=dict(color=shape.get('color', 'gray'), dash=shape.get('dash', 'dash')),
                            row=3 if show_volume else 2, col=1
                        )
    # 设置动态时间轴刻度
    interval_map = {
        '1m': {'tickformat': '%H:%M:%S', 'base_dtick': 1800000},
        '5m': {'tickformat': '%H:%M', 'base_dtick': 300000},
        '15m': {'tickformat': '%H:%M', 'base_dtick': 900000},
        '30m': {'tickformat': '%H:%M', 'base_dtick': 1800000},
        '1h': {'tickformat': '%m-%d %H:%M', 'base_dtick': 3600000 * 4},
        '4h': {'tickformat': '%m-%d %H:%M', 'base_dtick': 3600000 * 12},
        '1D': {'tickformat': '%Y-%m-%d', 'base_dtick': 86400000},
        '1W': {'tickformat': '%Y-%m-%d', 'base_dtick': 86400000 * 7},
        '1M': {'tickformat': '%Y-%m', 'base_dtick': 86400000 * 30}
    }
    time_span = (range_end - range_start).total_seconds() / 60
    candle_count = time_span / interval_minutes.get(cycle, 1)
    target_ticks = 15
    base_dtick = interval_map.get(cycle, {'base_dtick': 1800000})['base_dtick']
    adjusted_dtick = base_dtick * (candle_count // target_ticks) if candle_count > target_ticks else base_dtick
    xaxis_layout = dict(
        range=[range_start, range_end],
        rangeselector=dict(buttons=[
            dict(count=1, label="1h", step="hour", stepmode="backward"),
            dict(count=1, label="1d", step="day", stepmode="backward"),
            dict(step="all", label="全部")
        ]),
        type="date",
        tickformat=interval_map.get(cycle, {'tickformat': '%H:%M:%S'})['tickformat'],
        dtick=adjusted_dtick
    )
    fig.update_layout(
        xaxis_rangeslider_visible=True, xaxis_rangeslider=dict(range=[range_start, range_end]),
        dragmode='pan', template='plotly_dark', hovermode='x unified',
        showlegend=True, height=800 if show_indicators else 600, width=1200,
        xaxis=xaxis_layout
    )
    for i in range(2, rows + 1):
        fig.update_layout(**{f'xaxis{i}': xaxis_layout})
    logger.info(f"图表渲染耗时：{time.time() - start_time:.2f}秒")
    return fig

# 初始化 Dash 应用
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, 'assets/custom.css'], suppress_callback_exceptions=True)

# 检查 API 健康状态
if not check_api_health():
    logger.warning("API 不可用，可能会影响数据加载")

# 设置时间范围
MIN_DATE, MAX_DATE = get_api_time_range()
# 从最早时间开始加载数据
DEFAULT_START_DATE = MIN_DATE.replace(second=0, microsecond=0)  # 精确到分钟
DEFAULT_END_DATE = (DEFAULT_START_DATE + timedelta(minutes=2000)).replace(second=0, microsecond=0)

# 加载初始数据（从最早时间开始加载）
KLINE_CACHE = {}
all_data = None
attempts = 0
max_attempts = 5
current_start_date = DEFAULT_START_DATE
while attempts < max_attempts:
    logger.info(f"尝试加载初始数据，尝试次数：{attempts + 1}，开始时间：{current_start_date}")
    all_data = load_and_resample_data(current_start_date, '1m')
    if not all_data.empty:
        break
    attempts += 1
    current_start_date += timedelta(days=1)  # 向前尝试一天

if all_data.empty:
    logger.error("初始数据加载失败，使用空数据")
    all_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
all_data.index = pd.to_datetime(all_data.index)
all_data.index.name = 'timestamp'

# 设置初始时间轴范围（最新的 300 根 K 线）
if not all_data.empty:
    initial_end = all_data.index.max()
    initial_start = all_data.index[-300] if len(all_data) >= 300 else all_data.index.min()
    initial_xaxis_range = [initial_start.isoformat(), initial_end.isoformat()]
else:
    initial_xaxis_range = [DEFAULT_START_DATE.isoformat(), min(DEFAULT_END_DATE, MAX_DATE).isoformat()]

# 创建初始图表
initial_plotly_fig = create_plotly_kline_plot(
    all_data, all_data, xaxis_range=initial_xaxis_range,
    show_volume=True, show_indicators=False, show_signals=False, cycle='1m'
)

# 初始化回测缓存
BACKTEST_CACHE = {}

# 定义 Dash 应用布局
app.layout = dbc.Container([
    html.H1('LEAUSDT 策略回测', style={'textAlign': 'center', 'color': 'white'}),
    dbc.Row([
        # 左侧控制面板
        dbc.Col([
            html.Label('选择策略（可多选）', style={'color': 'white'}),
            dcc.Dropdown(
                id='strategy-select',
                options=[{'label': STRATEGIES[k]['display_name'], 'value': k} for k in STRATEGIES.keys()],
                value=None, multi=True,
                placeholder='请选择策略',
                style={'backgroundColor': '#333', 'color': 'white', 'borderColor': '#555'}
            ),
            html.Br(),
            html.Label('选择回测时间范围', style={'color': 'white'}),
            dcc.DatePickerRange(
                id='date-range',
                min_date_allowed=MIN_DATE,
                max_date_allowed=MAX_DATE,
                start_date=DEFAULT_START_DATE,
                end_date=DEFAULT_END_DATE,
                display_format='YYYY-MM-DD HH:mm',  # 显示小时和分钟
                style={'backgroundColor': '#333', 'color': 'white', 'borderColor': '#555'}
            ),
            html.Br(),
            html.Label('图层显示', style={'color': 'white'}),
            dcc.Checklist(
                id='layer-select',
                options=[
                    {'label': '显示成交量', 'value': 'volume'},
                    {'label': '显示指标', 'value': 'indicators'},
                    {'label': '显示买卖点', 'value': 'signals'}
                ],
                value=['volume', 'indicators', 'signals'],
                style={'color': 'white'}
            ),
            html.Br(),
            html.Label('加载历史数据', style={'color': 'white'}),
            dbc.Button('加载更多', id='load-more', color='secondary', n_clicks=0),
            html.Br(),
            html.Label('添加标注', style={'color': 'white'}),
            dcc.Input(id='annotation-time', type='text', placeholder='时间 (YYYY-MM-DD HH:MM:SS)', style={'width': '100%', 'marginBottom': '5px'}),
            dcc.Input(id='annotation-text', type='text', placeholder='标注文本', style={'width': '100%', 'marginBottom': '5px'}),
            dbc.Button('添加垂直线', id='add-annotation', color='primary', n_clicks=0),
            html.Div(id='strategy-params', style={'color': 'white'}),
        ], width=3),
        # 右侧图表和结果区域
        dbc.Col([
            html.Div(
                style={'position': 'relative'},
                children=[
                    html.Div(
                        children=[
                            html.Label('选择周期', style={'color': 'white', 'marginRight': '10px'}),
                            dcc.Dropdown(
                                id='cycle-select',
                                options=[
                                    {'label': '1分钟 (1m)', 'value': '1m'}, {'label': '5分钟 (5m)', 'value': '5m'},
                                    {'label': '15分钟 (15m)', 'value': '15m'}, {'label': '30分钟 (30m)', 'value': '30m'},
                                    {'label': '1小时 (1h)', 'value': '1h'}, {'label': '4小时 (4h)', 'value': '4h'},
                                    {'label': '1天 (1D)', 'value': '1D'}, {'label': '1周 (1W)', 'value': '1W'},
                                    {'label': '1月 (1M)', 'value': '1M'}
                                ],
                                value='1m',
                                className='cycle-dropdown',
                                style={
                                    'backgroundColor': '#ffffff',
                                    'color': '#000000',
                                    'borderColor': '#000000',
                                    'width': '150px',
                                    'display': 'inline-block',
                                    'verticalAlign': 'middle'
                                }
                            ),
                        ],
                        style={
                            'position': 'absolute',
                            'top': '10px',
                            'left': '10px',
                            'zIndex': 1000,
                            'display': 'flex',
                            'alignItems': 'center'
                        }
                    ),
                    dcc.Loading(id='loading', children=[
                        html.Div(id='chart-container', children=[
                            dcc.Graph(id='plotly-kline-plot', figure=initial_plotly_fig, config={'scrollZoom': True, 'doubleClick': 'reset'})
                        ])
                    ]),
                ]
            ),
            html.Div(id='stats-output', style={'color': 'white'}),
            html.Br(),
            html.Label('交易详情', style={'color': 'white'}),
            dash_table.DataTable(
                id='trade-table',
                columns=[
                    {'name': '时间', 'id': 'EntryTime'},
                    {'name': '类型', 'id': 'Type'},
                    {'name': '价格', 'id': 'EntryPrice'},
                    {'name': '策略', 'id': 'Strategy'}
                ],
                data=[],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '5px', 'color': 'white', 'backgroundColor': '#333'},
                style_header={'backgroundColor': '#444', 'color': 'white'},
                page_size=10
            )
        ], width=9)
    ]),
    dcc.Store(id='data-store', data=all_data.to_json(date_format='iso', orient='split')),
    dcc.Store(id='backtest-data-store', data=all_data.to_json(date_format='iso', orient='split')),
    dcc.Store(id='xaxis-range', data=initial_xaxis_range),
    dcc.Store(id='backtest-results', data=json.dumps([])),
    dcc.Store(id='annotations', data=[]),
], fluid=True)

# 回调函数：更新时间轴范围
@app.callback(
    Output('xaxis-range', 'data'),
    Input('plotly-kline-plot', 'relayoutData'),
    State('xaxis-range', 'data'),
    prevent_initial_call=True
)
def update_xaxis_range(relayout_data, current_range):
    last_update_time = getattr(update_xaxis_range, '_last_update_time', 0)
    current_time = time.time()
    if current_time - last_update_time < 1.5:
        logger.debug("时间轴更新被节流")
        return no_update
    update_xaxis_range._last_update_time = current_time

    if not relayout_data or not isinstance(relayout_data, dict) or ('xaxis.range[0]' not in relayout_data and 'xaxis.range[1]' not in relayout_data):
        logger.debug("无效的 relayoutData，跳过更新")
        return no_update

    try:
        range_start = pd.to_datetime(relayout_data.get('xaxis.range[0]', current_range[0])).floor('s')
        range_end = pd.to_datetime(relayout_data.get('xaxis.range[1]', current_range[1])).floor('s')
        if range_start > range_end:
            range_start, range_end = range_end, range_start
        if range_start < MIN_DATE:
            range_start = MIN_DATE
        if range_end > MAX_DATE:
            range_end = MAX_DATE
        new_range = [range_start.isoformat(), range_end.isoformat()]
        # 检查时间跨度变化
        current_span = (pd.to_datetime(current_range[1]) - pd.to_datetime(current_range[0])).total_seconds()
        new_span = (range_end - range_start).total_seconds()
        if abs(new_span - current_span) / current_span < 0.1 and abs(new_span - current_span) < 60:  # 小于10%且小于1分钟
            logger.debug("时间跨度变化过小，跳过更新")
            return no_update
        logger.debug(f"更新时间轴范围：{new_range}")
        return new_range
    except Exception as e:
        logger.error(f"relayoutData xaxis_range 转换失败：{relayout_data}, 错误：{e}")
        return no_update

# 回调函数：加载更多数据
@app.callback(
    [Output('data-store', 'data', allow_duplicate=True),
     Output('backtest-data-store', 'data', allow_duplicate=True)],
    [Input('xaxis-range', 'data'),
     Input('cycle-select', 'value')],
    [State('data-store', 'data'),
     State('backtest-data-store', 'data')],
    prevent_initial_call=True
)
def load_more_data(xaxis_range, cycle, stored_data, backtest_data):
    global all_data
    # 节流控制
    last_load_time = getattr(load_more_data, '_last_load_time', 0)
    current_time = time.time()
    if current_time - last_load_time < 1.0:
        logger.debug("数据加载被节流")
        return no_update, no_update
    load_more_data._last_load_time = current_time

    # 反序列化数据
    try:
        all_data = pd.read_json(StringIO(stored_data), orient='split')
        all_data.index = pd.to_datetime(all_data.index)
        all_data.index.name = 'timestamp'
    except Exception as e:
        logger.error(f"数据反序列化失败：{e}")
        return no_update, no_update

    # 检查时间范围和 K 线数量
    range_start = pd.to_datetime(xaxis_range[0]).floor('s')
    range_end = pd.to_datetime(xaxis_range[1]).floor('s')
    interval_minutes = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1D': 1440, '1W': 10080, '1M': 43200}
    cycle_minutes = interval_minutes.get(cycle, 1)
    
    # 计算当前范围内的 K 线数量
    visible_data = all_data[(all_data.index >= range_start) & (all_data.index <= range_end)]
    visible_candles = len(visible_data)
    logger.debug(f"当前可见 K 线数量：{visible_candles}, 范围：{range_start} 至 {range_end}, 数据边界：{all_data.index.min()} 至 {all_data.index.max()}")

    # 检查是否需要加载更多数据（左边界）
    data_updated = False
    if (visible_candles < 300 or range_start < all_data.index.min()) and range_start > MIN_DATE:
        load_candles_1m = 300 if cycle == '1m' else 300 * (cycle_minutes // interval_minutes['1m'])
        load_minutes = load_candles_1m * interval_minutes['1m']
        load_end = max(all_data.index.min(), range_start)
        load_start = load_end - timedelta(minutes=load_minutes)
        if load_start < MIN_DATE:
            load_start = MIN_DATE
        logger.info(f"加载历史数据：开始时间={load_start}, 周期={cycle}, 加载 1m K 线数量={load_candles_1m}")
        new_data = load_kline_data('LEAUSDT', '1m', start_timestamp=load_start, limit=load_candles_1m)
        if new_data is not None and not new_data.empty:
            all_data = pd.concat([new_data, all_data]).sort_index()
            all_data = all_data[~all_data.index.duplicated(keep='last')]
            all_data = resample_kline_data(all_data, cycle)
            data_updated = True
            logger.info(f"加载新数据成功，新增 K 线数量：{len(new_data)}")
        else:
            logger.warning(f"未加载到新数据：开始时间={load_start}")

    if not data_updated:
        return no_update, no_update

    return all_data.to_json(date_format='iso', orient='split'), all_data.to_json(date_format='iso', orient='split')

# 回调函数：更新参数和数据
@app.callback(
    [Output('strategy-params', 'children'),
     Output('data-store', 'data', allow_duplicate=True),
     Output('backtest-data-store', 'data', allow_duplicate=True),
     Output('backtest-results', 'data')],
    [Input('cycle-select', 'value'),
     Input('strategy-select', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input({'type': 'param', 'index': dash.ALL}, 'value')],
    [State('data-store', 'data'),
     State('backtest-data-store', 'data'),
     State('strategy-params', 'children')],
    prevent_initial_call=True
)
def update_params_and_data(cycle, strategies, start_date, end_date, param_values, stored_data, backtest_data, param_children):
    global all_data
    start_time = pd.to_datetime(start_date).floor('s')
    end_time = pd.to_datetime(end_date).floor('s')
    if start_time > end_time:
        logger.warning(f"start_date ({start_time}) 晚于 end_date ({end_time})，交换两者")
        start_time, end_time = end_time, start_time
    display_data = load_and_resample_data(start_time, cycle)
    backtest_data = display_data.copy()
    all_data = display_data.copy()
    display_data.index.name = 'timestamp'
    backtest_data.index.name = 'timestamp'
    inputs = []
    params_dict = {}
    if strategies:
        for strategy in strategies:
            if strategy in STRATEGIES:
                param_schema = STRATEGIES[strategy]['param_schema']
                for param, schema in param_schema.items():
                    if param == 'size':
                        continue
                    input_type = 'number' if schema['type'] in ['integer', 'float'] else 'text'
                    inputs.extend([
                        html.Label(f"{strategy}: {param.capitalize()} ({schema['description']})", style={'color': 'white', 'marginTop': '10px'}),
                        dcc.Input(
                            id={'type': 'param', 'index': f"{strategy}_{param}"},
                            type=input_type, value=schema['default'],
                            min=schema.get('min', None), max=schema.get('max', None),
                            step=0.1 if schema['type'] == 'float' else 1,
                            debounce=True,
                            style={'backgroundColor': '#333', 'color': 'white', 'borderColor': '#555', 'width': '100%'}
                        )
                    ])
    if param_children:
        param_names = []
        for child in param_children[::2]:
            try:
                label_content = child['props']['children']
                strategy_name, param_desc = label_content.split(': ', 1)
                param_name = param_desc.split(' (')[0].lower()
                param_key = f"{strategy_name}_{param_name}"
                param_names.append(param_key)
            except Exception as e:
                logger.error(f"解析参数标签失败：{label_content}, 错误：{e}")
                continue
        for name, value in zip(param_names, param_values):
            if name and value is not None:
                try:
                    strategy, param = name.split('_', 1)
                    schema = STRATEGIES[strategy]['param_schema'][param]
                    if schema['type'] == 'integer':
                        params_dict.setdefault(strategy, {})[param] = int(value)
                    elif schema['type'] == 'float':
                        params_dict.setdefault(strategy, {})[param] = float(value)
                    else:
                        params_dict.setdefault(strategy, {})[param] = value
                except (ValueError, KeyError) as e:
                    logger.error(f"参数 {name} 转换失败：{value}, 错误：{e}")
                    try:
                        strategy, param = name.split('_', 1)
                        params_dict.setdefault(strategy, {})[param] = STRATEGIES[strategy]['param_schema'][param]['default']
                    except Exception as inner_e:
                        logger.error(f"参数默认值设置失败：{inner_e}, 策略={strategy}, 参数={param}")
    backtest_results = []
    if strategies:
        for strategy in strategies:
            params = params_dict.get(strategy, {})
            params['size'] = STRATEGIES[strategy]['params'].get('size', 0.1)
            strategy_instance = STRATEGIES[strategy]['class']()
            # 根据回测时间范围和周期加载数据
            interval_minutes = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1D': 1440, '1W': 10080, '1M': 43200}
            cycle_minutes = interval_minutes.get(cycle, 1)
            required_1m_candles = int((end_time - start_time).total_seconds() / 60 / interval_minutes['1m'])
            load_start = start_time
            backtest_1m_data = load_kline_data('LEAUSDT', '1m', start_timestamp=load_start, limit=required_1m_candles)
            if backtest_1m_data.empty:
                logger.error(f"回测数据加载失败：开始时间={load_start}")
                backtest_1m_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            backtest_data_cycle = resample_kline_data(backtest_1m_data, cycle)
            cache_key = f"{strategy}_{hash(str(params))}_{hash(str(backtest_data_cycle.index[0]))}_{hash(str(backtest_data_cycle.index[-1]))}"
            if cache_key in BACKTEST_CACHE:
                stats, buy_signals, sell_signals = BACKTEST_CACHE[cache_key]
            else:
                try:
                    stats, buy_signals, sell_signals = strategy_instance.run_backtest(backtest_data_cycle, params)
                    if not buy_signals.empty and 'EntryTime' in buy_signals.columns:
                        buy_signals = buy_signals[buy_signals['EntryTime'].between(start_time, end_time)]
                    if not sell_signals.empty and 'EntryTime' in sell_signals.columns:
                        sell_signals = sell_signals[sell_signals['EntryTime'].between(start_time, end_time)]
                    BACKTEST_CACHE[cache_key] = (stats, buy_signals, sell_signals)
                except Exception as e:
                    logger.error(f"回测失败（{strategy}）：{e}")
                    stats, buy_signals, sell_signals = None, pd.DataFrame(), pd.DataFrame()
            indicator_data = None
            indicator_config = {}
            try:
                values = strategy_instance.compute_indicator(backtest_data_cycle, params)
                if values is not None:
                    indicator_series = pd.Series(values, index=backtest_data_cycle.index, name=strategy.upper())
                    indicator_df = indicator_series.reset_index()
                    indicator_data = {
                        'name': strategy.upper(),
                        'values': indicator_df[strategy.upper()].values.tolist()
                    }
                    indicator_config = strategy_instance.indicator_config(backtest_data_cycle, params) or {}
            except Exception as e:
                logger.error(f"指标计算错误（{strategy}）：{e}")
            backtest_results.append({
                'strategy': strategy,
                'params': params,
                'stats': stats if stats is not None else {},
                'buy_signals': buy_signals.to_json(date_format='iso', orient='split') if buy_signals is not None and not buy_signals.empty else None,
                'sell_signals': sell_signals.to_json(date_format='iso', orient='split') if sell_signals is not None and not sell_signals.empty else None,
                'indicator_data': indicator_data,
                'indicator_config': indicator_config
            })
    return (
        inputs,
        display_data.to_json(date_format='iso', orient='split'),
        backtest_data.to_json(date_format='iso', orient='split'),
        json.dumps(backtest_results)
    )

# 回调函数：更新图表和结果
@app.callback(
    [Output('chart-container', 'children'),
     Output('stats-output', 'children'),
     Output('trade-table', 'data')],
    [Input('data-store', 'data'),
     Input('backtest-data-store', 'data'),
     Input('layer-select', 'value'),
     Input('xaxis-range', 'data'),
     Input('backtest-results', 'data'),
     Input('cycle-select', 'value')],
    [State('plotly-kline-plot', 'figure')],
    prevent_initial_call=True
)
def update_plot(display_data, backtest_data, layer_select, xaxis_range, backtest_results, cycle, current_fig):
    # 节流控制
    last_update_time = getattr(update_plot, '_last_update_time', 0)
    current_time = time.time()
    if current_time - last_update_time < 1.5:
        logger.debug("图表更新被节流")
        return no_update, no_update, no_update
    update_plot._last_update_time = current_time

    # 反序列化数据
    try:
        display_data = pd.read_json(StringIO(display_data), orient='split')
        display_data.index = pd.to_datetime(display_data.index)
        display_data.index.name = 'timestamp'
        backtest_data = pd.read_json(StringIO(backtest_data), orient='split')
        backtest_data.index = pd.to_datetime(backtest_data.index)
        backtest_data.index.name = 'timestamp'
    except Exception as e:
        logger.error(f"数据反序列化失败：{e}")
        return no_update, no_update, no_update

    # 处理回测结果
    stats_list, buy_signals_list, sell_signals_list, indicator_data_list, indicator_config_list = [], [], [], [], []
    if backtest_results:
        try:
            backtest_results = json.loads(backtest_results)
            for result in backtest_results:
                strategy = result.get('strategy', None)
                params = result.get('params', {})
                stats = result.get('stats', {})
                buy_signals = pd.read_json(StringIO(result['buy_signals']), orient='split') if result.get('buy_signals') else pd.DataFrame()
                buy_signals.index = pd.to_datetime(buy_signals.index)
                if 'EntryTime' in buy_signals.columns:
                    buy_signals['EntryTime'] = pd.to_datetime(buy_signals['EntryTime'])
                sell_signals = pd.read_json(StringIO(result['sell_signals']), orient='split') if result.get('sell_signals') else pd.DataFrame()
                sell_signals.index = pd.to_datetime(sell_signals.index)
                if 'EntryTime' in sell_signals.columns:
                    sell_signals['EntryTime'] = pd.to_datetime(sell_signals['EntryTime'])
                indicator_data = result.get('indicator_data', None)
                indicator_config = result.get('indicator_config', {})
                stats_list.append((strategy, params, stats))
                buy_signals_list.append((strategy, buy_signals))
                sell_signals_list.append((strategy, sell_signals))
                indicator_data_list.append((strategy, indicator_data))
                indicator_config_list.append((strategy, indicator_config))
        except Exception as e:
            logger.error(f"处理 backtest_results 失败：{e}")
            backtest_results = []
    show_volume = 'volume' in layer_select
    show_indicators = 'indicators' in layer_select
    show_signals = 'signals' in layer_select
    combined_buy_signals = None
    combined_sell_signals = None
    combined_indicator_data = []
    combined_indicator_config = []
    if backtest_results:
        for strategy, buy_signals in buy_signals_list:
            if combined_buy_signals is None:
                combined_buy_signals = buy_signals.copy()
                combined_buy_signals['Strategy'] = strategy
            else:
                buy_signals['Strategy'] = strategy
                combined_buy_signals = pd.concat([combined_buy_signals, buy_signals])
        for strategy, sell_signals in sell_signals_list:
            if combined_sell_signals is None:
                combined_sell_signals = sell_signals.copy()
                combined_sell_signals['Strategy'] = strategy
            else:
                sell_signals['Strategy'] = strategy
                combined_sell_signals = pd.concat([combined_sell_signals, sell_signals])
        for strategy, indicator_data in indicator_data_list:
            if indicator_data:
                indicator_data['name'] = f"{strategy.upper()}_{indicator_data.get('name', 'Indicator')}"
                combined_indicator_data.append(indicator_data)
        combined_indicator_config = [cfg[1] for cfg in indicator_config_list]
    if combined_buy_signals is not None:
        combined_buy_signals = combined_buy_signals.sort_values('EntryTime').drop_duplicates(subset=['EntryTime', 'Strategy'], keep='last')
    if combined_sell_signals is not None:
        combined_sell_signals = combined_sell_signals.sort_values('EntryTime').drop_duplicates(subset=['EntryTime', 'Strategy'], keep='last')
    # 绘制图表
    fig = create_plotly_kline_plot(
        display_data, backtest_data,
        combined_buy_signals if show_signals else None,
        combined_sell_signals if show_signals else None,
        combined_indicator_data if show_indicators else None,
        combined_indicator_config if show_indicators else None,
        xaxis_range=xaxis_range, show_volume=show_volume,
        show_indicators=show_indicators, show_signals=show_signals,
        cycle=cycle
    )
    chart_component = dcc.Graph(id='plotly-kline-plot', figure=fig, config={'scrollZoom': True, 'doubleClick': 'reset'})
    # 生成统计信息
    stats_text = []
    trade_data = []
    for strategy, params, stats in stats_list:
        stats_text.extend([
            html.H4(f"策略: {STRATEGIES.get(strategy, {}).get('display_name', '未知策略')}", style={'color': 'white'}),
            html.P(f"描述: {STRATEGIES.get(strategy, {}).get('description', '')}", style={'color': 'white'}),
            html.P(f"参数: {params}", style={'color': 'white'}),
            html.Hr(),
            html.P(f"总回报: {stats.get('Return [%]', 0):.2f}%", style={'color': 'white'}),
            html.P(f"最大回撤: {stats.get('Max. Drawdown [%]', 0):.2f}%", style={'color': 'white'}),
            html.P(f"交易次数: {stats.get('# Trades', 0)}", style={'color': 'white'}),
            html.P(f"胜率: {stats.get('Win Rate [%]', 0):.2f}%", style={'color': 'white'}),
            html.P(f"夏普比率: {stats.get('Sharpe Ratio', 0):.2f}", style={'color': 'white'}),
            html.P(f"年化波动率: {stats.get('Annualized Volatility [%]', 0):.2f}%", style={'color': 'white'})
        ])
        if combined_buy_signals is not None and not combined_buy_signals.empty and 'EntryTime' in combined_buy_signals.columns:
            buy_signals = combined_buy_signals[combined_buy_signals['Strategy'] == strategy]
            buy_signals['Type'] = '买入'
            trade_data.extend(buy_signals[['EntryTime', 'Type', 'EntryPrice', 'Strategy']].to_dict('records'))
        if combined_sell_signals is not None and not combined_sell_signals.empty and 'EntryTime' in combined_sell_signals.columns:
            sell_signals = combined_sell_signals[combined_sell_signals['Strategy'] == strategy]
            sell_signals['Type'] = '卖出'
            trade_data.extend(sell_signals[['EntryTime', 'Type', 'EntryPrice', 'Strategy']].to_dict('records'))
    trade_data = sorted(trade_data, key=lambda x: x['EntryTime']) if trade_data else []
    return [chart_component], stats_text, trade_data

# 回调函数：添加垂直线标注
@app.callback(
    Output('plotly-kline-plot', 'figure'),
    Input('add-annotation', 'n_clicks'),
    [State('plotly-kline-plot', 'figure'),
     State('xaxis-range', 'data'),
     State('annotation-time', 'value'),
     State('annotation-text', 'value'),
     State('annotations', 'data')],
    prevent_initial_call=True
)
def add_annotation(n_clicks, current_fig, xaxis_range, annotation_time, annotation_text, annotations):
    if n_clicks == 0:
        return no_update
    fig = go.Figure(current_fig)
    try:
        annotation_time = pd.to_datetime(annotation_time) if annotation_time else pd.to_datetime(xaxis_range[1])
    except:
        annotation_time = pd.to_datetime(xaxis_range[1])
    fig.add_vline(
        x=annotation_time, line_dash='dash', line_color='yellow',
        annotation_text=annotation_text or '事件'
    )
    annotations.append({'time': annotation_time.isoformat(), 'text': annotation_text or '事件'})
    return fig

# 主程序入口
if __name__ == '__main__':
    if not os.path.exists(TEST_DATA_PATH):
        export_test_data(TEST_DATA_PATH)
    else:
        logger.info(f"测试数据已存在，跳过导出：{TEST_DATA_PATH}")
    app.run(debug=False)  # 关闭调试模式，避免重复日志