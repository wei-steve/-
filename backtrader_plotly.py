import pandas as pd
import sqlite3
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
from strategy_registry import registry
from backtest_engine import backtest_engine

# 设置日志级别为 WARNING，仅输出关键信息
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
handler = logging.StreamHandler()
handler.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
file_handler = logging.FileHandler('debug.log')
file_handler.setLevel(logging.WARNING)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 加载策略
sys.path.append(r'D:\策略研究\回测功能\strategies')
import rsi_strategy
import bollinger_strategy

# 获取所有策略类
strategy_classes = registry.get_all_strategies()

# 创建 STRATEGIES 字典，提前解析 display_name 和 description
STRATEGIES = {}
for strategy_name, strategy_class in strategy_classes.items():
    strategy_instance = strategy_class()
    STRATEGIES[strategy_name] = {
        'class': strategy_class,
        'display_name': strategy_instance.display_name,
        'description': strategy_instance.description,
        'params': strategy_instance.params,
        'param_schema': strategy_instance.param_schema
    }

# 查询数据库时间范围
def get_db_time_range(db_path, coin_name='LEAUSDT', interval='1m'):
    try:
        conn = sqlite3.connect(db_path)
        query = f"""
            SELECT MIN(timestamp), MAX(timestamp)
            FROM kline_data
            WHERE coin_name = ? AND interval = ?
        """
        cursor = conn.cursor()
        cursor.execute(query, (coin_name, interval))
        min_timestamp, max_timestamp = cursor.fetchone()
        conn.close()
        if min_timestamp and max_timestamp:
            min_time = pd.to_datetime(min_timestamp, unit='s')
            max_time = pd.to_datetime(max_timestamp, unit='s')
            return min_time, max_time
        return datetime.now() - timedelta(days=180), datetime.now()
    except Exception as e:
        logger.error(f"查询时间范围错误：{e}")
        return datetime.now() - timedelta(days=180), datetime.now()

# 数据加载函数
def load_kline_data(db_path, coin_name='LEAUSDT', interval='1m', start_timestamp=None, end_timestamp=None, limit=None, chunk_size=10000):
    cache_key = f"{coin_name}_{interval}_{start_timestamp}_{end_timestamp}_{limit}"
    if cache_key in KLINE_CACHE:
        return KLINE_CACHE[cache_key]
    
    try:
        conn = sqlite3.connect(db_path)
        data_chunks = []
        offset = 0
        query_limit = limit if limit else -1
        while True:
            query = f"""
                SELECT timestamp, open, high, low, close, volume
                FROM kline_data
                WHERE coin_name = ? AND interval = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
                LIMIT ? OFFSET ?
            """
            params = (coin_name, interval, int(start_timestamp.timestamp()), int(end_timestamp.timestamp()), chunk_size, offset)
            df_chunk = pd.read_sql_query(query, conn, params=params)
            if df_chunk.empty:
                break
            df_chunk['timestamp'] = pd.to_datetime(df_chunk['timestamp'], unit='s')
            df_chunk.set_index('timestamp', inplace=True)
            df_chunk.columns = ['open', 'high', 'low', 'close', 'volume']
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df_chunk[col] = pd.to_numeric(df_chunk[col], errors='coerce')
            data_chunks.append(df_chunk)
            offset += chunk_size
            if limit and offset >= limit:
                break
        conn.close()
        if not data_chunks:
            logger.warning("数据库返回空数据")
            return pd.DataFrame()
        data = pd.concat(data_chunks).sort_index()
        data = data[~data.index.duplicated(keep='last')]
        data = data.dropna()
        if limit:
            data = data.tail(limit)
        if not data.index.is_monotonic_increasing:
            data = data.sort_index()
        KLINE_CACHE[cache_key] = data
        return data
    except Exception as e:
        logger.error(f"数据加载错误：{e}")
        return pd.DataFrame()

# Plotly 绘图函数
def create_plotly_kline_plot(
    data,
    backtest_data,
    buy_signals=None,
    sell_signals=None,
    indicator_data=None,
    indicator_config=None,
    xaxis_range=None,
    show_volume=False
):
    if data.empty:
        logger.warning("无数据，无法绘图")
        fig = go.Figure()
        default_start = pd.to_datetime('2025-04-19 00:00:00')
        default_end = pd.to_datetime('2025-04-19 23:59:59')
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            dragmode='pan',
            template='plotly_dark',
            hovermode='x unified',
            showlegend=True,
            height=600,
            width=1200,
            xaxis=dict(
                range=[default_start, default_end],
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label="1h", step="hour", stepmode="backward"),
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(step="all", label="全部")
                    ]
                ),
                type="date",
                tickformat='%H:%M:%S'
            )
        )
        return fig
    
    if not isinstance(data.index, pd.DatetimeIndex):
        logger.error(f"数据索引不是 DatetimeIndex：{data.index}")
        data.index = pd.to_datetime(data.index)

    if not data.index.is_monotonic_increasing:
        data = data.sort_index()

    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logger.error(f"数据缺少必要列：{missing_columns}")
        return go.Figure()
    
    for col in ['open', 'high', 'low', 'close']:
        if not pd.to_numeric(data[col], errors='coerce').notnull().all():
            logger.warning(f"原始数据列 {col} 包含 NaN 或非数值数据")
            data = data.dropna(subset=[col])

    if data.empty:
        logger.warning("数据清洗后为空，使用默认数据")
        default_data = pd.DataFrame({
            'open': [0.0050],
            'high': [0.0051],
            'low': [0.0049],
            'close': [0.0050],
            'volume': [1000]
        }, index=[pd.to_datetime('2025-04-19 12:07:00')])
        data = default_data

    if xaxis_range:
        try:
            range_start = pd.to_datetime(xaxis_range[0])
            range_end = pd.to_datetime(xaxis_range[1])
            if range_start > range_end:
                logger.warning(f"xaxis_range 无效，range_start ({range_start}) 晚于 range_end ({range_end})，交换两者")
                range_start, range_end = range_end, range_start
                xaxis_range = [range_start.isoformat(), range_end.isoformat()]
            if range_start < data.index.min():
                range_start = data.index.min()
                logger.warning(f"xaxis_range 开始时间早于数据范围，调整为：{range_start}")
            if range_end > data.index.max():
                range_end = data.index.max()
                logger.warning(f"xaxis_range 结束时间晚于数据范围，调整为：{range_end}")
        except Exception as e:
            logger.error(f"xaxis_range 转换失败：{xaxis_range}, 错误：{e}")
            range_start = data.index.min()
            range_end = data.index.max()
    else:
        range_end = data.index.max()
        range_start = data.index[-300] if len(data) >= 300 else data.index.min()
    
    # 裁剪 K 线显示数据
    display_data = data.loc[range_start:range_end]
    if display_data.empty:
        logger.warning(f"xaxis_range [{range_start}, {range_end}] 无数据，数据范围：{data.index.min()} 到 {data.index.max()}")
        display_data = data  # 使用整个数据范围
        range_start = data.index.min()
        range_end = data.index.max()
        xaxis_range = [range_start.isoformat(), range_end.isoformat()]
    
    for col in ['open', 'high', 'low', 'close']:
        if not pd.to_numeric(display_data[col], errors='coerce').notnull().all():
            logger.warning(f"裁剪后的 display_data 列 {col} 包含 NaN 或非数值数据")
            display_data = display_data.dropna(subset=[col])

    if display_data.empty:
        logger.warning("裁剪后 display_data 为空，使用默认数据")
        display_data = pd.DataFrame({
            'open': [0.0050],
            'high': [0.0051],
            'low': [0.0049],
            'close': [0.0050],
            'volume': [1000]
        }, index=[pd.to_datetime('2025-04-19 12:07:00')])
        range_start = display_data.index.min()
        range_end = display_data.index.max()
        xaxis_range = [range_start.isoformat(), range_end.isoformat()]

    # 动态确定子图数量和高度
    rows = 1  # 至少有一个 K 线图
    row_heights = [1.0]  # 默认只显示 K 线图
    subplot_titles = ['K线图']

    if show_volume:
        rows += 1
        row_heights = [0.7, 0.3] if rows == 2 else [0.5, 0.2, 0.3]
        subplot_titles.append('成交量')

    if indicator_data:
        rows += 1
        row_heights = [0.5, 0.2, 0.3] if rows == 3 else [0.7, 0.3]
        subplot_titles.append(indicator_data.get('name', 'Indicator'))
    
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=subplot_titles,
        row_heights=row_heights
    )
    
    # 添加 K 线图
    fig.add_trace(
        go.Candlestick(
            x=display_data.index,
            open=display_data['open'],
            high=display_data['high'],
            low=display_data['low'],
            close=display_data['close'],
            name='K线'
        ),
        row=1,
        col=1
    )
    
    # 添加买卖点（基于 backtest_data，但仅显示在 xaxis_range 范围内）
    if buy_signals is not None and not buy_signals.empty:
        buy_signals_filtered = buy_signals[buy_signals['EntryTime'].between(range_start, range_end)]
        if not buy_signals_filtered.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals_filtered['EntryTime'],
                    y=buy_signals_filtered['EntryPrice'],
                    mode='markers',
                    name='买入',
                    marker=dict(symbol='triangle-up', size=10, color='green'),
                    text=[f"买入价格: {price:.2f}" for price in buy_signals_filtered['EntryPrice']],
                    hoverinfo='text+x'
                ),
                row=1,
                col=1
            )
    if sell_signals is not None and not sell_signals.empty:
        sell_signals_filtered = sell_signals[sell_signals['EntryTime'].between(range_start, range_end)]
        if not sell_signals_filtered.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals_filtered['EntryTime'],
                    y=sell_signals_filtered['EntryPrice'],
                    mode='markers',
                    name='卖出',
                    marker=dict(symbol='triangle-down', size=10, color='red'),
                    text=[f"卖出价格: {price:.2f}" for price in sell_signals_filtered['EntryPrice']],
                    hoverinfo='text+x'
                ),
                row=1,
                col=1
            )
    
    # 添加成交量（基于 display_data）
    if show_volume:
        fig.add_trace(
            go.Bar(
                x=display_data.index,
                y=display_data['volume'],
                name='成交量',
                marker_color='blue'
            ),
            row=2 if not indicator_data else 2,
            col=1
        )
    
    # 添加指标（基于 backtest_data，但仅显示在 xaxis_range 范围内）
    if indicator_data and 'values' in indicator_data:
        indicator_series = pd.Series(indicator_data['values'], index=backtest_data.index)
        indicator_series_filtered = indicator_series.loc[range_start:range_end]
        if indicator_series_filtered.empty:
            # 如果指标数据为空，使用整个 backtest_data 的指标数据，并在图表上显示提示
            indicator_series_filtered = indicator_series
            fig.add_annotation(
                text="指标数据超出当前时间范围，请调整时间区间",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="red"),
                align="center",
                bordercolor="red",
                borderwidth=2,
                borderpad=4,
                bgcolor="rgba(255, 255, 255, 0.8)"
            )
        fig.add_trace(
            go.Scatter(
                x=indicator_series_filtered.index,
                y=indicator_series_filtered,
                name=indicator_data.get('name', 'Indicator'),
                line=dict(color='purple')
            ),
            row=3 if show_volume else 2,
            col=1
        )
        if indicator_config:
            for shape in indicator_config.get('shapes', []):
                fig.add_shape(
                    type='line',
                    x0=max(range_start, indicator_series_filtered.index.min()),
                    x1=min(range_end, indicator_series_filtered.index.max()),
                    y0=shape['y'],
                    y1=shape['y'],
                    line=dict(color=shape.get('color', 'gray'), dash=shape.get('dash', 'dash')),
                    row=3 if show_volume else 2,
                    col=1
                )
    
    # 强制设置时间轴范围
    actual_start = display_data.index.min()
    actual_end = display_data.index.max()
    xaxis_layout = dict(
        range=[actual_start, actual_end],
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1h", step="hour", stepmode="backward"),
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(step="all", label="全部")
            ]
        ),
        type="date",
        tickformat='%H:%M:%S',
        dtick=1800000
    )
    
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        dragmode='pan',
        template='plotly_dark',
        hovermode='x unified',
        showlegend=True,
        height=800 if indicator_data else 600,
        width=1200,
        xaxis=xaxis_layout
    )
    
    for i in range(2, rows + 1):
        fig.update_layout(**{f'xaxis{i}': xaxis_layout})

    return fig

# Dash 应用
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)

# 数据库路径
DB_PATH = r'D:\策略研究\kline_db_new\kline_data_LEAUSDT.db'

# 时间范围
MIN_DATE, MAX_DATE = get_db_time_range(DB_PATH)
DEFAULT_END_DATE = MAX_DATE
DEFAULT_START_DATE = DEFAULT_END_DATE - timedelta(minutes=300)

# 初始数据
KLINE_CACHE = {}
initial_data = load_kline_data(DB_PATH, start_timestamp=DEFAULT_START_DATE, end_timestamp=DEFAULT_END_DATE, limit=300)
if initial_data.empty:
    logger.error("初始数据加载失败，使用空数据")
    initial_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
initial_data.index = pd.to_datetime(initial_data.index)
initial_data.index.name = 'timestamp'

# 设置 xaxis-range 初始值
initial_xaxis_range = [initial_data.index.min().isoformat(), initial_data.index.max().isoformat()] if not initial_data.empty else [DEFAULT_START_DATE.isoformat(), DEFAULT_END_DATE.isoformat()]

initial_plotly_fig = create_plotly_kline_plot(initial_data, initial_data, xaxis_range=initial_xaxis_range, show_volume=False)
initial_lwc_data = {
    'candlestick': initial_data.reset_index().rename(columns={'timestamp': 'time'})[['time', 'open', 'high', 'low', 'close']]
        .assign(time=lambda x: x['time'].apply(lambda t: t.isoformat()))
        .to_dict('records'),
    'volume': initial_data.reset_index().rename(columns={'timestamp': 'time', 'volume': 'value'})[['time', 'value']]
        .assign(time=lambda x: x['time'].apply(lambda t: t.isoformat()), color='rgba(0, 150, 255, 0.5)')
        .to_dict('records'),
    'indicator': None,
    'buy_signals': [],
    'sell_signals': [],
    'xaxis_range': initial_xaxis_range
}

# 回测缓存
BACKTEST_CACHE = {}

# 布局
app.layout = dbc.Container([
    html.H1('LEAUSDT 策略回测', style={'textAlign': 'center', 'color': 'white'}),
    dbc.Row([
        dbc.Col([
            html.Label('选择策略', style={'color': 'white'}),
            dcc.Dropdown(
                id='strategy-select',
                options=[{'label': STRATEGIES[k]['display_name'], 'value': k} for k in STRATEGIES.keys()],
                value=None,
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
                style={'backgroundColor': '#333', 'color': 'white', 'borderColor': '#555'}
            ),
            html.Br(),
            html.Label('图表类型', style={'color': 'white'}),
            dcc.Dropdown(
                id='chart-type',
                options=[
                    {'label': 'Plotly', 'value': 'plotly'},
                    {'label': 'Lightweight Charts', 'value': 'lwc'}
                ],
                value='plotly',
                style={'backgroundColor': '#333', 'color': 'white', 'borderColor': '#555'}
            ),
            html.Br(),
            html.Label('显示成交量', style={'color': 'white'}),
            dcc.Checklist(
                id='show-volume',
                options=[{'label': '显示成交量', 'value': 'volume'}],
                value=['volume'],
                style={'color': 'white'}
            ),
            html.Br(),
            html.Label('加载历史数据', style={'color': 'white'}),
            dbc.Button('加载更多', id='load-more', color='secondary', n_clicks=0),
            html.Div(id='strategy-params', style={'color': 'white'}),
        ], width=3),
        dbc.Col([
            dcc.Loading(id='loading', children=[
                html.Div(id='chart-container', children=[
                    dcc.Graph(id='plotly-kline-plot', figure=initial_plotly_fig, config={'scrollZoom': True, 'doubleClick': 'reset'})
                ])
            ]),
            html.Div(id='stats-output', style={'color': 'white'}),
            html.Br(),
            html.Label('交易详情', style={'color': 'white'}),
            dash_table.DataTable(
                id='trade-table',
                columns=[
                    {'name': '时间', 'id': 'EntryTime'},
                    {'name': '类型', 'id': 'Type'},
                    {'name': '价格', 'id': 'EntryPrice'},
                ],
                data=[],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '5px', 'color': 'white', 'backgroundColor': '#333'},
                style_header={'backgroundColor': '#444', 'color': 'white'},
                page_size=10
            )
        ], width=9)
    ]),
    dcc.Store(id='data-store', data=initial_data.to_json(date_format='iso', orient='split')),
    dcc.Store(id='backtest-data-store', data=initial_data.to_json(date_format='iso', orient='split')),
    dcc.Store(id='lwc-data-store', data=json.dumps(initial_lwc_data)),
    dcc.Store(id='xaxis-range', data=initial_xaxis_range),
    dcc.Store(id='backtest-results', data=json.dumps({})),
], fluid=True)

# 参数输入和数据加载
@app.callback(
    [Output('strategy-params', 'children'),
     Output('data-store', 'data'),
     Output('backtest-data-store', 'data'),
     Output('lwc-data-store', 'data'),
     Output('backtest-results', 'data')],
    [Input('strategy-select', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input({'type': 'param', 'index': dash.ALL}, 'value'),
     Input('xaxis-range', 'data')],
    [State('data-store', 'data'),
     State('backtest-data-store', 'data'),
     State('strategy-params', 'children')],
    prevent_initial_call=True
)
def update_params_and_data(strategy, start_date, end_date, param_values, xaxis_range, stored_data, backtest_data, param_children):
    # 加载显示数据（基于 xaxis-range）
    display_data = pd.read_json(StringIO(stored_data), orient='split')
    display_data.index = pd.to_datetime(display_data.index)

    # 确保 start_date 不晚于 end_date
    start_time = pd.to_datetime(start_date)
    end_time = pd.to_datetime(end_date)
    if start_time > end_time:
        logger.warning(f"start_date ({start_time}) 晚于 end_date ({end_time})，交换两者")
        start_time, end_time = end_time, start_time

    # 加载回测数据（基于 date-range）
    backtest_start = start_time - timedelta(minutes=300)  # 额外加载前 300 分钟数据以确保指标计算
    backtest_end = end_time
    backtest_data = load_kline_data(DB_PATH, start_timestamp=backtest_start, end_timestamp=backtest_end, limit=5000)
    if backtest_data.empty:
        logger.warning("回测数据加载失败，使用现有数据")
        backtest_data = pd.read_json(StringIO(stored_data), orient='split')
        backtest_data.index = pd.to_datetime(backtest_data.index)

    # 加载显示数据（基于 xaxis-range，不随 date-range 变化）
    if xaxis_range:
        try:
            range_start = pd.to_datetime(xaxis_range[0])
            range_end = pd.to_datetime(xaxis_range[1])
            if range_start > range_end:
                logger.warning(f"xaxis_range 无效，range_start ({range_start}) 晚于 range_end ({range_end})，交换两者")
                range_start, range_end = range_end, range_start
        except Exception as e:
            logger.error(f"xaxis_range 转换失败：{xaxis_range}, 错误：{e}")
            range_start = display_data.index.min()
            range_end = display_data.index.max()
    else:
        range_start = display_data.index[-300] if len(display_data) >= 300 else display_data.index.min()
        range_end = display_data.index.max()

    # 确保 display_data 覆盖 xaxis_range
    display_start = range_start - timedelta(minutes=300)
    display_end = range_end + timedelta(minutes=300)
    display_data = load_kline_data(DB_PATH, start_timestamp=display_start, end_timestamp=display_end, limit=5000)
    if display_data.empty:
        logger.warning("显示数据加载失败，使用现有数据")
        display_data = pd.read_json(StringIO(stored_data), orient='split')
        display_data.index = pd.to_datetime(display_data.index)

    # 确保索引名称为 timestamp
    display_data.index.name = 'timestamp'
    backtest_data.index.name = 'timestamp'

    # 准备 Lightweight Charts 数据
    data_reset = display_data.reset_index().rename(columns={'timestamp': 'time'})
    lwc_data = {
        'candlestick': data_reset[['time', 'open', 'high', 'low', 'close']]
            .assign(time=lambda x: x['time'].apply(lambda t: t.isoformat()))
            .to_dict('records'),
        'volume': data_reset.rename(columns={'volume': 'value'})[['time', 'value']]
            .assign(time=lambda x: x['time'].apply(lambda t: t.isoformat()), color='rgba(0, 150, 255, 0.5)')
            .to_dict('records'),
        'indicator': None,
        'buy_signals': [],
        'sell_signals': [],
        'xaxis_range': [range_start.isoformat(), range_end.isoformat()]
    }
    
    inputs = []
    if strategy and strategy in STRATEGIES:
        param_schema = STRATEGIES[strategy]['param_schema']
        for param, schema in param_schema.items():
            if param == 'size':
                continue
            input_type = 'number' if schema['type'] in ['integer', 'float'] else 'text'
            inputs.extend([
                html.Label(f"{param.capitalize()} ({schema['description']})", style={'color': 'white', 'marginTop': '10px'}),
                dcc.Input(
                    id={'type': 'param', 'index': param},
                    type=input_type,
                    value=schema['default'],
                    min=schema.get('min', None),
                    max=schema.get('max', None),
                    step=0.1 if schema['type'] == 'float' else 1,
                    debounce=True,
                    style={'backgroundColor': '#333', 'color': 'white', 'borderColor': '#555', 'width': '100%'}
                )
            ])

    # 计算回测结果，基于 backtest_data
    backtest_results = {}
    params = {}
    if param_children:
        param_names = []
        for child in param_children[::2]:
            label_content = child['props']['children']
            if isinstance(label_content, str):
                param_name = label_content.split(' (')[0].lower()
            else:
                try:
                    param_name = label_content[0]['props']['children'].split(' (')[0].lower()
                except (TypeError, IndexError, KeyError) as e:
                    logger.error(f"无法提取参数名，child: {child}, 错误: {e}")
                    param_name = ''
            param_names.append(param_name)
        
        for name, value in zip(param_names, param_values):
            if name and value is not None:
                params[name] = value
    if strategy:
        params['size'] = STRATEGIES[strategy]['params'].get('size', 0.1)

        strategy_instance = STRATEGIES[strategy]['class']()
        cache_key = f"{strategy}_{hash(str(params))}_{hash(str(backtest_data.index[0]))}_{hash(str(backtest_data.index[-1]))}"
        if cache_key in BACKTEST_CACHE:
            stats, buy_signals, sell_signals = BACKTEST_CACHE[cache_key]
        else:
            try:
                stats, buy_signals, sell_signals = strategy_instance.run_backtest(backtest_data, params)
                # 过滤买卖点，仅保留 date-range 内的交易
                if not buy_signals.empty and 'EntryTime' in buy_signals.columns:
                    buy_signals = buy_signals[buy_signals['EntryTime'].between(start_time, end_time)]
                if not sell_signals.empty and 'EntryTime' in sell_signals.columns:
                    sell_signals = sell_signals[sell_signals['EntryTime'].between(start_time, end_time)]
                BACKTEST_CACHE[cache_key] = (stats, buy_signals, sell_signals)
            except Exception as e:
                logger.error(f"回测失败：{e}")
                stats, buy_signals, sell_signals = None, pd.DataFrame(), pd.DataFrame()

        backtest_results = {
            'strategy': strategy,
            'params': params,
            'stats': stats if stats is not None else {},
            'buy_signals': buy_signals.to_json(date_format='iso', orient='split') if buy_signals is not None and not buy_signals.empty else None,
            'sell_signals': sell_signals.to_json(date_format='iso', orient='split') if sell_signals is not None and not sell_signals.empty else None,
            'indicator_data': None,
            'indicator_config': {}
        }

        try:
            values = strategy_instance.compute_indicator(backtest_data, params)
            indicator_series = pd.Series(values, index=backtest_data.index, name=strategy.upper())
            indicator_df = indicator_series.reset_index()
            indicator_df.columns = ['timestamp', strategy.upper()]
            backtest_results['indicator_data'] = {
                'name': strategy.upper(),
                'values': indicator_df.to_json(date_format='iso', orient='split')
            }
            backtest_results['indicator_config'] = strategy_instance.indicator_config(backtest_data, params)
        except Exception as e:
            logger.error(f"指标计算错误（{strategy}）：{e}")
            backtest_results['indicator_data'] = None
            backtest_results['indicator_config'] = {}

    return (
        inputs,
        display_data.to_json(date_format='iso', orient='split'),
        backtest_data.to_json(date_format='iso', orient='split'),
        json.dumps(lwc_data),
        json.dumps(backtest_results)
    )

# 更新图表和结果
@app.callback(
    [Output('chart-container', 'children'),
     Output('stats-output', 'children'),
     Output('trade-table', 'data')],
    [Input('data-store', 'data'),
     Input('backtest-data-store', 'data'),
     Input('lwc-data-store', 'data'),
     Input('chart-type', 'value'),
     Input('show-volume', 'value'),
     Input('xaxis-range', 'data'),
     Input('backtest-results', 'data')],
    prevent_initial_call=True
)
def update_plot(display_data, backtest_data, lwc_data, chart_type, show_volume, xaxis_range, backtest_results):
    display_data = pd.read_json(StringIO(display_data), orient='split')
    display_data.index = pd.to_datetime(display_data.index)
    display_data.index.name = 'timestamp'

    backtest_data = pd.read_json(StringIO(backtest_data), orient='split')
    backtest_data.index = pd.to_datetime(backtest_data.index)
    backtest_data.index.name = 'timestamp'

    lwc_data = json.loads(lwc_data)
    
    stats, buy_signals, sell_signals, indicator_data, indicator_config = None, None, None, None, {}
    lwc_data['indicator'] = None
    lwc_data['buy_signals'] = []
    lwc_data['sell_signals'] = []
    
    strategy = None
    if backtest_results:
        try:
            backtest_results = json.loads(backtest_results)
            strategy = backtest_results.get('strategy', None)
            params = backtest_results.get('params', {})
            stats = backtest_results.get('stats', {})
            buy_signals = pd.read_json(StringIO(backtest_results['buy_signals']), orient='split') if backtest_results.get('buy_signals') else pd.DataFrame()
            buy_signals.index = pd.to_datetime(buy_signals.index)
            if 'EntryTime' in buy_signals.columns:
                buy_signals['EntryTime'] = pd.to_datetime(buy_signals['EntryTime'])
            sell_signals = pd.read_json(StringIO(backtest_results['sell_signals']), orient='split') if backtest_results.get('sell_signals') else pd.DataFrame()
            sell_signals.index = pd.to_datetime(sell_signals.index)
            if 'EntryTime' in sell_signals.columns:
                sell_signals['EntryTime'] = pd.to_datetime(sell_signals['EntryTime'])
            if backtest_results.get('indicator_data'):
                indicator_df = pd.read_json(StringIO(backtest_results['indicator_data']['values']), orient='split')
                indicator_df['timestamp'] = pd.to_datetime(indicator_df['timestamp'])
                indicator_name = backtest_results['indicator_data']['name']
                indicator_values = pd.Series(indicator_df[indicator_name].values, index=indicator_df['timestamp'], name=indicator_name)
                indicator_data = {
                    'name': indicator_name,
                    'values': indicator_values
                }
            indicator_config = backtest_results.get('indicator_config', {})
            
            if not buy_signals.empty and 'EntryTime' in buy_signals.columns:
                lwc_data['buy_signals'] = buy_signals.rename(columns={'EntryTime': 'time', 'EntryPrice': 'value'}) \
                    .assign(
                        position='aboveBar',
                        shape='arrowUp',
                        color='green'
                    ).to_dict('records')
            if not sell_signals.empty and 'EntryTime' in sell_signals.columns:
                lwc_data['sell_signals'] = sell_signals.rename(columns={'EntryTime': 'time', 'EntryPrice': 'value'}) \
                    .assign(
                        position='belowBar',
                        shape='arrowDown',
                        color='red'
                    ).to_dict('records')
            if indicator_data:
                indicator_values = indicator_data['values']
                indicator_values = indicator_values[~indicator_values.isna()]
                indicator_name = indicator_values.name if indicator_values.name is not None else 'indicator'
                indicator_df = indicator_values.reset_index()
                indicator_df.columns = ['timestamp', indicator_name]
                indicator_df = indicator_df.rename(columns={'timestamp': 'time'})
                lwc_data['indicator'] = {
                    'name': indicator_data['name'],
                    'values': indicator_df[['time', indicator_name]]
                        .rename(columns={indicator_name: 'value'})
                        .assign(time=lambda x: x['time'].apply(lambda t: t.isoformat()))
                        .to_dict('records'),
                    'shapes': indicator_config.get('shapes', [])
                }
        except Exception as e:
            logger.error(f"处理 backtest_results 失败：{e}")
            backtest_results = {}

    if chart_type == 'plotly':
        show_volume = 'volume' in show_volume
        fig = create_plotly_kline_plot(
            display_data,
            backtest_data,
            buy_signals if strategy else None,
            sell_signals if strategy else None,
            indicator_data if strategy else None,
            indicator_config if strategy else None,
            xaxis_range=xaxis_range,
            show_volume=show_volume
        )
        chart_component = dcc.Graph(id='plotly-kline-plot', figure=fig, config={'scrollZoom': True, 'doubleClick': 'reset'})
    else:
        chart_component = html.Iframe(
            id='lwc-iframe',
            srcDoc=open('D:/策略研究/回测功能/assets/chart.html', 'r').read(),
            style={'width': '100%', 'height': '800px' if indicator_data else '600px'}
        )
    
    stats_text = []
    trade_data = []
    if stats and strategy:
        stats_text = [
            html.H4(f"策略: {STRATEGIES.get(strategy, {}).get('display_name', '未知策略')}", style={'color': 'white'}),
            html.P(f"描述: {STRATEGIES.get(strategy, {}).get('description', '')}", style={'color': 'white'}),
            html.P(f"参数: {params}", style={'color': 'white'}),
            html.Hr(),
            html.P(f"总回报: {stats.get('Return [%]', 0):.2f}%", style={'color': 'white'}),
            html.P(f"最大回撤: {stats.get('Max. Drawdown [%]', 0):.2f}%", style={'color': 'white'}),
            html.P(f"交易次数: {stats.get('# Trades', 0)}", style={'color': 'white'}),
            html.P(f"胜率: {stats.get('Win Rate [%]', 0):.2f}%", style={'color': 'white'})
        ]
        if buy_signals is not None and not buy_signals.empty and 'EntryTime' in buy_signals.columns:
            buy_signals['Type'] = '买入'
            trade_data.extend(buy_signals[['EntryTime', 'Type', 'EntryPrice']].to_dict('records'))
        if sell_signals is not None and not sell_signals.empty and 'EntryTime' in sell_signals.columns:
            sell_signals['Type'] = '卖出'
            trade_data.extend(sell_signals[['EntryTime', 'Type', 'EntryPrice']].to_dict('records'))
        trade_data = sorted(trade_data, key=lambda x: x['EntryTime']) if trade_data else []
    
    return [chart_component], stats_text, trade_data

# 同步 Lightweight Charts 数据
@app.callback(
    Output('lwc-data-store', 'data', allow_duplicate=True),
    [Input('chart-type', 'value'),
     Input('data-store', 'data'),
     Input('xaxis-range', 'data'),
     Input('backtest-results', 'data')],
    [State('lwc-data-store', 'data')],
    prevent_initial_call=True
)
def sync_lwc_data(chart_type, stored_data, xaxis_range, backtest_results, lwc_data):
    if chart_type != 'lwc':
        return no_update
    data = pd.read_json(StringIO(stored_data), orient='split')
    data.index = pd.to_datetime(data.index)
    data.index.name = 'timestamp'
    
    if xaxis_range:
        try:
            range_start = pd.to_datetime(xaxis_range[0])
            range_end = pd.to_datetime(xaxis_range[1])
            if range_start > range_end:
                logger.warning(f"xaxis_range 无效，range_start ({range_start}) 晚于 range_end ({range_end})，交换两者")
                range_start, range_end = range_end, range_start
                xaxis_range = [range_start.isoformat(), range_end.isoformat()]
            display_data = data.loc[range_start:range_end]
        except Exception as e:
            logger.error(f"xaxis_range 转换失败：{xaxis_range}, 错误：{e}")
            display_data = data
    else:
        display_data = data

    data_reset = display_data.reset_index()
    if 'timestamp' not in data_reset.columns:
        logger.error("timestamp 列不存在，检查数据结构")
        raise KeyError("timestamp 列不存在")
    
    data_reset = data_reset.rename(columns={'timestamp': 'time'})
    
    lwc_data = {
        'candlestick': data_reset[['time', 'open', 'high', 'low', 'close']]
            .assign(time=lambda x: x['time'].apply(lambda t: t.isoformat()))
            .to_dict('records'),
        'volume': data_reset.rename(columns={'volume': 'value'})[['time', 'value']]
            .assign(time=lambda x: x['time'].apply(lambda t: t.isoformat()), color='rgba(0, 150, 255, 0.5)')
            .to_dict('records'),
        'indicator': None,
        'buy_signals': [],
        'sell_signals': [],
        'xaxis_range': xaxis_range
    }
    
    if backtest_results:
        try:
            backtest_results = json.loads(backtest_results)
            buy_signals = pd.read_json(StringIO(backtest_results['buy_signals']), orient='split') if backtest_results.get('buy_signals') else pd.DataFrame()
            buy_signals.index = pd.to_datetime(buy_signals.index)
            if 'EntryTime' in buy_signals.columns:
                buy_signals['EntryTime'] = pd.to_datetime(buy_signals['EntryTime'])
            sell_signals = pd.read_json(StringIO(backtest_results['sell_signals']), orient='split') if backtest_results.get('sell_signals') else pd.DataFrame()
            sell_signals.index = pd.to_datetime(sell_signals.index)
            if 'EntryTime' in sell_signals.columns:
                sell_signals['EntryTime'] = pd.to_datetime(sell_signals['EntryTime'])
            
            if not buy_signals.empty and 'EntryTime' in buy_signals.columns:
                lwc_data['buy_signals'] = buy_signals.rename(columns={'EntryTime': 'time', 'EntryPrice': 'value'}) \
                    .assign(
                        position='aboveBar',
                        shape='arrowUp',
                        color='green'
                    ).to_dict('records')
            if not sell_signals.empty and 'EntryTime' in sell_signals.columns:
                lwc_data['sell_signals'] = sell_signals.rename(columns={'EntryTime': 'time', 'EntryPrice': 'value'}) \
                    .assign(
                        position='belowBar',
                        shape='arrowDown',
                        color='red'
                    ).to_dict('records')
            
            if backtest_results.get('indicator_data'):
                indicator_df = pd.read_json(StringIO(backtest_results['indicator_data']['values']), orient='split')
                indicator_df['timestamp'] = pd.to_datetime(indicator_df['timestamp'])
                indicator_name = backtest_results['indicator_data']['name']
                indicator_values = pd.Series(indicator_df[indicator_name].values, index=indicator_df['timestamp'], name=indicator_name)
                indicator_values = indicator_values[~indicator_values.isna()]
                indicator_df = indicator_values.reset_index()
                indicator_df.columns = ['timestamp', indicator_name]
                indicator_df = indicator_df.rename(columns={'timestamp': 'time'})
                lwc_data['indicator'] = {
                    'name': backtest_results['indicator_data']['name'],
                    'values': indicator_df[['time', indicator_name]]
                        .rename(columns={indicator_name: 'value'})
                        .assign(time=lambda x: x['time'].apply(lambda t: t.isoformat()))
                        .to_dict('records'),
                    'shapes': backtest_results.get('indicator_config', {}).get('shapes', [])
                }
        except Exception as e:
            logger.error(f"处理 backtest_results 失败：{e}")
    
    return json.dumps(lwc_data)

# 同步 Plotly 缩放范围
@app.callback(
    Output('xaxis-range', 'data'),
    [Input('plotly-kline-plot', 'relayoutData')],
    [State('xaxis-range', 'data')],
    prevent_initial_call=True
)
def update_xaxis_range(relayout_data, current_range):
    # 仅当用户拖动图表时更新 xaxis-range，date-range 变化不再影响 xaxis-range
    if relayout_data and 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
        new_range = [relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']]
        try:
            range_start = pd.to_datetime(new_range[0])
            range_end = pd.to_datetime(new_range[1])
            if range_start > range_end:
                logger.warning(f"relayoutData xaxis_range 无效，range_start ({range_start}) 晚于 range_end ({range_end})，交换两者")
                range_start, range_end = range_end, range_start
                new_range = [range_start.isoformat(), range_end.isoformat()]
        except Exception as e:
            logger.error(f"relayoutData xaxis_range 转换失败：{new_range}, 错误：{e}")
            return current_range
        return new_range
    return current_range

# 运行
if __name__ == '__main__':
    app.run(debug=True)