
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

# 设置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.StreamHandler(),
    logging.FileHandler('debug.log')
])
logger = logging.getLogger(__name__)

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
            logger.info(f"数据库时间范围：{min_time} 到 {max_time}")
            return min_time, max_time
        return datetime.now() - timedelta(days=180), datetime.now()
    except Exception as e:
        logger.error(f"查询时间范围错误：{e}")
        return datetime.now() - timedelta(days=180), datetime.now()

# 数据加载函数
def load_kline_data(db_path, coin_name='LEAUSDT', interval='1m', start_timestamp=None, end_timestamp=None, limit=None, chunk_size=10000):
    cache_key = f"{coin_name}_{interval}_{start_timestamp}_{end_timestamp}_{limit}"
    if cache_key in KLINE_CACHE:
        logger.info(f"从缓存加载数据：{cache_key}")
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
            data_chunks.append(df_chunk)
            offset += chunk_size
            logger.debug(f"加载数据分片：{len(df_chunk)} 条，offset: {offset}")
            if limit and offset >= limit:
                break
        conn.close()
        if not data_chunks:
            logger.warning("数据库返回空数据")
            return pd.DataFrame()
        data = pd.concat(data_chunks).sort_index()
        data = data[~data.index.duplicated(keep='last')]
        data = data[(data[['open', 'high', 'low', 'close']] > 0).all(axis=1)].dropna()
        if limit:
            data = data.tail(limit)
        KLINE_CACHE[cache_key] = data
        logger.info(f"加载数据：{len(data)} 条，时间范围：{data.index.min()} 到 {data.index.max()}")
        return data
    except Exception as e:
        logger.error(f"数据加载错误：{e}")
        return pd.DataFrame()

# Plotly 绘图函数
def create_plotly_kline_plot(data, buy_signals=None, sell_signals=None, indicator_data=None, indicator_config=None, xaxis_range=None, max_display=1000):
    if data.empty:
        logger.warning("无数据，无法绘图")
        return go.Figure()
    
    if xaxis_range:
        range_start = pd.to_datetime(xaxis_range[0])
        range_end = pd.to_datetime(xaxis_range[1])
    else:
        range_end = data.index.max()
        range_start = data.index[-300] if len(data) >= 300 else data.index.min()
    
    display_data = data.loc[range_start:range_end]
    if display_data.empty:
        logger.warning(f"xaxis_range [{range_start}, {range_end}] 无数据")
        return go.Figure()
    
    if len(display_data) > max_display:
        display_data = display_data.tail(max_display)
        logger.debug(f"限制渲染数据量为 {max_display} 条")
    
    rows = 2 + (1 if indicator_data else 0)
    row_heights = [0.5, 0.2, 0.3] if indicator_data else [0.7, 0.3]
    subplot_titles = ['K线图', '成交量', indicator_data.get('name', 'Indicator')] if indicator_data else ['K线图', '成交量']
    
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=subplot_titles, row_heights=row_heights)
    
    fig.add_trace(
        go.Candlestick(
            x=display_data.index,
            open=display_data['open'], high=display_data['high'],
            low=display_data['low'], close=display_data['close'],
            name='K线'
        ),
        row=1, col=1
    )
    
    if buy_signals is not None and not buy_signals.empty:
        buy_signals = buy_signals[buy_signals['EntryTime'].between(range_start, range_end)]
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals['EntryTime'], y=buy_signals['EntryPrice'],
                    mode='markers', name='买入',
                    marker=dict(symbol='triangle-up', size=10, color='green'),
                    text=[f"买入价格: {price:.2f}" for price in buy_signals['EntryPrice']],
                    hoverinfo='text+x'
                ),
                row=1, col=1
            )
    if sell_signals is not None and not sell_signals.empty:
        sell_signals = sell_signals[sell_signals['EntryTime'].between(range_start, range_end)]
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals['EntryTime'], y=sell_signals['EntryPrice'],
                    mode='markers', name='卖出',
                    marker=dict(symbol='triangle-down', size=10, color='red'),
                    text=[f"卖出价格: {price:.2f}" for price in sell_signals['EntryPrice']],
                    hoverinfo='text+x'
                ),
                row=1, col=1
            )
    
    fig.add_trace(
        go.Bar(x=display_data.index, y=display_data['volume'], name='成交量', marker_color='blue'),
        row=2, col=1
    )
    
    if indicator_data and 'values' in indicator_data:
        indicator_series = pd.Series(indicator_data['values'], index=data.index).loc[range_start:range_end]
        if len(indicator_series) > max_display:
            indicator_series = indicator_series.tail(max_display)
        fig.add_trace(
            go.Scatter(x=indicator_series.index, y=indicator_series, name=indicator_data.get('name', 'Indicator'), line=dict(color='purple')),
            row=3, col=1
        )
        if indicator_config:
            for shape in indicator_config.get('shapes', []):
                fig.add_shape(
                    type='line', x0=range_start, x1=range_end,
                    y0=shape['y'], y1=shape['y'],
                    line=dict(color=shape.get('color', 'gray'), dash=shape.get('dash', 'dash')),
                    row=3, col=1
                )
    
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        dragmode='pan',
        template='plotly_dark',
        hovermode='x unified',
        showlegend=True,
        height=800 if indicator_data else 600,
        width=1200,
        xaxis=dict(
            range=[range_start, range_end],
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(step="all", label="全部")
                ]
            ),
            type="date",
            tickformat='%Y-%m-%d %H:%M:%S',
            dtick=3600000 * 4
        )
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=['sat', 'mon'])])
    
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
initial_plotly_fig = create_plotly_kline_plot(initial_data)
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
    'xaxis_range': None
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
    dcc.Store(id='lwc-data-store', data=json.dumps(initial_lwc_data)),
    dcc.Store(id='xaxis-range', data=None),
], fluid=True)

# 参数输入和数据加载
@app.callback(
    [Output('strategy-params', 'children'),
     Output('data-store', 'data'),
     Output('lwc-data-store', 'data')],
    [Input('strategy-select', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('xaxis-range', 'data')],
    [State('data-store', 'data')],
    prevent_initial_call=True
)
def update_params_and_data(strategy, start_date, end_date, xaxis_range, stored_data):
    logger.debug(f"update_params_and_data 触发，策略: {strategy}, 时间范围: {start_date} 到 {end_date}, xaxis_range: {xaxis_range}")
    start_time = pd.to_datetime(start_date)
    end_time = pd.to_datetime(end_date)

    if xaxis_range:
        range_start = pd.to_datetime(xaxis_range[0])
        range_end = pd.to_datetime(xaxis_range[1])
        current_data = pd.read_json(StringIO(stored_data), orient='split')
        current_data.index = pd.to_datetime(current_data.index)
        # 显式设置索引名称，确保一致性
        current_data.index.name = 'timestamp'
        logger.debug(f"current_data 索引名: {current_data.index.name}, 列名: {list(current_data.columns)}")
        if range_start < current_data.index.min() or range_end > current_data.index.max():
            new_start = min(start_time, range_start) - timedelta(minutes=300)
            new_end = max(end_time, range_end) + timedelta(minutes=300)
            data = load_kline_data(DB_PATH, start_timestamp=new_start, end_timestamp=new_end)
        else:
            data = current_data
    else:
        data = load_kline_data(DB_PATH, start_timestamp=start_time, end_timestamp=end_time, limit=300)

    if data.empty:
        logger.warning("加载数据失败，使用现有数据")
        data = pd.read_json(StringIO(stored_data), orient='split')
        data.index = pd.to_datetime(data.index)
    else:
        logger.info(f"成功加载数据：{len(data)} 条，时间范围：{data.index.min()} 到 {data.index.max()}")

    # 确保 data 的索引名称为 timestamp
    data.index.name = 'timestamp'
    logger.debug(f"data 索引名: {data.index.name}, 列名: {list(data.columns)}")

    # 调试：检查 reset_index 后的列名
    data_reset = data.reset_index()
    logger.debug(f"data_reset 列名: {list(data_reset.columns)}")

    # 重命名列并选择需要的列
    data_reset = data_reset.rename(columns={'timestamp': 'time'})
    logger.debug(f"data_reset 重命名后列名: {list(data_reset.columns)}")

    BACKTEST_CACHE.clear()
    logger.info("切换策略或时间范围，清空回测缓存")
    
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
    logger.debug(f"生成参数输入框：{strategy}, 参数 {param_schema if strategy else {}}")
    return inputs, data.to_json(date_format='iso', orient='split'), json.dumps(lwc_data)

# 更新图表和结果
@app.callback(
    [Output('chart-container', 'children'),
     Output('stats-output', 'children'),
     Output('trade-table', 'data')],
    [Input('strategy-select', 'value'),
     Input('data-store', 'data'),
     Input('lwc-data-store', 'data'),
     Input('chart-type', 'value'),
     Input({'type': 'param', 'index': dash.ALL}, 'value')],
    [State('strategy-params', 'children'),
     State('xaxis-range', 'data')],
    prevent_initial_call=True
)
def update_plot(strategy, stored_data, lwc_data, chart_type, param_values, param_children, xaxis_range):
    logger.debug(f"update_plot 触发，策略: {strategy}, 图表类型: {chart_type}")
    data = pd.read_json(StringIO(stored_data), orient='split')
    data.index = pd.to_datetime(data.index)
    # 确保索引名称
    data.index.name = 'timestamp'
    logger.debug(f"update_plot data 索引名: {data.index.name}, 列名: {list(data.columns)}")
    lwc_data = json.loads(lwc_data)
    
    params = {}
    if param_children:
        # 提取参数名，处理 children 可能是字符串的情况
        param_names = []
        for child in param_children[::2]:  # 每隔一个元素（Label 组件）
            label_content = child['props']['children']
            if isinstance(label_content, str):
                # 如果 children 是字符串，直接处理
                param_name = label_content.split(' (')[0].lower()
            else:
                # 如果 children 是一个列表（嵌套组件），按原逻辑处理
                try:
                    param_name = label_content[0]['props']['children'].split(' (')[0].lower()
                except (TypeError, IndexError, KeyError) as e:
                    logger.error(f"无法提取参数名，child: {child}, 错误: {e}")
                    param_name = ''
            param_names.append(param_name)
        logger.debug(f"提取的参数名: {param_names}")
        
        for name, value in zip(param_names, param_values):
            if name and value is not None:
                params[name] = value
    if strategy:
        params['size'] = STRATEGIES[strategy]['params'].get('size', 0.1)
    
    stats, buy_signals, sell_signals = None, None, None
    indicator_data = None
    indicator_config = {}
    lwc_data['indicator'] = None
    lwc_data['buy_signals'] = []
    lwc_data['sell_signals'] = []
    
    if strategy and strategy in STRATEGIES:
        strategy_instance = STRATEGIES[strategy]['class']()
        cache_key = f"{strategy}_{hash(str(params))}_{hash(str(data.index[0]))}_{hash(str(data.index[-1]))}"
        if cache_key in BACKTEST_CACHE:
            stats, buy_signals, sell_signals = BACKTEST_CACHE[cache_key]
        else:
            logger.info(f"无缓存，运行回测：{strategy}")
            try:
                stats, buy_signals, sell_signals = strategy_instance.run_backtest(data, params)
                BACKTEST_CACHE[cache_key] = (stats, buy_signals, sell_signals)
            except Exception as e:
                logger.error(f"回测失败：{e}")
                error_msg = f"回测失败：{str(e)}"
                if chart_type == 'plotly':
                    fig = create_plotly_kline_plot(data, xaxis_range=xaxis_range)
                    return [dcc.Graph(id='plotly-kline-plot', figure=fig)], [html.P(error_msg, style={'color': 'white'})], []
                else:
                    return [html.Iframe(id='lwc-iframe', srcDoc=open('D:/策略研究/回测功能/assets/chart.html', 'r').read(), style={'width': '100%', 'height': '600px'})], [html.P(error_msg, style={'color': 'white'})], []
        
        try:
            values = strategy_instance.compute_indicator(data, params)
            indicator_data = {
                'name': strategy.upper(),
                'values': values
            }
            indicator_config = strategy_instance.indicator_config(data, params)
            
            indicator_values = pd.Series(values, index=data.index)
            indicator_values = indicator_values[~indicator_values.isna()]
            lwc_data['indicator'] = {
                'name': indicator_data['name'],
                'values': indicator_values.reset_index().rename(columns={'timestamp': 'time'})[['time', indicator_values.name]]
                    .rename(columns={indicator_values.name: 'value'})
                    .assign(time=lambda x: x['time'].apply(lambda t: t.isoformat()))
                    .to_dict('records'),
                'shapes': indicator_config.get('shapes', [])
            }
        except Exception as e:
            logger.error(f"指标计算错误（{strategy}）：{e}")
        
        if not buy_signals.empty:
            lwc_data['buy_signals'] = buy_signals.rename(columns={'EntryTime': 'time', 'EntryPrice': 'value'}) \
                .assign(
                    time=lambda x: x['time'].apply(lambda t: t.isoformat()),
                    position='aboveBar',
                    shape='arrowUp',
                    color='green'
                ).to_dict('records')
        if not sell_signals.empty:
            lwc_data['sell_signals'] = sell_signals.rename(columns={'EntryTime': 'time', 'EntryPrice': 'value'}) \
                .assign(
                    time=lambda x: x['time'].apply(lambda t: t.isoformat()),
                    position='belowBar',
                    shape='arrowDown',
                    color='red'
                ).to_dict('records')
    
    if chart_type == 'plotly':
        fig = create_plotly_kline_plot(data, buy_signals, sell_signals, indicator_data, indicator_config, xaxis_range)
        chart_component = dcc.Graph(id='plotly-kline-plot', figure=fig, config={'scrollZoom': True, 'doubleClick': 'reset'})
    else:
        chart_component = html.Iframe(
            id='lwc-iframe',
            srcDoc=open('D:/策略研究/回测功能/assets/chart.html', 'r').read(),
            style={'width': '100%', 'height': '800px' if indicator_data else '600px'}
        )
    
    stats_text = []
    trade_data = []
    if stats:
        stats_text = [
            html.H4(f"策略: {STRATEGIES[strategy]['display_name']}", style={'color': 'white'}),
            html.P(f"描述: {STRATEGIES[strategy]['description']}", style={'color': 'white'}),
            html.P(f"参数: {params}", style={'color': 'white'}),
            html.Hr(),
            html.P(f"总回报: {stats['Return [%]']:.2f}%", style={'color': 'white'}),
            html.P(f"最大回撤: {stats['Max. Drawdown [%]']:.2f}%", style={'color': 'white'}),
            html.P(f"交易次数: {stats['# Trades']}", style={'color': 'white'}),
            html.P(f"胜率: {stats['Win Rate [%]']:.2f}%", style={'color': 'white'})
        ]
        if buy_signals is not None and not buy_signals.empty:
            buy_signals['Type'] = '买入'
            trade_data.extend(buy_signals[['EntryTime', 'Type', 'EntryPrice']].to_dict('records'))
        if sell_signals is not None and not sell_signals.empty:
            sell_signals['Type'] = '卖出'
            trade_data.extend(sell_signals[['EntryTime', 'Type', 'EntryPrice']].to_dict('records'))
        trade_data = sorted(trade_data, key=lambda x: x['EntryTime'])
    
    logger.debug(f"图表更新完成，数据长度: {len(data)}, 交易数量: {len(trade_data)}")
    return [chart_component], stats_text, trade_data

# 同步 Lightweight Charts 数据
@app.callback(
    Output('lwc-data-store', 'data', allow_duplicate=True),
    [Input('chart-type', 'value'),
     Input('strategy-select', 'value'),
     Input('data-store', 'data'),
     Input({'type': 'param', 'index': dash.ALL}, 'value')],
    [State('strategy-params', 'children'),
     State('lwc-data-store', 'data'),
     State('xaxis-range', 'data')],
    prevent_initial_call=True
)
def sync_lwc_data(chart_type, strategy, stored_data, param_values, param_children, lwc_data, xaxis_range):
    if chart_type != 'lwc':
        return no_update
    data = pd.read_json(StringIO(stored_data), orient='split')
    data.index = pd.to_datetime(data.index)
    data.index.name = 'timestamp'
    logger.debug(f"sync_lwc_data data 索引名: {data.index.name}, 列名: {list(data.columns)}")
    
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
        logger.debug(f"sync_lwc_data 提取的参数名: {param_names}")
        
        for name, value in zip(param_names, param_values):
            if name and value is not None:
                params[name] = value
    if strategy:
        params['size'] = STRATEGIES[strategy]['params'].get('size', 0.1)
    
    data_reset = data.reset_index()
    logger.debug(f"sync_lwc_data data_reset 列名: {list(data_reset.columns)}")
    if 'timestamp' not in data_reset.columns:
        logger.error("timestamp 列不存在，检查数据结构")
        raise KeyError("timestamp 列不存在")
    
    data_reset = data_reset.rename(columns={'timestamp': 'time'})
    logger.debug(f"sync_lwc_data data_reset 重命名后列名: {list(data_reset.columns)}")
    
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
    
    if strategy and strategy in STRATEGIES:
        strategy_instance = STRATEGIES[strategy]['class']()
        stats, buy_signals, sell_signals = strategy_instance.run_backtest(data, params)
        try:
            values = strategy_instance.compute_indicator(data, params)
            indicator_series = pd.Series(values, index=data.index)
            indicator_series = indicator_series[~indicator_series.isna()]
            lwc_data['indicator'] = {
                'name': strategy.upper(),
                'values': indicator_series.reset_index().rename(columns={'timestamp': 'time'})[['time', indicator_series.name]]
                    .rename(columns={indicator_series.name: 'value'})
                    .assign(time=lambda x: x['time'].apply(lambda t: t.isoformat()))
                    .to_dict('records'),
                'shapes': strategy_instance.indicator_config(data, params).get('shapes', [])
            }
        except Exception as e:
            logger.error(f"指标计算错误（{strategy}）：{e}")
        if not buy_signals.empty:
            lwc_data['buy_signals'] = buy_signals.rename(columns={'EntryTime': 'time', 'EntryPrice': 'value'}) \
                .assign(
                    time=lambda x: x['time'].apply(lambda t: t.isoformat()),
                    position='aboveBar',
                    shape='arrowUp',
                    color='green'
                ).to_dict('records')
        if not sell_signals.empty:
            lwc_data['sell_signals'] = sell_signals.rename(columns={'EntryTime': 'time', 'EntryPrice': 'value'}) \
                .assign(
                    time=lambda x: x['time'].apply(lambda t: t.isoformat()),
                    position='belowBar',
                    shape='arrowDown',
                    color='red'
                ).to_dict('records')
    
    return json.dumps(lwc_data)

# 同步 Plotly 缩放范围
@app.callback(
    Output('xaxis-range', 'data'),
    Input('plotly-kline-plot', 'relayoutData'),
    State('xaxis-range', 'data'),
    prevent_initial_call=True
)
def update_xaxis_range(relayout_data, current_range):
    if relayout_data and 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
        new_range = [relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']]
        logger.debug(f"更新 xaxis-range: {new_range}")
        return new_range
    return current_range

# 运行
if __name__ == '__main__':
    app.run(debug=True)
