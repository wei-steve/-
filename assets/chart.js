document.addEventListener('DOMContentLoaded', function() {
    const chartContainer = document.getElementById('lightweight-chart');
    if (!chartContainer) return;

    const chart = LightweightCharts.createChart(chartContainer, {
        width: 1200,
        height: 600,
        timeScale: {
            timeVisible: true,
            secondsVisible: true,
            borderColor: '#555',
        },
        rightPriceScale: {
            borderColor: '#555',
        },
        layout: {
            background: { type: 'solid', color: '#222' },
            textColor: '#DDD',
        },
        grid: {
            vertLines: { color: '#444' },
            horzLines: { color: '#444' },
        },
    });

    let candlestickSeries, volumeSeries, indicatorSeries;

    function updateChart(data) {
        // 清空现有系列
        if (candlestickSeries) chart.removeSeries(candlestickSeries);
        if (volumeSeries) chart.removeSeries(volumeSeries);
        if (indicatorSeries) chart.removeSeries(indicatorSeries);

        // K 线
        candlestickSeries = chart.addCandlestickSeries({
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderVisible: false,
            wickUpColor: '#26a69a',
            wickDownColor: '#ef5350',
        });
        candlestickSeries.setData(data.candlestick.map(d => ({
            time: new Date(d.time).getTime() / 1000,
            open: d.open,
            high: d.high,
            low: d.low,
            close: d.close
        })));

        // 成交量
        volumeSeries = chart.addHistogramSeries({
            priceFormat: { type: 'volume' },
            priceScaleId: '',
            scaleMargins: { top: 0.7, bottom: 0 },
        });
        volumeSeries.setData(data.volume.map(d => ({
            time: new Date(d.time).getTime() / 1000,
            value: d.value,
            color: d.color
        })));

        // 指标
        if (data.indicator && data.indicator.values) {
            chart.resize(1200, 800);
            indicatorSeries = chart.addLineSeries({
                priceScaleId: 'indicator',
                color: 'purple',
                lineWidth: 2,
            });
            indicatorSeries.setData(data.indicator.values.map(d => ({
                time: new Date(d.time).getTime() / 1000,
                value: d.value
            })));

            // 指标水平线
            if (data.indicator.shapes) {
                data.indicator.shapes.forEach(shape => {
                    indicatorSeries.createPriceLine({
                        price: shape.y,
                        color: shape.color,
                        lineStyle: LightweightCharts.LineStyle.Dashed,
                        lineWidth: 1,
                    });
                });
            }
        } else {
            chart.resize(1200, 600);
        }

        // 买卖信号
        const markers = [];
        if (data.buy_signals) {
            data.buy_signals.forEach(sig => {
                markers.push({
                    time: new Date(sig.time).getTime() / 1000,
                    position: sig.position,
                    shape: sig.shape,
                    color: sig.color,
                    text: `买入: ${sig.value.toFixed(2)}`
                });
            });
        }
        if (data.sell_signals) {
            data.sell_signals.forEach(sig => {
                markers.push({
                    time: new Date(sig.time).getTime() / 1000,
                    position: sig.position,
                    shape: sig.shape,
                    color: sig.color,
                    text: `卖出: ${sig.value.toFixed(2)}`
                });
            });
        }
        candlestickSeries.setMarkers(markers);

        // 同步缩放范围
        if (data.xaxis_range) {
            chart.timeScale().setVisibleRange({
                from: new Date(data.xaxis_range[0]).getTime() / 1000,
                to: new Date(data.xaxis_range[1]).getTime() / 1000
            });
        } else {
            chart.timeScale().fitContent();
        }
    }

    // 从 Dash 获取数据
    function loadData() {
        const lwcData = document.getElementById('lwc-data-store');
        if (lwcData && lwcData.value) {
            const data = JSON.parse(lwcData.value);
            updateChart(data);
        }
    }

    // 监听数据变化
    const observer = new MutationObserver(loadData);
    observer.observe(document.getElementById('lwc-data-store'), { attributes: true, childList: true, subtree: true });

    // 监听缩放范围变化
    chart.timeScale().subscribeVisibleTimeRangeChange(range => {
        if (range) {
            const lwcData = document.getElementById('lwc-data-store');
            if (lwcData) {
                const data = JSON.parse(lwcData.value);
                data.xaxis_range = [
                    new Date(range.from * 1000).toISOString(),
                    new Date(range.to * 1000).toISOString()
                ];
                lwcData.value = JSON.stringify(data);
            }
        }
    });

    // 初始加载
    loadData();
});