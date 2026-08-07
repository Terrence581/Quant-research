# factor2_2conditions Long-only 工作流程

## 1. 策略概述

信号在 T 日收盘后产生，实际买入发生在下一交易日 T+1 开盘。策略只做多，不做空。

本策略是两条件规则信号策略，不构建因子，不做因子排序，不做 3sigma 去极值，不做 z-score 标准化，也不做 IC / RankIC。

最终股票池只由两条条件决定：

```text
条件1：close_T > max(close_{T-1}, close_{T-2}, ..., close_{T-126})

条件2：最近63个交易记录内，未复权收盘价最大回撤 < -20%
```

其中 `close_price` 来自数据库原始未复权收盘价 `Clsprc`。

## 2. 默认参数

代码位置：`factor2_2conditions.py`

```python
DEFAULT_HIGH_LOOKBACK_DAYS = 126
DEFAULT_DRAWDOWN_LOOKBACK_DAYS = 63
DEFAULT_DRAWDOWN_THRESHOLD = 0.20
DEFAULT_HOLDING_DAYS = 63
DEFAULT_MARKET_TYPES = {1, 4}
```

输出目录由以下函数生成：

```python
build_long_only_experiment_folder_name()
```

默认目录格式：

```text
factor2_2conditions_output/
  longonly_high126_dd63_hd63_mkt1-4/
```

## 3. 数据读取与缓存

主流程函数：

```python
main()
```

数据读取优先级：

1. 先检查本地清洗缓存。
2. 如果 `factor2_2conditions_output/_local_data_cache` 中清洗缓存存在且没有设置 `--refresh-clean-data`，直接读取该缓存。
3. 如果新策略输出目录中没有清洗缓存，则检查已有 `factor2_LongOnly_output/_local_data_cache` 中同参数清洗缓存。
4. 如果清洗缓存不存在，则先找本地原始行情缓存。
5. 如果原始行情缓存也不存在，才连接 MySQL 导出原始行情。

复用的基础函数来自 `factor2_LongOnly.py`：

```python
build_local_raw_data_cache_path()
build_local_clean_data_cache_paths()
load_clean_market_data_from_cache()
load_raw_market_data_with_cache()
clean_market_data()
write_clean_market_data_cache()
```

清洗逻辑保持原工作流：

```text
统一字段名
删除 stock_code / trade_date 缺失记录
保留指定 market_type
删除价格小于等于0的记录
剔除 ST/PT
剔除流通市值低于阈值的记录
按 stock_code + trade_date 排序和去重
辅助字段按股票内部前向填充
```

## 4. 条件1：T日严格突破前高

实现函数：

```python
calculate_one_stock_two_condition_signal()
calculate_raw_two_condition_signals()
```

对每只股票逐日计算：

```text
prior_high_close = max(close_{T-1}, ..., close_{T-126})
condition1 = close_T > prior_high_close
```

注意：

```text
历史窗口不包含 T 日
必须严格大于，不接受等于
窗口必须完整，不足126个有效交易记录不能入池
如果最高收盘价在窗口内出现多次，记录第一次出现日期
```

诊断字段：

```text
lookback_window_start
prior_high_trade_date
prior_high_close_price
high_lookback_valid_days
breakout_return
high_breakout_filter
```

其中：

```text
breakout_return = close_T / prior_high_close - 1
```

这个字段只用于诊断，不用于排序。

## 5. 条件2：最近63日最大回撤

实现函数：

```python
calculate_price_drawdown_window()
calculate_one_stock_two_condition_signal()
```

窗口包含 T 日：

```text
T-62, T-61, ..., T
```

使用未复权收盘价计算：

```text
running_high_d = max(close from window_start to d)
drawdown_d = close_d / running_high_d - 1
rolling_max_drawdown = min(drawdown_d)
```

通过条件：

```text
rolling_max_drawdown < -0.20
```

注意：

```text
窗口必须完整，不足63个有效交易记录不能入池
价格无效或缺失时不能入池
回撤高点日期取产生该次最大回撤前的第一次 running high 日期
回撤低点日期取最大回撤发生日期
```

诊断字段：

```text
drawdown_window_start
drawdown_lookback_valid_days
rolling_max_drawdown
drawdown_peak_trade_date
drawdown_peak_close_price
drawdown_trough_trade_date
drawdown_trough_close_price
drawdown_filter
```

## 6. 最终信号池

实现函数：

```python
calculate_raw_two_condition_signals()
build_all_signals()
build_screening_reason()
screening_summary()
screening_filter_step_summary()
```

最终信号：

```text
passes_signal_filters = high_breakout_filter & drawdown_filter
```

只有 `passes_signal_filters=True` 的股票-日期记录进入：

```text
03_factor2_2conditions_long_only_signal_pool.csv
```

全量观察记录输出到：

```text
03_factor2_2conditions_long_only_all_stocks.csv
```

筛选失败原因字段：

```text
screening_reason
```

可能取值：

```text
insufficient_high_lookback_history
not_strict_breakout_above_prior_high
insufficient_drawdown_lookback_history
drawdown_not_deep_enough
passed
```

## 7. T+1 开盘交易过滤

实现函数：

```python
attach_long_only_execution_and_returns()
build_tradable_open_buy_dates()
```

交易规则：

```text
T 日收盘后生成信号
T+1 开盘买入
如果 T+1 开盘价缺失，不买入
如果 T+1 开盘一字涨停或一字跌停，不买入
同一信号日内所有可交易股票等权
```

输出文件：

```text
04_factor2_2conditions_long_only_signal_pool_with_forward_returns.csv
04_factor2_2conditions_long_only_tradable_open_buy_dates.csv
```

## 8. 持有期收益

实现函数：

```python
calculate_forward_holding_returns_sum()
```

持有期收益不用累乘，改为 `Dretwd` 日收益求和：

```text
next_period_return_before_trade_filter
  = sum(Dretwd_{T+1}, ..., Dretwd_{T+holding_days})
```

不使用：

```text
product(1 + Dretwd) - 1
```

如果持有期内 `Dretwd` 不完整，则该条持有期收益记为空，并被交易过滤为不可用。

## 9. 每日组合收益和 NAV

实现函数：

```python
calculate_long_only_returns()
attach_additive_long_only_nav()
```

每日组合收益：

```text
daily_portfolio_return
  = sum(active_stock_daily_Dretwd * stock_weight / holding_days)
```

其中：

```text
每个信号 sleeve 占组合资金 1 / holding_days
同一信号 sleeve 内股票等权
多个历史信号 sleeve 可同时活跃
```

NAV 使用累加，不使用复利：

```text
NAV_0 = 1.0
NAV_t = NAV_{t-1} + long_only_return_t
```

代码口径：

```python
current_nav = current_nav + period_return
```

输出文件：

```text
05_factor2_2conditions_long_only_holding_period_returns.csv
05_factor2_2conditions_long_only_nav_curve.png
```

## 10. 绩效和回撤

实现函数：

```python
calculate_long_only_performance_attribution()
```

绩效表输出：

```text
06_factor2_2conditions_long_only_portfolio_performance_table.csv
06_factor2_2conditions_long_only_portfolio_performance_detail.csv
factor2_2conditions_long_only_performance_summary.csv
```

组合最大回撤基于 `long_only_nav` 计算：

```text
running_max_t = max(NAV_1, ..., NAV_t)
drawdown_t = NAV_t / running_max_t - 1
max_drawdown = min(drawdown_t)
```

输出文件：

```text
07_factor2_2conditions_long_only_drawdown_series.csv
08_factor2_2conditions_long_only_yearly_performance.csv
```

## 11. 交易明细图

实现函数：

```python
output_trade_detail_image()
```

图片文件：

```text
15_factor2_2conditions_long_only_trade_detail_YYYYMMDD.png
```

显示字段：

```text
买入日期
信号日期
股票代码
买入开盘价
组合权重
T日收盘价
前高日期
前高收盘价
突破幅度
回撤窗口起点
最大回撤
回撤高点日期
回撤低点日期
持有结束日期
```

## 12. Diagnostics

诊断脚本：

```text
factor2_2conditions_diagnostics.py
```

诊断脚本从 `factor2_2conditions.py` 读取默认参数，并自动推导默认输入目录。

保留的诊断内容：

```text
信号质量摘要
交易过滤原因
每日信号数量
现金持有原因
收益分布
NAV / 回撤复核
筛选步骤摘要
指定交易日买入明细图
```

删除的诊断内容：

```text
IC
RankIC
因子分组
3sigma 去极值
z-score 标准化
```

运行示例：

```powershell
& D:\Anaconda\python.exe "D:\Desktop\CINDA qr\factor1_strong_momentum\factor2_2conditions.py"

& D:\Anaconda\python.exe "D:\Desktop\CINDA qr\factor1_strong_momentum\factor2_2conditions_diagnostics.py" --trade-detail-date 2019-09-09
```
