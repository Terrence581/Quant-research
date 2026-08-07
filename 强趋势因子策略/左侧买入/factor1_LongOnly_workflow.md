# 强势股回调 Long-only 策略工作流程

## 1. 策略概述

本策略是一个强势股回调事件型 long-only 策略。

策略目标不是构造多空组合，而是检验：在 A 股主板股票中，满足强上涨、近高点、充分回撤、回调至阶段低点、且 peak 前涨跌幅波动不过度异常的股票，未来持有期内是否具有正向收益。

信号在 `T` 日收盘后产生，实际买入发生在下一交易日 `T+1` 开盘。策略只做多，不做空。

## 2. 市场与时间范围

市场范围：

```text
A 股主板股票
market_type = {1, 4}
```

测试时间范围：

```text
2018-01-01 至 2021-12-31
```

当前代码只保留 `market_type={1,4}`，对应上证 A 股和深证 A 股主板。

## 3. 当前核心参数

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `DEFAULT_START_DATE` | `2018-01-01` | 回测开始日期 |
| `DEFAULT_END_DATE` | `2021-12-31` | 回测结束日期 |
| `DEFAULT_MARKET_TYPES` | `{1, 4}` | 只保留 A 股主板 |
| `DEFAULT_LOOKBACK_DAYS` | `189` | 回看窗口，约 9 个月交易日 |
| `DEFAULT_MAX_PEAK_AGE_DAYS` | `63` | peak 距离 T 日最多 63 个交易日 |
| `DEFAULT_HOLDING_DAYS` | `63` | 持有期 63 个交易日 |
| `DEFAULT_RISE_THRESHOLD` | `0.50` | 低点到 peak 涨幅超过 50% |
| `DEFAULT_DRAWDOWN_THRESHOLD` | `0.10` | peak 到 T 日回撤超过 10% |
| `DEFAULT_ZSCORE_WINDOW` | `63` | ChangeRatio 时序 z-score 历史窗口 |
| `DEFAULT_ZSCORE_LIMIT` | `2.5` | peak 前 abs(z-score) 最大值不能超过 2.5 |
| `DEFAULT_ZSCORE_DDOF` | `0` | 标准差自由度 |
| `DEFAULT_MIN_FLOAT_MARKET_VALUE` | `5,000,000` | 流通市值下限 |

## 4. 数据读取与缓存

程序先检查本地是否已有原始行情缓存和清洗后行情缓存。

如果本地缓存存在，且没有设置强制刷新，则直接读取本地缓存，避免重复从数据库读取。

如果本地缓存不存在，或用户要求刷新数据，则从 MySQL 数据库读取原始行情数据，并保存到本地缓存。

数据库密码必须通过环境变量提供：

```powershell
$env:MYSQL_PASSWORD = "你的数据库密码"
```

如果环境变量 `MYSQL_PASSWORD` 缺失，程序会直接报错并停止运行。

## 5. 基础数据清洗

读取原始行情后，程序执行以下清洗步骤：

1. 统一数据库字段名为策略内部标准字段名。
2. 删除 `trade_date` 或 `stock_code` 缺失的记录。
3. 只保留 `market_type={1,4}` 的主板股票。
4. 删除开盘价、最高价、最低价、收盘价中任一价格小于等于 0 的记录。
5. 剔除 ST/PT 股票。
6. 剔除流通市值低于 `DEFAULT_MIN_FLOAT_MARKET_VALUE` 的记录。
7. 按 `stock_code` 和 `trade_date` 排序。
8. 对可前值填充的辅助字段，在每只股票内部向前填充。
9. 按 `stock_code + trade_date` 去重。

清洗日志会输出到：

```text
cleaning_step_log.csv
cleaning_exclusion_reason_summary.csv
missing_value_ffill_summary.csv
```

## 6. ChangeRatio 时序 z-score

策略使用数据库中的 `ChangeRatio` 字段计算涨跌幅的时序 z-score。

这里的 z-score 是每只股票自己的时间序列 z-score，不是截面 z-score。

对每只股票，某一天的 z-score 使用该股票此前 `DEFAULT_ZSCORE_WINDOW` 个交易日的 `ChangeRatio` 均值和标准差计算：

```text
change_ratio_zscore_t =
    (ChangeRatio_t - mean(ChangeRatio_{t-63:t-1}))
    / std(ChangeRatio_{t-63:t-1})
```

注意：

1. 只使用当日之前的数据，不使用未来信息。
2. 当前窗口长度为 63 个交易日。
3. 历史窗口不足 63 日时，该日 z-score 为空。
4. 如果标准差为空或等于 0，则 z-score 为空。

## 7. 五重筛选逻辑

假设观察日为 `T`，程序对每只股票在 `T` 日执行五重筛选。

### 7.1 确定唯一 peak

在每只股票最近 `DEFAULT_LOOKBACK_DAYS=189` 个交易记录内，寻找第一次出现的最高收盘价，并定义为唯一 `peak`。

后续所有筛选条件都围绕同一个 `peak` 展开。

如果最高收盘价在窗口内出现多次，只取第一次出现的最高收盘价作为 `peak`。

### 7.2 第一重筛选：低点到 peak 强上涨

在 lookback 窗口内，找到 `peak` 之前的最低收盘价。

要求从该低点到唯一 `peak` 的累计涨幅超过 50%：

```text
low_to_peak_return > 50%
```

该收益使用 `Dretwd` 累乘计算，不使用未复权价格直接计算。

### 7.3 第二重筛选：peak 距离 T 日足够近

要求唯一 `peak` 距离观察日 `T` 不超过 63 个交易日：

```text
peak_trade_date >= T 日向前第 63 个交易日
```

也就是说，`peak` 必须出现在 T 日前最近 63 个交易日内。

### 7.4 第三重筛选：peak 到 T 日充分回撤

要求从唯一 `peak` 到观察日 `T` 的累计收益小于 `-10%`：

```text
drawdown_from_peak_to_signal < -10%
```

该回撤同样使用 `Dretwd` 累乘计算。

### 7.5 第四重筛选：T 日是 peak 后最低点

要求观察日 `T` 的收盘价，是从唯一 `peak` 之后到 `T` 日之间的最低收盘价：

```text
T 日收盘价 = peak 后截至 T 日的最低收盘价
```

如果存在并列最低，代码按最后一次最低点处理，因此只有当前 T 日正好处在该最低点位置时才通过。

### 7.6 第五重筛选：peak 前 z-score 不过度异常

取唯一 `peak` 前 `DEFAULT_ZSCORE_WINDOW=63` 个 `ChangeRatio` 时序 z-score。

要求该窗口完整，且窗口内 z-score 绝对值最大值不超过阈值：

```text
max(abs(change_ratio_zscore)) <= DEFAULT_ZSCORE_LIMIT
```

当前参数为：

```text
max(abs(change_ratio_zscore)) <= 2.5
```

如果股票在 peak 前不足 63 个有效 z-score，则不能进入最终股票池。

### 7.7 最终入池条件

只有同时满足五条筛选条件的股票，才能进入最终 long-only 信号池。

任意一条不满足，都不会进入最终股票池。

最终信号池输出为：

```text
03_factor1_long_only_screened_stock_pool.csv
```

## 8. 删除历史不足的信号

虽然计算过程中需要保留足够的历史数据用于 lookback 和 z-score 计算，但正式输出只保留回测区间内且历史足够的信号。

代码会过滤掉历史有效交易日数少于 `DEFAULT_LOOKBACK_DAYS=189` 的记录：

```text
full_history_valid_days >= lookback_days
```

因此，2018 年开始后前一段历史不足的信号不会进入最终输出。

## 9. 因子值定义

通过五重筛选后，策略将 peak 到 T 日的回撤收益作为因子值：

```text
momentum_raw = drawdown_from_peak_to_signal
```

该值主要用于：

1. 每日交易明细中按因子值排序。
2. 信号池内部 IC / RankIC 检验。
3. 诊断分析。

当前 long-only 主流程不是截面分组多空回测，因此不会在主流程中用 3sigma 和截面 z-score 构造多空组合。

## 10. T+1 开盘可交易过滤

所有股票都是在 `T` 日收盘后出信号。

实际买入发生在下一交易日 `T+1` 开盘。

程序会把 T 日信号映射到 T+1 实际买入日：

```text
holding_start_trade_date = next_trade_date
```

然后检查 T+1 开盘是否可以买入。

以下情况会被视为不可交易：

1. T+1 开盘价缺失。
2. 未来持有期收益缺失。
3. T+1 开盘价等于涨停价。
4. T+1 开盘价等于跌停价。

只有满足以下条件的股票才进入实际可买股票池：

```text
is_tradable_next_open = True
```

交易过滤原因保存在：

```text
trade_filter_reason
```

## 11. 未来持有期收益计算

单只股票事件收益定义为从 `T+1` 到 `T+holding_days` 的 `Dretwd` 累乘收益。

当前持有期为 63 个交易日，因此：

```text
next_period_return =
    product(1 + Dretwd_{T+1 到 T+63}) - 1
```

该字段在交易过滤前为：

```text
next_period_return_before_trade_filter
```

通过 T+1 开盘可交易过滤后，保留为：

```text
next_period_return
```

如果股票 T+1 不可交易，则 `next_period_return` 为空，不进入真实可买股票池。

## 12. 每日买入股票排序

在同一个实际买入日内，所有可交易股票按因子值从大到小排序。

排序字段为：

```text
momentum_raw
```

由于：

```text
momentum_raw = drawdown_from_peak_to_signal
```

所以排序本质上是按 peak 到 T 日的回撤收益排序。

排序结果用于交易明细图片和人工检查，不改变 long-only 组合的等权买入口径。

## 13. Long-only 组合构造

策略不做空，只做多。

同一个信号日 T 的所有 T+1 可买股票构成一个 cohort。

cohort 内部股票等权：

```text
单只股票权重 = 1 / 当日可买股票数量
```

每个 cohort 占用组合资金的：

```text
1 / holding_days
```

当前 `holding_days=63`，因此每个新 cohort 占用组合资金的 `1/63`。

每天可能同时持有过去 63 个交易日产生的多个 cohort，因此组合是滚动持仓结构。

## 14. 每日组合收益计算

每日个股收益使用当日 `Dretwd`。

每个活跃持仓股票在某一天的收益贡献为：

```text
个股当日收益贡献 =
    个股当日 Dretwd
    * cohort 内等权权重
    / holding_days
```

每日组合收益为当天所有活跃持仓股票收益贡献之和：

```text
long_only_return_t =
    sum(个股当日 Dretwd * cohort 内等权权重 / holding_days)
```

如果当天没有任何活跃持仓，则：

```text
long_only_return_t = 0
```

这意味着没有信号且没有历史持仓时，组合资金保持现金，收益记为 0。

如果当天没有新信号，但仍有过去信号形成的持仓，则 NAV 会根据这些历史持仓的当日 `Dretwd` 变化。

## 15. NAV 计算

初始 NAV 为 1。

当前代码使用加法净值口径：

```text
NAV_{t+1} = NAV_t + long_only_return_t
```

也就是说：

```text
long_only_nav_t =
    1 + cumulative_sum(long_only_return)
```

该口径不是复利累乘：

```text
不是 NAV_t = product(1 + long_only_return_t)
```

这样做是为了避免在事件收益或滚动 sleeve 结构下重复复利。

## 16. 绩效指标

基于每日 `long_only_return` 和 `long_only_nav`，程序计算：

1. Final NAV
2. 累计收益
3. 年化收益
4. 年化波动
5. Sharpe
6. 最大回撤
7. 胜率
8. 持仓期数
9. 空仓期数

NAV 图中显示以下核心指标：

```text
Final NAV
CumRet
Sharpe
MaxDD
```

## 17. 主策略输出文件

主策略运行后，主要输出文件包括：

| 文件 | 内容 |
|---|---|
| `03_strong_momentum_factor_all_stocks.csv` | 所有股票每日筛选状态和未通过原因 |
| `03_factor1_long_only_screened_stock_pool.csv` | 五重筛选全部通过后的最终信号池 |
| `04_factor1_long_only_signal_pool_with_forward_returns.csv` | 加入 T+1 可交易状态和未来收益后的信号池 |
| `04_factor1_long_only_tradable_open_buy_dates.csv` | 实际买入交易日、可买股票数、对应信号日 |
| `05_factor1_long_only_holding_period_returns.csv` | long-only 每日收益和 NAV |
| `05_factor1_long_only_nav_curve.png` | long-only NAV 曲线 |
| `06_factor1_long_only_portfolio_performance_table.csv` | 组合绩效表 |
| `06_factor1_long_only_portfolio_performance_detail.csv` | 组合绩效明细 |
| `factor1_long_only_performance_summary.csv` | 绩效汇总 |
| `07_factor1_long_only_drawdown_series.csv` | 回撤序列 |
| `08_factor1_long_only_yearly_performance.csv` | 年度表现 |
| `strong_momentum_screening_summary.csv` | 筛选结果摘要 |
| `strong_momentum_filter_step_summary.csv` | 五重筛选漏斗 |
| `factor1_long_only_step_log.csv` | 主流程步骤日志 |
| `run_summary.csv` | 运行参数和输出路径摘要 |

## 18. 诊断脚本工作流程

诊断脚本为：

```text
factor1_LongOnly_diagnostics.py
```

它读取主策略输出，并生成以下诊断：

1. 检查主策略输出文件是否完整。
2. 读取最终信号池、未来收益文件、每日 NAV 文件和回撤文件。
3. 统计原始信号数量、T+1 可交易信号数量、空仓天数和持仓天数。
4. 汇总交易过滤原因。
5. 统计每日信号数和可交易信号数。
6. 计算收益分布。
7. 计算最终信号池内部 IC 和 RankIC。
8. 绘制累计 IC 曲线。
9. 绘制筛选漏斗。
10. 输出指定交易日实际可以买入股票的图片。

如果指定交易日不是交易日，或属于历史不足阶段，或当天没有符合条件且可在开盘买入的股票，诊断脚本会输出提示图片。

## 19. 指定交易日买入明细

可以在诊断脚本中修改参数：

```text
DEFAULT_TRADE_DETAIL_DATE
```

也可以通过命令行传入：

```powershell
python .\factor1_LongOnly_diagnostics.py --trade-date 2019-07-19
```

这里输入的是实际买入日，即：

```text
holding_start_trade_date
```

不是信号日 `trade_date`。

交易明细图片会按当日可买股票的 `momentum_raw` 从大到小排序。

## 20. 简化流程图

```text
读取 SQL 或本地缓存行情
-> 清洗数据并前值填充
-> 计算每只股票 ChangeRatio 时序 z-score
-> 每个 T 日寻找 189 日窗口内第一次最高收盘价 peak
-> 围绕同一个 peak 执行五重筛选
-> 删除历史不足 189 日的信号
-> 得到最终 long-only 信号池
-> T+1 开盘检查涨跌停和数据缺失
-> 保留实际可买股票
-> 用 Dretwd 计算 T+1 到 T+63 事件收益
-> 将每日新信号展开成 63 日滚动持仓 sleeve
-> 计算每日 long_only_return
-> 按 NAV_{t+1}=NAV_t+long_only_return_t 生成净值
-> 输出 CSV、NAV 图、绩效表和诊断图
```

## 21. 核心口径总结

1. 信号日是 `T` 日。
2. 实际买入日是 `T+1` 开盘。
3. 五重筛选必须全部围绕同一个唯一 `peak`。
4. 只有五重筛选全部满足，股票才进入最终信号池。
5. 只有 T+1 开盘可交易，股票才进入真实可买股票池。
6. 历史路径收益和未来持有期收益都使用 `Dretwd` 累乘。
7. 原始未复权价格不用于收益率计算。
8. 策略只做多，不构造空头组合。
9. 每个信号日 cohort 内股票等权。
10. 每个 cohort 占用 `1 / holding_days` 的组合资金。
11. 每日 NAV 使用加法口径，不使用重复复利。
