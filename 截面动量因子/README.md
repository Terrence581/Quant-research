# 截面动量因子

本文件夹用于研究 A 股截面动量因子，覆盖从行情数据读取、样本清洗、因子计算、分层组合、未来收益、IC 检验、多空价差组合到图表诊断的完整流程。

当前框架支持两类调仓频率：

- 日频：按交易日回看和持有。
- 月频：按月末交易日形成信号，支持 `skip` 参数跳过最近月份。

核心回测原则：

- 因子在 T 日收盘后计算。
- T+1 日开盘前准备交易。
- 不在原始数据阶段提前剔除涨跌停股票。
- 只在选出多头组和空头组后，根据 T+1 开盘价是否触及涨跌停价进行交易过滤。
- 未来收益使用 `return_without_dividend` 累乘计算，避免简单累加导致的收益口径偏差。
- 多空价差组合收益定义为：高动量组收益 - 低动量组收益。

---

## 1. 文件结构

```text
截面动量因子/
├── README.md
├── calculate_momentum_factor_sql.py        # 主计算脚本
├── momentum_factor_diagnostics.py          # 诊断图表脚本
└── plot_momentum_factor_figures.py         # 报告图表汇总脚本
```

---

## 2. 脚本说明

### 2.1 主计算脚本

`calculate_momentum_factor_sql.py`

负责完整回测主流程：

1. 从 MySQL 或本地缓存读取 A 股行情数据。
2. 统一字段名和日期格式。
3. 执行基础样本清洗。
4. 构造日频或月频调仓日历。
5. 计算原始动量因子。
6. 进行 3sigma 缩尾和 Z-score 标准化。
7. 按截面因子值排序并分组。
8. 计算未来持有期收益。
9. 在 T+1 开盘阶段过滤一字涨停或一字跌停股票。
10. 计算分组等权收益、多空价差收益、净值、回撤和绩效指标。
11. 输出 CSV 结果文件。

### 2.2 诊断脚本

`momentum_factor_diagnostics.py`

负责读取主脚本输出结果，生成诊断表和图：

1. 自动读取主脚本默认参数。
2. 自动识别日频或月频输出目录。
3. 读取分组收益、多空收益、IC 汇总等小体积文件。
4. 绘制 IC 曲线、累计 IC 曲线、分层净值曲线、多空净值曲线。
5. 输出排序检验表和诊断摘要。

该脚本不依赖大体积个股明细文件，因此可以在不保存 2GB 级别中间文件的情况下运行。

### 2.3 报告图表脚本

`plot_momentum_factor_figures.py`

用于生成报告需要的汇总图表，例如：

- 不同 lookback 参数下的 IC Mean。
- 不同 holding 参数下的因子有效期。
- Group1 至 Group10 最终收益或最终 NAV 柱状图。
- Group1 至 Group10 绩效表。

---

## 3. 当前默认参数

主脚本中的主要默认参数如下：

| 参数 | 当前默认值 | 含义 |
|---|---:|---|
| `DEFAULT_START_DATE` | `2015-01-01` | 样本开始日期 |
| `DEFAULT_END_DATE` | `2026-01-01` | 样本结束日期 |
| `DEFAULT_REBALANCE_FREQUENCY` | `monthly` | 当前默认月频调仓 |
| `DEFAULT_LOOKBACK_DAYS` | `50` | 日频回看交易日数 |
| `DEFAULT_HOLDING_DAYS` | `5` | 日频持有交易日数 |
| `DEFAULT_LOOKBACK_MONTHS` | `3` | 月频回看月数 |
| `DEFAULT_HOLDING_MONTHS` | `1` | 月频持有月数 |
| `DEFAULT_S` | `0` | 月频 skip 参数 |
| `DEFAULT_GROUP_NUM` | `10` | 分组数量 |
| `DEFAULT_MIN_FLOAT_MARKET_VALUE` | `5,000,000` | 流通市值下限 |
| `DEFAULT_MONTHLY_LOOKBACK_MIN_TRADING_DAYS` | `21` | 月频回看最少有效交易日 |

---

## 4. 因子定义

### 4.1 日频动量

日频动量以 T 日为当前截面，使用 T 日之前的交易日数据计算。

例如 5 日动量：

```text
momentum = 区间末收盘价 / 区间首日收盘价 - 1
```

其中区间为 T-5 至 T-1 共 5 个交易日。

### 4.2 月频动量

月频动量使用经典动量框架：

```text
MOM_i(t, T, s) = product(1 + r_i,t-k) - 1
```

其中：

- `T`：回看窗口总月数。
- `s`：跳过最近月数。
- `r_i,t-k`：股票 i 在过去第 k 期的收益。

当前实现中，月频动量使用 `return_without_dividend` 在回看区间内进行日收益率累乘计算：

```text
monthly_momentum = product(1 + return_without_dividend) - 1
```

---

## 5. 交易与收益口径

### 5.1 调仓时点

```text
T 日收盘后：
  计算因子、缩尾、标准化、排序、分组

T+1 日开盘：
  检查多头组和空头组股票是否一字涨停或一字跌停
  不可交易股票从当期组合中剔除

持有期结束：
  使用持有区间内日收益率累乘得到未来收益
```

### 5.2 涨跌停过滤

涨跌停过滤只发生在选出多头组和空头组之后。

判断逻辑：

```text
next_open_price == next_limit_up_price
或
next_open_price == next_limit_down_price
```

这样处理可以避免在 T 日之前错误使用 T+1 信息，也避免提前改变截面分组结果。

### 5.3 分组收益

每组内部股票等权：

```text
group_return = mean(stock_forward_return)
```

高动量组和低动量组定义：

```text
Group1 = 最高动量组
GroupN = 最低动量组
```

多空价差组合：

```text
long_short_return = high_momentum_group_return - low_momentum_group_return
```

净值：

```text
NAV_t = product(1 + return_t)
```

---

## 6. 运行方式

### 6.1 环境依赖

主要 Python 依赖：

```text
numpy
pandas
matplotlib
```

建议使用 Python 3.10 或以上版本。

### 6.2 运行主计算脚本

在当前文件夹中运行：

```bash
python calculate_momentum_factor_sql.py
```

也可以在仓库根目录运行：

```bash
python 截面动量因子/calculate_momentum_factor_sql.py
```

### 6.3 运行诊断脚本

在当前文件夹中运行：

```bash
python momentum_factor_diagnostics.py
```

也可以在仓库根目录运行：

```bash
python 截面动量因子/momentum_factor_diagnostics.py
```

### 6.4 运行报告图表脚本

在当前文件夹中运行：

```bash
python plot_momentum_factor_figures.py
```

也可以在仓库根目录运行：

```bash
python 截面动量因子/plot_momentum_factor_figures.py
```

---

## 7. 输出目录

输出结果默认保存在当前文件夹下：

```text
output/
```

日频输出目录格式：

```text
daily_lb{lookback_days}_hd{holding_days}_g{group_num}_mkt{market_tag}
```

月频输出目录格式：

```text
monthly_lb{lookback_months}m_s{s}_hd{holding_months}m_g{group_num}_mkt{market_tag}
```

---

## 8. 主要输出文件

| 文件 | 内容 |
|---|---|
| `09_quantile_equal_weight_returns.csv` | Group1 至 GroupN 分组等权收益 |
| `10_long_short_hedge_returns.csv` | 高动量组、低动量组、多空价差组合收益和净值 |
| `12_momentum_ic_series.csv` | 每期 IC 和 RankIC |
| `12_momentum_ic_ir_summary.csv` | IC / RankIC 汇总统计 |
| `12_momentum_factor_value_statistics.csv` | 因子值分布统计 |
| `12_momentum_factor_input_summary.csv` | 因子输入样本统计 |
| `momentum_performance_summary.csv` | 组合绩效汇总 |
| `momentum_drawdown_series.csv` | 回撤序列 |
| `momentum_yearly_performance.csv` | 年度表现 |
| `run_summary.csv` | 本次运行参数和样本数量摘要 |

诊断图表常见输出：

| 文件 | 内容 |
|---|---|
| `12_momentum_daily_ic_curve.png` | 单期 IC 曲线 |
| `12_momentum_cumulative_ic_curve.png` | 累计 IC 曲线 |
| `13_momentum_quantile_nav_curve.png` | 分层组合净值曲线 |
| `14_momentum_long_short_nav_curve.png` | 多空价差组合净值曲线 |
| `11_momentum_sort_test_table.png` | 排序检验绩效表 |

---

## 9. 绩效指标

| 指标 | 计算方式 |
|---|---|
| 累计收益 | `最后一期 NAV - 1` |
| 年化收益 | `最后一期 NAV ** (年化期数 / 样本期数) - 1` |
| 年化波动 | `收益率标准差 * sqrt(年化期数)` |
| Sharpe | `年化收益 / 年化波动` |
| 最大回撤 | `NAV / 历史最高 NAV - 1` 的最小值 |
| 胜率 | `收益率 > 0` 的期数占比 |
| 样本期数 | 有效收益观测数量 |

年化期数：

```text
日频 = 252 / holding_days
月频 = 12 / holding_months
```

---

## 10. 注意事项

1. 不要对涨跌停状态做前值填充。
2. 不要在原始行情阶段提前剔除涨跌停股票。
3. 不要使用 T+1 之后的数据计算 T 日因子。
4. 未来收益应使用日收益率累乘，不应简单累加。
5. 如果输出文件被 Excel、Word 或图片查看器打开，重新运行脚本时可能出现 `PermissionError`。
6. 大体积个股明细文件默认可以关闭；诊断脚本优先读取小体积汇总文件。
