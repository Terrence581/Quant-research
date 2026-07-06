# A股动量策略研究仓库

本仓库包含两个 A 股动量相关研究模块：

1. `截面动量因子/`：面向全市场截面的动量因子研究框架，重点检验动量因子在股票截面上的排序能力，并输出分层组合、多空组合、IC/RankIC 和诊断图表。
2. `strong_momentum_LongOnly/`：面向强势股回调事件的 long-only 策略研究模块，重点检验满足强动量回调路径筛选的股票，在后续持有期内是否具有正向收益。

两个模块都遵循“信号在 T 日收盘后形成、交易在 T+1 执行、未来收益不能提前泄露”的回测原则，但研究目标、股票池构造方式和组合收益口径不同。

---

## 1. 项目文件

```text
.
├── README.md
├── 截面动量因子/
│   ├── calculate_momentum_factor_sql.py        # 截面动量因子主计算脚本
│   ├── momentum_factor_diagnostics.py          # 截面动量因子诊断图表脚本
│   └── plot_momentum_factor_figures.py         # 报告图表汇总脚本
└── strong_momentum_LongOnly/
    ├── factor1_LongOnly.py                     # 强势股回调 long-only 主策略脚本
    ├── factor1_LongOnly_diagnostics.py         # long-only 策略诊断脚本
    └── README.md                               # long-only 策略说明
```

---

## 2. 两个模块的区别

| 维度 | `截面动量因子/` | `strong_momentum_LongOnly/` |
|---|---|---|
| 研究目标 | 检验截面动量因子的排序能力 | 检验强势股回调事件信号的未来收益有效性 |
| 信号对象 | 每个调仓日上的全市场股票截面 | 每个观察日满足路径条件的事件股票池 |
| 信号定义 | 过去一段时间收益率或月频动量 | 强上涨、近高点、回撤、创新低、ChangeRatio z-score 约束 |
| 交易方向 | 分层组合，并构造高动量组减低动量组的多空组合 | 只做多，不构造空头组合 |
| 交易时点 | T 日收盘形成因子，T+1 开盘检查可交易性 | T 日出信号，T+1 开盘检查可交易性并买入 |
| 收益口径 | 分组收益、多空价差收益、组合 NAV | long-only 每日收益、long-only NAV、交易日明细 |
| 主要诊断 | IC、RankIC、分层净值、多空净值、排序检验 | 信号池漏斗、可交易过滤、IC/RankIC、NAV、指定交易日买入明细 |

---

## 3. `截面动量因子/` 模块

该模块用于研究 A 股截面动量因子，覆盖从行情数据读取、样本清洗、因子计算、分层组合、未来收益、IC 检验、多空价差组合到图表诊断的完整流程。

当前框架支持两类调仓频率：

- 日频：按交易日回看和持有。
- 月频：按月末交易日形成信号，支持 `skip` 参数跳过最近月份。

### 3.1 主计算脚本

`截面动量因子/calculate_momentum_factor_sql.py`

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

### 3.2 诊断脚本

`截面动量因子/momentum_factor_diagnostics.py`

负责读取主脚本输出结果，生成诊断表和图：

1. 自动读取主脚本默认参数。
2. 自动识别日频或月频输出目录。
3. 读取分组收益、多空收益、IC 汇总等小体积文件。
4. 绘制 IC 曲线、累计 IC 曲线、分层净值曲线、多空净值曲线。
5. 输出排序检验表和诊断摘要。

该脚本不依赖大体积个股明细文件，因此可以在不保存 2GB 级别中间文件的情况下运行。

### 3.3 报告图表脚本

`截面动量因子/plot_momentum_factor_figures.py`

用于生成报告需要的汇总图表，例如：

- 不同 lookback 参数下的 IC Mean。
- 不同 holding 参数下的因子有效期。
- Group1 至 Group10 最终收益或最终 NAV 柱状图。
- Group1 至 Group10 绩效表。

### 3.4 截面动量因子定义

日频动量以 T 日为当前截面，使用 T 日之前的交易日数据计算。

例如 5 日动量：

```text
momentum = 区间末收盘价 / 区间首日收盘价 - 1
```

其中区间为 T-5 至 T-1 共 5 个交易日。

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

### 3.5 截面动量输出

输出结果默认保存在：

```text
截面动量因子/output/
```

常见输出文件：

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

## 4. `strong_momentum_LongOnly/` 模块

该模块包含一个面向 A 股主板股票的强势股回调事件型 long-only 策略。策略只做多，不构造空头组合，重点检验满足筛选条件的股票在未来持有期内是否具有正向收益。

更详细的策略说明见：`strong_momentum_LongOnly/README.md`。

### 4.1 主策略脚本

`strong_momentum_LongOnly/factor1_LongOnly.py`

负责完整策略流程：

1. 从数据库或本地缓存读取行情数据。
2. 清洗样本并保留指定市场类型。
3. 对每只股票计算 `ChangeRatio` 的时序 z-score。
4. 在每个观察日执行强势股回调五条筛选。
5. 生成最终信号池。
6. 检查 T+1 开盘是否可买入。
7. 使用 `Dretwd` 累乘计算未来持有期收益。
8. 计算 long-only 每日收益和 NAV。
9. 输出信号池、可交易明细、绩效表和 NAV 图。

### 4.2 诊断脚本

`strong_momentum_LongOnly/factor1_LongOnly_diagnostics.py`

负责读取主策略输出，生成诊断结果：

1. 检查主策略输出文件是否完整。
2. 汇总最终信号池和 T+1 可交易股票池。
3. 生成筛选漏斗和交易过滤诊断。
4. 计算信号池 IC / RankIC。
5. 绘制累计 IC 和 long-only NAV 图。
6. 输出指定交易日实际可以买入的股票明细图片。

### 4.3 五条筛选逻辑

每个观察日 `T`，脚本在每只股票的 lookback 窗口内寻找第一次出现的最高收盘价，并将其定义为唯一 `peak`。股票只有同时满足以下五条条件，才会进入最终 long-only 信号池：

1. 在过去 lookback 窗口内，从 peak 前低点到唯一 `peak` 的累计涨幅超过设定阈值。
2. 唯一 `peak` 距离观察日 `T` 不超过设定的最大交易日数。
3. 从 `peak` 到 `T` 的回撤超过设定阈值。
4. `T` 日收盘价是该 `peak` 之后截至 `T` 的最低收盘价。
5. `peak` 前窗口内 `ChangeRatio` 时序 z-score 的最大绝对值不超过设定阈值，并且该 z-score 历史窗口长度完整。

### 4.4 Long-only 交易与收益口径

- 信号在 `T` 日收盘后产生。
- 实际买入发生在下一交易日 `T+1` 开盘。
- 如果股票在 `T+1` 开盘价等于涨停价或跌停价，则视为不可交易，不进入实际可买股票池。
- 未来持有期收益使用 `Dretwd` 累乘计算，不使用未复权价格计算。
- 策略只做多，不建立空头组合。
- 每日 NAV 使用研究脚本中的加法净值口径：

```text
NAV_{t+1} = NAV_t + long_only_return_t
```

### 4.5 Long-only 输出

主策略常见输出包括：

| 文件 | 内容 |
|---|---|
| `03_factor1_long_only_screened_stock_pool.csv` | 五条筛选全部满足后的最终信号池 |
| `04_factor1_long_only_signal_pool_with_forward_returns.csv` | 包含 T+1 开盘可交易状态和未来收益的信号池 |
| `04_factor1_long_only_tradable_open_buy_dates.csv` | 实际买入交易日、可买股票数和对应信号日 |
| `05_factor1_long_only_holding_period_returns.csv` | 每日 long-only 收益和 NAV 序列 |
| `05_factor1_long_only_nav_curve.png` | NAV 曲线，图中包含 Final NAV、累计收益、Sharpe 和最大回撤 |

---

## 5. 共同回测原则

两个模块都遵循以下原则：

1. 不使用 T+1 之后的数据计算 T 日信号。
2. 不在原始行情阶段提前剔除涨跌停股票。
3. T 日收盘后形成信号，T+1 开盘检查可交易性。
4. 涨跌停过滤只发生在信号或组合已经形成之后。
5. 未来收益使用日收益率累乘，不应简单累加。
6. 如果输出文件被 Excel、Word 或图片查看器打开，重新运行脚本时可能出现 `PermissionError`。

---

## 6. 运行方式

### 6.1 运行截面动量因子模块

```bash
python 截面动量因子/calculate_momentum_factor_sql.py
python 截面动量因子/momentum_factor_diagnostics.py
python 截面动量因子/plot_momentum_factor_figures.py
```

也可以先进入文件夹再运行：

```bash
cd 截面动量因子
python calculate_momentum_factor_sql.py
python momentum_factor_diagnostics.py
python plot_momentum_factor_figures.py
```

### 6.2 运行 Long-only 模块

运行主策略前，需要通过环境变量提供数据库密码：

```powershell
$env:MYSQL_PASSWORD = "你的数据库密码"
cd strong_momentum_LongOnly
python .\factor1_LongOnly.py
python .\factor1_LongOnly_diagnostics.py
```

如果需要查看某个实际买入日的具体交易股票，可以修改 `factor1_LongOnly_diagnostics.py` 中的 `DEFAULT_TRADE_DETAIL_DATE`，或通过命令行传入：

```powershell
python .\factor1_LongOnly_diagnostics.py --trade-date 2019-07-19
```

这里的日期是实际买入日，即 `holding_start_trade_date`，不是信号日 `trade_date`。

---

## 7. 绩效和 IC 指标

两个模块都会用到部分通用评估指标：

| 指标 | 含义 |
|---|---|
| 累计收益 | 最终 NAV 相对初始 NAV 的变化 |
| 年化收益 | 将样本期收益按年化期数折算 |
| 年化波动 | 收益率标准差乘以年化系数 |
| Sharpe | 年化收益 / 年化波动 |
| 最大回撤 | NAV 相对历史最高点的最大跌幅 |
| IC | 因子值或信号强度与未来收益的截面 Pearson 相关 |
| RankIC | 因子值或信号强度与未来收益的截面 Spearman 秩相关 |

解释：

- IC Mean > 0：因子值越高，未来收益越高。
- IC Mean < 0：因子值越高，未来收益越低，可能存在反转效应。
- t 值绝对值越大，统计显著性越强。
- p 值越小，统计显著性越强。

---

## 8. 项目状态

当前项目已实现：

- 日频截面动量因子研究。
- 月频截面动量因子研究。
- 月频 `skip` 参数。
- Group1 至 Group10 分层组合。
- 高动量组减低动量组的多空组合。
- 截面动量 IC、RankIC、绩效指标和回撤统计。
- 强势股回调 long-only 事件策略。
- 强势股回调五条筛选和 T+1 开盘可交易过滤。
- Long-only NAV、交易日明细、IC/RankIC 和诊断图表。

后续可扩展方向：

- 多因子横向比较。
- 参数网格搜索。
- 行业中性化版本。
- 风格暴露分析。
- 交易成本和冲击成本建模。
