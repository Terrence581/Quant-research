# 强势股回调 Long-only 策略

本文件夹包含一个面向 A 股主板股票的强势股回调事件型 long-only 策略研究脚本。策略只做多，不构造空头组合，重点检验满足筛选条件的股票在未来持有期内是否具有正向收益。

## 文件说明

- `factor1_LongOnly.py`：主策略脚本。负责读取和清洗行情数据、执行五条筛选规则、检查 T+1 开盘可交易性、使用 `Dretwd` 计算未来收益、生成 long-only 每日 NAV，并输出 CSV 和图表。
- `factor1_LongOnly_diagnostics.py`：诊断脚本。读取主策略输出，生成信号质量、交易过滤原因、收益分布、IC/RankIC、筛选漏斗，以及指定交易日买入明细图片。

## 筛选逻辑

每个观察日 `T`，脚本在每只股票的 lookback 窗口内寻找第一次出现的最高收盘价，并将其定义为唯一 `peak`。股票只有同时满足以下五条条件，才会进入最终 long-only 信号池：

1. 在过去 lookback 窗口内，从 peak 前低点到唯一 `peak` 的累计涨幅超过设定阈值。
2. 唯一 `peak` 距离观察日 `T` 不超过设定的最大交易日数。
3. 从 `peak` 到 `T` 的回撤超过设定阈值。
4. `T` 日收盘价是该 `peak` 之后截至 `T` 的最低收盘价。
5. `peak` 前窗口内 `ChangeRatio` 时序 z-score 的最大绝对值不超过设定阈值，并且该 z-score 历史窗口长度完整。

## 交易与收益口径

- 信号在 `T` 日收盘后产生。
- 可交易股票在下一交易日 `T+1` 开盘买入。
- 如果股票在 `T+1` 开盘价等于涨停价或跌停价，则视为不可交易，不进入实际可买股票池。
- 未来持有期收益使用 `Dretwd` 累乘计算，不使用未复权价格计算。
- 策略只做多，不建立空头组合。
- 每日 NAV 使用研究脚本中的加法净值口径：

```text
NAV_{t+1} = NAV_t + long_only_return_t
```

## 主要输出

主策略常见输出包括：

- `03_factor1_long_only_screened_stock_pool.csv`：五条筛选全部满足后的最终信号池。
- `04_factor1_long_only_signal_pool_with_forward_returns.csv`：包含 T+1 开盘可交易状态和未来收益的信号池。
- `04_factor1_long_only_tradable_open_buy_dates.csv`：实际买入交易日、可买股票数和对应信号日。
- `05_factor1_long_only_holding_period_returns.csv`：每日 long-only 收益和 NAV 序列。
- `05_factor1_long_only_nav_curve.png`：NAV 曲线，图中包含 Final NAV、累计收益、Sharpe 和最大回撤。

## 使用方法

运行主策略前，需要通过环境变量提供数据库密码：

```powershell
$env:MYSQL_PASSWORD = "你的数据库密码"
python .\factor1_LongOnly.py
```

主策略输出生成后，再运行诊断脚本：

```powershell
python .\factor1_LongOnly_diagnostics.py
```

如果需要查看某个实际买入日的具体交易股票，可以修改 `factor1_LongOnly_diagnostics.py` 中的 `DEFAULT_TRADE_DETAIL_DATE`，或通过命令行传入：

```powershell
python .\factor1_LongOnly_diagnostics.py --trade-date 2019-07-19
```

这里的日期是实际买入日，即 `holding_start_trade_date`，不是信号日 `trade_date`。
