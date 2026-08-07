# A 股量化研究与因子回测

本仓库汇总 A 股量化研究项目，覆盖截面动量、强趋势事件策略、QTrend20、财务增长因子，以及 LLM 因子挖掘 Agent 设计文档。各模块用于因子构造、信号筛选、分组回测、IC/RankIC 检验、组合净值和诊断分析。

## 快速导航

| 模块 | 主要内容 |
|---|---|
| [截面动量因子](./截面动量因子/) | 日频/月频动量、分组组合、多空组合、IC/RankIC 和诊断图 |
| [强趋势因子策略](./强趋势因子策略/) | 左侧回调低点 long-only、右侧回撤后突破 |
| [QTrend20](./QTrend20/) | 20 日价格路径平稳度因子和 Return20 分层 |
| [财务因子](./财务因子/) | 营收、净利润同比/环比增长因子 |
| [LLM Agent架构](./LLM%20Agent架构/) | A 股量化因子挖掘 Agent 的完整与精简架构 |

## 项目结构

```text
.
├── 截面动量因子/
│   ├── calculate_momentum_factor_sql.py
│   ├── momentum_factor_diagnostics.py
│   └── plot_momentum_factor_figures.py
├── 强趋势因子策略/
│   ├── 左侧买入/
│   │   ├── factor1_LongOnly.py
│   │   └── factor1_LongOnly_diagnostics.py
│   └── 右侧买入/
│       ├── factor2_2conditions.py
│       └── factor2_2conditions_diagnostics.py
├── QTrend20/
│   ├── qtrend20_factor_v3.py
│   └── qtrend20_factor_diagnostics_v3.py
├── 财务因子/
│   ├── Sales_YoY_Q1_factor.py
│   ├── Revenue_QoQ_weekly_factor.py
│   ├── Net_Profit_YoY_weekly_factor.py
│   ├── Net_Profit_QoQ_weekly_factor.py
│   └── *_analysis.py
└── LLM Agent架构/
    ├── LLM_A股量化因子挖掘Agent_完整项目架构.md
    └── LLM驱动的A股量化因子挖掘Agent_精炼修订版.md
```

## 1. 截面动量因子

目录：`截面动量因子/`。

主脚本支持日频和月频调仓，完成行情读取、样本清洗、动量计算、3-sigma 缩尾、Z-score 标准化、分组、T+1 可交易过滤、未来收益、NAV、IC/RankIC 和回撤统计。

默认月频配置：2015-01-01 至 2026-01-01，回看 3 个月，跳过最近 0 个月，持有 1 个月，10 组等权。实际参数以脚本顶部常量或命令行参数为准。

```bash
python 截面动量因子/calculate_momentum_factor_sql.py
python 截面动量因子/momentum_factor_diagnostics.py
python 截面动量因子/plot_momentum_factor_figures.py
```

## 2. 强趋势因子策略

### 左侧买入：强势股回调低点

入口：`强趋势因子策略/左侧买入/factor1_LongOnly.py`。

在最近 189 个交易日内寻找第一次出现的最高收盘价，并检查峰值前涨幅超过 50%、峰值距信号日不超过 63 日、回撤超过 10%、信号日处于峰值后阶段低点，以及峰值前 63 日 ChangeRatio 时序 Z-score 约束不超过 2.5。

当前代码默认参数为 2019-01-01 至 2021-12-31、主板 `market_type={1,4}`、持有 63 个交易日、等权 long-only。

### 右侧买入：回撤后突破

入口：`强趋势因子策略/右侧买入/factor2_2conditions.py`。

条件为：T 日收盘价严格突破此前 211 个交易日最高收盘价；同时，包含 T 日的最近 84 个交易日滚动最大回撤超过 23%。默认区间为 2018-01-01 至 2021-12-31，持有 84 个交易日，主板等权 long-only。

两类策略均采用 T 日收盘形成信号、T+1 开盘买入，并在执行阶段过滤一字涨跌停股票。

## 3. QTrend20

入口：

- `QTrend20/qtrend20_factor_v3.py`
- `QTrend20/qtrend20_factor_diagnostics_v3.py`

QTrend20 衡量 20 日收盘价路径相对首尾连线的偏离程度；因子值越小，价格路径越平稳。主回测先筛选 Return20 > 0，再按标准化 QTrend20 分组，输出 G1 多头、G1-G10 多空、IC/RankIC、因子有效期和 Return20 分层结果。

默认配置：2019-09-01 至 2024-09-01，回看 20 个交易日，持有 5 个交易日，10 组等权。

```bash
cd QTrend20
python qtrend20_factor_v3.py
python qtrend20_factor_diagnostics_v3.py
```

## 4. 财务因子

目录：`财务因子/`。

当前包含四类因子及对应分析脚本：

- `Sales_YoY_Q1_factor.py`：单季度营收同比增速；
- `Revenue_QoQ_weekly_factor.py`：单季度营收环比增速；
- `Net_Profit_YoY_weekly_factor.py`：单季度净利润同比增速；
- `Net_Profit_QoQ_weekly_factor.py`：单季度净利润环比增速。

`*_analysis.py` 脚本用于生成多空净值、IC/RankIC、分组收益和诊断结果。工作汇总中的 ROE > 5% 与 ROE > 10% 属于这些财务因子实验的筛选分支。

## 5. LLM 因子挖掘 Agent

目录：`LLM Agent架构/`。

- `LLM_A股量化因子挖掘Agent_完整项目架构.md`：完整 Agent 分层、任务流和审查架构；
- `LLM驱动的A股量化因子挖掘Agent_精炼修订版.md`：精简后的执行方案。

## 共同回测原则

- T 日收盘后形成信号，T+1 执行交易；
- 不使用信号形成之后的数据计算 T 日因子；
- 涨跌停过滤发生在交易执行阶段；
- 组合通常采用等权，具体口径以各模块脚本为准；
- 输出文件被 Excel 或图片查看器占用时，重新运行可能出现 `PermissionError`。

运行脚本前，请先检查数据缓存、MySQL 连接、输出目录和本机路径配置。
