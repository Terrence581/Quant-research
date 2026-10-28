# A-Share Quantitative Research and Factor Backtesting

This repository brings together quantitative research projects for China's A-share market, covering cross-sectional momentum, strong-trend event strategies, QTrend20, financial growth factors, and design documents for an LLM factor-mining agent. The modules support factor construction, signal screening, portfolio-group backtesting, information coefficient (IC) and rank information coefficient (RankIC) analysis, portfolio net asset value (NAV) calculations, and diagnostics.

## Quick Navigation

| Module | Main Features |
|---|---|
| [Cross-Sectional Momentum](./截面动量因子/) | Daily/monthly momentum, grouped portfolios, long–short portfolios, IC/RankIC, and diagnostic plots |
| [Strong-Trend Factor Strategies](./强趋势因子策略/) | Long-only entries at pullback lows and breakout entries following drawdowns |
| [QTrend20](./QTrend20/) | A 20-day price-path smoothness factor and analysis stratified by Return20 |
| [Financial Factors](./财务因子/) | Year-over-year and quarter-over-quarter growth factors for revenue and net profit |
| [LLM Agent Architecture](./LLM%20Agent架构/) | Full and streamlined architectures for an A-share factor-mining agent |

## Repository Structure

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

## 1. Cross-Sectional Momentum

Directory: `截面动量因子/`.

The main script supports daily and monthly rebalancing. It handles market data loading, sample cleaning, momentum calculation, 3-sigma winsorization, Z-score standardization, portfolio grouping, T+1 tradability filtering, forward returns, NAV, IC/RankIC, and drawdown statistics.

The default monthly configuration runs from 2015-01-01 to 2026-01-01, with a 3-month lookback, a 0-month skip period, a 1-month holding period, and 10 equally weighted groups. The constants at the top of the script or the command-line arguments determine the actual configuration.

```bash
python 截面动量因子/calculate_momentum_factor_sql.py
python 截面动量因子/momentum_factor_diagnostics.py
python 截面动量因子/plot_momentum_factor_figures.py
```

## 2. Strong-Trend Factor Strategies

### Pullback Entry: Buying Strong Stocks at Pullback Lows

Entry point: `强趋势因子策略/左侧买入/factor1_LongOnly.py`.

The strategy identifies the first occurrence of the highest closing price within the most recent 189 trading days and checks the following conditions:

- The price increase before the peak exceeds 50%.
- The peak is no more than 63 trading days before the signal date.
- The drawdown exceeds 10%.
- The signal date marks a low in the post-peak period.
- The time-series Z-score constraint for ChangeRatio over the 63 trading days before the peak does not exceed 2.5.

The current code defaults to 2019-01-01 through 2021-12-31, main-board stocks with `market_type={1,4}`, a 63-trading-day holding period, and an equally weighted long-only portfolio.

### Breakout Entry: Buying Breakouts After Drawdowns

Entry point: `强趋势因子策略/右侧买入/factor2_2conditions.py`.

The closing price on day T must strictly exceed the highest closing price over the preceding 211 trading days. At the same time, the rolling maximum drawdown over the most recent 84 trading days, including day T, must exceed 23%. The default backtest runs from 2018-01-01 to 2021-12-31, with an 84-trading-day holding period and an equally weighted long-only portfolio of main-board stocks.

Both strategies generate signals at the close of day T and buy at the open of T+1. Stocks locked at the upper or lower price limit throughout the trading session are excluded at the execution stage.

## 3. QTrend20

Entry points:

- `QTrend20/qtrend20_factor_v3.py`
- `QTrend20/qtrend20_factor_diagnostics_v3.py`

QTrend20 measures how far the 20-day closing-price path deviates from the straight line connecting its first and last observations. A lower factor value indicates a smoother price path. The main backtest first filters for Return20 > 0, then groups stocks by standardized QTrend20. Outputs include the G1 long portfolio, the G1-G10 long–short portfolio, IC/RankIC, factor effectiveness across holding horizons, and results stratified by Return20.

The default configuration runs from 2019-09-01 to 2024-09-01, with a 20-trading-day lookback, a 5-trading-day holding period, and 10 equally weighted groups.

```bash
cd QTrend20
python qtrend20_factor_v3.py
python qtrend20_factor_diagnostics_v3.py
```

## 4. Financial Factors

Directory: `财务因子/`.

The repository currently includes four factor types and their corresponding analysis scripts:

- `Sales_YoY_Q1_factor.py`: year-over-year revenue growth for an individual quarter.
- `Revenue_QoQ_weekly_factor.py`: quarter-over-quarter revenue growth for an individual quarter.
- `Net_Profit_YoY_weekly_factor.py`: year-over-year net profit growth for an individual quarter.
- `Net_Profit_QoQ_weekly_factor.py`: quarter-over-quarter net profit growth for an individual quarter.

The `*_analysis.py` scripts generate long–short NAV curves, IC/RankIC, portfolio-group returns, and diagnostics. The ROE > 5% and ROE > 10% settings mentioned in the work summary are screening variants of these financial factor experiments.

## 5. LLM Factor-Mining Agent

Directory: `LLM Agent架构/`.

- `LLM_A股量化因子挖掘Agent_完整项目架构.md`: the full architecture, including agent layers, task flows, and review mechanisms.
- `LLM驱动的A股量化因子挖掘Agent_精炼修订版.md`: a streamlined execution plan.

## Shared Backtesting Principles

- Generate signals after the close of day T and execute trades on T+1.
- Do not use data from after signal formation to calculate factors for day T.
- Apply upper/lower price-limit filters at the trade execution stage.
- Portfolios generally use equal weighting; consult each module's scripts for the exact methodology.
- Rerunning a script may raise a `PermissionError` if an output file is open in Excel or an image viewer.

Before running the scripts, check the data cache, MySQL connection, output directories, and local path configuration.
