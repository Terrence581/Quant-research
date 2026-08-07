# -*- coding: utf-8 -*-
"""
单季度净利润同比增长率(Net Profit YoY, single quarter)因子：周频 Rank IC 回测。

参照 财务因子/Sales_YoY_single_quarter_factor.py 的框架搭建，仅将标的指标由营业收入替换为
净利润（financial_data.metric_code = B002000000），时序规则完全复用本目录
《基本面因子回测时序规则.md》《营业收入同比增长率因子构造.md》的约定（对营收成立的拆解逻辑
同样适用于净利润，只是净利润可能为负，见下文口径说明）：

- 单季度拆解：Q1=一季报累计；Q2=半年报−一季报；Q3=三季报−半年报；Q4=年报−三季报。
- 单季度同比增长率 Qn_yoy =（本年 Qn−去年同季 Qn）/ |去年同季 Qn|（分母取绝对值，
  使增长方向在去年基数为负/亏损时仍保持一致）。
- 时点可得＝固定披露截止日（防未来函数，全市场同步切换）：Q1→当年4/30、
  半年报→当年8/31、Q3→当年10/31、年报→次年4/30。
- 春季窗口 [次年4/30, 次年8/31) 固定使用当年 Q1 填充（因子只保留季度 {1,2,3}，不含 Q4），
  与 Sales_YoY_single_quarter_factor.py 完全一致。
- 周频 Rank IC 回测：每周最后一个交易日为信号日 T，T+1 开盘执行，持有一周。
- 全 A 股票池（Markettype∈{1,4,16,32,64}），剔除 ST/PT、停牌；默认不设流通市值下限。

运行：  D:/Anaconda/python.exe 财务因子/Net_Profit_YoY_weekly_factor.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# ============================================================
# 1. 配置（对齐现有财务因子框架）
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "net_profit_yoy_weekly_output"
CACHE_DIR = PROJECT_DIR / "_local_data_cache"

MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "Shx20220717")
MYSQL_DATABASE = "1626_astock"

NET_PROFIT_METRIC_CODE = "B002000000"   # 净利润
FINANCIAL_STATEMENT_TYPE = "A"
FINANCIAL_IF_CORRECT = "0"

# ROE 筛选：profitability_data 中 净资产收益率(ROE)TTM 字段（CSMAR FI_T5，小数形式，
# 如 0.162 = 16.2%）。阈值以百分数给出，默认 10 即 ROE TTM > 10%。
ROE_TTM_FIELD = "F050504C"
ROE_FILTER_ENABLED = False        # False 则不做 ROE 筛选
ROE_MIN_PCT = 5.0               # ROE TTM 筛选阈值（百分数），仅保留严格大于该值的股票
DEFAULT_ROE_MIN_PCT = ROE_MIN_PCT  # 兼容旧引用

# 默认研究区间：与 Sales_YoY_Q1_factor.py 保持一致。
DEFAULT_START_DATE = "2014-01-03"
DEFAULT_END_DATE = "2024-09-30"

DEFAULT_GROUP_NUM = 10
DEFAULT_RETURN_COLUMN = "return_without_dividend"   # Dretnd
# 与 Q1 模板一致：默认不设流通市值下限。
DEFAULT_MIN_FLOAT_MARKET_VALUE = 5_000_000.0
DEFAULT_MARKET_TYPES = {1, 4, 16, 32, 64}
WEEKS_PER_YEAR = 52.0

# Trdsta 中 ST/*ST/PT 等状态编码；LimitStatus：1=涨停,-1=跌停。
ST_OR_PT_STATUS_VALUES = {2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16}
LIMIT_UP_OR_DOWN_VALUES = {-1, 1}
PRICE_COMPARE_TOLERANCE = 1e-6

STANDARD_PERIOD_MMDD = ["03-31", "06-30", "09-30", "12-31"]


# ============================================================
# 2. 数据读取（MySQL + 本地 parquet 缓存）
# ============================================================
def get_engine():
    url = (
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}?charset=utf8mb4"
    )
    return create_engine(url)


def load_financial_net_profit(start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    """读取净利润长表；多取两年历史用于单季度同比（同比需去年同季，同季需再上一季累计）。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"net_profit_{NET_PROFIT_METRIC_CODE}_{start_date}_{end_date}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    hist_start = (pd.Timestamp(start_date) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    sql = f"""
    SELECT Stkcd, Accper, DeclareDate, metric_value
    FROM financial_data
    WHERE metric_code = '{NET_PROFIT_METRIC_CODE}'
      AND Typrep = '{FINANCIAL_STATEMENT_TYPE}'
      AND IfCorrect = '{FINANCIAL_IF_CORRECT}'
      AND Accper BETWEEN '{hist_start}' AND '{end_date}'
      AND DATE_FORMAT(Accper, '%%m-%%d') <> '01-01'
    """
    engine = get_engine()
    df = pd.read_sql(sql, engine)
    engine.dispose()
    df.to_parquet(cache, index=False)
    return df


def load_market(start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    """读取回测所需的行情列。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"net_profit_market_{start_date}_{end_date}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    sql = f"""
    SELECT Stkcd, Trddt, Opnprc, Clsprc, Dnshrtrd, Dnvaltrd, Dsmvosd,
           Dretnd, Dretwd, Markettype, Trdsta, LimitUp, LimitDown, LimitStatus
    FROM all_market_data
    WHERE Trddt BETWEEN '{start_date}' AND '{end_date}'
    """
    engine = get_engine()
    df = pd.read_sql(sql, engine)
    engine.dispose()
    df.to_parquet(cache, index=False)
    return df


def load_roe_ttm(start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    """读取盈利能力表的 净资产收益率(ROE)TTM（F050504C）；多取两年历史覆盖回测期初。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"net_profit_yoy_roe_ttm_{ROE_TTM_FIELD}_{start_date}_{end_date}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    hist_start = (pd.Timestamp(start_date) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    sql = f"""
    SELECT Stkcd, Accper, {ROE_TTM_FIELD} AS roe_ttm
    FROM profitability_data
    WHERE Typrep = '{FINANCIAL_STATEMENT_TYPE}'
      AND Accper BETWEEN '{hist_start}' AND '{end_date}'
      AND DATE_FORMAT(Accper, '%%m-%%d') <> '01-01'
    """
    engine = get_engine()
    df = pd.read_sql(sql, engine)
    engine.dispose()
    df.to_parquet(cache, index=False)
    return df


# ============================================================
# 3. 单季度净利润同比增长率因子
# ============================================================
def parse_latest_declare_date(value) -> pd.Timestamp:
    """DeclareDate 多日期取最晚，避免提前使用更正后数据。"""
    if pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    if not text or text == r"\N":
        return pd.NaT
    parts = [p.strip() for p in text.split(",") if p.strip() and p.strip() != r"\N"]
    if not parts:
        return pd.NaT
    dates = pd.to_datetime(parts, errors="coerce")
    valid = [d for d in dates if pd.notna(d)]
    return max(valid) if valid else pd.NaT


def regulatory_deadline(report_date: pd.Timestamp) -> pd.Timestamp:
    """缺 DeclareDate 时的监管披露截止日（点时可得回退，防未来函数）。"""
    if pd.isna(report_date):
        return pd.NaT
    y, mmdd = report_date.year, report_date.strftime("%m-%d")
    if mmdd == "03-31":
        return pd.Timestamp(y, 4, 30)
    if mmdd == "06-30":
        return pd.Timestamp(y, 8, 31)
    if mmdd == "09-30":
        return pd.Timestamp(y, 10, 31)
    if mmdd == "12-31":
        return pd.Timestamp(y + 1, 4, 30)
    return report_date


def prepare_net_profit_yoy_single_quarter(
    raw_financial: pd.DataFrame,
    availability: str = "deadline",
) -> pd.DataFrame:
    """从 financial_data 计算单季度净利润同比增长率。

    availability="deadline"：因子在**固定披露截止日**后才可用（Q1→当年4/30、
        半年报→当年8/31、Q3→当年10/31、年报→次年4/30），与
        《营业收入同比增长率因子构造.md》一致，全市场同步切换；
        "declare_date"：改用实际 DeclareDate（多日期取最晚，缺失回退截止日）。
    净利润可能为负（亏损），增长率使用
    （本年单季−去年同季单季）/ |去年同季单季|；分母为 0 的记录剔除。
    春季窗口 [次年4/30,次年8/31) 固定使用当年 Q1，因子仅保留 Q1、Q2、Q3。
    """
    fin = raw_financial.copy()
    fin["stock_code"] = fin["Stkcd"].astype("string").str.strip().str.zfill(6)
    fin["report_date"] = pd.to_datetime(fin["Accper"], errors="coerce")
    fin["net_profit_cum"] = pd.to_numeric(fin["metric_value"], errors="coerce")
    fin = fin.loc[fin["report_date"].dt.strftime("%m-%d").isin(STANDARD_PERIOD_MMDD)].copy()
    fin = fin.dropna(subset=["stock_code", "report_date", "net_profit_cum"])

    # 时点可得日。默认按固定披露截止日（文档口径），可选实际 DeclareDate。
    if availability == "declare_date":
        declare = fin["DeclareDate"].apply(parse_latest_declare_date)
        fin["available_date"] = declare.where(
            declare.notna(), fin["report_date"].apply(regulatory_deadline)
        )
    else:
        fin["available_date"] = fin["report_date"].apply(regulatory_deadline)
    fin = fin.sort_values(["stock_code", "report_date", "available_date"])
    fin = fin.drop_duplicates(["stock_code", "report_date"], keep="last")

    # 单季度净利润 = 本期累计 − 上一季度累计（Q1 直接取累计）。
    fin["prev_q_end"] = (
        (fin["report_date"].dt.to_period("Q") - 1).dt.to_timestamp(how="end").dt.normalize()
    )
    prev_cum = fin[["stock_code", "report_date", "net_profit_cum"]].rename(
        columns={"report_date": "prev_q_end", "net_profit_cum": "prev_cum"}
    )
    fin = fin.merge(prev_cum, on=["stock_code", "prev_q_end"], how="left")
    is_q1 = fin["report_date"].dt.quarter.eq(1)
    fin["net_profit_sq"] = fin["net_profit_cum"].where(is_q1, fin["net_profit_cum"] - fin["prev_cum"])

    # 单季度同比增长率 =（本年单季−去年同季单季）/ |去年同季单季|。
    fin["prior_year_end"] = fin["report_date"] - pd.DateOffset(years=1)
    prev_year = fin[["stock_code", "report_date", "net_profit_sq"]].rename(
        columns={"report_date": "prior_year_end", "net_profit_sq": "prev_year_sq"}
    )
    fin = fin.merge(prev_year, on=["stock_code", "prior_year_end"], how="left")

    valid = (
        fin["net_profit_sq"].notna()
        & fin["prev_year_sq"].notna()
        & fin["prev_year_sq"].ne(0)
    )
    fin["net_profit_yoy_sq"] = np.nan
    fin.loc[valid, "net_profit_yoy_sq"] = (
        (fin.loc[valid, "net_profit_sq"] - fin.loc[valid, "prev_year_sq"])
        / fin.loc[valid, "prev_year_sq"].abs()
    )
    factor = fin.loc[valid, [
        "stock_code", "report_date", "available_date",
        "net_profit_cum", "net_profit_sq", "prev_year_sq", "net_profit_yoy_sq",
    ]].copy()
    # 单季度拆解需相邻季度，故在算完同比后仅保留 next_q1_fill 使用的 Q1、Q2、Q3。
    factor = factor.loc[factor["report_date"].dt.quarter.isin((1, 2, 3))].copy()
    factor["report_quarter"] = factor["report_date"].dt.quarter
    factor["available_date"] = pd.to_datetime(factor["available_date"])
    return factor.sort_values(["stock_code", "available_date", "report_date"]).reset_index(drop=True)


def prepare_roe_ttm(raw_roe: pd.DataFrame) -> pd.DataFrame:
    """清洗 净资产收益率(ROE)TTM 长表，给出点时可得日（固定披露截止日，与因子口径一致）。

    profitability_data 无 DeclareDate 列，统一用 regulatory_deadline 作为可得日，
    避免未来函数。roe_ttm 为小数形式（0.162 = 16.2%）。
    """
    roe = raw_roe.copy()
    roe["stock_code"] = roe["Stkcd"].astype("string").str.strip().str.zfill(6)
    roe["report_date"] = pd.to_datetime(roe["Accper"], errors="coerce")
    roe["roe_ttm"] = pd.to_numeric(roe["roe_ttm"], errors="coerce")
    roe = roe.loc[roe["report_date"].dt.strftime("%m-%d").isin(STANDARD_PERIOD_MMDD)].copy()
    roe = roe.dropna(subset=["stock_code", "report_date", "roe_ttm"])
    roe["available_date"] = roe["report_date"].apply(regulatory_deadline)
    roe = roe.sort_values(["stock_code", "report_date"])
    roe = roe.drop_duplicates(["stock_code", "report_date"], keep="last")
    return (roe[["stock_code", "report_date", "available_date", "roe_ttm"]]
            .sort_values(["stock_code", "available_date", "report_date"])
            .reset_index(drop=True))


def attach_roe_pit(signal_factor: pd.DataFrame, roe: pd.DataFrame) -> pd.DataFrame:
    """对每个 (股票, 信号日) 取 available_date <= 信号日 的最新 ROE TTM。

    双键排序 ["available_date", "report_date"]：每年4/30年报与一季报同日可得，
    平局必须确定性地取最新报告期（一季报），结果与排序算法稳定性无关。
    """
    left = signal_factor.sort_values("signal_date").copy()
    right = roe.sort_values(["available_date", "report_date"])[
        ["stock_code", "available_date", "roe_ttm"]
    ].copy()
    merged = pd.merge_asof(
        left, right,
        left_on="signal_date", right_on="available_date",
        by="stock_code", direction="backward",
    )
    return merged.drop(columns=["available_date"], errors="ignore")


def apply_roe_filter(signal_factor: pd.DataFrame, roe: pd.DataFrame,
                     roe_min_pct: float) -> pd.DataFrame:
    """ROE TTM 筛选：仅保留点时可得 ROE TTM > roe_min_pct（百分数）的记录。

    无 ROE 数据（NaN）的记录视为不满足条件，一并剔除（"只保留大于阈值"的严格口径）。
    """
    merged = attach_roe_pit(signal_factor, roe)
    return merged.loc[merged["roe_ttm"] > roe_min_pct / 100.0].copy()


# ============================================================
# 4. 行情清洗与股票池
# ============================================================


def clean_market_data(raw: pd.DataFrame, market_types: set[int],
                      min_float: float, return_column: str) -> pd.DataFrame:
    """标准化 + 股票池过滤（非A股、ST/PT、停牌、可选市值下限）。保留涨跌停价用于 T+1 过滤。"""
    d = raw.rename(columns={
        "Stkcd": "stock_code", "Trddt": "trade_date", "Opnprc": "open_price",
        "Clsprc": "close_price", "Dnshrtrd": "volume", "Dnvaltrd": "amount",
        "Dsmvosd": "float_market_value", "Dretnd": "return_without_dividend",
        "Dretwd": "return_with_dividend", "Markettype": "market_type",
        "Trdsta": "trade_status", "LimitUp": "limit_up_price",
        "LimitDown": "limit_down_price", "LimitStatus": "limit_status",
    }).copy()
    d["stock_code"] = d["stock_code"].astype("string").str.strip().str.zfill(6)
    d["trade_date"] = pd.to_datetime(d["trade_date"], errors="coerce")
    for col in ["open_price", "close_price", "volume", "amount", "float_market_value",
                "return_without_dividend", "return_with_dividend", "market_type",
                "trade_status", "limit_up_price", "limit_down_price", "limit_status"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    required_columns = [
        "stock_code", "trade_date", "open_price", "close_price", "volume", "amount",
        "float_market_value", "market_type", "trade_status", "limit_up_price",
        "limit_down_price", "limit_status", return_column,
    ]
    missing_columns = [column for column in required_columns if column not in d.columns]
    if missing_columns:
        raise KeyError(f"原始行情缺少清洗必需字段：{missing_columns}")
    d = d.dropna(subset=required_columns).copy()
    d = d.sort_values(["stock_code", "trade_date"]).drop_duplicates(
        ["stock_code", "trade_date"], keep="last")

    is_a = d["market_type"].isin(market_types)
    is_st = d["trade_status"].isin(ST_OR_PT_STATUS_VALUES)
    is_suspended = (
        (d["volume"].notna() & d["volume"].le(0))
        | (d["amount"].notna() & d["amount"].le(0))
    )
    keep = is_a & ~is_st & ~is_suspended
    if min_float > 0:
        keep = keep & ~d["float_market_value"].lt(min_float)
    return d.loc[keep].reset_index(drop=True)


# ============================================================
# 5. 周频日历、点时可得因子、前瞻周收益
# ============================================================
def build_weekly_signals(trade_dates: pd.Series) -> pd.DataFrame:
    """每周最后一个交易日为信号日；给出下一交易日（开仓日）与下一信号日。"""
    dates = pd.Series(pd.to_datetime(trade_dates.unique())).sort_values().reset_index(drop=True)
    cal = pd.DataFrame({"trade_date": dates})
    iso = cal["trade_date"].dt.isocalendar()
    cal["yw"] = iso["year"].astype(int) * 100 + iso["week"].astype(int)
    signal = cal.groupby("yw")["trade_date"].max().sort_values().reset_index(drop=True)
    sig = pd.DataFrame({"signal_date": signal})
    # 下一交易日 = 开仓日。
    next_map = dict(zip(dates, dates.shift(-1)))
    sig["entry_date"] = sig["signal_date"].map(next_map)
    sig["next_signal_date"] = sig["signal_date"].shift(-1)
    return sig


def attach_factor_pit(universe_at_signal: pd.DataFrame, factor: pd.DataFrame) -> pd.DataFrame:
    """对每个 (股票, 信号日) 取 available_date <= 信号日 的最新单季度同比增长率因子。"""
    left = universe_at_signal.sort_values("signal_date").copy()
    # 双键排序：同一 available_date 出现多条时保证 merge_asof 平局取 report_date 最新的一条
    right = factor.sort_values(["available_date", "report_date"])[
        ["stock_code", "available_date", "report_date", "net_profit_yoy_sq"]
    ].copy()
    merged = pd.merge_asof(
        left, right,
        left_on="signal_date", right_on="available_date",
        by="stock_code", direction="backward",
    )
    return merged.dropna(subset=["net_profit_yoy_sq"]).copy()


def compute_forward_week_return(market_std: pd.DataFrame, signals: pd.DataFrame,
                                return_column: str) -> pd.DataFrame:
    """每只股票的前瞻一周收益：把每个交易日收益归入其之前最近的信号日，按周复利。"""
    sig_dates = signals[["signal_date"]].dropna().sort_values("signal_date")
    ret = market_std[["stock_code", "trade_date", return_column]].dropna(
        subset=[return_column]).sort_values("trade_date").copy()
    # 交易日 d 的收益归入 严格小于 d 的最近信号日（即持有区间 (T, 下一T] 内的收益）。
    ret = pd.merge_asof(
        ret, sig_dates, left_on="trade_date", right_on="signal_date",
        direction="backward", allow_exact_matches=False,
    )
    ret = ret.dropna(subset=["signal_date"])
    grp = ret.groupby(["stock_code", "signal_date"])[return_column]
    fwd = (grp.apply(lambda s: np.prod(1.0 + s.values) - 1.0)
           .rename("forward_week_return").reset_index())
    return fwd


# ============================================================
# 6. 截面处理：3sigma 去极值、zscore、分组
# ============================================================
def process_cross_section(signal_factor: pd.DataFrame, group_num: int) -> pd.DataFrame:
    """按信号日截面：3σ 剔除极值 → zscore → 降序分 group_num 组（第1组=因子最高=多头）。"""
    df = signal_factor.dropna(subset=["net_profit_yoy_sq"]).copy()
    g = df.groupby("signal_date")["net_profit_yoy_sq"]
    mean = g.transform("mean")
    std = g.transform(lambda s: s.std(ddof=0))
    lower, upper = mean - 3.0 * std, mean + 3.0 * std
    df = df.loc[(df["net_profit_yoy_sq"] >= lower) & (df["net_profit_yoy_sq"] <= upper)].copy()

    g2 = df.groupby("signal_date")["net_profit_yoy_sq"]
    mean2 = g2.transform("mean")
    std2 = g2.transform(lambda s: s.std(ddof=0))
    df["zscore"] = np.where(std2.gt(0), (df["net_profit_yoy_sq"] - mean2) / std2, np.nan)
    df = df.dropna(subset=["zscore"]).copy()

    df = df.sort_values(["signal_date", "zscore", "stock_code"],
                        ascending=[True, False, True]).reset_index(drop=True)
    df["rank_desc"] = df.groupby("signal_date").cumcount() + 1
    df["n"] = df.groupby("signal_date")["stock_code"].transform("count")
    df["quantile_group"] = (((df["rank_desc"] - 1) * group_num) // df["n"] + 1).clip(1, group_num)
    return df


def apply_t1_tradability(grouped: pd.DataFrame, market_std: pd.DataFrame,
                         group_num: int) -> pd.DataFrame:
    """T+1 开盘可交易性过滤：开仓日开盘触及涨跌停或缺记录的多空两端股票剔除。"""
    entry = market_std[["stock_code", "trade_date", "open_price",
                        "limit_up_price", "limit_down_price"]].rename(
        columns={"trade_date": "entry_date", "open_price": "entry_open",
                 "limit_up_price": "entry_limit_up", "limit_down_price": "entry_limit_down"})
    g = grouped.merge(entry, on=["stock_code", "entry_date"], how="left")
    open_lu = (g["entry_open"] - g["entry_limit_up"]).abs().le(PRICE_COMPARE_TOLERANCE)
    open_ld = (g["entry_open"] - g["entry_limit_down"]).abs().le(PRICE_COMPARE_TOLERANCE)
    is_limit = (open_lu | open_ld).fillna(False)
    missing = (
        g[["entry_open", "entry_limit_up", "entry_limit_down"]].isna().any(axis=1)
        | g["forward_week_return"].isna()
    )
    is_ls_candidate = g["quantile_group"].isin([1, group_num])
    g["is_entry_open_limit"] = is_limit
    g["tradable_at_entry"] = (~is_limit & ~missing).fillna(False)
    g["tradable_ls"] = (is_ls_candidate & g["tradable_at_entry"]).fillna(False)
    return g


# ============================================================
# 7. 多空组合、周频 Rank IC、绩效
# ============================================================
def weekly_long_short(graded: pd.DataFrame, group_num: int) -> pd.DataFrame:
    """周频多空：多头=第1组，空头=第group_num组，等权，逐周复利。仅用可交易股票。"""
    cand = graded.loc[graded["tradable_ls"] & graded["quantile_group"].isin([1, group_num])].copy()
    high = (cand.loc[cand["quantile_group"].eq(1)]
            .groupby("signal_date")["forward_week_return"].mean().rename("high_return"))
    low = (cand.loc[cand["quantile_group"].eq(group_num)]
           .groupby("signal_date")["forward_week_return"].mean().rename("low_return"))
    ls = pd.concat([high, low], axis=1).dropna().reset_index()
    ls["long_short_return"] = ls["high_return"] - ls["low_return"]
    ls = ls.sort_values("signal_date").reset_index(drop=True)
    ls["long_short_nav"] = (1.0 + ls["long_short_return"]).cumprod()
    ls["high_nav"] = (1.0 + ls["high_return"]).cumprod()
    ls["low_nav"] = (1.0 + ls["low_return"]).cumprod()
    return ls


def quantile_group_returns(graded: pd.DataFrame, group_num: int) -> pd.DataFrame:
    """各分组周频等权收益与净值（用于分组净值图/单调性检查，观察口径不做交易过滤）。"""
    obs = graded.loc[graded["tradable_at_entry"]].dropna(subset=["forward_week_return"]).copy()
    gr = (obs.groupby(["signal_date", "quantile_group"])["forward_week_return"]
          .mean().reset_index())
    piv = gr.pivot(index="signal_date", columns="quantile_group",
                   values="forward_week_return").sort_index()
    nav = (1.0 + piv).cumprod()
    nav.columns = [f"group_{int(c)}_nav" for c in nav.columns]
    piv.columns = [f"group_{int(c)}_return" for c in piv.columns]
    return pd.concat([piv, nav], axis=1).reset_index()


def weekly_rank_ic(graded: pd.DataFrame) -> pd.DataFrame:
    """周频 Rank IC = 截面 zscore 与前瞻周收益的 Spearman（观察口径，不做交易过滤）。"""
    valid = graded.dropna(subset=["zscore", "forward_week_return"]).copy()
    rows = []
    for sig, one in valid.groupby("signal_date"):
        if len(one) < 5:
            continue
        fr = one["zscore"].rank(method="average")
        rr = one["forward_week_return"].rank(method="average")
        rows.append({
            "signal_date": sig,
            "ic": one["zscore"].corr(one["forward_week_return"], method="pearson"),
            "rank_ic": fr.corr(rr, method="pearson"),
            "n": len(one),
        })
    ic = pd.DataFrame(rows).sort_values("signal_date").reset_index(drop=True)
    if not ic.empty:
        ic["cumulative_rank_ic"] = ic["rank_ic"].cumsum()
        ic["cumulative_ic"] = ic["ic"].cumsum()
    return ic


def drawdown_with_dates(nav: pd.Series, dates: pd.Series):
    """最大回撤及其峰/谷日期。"""
    nav = pd.Series(pd.to_numeric(pd.Series(nav), errors="coerce").to_numpy())
    dates = pd.Series(pd.to_datetime(pd.Series(dates)).to_numpy())
    peak = nav.cummax()
    dd = nav / peak - 1.0
    trough_i = int(dd.idxmin())
    max_dd = float(dd.min())
    peak_i = int(nav.iloc[:trough_i + 1].idxmax()) if trough_i >= 0 else 0
    return max_dd, dates.iloc[peak_i], dates.iloc[trough_i]


def performance_metrics(ls: pd.DataFrame, name: str) -> dict:
    """周频组合绩效：总收益、年化、波动、夏普、最大回撤及起止、月度胜率。"""
    r = pd.to_numeric(ls["long_short_return"], errors="coerce").dropna()
    nav = pd.to_numeric(ls["long_short_nav"], errors="coerce").dropna()
    dates = pd.to_datetime(ls.loc[r.index, "signal_date"])
    n = len(r)
    if n == 0:
        return {"portfolio": name, "n": 0}
    total_return = float(nav.iloc[-1] - 1.0)
    annual_return = float(nav.iloc[-1] ** (WEEKS_PER_YEAR / n) - 1.0)
    annual_vol = float(r.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR))
    sharpe = annual_return / annual_vol if annual_vol else np.nan
    max_dd, dd_start, dd_end = drawdown_with_dates(nav.values, dates.values)
    # 月度胜率：把周频多空收益按自然月复利，统计正收益月占比。
    monthly = (pd.DataFrame({"date": dates.values, "r": r.values})
               .set_index("date")["r"]
               .resample("ME").apply(lambda s: np.prod(1.0 + s.values) - 1.0))
    monthly = monthly.dropna()
    monthly_win = float((monthly > 0).mean()) if len(monthly) else np.nan
    return {
        "portfolio": name, "n_weeks": n,
        "total_return": total_return, "annual_return": annual_return,
        "annual_volatility": annual_vol, "sharpe": sharpe,
        "max_drawdown": max_dd,
        "max_dd_start": pd.Timestamp(dd_start).strftime("%Y-%m-%d"),
        "max_dd_end": pd.Timestamp(dd_end).strftime("%Y-%m-%d"),
        "monthly_win_rate": monthly_win,
    }


def performance_metrics_for_period(
    ls: pd.DataFrame,
    name: str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> dict:
    """按给定区间将净值重置为 1，再使用统一绩效函数计算多空表现。"""
    period = ls.copy()
    if start is not None:
        period = period.loc[period["signal_date"] >= start]
    if end is not None:
        period = period.loc[period["signal_date"] <= end]
    period = period.sort_values("signal_date").copy()
    period["long_short_nav"] = (1.0 + period["long_short_return"]).cumprod()
    return performance_metrics(period, name)


def format_performance_summary(label: str, metrics: dict) -> str:
    """格式化全区间多空绩效。"""
    if not metrics.get("n_weeks"):
        return f"  {label}：无有效周收益记录"
    return (
        f"  {label}：周数 {metrics['n_weeks']}，总收益 {metrics['total_return']*100:.2f}%，"
        f"年化收益 {metrics['annual_return']*100:.2f}%，"
        f"年化波动率 {metrics['annual_volatility']*100:.2f}%，"
        f"Sharpe {metrics['sharpe']:.2f}，最大回撤 {metrics['max_drawdown']*100:.2f}%"
        f"（{metrics['max_dd_start']}~{metrics['max_dd_end']}），"
        f"月度胜率 {metrics['monthly_win_rate']*100:.2f}%"
    )


def mean_rank_ic(ic: pd.DataFrame, start=None, end=None) -> float:
    s = ic
    if start is not None:
        s = s.loc[s["signal_date"] >= start]
    if end is not None:
        s = s.loc[s["signal_date"] <= end]
    return float(s["rank_ic"].mean()) if len(s) else np.nan


# ============================================================
# 8. 主流程
# ============================================================
# 春季窗口 [次年4/30, 次年8/31) 仅使用当年 Q1。
OUTPUT_VARIANT_NAME = "next_q1_fill"
OUTPUT_VARIANT_DESCRIPTION = "次年Q1填充法（春季用当年Q1）"


def backtest_variant(factor: pd.DataFrame, universe: pd.DataFrame, fwd: pd.DataFrame,
                     market_std: pd.DataFrame, signals: pd.DataFrame, group_num: int,
                     out_dir: Path, roe: pd.DataFrame | None = None,
                     roe_min_pct: float = DEFAULT_ROE_MIN_PCT) -> dict:
    """给定单季度净利润同比增长率因子，做周频 10 组多空 + RankIC + 绩效，落盘到 out_dir。

    roe 非 None 时先做 ROE TTM > roe_min_pct 的点时可得筛选，再进入截面处理。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    signal_factor = attach_factor_pit(universe, factor)
    n_before_roe = len(signal_factor)
    if roe is not None:
        signal_factor = apply_roe_filter(signal_factor, roe, roe_min_pct)
        print(f"      ROE TTM > {roe_min_pct:g}% 筛选：{n_before_roe} → {len(signal_factor)} "
              f"条 (股票, 信号日) 记录")
    else:
        signal_factor["roe_ttm"] = np.nan
    signal_factor = signal_factor.merge(fwd, on=["stock_code", "signal_date"], how="left")

    graded = process_cross_section(signal_factor, group_num)
    graded = graded.merge(
        signal_factor[["stock_code", "signal_date", "entry_date", "forward_week_return"]]
        .drop_duplicates(["stock_code", "signal_date"]),
        on=["stock_code", "signal_date"], how="left", suffixes=("", "_dup"))
    if "forward_week_return_dup" in graded.columns:
        graded["forward_week_return"] = graded["forward_week_return"].fillna(graded["forward_week_return_dup"])
        graded = graded.drop(columns=[c for c in graded.columns if c.endswith("_dup")])
    graded = apply_t1_tradability(graded, market_std, group_num)

    ls = weekly_long_short(graded, group_num)
    groups = quantile_group_returns(graded, group_num)
    ic = weekly_rank_ic(graded)

    factor.to_csv(out_dir / "01_net_profit_yoy_weekly_factor.csv", index=False, encoding="utf-8-sig")
    graded[["signal_date", "stock_code", "net_profit_yoy_sq", "roe_ttm", "zscore", "rank_desc",
            "quantile_group", "entry_date", "forward_week_return", "is_entry_open_limit",
            "tradable_at_entry", "tradable_ls"]].to_csv(
        out_dir / "02_weekly_graded_factor.csv", index=False, encoding="utf-8-sig")
    ls.to_csv(out_dir / "03_weekly_long_short.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(out_dir / "04_weekly_group_nav.csv", index=False, encoding="utf-8-sig")
    ic.to_csv(out_dir / "05_weekly_rank_ic.csv", index=False, encoding="utf-8-sig")

    perf_full = performance_metrics_for_period(ls, "long_short_full")
    pd.DataFrame([perf_full]).to_csv(
        out_dir / "06_performance_metrics.csv", index=False, encoding="utf-8-sig")

    return {
        "ric_full": mean_rank_ic(ic),
        "perf_full": perf_full,
        "factor_rows": len(factor), "n_stocks": factor["stock_code"].nunique() if len(factor) else 0,
    }


def run(start_date: str, end_date: str, group_num: int, return_column: str,
        market_types: set[int], min_float: float, availability: str, refresh: bool,
        roe_min_pct: float | None = DEFAULT_ROE_MIN_PCT) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/4] 读取财务数据（净利润）...")
    raw_fin = load_financial_net_profit(start_date, end_date, refresh)

    roe = None
    if roe_min_pct is not None:
        print(f"      读取盈利能力数据（净资产收益率ROE TTM，筛选阈值 > {roe_min_pct:g}%）...")
        roe = prepare_roe_ttm(load_roe_ttm(start_date, end_date, refresh))
        print(f"      ROE TTM 记录数：{len(roe)}，股票 {roe['stock_code'].nunique()} 只")

    print("[2/4] 读取并清洗行情数据 ...")
    raw_mkt = load_market(start_date, end_date, refresh)
    market_std = clean_market_data(raw_mkt, market_types, min_float, return_column)
    print(f"      清洗后行情记录数：{len(market_std)}，交易日 "
          f"{market_std['trade_date'].min().date()} ~ {market_std['trade_date'].max().date()}")

    print("[3/4] 构建周频信号日历与前瞻一周收益 ...")
    signals = build_weekly_signals(market_std["trade_date"])
    signals = signals.loc[signals["entry_date"].notna() & signals["next_signal_date"].notna()].copy()
    universe = market_std[["stock_code", "trade_date"]].rename(columns={"trade_date": "signal_date"})
    universe = universe.merge(signals[["signal_date", "entry_date"]], on="signal_date", how="inner")
    fwd = compute_forward_week_return(market_std, signals, return_column)
    print(f"      周信号数：{signals['signal_date'].nunique()}；时点可得口径：{availability}")

    print("[4/4] 使用 next_q1_fill 做周频 10 组多空 / RankIC / 绩效 ...")
    factor = prepare_net_profit_yoy_single_quarter(raw_fin, availability=availability)
    out_dir = OUTPUT_DIR / f"{OUTPUT_VARIANT_NAME}_roe_gt_{roe_min_pct:g}" if roe is not None else OUTPUT_DIR / OUTPUT_VARIANT_NAME
    summary = backtest_variant(
        factor, universe, fwd, market_std, signals, group_num, out_dir,
        roe=roe, roe_min_pct=roe_min_pct if roe_min_pct is not None else DEFAULT_ROE_MIN_PCT,
    )
    print(f"      [{out_dir.name}] 因子记录 {summary['factor_rows']}，"
          f"股票 {summary['n_stocks']} → {out_dir}")

    print("\n================= 单季度净利润同比增长率因子(Net_Profit_YoY)周频回测结果摘要 =================")
    pf = summary["perf_full"]
    print(f"\n【{out_dir.name}】{OUTPUT_VARIANT_DESCRIPTION}"
          + (f" + ROE TTM > {roe_min_pct:g}% 筛选" if roe is not None else ""))
    print(f"  周频 Rank IC 均值：{summary['ric_full']*100:.2f}%")
    print(format_performance_summary("多空(全区间)", pf))
    print(f"\n输出根目录：{OUTPUT_DIR}")


def parse_args():
    p = argparse.ArgumentParser(description="单季度净利润同比增长率因子(Net_Profit_YoY)周频回测（文档口径）")
    p.add_argument("--start", default=DEFAULT_START_DATE)
    p.add_argument("--end", default=DEFAULT_END_DATE)
    p.add_argument("--group-num", type=int, default=DEFAULT_GROUP_NUM)
    p.add_argument("--return-column", default=DEFAULT_RETURN_COLUMN,
                   choices=["return_without_dividend", "return_with_dividend"])
    p.add_argument("--min-float", type=float, default=DEFAULT_MIN_FLOAT_MARKET_VALUE)
    p.add_argument("--availability", default="deadline", choices=["deadline", "declare_date"],
                   help="时点可得口径：deadline=固定披露截止日(文档口径)；declare_date=实际披露日")
    p.add_argument("--roe-min", type=float, default=None,
                   help="ROE TTM 筛选阈值（百分数），默认取脚本参数区 ROE_MIN_PCT（当前 "
                        f"{ROE_MIN_PCT:g}）；传入则临时覆盖参数区设置")
    p.add_argument("--no-roe-filter", action="store_true",
                   help="临时关闭 ROE TTM 筛选（等效于参数区 ROE_FILTER_ENABLED = False）")
    p.add_argument("--refresh", action="store_true", help="强制重新查询 MySQL")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # 参数区为默认来源：ROE_FILTER_ENABLED 控制是否筛选，ROE_MIN_PCT 为阈值；
    # 命令行 --roe-min / --no-roe-filter 仅作临时覆盖。
    roe_min_pct = None
    if ROE_FILTER_ENABLED and not args.no_roe_filter:
        roe_min_pct = args.roe_min if args.roe_min is not None else ROE_MIN_PCT
    run(args.start, args.end, args.group_num, args.return_column,
        DEFAULT_MARKET_TYPES, args.min_float, args.availability, args.refresh,
        roe_min_pct=roe_min_pct)
