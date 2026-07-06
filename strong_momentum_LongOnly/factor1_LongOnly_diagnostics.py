from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. 基础配置
# ============================================================


def load_long_only_config():
    """Read defaults from factor1_LongOnly.py when it is available."""

    # 诊断脚本优先复用主策略文件里的默认参数，避免两个文件参数漂移。
    candidates = [
        Path(__file__).with_name("factor1_LongOnly.py"),
        Path(r"D:\Desktop\factor1_LongOnly.py"),
    ]
    for module_path in candidates:
        if not module_path.exists():
            continue
        spec = importlib.util.spec_from_file_location("factor1_long_only_config", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return None


LONG_ONLY_CONFIG = load_long_only_config()
DEFAULT_OUTPUT_ROOT = getattr(
    LONG_ONLY_CONFIG,
    "DEFAULT_OUTPUT_ROOT",
    Path(r"D:\Desktop\CINDA qr\factor1_strong_momentum\factor1_LongOnly_output"),
)
DEFAULT_LOOKBACK_DAYS = getattr(LONG_ONLY_CONFIG, "DEFAULT_LOOKBACK_DAYS", 147)
DEFAULT_MAX_PEAK_AGE_DAYS = getattr(LONG_ONLY_CONFIG, "DEFAULT_MAX_PEAK_AGE_DAYS", 42)
DEFAULT_HOLDING_DAYS = getattr(LONG_ONLY_CONFIG, "DEFAULT_HOLDING_DAYS", 21)
DEFAULT_MARKET_TYPES = getattr(LONG_ONLY_CONFIG, "DEFAULT_MARKET_TYPES", {1, 4})
TRADING_DAYS_PER_YEAR = getattr(LONG_ONLY_CONFIG, "TRADING_DAYS_PER_YEAR", 252)


def market_types_to_tag(market_types: set[int]) -> str:
    return "-".join(str(item) for item in sorted(market_types)) if market_types else "na"


def build_default_input_dir(output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    # 默认诊断目录和主策略输出目录保持一致：longonly_lb{lookback}_hd{holding}_mkt{market_types}。
    if LONG_ONLY_CONFIG is not None and hasattr(LONG_ONLY_CONFIG, "build_long_only_experiment_folder_name"):
        folder_name = LONG_ONLY_CONFIG.build_long_only_experiment_folder_name(
            DEFAULT_LOOKBACK_DAYS,
            DEFAULT_HOLDING_DAYS,
            DEFAULT_MARKET_TYPES,
        )
    else:
        folder_name = f"longonly_lb{DEFAULT_LOOKBACK_DAYS}_hd{DEFAULT_HOLDING_DAYS}_mkt{market_types_to_tag(DEFAULT_MARKET_TYPES)}"
    return output_root / folder_name


DEFAULT_INPUT_DIR = build_default_input_dir(DEFAULT_OUTPUT_ROOT)
DEFAULT_TRADE_DETAIL_DATE = "2019-11-22"  # Change this to inspect another execution date, or override with --trade-date.
FACTOR_SORT_COLUMN = "momentum_raw"


DATE_COLUMNS = [
    "trade_date",
    "next_trade_date",
    "holding_start_trade_date",
    "holding_end_trade_date",
    "holding_start_trade_date_min",
    "holding_start_trade_date_max",
    "holding_end_trade_date_min",
    "holding_end_trade_date_max",
    "active_signal_trade_date_min",
    "active_signal_trade_date_max",
    "momentum_start_date",
    "momentum_end_date",
    "lookback_window_start",
    "pre_peak_low_trade_date",
    "pre_peak_zscore_window_start",
    "peak_trade_date",
    "post_peak_low_trade_date",
    "max_peak_age_start",
]

BOOLEAN_COLUMNS = [
    "has_pre_peak_low",
    "signal_is_post_peak_low",
    "strong_rise_filter",
    "peak_age_filter",
    "drawdown_filter",
    "post_peak_low_filter",
    "pre_peak_zscore_filter",
    "passes_strong_momentum_filters",
    "has_complete_holding_return",
    "is_next_open_limit_up",
    "is_next_open_limit_down",
    "is_next_open_one_word_limit",
    "is_next_close_limit_up",
    "is_next_close_limit_down",
    "is_next_close_limit",
    "is_tradable_next_open",
    "is_tradable_next_close",
    "has_raw_signal",
    "has_tradable_signal",
    "has_active_position",
]

NUMERIC_COLUMNS = [
    "signal_close_price",
    "return_without_dividend",
    "momentum_start_close_price",
    "momentum_end_close_price",
    "pre_peak_low_close_price",
    "peak_close_price",
    "post_peak_low_close_price",
    "full_history_valid_days",
    "lookback_valid_days",
    "lookback_limit_days_count",
    "low_to_peak_return",
    "drawdown_from_peak_to_signal",
    "pre_peak_zscore_valid_days",
    "pre_peak_max_abs_zscore",
    "zscore_window",
    "zscore_limit",
    "zscore_ddof",
    "momentum_raw",
    "strong_momentum_raw",
    "holding_entry_open_price",
    "holding_entry_close_price",
    "holding_entry_price",
    "holding_exit_close_price",
    "holding_end_close_price",
    "holding_terminal_offset_days",
    "next_period_return_before_trade_filter",
    "future_return_valid_days",
    "next_open_price",
    "next_high_price",
    "next_low_price",
    "next_close_price",
    "next_limit_up_price",
    "next_limit_down_price",
    "next_limit_status",
    "next_period_return",
    "holding_days",
    "buy_rank_by_factor_desc",
    "long_only_weight_equal",
    "long_only_return_sum",
    "raw_signal_stock_count",
    "tradable_signal_stock_count",
    "next_close_limit_excluded_count",
    "next_open_limit_excluded_count",
    "missing_next_record_count",
    "active_sleeve_return_sum",
    "active_stock_lot_count",
    "active_stock_count",
    "active_signal_sleeve_count",
    "active_missing_stock_return_count",
    "portfolio_invested_fraction",
    "avg_signal_momentum_raw",
    "avg_low_to_peak_return",
    "avg_drawdown_from_peak_to_signal",
    "long_only_return",
    "long_only_nav",
    "nav",
    "running_max",
    "drawdown",
]


# ============================================================
# 2. 参数和通用工具
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "factor1_LongOnly 诊断：检查 long-only 信号池、T+1 可交易过滤、"
            "组合 NAV、回撤、收益分布和现金持有原因。"
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--holding-days", type=int, default=None)
    parser.add_argument(
        "--trade-detail-date",
        "--trade-date",
        default=DEFAULT_TRADE_DETAIL_DATE,
        help="输出指定真实交易执行日的买入股票明细图片，格式 YYYY-MM-DD。该日对应 T+1 开盘买入。",
    )
    return parser.parse_args()


def write_csv(data: pd.DataFrame, output_path: Path, index: bool = False) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=index, encoding="utf-8-sig")
    return output_path


def numbered_path(output_dir: Path, number: int, base_name: str, suffix: str) -> Path:
    return output_dir / f"{number:02d}_{base_name}{suffix}"


def set_chinese_font() -> None:
    candidates = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    available_fonts = {font.name for font in plt.matplotlib.font_manager.fontManager.ttflist}
    for font_name in candidates:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def set_yearly_xaxis(ax: plt.Axes, dates: pd.Series) -> None:
    clean_dates = pd.to_datetime(dates, errors="coerce").dropna()
    if clean_dates.empty:
        return
    ax.set_xlim(pd.Timestamp(clean_dates.min().year, 1, 1), pd.Timestamp(clean_dates.max().year, 12, 31))
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelrotation=0)


def first_existing_file(input_dir: Path, names: list[str]) -> Path:
    for name in names:
        path = input_dir / name
        if path.exists():
            return path
    tried = "\n".join(str(input_dir / name) for name in names)
    raise SystemExit(f"缺少必要输入文件，已尝试：\n{tried}")


def optional_file(input_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        path = input_dir / name
        if path.exists():
            return path
    return None


def validate_required_columns(data: pd.DataFrame, path: Path, required_columns: list[str], file_label: str) -> None:
    missing_columns = [col for col in required_columns if col not in data.columns]
    if not missing_columns:
        return
    available_columns = ", ".join(map(str, data.columns[:40]))
    raise SystemExit(
        f"{file_label} 文件字段不完整：{path}\n"
        f"缺少字段：{', '.join(missing_columns)}\n"
        f"当前字段：{available_columns if available_columns else '(无字段)'}\n"
        "请先重新运行 factor1_LongOnly.py 生成完整回测输出，再运行 diagnostics。"
    )


def read_run_summary(input_dir: Path) -> dict[str, str]:
    path = input_dir / "run_summary.csv"
    if not path.exists():
        return {}
    data = pd.read_csv(path, dtype=str)
    if not {"metric", "value"}.issubset(data.columns):
        return {}
    return dict(zip(data["metric"].astype(str), data["value"].astype(str)))


def to_int(value: object, default: int) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator is None or pd.isna(denominator) or denominator == 0:
        return math.nan
    return float(numerator) / float(denominator)


def coerce_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
    }
    return series.astype("string").str.strip().str.lower().map(mapping).fillna(False).astype(bool)


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    if "stock_code" in output.columns:
        output["stock_code"] = output["stock_code"].astype("string").str.strip().str.zfill(6)
    for col in DATE_COLUMNS:
        if col in output.columns:
            output[col] = pd.to_datetime(output[col], errors="coerce")
    for col in NUMERIC_COLUMNS:
        if col in output.columns:
            output[col] = pd.to_numeric(output[col], errors="coerce")
    for col in BOOLEAN_COLUMNS:
        if col in output.columns:
            output[col] = coerce_boolean(output[col])
    return output


def sort_by_execution_date_and_factor(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    sort_date_col = "holding_start_trade_date" if "holding_start_trade_date" in output.columns else "trade_date"
    factor_col = FACTOR_SORT_COLUMN if FACTOR_SORT_COLUMN in output.columns else "stock_code"
    sort_cols = [sort_date_col]
    ascending = [True]
    if factor_col in output.columns:
        sort_cols.append(factor_col)
        ascending.append(False if factor_col != "stock_code" else True)
    if "stock_code" in output.columns and "stock_code" not in sort_cols:
        sort_cols.append("stock_code")
        ascending.append(True)
    return output.sort_values(sort_cols, ascending=ascending, na_position="last").reset_index(drop=True)


def attach_buy_rank_by_factor_desc(data: pd.DataFrame, tradable_col: str = "is_tradable_next_open") -> pd.DataFrame:
    output = data.copy()
    output["buy_rank_by_factor_desc"] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    if output.empty or FACTOR_SORT_COLUMN not in output.columns or "holding_start_trade_date" not in output.columns:
        return output
    tradable_mask = (
        output[tradable_col].fillna(False).astype(bool)
        if tradable_col in output.columns
        else pd.Series(True, index=output.index)
    )
    rank_mask = (
        tradable_mask
        & output["holding_start_trade_date"].notna()
        & pd.to_numeric(output[FACTOR_SORT_COLUMN], errors="coerce").notna()
    )
    if not rank_mask.any():
        return output
    ranked = output.loc[rank_mask].sort_values(
        ["holding_start_trade_date", FACTOR_SORT_COLUMN, "stock_code"],
        ascending=[True, False, True],
        na_position="last",
    )
    ranks = ranked.groupby("holding_start_trade_date", sort=False).cumcount() + 1
    output.loc[ranked.index, "buy_rank_by_factor_desc"] = pd.Series(ranks.to_numpy(), index=ranked.index, dtype="Int64")
    return output


def two_sided_t_pvalue(t_value: float, degrees_of_freedom: int) -> float:
    if pd.isna(t_value) or degrees_of_freedom <= 0:
        return math.nan
    try:
        from scipy import stats

        return float(2.0 * stats.t.sf(abs(t_value), df=degrees_of_freedom))
    except Exception:
        return float(math.erfc(abs(t_value) / math.sqrt(2.0)))


def mean_std_t_p(series: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    n = len(clean)
    if n == 0:
        return {"mean": math.nan, "std": math.nan, "t_value": math.nan, "p_value": math.nan, "observation_count": 0}
    mean_value = float(clean.mean())
    std_value = float(clean.std(ddof=1))
    if n <= 1 or pd.isna(std_value) or std_value == 0:
        t_value = math.nan
        p_value = math.nan
    else:
        t_value = mean_value / (std_value / math.sqrt(n))
        p_value = two_sided_t_pvalue(t_value, n - 1)
    return {"mean": mean_value, "std": std_value, "t_value": t_value, "p_value": p_value, "observation_count": n}


def format_metric_value(metric: str, value: object) -> str:
    if pd.isna(value):
        return ""
    metric_lower = str(metric).lower()
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if any(key in metric_lower for key in ["ratio", "return", "drawdown", "volatility", "win_rate", "positive"]):
        return f"{numeric_value:.2%}"
    if any(key in metric_lower for key in ["count", "rows", "days", "records", "stock"]):
        return f"{numeric_value:,.0f}"
    return f"{numeric_value:.6g}"


# ============================================================
# 3. 数据读取
# ============================================================


def load_signal_pool(input_dir: Path) -> pd.DataFrame:
    # 交易诊断的核心输入：五重筛选后并已匹配 T+1 开盘可交易状态的信号池。
    path = first_existing_file(input_dir, ["04_factor1_long_only_signal_pool_with_forward_returns.csv"])
    data = pd.read_csv(path, dtype={"stock_code": "string"}, low_memory=False)
    validate_required_columns(
        data,
        path,
        [
            "stock_code",
            "trade_date",
            "holding_start_trade_date",
            "is_tradable_next_open",
            "next_period_return_before_trade_filter",
            "next_period_return",
            "momentum_raw",
        ],
        "信号池",
    )
    return normalize_columns(data)


def load_screened_stock_pool(input_dir: Path) -> pd.DataFrame:
    path = optional_file(input_dir, ["03_factor1_long_only_screened_stock_pool.csv"])
    if path is None:
        return pd.DataFrame()
    return normalize_columns(pd.read_csv(path, dtype={"stock_code": "string"}, low_memory=False))


def load_long_only_returns(input_dir: Path) -> pd.DataFrame:
    # 每日 long-only 收益和 NAV 序列，是绩效复核和收益分布诊断的核心输入。
    path = first_existing_file(input_dir, ["05_factor1_long_only_holding_period_returns.csv"])
    data = pd.read_csv(path, low_memory=False)
    validate_required_columns(
        data,
        path,
        ["trade_date", "long_only_return", "long_only_nav"],
        "每日收益/NAV",
    )
    return normalize_columns(data)


def load_drawdown_series(input_dir: Path) -> pd.DataFrame:
    path = optional_file(input_dir, ["07_factor1_long_only_drawdown_series.csv"])
    if path is None:
        return pd.DataFrame()
    return normalize_columns(pd.read_csv(path, low_memory=False))


def load_optional_summary(input_dir: Path, names: list[str]) -> pd.DataFrame:
    path = optional_file(input_dir, names)
    if path is None:
        return pd.DataFrame()
    return normalize_columns(pd.read_csv(path, low_memory=False))


# ============================================================
# 4. 诊断表
# ============================================================


def calculate_drawdown_series(nav_series: pd.Series) -> pd.DataFrame:
    # 回撤统一基于 long_only_nav 的历史 running max 计算。
    nav = pd.to_numeric(nav_series, errors="coerce")
    running_max = nav.cummax()
    drawdown = nav / running_max - 1.0
    return pd.DataFrame({"nav": nav, "running_max": running_max, "drawdown": drawdown})


def calculate_performance_metrics(
    return_series: pd.Series,
    annualization_periods: float,
    nav_series: pd.Series | None = None,
) -> dict[str, float]:
    # 绩效复核使用主策略的加法 NAV 口径：累计收益 = final_nav - 1。
    clean = pd.to_numeric(return_series, errors="coerce").dropna()
    if clean.empty:
        return {
            "cumulative_return": math.nan,
            "annual_return": math.nan,
            "annual_volatility": math.nan,
            "sharpe_ratio": math.nan,
            "max_drawdown": math.nan,
            "win_rate": math.nan,
            "observation_count": 0,
            "return_curve_method": "additive_daily_return_nav",
        }
    nav = (
        pd.to_numeric(nav_series, errors="coerce").dropna()
        if nav_series is not None
        else 1.0 + clean.cumsum()
    )
    final_nav = float(nav.iloc[-1]) if not nav.empty else math.nan
    cumulative_return = final_nav - 1.0 if pd.notna(final_nav) else math.nan
    annual_return = float(clean.mean() * annualization_periods)
    annual_volatility = float(clean.std(ddof=1) * math.sqrt(annualization_periods)) if len(clean) > 1 else math.nan
    sharpe_ratio = annual_return / annual_volatility if annual_volatility and not pd.isna(annual_volatility) else math.nan
    max_drawdown = float(calculate_drawdown_series(nav)["drawdown"].min()) if not nav.empty else math.nan
    win_rate = float((clean > 0).mean())
    return {
        "cumulative_return": cumulative_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "observation_count": len(clean),
        "return_curve_method": "additive_daily_return_nav",
    }


def build_signal_quality_summary(signal_pool: pd.DataFrame, long_only_returns: pd.DataFrame) -> pd.DataFrame:
    # 汇总信号数量、T+1 开盘可交易数量、涨跌停/缺失过滤和事件收益分布。
    tradable_col = "is_tradable_next_open" if "is_tradable_next_open" in signal_pool else "is_tradable_next_close"
    limit_count_col = "next_open_limit_excluded_count" if "next_open_limit_excluded_count" in long_only_returns else "next_close_limit_excluded_count"
    tradable = signal_pool[tradable_col] if tradable_col in signal_pool else pd.Series(False, index=signal_pool.index)
    complete_return = (
        signal_pool["has_complete_holding_return"]
        if "has_complete_holding_return" in signal_pool
        else pd.Series(False, index=signal_pool.index)
    )
    stock_return = pd.to_numeric(signal_pool.loc[tradable, "next_period_return"], errors="coerce")
    before_filter_return = pd.to_numeric(signal_pool.get("next_period_return_before_trade_filter", pd.Series(dtype="float64")), errors="coerce")
    raw_signal_count = len(signal_pool)
    tradable_count = int(tradable.sum())
    complete_count = int(complete_return.sum())
    active_col = "has_active_position" if "has_active_position" in long_only_returns.columns else "has_tradable_signal"
    active_mask = long_only_returns.get(active_col, pd.Series(False, index=long_only_returns.index)).fillna(False).astype(bool)
    cash_days = int((~active_mask).sum())
    invested_days = int(active_mask.sum())

    rows = [
        ["signal_records_before_trade_filter", raw_signal_count],
        ["signal_stock_count", signal_pool["stock_code"].nunique() if "stock_code" in signal_pool else 0],
        ["signal_trade_date_count", signal_pool["trade_date"].nunique() if "trade_date" in signal_pool else 0],
        ["tradable_signal_records", tradable_count],
        ["tradable_signal_ratio", safe_ratio(tradable_count, raw_signal_count)],
        ["complete_holding_return_records", complete_count],
        ["complete_holding_return_ratio", safe_ratio(complete_count, raw_signal_count)],
        ["avg_stock_return_before_trade_filter", float(before_filter_return.mean()) if not before_filter_return.dropna().empty else math.nan],
        ["avg_stock_return_after_trade_filter", float(stock_return.mean()) if not stock_return.dropna().empty else math.nan],
        ["positive_stock_return_ratio_after_trade_filter", float((stock_return.dropna() > 0).mean()) if not stock_return.dropna().empty else math.nan],
        ["portfolio_period_count", len(long_only_returns)],
        ["invested_period_count", invested_days],
        ["cash_period_count", cash_days],
        ["cash_period_ratio", safe_ratio(cash_days, len(long_only_returns))],
        ["avg_raw_signal_stock_count_per_day", float(long_only_returns["raw_signal_stock_count"].mean()) if "raw_signal_stock_count" in long_only_returns else math.nan],
        ["avg_tradable_signal_stock_count_per_day", float(long_only_returns["tradable_signal_stock_count"].mean()) if "tradable_signal_stock_count" in long_only_returns else math.nan],
        ["avg_active_signal_sleeve_count_per_day", float(long_only_returns["active_signal_sleeve_count"].mean()) if "active_signal_sleeve_count" in long_only_returns else math.nan],
        ["avg_portfolio_invested_fraction", float(long_only_returns["portfolio_invested_fraction"].mean()) if "portfolio_invested_fraction" in long_only_returns else math.nan],
        ["avg_next_open_limit_excluded_count_per_day", float(long_only_returns[limit_count_col].mean()) if limit_count_col in long_only_returns else math.nan],
        ["avg_missing_next_record_count_per_day", float(long_only_returns["missing_next_record_count"].mean()) if "missing_next_record_count" in long_only_returns else math.nan],
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def build_trade_filter_reason_summary(signal_pool: pd.DataFrame) -> pd.DataFrame:
    # 按交易过滤原因统计哪些信号没有进入实际 T+1 开盘买入组合。
    if "trade_filter_reason" not in signal_pool.columns:
        return pd.DataFrame(columns=["trade_filter_reason", "record_count", "stock_count", "trade_date_count", "avg_next_period_return"])
    summary = (
        signal_pool.groupby("trade_filter_reason", dropna=False)
        .agg(
            record_count=("trade_filter_reason", "size"),
            stock_count=("stock_code", "nunique") if "stock_code" in signal_pool else ("trade_filter_reason", "size"),
            trade_date_count=("trade_date", "nunique") if "trade_date" in signal_pool else ("trade_filter_reason", "size"),
            avg_next_period_return=("next_period_return", "mean") if "next_period_return" in signal_pool else ("trade_filter_reason", "size"),
        )
        .reset_index()
    )
    summary["record_ratio"] = summary["record_count"] / max(len(signal_pool), 1)
    return summary.sort_values(["record_count", "trade_filter_reason"], ascending=[False, True]).reset_index(drop=True)


def build_daily_signal_diagnostics(long_only_returns: pd.DataFrame) -> pd.DataFrame:
    # 从每日 NAV 明细中抽取信号数、活跃持仓数、现金状态和当日组合收益。
    daily = long_only_returns.copy()
    limit_count_col = "next_open_limit_excluded_count" if "next_open_limit_excluded_count" in daily else "next_close_limit_excluded_count"
    daily["raw_to_tradable_ratio"] = daily.apply(
        lambda row: safe_ratio(row.get("tradable_signal_stock_count", math.nan), row.get("raw_signal_stock_count", math.nan)),
        axis=1,
    )
    daily["next_open_limit_exclusion_ratio"] = daily.apply(
        lambda row: safe_ratio(row.get(limit_count_col, math.nan), row.get("raw_signal_stock_count", math.nan)),
        axis=1,
    )
    daily["missing_next_record_ratio"] = daily.apply(
        lambda row: safe_ratio(row.get("missing_next_record_count", math.nan), row.get("raw_signal_stock_count", math.nan)),
        axis=1,
    )
    keep_cols = [
        "trade_date",
        "next_trade_date",
        "holding_start_trade_date",
        "holding_end_trade_date",
        "raw_signal_stock_count",
        "tradable_signal_stock_count",
        "active_signal_sleeve_count",
        "active_stock_lot_count",
        "active_stock_count",
        "portfolio_invested_fraction",
        "raw_to_tradable_ratio",
        "next_close_limit_excluded_count",
        "next_open_limit_excluded_count",
        "next_open_limit_exclusion_ratio",
        "missing_next_record_count",
        "missing_next_record_ratio",
        "avg_signal_momentum_raw",
        "avg_low_to_peak_return",
        "avg_drawdown_from_peak_to_signal",
        "long_only_return",
        "long_only_nav",
        "has_raw_signal",
        "has_tradable_signal",
        "has_active_position",
        "cash_reason",
    ]
    return daily[[col for col in keep_cols if col in daily.columns]].sort_values("trade_date").reset_index(drop=True)


def build_cash_reason_summary(long_only_returns: pd.DataFrame) -> pd.DataFrame:
    if "cash_reason" not in long_only_returns.columns:
        return pd.DataFrame(columns=["cash_reason", "period_count", "period_ratio", "avg_long_only_return"])
    summary = (
        long_only_returns.groupby("cash_reason", dropna=False)
        .agg(
            period_count=("cash_reason", "size"),
            avg_long_only_return=("long_only_return", "mean") if "long_only_return" in long_only_returns else ("cash_reason", "size"),
            avg_raw_signal_stock_count=("raw_signal_stock_count", "mean") if "raw_signal_stock_count" in long_only_returns else ("cash_reason", "size"),
            avg_tradable_signal_stock_count=("tradable_signal_stock_count", "mean") if "tradable_signal_stock_count" in long_only_returns else ("cash_reason", "size"),
        )
        .reset_index()
    )
    summary["period_ratio"] = summary["period_count"] / max(len(long_only_returns), 1)
    return summary.sort_values("period_count", ascending=False).reset_index(drop=True)


def build_return_distribution(signal_pool: pd.DataFrame, long_only_returns: pd.DataFrame, holding_days: int) -> pd.DataFrame:
    annualization_periods = TRADING_DAYS_PER_YEAR
    series_map = {
        "portfolio_long_only_return": long_only_returns.get("long_only_return", pd.Series(dtype="float64")),
        "stock_next_period_return_before_trade_filter": signal_pool.get("next_period_return_before_trade_filter", pd.Series(dtype="float64")),
    }
    tradable_col = "is_tradable_next_open" if "is_tradable_next_open" in signal_pool.columns else "is_tradable_next_close"
    if tradable_col in signal_pool.columns:
        series_map["stock_next_period_return_after_trade_filter"] = signal_pool.loc[
            signal_pool[tradable_col],
            "next_period_return",
        ]
    rows = []
    for name, series in series_map.items():
        values = pd.to_numeric(series, errors="coerce").dropna()
        stats = mean_std_t_p(values)
        row = {
            "return_series": name,
            **stats,
            "min": float(values.min()) if not values.empty else math.nan,
            "p01": float(values.quantile(0.01)) if not values.empty else math.nan,
            "p05": float(values.quantile(0.05)) if not values.empty else math.nan,
            "p25": float(values.quantile(0.25)) if not values.empty else math.nan,
            "median": float(values.median()) if not values.empty else math.nan,
            "p75": float(values.quantile(0.75)) if not values.empty else math.nan,
            "p95": float(values.quantile(0.95)) if not values.empty else math.nan,
            "p99": float(values.quantile(0.99)) if not values.empty else math.nan,
            "max": float(values.max()) if not values.empty else math.nan,
            "positive_ratio": float((values > 0).mean()) if not values.empty else math.nan,
            "negative_ratio": float((values < 0).mean()) if not values.empty else math.nan,
        }
        if name == "portfolio_long_only_return":
            row["annualized_volatility"] = float(values.std(ddof=1) * math.sqrt(annualization_periods)) if len(values) > 1 else math.nan
        else:
            row["annualized_volatility"] = math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_nav_drawdown_table(long_only_returns: pd.DataFrame, drawdown_series: pd.DataFrame) -> pd.DataFrame:
    base = long_only_returns[["trade_date", "long_only_return"]].copy()
    base["long_only_return"] = pd.to_numeric(base["long_only_return"], errors="coerce").fillna(0.0)
    if "long_only_nav" in long_only_returns.columns:
        base["long_only_nav"] = pd.to_numeric(long_only_returns["long_only_nav"], errors="coerce")
    else:
        base["long_only_nav"] = 1.0 + base["long_only_return"].cumsum()
    base["nav_curve_semantics"] = "additive_nav_from_daily_long_only_return"
    drawdown = calculate_drawdown_series(base["long_only_nav"])
    drawdown["trade_date"] = base["trade_date"].values
    cols = ["trade_date", "nav", "running_max", "drawdown"]
    output = base.merge(drawdown[[col for col in cols if col in drawdown.columns]], on="trade_date", how="left")
    if "nav" not in output.columns:
        output["nav"] = output["long_only_nav"]
    return output.sort_values("trade_date").reset_index(drop=True)


def build_monthly_performance(long_only_returns: pd.DataFrame) -> pd.DataFrame:
    data = long_only_returns[["trade_date", "long_only_return"]].dropna(subset=["trade_date"]).copy()
    if data.empty:
        return pd.DataFrame(columns=["year_month", "monthly_return", "observation_count"])
    data["year_month"] = data["trade_date"].dt.to_period("M").astype(str)
    summary = (
        data.groupby("year_month")
        .agg(
            monthly_return=("long_only_return", lambda x: float(pd.to_numeric(x, errors="coerce").fillna(0.0).sum())),
            observation_count=("long_only_return", "size"),
        )
        .reset_index()
    )
    return summary


def build_performance_recheck(long_only_returns: pd.DataFrame, holding_days: int) -> pd.DataFrame:
    metrics = calculate_performance_metrics(
        long_only_returns.get("long_only_return", pd.Series(dtype="float64")),
        annualization_periods=TRADING_DAYS_PER_YEAR,
        nav_series=long_only_returns.get("long_only_nav"),
    )
    return pd.DataFrame([{"portfolio": "factor1_long_only_recheck", **metrics, "holding_days": holding_days, "annualization_periods_per_year": TRADING_DAYS_PER_YEAR}])


def calculate_signal_pool_ic(signal_pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # IC 只检验最终信号池内部 momentum_raw 对未来收益的横截面排序/线性解释能力。
    required = {"trade_date", "momentum_raw", "next_period_return"}
    if not required.issubset(signal_pool.columns):
        return pd.DataFrame(), pd.DataFrame()
    data = signal_pool.copy()
    tradable_col = "is_tradable_next_open" if "is_tradable_next_open" in data.columns else "is_tradable_next_close"
    if tradable_col in data.columns:
        data = data.loc[data[tradable_col]].copy()
    valid = data[["trade_date", "momentum_raw", "next_period_return"]].dropna().copy()
    rows = []
    for trade_date, one_day in valid.groupby("trade_date"):
        if len(one_day) < 2:
            continue
        rows.append(
            {
                "trade_date": trade_date,
                "ic": one_day["momentum_raw"].corr(one_day["next_period_return"], method="pearson"),
                "rank_ic": one_day["momentum_raw"].corr(one_day["next_period_return"], method="spearman"),
                "stock_count": len(one_day),
            }
        )
    ic_series = pd.DataFrame(rows)
    if ic_series.empty:
        return pd.DataFrame(), pd.DataFrame()
    ic_series = ic_series.sort_values("trade_date").reset_index(drop=True)
    ic_series["cumulative_ic"] = ic_series["ic"].cumsum()
    ic_series["cumulative_rank_ic"] = ic_series["rank_ic"].cumsum()

    summary_rows = []
    for metric, col in [("SignalPoolIC", "ic"), ("SignalPoolRankIC", "rank_ic")]:
        stats = mean_std_t_p(ic_series[col])
        ir = stats["mean"] / stats["std"] if stats["std"] and not pd.isna(stats["std"]) else math.nan
        summary_rows.append({"metric": metric, **stats, "ir": ir})
    return ic_series, pd.DataFrame(summary_rows)


# ============================================================
# 5. 绘图
# ============================================================


def save_metric_table_image(table: pd.DataFrame, title: str, output_path: Path) -> None:
    display = table.copy()
    if {"metric", "value"}.issubset(display.columns):
        display["value"] = [format_metric_value(metric, value) for metric, value in zip(display["metric"], display["value"])]
    max_rows = min(len(display), 28)
    display = display.head(max_rows)
    fig, ax = plt.subplots(figsize=(11.5, max(3.2, 0.38 * (len(display) + 2))), dpi=170)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    table_artist = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(9.2)
    table_artist.scale(1.0, 1.32)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def format_nav_plot_metric(value: object, percent: bool = False) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "NA"
    return f"{numeric:.2%}" if percent else f"{numeric:.4f}"


def plot_long_only_nav(nav_drawdown: pd.DataFrame, output_path: Path) -> None:
    # 该函数保留给需要单独画 diagnostics NAV 时使用；当前主流程不再重复输出 NAV 图。
    plot_data = nav_drawdown.sort_values("trade_date").copy()
    nav_series = pd.to_numeric(plot_data.get("long_only_nav", pd.Series(dtype="float64")), errors="coerce")
    if "long_only_return" in plot_data:
        return_series = pd.to_numeric(plot_data["long_only_return"], errors="coerce")
    else:
        return_series = nav_series.diff()
        first_valid_index = nav_series.first_valid_index()
        if first_valid_index is not None:
            return_series.loc[first_valid_index] = nav_series.loc[first_valid_index] - 1.0
    metrics = calculate_performance_metrics(
        return_series,
        TRADING_DAYS_PER_YEAR,
        nav_series=nav_series,
    )
    clean_nav = nav_series.dropna()
    final_nav = float(clean_nav.iloc[-1]) if not clean_nav.empty else math.nan
    metric_text = "\n".join(
        [
            f"Final NAV: {format_nav_plot_metric(final_nav)}",
            f"CumRet: {format_nav_plot_metric(metrics.get('cumulative_return'), percent=True)}",
            f"Sharpe: {format_nav_plot_metric(metrics.get('sharpe_ratio'))}",
            f"MaxDD: {format_nav_plot_metric(metrics.get('max_drawdown'), percent=True)}",
        ]
    )

    fig, ax = plt.subplots(figsize=(13, 5.4), dpi=170)
    ax.plot(
        plot_data["trade_date"],
        nav_series,
        linewidth=1.15,
        color="#1f77b4",
        label="Long-only NAV",
    )
    ax.axhline(1.0, color="#666666", linestyle="--", linewidth=0.8)
    ax.text(
        0.99,
        0.98,
        metric_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#999999", "alpha": 0.88},
    )
    ax.set_title("factor1 Long-only NAV")
    ax.set_xlabel("时间")
    ax.set_ylabel("NAV")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.22)
    set_yearly_xaxis(ax, plot_data["trade_date"])
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_signal_counts(daily_signal: pd.DataFrame, output_path: Path) -> None:
    plot_data = daily_signal.sort_values("trade_date")
    limit_count_col = "next_open_limit_excluded_count" if "next_open_limit_excluded_count" in plot_data else "next_close_limit_excluded_count"
    fig, ax = plt.subplots(figsize=(13, 5.8), dpi=170)
    if "raw_signal_stock_count" in plot_data:
        ax.plot(plot_data["trade_date"], plot_data["raw_signal_stock_count"], linewidth=1.0, label="原始信号数", color="#4e79a7")
    if "tradable_signal_stock_count" in plot_data:
        ax.plot(plot_data["trade_date"], plot_data["tradable_signal_stock_count"], linewidth=1.0, label="T+1开盘可交易信号数", color="#59a14f")
    if limit_count_col in plot_data:
        ax.bar(plot_data["trade_date"], plot_data[limit_count_col], width=1.0, alpha=0.35, label="次日开盘涨跌停剔除数", color="#f28e2b")
    ax.set_title("factor1 Long-only 每日信号数量")
    ax.set_xlabel("时间")
    ax.set_ylabel("股票数")
    ax.legend()
    ax.grid(axis="y", alpha=0.22)
    set_yearly_xaxis(ax, plot_data["trade_date"])
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_return_distribution(signal_pool: pd.DataFrame, long_only_returns: pd.DataFrame, output_path: Path) -> None:
    portfolio_returns = pd.to_numeric(long_only_returns.get("long_only_return", pd.Series(dtype="float64")), errors="coerce").dropna()
    tradable_col = "is_tradable_next_open" if "is_tradable_next_open" in signal_pool.columns else "is_tradable_next_close"
    if tradable_col in signal_pool.columns and "next_period_return" in signal_pool.columns:
        stock_returns = pd.to_numeric(signal_pool.loc[signal_pool[tradable_col], "next_period_return"], errors="coerce").dropna()
    else:
        stock_returns = pd.Series(dtype="float64")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=170)
    axes[0].hist(portfolio_returns, bins=40, color="#4e79a7", alpha=0.82)
    axes[0].axvline(0.0, color="#666666", linestyle="--", linewidth=0.8)
    axes[0].set_title("组合日收益分布")
    axes[0].set_xlabel("收益率")
    axes[0].set_ylabel("期数")
    axes[0].grid(axis="y", alpha=0.22)

    axes[1].hist(stock_returns, bins=60, color="#59a14f", alpha=0.82)
    axes[1].axvline(0.0, color="#666666", linestyle="--", linewidth=0.8)
    axes[1].set_title("可交易信号个股持有期收益分布")
    axes[1].set_xlabel("收益率")
    axes[1].set_ylabel("记录数")
    axes[1].grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_trade_filter_reason(trade_filter_summary: pd.DataFrame, output_path: Path) -> None:
    if trade_filter_summary.empty:
        return
    plot_data = trade_filter_summary.sort_values("record_count", ascending=True).tail(12)
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=170)
    ax.barh(plot_data["trade_filter_reason"].astype(str), plot_data["record_count"], color="#4e79a7", alpha=0.85)
    ax.set_title("T+1 交易过滤原因分布")
    ax.set_xlabel("记录数")
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_signal_pool_cumulative_ic(ic_series: pd.DataFrame, output_path: Path) -> None:
    # 累计 IC 图只保留一张，用于查看信号池内部排序能力是否稳定。
    if ic_series.empty:
        return
    plot_data = ic_series.sort_values("trade_date").copy()
    ic_values = pd.to_numeric(plot_data["ic"], errors="coerce").dropna()
    ic_mean = float(ic_values.mean()) if not ic_values.empty else math.nan
    ic_std = float(ic_values.std(ddof=1)) if len(ic_values) > 1 else math.nan
    ic_ir = ic_mean / ic_std if ic_std and not pd.isna(ic_std) else math.nan
    ic_positive_ratio = float((ic_values > 0).mean()) if not ic_values.empty else math.nan
    stats_title = (
        f"IC Mean = {ic_mean:.4f}    IC Std = {ic_std:.4f}    "
        f"IR = {ic_ir:.4f}    IC > 0 = {ic_positive_ratio:.2%}"
    )
    if "cumulative_ic" not in plot_data.columns:
        plot_data["cumulative_ic"] = pd.to_numeric(plot_data["ic"], errors="coerce").cumsum()

    fig, ax = plt.subplots(figsize=(13, 5.4), dpi=170)
    ax.plot(plot_data["trade_date"], plot_data["cumulative_ic"], linewidth=1.15, color="#1f77b4", label="Cumulative IC")
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=0.8)
    fig.suptitle(stats_title, fontsize=11, y=0.995)
    ax.set_title("factor1 Long-only 信号池累计 IC 走势", fontsize=13)
    ax.set_xlabel("时间")
    ax.set_ylabel("累计 IC")
    ax.legend()
    ax.grid(alpha=0.22)
    set_yearly_xaxis(ax, plot_data["trade_date"])
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_optional_trade_detail_date(text: str | None) -> pd.Timestamp | None:
    if text is None or str(text).strip() == "":
        return None
    parsed = pd.to_datetime(str(text).strip(), errors="coerce")
    if pd.isna(parsed):
        raise SystemExit(f"--trade-detail-date 日期格式不正确：{text}。请使用 YYYY-MM-DD。")
    return pd.Timestamp(parsed).normalize()


def save_trade_detail_message_image(message: str, trade_date: pd.Timestamp, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 3.6), dpi=170)
    ax.axis("off")
    ax.text(0.5, 0.58, message, ha="center", va="center", fontsize=15, wrap=True)
    ax.text(
        0.5,
        0.28,
        f"查询日期：{trade_date.strftime('%Y-%m-%d')}",
        ha="center",
        va="center",
        fontsize=11,
        color="#666666",
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def output_trade_detail_image(
    trade_detail_date: pd.Timestamp | None,
    signal_pool: pd.DataFrame,
    long_only_returns: pd.DataFrame,
    output_dir: Path,
    lookback_days: int,
) -> Path | None:
    # 按实际买入日 holding_start_trade_date 查询；非交易日或无可交易股票时输出提示图。
    if trade_detail_date is None:
        return None

    trade_date = pd.Timestamp(trade_detail_date).normalize()
    output_path = numbered_path(output_dir, 15, f"factor1_long_only_trade_detail_{trade_date.strftime('%Y%m%d')}", ".png")
    # 使用 long_only_returns 的交易日历校验输入日期，避免对非交易日生成空表误判。
    trade_calendar = (
        pd.to_datetime(long_only_returns.get("trade_date", pd.Series(dtype="datetime64[ns]")), errors="coerce")
        .dropna()
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if trade_calendar.empty or not trade_calendar.eq(trade_date).any():
        message = "该日期不是 long-only 输出中的交易日，请输入正确交易日期。"
        print(message)
        return save_trade_detail_message_image(message, trade_date, output_path)

    first_available_trade_date = trade_calendar.iloc[lookback_days] if len(trade_calendar) > lookback_days else pd.NaT
    if pd.notna(first_available_trade_date) and trade_date < pd.Timestamp(first_available_trade_date).normalize():
        message = (
            f"该日期早于策略首个可用交易执行日 {pd.Timestamp(first_available_trade_date).strftime('%Y-%m-%d')}。"
            "2018 年前约 7 个自然月因历史不足 147 个交易日不可用，请输入正确日期。"
        )
        print(message)
        return save_trade_detail_message_image(message, trade_date, output_path)

    # 只展示该买入日真正通过 T+1 开盘可交易过滤的股票。
    if signal_pool.empty or "holding_start_trade_date" not in signal_pool.columns:
        day_trades = pd.DataFrame()
    else:
        tradable_col = "is_tradable_next_open" if "is_tradable_next_open" in signal_pool.columns else "is_tradable_next_close"
        tradable_mask = signal_pool[tradable_col].fillna(False).astype(bool) if tradable_col in signal_pool.columns else pd.Series(False, index=signal_pool.index)
        day_trades = signal_pool.loc[
            signal_pool["holding_start_trade_date"].dt.normalize().eq(trade_date)
            & tradable_mask
        ].copy()

    if day_trades.empty:
        message = "该交易日没有符合条件且可在开盘买入的股票。"
        print(message)
        return save_trade_detail_message_image(message, trade_date, output_path)

    # 明细表按因子值从大到小排序，便于查看当天买入优先级。
    display = sort_by_execution_date_and_factor(attach_buy_rank_by_factor_desc(day_trades, tradable_col=tradable_col))
    display["因子排名"] = pd.to_numeric(display.get("buy_rank_by_factor_desc", pd.Series(dtype="float64")), errors="coerce").map(
        lambda x: "" if pd.isna(x) else f"{int(x)}"
    )
    display["买入日期"] = display["holding_start_trade_date"].dt.strftime("%Y-%m-%d")
    display["信号日期"] = display["trade_date"].dt.strftime("%Y-%m-%d")
    display["股票代码"] = display["stock_code"].astype(str)
    display["因子值"] = pd.to_numeric(display.get(FACTOR_SORT_COLUMN, pd.Series(dtype="float64")), errors="coerce").map(
        lambda x: "" if pd.isna(x) else f"{x:.6f}"
    )
    display["买入开盘价"] = pd.to_numeric(display.get("holding_entry_open_price", pd.Series(dtype="float64")), errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    display["组合权重"] = pd.to_numeric(display.get("long_only_weight_equal", pd.Series(dtype="float64")), errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    display["低点到峰值涨幅"] = pd.to_numeric(display.get("low_to_peak_return", pd.Series(dtype="float64")), errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    display["峰值回撤"] = pd.to_numeric(display.get("drawdown_from_peak_to_signal", pd.Series(dtype="float64")), errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    display["峰值日期"] = pd.to_datetime(display.get("peak_trade_date", pd.Series(dtype="datetime64[ns]")), errors="coerce").dt.strftime("%Y-%m-%d")
    display["持有结束日期"] = pd.to_datetime(display.get("holding_end_trade_date", pd.Series(dtype="datetime64[ns]")), errors="coerce").dt.strftime("%Y-%m-%d")
    table_columns = ["因子排名", "买入日期", "信号日期", "股票代码", "因子值", "买入开盘价", "组合权重", "低点到峰值涨幅", "峰值回撤", "峰值日期", "持有结束日期"]
    table_data = display[table_columns]

    row_count = len(table_data)
    fig_height = min(max(3.8, 1.3 + 0.38 * (row_count + 1)), 20)
    fig, ax = plt.subplots(figsize=(15.5, fig_height), dpi=170)
    ax.axis("off")
    ax.set_title(f"{trade_date.strftime('%Y-%m-%d')} 开盘买入股票明细（共 {row_count} 只）", fontsize=16, pad=16)
    table = ax.table(
        cellText=table_data.values,
        colLabels=table_columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.28)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f4e79")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f4f7fb")
        cell.set_edgecolor("#d0d7de")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"已输出交易明细图片：{output_path}")
    return output_path


# ============================================================
# 6. 主流程
# ============================================================


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    set_chinese_font()
    trade_detail_date = parse_optional_trade_detail_date(args.trade_detail_date)

    run_summary = read_run_summary(input_dir)
    holding_days = args.holding_days if args.holding_days is not None else to_int(run_summary.get("holding_days"), DEFAULT_HOLDING_DAYS)

    print("1/6 正在读取 factor1_LongOnly 回测输出...")
    # 第 1 步：读取主策略已经生成的信号池、每日收益/NAV、回撤和筛选摘要。
    screened_stock_pool = load_screened_stock_pool(input_dir)
    signal_pool = load_signal_pool(input_dir)
    long_only_returns = load_long_only_returns(input_dir)
    drawdown_series = load_drawdown_series(input_dir)
    portfolio_detail = load_optional_summary(input_dir, ["06_factor1_long_only_portfolio_performance_detail.csv"])
    yearly_performance = load_optional_summary(input_dir, ["08_factor1_long_only_yearly_performance.csv"])
    filter_step_summary = load_optional_summary(input_dir, ["strong_momentum_filter_step_summary.csv"])
    screening_reason_summary = load_optional_summary(input_dir, ["strong_momentum_screening_summary.csv"])

    print("2/6 正在生成信号池和交易过滤诊断...")
    # 第 2 步：复核信号质量、T+1 交易过滤原因、每日信号数量和现金持有原因。
    signal_quality_summary = build_signal_quality_summary(signal_pool, long_only_returns)
    trade_filter_summary = build_trade_filter_reason_summary(signal_pool)
    daily_signal = build_daily_signal_diagnostics(long_only_returns)
    cash_reason_summary = build_cash_reason_summary(long_only_returns)
    write_csv(signal_quality_summary, numbered_path(output_dir, 9, "factor1_long_only_signal_quality_summary", ".csv"))
    write_csv(trade_filter_summary, numbered_path(output_dir, 9, "factor1_long_only_trade_filter_reason_summary", ".csv"))
    write_csv(daily_signal, numbered_path(output_dir, 9, "factor1_long_only_daily_signal_diagnostics", ".csv"))
    write_csv(cash_reason_summary, numbered_path(output_dir, 9, "factor1_long_only_cash_reason_summary", ".csv"))
    save_metric_table_image(
        signal_quality_summary,
        "factor1 Long-only 信号质量摘要",
        numbered_path(output_dir, 9, "factor1_long_only_signal_quality_summary", ".png"),
    )

    print("3/6 正在生成组合绩效、NAV 和收益分布诊断...")
    # 第 3 步：基于 long_only_returns 复算绩效、月度收益、NAV/回撤序列和收益分布。
    nav_drawdown = build_nav_drawdown_table(long_only_returns, drawdown_series)
    monthly_performance = build_monthly_performance(long_only_returns)
    performance_recheck = build_performance_recheck(long_only_returns, holding_days)
    return_distribution = build_return_distribution(signal_pool, long_only_returns, holding_days)
    write_csv(nav_drawdown, numbered_path(output_dir, 10, "factor1_long_only_nav_drawdown_series", ".csv"))
    write_csv(monthly_performance, numbered_path(output_dir, 10, "factor1_long_only_monthly_performance", ".csv"))
    write_csv(performance_recheck, numbered_path(output_dir, 10, "factor1_long_only_performance_recheck", ".csv"))
    write_csv(return_distribution, numbered_path(output_dir, 11, "factor1_long_only_return_distribution", ".csv"))
    if not performance_recheck.empty:
        print("Portfolio performance recheck:")
        print(performance_recheck.to_string(index=False))
    portfolio_detail = pd.DataFrame()
    if not portfolio_detail.empty:
        write_csv(portfolio_detail, numbered_path(output_dir, 10, "factor1_long_only_portfolio_performance_detail_copy", ".csv"))
    if not yearly_performance.empty:
        write_csv(yearly_performance, numbered_path(output_dir, 10, "factor1_long_only_yearly_performance_copy", ".csv"))

    print("4/6 正在绘制 long-only 诊断图表...")
    # 第 4 步：只输出收益分布、信号数量和交易过滤图；NAV 图由主策略 05 文件负责。
    plot_return_distribution(signal_pool, long_only_returns, numbered_path(output_dir, 11, "factor1_long_only_return_distribution", ".png"))
    plot_signal_counts(daily_signal, numbered_path(output_dir, 12, "factor1_long_only_signal_count_curve", ".png"))
    plot_trade_filter_reason(trade_filter_summary, numbered_path(output_dir, 12, "factor1_long_only_trade_filter_reason_bar", ".png"))
    trade_detail_image_path = output_trade_detail_image(
        trade_detail_date,
        signal_pool=signal_pool,
        long_only_returns=long_only_returns,
        output_dir=output_dir,
        lookback_days=to_int(run_summary.get("lookback_days"), DEFAULT_LOOKBACK_DAYS),
    )

    print("5/6 正在生成信号池 IC 和筛选摘要...")
    # 第 5 步：计算最终信号池内部的 IC/RankIC，并输出累计 IC 图和筛选漏斗摘要。
    ic_series, ic_summary = calculate_signal_pool_ic(signal_pool)
    write_csv(ic_series, numbered_path(output_dir, 13, "factor1_long_only_signal_pool_ic_series", ".csv"))
    write_csv(ic_summary, numbered_path(output_dir, 13, "factor1_long_only_signal_pool_ic_ir_summary", ".csv"))
    plot_signal_pool_cumulative_ic(ic_series, numbered_path(output_dir, 13, "factor1_long_only_cumulative_ic_curve", ".png"))
    write_csv(filter_step_summary, numbered_path(output_dir, 14, "factor1_long_only_filter_step_summary", ".csv"))
    write_csv(screening_reason_summary, numbered_path(output_dir, 14, "factor1_long_only_screening_reason_summary", ".csv"))

    diagnostic_summary = pd.DataFrame(
        # 第 6 步：记录 diagnostics 使用的输入目录、核心参数和输出图表路径。
        [
            ["input_dir", str(input_dir)],
            ["output_dir", str(output_dir)],
            ["factor_name", run_summary.get("factor_name", "factor1_LongOnly")],
            ["factor_version", run_summary.get("factor_version", "")],
            ["strategy_type", run_summary.get("strategy_type", "long_only_dretwd_real_market_rolling_sleeve_additive_nav")],
            ["lookback_days", to_int(run_summary.get("lookback_days"), DEFAULT_LOOKBACK_DAYS)],
            ["lookback_unit", run_summary.get("lookback_unit", "trading_days")],
            ["peak_selection_rule", run_summary.get("peak_selection_rule", "first_occurrence_of_highest_close_in_lookback_window")],
            ["max_peak_age_days", to_int(run_summary.get("max_peak_age_days"), DEFAULT_MAX_PEAK_AGE_DAYS)],
            ["holding_days", holding_days],
            ["holding_unit", run_summary.get("holding_unit", "trading_days")],
            ["entry_rule", run_summary.get("entry_rule", "next_trade_date_open")],
            ["exit_rule", run_summary.get("exit_rule", f"close_after_{holding_days}_close_to_close_return_intervals")],
            ["return_horizon", run_summary.get("return_horizon", f"T+1_Dretwd_to_T+{holding_days}_Dretwd")],
            ["holding_terminal_offset_days", run_summary.get("holding_terminal_offset_days", holding_days)],
            ["weighting_scheme", run_summary.get("weighting_scheme", f"rolling_{holding_days}_equal_capital_sleeves_equal_weight_stocks")],
            ["portfolio_model", run_summary.get("portfolio_model", "rolling_sleeve_dretwd_additive_nav")],
            ["nav_curve_semantics", run_summary.get("nav_curve_semantics", "additive_nav_from_daily_long_only_return")],
            ["short_side", run_summary.get("short_side", "none")],
            ["screened_stock_pool_rows", len(screened_stock_pool)],
            ["signal_pool_rows", len(signal_pool)],
            ["long_only_return_rows", len(long_only_returns)],
            ["diagnostic_signal_quality_rows", len(signal_quality_summary)],
            ["diagnostic_trade_filter_reason_rows", len(trade_filter_summary)],
            ["diagnostic_daily_signal_rows", len(daily_signal)],
            ["diagnostic_return_distribution_rows", len(return_distribution)],
            ["diagnostic_ic_observation_count", len(ic_series)],
            ["trade_detail_date", "" if trade_detail_date is None else trade_detail_date.strftime("%Y-%m-%d")],
            ["trade_detail_image_path", "" if trade_detail_image_path is None else str(trade_detail_image_path)],
        ],
        columns=["metric", "value"],
    )
    write_csv(diagnostic_summary, numbered_path(output_dir, 14, "factor1_LongOnly_diagnostic_run_summary", ".csv"))

    print("6/6 factor1_LongOnly 诊断完成。")
    print(f"输出目录：{output_dir}")
    if trade_detail_image_path is not None:
        print(f"交易明细图片：{trade_detail_image_path}")
    if not portfolio_detail.empty:
        print(f"组合绩效：\n{portfolio_detail.to_string(index=False)}")
    if not ic_summary.empty:
        print(f"信号池 IC/IR 摘要：\n{ic_summary.to_string(index=False)}")


if __name__ == "__main__":
    main()
