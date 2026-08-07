from __future__ import annotations

import argparse
import importlib.util
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 0. 手动参数区
# ============================================================


PARAM_START_DATE = "2018-01-01"
PARAM_END_DATE = "2021-12-31"
PARAM_MARKET_TYPES = {1, 4}
PARAM_MIN_FLOAT_MARKET_VALUE = 5_000_000.0

PARAM_HIGH_LOOKBACK_DAYS = 211
PARAM_DRAWDOWN_LOOKBACK_DAYS = 84
PARAM_DRAWDOWN_THRESHOLD = 0.23
PARAM_HOLDING_DAYS = 84

# 填写真实买入日期即可查看当日开盘买入股票明细；不需要时设为 None。
# 例：PARAM_TRADE_DETAIL_DATE = "2019-09-09"
PARAM_TRADE_DETAIL_DATE = None

PARAM_OUTPUT_ROOT = Path(r"D:\Desktop\CINDA qr\factor1_strong_momentum\factor2_2conditions_output")
PARAM_KEEP_TEMP = False
PARAM_REFRESH_LOCAL_DATA = False
PARAM_REFRESH_CLEAN_DATA = False
PARAM_FORCE_RECALCULATE_SIGNAL = False


# ============================================================
# 1. 基础配置
# ============================================================


def load_sibling_module(module_name: str, file_name: str):
    module_path = Path(__file__).resolve().with_name(file_name)
    if not module_path.exists():
        raise FileNotFoundError(f"找不到依赖文件：{module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"无法加载模块：{module_path}")
    spec.loader.exec_module(module)
    return module


BASE = load_sibling_module("factor2_longonly_base", "factor2_LongOnly.py")

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PARAM_OUTPUT_ROOT
LOCAL_RAW_DATA_CACHE_DIR = DEFAULT_OUTPUT_ROOT / "_local_data_cache"
LOCAL_CLEAN_DATA_CACHE_DIR = DEFAULT_OUTPUT_ROOT / "_local_data_cache"
FALLBACK_DATA_CACHE_DIR = Path(r"D:\Desktop\CINDA qr\factor1_strong_momentum\factor2_LongOnly_output\_local_data_cache")

DEFAULT_START_DATE = PARAM_START_DATE
DEFAULT_END_DATE = PARAM_END_DATE
DEFAULT_HOLDING_DAYS = PARAM_HOLDING_DAYS
DEFAULT_MARKET_TYPES = set(PARAM_MARKET_TYPES)
DEFAULT_MIN_FLOAT_MARKET_VALUE = PARAM_MIN_FLOAT_MARKET_VALUE

DEFAULT_HIGH_LOOKBACK_DAYS = PARAM_HIGH_LOOKBACK_DAYS
DEFAULT_DRAWDOWN_LOOKBACK_DAYS = PARAM_DRAWDOWN_LOOKBACK_DAYS
DEFAULT_DRAWDOWN_THRESHOLD = PARAM_DRAWDOWN_THRESHOLD
DEFAULT_TRADE_DETAIL_DATE = PARAM_TRADE_DETAIL_DATE
DEFAULT_KEEP_TEMP = PARAM_KEEP_TEMP
DEFAULT_REFRESH_LOCAL_DATA = PARAM_REFRESH_LOCAL_DATA
DEFAULT_REFRESH_CLEAN_DATA = PARAM_REFRESH_CLEAN_DATA
DEFAULT_FORCE_RECALCULATE_SIGNAL = PARAM_FORCE_RECALCULATE_SIGNAL
FACTOR_VERSION = "factor2_2conditions_half_year_high_and_three_month_drawdown_v1"

TRADING_DAYS_PER_YEAR = getattr(BASE, "TRADING_DAYS_PER_YEAR", 252)
PRICE_COMPARE_TOLERANCE = getattr(BASE, "PRICE_COMPARE_TOLERANCE", 1e-6)
KEY_COLUMNS = ["stock_code", "trade_date"]

DATE_COLUMNS = [
    "trade_date",
    "capital_change_date",
    "lookback_window_start",
    "prior_high_trade_date",
    "drawdown_window_start",
    "drawdown_peak_trade_date",
    "drawdown_trough_trade_date",
    "next_trade_date",
    "holding_start_trade_date",
    "holding_end_trade_date",
    "active_signal_trade_date_min",
    "active_signal_trade_date_max",
    "holding_start_trade_date_min",
    "holding_start_trade_date_max",
    "holding_end_trade_date_min",
    "holding_end_trade_date_max",
]


# ============================================================
# 2. 参数和通用工具
# ============================================================


def parse_market_types(text: str | None) -> set[int]:
    if text is None or str(text).strip() == "":
        return set(DEFAULT_MARKET_TYPES)
    values: set[int] = set()
    for token in str(text).replace("，", ",").split(","):
        token = token.strip()
        if token:
            values.add(int(token))
    return values or set(DEFAULT_MARKET_TYPES)


def market_types_to_tag(market_types: set[int]) -> str:
    return BASE.market_types_to_tag(market_types)


def build_raw_data_cache_path_in(cache_dir: Path, start_date: str, end_date: str, market_types: set[int]) -> Path:
    start_tag = start_date.replace("-", "")
    end_tag = end_date.replace("-", "")
    market_tag = market_types_to_tag(market_types)
    return cache_dir / f"{BASE.MARKET_TABLE}_raw_sql_{start_tag}_{end_tag}_mkt{market_tag}.csv"


def build_clean_data_cache_paths_in(
    cache_dir: Path,
    start_date: str,
    end_date: str,
    min_float_market_value: float,
    market_types: set[int],
) -> dict[str, Path]:
    start_tag = start_date.replace("-", "")
    end_tag = end_date.replace("-", "")
    market_tag = market_types_to_tag(market_types)
    min_value_tag = BASE.build_cache_number_tag(min_float_market_value)
    prefix = f"{BASE.MARKET_TABLE}_cleaned_ffill_{start_tag}_{end_tag}_mkt{market_tag}_minmv{min_value_tag}"
    return {
        "clean_data": cache_dir / f"{prefix}.csv",
        "cleaning_log": cache_dir / f"{prefix}_cleaning_step_log.csv",
        "missing_summary": cache_dir / f"{prefix}_missing_value_ffill_summary.csv",
        "exclusion_summary": cache_dir / f"{prefix}_exclusion_reason_summary.csv",
    }


def build_long_only_experiment_folder_name(
    high_lookback_days: int,
    drawdown_lookback_days: int,
    holding_days: int,
    market_types: set[int],
) -> str:
    return (
        f"longonly_high{int(high_lookback_days)}"
        f"_dd{int(drawdown_lookback_days)}"
        f"_hd{int(holding_days)}"
        f"_mkt{market_types_to_tag(market_types)}"
    )


def build_default_experiment_dir(output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    return output_root / build_long_only_experiment_folder_name(
        DEFAULT_HIGH_LOOKBACK_DAYS,
        DEFAULT_DRAWDOWN_LOOKBACK_DAYS,
        DEFAULT_HOLDING_DAYS,
        DEFAULT_MARKET_TYPES,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "factor2_2conditions long-only：T日收盘价严格突破前126个交易记录最高收盘价，"
            "且近63个交易记录未复权收盘价最大回撤超过20%；T+1开盘买入，"
            "持有期收益和NAV均用收益累加口径。"
        )
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--market-types", default=",".join(str(item) for item in sorted(DEFAULT_MARKET_TYPES)))
    parser.add_argument("--min-float-market-value", type=float, default=DEFAULT_MIN_FLOAT_MARKET_VALUE)
    parser.add_argument("--high-lookback-days", type=int, default=DEFAULT_HIGH_LOOKBACK_DAYS)
    parser.add_argument("--drawdown-lookback-days", type=int, default=DEFAULT_DRAWDOWN_LOOKBACK_DAYS)
    parser.add_argument("--drawdown-threshold", type=float, default=DEFAULT_DRAWDOWN_THRESHOLD)
    parser.add_argument("--holding-days", type=int, default=DEFAULT_HOLDING_DAYS)
    parser.add_argument(
        "--trade-detail-date",
        "--trade-date",
        default=DEFAULT_TRADE_DETAIL_DATE,
        help="输出指定真实交易执行日的买入股票明细图片，格式 YYYY-MM-DD。该日对应 T+1 开盘买入。",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--keep-temp", action="store_true", default=DEFAULT_KEEP_TEMP)
    parser.add_argument("--refresh-local-data", action="store_true", default=DEFAULT_REFRESH_LOCAL_DATA)
    parser.add_argument("--refresh-clean-data", action="store_true", default=DEFAULT_REFRESH_CLEAN_DATA)
    parser.add_argument(
        "--force-recalculate-signal",
        "--force-recalculate-factor",
        action="store_true",
        default=DEFAULT_FORCE_RECALCULATE_SIGNAL,
        help="忽略已有信号结果文件，重新执行两条件筛选和绩效输出。",
    )
    return parser.parse_args()


def normalize_date_columns(data: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    output = data.copy()
    for col in columns or DATE_COLUMNS:
        if col in output.columns:
            output[col] = pd.to_datetime(output[col], errors="coerce")
    return output


def format_signal_dates(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    for col in DATE_COLUMNS:
        if col in output.columns:
            output[col] = pd.to_datetime(output[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return output


def write_csv(data: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data.to_csv(output_path, index=False, encoding="utf-8-sig")
        return output_path
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = output_path.with_name(f"{output_path.stem}_{timestamp}{output_path.suffix}")
        data.to_csv(fallback_path, index=False, encoding="utf-8-sig")
        print(f"目标文件被占用，已写入备用文件：{fallback_path}")
        return fallback_path


def safe_nunique(data: pd.DataFrame, column: str) -> int:
    return int(data[column].nunique()) if column in data.columns else 0


def parse_optional_trade_detail_date(text: str | None) -> pd.Timestamp | None:
    if text is None or str(text).strip() == "":
        return None
    parsed = pd.to_datetime(str(text).strip(), errors="coerce")
    if pd.isna(parsed):
        raise SystemExit(f"--trade-detail-date 日期格式不正确：{text}。请使用 YYYY-MM-DD。")
    return pd.Timestamp(parsed).normalize()


def sort_by_execution_date_and_stock(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    sort_date_col = "holding_start_trade_date" if "holding_start_trade_date" in output.columns else "trade_date"
    sort_cols = [sort_date_col]
    if "stock_code" in output.columns:
        sort_cols.append("stock_code")
    return output.sort_values(sort_cols, ascending=True, na_position="last").reset_index(drop=True)


# ============================================================
# 3. 两条件信号筛选
# ============================================================


def calculate_price_drawdown_window(prices: np.ndarray) -> tuple[float, int, int]:
    if prices.size == 0 or not np.isfinite(prices).all() or (prices <= 0).any():
        return math.nan, -1, -1
    running_high = np.maximum.accumulate(prices)
    drawdowns = prices / running_high - 1.0
    trough_offset = int(np.argmin(drawdowns))
    peak_price = running_high[trough_offset]
    peak_candidates = np.flatnonzero(prices[: trough_offset + 1] == peak_price)
    peak_offset = int(peak_candidates[0]) if peak_candidates.size else trough_offset
    return float(drawdowns[trough_offset]), peak_offset, trough_offset


def build_screening_reason(data: pd.DataFrame) -> pd.Series:
    conditions = [
        ~data["has_full_high_lookback"].fillna(False),
        ~data["high_breakout_filter"].fillna(False),
        ~data["has_full_drawdown_lookback"].fillna(False),
        ~data["drawdown_filter"].fillna(False),
        data["passes_signal_filters"].fillna(False),
    ]
    choices = [
        "insufficient_high_lookback_history",
        "not_strict_breakout_above_prior_high",
        "insufficient_drawdown_lookback_history",
        "drawdown_not_deep_enough",
        "passed",
    ]
    return pd.Series(np.select(conditions, choices, default="unknown"), index=data.index)


def calculate_one_stock_two_condition_signal(
    stock_data: pd.DataFrame,
    high_lookback_days: int,
    drawdown_lookback_days: int,
    drawdown_threshold: float,
) -> pd.DataFrame:
    stock = stock_data.sort_values("trade_date").reset_index(drop=True).copy()
    close_values = pd.to_numeric(stock["close_price"], errors="coerce").to_numpy(dtype="float64")
    dates = pd.to_datetime(stock["trade_date"], errors="coerce").reset_index(drop=True)

    high_lookback_days = max(int(high_lookback_days), 1)
    drawdown_lookback_days = max(int(drawdown_lookback_days), 1)

    prior_high_dates: list[pd.Timestamp] = []
    prior_high_prices: list[float] = []
    high_window_starts: list[pd.Timestamp] = []
    high_valid_days: list[int] = []
    drawdown_window_starts: list[pd.Timestamp] = []
    drawdown_valid_days: list[int] = []
    rolling_max_drawdowns: list[float] = []
    drawdown_peak_dates: list[pd.Timestamp] = []
    drawdown_peak_prices: list[float] = []
    drawdown_trough_dates: list[pd.Timestamp] = []
    drawdown_trough_prices: list[float] = []
    breakout_returns: list[float] = []
    full_history_valid_days: list[int] = []

    valid_seen = 0
    for i, close_t in enumerate(close_values):
        prior_valid_seen = valid_seen
        is_valid_close_t = np.isfinite(close_t) and close_t > 0
        if is_valid_close_t:
            valid_seen += 1

        high_start_pos = max(0, i - high_lookback_days)
        high_window = close_values[high_start_pos:i]
        valid_high_offsets = np.flatnonzero(np.isfinite(high_window) & (high_window > 0))
        high_window_starts.append(dates.iloc[high_start_pos] if i > 0 else pd.NaT)
        high_valid_days.append(int(valid_high_offsets.size))

        if valid_high_offsets.size > 0:
            valid_high_prices = high_window[valid_high_offsets]
            first_high_offset = int(valid_high_offsets[np.argmax(valid_high_prices)])
            prior_high_pos = high_start_pos + first_high_offset
            prior_high_price = float(close_values[prior_high_pos])
            prior_high_dates.append(dates.iloc[prior_high_pos])
            prior_high_prices.append(prior_high_price)
            breakout_returns.append(
                float(close_t / prior_high_price - 1.0)
                if is_valid_close_t and prior_high_price > 0
                else math.nan
            )
        else:
            prior_high_dates.append(pd.NaT)
            prior_high_prices.append(math.nan)
            breakout_returns.append(math.nan)

        dd_start_pos = max(0, i - drawdown_lookback_days + 1)
        dd_window = close_values[dd_start_pos : i + 1]
        valid_dd_mask = np.isfinite(dd_window) & (dd_window > 0)
        drawdown_window_starts.append(dates.iloc[dd_start_pos])
        drawdown_valid_days.append(int(valid_dd_mask.sum()))
        if dd_window.size == drawdown_lookback_days and valid_dd_mask.all():
            max_drawdown, peak_offset, trough_offset = calculate_price_drawdown_window(dd_window)
            peak_pos = dd_start_pos + peak_offset if peak_offset >= 0 else -1
            trough_pos = dd_start_pos + trough_offset if trough_offset >= 0 else -1
            rolling_max_drawdowns.append(max_drawdown)
            drawdown_peak_dates.append(dates.iloc[peak_pos] if peak_pos >= 0 else pd.NaT)
            drawdown_peak_prices.append(float(close_values[peak_pos]) if peak_pos >= 0 else math.nan)
            drawdown_trough_dates.append(dates.iloc[trough_pos] if trough_pos >= 0 else pd.NaT)
            drawdown_trough_prices.append(float(close_values[trough_pos]) if trough_pos >= 0 else math.nan)
        else:
            rolling_max_drawdowns.append(math.nan)
            drawdown_peak_dates.append(pd.NaT)
            drawdown_peak_prices.append(math.nan)
            drawdown_trough_dates.append(pd.NaT)
            drawdown_trough_prices.append(math.nan)

        full_history_valid_days.append(prior_valid_seen)

    stock["lookback_window_start"] = pd.to_datetime(high_window_starts)
    stock["prior_high_trade_date"] = pd.to_datetime(prior_high_dates)
    stock["prior_high_close_price"] = pd.to_numeric(prior_high_prices, errors="coerce")
    stock["high_lookback_valid_days"] = pd.Series(high_valid_days, dtype="Int64")
    stock["drawdown_window_start"] = pd.to_datetime(drawdown_window_starts)
    stock["drawdown_lookback_valid_days"] = pd.Series(drawdown_valid_days, dtype="Int64")
    stock["rolling_max_drawdown"] = pd.to_numeric(rolling_max_drawdowns, errors="coerce")
    stock["drawdown_peak_trade_date"] = pd.to_datetime(drawdown_peak_dates)
    stock["drawdown_peak_close_price"] = pd.to_numeric(drawdown_peak_prices, errors="coerce")
    stock["drawdown_trough_trade_date"] = pd.to_datetime(drawdown_trough_dates)
    stock["drawdown_trough_close_price"] = pd.to_numeric(drawdown_trough_prices, errors="coerce")
    stock["breakout_return"] = pd.to_numeric(breakout_returns, errors="coerce")
    stock["full_history_valid_days"] = full_history_valid_days

    stock["has_full_high_lookback"] = stock["high_lookback_valid_days"].eq(high_lookback_days)
    stock["has_full_drawdown_lookback"] = stock["drawdown_lookback_valid_days"].eq(drawdown_lookback_days)
    stock["high_breakout_filter"] = (
        stock["has_full_high_lookback"]
        & pd.to_numeric(stock["close_price"], errors="coerce").gt(stock["prior_high_close_price"])
    )
    stock["drawdown_filter"] = (
        stock["has_full_drawdown_lookback"]
        & stock["rolling_max_drawdown"].lt(-float(drawdown_threshold))
    )
    stock["passes_signal_filters"] = stock["high_breakout_filter"] & stock["drawdown_filter"]
    stock["screening_reason"] = build_screening_reason(stock)
    return stock


def calculate_raw_two_condition_signals(
    clean_data: pd.DataFrame,
    high_lookback_days: int,
    drawdown_lookback_days: int,
    drawdown_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = normalize_date_columns(clean_data).sort_values(KEY_COLUMNS).reset_index(drop=True)
    parts = [
        calculate_one_stock_two_condition_signal(
            one_stock,
            high_lookback_days=high_lookback_days,
            drawdown_lookback_days=drawdown_lookback_days,
            drawdown_threshold=drawdown_threshold,
        )
        for _, one_stock in data.groupby("stock_code", sort=False)
    ]
    signals = pd.concat(parts, ignore_index=True) if parts else data.iloc[0:0].copy()
    signals = signals.rename(columns={"close_price": "signal_close_price"})
    signals["high_lookback_days"] = high_lookback_days
    signals["drawdown_lookback_days"] = drawdown_lookback_days
    signals["drawdown_threshold"] = drawdown_threshold
    signals["rebalance_frequency"] = "daily"

    ordered_columns = [
        "stock_code",
        "trade_date",
        "signal_close_price",
        "lookback_window_start",
        "prior_high_trade_date",
        "prior_high_close_price",
        "high_lookback_valid_days",
        "breakout_return",
        "drawdown_window_start",
        "drawdown_lookback_valid_days",
        "rolling_max_drawdown",
        "drawdown_peak_trade_date",
        "drawdown_peak_close_price",
        "drawdown_trough_trade_date",
        "drawdown_trough_close_price",
        "full_history_valid_days",
        "has_full_high_lookback",
        "has_full_drawdown_lookback",
        "high_breakout_filter",
        "drawdown_filter",
        "passes_signal_filters",
        "screening_reason",
        "high_lookback_days",
        "drawdown_lookback_days",
        "drawdown_threshold",
        "rebalance_frequency",
    ]
    signals = signals[[col for col in ordered_columns if col in signals.columns]].copy()
    valid_signal = signals.loc[signals["passes_signal_filters"].fillna(False)].copy()
    missing_signal = signals.loc[~signals["passes_signal_filters"].fillna(False)].copy()
    return valid_signal, missing_signal


def build_all_signals(valid_signal: pd.DataFrame, missing_signal: pd.DataFrame) -> pd.DataFrame:
    parts = [part for part in [valid_signal, missing_signal] if not part.empty]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(["trade_date", "stock_code"]).reset_index(drop=True)


def screening_summary(all_signal: pd.DataFrame) -> pd.DataFrame:
    if all_signal.empty:
        return pd.DataFrame(columns=["screening_reason", "record_count", "stock_count", "trade_date_count"])
    return (
        all_signal.groupby("screening_reason", dropna=False)
        .agg(
            record_count=("screening_reason", "size"),
            stock_count=("stock_code", "nunique"),
            trade_date_count=("trade_date", "nunique"),
        )
        .reset_index()
        .sort_values("record_count", ascending=False)
        .reset_index(drop=True)
    )


def screening_filter_step_summary(
    all_signal: pd.DataFrame,
    high_lookback_days: int,
    drawdown_lookback_days: int,
    drawdown_threshold: float,
) -> pd.DataFrame:
    columns = ["step", "description", "record_count", "stock_count", "trade_date_count"]
    if all_signal.empty:
        return pd.DataFrame(columns=columns)
    step1 = all_signal["high_breakout_filter"].fillna(False)
    step2 = step1 & all_signal["drawdown_filter"].fillna(False)
    masks = [
        ("observation_pool", "T日可观察股票池", pd.Series(True, index=all_signal.index)),
        (
            "step1_strict_high_breakout",
            f"T日未复权收盘价严格大于T-1至T-{high_lookback_days}最高收盘价",
            step1,
        ),
        (
            "step2_recent_drawdown",
            f"最近{drawdown_lookback_days}个交易记录未复权收盘价最大回撤超过{drawdown_threshold:.0%}",
            step2,
        ),
    ]
    rows = []
    for step, description, mask in masks:
        data = all_signal.loc[mask.fillna(False)]
        rows.append(
            {
                "step": step,
                "description": description,
                "record_count": len(data),
                "stock_count": safe_nunique(data, "stock_code"),
                "trade_date_count": safe_nunique(data, "trade_date"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


# ============================================================
# 4. T+1 开盘交易过滤、持有期收益和每日 NAV
# ============================================================


def calculate_forward_holding_returns_sum(clean_data: pd.DataFrame, holding_days: int) -> pd.DataFrame:
    data = normalize_date_columns(clean_data).sort_values(KEY_COLUMNS)[
        ["stock_code", "trade_date", "open_price", "close_price", "return_with_dividend"]
    ].copy()
    data["open_price"] = pd.to_numeric(data["open_price"], errors="coerce")
    data["close_price"] = pd.to_numeric(data["close_price"], errors="coerce")
    data["return_with_dividend"] = pd.to_numeric(data["return_with_dividend"], errors="coerce")
    grouped = data.groupby("stock_code", sort=False)
    terminal_offset_days = max(int(holding_days), 1)

    entry_open = grouped["open_price"].shift(-1)
    entry_close = grouped["close_price"].shift(-1)
    holding_end_close = grouped["close_price"].shift(-terminal_offset_days)
    complete_mask = entry_open.gt(0) & holding_end_close.gt(0)
    data["future_return_valid_days"] = 0
    data["holding_period_return_sum"] = 0.0
    for offset in range(1, terminal_offset_days + 1):
        shifted_return = grouped["return_with_dividend"].shift(-offset)
        valid = shifted_return.notna()
        data["future_return_valid_days"] += valid.astype("int64")
        data["holding_period_return_sum"] += shifted_return.fillna(0.0)
    complete_mask &= data["future_return_valid_days"].eq(terminal_offset_days)

    data["holding_entry_open_price"] = entry_open
    data["holding_entry_close_price"] = entry_close
    data["holding_entry_price"] = entry_open
    data["holding_entry_price_type"] = "T+1_open"
    data["holding_end_close_price"] = holding_end_close
    data["holding_exit_close_price"] = holding_end_close
    data["holding_terminal_offset_days"] = terminal_offset_days
    data["next_period_return_before_trade_filter"] = data["holding_period_return_sum"]
    data.loc[~complete_mask, "next_period_return_before_trade_filter"] = np.nan
    data["has_complete_holding_return"] = complete_mask
    data["forward_return_rule"] = f"buy_t_plus_1_open_hold_{holding_days}_days_sum_dretwd_t_plus_1_to_t_plus_{terminal_offset_days}"
    return data[
        [
            "stock_code",
            "trade_date",
            "holding_entry_open_price",
            "holding_entry_close_price",
            "holding_entry_price",
            "holding_entry_price_type",
            "holding_end_close_price",
            "holding_exit_close_price",
            "holding_terminal_offset_days",
            "next_period_return_before_trade_filter",
            "future_return_valid_days",
            "has_complete_holding_return",
            "forward_return_rule",
        ]
    ]


def attach_long_only_execution_and_returns(
    signal_pool: pd.DataFrame,
    clean_data: pd.DataFrame,
    holding_days: int,
) -> pd.DataFrame:
    if signal_pool.empty:
        columns = list(signal_pool.columns) + [
            "next_trade_date",
            "holding_start_trade_date",
            "holding_end_trade_date",
            "holding_entry_open_price",
            "holding_entry_close_price",
            "holding_entry_price",
            "holding_entry_price_type",
            "holding_end_close_price",
            "holding_exit_close_price",
            "holding_terminal_offset_days",
            "next_period_return_before_trade_filter",
            "next_period_return",
            "has_complete_holding_return",
            "forward_return_rule",
            "next_open_price",
            "next_high_price",
            "next_low_price",
            "next_close_price",
            "next_limit_up_price",
            "next_limit_down_price",
            "next_limit_status",
            "is_next_open_limit_up",
            "is_next_open_limit_down",
            "is_next_open_one_word_limit",
            "is_next_close_limit_up",
            "is_next_close_limit_down",
            "is_next_close_limit",
            "is_tradable_next_open",
            "is_tradable_next_close",
            "trade_filter_reason",
            "long_only_weight_equal",
        ]
        return pd.DataFrame(columns=columns)

    signals = normalize_date_columns(signal_pool).copy()
    clean = normalize_date_columns(clean_data).copy()

    trade_calendar = clean[["trade_date"]].drop_duplicates().sort_values("trade_date").reset_index(drop=True)
    terminal_offset_days = max(int(holding_days), 1)
    trade_calendar["next_trade_date"] = trade_calendar["trade_date"].shift(-1)
    trade_calendar["holding_start_trade_date"] = trade_calendar["next_trade_date"]
    trade_calendar["holding_end_trade_date"] = trade_calendar["trade_date"].shift(-terminal_offset_days)
    signals = signals.merge(trade_calendar, on="trade_date", how="left")

    forward_returns = calculate_forward_holding_returns_sum(clean, holding_days)
    signals = signals.merge(forward_returns, on=["stock_code", "trade_date"], how="left")

    entry_cols = [
        "stock_code",
        "trade_date",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "limit_up_price",
        "limit_down_price",
        "limit_status",
    ]
    entry_data = clean.sort_values(KEY_COLUMNS)[entry_cols].rename(
        columns={
            "trade_date": "next_trade_date",
            "open_price": "next_open_price",
            "high_price": "next_high_price",
            "low_price": "next_low_price",
            "close_price": "next_close_price",
            "limit_up_price": "next_limit_up_price",
            "limit_down_price": "next_limit_down_price",
            "limit_status": "next_limit_status",
        }
    )
    signals = signals.merge(entry_data, on=["stock_code", "next_trade_date"], how="left")

    close_equals_limit_up = (signals["next_close_price"] - signals["next_limit_up_price"]).abs().le(PRICE_COMPARE_TOLERANCE)
    close_equals_limit_down = (signals["next_close_price"] - signals["next_limit_down_price"]).abs().le(PRICE_COMPARE_TOLERANCE)
    signals["is_next_close_limit_up"] = close_equals_limit_up.fillna(False).astype(bool)
    signals["is_next_close_limit_down"] = close_equals_limit_down.fillna(False).astype(bool)
    signals["is_next_close_limit"] = signals["is_next_close_limit_up"] | signals["is_next_close_limit_down"]
    open_equals_limit_up = (signals["next_open_price"] - signals["next_limit_up_price"]).abs().le(PRICE_COMPARE_TOLERANCE)
    open_equals_limit_down = (signals["next_open_price"] - signals["next_limit_down_price"]).abs().le(PRICE_COMPARE_TOLERANCE)
    signals["is_next_open_limit_up"] = open_equals_limit_up.fillna(False).astype(bool)
    signals["is_next_open_limit_down"] = open_equals_limit_down.fillna(False).astype(bool)
    signals["is_next_open_one_word_limit"] = signals["is_next_open_limit_up"] | signals["is_next_open_limit_down"]

    missing_next = signals["next_open_price"].isna() | signals["next_period_return_before_trade_filter"].isna()
    signals["is_tradable_next_open"] = (~missing_next & ~signals["is_next_open_one_word_limit"]).fillna(False).astype(bool)
    signals["is_tradable_next_close"] = signals["is_tradable_next_open"]
    signals["next_period_return"] = signals["next_period_return_before_trade_filter"].where(signals["is_tradable_next_open"])
    signals["trade_filter_reason"] = np.select(
        [
            missing_next.to_numpy(dtype=bool),
            signals["is_next_open_limit_up"].to_numpy(dtype=bool),
            signals["is_next_open_limit_down"].to_numpy(dtype=bool),
        ],
        [
            "missing_next_open_record_or_return",
            "next_open_limit_up",
            "next_open_limit_down",
        ],
        default="tradable_next_open",
    )
    signals["next_period_return_column_used"] = "sum_dretwd_tplus1_to_holding_end_open_entry"
    signals["rebalance_frequency"] = "daily"
    signals["holding_days"] = holding_days
    signals["holding_terminal_offset_days"] = terminal_offset_days

    tradable_count = signals.groupby("trade_date")["is_tradable_next_open"].transform("sum")
    signals["long_only_weight_equal"] = np.where(
        signals["is_tradable_next_open"] & tradable_count.gt(0),
        1.0 / tradable_count.replace(0, np.nan),
        0.0,
    )
    return sort_by_execution_date_and_stock(signals)


def build_tradable_open_buy_dates(signal_pool_with_returns: pd.DataFrame) -> pd.DataFrame:
    columns = ["交易日", "可买股票数", "对应信号日"]
    if signal_pool_with_returns.empty or "holding_start_trade_date" not in signal_pool_with_returns.columns:
        return pd.DataFrame(columns=columns)
    data = normalize_date_columns(signal_pool_with_returns, columns=["trade_date", "holding_start_trade_date"])
    tradable_col = "is_tradable_next_open" if "is_tradable_next_open" in data.columns else "is_tradable_next_close"
    if tradable_col not in data.columns:
        return pd.DataFrame(columns=columns)

    tradable = data.loc[data[tradable_col].fillna(False)].copy()
    tradable = tradable.loc[tradable["holding_start_trade_date"].notna()]
    if tradable.empty:
        return pd.DataFrame(columns=columns)

    def join_signal_dates(series: pd.Series) -> str:
        dates = pd.to_datetime(series, errors="coerce").dropna().dt.strftime("%Y-%m-%d")
        return ",".join(sorted(dates.unique()))

    summary = (
        tradable.groupby("holding_start_trade_date", dropna=False)
        .agg(
            可买股票数=("stock_code", "nunique"),
            对应信号日=("trade_date", join_signal_dates),
        )
        .reset_index()
        .rename(columns={"holding_start_trade_date": "交易日"})
    )
    summary["交易日"] = pd.to_datetime(summary["交易日"], errors="coerce").dt.strftime("%Y-%m-%d")
    return summary.sort_values("交易日").reset_index(drop=True)


def attach_additive_long_only_nav(portfolio: pd.DataFrame) -> pd.DataFrame:
    output = portfolio.copy()
    returns = pd.to_numeric(output["long_only_return"], errors="coerce").fillna(0.0)
    nav_start_values = []
    nav_end_values = []
    current_nav = 1.0
    for period_return in returns.to_numpy(dtype="float64"):
        nav_start_values.append(current_nav)
        current_nav = current_nav + period_return
        nav_end_values.append(current_nav)
    output["long_only_nav_start"] = nav_start_values
    output["long_only_nav"] = nav_end_values
    output["nav_update_rule"] = "long_only_nav_t_plus_1=long_only_nav_t+daily_long_only_return_t"
    output["nav_curve_semantics"] = "additive_nav_from_daily_long_only_return"
    return output


def calculate_long_only_returns(
    signal_pool_with_returns: pd.DataFrame,
    clean_data: pd.DataFrame,
    holding_days: int,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    clean = normalize_date_columns(clean_data).copy()
    clean["return_with_dividend"] = pd.to_numeric(clean["return_with_dividend"], errors="coerce")
    holding_days = max(int(holding_days), 1)

    calendar_all = clean[["trade_date"]].drop_duplicates().sort_values("trade_date").reset_index(drop=True)
    calendar_all["calendar_index"] = np.arange(len(calendar_all), dtype="int64")
    calendar_all["next_trade_date"] = calendar_all["trade_date"].shift(-1)
    calendar_all["holding_start_trade_date"] = calendar_all["next_trade_date"]
    calendar_all["holding_end_trade_date"] = calendar_all["trade_date"].shift(-holding_days)

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    portfolio = calendar_all.loc[calendar_all["trade_date"].between(start_ts, end_ts)].copy()

    stock_daily_returns = clean[["stock_code", "trade_date", "return_with_dividend"]].sort_values(KEY_COLUMNS).copy()
    stock_daily_returns["daily_stock_return"] = pd.to_numeric(stock_daily_returns["return_with_dividend"], errors="coerce")
    stock_daily_returns = stock_daily_returns[["stock_code", "trade_date", "daily_stock_return"]]

    if signal_pool_with_returns.empty:
        signal_agg = pd.DataFrame(columns=["trade_date"])
        active_agg = pd.DataFrame(columns=["trade_date"])
    else:
        signals = normalize_date_columns(signal_pool_with_returns).copy()
        signals = signals.loc[signals["trade_date"].between(start_ts, end_ts)].copy()
        tradable_col = "is_tradable_next_open" if "is_tradable_next_open" in signals.columns else "is_tradable_next_close"
        limit_col = "is_next_open_one_word_limit" if "is_next_open_one_word_limit" in signals.columns else "is_next_close_limit"
        entry_price_col = "next_open_price" if "next_open_price" in signals.columns else "next_close_price"

        tradable_mask = signals[tradable_col].fillna(False).astype(bool) if tradable_col in signals.columns else pd.Series(False, index=signals.index)
        limit_mask = signals[limit_col].fillna(False).astype(bool) if limit_col in signals.columns else pd.Series(False, index=signals.index)
        missing_next = pd.Series(False, index=signals.index)
        if entry_price_col in signals.columns:
            missing_next |= signals[entry_price_col].isna()
        if "next_period_return_before_trade_filter" in signals.columns:
            missing_next |= signals["next_period_return_before_trade_filter"].isna()

        signals["long_only_weight_equal"] = pd.to_numeric(signals["long_only_weight_equal"], errors="coerce").fillna(0.0)
        for col in ["breakout_return", "rolling_max_drawdown"]:
            if col not in signals.columns:
                signals[col] = np.nan
        for col in ["holding_start_trade_date", "holding_end_trade_date"]:
            if col not in signals.columns:
                signals[col] = pd.NaT

        signals["raw_signal_stock_count_flag"] = 1
        signals["tradable_signal_stock_count_flag"] = tradable_mask.astype("int64")
        signals["next_open_limit_excluded_flag"] = limit_mask.astype("int64")
        signals["missing_next_record_flag"] = missing_next.astype("int64")

        signal_agg = (
            signals.groupby("trade_date")
            .agg(
                raw_signal_stock_count=("raw_signal_stock_count_flag", "sum"),
                tradable_signal_stock_count=("tradable_signal_stock_count_flag", "sum"),
                next_close_limit_excluded_count=("next_open_limit_excluded_flag", "sum"),
                next_open_limit_excluded_count=("next_open_limit_excluded_flag", "sum"),
                missing_next_record_count=("missing_next_record_flag", "sum"),
                avg_breakout_return=("breakout_return", "mean"),
                avg_rolling_max_drawdown=("rolling_max_drawdown", "mean"),
                holding_start_trade_date_min=("holding_start_trade_date", "min"),
                holding_start_trade_date_max=("holding_start_trade_date", "max"),
                holding_end_trade_date_min=("holding_end_trade_date", "min"),
                holding_end_trade_date_max=("holding_end_trade_date", "max"),
            )
            .reset_index()
        )

        date_to_index = dict(zip(calendar_all["trade_date"], calendar_all["calendar_index"]))
        calendar_dates = calendar_all["trade_date"].to_numpy(dtype="datetime64[ns]")
        signals["entry_calendar_index"] = signals["holding_start_trade_date"].map(date_to_index)
        signals["entry_calendar_index"] = pd.to_numeric(signals["entry_calendar_index"], errors="coerce")

        tradable = signals.loc[
            tradable_mask
            & signals["long_only_weight_equal"].gt(0)
            & signals["entry_calendar_index"].notna()
        ].copy()
        tradable["entry_calendar_index"] = tradable["entry_calendar_index"].astype("int64")
        tradable["signal_trade_date"] = tradable["trade_date"]

        active_parts = []
        for active_holding_day in range(1, holding_days + 1):
            part = tradable[["signal_trade_date", "stock_code", "long_only_weight_equal", "entry_calendar_index"]].copy()
            part["active_holding_day"] = active_holding_day
            part["portfolio_calendar_index"] = part["entry_calendar_index"] + active_holding_day - 1
            part = part.loc[part["portfolio_calendar_index"].between(0, len(calendar_dates) - 1)].copy()
            if part.empty:
                continue
            part["trade_date"] = calendar_dates[part["portfolio_calendar_index"].to_numpy(dtype="int64")]
            active_parts.append(part)

        if active_parts:
            active_positions = pd.concat(active_parts, ignore_index=True)
            active_positions = active_positions.merge(stock_daily_returns, on=["stock_code", "trade_date"], how="left")
            active_positions["missing_active_stock_return_flag"] = active_positions["daily_stock_return"].isna().astype("int64")
            active_positions["daily_stock_return"] = active_positions["daily_stock_return"].fillna(0.0)
            active_positions["active_sleeve_weighted_return"] = active_positions["long_only_weight_equal"] * active_positions["daily_stock_return"]
            active_positions["portfolio_return_component"] = active_positions["active_sleeve_weighted_return"] / holding_days
            active_agg = (
                active_positions.groupby("trade_date")
                .agg(
                    active_sleeve_return_sum=("active_sleeve_weighted_return", "sum"),
                    long_only_return=("portfolio_return_component", "sum"),
                    active_stock_lot_count=("stock_code", "size"),
                    active_stock_count=("stock_code", "nunique"),
                    active_signal_sleeve_count=("signal_trade_date", "nunique"),
                    active_missing_stock_return_count=("missing_active_stock_return_flag", "sum"),
                    active_signal_trade_date_min=("signal_trade_date", "min"),
                    active_signal_trade_date_max=("signal_trade_date", "max"),
                )
                .reset_index()
            )
        else:
            active_agg = pd.DataFrame(columns=["trade_date"])

    portfolio = portfolio.merge(signal_agg, on="trade_date", how="left").merge(active_agg, on="trade_date", how="left")
    count_cols = [
        "raw_signal_stock_count",
        "tradable_signal_stock_count",
        "next_close_limit_excluded_count",
        "next_open_limit_excluded_count",
        "missing_next_record_count",
        "active_stock_lot_count",
        "active_stock_count",
        "active_signal_sleeve_count",
        "active_missing_stock_return_count",
    ]
    for col in count_cols:
        if col not in portfolio.columns:
            portfolio[col] = 0
        portfolio[col] = portfolio[col].fillna(0).astype("Int64")

    for col in ["active_sleeve_return_sum", "long_only_return"]:
        if col not in portfolio.columns:
            portfolio[col] = 0.0
        portfolio[col] = pd.to_numeric(portfolio[col], errors="coerce").fillna(0.0)
    for col in ["avg_breakout_return", "avg_rolling_max_drawdown"]:
        if col not in portfolio.columns:
            portfolio[col] = np.nan

    portfolio["long_only_return_sum"] = portfolio["long_only_return"]
    portfolio = attach_additive_long_only_nav(portfolio)
    portfolio["has_raw_signal"] = portfolio["raw_signal_stock_count"].gt(0)
    portfolio["has_tradable_signal"] = portfolio["tradable_signal_stock_count"].gt(0)
    portfolio["has_active_position"] = portfolio["active_signal_sleeve_count"].gt(0)
    portfolio["portfolio_invested_fraction"] = (
        pd.to_numeric(portfolio["active_signal_sleeve_count"], errors="coerce").fillna(0.0) / holding_days
    ).clip(lower=0.0, upper=1.0)
    portfolio["cash_reason"] = np.select(
        [
            ~portfolio["has_active_position"].to_numpy(dtype=bool),
            portfolio["has_active_position"].to_numpy(dtype=bool) & portfolio["portfolio_invested_fraction"].lt(1.0).to_numpy(dtype=bool),
        ],
        [
            "no_active_position_hold_cash",
            "partially_invested_rolling_sleeves",
        ],
        default="fully_invested_rolling_sleeves",
    )
    portfolio["rebalance_frequency"] = "daily_signal_generation_entry_next_open"
    portfolio["holding_days"] = holding_days
    portfolio["holding_terminal_offset_days"] = holding_days
    portfolio["weighting_scheme"] = f"rolling_{holding_days}_equal_capital_sleeves_equal_weight_stocks"
    portfolio["portfolio_model"] = "rolling_sleeve_sum_dretwd_additive_nav"
    return portfolio.sort_values("trade_date").reset_index(drop=True)


# ============================================================
# 5. 绩效、日志和图表
# ============================================================


def calculate_long_only_performance_attribution(
    long_only_returns: pd.DataFrame,
    holding_days: int,
    annualization_periods: int = TRADING_DAYS_PER_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    active_mask = (
        long_only_returns["has_active_position"].fillna(False).astype(bool)
        if "has_active_position" in long_only_returns.columns
        else long_only_returns["has_tradable_signal"].fillna(False).astype(bool)
    )
    metrics = BASE.calculate_additive_nav_metrics(
        long_only_returns["long_only_return"],
        long_only_returns.get("long_only_nav"),
        annualization_periods,
    )
    performance_summary = pd.DataFrame(
        [
            {
                "portfolio": "factor2_2conditions_long_only",
                **metrics,
                "annualization_periods_per_year": annualization_periods,
                "holding_days": holding_days,
                "cash_period_count": int((~active_mask).sum()),
                "invested_period_count": int(active_mask.sum()),
                "cash_period_ratio": float((~active_mask).mean()) if len(long_only_returns) else math.nan,
                "avg_raw_signal_stock_count": float(pd.to_numeric(long_only_returns["raw_signal_stock_count"], errors="coerce").mean()) if len(long_only_returns) else math.nan,
                "avg_tradable_signal_stock_count": float(pd.to_numeric(long_only_returns["tradable_signal_stock_count"], errors="coerce").mean()) if len(long_only_returns) else math.nan,
                "avg_active_signal_sleeve_count": float(pd.to_numeric(long_only_returns.get("active_signal_sleeve_count", pd.Series(dtype="float64")), errors="coerce").mean()) if len(long_only_returns) else math.nan,
                "avg_portfolio_invested_fraction": float(pd.to_numeric(long_only_returns.get("portfolio_invested_fraction", pd.Series(dtype="float64")), errors="coerce").mean()) if len(long_only_returns) else math.nan,
            }
        ]
    )

    drawdown = BASE.calculate_drawdown_series(pd.to_numeric(long_only_returns["long_only_nav"], errors="coerce"))
    drawdown["trade_date"] = long_only_returns["trade_date"].values
    drawdown["portfolio"] = "factor2_2conditions_long_only"

    yearly_rows = []
    temp = long_only_returns[["trade_date", "long_only_return"]].copy()
    temp["year"] = pd.to_datetime(temp["trade_date"]).dt.year
    for year, one_year in temp.groupby("year"):
        year_return = float(pd.to_numeric(one_year["long_only_return"], errors="coerce").dropna().sum())
        yearly_rows.append({"portfolio": "factor2_2conditions_long_only", "year": year, "year_return": year_return, "observation_count": len(one_year)})
    return performance_summary, drawdown, pd.DataFrame(yearly_rows)


def long_only_step_log(
    raw_data: pd.DataFrame,
    clean_data: pd.DataFrame,
    all_signal: pd.DataFrame,
    signal_pool: pd.DataFrame,
    signal_pool_with_returns: pd.DataFrame,
    long_only_returns: pd.DataFrame,
) -> pd.DataFrame:
    tradable_col = "is_tradable_next_open" if "is_tradable_next_open" in signal_pool_with_returns.columns else "is_tradable_next_close"
    tradable_signals = (
        signal_pool_with_returns.loc[signal_pool_with_returns[tradable_col].fillna(False)]
        if tradable_col in signal_pool_with_returns.columns
        else pd.DataFrame()
    )
    rows = [
        ["sql_raw_market_data", len(raw_data), safe_nunique(raw_data, "Stkcd"), safe_nunique(raw_data, "Trddt")],
        ["cleaned_stock_pool", len(clean_data), safe_nunique(clean_data, "stock_code"), safe_nunique(clean_data, "trade_date")],
        ["all_two_condition_records", len(all_signal), safe_nunique(all_signal, "stock_code"), safe_nunique(all_signal, "trade_date")],
        ["screened_long_only_signal_pool", len(signal_pool), safe_nunique(signal_pool, "stock_code"), safe_nunique(signal_pool, "trade_date")],
        ["signal_pool_with_forward_returns", len(signal_pool_with_returns), safe_nunique(signal_pool_with_returns, "stock_code"), safe_nunique(signal_pool_with_returns, "trade_date")],
        ["tradable_long_only_signal_records", int(tradable_signals[tradable_col].sum()) if tradable_col in tradable_signals else 0, safe_nunique(tradable_signals, "stock_code"), safe_nunique(tradable_signals, "trade_date")],
        ["long_only_return_rows", len(long_only_returns), None, safe_nunique(long_only_returns, "trade_date")],
    ]
    return pd.DataFrame(rows, columns=["step", "rows", "entity_count", "trade_date_count"])


def build_trade_detail_message(message: str, output_path: Path, trade_date: pd.Timestamp) -> Path:
    import matplotlib.pyplot as plt

    write_csv(
        pd.DataFrame([{"查询日期": trade_date.strftime("%Y-%m-%d"), "说明": message}]),
        output_path.with_suffix(".csv"),
    )
    BASE.configure_matplotlib_chinese_font(plt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 3.6), dpi=180)
    ax.axis("off")
    ax.text(0.5, 0.58, message, ha="center", va="center", fontsize=15, wrap=True)
    ax.text(0.5, 0.28, f"查询日期：{trade_date.strftime('%Y-%m-%d')}", ha="center", va="center", fontsize=11, color="#666666")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def output_trade_detail_image(
    trade_detail_date: pd.Timestamp | None,
    clean_data: pd.DataFrame,
    signal_pool_with_returns: pd.DataFrame,
    output_dir: Path,
    high_lookback_days: int,
) -> Path | None:
    if trade_detail_date is None:
        return None

    trade_date = pd.Timestamp(trade_detail_date).normalize()
    image_path = output_dir / f"15_factor2_2conditions_long_only_trade_detail_{trade_date.strftime('%Y%m%d')}.png"
    calendar = normalize_date_columns(clean_data)[["trade_date"]].drop_duplicates().sort_values("trade_date")
    calendar_dates = pd.to_datetime(calendar["trade_date"]).dt.normalize()

    if not calendar_dates.eq(trade_date).any():
        message = "该日期不是清洗后股票池的交易日，请输入正确交易日期。"
        print(message)
        return build_trade_detail_message(message, image_path, trade_date)

    if len(calendar_dates) > high_lookback_days + 1:
        first_available_trade_date = pd.Timestamp(calendar_dates.iloc[high_lookback_days + 1]).normalize()
        if trade_date < first_available_trade_date:
            message = f"该日期早于策略首个可用交易执行日 {first_available_trade_date.strftime('%Y-%m-%d')}。"
            print(message)
            return build_trade_detail_message(message, image_path, trade_date)

    signals = normalize_date_columns(signal_pool_with_returns).copy()
    if signals.empty:
        day_trades = pd.DataFrame()
    else:
        tradable_col = "is_tradable_next_open" if "is_tradable_next_open" in signals.columns else "is_tradable_next_close"
        tradable_mask = signals[tradable_col].fillna(False) if tradable_col in signals.columns else pd.Series(False, index=signals.index)
        day_trades = signals.loc[
            signals["holding_start_trade_date"].dt.normalize().eq(trade_date)
            & tradable_mask
        ].copy()

    if day_trades.empty:
        message = "该交易日没有符合条件且可在开盘买入的股票。"
        print(message)
        return build_trade_detail_message(message, image_path, trade_date)

    display = sort_by_execution_date_and_stock(day_trades)
    display["买入日期"] = display["holding_start_trade_date"].dt.strftime("%Y-%m-%d")
    display["信号日期"] = display["trade_date"].dt.strftime("%Y-%m-%d")
    display["股票代码"] = display["stock_code"].astype(str)
    display["买入开盘价"] = pd.to_numeric(display["holding_entry_open_price"], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    display["组合权重"] = pd.to_numeric(display["long_only_weight_equal"], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    display["T日收盘价"] = pd.to_numeric(display["signal_close_price"], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    display["前高日期"] = pd.to_datetime(display["prior_high_trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    display["前高收盘价"] = pd.to_numeric(display["prior_high_close_price"], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    display["突破幅度"] = pd.to_numeric(display["breakout_return"], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    display["回撤窗口起点"] = pd.to_datetime(display["drawdown_window_start"], errors="coerce").dt.strftime("%Y-%m-%d")
    display["最大回撤"] = pd.to_numeric(display["rolling_max_drawdown"], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    display["回撤高点日期"] = pd.to_datetime(display["drawdown_peak_trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    display["回撤低点日期"] = pd.to_datetime(display["drawdown_trough_trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    display["持有期收益"] = pd.to_numeric(display["next_period_return_before_trade_filter"], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    display["持有结束日期"] = pd.to_datetime(display["holding_end_trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    table_columns = ["买入日期", "信号日期", "股票代码", "买入开盘价", "组合权重", "T日收盘价", "前高日期", "前高收盘价", "突破幅度", "回撤窗口起点", "最大回撤", "回撤高点日期", "回撤低点日期", "持有期收益", "持有结束日期"]
    table_data = display[table_columns]
    csv_path = image_path.with_suffix(".csv")
    write_csv(table_data, csv_path)

    print(f"{trade_date.strftime('%Y-%m-%d')} 开盘买入股票明细（共 {len(table_data)} 只）：")
    print(table_data.head(50).to_string(index=False))
    if len(table_data) > 50:
        print(f"仅在终端显示前 50 行，完整明细见：{csv_path}")

    import matplotlib.pyplot as plt

    BASE.configure_matplotlib_chinese_font(plt)
    row_count = len(table_data)
    fig_height = min(max(3.8, 1.3 + 0.38 * (row_count + 1)), 20)
    fig, ax = plt.subplots(figsize=(20.5, fig_height), dpi=180)
    ax.axis("off")
    ax.set_title(f"{trade_date.strftime('%Y-%m-%d')} 开盘买入股票明细（共 {row_count} 只）", fontsize=16, pad=16)
    table = ax.table(cellText=table_data.values, colLabels=table_columns, loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.28)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f4e79")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f4f7fb")
        cell.set_edgecolor("#d0d7de")
    fig.tight_layout()
    image_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"已输出交易明细CSV：{csv_path}")
    print(f"已输出交易明细图片：{image_path}")
    return image_path


# ============================================================
# 6. 缓存和主流程
# ============================================================


def required_long_only_result_files(output_dir: Path) -> list[Path]:
    return [
        output_dir / "03_factor2_2conditions_long_only_all_stocks.csv",
        output_dir / "03_factor2_2conditions_long_only_signal_pool.csv",
        output_dir / "04_factor2_2conditions_long_only_signal_pool_with_forward_returns.csv",
        output_dir / "05_factor2_2conditions_long_only_holding_period_returns.csv",
        output_dir / "05_factor2_2conditions_long_only_nav_curve.png",
        output_dir / "06_factor2_2conditions_long_only_portfolio_performance_table.csv",
        output_dir / "06_factor2_2conditions_long_only_portfolio_performance_detail.csv",
        output_dir / "07_factor2_2conditions_long_only_drawdown_series.csv",
        output_dir / "08_factor2_2conditions_long_only_yearly_performance.csv",
        output_dir / "factor2_2conditions_filter_step_summary.csv",
        output_dir / "run_summary.csv",
    ]


def read_existing_run_summary(output_dir: Path) -> dict[str, str]:
    return BASE.read_existing_run_summary(output_dir)


def has_complete_long_only_result_cache(output_dir: Path, args: argparse.Namespace, market_types: set[int]) -> bool:
    if not all(path.exists() for path in required_long_only_result_files(output_dir)):
        return False
    summary = read_existing_run_summary(output_dir)
    expected = {
        "factor_version": FACTOR_VERSION,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "holding_days": str(args.holding_days),
        "selected_market_types": ",".join(str(item) for item in sorted(market_types)),
        "high_lookback_days": str(args.high_lookback_days),
        "drawdown_lookback_days": str(args.drawdown_lookback_days),
        "drawdown_threshold": str(args.drawdown_threshold),
        "min_float_market_value_dsmvosd_unit": str(args.min_float_market_value),
    }
    return all(str(summary.get(key, "")) == value for key, value in expected.items())


def print_reused_long_only_result_cache(output_dir: Path) -> None:
    print("发现完整 factor2_2conditions 结果缓存，直接复用并跳过本次重算。")
    print(f"结果目录：{output_dir}")
    for path in required_long_only_result_files(output_dir):
        print(f"已存在：{path.name}")


def main() -> None:
    args = parse_args()
    market_types = parse_market_types(args.market_types)
    trade_detail_date = parse_optional_trade_detail_date(args.trade_detail_date)
    output_root: Path = args.output_dir
    experiment_folder_name = build_long_only_experiment_folder_name(
        args.high_lookback_days,
        args.drawdown_lookback_days,
        args.holding_days,
        market_types,
    )
    output_dir = output_root / experiment_folder_name

    global LOCAL_RAW_DATA_CACHE_DIR, LOCAL_CLEAN_DATA_CACHE_DIR
    LOCAL_RAW_DATA_CACHE_DIR = output_root / "_local_data_cache"
    LOCAL_CLEAN_DATA_CACHE_DIR = output_root / "_local_data_cache"
    BASE.LOCAL_RAW_DATA_CACHE_DIR = LOCAL_RAW_DATA_CACHE_DIR
    BASE.LOCAL_CLEAN_DATA_CACHE_DIR = LOCAL_CLEAN_DATA_CACHE_DIR

    output_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_tsv_path = output_root / "_raw_market_data_for_factor2_2conditions.tsv"
    primary_cache_dir = output_root / "_local_data_cache"
    raw_cache_path = build_raw_data_cache_path_in(primary_cache_dir, args.start_date, args.end_date, market_types)
    clean_cache_paths = build_clean_data_cache_paths_in(
        primary_cache_dir,
        args.start_date,
        args.end_date,
        args.min_float_market_value,
        market_types,
    )
    fallback_raw_cache_path = build_raw_data_cache_path_in(FALLBACK_DATA_CACHE_DIR, args.start_date, args.end_date, market_types)
    fallback_clean_cache_paths = build_clean_data_cache_paths_in(
        FALLBACK_DATA_CACHE_DIR,
        args.start_date,
        args.end_date,
        args.min_float_market_value,
        market_types,
    )
    if (
        not clean_cache_paths["clean_data"].exists()
        and fallback_clean_cache_paths["clean_data"].exists()
        and not args.refresh_clean_data
    ):
        clean_cache_paths = fallback_clean_cache_paths

    refresh_requested = args.refresh_local_data or args.refresh_clean_data
    if has_complete_long_only_result_cache(output_dir, args, market_types) and not args.force_recalculate_signal and not refresh_requested:
        print_reused_long_only_result_cache(output_dir)
        if trade_detail_date is not None:
            if clean_cache_paths["clean_data"].exists():
                clean_data, _, _, _ = BASE.load_clean_market_data_from_cache(clean_cache_paths)
            else:
                cached_returns_path = output_dir / "05_factor2_2conditions_long_only_holding_period_returns.csv"
                clean_data = pd.read_csv(cached_returns_path, usecols=["trade_date"])
            cached_signal_path = output_dir / "04_factor2_2conditions_long_only_signal_pool_with_forward_returns.csv"
            signal_pool_with_returns = pd.read_csv(cached_signal_path, dtype={"stock_code": str})
            trade_detail_image_path = output_trade_detail_image(
                trade_detail_date,
                clean_data=clean_data,
                signal_pool_with_returns=signal_pool_with_returns,
                output_dir=output_dir,
                high_lookback_days=args.high_lookback_days,
            )
            trade_detail_csv_path = (
                trade_detail_image_path.with_suffix(".csv")
                if trade_detail_image_path is not None and trade_detail_image_path.with_suffix(".csv").exists()
                else None
            )
            if trade_detail_csv_path is not None:
                print(f"交易明细CSV：{trade_detail_csv_path}")
            if trade_detail_image_path is not None:
                print(f"交易明细图片：{trade_detail_image_path}")
        return

    print("1/7 正在读取或生成清洗后行情数据...")
    raw_data = pd.DataFrame()
    raw_data_source = "not_loaded_because_clean_cache_reused"
    if clean_cache_paths["clean_data"].exists() and not args.refresh_clean_data:
        print(f"读取本地清洗行情缓存：{clean_cache_paths['clean_data']}")
        clean_data, cleaning_log, missing_summary, exclusion_summary = BASE.load_clean_market_data_from_cache(clean_cache_paths)
        clean_data_source = "local_clean_cache" if clean_cache_paths["clean_data"].is_relative_to(primary_cache_dir) else "fallback_factor2_longonly_clean_cache"
    else:
        if args.refresh_clean_data:
            print("收到 --refresh-clean-data，忽略清洗缓存并重新清洗。")
        else:
            print("未找到清洗缓存，将优先读取原始行情缓存；若原始缓存也不存在才连接 MySQL。")
        if (
            not raw_cache_path.exists()
            and fallback_raw_cache_path.exists()
            and not args.refresh_local_data
        ):
            BASE.LOCAL_RAW_DATA_CACHE_DIR = FALLBACK_DATA_CACHE_DIR
            raw_cache_path = fallback_raw_cache_path
        else:
            BASE.LOCAL_RAW_DATA_CACHE_DIR = primary_cache_dir
        raw_data, raw_cache_path, raw_data_source = BASE.load_raw_market_data_with_cache(
            args.start_date,
            args.end_date,
            market_types,
            temp_tsv_path,
            args.refresh_local_data,
            args.keep_temp,
        )
        clean_data, cleaning_log, missing_summary, exclusion_summary = BASE.clean_market_data(
            raw_data,
            min_float_market_value=args.min_float_market_value,
            market_types=market_types,
        )
        BASE.write_clean_market_data_cache(clean_data, cleaning_log, missing_summary, exclusion_summary, clean_cache_paths)
        clean_data_source = "fresh_cleaned_and_cached"

    print("2/7 正在执行两条件信号筛选...")
    raw_signal_all_dates, missing_signal_all_dates = calculate_raw_two_condition_signals(
        clean_data,
        high_lookback_days=args.high_lookback_days,
        drawdown_lookback_days=args.drawdown_lookback_days,
        drawdown_threshold=args.drawdown_threshold,
    )
    all_signal_all_dates = build_all_signals(raw_signal_all_dates, missing_signal_all_dates)

    start_ts = pd.to_datetime(args.start_date)
    end_ts = pd.to_datetime(args.end_date)
    raw_date_window = raw_signal_all_dates.loc[raw_signal_all_dates["trade_date"].between(start_ts, end_ts)].copy()
    all_date_window = all_signal_all_dates.loc[all_signal_all_dates["trade_date"].between(start_ts, end_ts)].copy()
    missing_date_window = missing_signal_all_dates.loc[missing_signal_all_dates["trade_date"].between(start_ts, end_ts)].copy()
    full_history_mask = (
        pd.to_numeric(all_date_window["full_history_valid_days"], errors="coerce").ge(args.high_lookback_days)
        & pd.to_numeric(all_date_window["drawdown_lookback_valid_days"], errors="coerce").eq(args.drawdown_lookback_days)
    )
    raw_history_mask = (
        pd.to_numeric(raw_date_window["full_history_valid_days"], errors="coerce").ge(args.high_lookback_days)
        & pd.to_numeric(raw_date_window["drawdown_lookback_valid_days"], errors="coerce").eq(args.drawdown_lookback_days)
    )
    missing_history_mask = (
        pd.to_numeric(missing_date_window["full_history_valid_days"], errors="coerce").ge(args.high_lookback_days)
        & pd.to_numeric(missing_date_window["drawdown_lookback_valid_days"], errors="coerce").eq(args.drawdown_lookback_days)
    )
    signal_pool = raw_date_window.loc[raw_history_mask].copy()
    all_signal = all_date_window.loc[full_history_mask].copy()
    insufficient_history_rows = int((~full_history_mask).sum())

    print("3/7 正在匹配 T+1 开盘可交易状态和未来持有期收益...")
    signal_pool_with_returns = attach_long_only_execution_and_returns(
        signal_pool,
        clean_data=clean_data,
        holding_days=args.holding_days,
    )

    trade_detail_image_path = output_trade_detail_image(
        trade_detail_date,
        clean_data=clean_data,
        signal_pool_with_returns=signal_pool_with_returns,
        output_dir=output_dir,
        high_lookback_days=args.high_lookback_days,
    )
    trade_detail_csv_path = (
        trade_detail_image_path.with_suffix(".csv")
        if trade_detail_image_path is not None and trade_detail_image_path.with_suffix(".csv").exists()
        else None
    )

    print("4/7 正在计算 long-only 滚动持仓每日收益和 NAV...")
    long_only_returns = calculate_long_only_returns(
        signal_pool_with_returns,
        clean_data=clean_data,
        holding_days=args.holding_days,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    print("5/7 正在计算绩效和摘要...")
    performance_summary, drawdown_series, yearly_performance = calculate_long_only_performance_attribution(
        long_only_returns,
        args.holding_days,
    )
    portfolio_performance_table, portfolio_performance_detail = BASE.build_long_only_performance_tables(performance_summary)
    screen_summary = screening_summary(all_signal)
    screen_filter_summary = screening_filter_step_summary(
        all_signal,
        args.high_lookback_days,
        args.drawdown_lookback_days,
        args.drawdown_threshold,
    )
    step_log = long_only_step_log(
        raw_data,
        clean_data,
        all_signal,
        signal_pool,
        signal_pool_with_returns,
        long_only_returns,
    )

    print("6/7 正在输出 CSV 文件...")
    write_csv(format_signal_dates(all_signal), output_dir / "03_factor2_2conditions_long_only_all_stocks.csv")
    write_csv(format_signal_dates(signal_pool), output_dir / "03_factor2_2conditions_long_only_signal_pool.csv")
    write_csv(format_signal_dates(signal_pool_with_returns), output_dir / "04_factor2_2conditions_long_only_signal_pool_with_forward_returns.csv")
    write_csv(build_tradable_open_buy_dates(signal_pool_with_returns), output_dir / "04_factor2_2conditions_long_only_tradable_open_buy_dates.csv")
    write_csv(format_signal_dates(long_only_returns), output_dir / "05_factor2_2conditions_long_only_holding_period_returns.csv")
    write_csv(portfolio_performance_table, output_dir / "06_factor2_2conditions_long_only_portfolio_performance_table.csv")
    write_csv(portfolio_performance_detail, output_dir / "06_factor2_2conditions_long_only_portfolio_performance_detail.csv")
    write_csv(format_signal_dates(performance_summary), output_dir / "factor2_2conditions_long_only_performance_summary.csv")
    write_csv(format_signal_dates(drawdown_series), output_dir / "07_factor2_2conditions_long_only_drawdown_series.csv")
    write_csv(format_signal_dates(yearly_performance), output_dir / "08_factor2_2conditions_long_only_yearly_performance.csv")
    write_csv(screen_summary, output_dir / "factor2_2conditions_screening_summary.csv")
    write_csv(screen_filter_summary, output_dir / "factor2_2conditions_filter_step_summary.csv")
    write_csv(cleaning_log, output_dir / "cleaning_step_log.csv")
    write_csv(exclusion_summary, output_dir / "cleaning_exclusion_reason_summary.csv")
    write_csv(missing_summary, output_dir / "missing_value_ffill_summary.csv")
    write_csv(step_log, output_dir / "factor2_2conditions_long_only_step_log.csv")
    nav_curve_path = BASE.plot_long_only_nav_curve(long_only_returns, output_dir / "05_factor2_2conditions_long_only_nav_curve.png")
    print(f"已输出 long-only NAV 曲线：{nav_curve_path}")

    print("7/7 正在输出运行摘要...")
    run_summary = pd.DataFrame(
        [
            ["factor_name", "factor2_2conditions"],
            ["factor_version", FACTOR_VERSION],
            ["strategy_type", "two_condition_long_only_sum_dretwd_additive_nav"],
            ["start_date", args.start_date],
            ["end_date", args.end_date],
            ["rebalance_frequency", "daily_signal_generation_entry_next_open"],
            ["high_lookback_days", args.high_lookback_days],
            ["high_breakout_rule", f"close_T > max(close_T-1_to_close_T-{args.high_lookback_days}) using raw unadjusted close"],
            ["drawdown_lookback_days", args.drawdown_lookback_days],
            ["drawdown_threshold", args.drawdown_threshold],
            ["drawdown_rule", f"min(close/running_high-1) over T-{args.drawdown_lookback_days - 1}..T raw close < -{args.drawdown_threshold:g}"],
            ["factor_formula", "none"],
            ["ranking_rule", "none_all_tradable_signals_equal_weight"],
            ["holding_days", args.holding_days],
            ["holding_unit", "trading_days"],
            ["entry_rule", "next_trade_date_open"],
            ["exit_rule", f"sum_Dretwd_through_T+{args.holding_days}"],
            ["return_horizon", f"T+1_Dretwd_to_T+{args.holding_days}_Dretwd_sum"],
            ["return_calculation_source", "Dretwd_return_with_dividend_summed_not_compounded"],
            ["forward_return_rule", f"sum_Dretwd_from_T+1_to_T+{args.holding_days}"],
            ["daily_nav_return_rule", "daily_active_positions_weighted_Dretwd"],
            ["nav_update_rule", "long_only_nav_t_plus_1=long_only_nav_t+daily_long_only_return_t"],
            ["nav_curve_semantics", "additive_nav_from_daily_long_only_return"],
            ["holding_terminal_offset_days", args.holding_days],
            ["weighting_scheme", f"rolling_{args.holding_days}_equal_capital_sleeves_equal_weight_stocks"],
            ["portfolio_model", "rolling_sleeve_sum_dretwd_additive_nav"],
            ["short_side", "none"],
            ["cash_return_when_no_active_position", "0"],
            ["selected_market_types", ",".join(str(item) for item in sorted(market_types))],
            ["market_types_folder_tag", market_types_to_tag(market_types)],
            ["experiment_folder_name", experiment_folder_name],
            ["output_root", str(output_root)],
            ["output_dir", str(output_dir)],
            ["raw_data_source", raw_data_source],
            ["local_raw_data_cache_path", str(raw_cache_path)],
            ["clean_data_source", clean_data_source],
            ["local_clean_data_cache_path", str(clean_cache_paths["clean_data"])],
            ["min_float_market_value_dsmvosd_unit", args.min_float_market_value],
            ["cleaned_rows", len(clean_data)],
            ["insufficient_history_rows", insufficient_history_rows],
            ["all_two_condition_rows", len(all_signal)],
            ["long_only_signal_rows_before_trade_filter", len(signal_pool)],
            ["signal_rows_with_forward_returns", len(signal_pool_with_returns)],
            ["tradable_long_only_signal_rows", int(signal_pool_with_returns["is_tradable_next_open"].sum()) if "is_tradable_next_open" in signal_pool_with_returns else 0],
            ["long_only_return_rows", len(long_only_returns)],
            ["cash_period_count", int((~long_only_returns["has_active_position"].fillna(False)).sum()) if "has_active_position" in long_only_returns else 0],
            ["invested_period_count", int(long_only_returns["has_active_position"].fillna(False).sum()) if "has_active_position" in long_only_returns else 0],
            ["trade_detail_date", "" if trade_detail_date is None else trade_detail_date.strftime("%Y-%m-%d")],
            ["trade_detail_csv_path", "" if trade_detail_csv_path is None else str(trade_detail_csv_path)],
            ["trade_detail_image_path", "" if trade_detail_image_path is None else str(trade_detail_image_path)],
            ["nav_curve_path", str(nav_curve_path)],
            ["portfolio_performance_table_rows", len(portfolio_performance_table)],
        ],
        columns=["metric", "value"],
    )
    write_csv(run_summary, output_dir / "run_summary.csv")

    print("factor2_2conditions 处理完成。")
    print(f"输出目录：{output_dir}")
    print(f"两条件信号记录数：{len(signal_pool):,}")
    tradable_rows = int(signal_pool_with_returns["is_tradable_next_open"].sum()) if "is_tradable_next_open" in signal_pool_with_returns else 0
    print(f"次日开盘可交易信号记录数：{tradable_rows:,}")
    print(f"long-only 每日 NAV 记录数：{len(long_only_returns):,}")
    if trade_detail_csv_path is not None:
        print(f"交易明细CSV：{trade_detail_csv_path}")
    if trade_detail_image_path is not None:
        print(f"交易明细图片：{trade_detail_image_path}")
    print("组合绩效表：")
    print(portfolio_performance_table.to_string(index=False))


if __name__ == "__main__":
    main()
