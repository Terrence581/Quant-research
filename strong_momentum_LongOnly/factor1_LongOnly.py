from __future__ import annotations

import argparse
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. 基础配置
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = Path(r"D:\Desktop\CINDA qr\factor1_strong_momentum\factor1_LongOnly_output")
LOCAL_RAW_DATA_CACHE_DIR = DEFAULT_OUTPUT_ROOT / "_local_data_cache"
LOCAL_CLEAN_DATA_CACHE_DIR = DEFAULT_OUTPUT_ROOT / "_local_data_cache"

MYSQL_EXE = os.environ.get("MYSQL_EXE", "mysql")
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD")
MYSQL_DATABASE = "1626_astock"
MARKET_TABLE = "all_market_data"

# 回测时间、持有期和股票池基础约束。
DEFAULT_START_DATE = "2018-01-01"       # 正式回测起始日
DEFAULT_END_DATE = "2021-12-31"
DEFAULT_HOLDING_DAYS = 63
DEFAULT_GROUP_NUM = 1  # long-only 只做多，不需要分组
DEFAULT_MIN_FLOAT_MARKET_VALUE = 5_000_000.0
DEFAULT_MIN_CROSS_SECTION_STOCK_COUNT = 1  # long-only 版本不要求截面分组最低数量

# 五重筛选的核心参数：市场范围、强上涨、回撤、lookback peak 和 peak 前波动约束。
DEFAULT_MARKET_TYPES = {1, 4}
DEFAULT_RISE_THRESHOLD = 0.50
DEFAULT_DRAWDOWN_THRESHOLD = 0.10
DEFAULT_LOOKBACK_DAYS = 189
DEFAULT_MAX_PEAK_AGE_DAYS = 63
DEFAULT_ZSCORE_WINDOW = 63        # 每只股票计算 ChangeRatio 时序 z-score 的历史窗口，要求 peak 前有完整窗口
DEFAULT_ZSCORE_LIMIT = 2.5        # peak 前窗口内 abs(z-score) 最大允许值
DEFAULT_ZSCORE_DDOF = 0           # 标准差自由度
DEFAULT_TRADE_DETAIL_DATE = None  # Example: "2019-07-11"; command line --trade-date can override it.
FACTOR_VERSION = "factor1_LongOnly_first_window_high_peak_dretwd_tplus1_open_additive_nav_pre_peak_time_zscore_v21"

TRADING_DAYS_PER_YEAR = 252
PRICE_COMPARE_TOLERANCE = 1e-6

# A 股市场类型：1=上证A股，4=深证A股，16=创业板，32=科创板，64=北证A股。
A_SHARE_MARKET_TYPES = {1, 4, 16, 32, 64}

# Trdsta 中带 ST、*ST、SST、S*ST、GST、G*ST、UST、U*ST、NST、N*ST 以及 PT 的状态。
ST_OR_PT_STATUS_VALUES = {2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16}
LIMIT_UP_OR_DOWN_VALUES = {-1, 1}

SOURCE_COLUMNS = [
    "Stkcd",
    "Trddt",
    "Opnprc",
    "Hiprc",
    "Loprc",
    "Clsprc",
    "Dnshrtrd",
    "Dnvaltrd",
    "Dsmvosd",
    "Dsmvtll",
    "Dretwd",
    "Dretnd",
    "Adjprcwd",
    "Adjprcnd",
    "Markettype",
    "Capchgdt",
    "Trdsta",
    "Ahshrtrd_D",
    "Ahvaltrd_D",
    "PreClosePrice",
    "ChangeRatio",
    "LimitDown",
    "LimitUp",
    "LimitStatus",
]

COLUMN_RENAME_MAP = {
    "Stkcd": "stock_code",
    "Trddt": "trade_date",
    "Opnprc": "open_price",
    "Hiprc": "high_price",
    "Loprc": "low_price",
    "Clsprc": "close_price",
    "Dnshrtrd": "volume",
    "Dnvaltrd": "amount",
    "Dsmvosd": "float_market_value",
    "Dsmvtll": "total_market_value",
    "Dretwd": "return_with_dividend",
    "Dretnd": "return_without_dividend",
    "Adjprcwd": "adj_close_with_dividend",
    "Adjprcnd": "adj_close_without_dividend",
    "Markettype": "market_type",
    "Capchgdt": "capital_change_date",
    "Trdsta": "trade_status",
    "Ahshrtrd_D": "after_hours_volume",
    "Ahvaltrd_D": "after_hours_amount",
    "PreClosePrice": "pre_close_price",
    "ChangeRatio": "change_ratio",
    "LimitDown": "limit_down_price",
    "LimitUp": "limit_up_price",
    "LimitStatus": "limit_status",
}

NUMERIC_COLUMNS = [
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "volume",
    "amount",
    "float_market_value",
    "total_market_value",
    "return_with_dividend",
    "return_without_dividend",
    "adj_close_with_dividend",
    "adj_close_without_dividend",
    "market_type",
    "trade_status",
    "after_hours_volume",
    "after_hours_amount",
    "pre_close_price",
    "change_ratio",
    "limit_down_price",
    "limit_up_price",
    "limit_status",
]

KEY_COLUMNS = ["stock_code", "trade_date"]
FACTOR_SORT_COLUMN = "momentum_raw"
DATE_COLUMNS = [
    "trade_date",
    "capital_change_date",
    "momentum_start_date",
    "momentum_end_date",
    "peak_trade_date",
    "pre_peak_low_trade_date",
    "post_peak_low_trade_date",
    "pre_peak_zscore_window_start",
    "lookback_window_start",
    "max_peak_age_start",
    "next_trade_date",
    "holding_start_trade_date",
    "holding_end_trade_date",
]
NO_FORWARD_FILL_COLUMNS = {"limit_status"}
FILL_COLUMNS = [
    col
    for col in COLUMN_RENAME_MAP.values()
    if col not in KEY_COLUMNS and col not in NO_FORWARD_FILL_COLUMNS
]


# ============================================================
# 2. 参数和通用工具
# ============================================================


def parse_market_types(text: str) -> set[int]:
    values: set[int] = set()
    for token in str(text).replace("，", ",").split(","):
        token = token.strip()
        if token:
            values.add(int(token))
    return values or set(DEFAULT_MARKET_TYPES)


def market_types_to_tag(market_types: set[int]) -> str:
    return "-".join(str(item) for item in sorted(market_types)) if market_types else "na"


def build_experiment_folder_name(
    lookback_days: int,
    holding_days: int,
    group_num: int,
    market_types: set[int],
) -> str:
    """Use the key backtest parameters to build one stable experiment folder name."""

    return f"lb{lookback_days}_hd{holding_days}_g{group_num}_mkt{market_types_to_tag(market_types)}"


def build_default_experiment_dir(output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    return output_root / build_experiment_folder_name(
        DEFAULT_LOOKBACK_DAYS,
        DEFAULT_HOLDING_DAYS,
        DEFAULT_GROUP_NUM,
        DEFAULT_MARKET_TYPES,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "计算 A 股强势因子策略1 long-only 版本：五重路径筛选后，"
            "T+1 开盘等权买入可交易信号，持有期内按每日 Dretwd 计算真实组合 NAV；"
            "未占用资金持有现金，收益按 0 处理。"
        )
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="正式回测起始日期，格式 YYYY-MM-DD。")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="测试结束日期，格式 YYYY-MM-DD。")
    parser.add_argument("--min-float-market-value", type=float, default=DEFAULT_MIN_FLOAT_MARKET_VALUE)
    parser.add_argument("--rise-threshold", type=float, default=DEFAULT_RISE_THRESHOLD)
    parser.add_argument("--drawdown-threshold", type=float, default=DEFAULT_DRAWDOWN_THRESHOLD)
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--max-peak-age-days", type=int, default=DEFAULT_MAX_PEAK_AGE_DAYS)
    parser.add_argument("--holding-days", type=int, default=DEFAULT_HOLDING_DAYS, help="持有期交易日数，long-only 默认 21。")
    parser.add_argument("--group-num", type=int, default=DEFAULT_GROUP_NUM, help="兼容旧脚本参数，long-only 版本不再使用分组。")
    parser.add_argument("--min-cross-section-stock-count", type=int, default=DEFAULT_MIN_CROSS_SECTION_STOCK_COUNT)
    parser.add_argument(
        "--trade-detail-date",
        "--trade-date",
        default=DEFAULT_TRADE_DETAIL_DATE,
        help="输出指定真实交易执行日的买入股票明细图片，格式 YYYY-MM-DD。该日对应 T+1 开盘买入。",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--refresh-local-data", action="store_true")
    parser.add_argument("--refresh-clean-data", action="store_true")
    parser.add_argument(
        "--force-recalculate-factor",
        action="store_true",
        help="忽略已有因子结果文件，重新执行因子计算、分组和绩效输出。",
    )
    return parser.parse_args()


def quote_identifier(identifier: str) -> str:
    return f"`{identifier.replace('`', '``')}`"


def normalize_date_columns(data: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    output = data.copy()
    for col in columns or DATE_COLUMNS:
        if col in output.columns:
            output[col] = pd.to_datetime(output[col], errors="coerce")
    return output


def format_factor_dates(data: pd.DataFrame) -> pd.DataFrame:
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


def build_cache_number_tag(value: float) -> str:
    numeric_value = float(value)
    return str(int(numeric_value)) if numeric_value.is_integer() else str(numeric_value).replace(".", "p")


# ============================================================
# 3. MySQL 读取和本地缓存
# ============================================================


def build_market_sql(start_date: str, end_date: str, market_types: set[int]) -> str:
    selected_columns = ",\n    ".join(quote_identifier(col) for col in SOURCE_COLUMNS)
    market_type_values = ", ".join(str(item) for item in sorted(market_types))
    return f"""
SELECT
    {selected_columns}
FROM {quote_identifier(MARKET_TABLE)}
WHERE {quote_identifier("Trddt")} BETWEEN '{start_date}' AND '{end_date}'
  AND {quote_identifier("Markettype")} IN ({market_type_values});
""".strip()


def export_mysql_to_tsv(sql: str, output_path: Path) -> None:
    if not MYSQL_PASSWORD:
        raise SystemExit("缺少 MYSQL_PASSWORD 环境变量，无法连接 MySQL。请先在环境变量中设置数据库密码。")
    env = os.environ.copy()
    env["MYSQL_PWD"] = MYSQL_PASSWORD
    command = [
        MYSQL_EXE,
        f"--host={MYSQL_HOST}",
        f"--port={MYSQL_PORT}",
        f"--user={MYSQL_USER}",
        f"--database={MYSQL_DATABASE}",
        "--default-character-set=utf8mb4",
        "--batch",
        "--raw",
        f"--execute={sql}",
    ]
    try:
        with output_path.open("wb") as output_file:
            result = subprocess.run(command, stdout=output_file, stderr=subprocess.PIPE, env=env, check=False)
    except FileNotFoundError as exc:
        raise SystemExit("没有找到 mysql 命令行客户端，请确认 mysql.exe 已加入 PATH 或设置 MYSQL_EXE。") from exc
    if result.returncode != 0:
        error_text = result.stderr.decode("utf-8", errors="ignore")
        raise SystemExit(f"MySQL 查询失败：\n{error_text}")


def read_raw_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype="string", na_values=["NULL", r"\N", ""], keep_default_na=True)


def read_raw_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype="string", na_values=["NULL", r"\N", ""], keep_default_na=True)


def build_local_raw_data_cache_path(start_date: str, end_date: str, market_types: set[int]) -> Path:
    start_tag = start_date.replace("-", "")
    end_tag = end_date.replace("-", "")
    market_tag = market_types_to_tag(market_types)
    return LOCAL_RAW_DATA_CACHE_DIR / f"{MARKET_TABLE}_raw_sql_{start_tag}_{end_tag}_mkt{market_tag}.csv"


def build_local_clean_data_cache_paths(
    start_date: str,
    end_date: str,
    min_float_market_value: float,
    market_types: set[int],
) -> dict[str, Path]:
    start_tag = start_date.replace("-", "")
    end_tag = end_date.replace("-", "")
    market_tag = market_types_to_tag(market_types)
    min_value_tag = build_cache_number_tag(min_float_market_value)
    prefix = f"{MARKET_TABLE}_cleaned_ffill_{start_tag}_{end_tag}_mkt{market_tag}_minmv{min_value_tag}"
    return {
        "clean_data": LOCAL_CLEAN_DATA_CACHE_DIR / f"{prefix}.csv",
        "cleaning_log": LOCAL_CLEAN_DATA_CACHE_DIR / f"{prefix}_cleaning_step_log.csv",
        "missing_summary": LOCAL_CLEAN_DATA_CACHE_DIR / f"{prefix}_missing_value_ffill_summary.csv",
        "exclusion_summary": LOCAL_CLEAN_DATA_CACHE_DIR / f"{prefix}_exclusion_reason_summary.csv",
    }


def load_raw_market_data_with_cache(
    start_date: str,
    end_date: str,
    market_types: set[int],
    temp_tsv_path: Path,
    refresh_local_data: bool,
    keep_temp: bool,
) -> tuple[pd.DataFrame, Path, str]:
    cache_path = build_local_raw_data_cache_path(start_date, end_date, market_types)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not refresh_local_data:
        print(f"读取本地原始行情缓存：{cache_path}")
        return read_raw_csv(cache_path), cache_path, "local_cache"

    print("正在从 MySQL 导出原始行情数据...")
    export_mysql_to_tsv(build_market_sql(start_date, end_date, market_types), temp_tsv_path)
    raw_data = read_raw_tsv(temp_tsv_path)
    if not keep_temp:
        temp_tsv_path.unlink(missing_ok=True)
    write_csv(raw_data, cache_path)
    return raw_data, cache_path, "mysql_export"


def read_clean_market_data_cache(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={"stock_code": "string"}, na_values=["NULL", r"\N", ""], keep_default_na=True)
    data["stock_code"] = data["stock_code"].astype("string").str.strip().str.zfill(6)
    data = normalize_date_columns(data, ["trade_date", "capital_change_date"])
    for col in NUMERIC_COLUMNS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in ["market_type", "trade_status", "limit_status"]:
        if col in data.columns:
            data[col] = data[col].round().astype("Int64")
    return data.sort_values(KEY_COLUMNS).reset_index(drop=True)


def read_optional_cache_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame(columns=columns)


def load_clean_market_data_from_cache(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        read_clean_market_data_cache(paths["clean_data"]),
        read_optional_cache_csv(paths["cleaning_log"], ["step", "before_rows", "after_rows", "removed_rows"]),
        read_optional_cache_csv(paths["missing_summary"], ["column", "raw_missing_count", "missing_after_ffill_count"]),
        read_optional_cache_csv(paths["exclusion_summary"], ["exclude_reason", "record_count", "stock_count", "trade_date_count"]),
    )


def write_clean_market_data_cache(
    clean_data: pd.DataFrame,
    cleaning_log: pd.DataFrame,
    missing_summary: pd.DataFrame,
    exclusion_summary: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    write_csv(format_factor_dates(clean_data), paths["clean_data"])
    write_csv(cleaning_log, paths["cleaning_log"])
    write_csv(missing_summary, paths["missing_summary"])
    write_csv(exclusion_summary, paths["exclusion_summary"])


# ============================================================
# 4. 清洗逻辑
# ============================================================


def append_step_log(rows: list[dict], step: str, before_rows: int, after_data: pd.DataFrame) -> None:
    rows.append(
        {
            "step": step,
            "before_rows": before_rows,
            "after_rows": len(after_data),
            "removed_rows": before_rows - len(after_data),
            "after_stock_count": safe_nunique(after_data, "stock_code"),
            "after_trade_date_count": safe_nunique(after_data, "trade_date"),
        }
    )


def standardize_columns(raw_data: pd.DataFrame) -> pd.DataFrame:
    data = raw_data.rename(columns=COLUMN_RENAME_MAP).copy()
    data["stock_code"] = data["stock_code"].astype("string").str.strip().str.zfill(6)
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data["capital_change_date"] = pd.to_datetime(data.get("capital_change_date"), errors="coerce")
    for col in NUMERIC_COLUMNS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in ["market_type", "trade_status", "limit_status"]:
        if col in data.columns:
            data[col] = data[col].round().astype("Int64")
    return data.sort_values(KEY_COLUMNS).reset_index(drop=True)


def summarize_missing_values(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in before.columns:
        raw_missing = int(before[col].isna().sum())
        after_missing = int(after[col].isna().sum()) if col in after.columns else raw_missing
        rows.append(
            {
                "column": col,
                "raw_missing_count": raw_missing,
                "missing_after_ffill_count": after_missing,
                "filled_by_ffill_count": max(raw_missing - after_missing, 0),
            }
        )
    return pd.DataFrame(rows)


def clean_market_data(
    raw_data: pd.DataFrame,
    min_float_market_value: float,
    market_types: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 先把数据库原始字段统一成策略内部字段名，后续所有模块只使用标准字段。
    data = standardize_columns(raw_data)
    log_rows: list[dict] = []
    exclusion_rows: list[dict] = []

    def apply_filter(step: str, mask: pd.Series) -> None:
        nonlocal data
        before_data = data
        before_rows = len(before_data)
        keep_mask = mask.fillna(False).astype(bool)
        removed = before_data.loc[~keep_mask].copy()
        data = before_data.loc[keep_mask].copy()
        append_step_log(log_rows, step, before_rows, data)
        exclusion_rows.append(
            {
                "exclude_reason": step,
                "record_count": len(removed),
                "stock_count": safe_nunique(removed, "stock_code"),
                "trade_date_count": safe_nunique(removed, "trade_date"),
            }
        )

    # 清洗顺序固定：关键字段、市场类型、有效价格、ST/PT 状态、流通市值。
    apply_filter("valid_key_columns", data["trade_date"].notna() & data["stock_code"].notna())

    apply_filter("selected_mainboard_market_types", data["market_type"].isin(market_types))

    valid_price = data[["open_price", "high_price", "low_price", "close_price"]].gt(0).all(axis=1)
    apply_filter("valid_positive_ohlc_prices", valid_price)

    apply_filter("exclude_st_or_pt_status", ~data["trade_status"].isin(ST_OR_PT_STATUS_VALUES))

    apply_filter("min_float_market_value", data["float_market_value"].ge(min_float_market_value))

    # 对可前向填充的行情辅助字段按股票补齐，再按 stock_code/trade_date 去重。
    before_ffill = data.copy()
    data = data.sort_values(KEY_COLUMNS).reset_index(drop=True)
    fill_cols = [col for col in FILL_COLUMNS if col in data.columns]
    data[fill_cols] = data.groupby("stock_code", group_keys=False)[fill_cols].ffill()
    data = data.drop_duplicates(KEY_COLUMNS, keep="last").sort_values(KEY_COLUMNS).reset_index(drop=True)

    cleaning_log = pd.DataFrame(log_rows)
    missing_summary = summarize_missing_values(before_ffill, data)
    exclusion_summary = pd.DataFrame(
        exclusion_rows,
        columns=["exclude_reason", "record_count", "stock_count", "trade_date_count"],
    )
    return data, cleaning_log, missing_summary, exclusion_summary


# ============================================================
# 5. 强动量因子计算
# ============================================================


def build_screening_reason(data: pd.DataFrame) -> pd.Series:
    conditions = [
        ~data["has_pre_peak_low"].fillna(False),
        ~data["strong_rise_filter"].fillna(False),
        ~data["peak_age_filter"].fillna(False),
        ~data["drawdown_filter"].fillna(False),
        ~data["post_peak_low_filter"].fillna(False),
        ~data["pre_peak_zscore_filter"].fillna(False),
        data["passes_strong_momentum_filters"].fillna(False),
    ]
    choices = [
        "missing_low_before_peak",
        "low_to_peak_rise_not_enough",
        "peak_too_old",
        "drawdown_not_enough",
        "signal_not_post_peak_low",
        "pre_peak_zscore_too_high_or_history_short",
        "passed",
    ]
    return pd.Series(np.select(conditions, choices, default="unknown"), index=data.index)


def cumulative_return_from_dretwd(
    return_values: np.ndarray,
    start_pos_exclusive: int,
    end_pos_inclusive: int,
) -> float:
    """Compound Dretwd for (start_pos_exclusive, end_pos_inclusive]."""

    if end_pos_inclusive <= start_pos_exclusive:
        return 0.0
    period_returns = return_values[start_pos_exclusive + 1 : end_pos_inclusive + 1]
    if period_returns.size == 0 or not np.isfinite(period_returns).all():
        return math.nan
    return float(np.prod(1.0 + period_returns) - 1.0)


def attach_change_ratio_zscore(data: pd.DataFrame, zscore_window: int, zscore_ddof: int) -> pd.DataFrame:
    output = data.copy()
    if "change_ratio" not in output.columns:
        output["change_ratio_zscore"] = np.nan
        return output
    zscore_window = max(int(zscore_window), 1)
    zscore_ddof = int(zscore_ddof)
    output["change_ratio"] = pd.to_numeric(output["change_ratio"], errors="coerce")
    output = output.sort_values(KEY_COLUMNS).reset_index(drop=True)

    def one_stock_time_zscore(series: pd.Series) -> pd.Series:
        # 每只股票单独做时序 z-score，且只使用当日前的历史窗口，避免未来信息。
        values = pd.to_numeric(series, errors="coerce")
        history = values.shift(1)
        rolling_mean = history.rolling(zscore_window, min_periods=zscore_window).mean()
        rolling_std = history.rolling(zscore_window, min_periods=zscore_window).std(ddof=zscore_ddof)
        zscore = (values - rolling_mean) / rolling_std
        return zscore.where(rolling_std.notna() & rolling_std.ne(0))

    output["change_ratio_zscore"] = (
        output.groupby("stock_code", group_keys=False)["change_ratio"]
        .transform(one_stock_time_zscore)
    )
    return output


def calculate_one_stock_strong_momentum(
    stock_data: pd.DataFrame,
    rise_threshold: float,
    drawdown_threshold: float,
    lookback_days: int,
    max_peak_age_days: int,
    zscore_window: int,
    zscore_limit: float,
) -> pd.DataFrame:
    """逐日计算单只股票的强动量形态。

    每个 T 日只在最近 lookback_days 个交易记录内取第一次出现的最高收盘价作为唯一 peak，
    后续五重筛选全部围绕这个 peak 判断。
    """

    stock = stock_data.sort_values("trade_date").reset_index(drop=True).copy()
    close_values = pd.to_numeric(stock["close_price"], errors="coerce").to_numpy(dtype="float64")
    if "return_with_dividend" in stock.columns:
        dretwd_values = pd.to_numeric(stock["return_with_dividend"], errors="coerce").to_numpy(dtype="float64")
    else:
        dretwd_values = np.full(len(stock), np.nan, dtype="float64")
    if "change_ratio_zscore" in stock.columns:
        change_zscore_values = pd.to_numeric(stock["change_ratio_zscore"], errors="coerce").to_numpy(dtype="float64")
    else:
        change_zscore_values = np.full(len(stock), np.nan, dtype="float64")
    dates = pd.to_datetime(stock["trade_date"], errors="coerce").reset_index(drop=True)
    if "limit_status" in stock.columns:
        limit_flags = pd.to_numeric(stock["limit_status"], errors="coerce").isin(LIMIT_UP_OR_DOWN_VALUES).astype("int64").to_numpy()
    else:
        limit_flags = np.zeros(len(stock), dtype="int64")
    limit_prefix = np.cumsum(limit_flags)
    lookback_days = max(int(lookback_days), 1)
    max_peak_age_days = max(int(max_peak_age_days), 1)
    zscore_window = max(int(zscore_window), 1)

    peak_dates: list[pd.Timestamp] = []
    peak_prices: list[float] = []
    low_dates: list[pd.Timestamp] = []
    low_prices: list[float] = []
    post_peak_low_dates: list[pd.Timestamp] = []
    post_peak_low_prices: list[float] = []
    low_to_peak_returns: list[float] = []
    drawdowns_from_peak_to_signal: list[float] = []
    lookback_window_starts: list[pd.Timestamp] = []
    max_peak_age_starts: list[pd.Timestamp] = []
    pre_peak_zscore_window_starts: list[pd.Timestamp] = []
    pre_peak_max_abs_zscores: list[float] = []
    pre_peak_zscore_valid_days: list[int | None] = []
    valid_days_from_peak: list[int | None] = []
    limit_days_from_peak: list[int | None] = []
    full_history_valid_days: list[int] = []

    valid_seen = 0

    for i, price in enumerate(close_values):
        is_valid_price = np.isfinite(price) and price > 0
        if is_valid_price:
            valid_seen += 1

        # 对每个 T 日，只在 lookback 窗口内取第一次出现的最高收盘价作为唯一 peak。
        window_start_pos = max(0, i - lookback_days + 1)
        max_peak_age_start_pos = max(0, i - max_peak_age_days + 1)
        lookback_window_starts.append(dates.iloc[window_start_pos])
        max_peak_age_starts.append(dates.iloc[max_peak_age_start_pos])
        window_prices = close_values[window_start_pos : i + 1]
        valid_window_mask = np.isfinite(window_prices) & (window_prices > 0)
        valid_window_offsets = np.flatnonzero(valid_window_mask)

        if valid_window_offsets.size == 0:
            peak_dates.append(pd.NaT)
            peak_prices.append(math.nan)
            low_dates.append(pd.NaT)
            low_prices.append(math.nan)
            post_peak_low_dates.append(pd.NaT)
            post_peak_low_prices.append(math.nan)
            low_to_peak_returns.append(math.nan)
            drawdowns_from_peak_to_signal.append(math.nan)
            pre_peak_zscore_window_starts.append(pd.NaT)
            pre_peak_max_abs_zscores.append(math.nan)
            pre_peak_zscore_valid_days.append(pd.NA)
            valid_days_from_peak.append(pd.NA)
            limit_days_from_peak.append(pd.NA)
            full_history_valid_days.append(valid_seen)
            continue

        valid_window_prices = window_prices[valid_window_offsets]
        first_peak_index = int(np.argmax(valid_window_prices))
        peak_offset = int(valid_window_offsets[first_peak_index])
        peak_pos = window_start_pos + peak_offset
        peak_price = float(close_values[peak_pos])
        # 第 5 条筛选使用 peak 前 zscore_window 个 ChangeRatio 时序 z-score 的最大绝对值。
        pre_peak_zscore_start_pos = max(0, peak_pos - zscore_window)
        pre_peak_abs_zscore_window = np.abs(change_zscore_values[pre_peak_zscore_start_pos:peak_pos])
        valid_pre_peak_zscore_window = pre_peak_abs_zscore_window[np.isfinite(pre_peak_abs_zscore_window)]
        valid_pre_peak_zscore_days = int(valid_pre_peak_zscore_window.size)
        pre_peak_max_abs_zscore = (
            float(valid_pre_peak_zscore_window.max())
            if valid_pre_peak_zscore_window.size > 0
            else math.nan
        )

        low_slice = close_values[window_start_pos:peak_pos]
        valid_low_mask = np.isfinite(low_slice) & (low_slice > 0)
        valid_low_offsets = np.flatnonzero(valid_low_mask)
        if valid_low_offsets.size > 0:
            low_offset = int(valid_low_offsets[np.argmin(low_slice[valid_low_offsets])])
            low_pos_for_peak = window_start_pos + low_offset
            low_price_for_peak = float(close_values[low_pos_for_peak])
        else:
            low_pos_for_peak = -1
            low_price_for_peak = math.nan

        # 第 4 条筛选要求 T 日是同一个 peak 之后截至 T 日的最低收盘价；并列最低时取最后一次。
        post_peak_slice = close_values[peak_pos : i + 1]
        valid_post_peak_mask = np.isfinite(post_peak_slice) & (post_peak_slice > 0)
        valid_post_peak_offsets = np.flatnonzero(valid_post_peak_mask)
        if valid_post_peak_offsets.size > 0:
            valid_post_peak_prices = post_peak_slice[valid_post_peak_offsets]
            reverse_low_index = int(np.argmin(valid_post_peak_prices[::-1]))
            post_low_offset = int(valid_post_peak_offsets[len(valid_post_peak_offsets) - 1 - reverse_low_index])
            post_low_pos = peak_pos + post_low_offset
            post_low_price = float(close_values[post_low_pos])
        else:
            post_low_pos = -1
            post_low_price = math.nan

        before_window_limit_count = limit_prefix[window_start_pos - 1] if window_start_pos > 0 else 0
        peak_dates.append(dates.iloc[peak_pos])
        peak_prices.append(peak_price)
        low_dates.append(dates.iloc[low_pos_for_peak] if low_pos_for_peak >= 0 else pd.NaT)
        low_prices.append(low_price_for_peak if low_pos_for_peak >= 0 else math.nan)
        post_peak_low_dates.append(dates.iloc[post_low_pos] if post_low_pos >= 0 else pd.NaT)
        post_peak_low_prices.append(post_low_price if post_low_pos >= 0 else math.nan)
        pre_peak_zscore_window_starts.append(dates.iloc[pre_peak_zscore_start_pos] if peak_pos > 0 else pd.NaT)
        pre_peak_max_abs_zscores.append(pre_peak_max_abs_zscore)
        pre_peak_zscore_valid_days.append(valid_pre_peak_zscore_days)
        low_to_peak_returns.append(
            cumulative_return_from_dretwd(dretwd_values, low_pos_for_peak, peak_pos)
            if low_pos_for_peak >= 0
            else math.nan
        )
        drawdowns_from_peak_to_signal.append(cumulative_return_from_dretwd(dretwd_values, peak_pos, i))
        full_history_valid_days.append(valid_seen)
        valid_days_from_peak.append(int(valid_window_offsets.size))
        limit_days_from_peak.append(int(limit_prefix[i] - before_window_limit_count))

    stock["peak_trade_date"] = pd.to_datetime(peak_dates)
    stock["peak_close_price"] = pd.to_numeric(peak_prices, errors="coerce")
    stock["pre_peak_low_trade_date"] = pd.to_datetime(low_dates)
    stock["pre_peak_low_close_price"] = pd.to_numeric(low_prices, errors="coerce")
    stock["post_peak_low_trade_date"] = pd.to_datetime(post_peak_low_dates)
    stock["post_peak_low_close_price"] = pd.to_numeric(post_peak_low_prices, errors="coerce")
    stock["full_history_valid_days"] = full_history_valid_days
    stock["lookback_valid_days"] = pd.Series(valid_days_from_peak, dtype="Int64")
    stock["lookback_limit_days_count"] = pd.Series(limit_days_from_peak, dtype="Int64")
    stock["lookback_has_limit_up_or_down"] = stock["lookback_limit_days_count"].fillna(0).gt(0)

    stock["has_pre_peak_low"] = (
        stock["pre_peak_low_trade_date"].notna()
        & stock["pre_peak_low_close_price"].notna()
        & stock["pre_peak_low_close_price"].gt(0)
        & stock["pre_peak_low_trade_date"].lt(stock["peak_trade_date"])
    )
    stock["low_to_peak_return"] = pd.to_numeric(low_to_peak_returns, errors="coerce")
    stock["drawdown_from_peak_to_signal"] = pd.to_numeric(drawdowns_from_peak_to_signal, errors="coerce")
    stock["signal_is_post_peak_low"] = (
        stock["trade_date"].eq(stock["post_peak_low_trade_date"])
        & (stock["close_price"] - stock["post_peak_low_close_price"]).abs().le(PRICE_COMPARE_TOLERANCE)
    )
    stock["lookback_window_start"] = pd.to_datetime(lookback_window_starts)
    stock["max_peak_age_start"] = pd.to_datetime(max_peak_age_starts)
    stock["pre_peak_zscore_window_start"] = pd.to_datetime(pre_peak_zscore_window_starts)
    stock["pre_peak_max_abs_zscore"] = pd.to_numeric(pre_peak_max_abs_zscores, errors="coerce")
    stock["pre_peak_zscore_valid_days"] = pd.Series(pre_peak_zscore_valid_days, dtype="Int64")
    # 五条路径筛选必须同时满足，后续最终股票池只接收全部通过的记录。
    stock["strong_rise_filter"] = stock["has_pre_peak_low"] & stock["low_to_peak_return"].gt(rise_threshold)
    stock["peak_age_filter"] = stock["strong_rise_filter"] & stock["peak_trade_date"].ge(stock["max_peak_age_start"])
    stock["drawdown_filter"] = stock["peak_age_filter"] & stock["drawdown_from_peak_to_signal"].lt(-drawdown_threshold)
    stock["post_peak_low_filter"] = stock["drawdown_filter"] & stock["signal_is_post_peak_low"]
    stock["pre_peak_zscore_filter"] = (
        stock["post_peak_low_filter"]
        & stock["pre_peak_zscore_valid_days"].eq(zscore_window)
        & stock["pre_peak_max_abs_zscore"].le(zscore_limit)
    )
    stock["passes_strong_momentum_filters"] = (
        stock["has_pre_peak_low"]
        & stock["strong_rise_filter"]
        & stock["peak_age_filter"]
        & stock["drawdown_filter"]
        & stock["post_peak_low_filter"]
        & stock["pre_peak_zscore_filter"]
    )
    stock["screening_reason"] = build_screening_reason(stock)

    # momentum_raw 只在五条筛选全通过时保留；否则置空，避免进入最终股票池。
    stock["momentum_raw"] = stock["drawdown_from_peak_to_signal"].where(stock["passes_strong_momentum_filters"])
    stock["strong_momentum_raw"] = stock["momentum_raw"]
    stock["momentum_start_date"] = stock["peak_trade_date"]
    stock["momentum_end_date"] = stock["trade_date"]
    stock["momentum_start_close_price"] = stock["peak_close_price"]
    stock["momentum_end_close_price"] = stock["close_price"]
    return stock


def calculate_raw_strong_momentum(
    clean_data: pd.DataFrame,
    rise_threshold: float,
    drawdown_threshold: float,
    lookback_days: int,
    max_peak_age_days: int,
    zscore_window: int,
    zscore_limit: float,
    zscore_ddof: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 先附加 ChangeRatio 时序 z-score，再逐股票逐日计算强动量形态。
    data = attach_change_ratio_zscore(
        normalize_date_columns(clean_data).sort_values(KEY_COLUMNS).reset_index(drop=True),
        zscore_window=zscore_window,
        zscore_ddof=zscore_ddof,
    )
    parts = [
        calculate_one_stock_strong_momentum(
            one_stock,
            rise_threshold=rise_threshold,
            drawdown_threshold=drawdown_threshold,
            lookback_days=lookback_days,
            max_peak_age_days=max_peak_age_days,
            zscore_window=zscore_window,
            zscore_limit=zscore_limit,
        )
        for _, one_stock in data.groupby("stock_code", sort=False)
    ]
    factor = pd.concat(parts, ignore_index=True) if parts else data.iloc[0:0].copy()
    factor = factor.rename(columns={"close_price": "signal_close_price"})
    factor["lookback_days"] = lookback_days
    factor["lookback_unit"] = "trading_days"
    factor["max_peak_age_days"] = max_peak_age_days
    factor["zscore_window"] = zscore_window
    factor["zscore_limit"] = zscore_limit
    factor["zscore_ddof"] = zscore_ddof
    factor["rebalance_frequency"] = "daily"

    ordered_columns = [
        "stock_code",
        "trade_date",
        "signal_close_price",
        "momentum_start_date",
        "momentum_end_date",
        "momentum_start_close_price",
        "momentum_end_close_price",
        "lookback_window_start",
        "pre_peak_low_trade_date",
        "pre_peak_low_close_price",
        "peak_trade_date",
        "peak_close_price",
        "pre_peak_zscore_window_start",
        "pre_peak_zscore_valid_days",
        "pre_peak_max_abs_zscore",
        "post_peak_low_trade_date",
        "post_peak_low_close_price",
        "full_history_valid_days",
        "lookback_valid_days",
        "lookback_limit_days_count",
        "lookback_has_limit_up_or_down",
        "low_to_peak_return",
        "drawdown_from_peak_to_signal",
        "max_peak_age_start",
        "has_pre_peak_low",
        "signal_is_post_peak_low",
        "strong_rise_filter",
        "peak_age_filter",
        "drawdown_filter",
        "post_peak_low_filter",
        "pre_peak_zscore_filter",
        "passes_strong_momentum_filters",
        "screening_reason",
        "momentum_raw",
        "strong_momentum_raw",
        "lookback_days",
        "lookback_unit",
        "max_peak_age_days",
        "zscore_window",
        "zscore_limit",
        "zscore_ddof",
        "rebalance_frequency",
    ]
    factor = factor[[col for col in ordered_columns if col in factor.columns]].copy()
    # 最终股票池只保留 momentum_raw 非空记录，即五条筛选全部满足的股票-日期。
    valid_factor = factor.loc[factor["momentum_raw"].notna()].copy()
    missing_factor = factor.loc[factor["momentum_raw"].isna()].copy()
    return valid_factor, missing_factor


def build_all_factor(raw_factor: pd.DataFrame, missing_factor: pd.DataFrame) -> pd.DataFrame:
    parts = [part for part in [raw_factor, missing_factor] if not part.empty]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(["trade_date", "stock_code"]).reset_index(drop=True)


# ============================================================
# 6. 截面处理、分组与回测
# ============================================================


def apply_3sigma_winsorization(raw_factor: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stats = (
        raw_factor.groupby("trade_date")["momentum_raw"]
        .agg(
            cross_section_mean="mean",
            cross_section_std=lambda series: series.std(ddof=0),
            cross_section_min="min",
            cross_section_max="max",
            cross_section_stock_count="count",
        )
        .reset_index()
    )
    stats["sigma_lower_bound"] = stats["cross_section_mean"] - 3.0 * stats["cross_section_std"]
    stats["sigma_upper_bound"] = stats["cross_section_mean"] + 3.0 * stats["cross_section_std"]
    factor = raw_factor.merge(stats, on="trade_date", how="left")
    valid_sigma = factor["cross_section_std"].notna() & factor["cross_section_std"].ne(0)
    lower_mask = valid_sigma & factor["momentum_raw"].lt(factor["sigma_lower_bound"])
    upper_mask = valid_sigma & factor["momentum_raw"].gt(factor["sigma_upper_bound"])
    factor["is_3sigma_extreme"] = lower_mask | upper_mask
    factor["extreme_direction"] = np.select(
        [lower_mask, upper_mask],
        ["lower_than_mean_minus_3std", "higher_than_mean_plus_3std"],
        default="within_3sigma",
    )
    factor["momentum_3sigma"] = factor["momentum_raw"]
    factor.loc[lower_mask, "momentum_3sigma"] = factor.loc[lower_mask, "sigma_lower_bound"]
    factor.loc[upper_mask, "momentum_3sigma"] = factor.loc[upper_mask, "sigma_upper_bound"]
    factor["strong_momentum_3sigma"] = factor["momentum_3sigma"]
    extreme_records = factor.loc[factor["is_3sigma_extreme"]].copy()
    extreme_count = extreme_records.groupby("trade_date").size().rename("extreme_count").reset_index()
    date_summary = stats.merge(extreme_count, on="trade_date", how="left")
    date_summary["extreme_count"] = date_summary["extreme_count"].fillna(0).astype("Int64")
    return factor, extreme_records, date_summary


def apply_zscore_standardization(factor_3sigma: pd.DataFrame) -> pd.DataFrame:
    stats = (
        factor_3sigma.groupby("trade_date")["momentum_3sigma"]
        .agg(zscore_mean="mean", zscore_std=lambda series: series.std(ddof=0), zscore_stock_count="count")
        .reset_index()
    )
    factor = factor_3sigma.merge(stats, on="trade_date", how="left")
    valid_std = factor["zscore_std"].notna() & factor["zscore_std"].ne(0)
    factor["momentum_zscore"] = np.nan
    factor.loc[valid_std, "momentum_zscore"] = (
        factor.loc[valid_std, "momentum_3sigma"] - factor.loc[valid_std, "zscore_mean"]
    ) / factor.loc[valid_std, "zscore_std"]
    factor["strong_momentum_zscore"] = factor["momentum_zscore"]
    return factor


def rank_standardized_factor(factor_zscore: pd.DataFrame) -> pd.DataFrame:
    ranked = factor_zscore.loc[factor_zscore["momentum_zscore"].notna()].sort_values(
        ["trade_date", "momentum_zscore", "stock_code"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    ranked["standardized_momentum_rank_desc"] = ranked.groupby("trade_date").cumcount() + 1
    ranked["cross_section_stock_count_after_zscore"] = ranked.groupby("trade_date")["stock_code"].transform("count")
    ranked["standardized_momentum_percentile_desc"] = (
        ranked["standardized_momentum_rank_desc"] / ranked["cross_section_stock_count_after_zscore"]
    )
    ranked["standardized_momentum_rank_desc"] = ranked["standardized_momentum_rank_desc"].astype("Int64")
    ranked["cross_section_stock_count_after_zscore"] = ranked["cross_section_stock_count_after_zscore"].astype("Int64")
    return ranked


def calculate_forward_holding_returns(clean_data: pd.DataFrame, holding_days: int) -> pd.DataFrame:
    # 未来事件收益统一使用 Dretwd 累乘，避免未复权价格带来的收益计算偏差。
    data = normalize_date_columns(clean_data).sort_values(KEY_COLUMNS)[
        ["stock_code", "trade_date", "open_price", "close_price", "return_with_dividend"]
    ].copy()
    data["open_price"] = pd.to_numeric(data["open_price"], errors="coerce")
    data["close_price"] = pd.to_numeric(data["close_price"], errors="coerce")
    data["return_with_dividend"] = pd.to_numeric(data["return_with_dividend"], errors="coerce")
    grouped = data.groupby("stock_code", sort=False)
    terminal_offset_days = max(int(holding_days), 1)
    # 信号在 T 日产生；T+1 开盘价用于交易可行性和买入价记录，收益区间为 T+1..T+holding_days。
    entry_open = grouped["open_price"].shift(-1)
    entry_close = grouped["close_price"].shift(-1)
    holding_end_close = grouped["close_price"].shift(-terminal_offset_days)
    complete_mask = entry_open.gt(0) & holding_end_close.gt(0)
    data["future_return_valid_days"] = 0
    data["holding_period_return_factor"] = 1.0
    for offset in range(1, terminal_offset_days + 1):
        shifted_return = grouped["return_with_dividend"].shift(-offset)
        valid = shifted_return.notna()
        data["future_return_valid_days"] += valid.astype("int64")
        data["holding_period_return_factor"] *= 1.0 + shifted_return.fillna(0.0)
    complete_mask &= data["future_return_valid_days"].eq(max(int(holding_days), 1))
    data["holding_entry_open_price"] = entry_open
    data["holding_entry_close_price"] = entry_close
    data["holding_entry_price"] = entry_open
    data["holding_entry_price_type"] = "T+1_open"
    data["holding_end_close_price"] = holding_end_close
    data["holding_exit_close_price"] = holding_end_close
    data["holding_terminal_offset_days"] = terminal_offset_days
    data["next_period_return_before_trade_filter"] = data["holding_period_return_factor"] - 1.0
    data.loc[~complete_mask, "next_period_return_before_trade_filter"] = np.nan
    data["has_complete_holding_return"] = complete_mask
    data["forward_return_rule"] = f"buy_t_plus_1_open_hold_{holding_days}_days_compound_dretwd_t_plus_1_to_t_plus_{terminal_offset_days}"
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


def assign_quantile_groups(
    ranked_factor: pd.DataFrame,
    clean_data: pd.DataFrame,
    group_num: int,
    holding_days: int,
    min_cross_section_stock_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped = normalize_date_columns(ranked_factor)
    clean_data = normalize_date_columns(clean_data)

    trade_calendar = clean_data[["trade_date"]].drop_duplicates().sort_values("trade_date").reset_index(drop=True)
    terminal_offset_days = max(int(holding_days), 1)
    trade_calendar["next_trade_date"] = trade_calendar["trade_date"].shift(-1)
    trade_calendar["holding_start_trade_date"] = trade_calendar["next_trade_date"]
    trade_calendar["holding_end_trade_date"] = trade_calendar["trade_date"].shift(-terminal_offset_days)
    grouped = grouped.merge(trade_calendar, on="trade_date", how="left")

    forward_returns = calculate_forward_holding_returns(clean_data, holding_days)
    grouped = grouped.merge(forward_returns, on=["stock_code", "trade_date"], how="left")

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
    entry_data = clean_data.sort_values(KEY_COLUMNS)[entry_cols].rename(
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
    grouped = grouped.merge(entry_data, on=["stock_code", "next_trade_date"], how="left")

    close_equals_limit_up = (grouped["next_close_price"] - grouped["next_limit_up_price"]).abs().le(PRICE_COMPARE_TOLERANCE)
    close_equals_limit_down = (grouped["next_close_price"] - grouped["next_limit_down_price"]).abs().le(PRICE_COMPARE_TOLERANCE)
    grouped["is_next_close_limit_up"] = close_equals_limit_up.fillna(False).astype(bool)
    grouped["is_next_close_limit_down"] = close_equals_limit_down.fillna(False).astype(bool)
    grouped["is_next_close_limit"] = grouped["is_next_close_limit_up"] | grouped["is_next_close_limit_down"]
    open_equals_limit_up = (grouped["next_open_price"] - grouped["next_limit_up_price"]).abs().le(PRICE_COMPARE_TOLERANCE)
    open_equals_limit_down = (grouped["next_open_price"] - grouped["next_limit_down_price"]).abs().le(PRICE_COMPARE_TOLERANCE)
    grouped["is_next_open_limit_up"] = open_equals_limit_up.fillna(False).astype(bool)
    grouped["is_next_open_limit_down"] = open_equals_limit_down.fillna(False).astype(bool)
    grouped["is_next_open_limit"] = grouped["is_next_open_limit_up"] | grouped["is_next_open_limit_down"]
    grouped["is_next_open_one_word_limit_up"] = grouped["is_next_open_limit_up"]
    grouped["is_next_open_one_word_limit_down"] = grouped["is_next_open_limit_down"]
    grouped["is_next_open_one_word_limit"] = grouped["is_next_open_limit"]

    missing_next = grouped["next_open_price"].isna() | grouped["next_period_return_before_trade_filter"].isna()
    grouped["is_tradable_next_open"] = (~missing_next & ~grouped["is_next_open_limit"]).fillna(False).astype(bool)
    grouped["is_tradable_next_close"] = grouped["is_tradable_next_open"]
    grouped["next_period_return"] = grouped["next_period_return_before_trade_filter"]
    grouped["trade_filter_reason"] = np.select(
        [
            missing_next.to_numpy(dtype=bool),
            grouped["is_next_open_limit_up"].to_numpy(dtype=bool),
            grouped["is_next_open_limit_down"].to_numpy(dtype=bool),
        ],
        [
            "missing_next_open_record_or_return",
            "next_open_limit_up",
            "next_open_limit_down",
        ],
        default="tradable_next_open",
    )
    grouped["next_period_return_column_used"] = "compound_dretwd_tplus1_to_holding_end_open_entry"
    grouped["rebalance_frequency"] = "daily"
    grouped["holding_days"] = holding_days
    grouped["holding_terminal_offset_days"] = terminal_offset_days
    grouped["group_num"] = group_num

    # The strategy forms groups only after next-open tradability is confirmed.
    tradable = grouped.loc[grouped["is_tradable_next_open"]].copy()
    grouping_min_count = max(int(min_cross_section_stock_count), int(group_num), 1)
    cross_section_stock_count = build_cross_section_stock_count(tradable, grouping_min_count)
    grouped, cross_section_excluded_records = filter_by_cross_section_count(tradable, cross_section_stock_count)

    grouped = grouped.sort_values(["trade_date", "momentum_zscore", "stock_code"], ascending=[True, False, True]).reset_index(drop=True)
    if not grouped.empty:
        grouped["standardized_momentum_rank_desc"] = grouped.groupby("trade_date").cumcount() + 1
        grouped["cross_section_stock_count_after_zscore"] = grouped.groupby("trade_date")["stock_code"].transform("count")
        grouped["standardized_momentum_percentile_desc"] = (
            grouped["standardized_momentum_rank_desc"] / grouped["cross_section_stock_count_after_zscore"]
        )
        grouped["tradable_momentum_rank_desc"] = grouped["standardized_momentum_rank_desc"]
        grouped["tradable_cross_section_stock_count"] = grouped["cross_section_stock_count_after_zscore"]
        grouped["tradable_momentum_percentile_desc"] = grouped["standardized_momentum_percentile_desc"]
        grouped["quantile_group"] = (
            ((grouped["standardized_momentum_rank_desc"] - 1) * group_num)
            // grouped["cross_section_stock_count_after_zscore"]
            + 1
        ).clip(lower=1, upper=group_num)
        grouped["standardized_momentum_rank_desc"] = grouped["standardized_momentum_rank_desc"].astype("Int64")
        grouped["cross_section_stock_count_after_zscore"] = grouped["cross_section_stock_count_after_zscore"].astype("Int64")
        grouped["tradable_momentum_rank_desc"] = grouped["tradable_momentum_rank_desc"].astype("Int64")
        grouped["tradable_cross_section_stock_count"] = grouped["tradable_cross_section_stock_count"].astype("Int64")
        grouped["quantile_group"] = grouped["quantile_group"].astype("Int64")
    else:
        grouped["tradable_momentum_rank_desc"] = pd.Series(dtype="Int64")
        grouped["tradable_cross_section_stock_count"] = pd.Series(dtype="Int64")
        grouped["tradable_momentum_percentile_desc"] = pd.Series(dtype="float64")
        grouped["quantile_group"] = pd.Series(dtype="Int64")

    grouped["is_long_short_trade_candidate"] = grouped["quantile_group"].isin([1, group_num]).fillna(False).astype(bool)
    grouped["is_tradable_long_short_next_open"] = grouped["is_long_short_trade_candidate"]
    grouped["next_period_return_after_long_short_trade_filter"] = grouped["next_period_return"]
    grouped["long_short_role"] = np.select(
        [
            grouped["quantile_group"].eq(1).fillna(False).to_numpy(dtype=bool),
            grouped["quantile_group"].eq(group_num).fillna(False).to_numpy(dtype=bool),
        ],
        ["long_top_group", "short_bottom_group"],
        default="middle_group",
    )
    tradable_stock_pool = grouped.copy()
    return grouped, tradable_stock_pool, cross_section_stock_count, cross_section_excluded_records


def calculate_quantile_portfolio_returns(grouped_factor: pd.DataFrame, group_num: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = grouped_factor.copy()
    data["is_group_return_record"] = data["next_period_return"].notna()
    data["is_tradable_long_short_return_record"] = data["next_period_return_after_long_short_trade_filter"].notna()
    data["next_open_limit_excluded_flag"] = (
        data["is_long_short_trade_candidate"] & data["is_next_open_limit"].fillna(False).astype(bool)
    ).astype("int64")
    data["missing_next_record_flag"] = (
        data["next_open_price"].isna() | data["next_period_return_before_trade_filter"].isna()
    ).astype("int64")

    quantile_returns = (
        data.groupby(["trade_date", "quantile_group"])
        .agg(
            group_return_sum=("next_period_return", lambda series: series.sum(min_count=1)),
            group_stock_count=("next_period_return", "count"),
            signal_group_stock_count=("stock_code", "count"),
            group_return_after_long_short_trade_filter_sum=(
                "next_period_return_after_long_short_trade_filter",
                lambda series: series.sum(min_count=1),
            ),
            group_stock_count_after_long_short_trade_filter=("next_period_return_after_long_short_trade_filter", "count"),
            next_open_limit_excluded_count=("next_open_limit_excluded_flag", "sum"),
            missing_next_record_count=("missing_next_record_flag", "sum"),
            group_avg_momentum_zscore=("momentum_zscore", "mean"),
            group_avg_momentum_raw=("momentum_raw", "mean"),
            next_trade_date_min=("next_trade_date", "min"),
            next_trade_date_max=("next_trade_date", "max"),
            holding_start_trade_date_min=("holding_start_trade_date", "min"),
            holding_start_trade_date_max=("holding_start_trade_date", "max"),
            holding_end_trade_date_min=("holding_end_trade_date", "min"),
            holding_end_trade_date_max=("holding_end_trade_date", "max"),
        )
        .reset_index()
    )
    quantile_returns["group_equal_weight_return"] = (
        quantile_returns["group_return_sum"] / quantile_returns["group_stock_count"].replace(0, np.nan)
    )
    quantile_returns["group_equal_weight_return_after_long_short_trade_filter"] = (
        quantile_returns["group_return_after_long_short_trade_filter_sum"]
        / quantile_returns["group_stock_count_after_long_short_trade_filter"].replace(0, np.nan)
    )
    quantile_returns["next_open_one_word_limit_excluded_count"] = quantile_returns["next_open_limit_excluded_count"]
    quantile_returns["rebalance_frequency"] = "daily"
    quantile_returns["holding_days"] = data["holding_days"].iloc[0] if not data.empty else DEFAULT_HOLDING_DAYS
    quantile_returns = quantile_returns.sort_values(["trade_date", "quantile_group"]).reset_index(drop=True)

    long_side = quantile_returns.loc[quantile_returns["quantile_group"].eq(1)].rename(
        columns={
            "group_equal_weight_return_after_long_short_trade_filter": "long_top_group_return",
            "group_return_after_long_short_trade_filter_sum": "long_top_group_return_sum",
            "group_stock_count_after_long_short_trade_filter": "long_top_group_stock_count",
            "signal_group_stock_count": "long_top_group_signal_stock_count",
            "next_open_limit_excluded_count": "long_top_group_next_open_limit_excluded_count",
            "missing_next_record_count": "long_top_group_missing_next_record_count",
        }
    )
    short_side = quantile_returns.loc[quantile_returns["quantile_group"].eq(group_num)].rename(
        columns={
            "group_equal_weight_return_after_long_short_trade_filter": "short_bottom_group_return",
            "group_return_after_long_short_trade_filter_sum": "short_bottom_group_return_sum",
            "group_stock_count_after_long_short_trade_filter": "short_bottom_group_stock_count",
            "signal_group_stock_count": "short_bottom_group_signal_stock_count",
            "next_open_limit_excluded_count": "short_bottom_group_next_open_limit_excluded_count",
            "missing_next_record_count": "short_bottom_group_missing_next_record_count",
        }
    )
    keep_long = [
        "trade_date",
        "long_top_group_return",
        "long_top_group_return_sum",
        "long_top_group_stock_count",
        "long_top_group_signal_stock_count",
        "long_top_group_next_open_limit_excluded_count",
        "long_top_group_missing_next_record_count",
        "next_trade_date_min",
        "next_trade_date_max",
        "holding_start_trade_date_min",
        "holding_start_trade_date_max",
        "holding_end_trade_date_min",
        "holding_end_trade_date_max",
        "rebalance_frequency",
        "holding_days",
    ]
    keep_short = [
        "trade_date",
        "short_bottom_group_return",
        "short_bottom_group_return_sum",
        "short_bottom_group_stock_count",
        "short_bottom_group_signal_stock_count",
        "short_bottom_group_next_open_limit_excluded_count",
        "short_bottom_group_missing_next_record_count",
    ]
    long_short = long_side[keep_long].merge(short_side[keep_short], on="trade_date", how="inner")
    long_short["long_short_spread_return"] = long_short["long_top_group_return"] - long_short["short_bottom_group_return"]
    long_short["long_top_group_nav"] = (1.0 + long_short["long_top_group_return"].fillna(0.0)).cumprod()
    long_short["short_bottom_group_nav"] = (1.0 + long_short["short_bottom_group_return"].fillna(0.0)).cumprod()
    long_short["long_short_spread_nav"] = (1.0 + long_short["long_short_spread_return"].fillna(0.0)).cumprod()
    return quantile_returns, long_short


def build_long_short_stock_members(grouped_factor: pd.DataFrame, group_num: int) -> pd.DataFrame:
    members = grouped_factor.loc[grouped_factor["quantile_group"].isin([1, group_num])].copy()
    members["portfolio_side"] = np.where(members["quantile_group"].eq(1), "long", "short")
    members["long_short_role"] = np.where(
        members["quantile_group"].eq(1),
        "long_top_group",
        "short_bottom_group",
    )
    return members.sort_values(["trade_date", "quantile_group", "standardized_momentum_rank_desc", "stock_code"])


# ============================================================
# 7. IC、绩效和摘要
# ============================================================


def calculate_ic_ir(factor_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    factor = factor_data.copy()
    valid = factor[["trade_date", "momentum_zscore", "next_period_return"]].dropna().copy()
    rows = []
    for trade_date, one_day in valid.groupby("trade_date"):
        if len(one_day) < 2:
            continue
        rows.append(
            {
                "trade_date": trade_date,
                "ic": one_day["momentum_zscore"].corr(one_day["next_period_return"], method="pearson"),
                "rank_ic": one_day["momentum_zscore"].corr(one_day["next_period_return"], method="spearman"),
                "stock_count": len(one_day),
            }
        )
    ic_series = pd.DataFrame(rows)
    if ic_series.empty:
        ic_series = pd.DataFrame(columns=["trade_date", "ic", "rank_ic", "stock_count", "cumulative_ic", "cumulative_rank_ic"])
        ic_summary = pd.DataFrame(columns=["metric", "mean", "std", "t_value", "p_value", "ir", "observation_count"])
        return ic_series, ic_summary

    ic_series = ic_series.sort_values("trade_date").reset_index(drop=True)
    ic_series["cumulative_ic"] = ic_series["ic"].cumsum()
    ic_series["cumulative_rank_ic"] = ic_series["rank_ic"].cumsum()
    summary_rows = []
    for metric, col in [("IC", "ic"), ("RankIC", "rank_ic")]:
        stats = mean_std_t_p(ic_series[col])
        ir = stats["mean"] / stats["std"] if stats["std"] and not pd.isna(stats["std"]) else math.nan
        summary_rows.append({"metric": metric, **stats, "ir": ir})
    return ic_series, pd.DataFrame(summary_rows)


def calculate_factor_value_statistics(factor_data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["momentum_raw", "momentum_3sigma", "momentum_zscore"]:
        if col not in factor_data.columns:
            continue
        values = pd.to_numeric(factor_data[col], errors="coerce").dropna()
        rows.append(
            {
                "factor_column": col,
                "count": len(values),
                "mean": values.mean(),
                "std": values.std(ddof=1),
                "min": values.min(),
                "p25": values.quantile(0.25),
                "median": values.median(),
                "p75": values.quantile(0.75),
                "max": values.max(),
            }
        )
    return pd.DataFrame(rows)


def summarize_factor_input(factor_data: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ["factor_rows", len(factor_data)],
        ["stock_count", safe_nunique(factor_data, "stock_code")],
        ["trade_date_count", safe_nunique(factor_data, "trade_date")],
        ["valid_factor_rows", int(factor_data["momentum_zscore"].notna().sum()) if "momentum_zscore" in factor_data else 0],
        ["valid_next_return_rows", int(factor_data["next_period_return"].notna().sum()) if "next_period_return" in factor_data else 0],
        ["next_period_return_column_used", ",".join(factor_data["next_period_return_column_used"].dropna().astype(str).unique()) if "next_period_return_column_used" in factor_data else ""],
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def calculate_drawdown_series(nav_series: pd.Series) -> pd.DataFrame:
    running_max = nav_series.cummax()
    drawdown = nav_series / running_max - 1.0
    return pd.DataFrame({"nav": nav_series, "running_max": running_max, "drawdown": drawdown})


def calculate_performance_metrics(return_series: pd.Series, annualization_periods: float) -> dict[str, float]:
    returns = pd.to_numeric(return_series, errors="coerce").dropna()
    if returns.empty:
        return {
            "observation_count": 0,
            "final_nav": math.nan,
            "cumulative_return": math.nan,
            "annual_return": math.nan,
            "annual_volatility": math.nan,
            "sharpe_ratio": math.nan,
            "max_drawdown": math.nan,
            "win_rate": math.nan,
        }
    nav = (1.0 + returns).cumprod()
    final_nav = float(nav.iloc[-1])
    annual_return = final_nav ** (annualization_periods / len(returns)) - 1.0 if final_nav > 0 else math.nan
    annual_volatility = float(returns.std(ddof=1) * math.sqrt(annualization_periods))
    sharpe = annual_return / annual_volatility if annual_volatility and not pd.isna(annual_volatility) else math.nan
    max_drawdown = float((nav / nav.cummax() - 1.0).min())
    return {
        "observation_count": len(returns),
        "final_nav": final_nav,
        "cumulative_return": final_nav - 1.0,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": float((returns > 0).mean()),
    }


def calculate_additive_nav_metrics(
    return_series: pd.Series,
    nav_series: pd.Series | None,
    annualization_periods: float,
) -> dict[str, float]:
    returns = pd.to_numeric(return_series, errors="coerce").dropna()
    if returns.empty:
        return {
            "observation_count": 0,
            "final_nav": math.nan,
            "cumulative_return": math.nan,
            "annual_return": math.nan,
            "annual_volatility": math.nan,
            "sharpe_ratio": math.nan,
            "max_drawdown": math.nan,
            "win_rate": math.nan,
            "return_curve_method": "additive_daily_return_nav",
        }

    nav = (
        pd.to_numeric(nav_series, errors="coerce").dropna()
        if nav_series is not None
        else pd.Series(dtype="float64")
    )
    cumulative_return = float(returns.sum())
    final_nav = float(nav.iloc[-1]) if not nav.empty else 1.0 + cumulative_return
    annual_return = float(returns.mean() * annualization_periods)
    annual_volatility = float(returns.std(ddof=1) * math.sqrt(annualization_periods)) if len(returns) > 1 else math.nan
    sharpe = annual_return / annual_volatility if annual_volatility and not pd.isna(annual_volatility) else math.nan
    max_drawdown = (
        float(calculate_drawdown_series(nav)["drawdown"].min())
        if not nav.empty
        else math.nan
    )
    return {
        "observation_count": len(returns),
        "final_nav": final_nav,
        "cumulative_return": cumulative_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": float((returns > 0).mean()),
        "return_curve_method": "additive_daily_return_nav",
    }


def calculate_performance_attribution(
    long_short_returns: pd.DataFrame,
    holding_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    annualization_periods = TRADING_DAYS_PER_YEAR / max(int(holding_days), 1)
    portfolios = [
        ("long_top_group", "long_top_group_return", "long_top_group_nav"),
        ("short_bottom_group", "short_bottom_group_return", "short_bottom_group_nav"),
        ("high_minus_low_spread", "long_short_spread_return", "long_short_spread_nav"),
    ]
    summary_rows = []
    drawdown_parts = []
    yearly_rows = []
    for portfolio, return_col, nav_col in portfolios:
        metrics = calculate_performance_metrics(long_short_returns[return_col], annualization_periods)
        summary_rows.append({"portfolio": portfolio, **metrics, "annualization_periods_per_year": annualization_periods})
        drawdown = calculate_drawdown_series(pd.to_numeric(long_short_returns[nav_col], errors="coerce"))
        drawdown["trade_date"] = long_short_returns["trade_date"].values
        drawdown["portfolio"] = portfolio
        drawdown_parts.append(drawdown)
        temp = long_short_returns[["trade_date", return_col]].copy()
        temp["year"] = pd.to_datetime(temp["trade_date"]).dt.year
        for year, one_year in temp.groupby("year"):
            year_return = (1.0 + pd.to_numeric(one_year[return_col], errors="coerce").dropna()).prod() - 1.0
            yearly_rows.append({"portfolio": portfolio, "year": year, "year_return": year_return, "observation_count": len(one_year)})
    return pd.DataFrame(summary_rows), pd.concat(drawdown_parts, ignore_index=True), pd.DataFrame(yearly_rows)


def format_percent(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "" if pd.isna(numeric) else f"{float(numeric):.2%}"


def format_number(value: object, digits: int = 3) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "" if pd.isna(numeric) else f"{float(numeric):.{digits}f}"


def build_portfolio_performance_tables(performance_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the terminal/report table for high momentum, low momentum, and spread portfolios."""

    portfolio_labels = {
        "long_top_group": "高动量组",
        "short_bottom_group": "低动量组",
        "high_minus_low_spread": "High-Low 价差组",
    }
    rows = []
    for portfolio, label in portfolio_labels.items():
        matched = performance_summary.loc[performance_summary["portfolio"].eq(portfolio)]
        if matched.empty:
            rows.append(
                {
                    "portfolio": portfolio,
                    "portfolio_label": label,
                    "cumulative_return": math.nan,
                    "annual_return": math.nan,
                    "annual_volatility": math.nan,
                    "sharpe": math.nan,
                    "max_drawdown": math.nan,
                    "win_rate": math.nan,
                    "sample_count": 0,
                }
            )
            continue
        row = matched.iloc[0]
        rows.append(
            {
                "portfolio": portfolio,
                "portfolio_label": label,
                "cumulative_return": row.get("cumulative_return", math.nan),
                "annual_return": row.get("annual_return", math.nan),
                "annual_volatility": row.get("annual_volatility", math.nan),
                "sharpe": row.get("sharpe_ratio", math.nan),
                "max_drawdown": row.get("max_drawdown", math.nan),
                "win_rate": row.get("win_rate", math.nan),
                "sample_count": int(row.get("observation_count", 0)) if pd.notna(row.get("observation_count", pd.NA)) else 0,
            }
        )

    detail = pd.DataFrame(rows)
    display = pd.DataFrame(
        {
            "组合": detail["portfolio_label"],
            "累计收益": detail["cumulative_return"].map(format_percent),
            "年化收益": detail["annual_return"].map(format_percent),
            "年化波动": detail["annual_volatility"].map(format_percent),
            "Sharpe": detail["sharpe"].map(format_number),
            "最大回撤": detail["max_drawdown"].map(format_percent),
            "胜率": detail["win_rate"].map(format_percent),
            "样本期数": detail["sample_count"].astype("Int64"),
        }
    )
    detail = detail[
        [
            "portfolio",
            "cumulative_return",
            "annual_return",
            "annual_volatility",
            "sharpe",
            "max_drawdown",
            "win_rate",
            "sample_count",
        ]
    ]
    return display, detail


def screening_summary(all_factor: pd.DataFrame) -> pd.DataFrame:
    if all_factor.empty:
        return pd.DataFrame(columns=["screening_reason", "record_count", "stock_count", "trade_date_count"])
    return (
        all_factor.groupby("screening_reason", dropna=False)
        .agg(record_count=("stock_code", "size"), stock_count=("stock_code", "nunique"), trade_date_count=("trade_date", "nunique"))
        .reset_index()
        .sort_values("record_count", ascending=False)
    )


def build_cross_section_stock_count(raw_factor: pd.DataFrame, min_count: int) -> pd.DataFrame:
    columns = ["trade_date", "screened_stock_count", "keep_for_grouping", "exclusion_reason"]
    if raw_factor.empty:
        return pd.DataFrame(columns=columns)
    count = (
        raw_factor.groupby("trade_date")["stock_code"]
        .nunique()
        .reset_index(name="screened_stock_count")
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    count["keep_for_grouping"] = count["screened_stock_count"].ge(min_count)
    count["exclusion_reason"] = np.where(
        count["keep_for_grouping"],
        "kept",
        f"cross_section_stock_count_less_than_{min_count}",
    )
    return count[columns]


def filter_by_cross_section_count(
    raw_factor: pd.DataFrame,
    cross_section_stock_count: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if raw_factor.empty or cross_section_stock_count.empty:
        return raw_factor.copy(), raw_factor.iloc[0:0].copy()
    keep_dates = set(cross_section_stock_count.loc[cross_section_stock_count["keep_for_grouping"], "trade_date"])
    keep_mask = raw_factor["trade_date"].isin(keep_dates)
    filtered = raw_factor.loc[keep_mask].copy()
    excluded = raw_factor.loc[~keep_mask].copy()
    if not excluded.empty:
        excluded["cross_section_exclusion_reason"] = "cross_section_stock_count_less_than_minimum"
    return filtered, excluded


def screening_filter_step_summary(
    all_factor: pd.DataFrame,
    lookback_days: int,
    max_peak_age_days: int,
    drawdown_threshold: float,
    zscore_window: int,
    zscore_limit: float,
) -> pd.DataFrame:
    columns = ["step", "description", "record_count", "stock_count", "trade_date_count"]
    if all_factor.empty:
        return pd.DataFrame(columns=columns)

    step1 = all_factor["has_pre_peak_low"].fillna(False) & all_factor["strong_rise_filter"].fillna(False)
    step2 = step1 & all_factor["peak_age_filter"].fillna(False)
    step3 = step2 & all_factor["drawdown_filter"].fillna(False)
    step4 = step3 & all_factor["post_peak_low_filter"].fillna(False)
    step5 = step4 & all_factor["pre_peak_zscore_filter"].fillna(False)
    masks = [
        ("observation_pool", "T日可观察股票池", pd.Series(True, index=all_factor.index)),
        ("step1_strong_rise", f"过去{lookback_days}个交易日内低点到唯一peak涨幅超过50%，peak为窗口内第一次最高收盘价", step1),
        ("step2_recent_peak", f"唯一peak距离T日不超过{max_peak_age_days}个交易日", step2),
        ("step3_peak_drawdown", f"唯一peak至T日跌幅超过{drawdown_threshold:.0%}", step3),
        ("step4_signal_is_low", "T日收盘价为peak后最低点", step4),
        ("step5_pre_peak_zscore", f"peak前{zscore_window}个交易日内ChangeRatio时序z-score绝对值最大值不超过{zscore_limit:g}", step5),
    ]
    rows = []
    for step, description, mask in masks:
        data = all_factor.loc[mask.fillna(False)]
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


def factor_step_log(
    raw_data: pd.DataFrame,
    clean_data: pd.DataFrame,
    all_factor: pd.DataFrame,
    raw_factor: pd.DataFrame,
    factor_3sigma: pd.DataFrame,
    factor_zscore: pd.DataFrame,
    ranked_factor: pd.DataFrame,
    grouped_factor: pd.DataFrame,
    quantile_returns: pd.DataFrame,
    long_short_returns: pd.DataFrame,
    ic_series: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        ["sql_raw_market_data", len(raw_data), safe_nunique(raw_data, "Stkcd"), safe_nunique(raw_data, "Trddt")],
        ["cleaned_mainboard_stock_pool", len(clean_data), safe_nunique(clean_data, "stock_code"), safe_nunique(clean_data, "trade_date")],
        ["all_strong_momentum_records", len(all_factor), safe_nunique(all_factor, "stock_code"), safe_nunique(all_factor, "trade_date")],
        ["valid_strong_momentum_records", len(raw_factor), safe_nunique(raw_factor, "stock_code"), safe_nunique(raw_factor, "trade_date")],
        ["three_sigma_processed", len(factor_3sigma), safe_nunique(factor_3sigma, "stock_code"), safe_nunique(factor_3sigma, "trade_date")],
        ["zscore_standardized", len(factor_zscore), safe_nunique(factor_zscore, "stock_code"), safe_nunique(factor_zscore, "trade_date")],
        ["ranked_factor", len(ranked_factor), safe_nunique(ranked_factor, "stock_code"), safe_nunique(ranked_factor, "trade_date")],
        ["quantile_grouped_factor", len(grouped_factor), safe_nunique(grouped_factor, "stock_code"), safe_nunique(grouped_factor, "trade_date")],
        ["quantile_equal_weight_returns", len(quantile_returns), safe_nunique(quantile_returns, "quantile_group"), safe_nunique(quantile_returns, "trade_date")],
        ["long_short_returns", len(long_short_returns), None, safe_nunique(long_short_returns, "trade_date")],
        ["ic_series", len(ic_series), None, safe_nunique(ic_series, "trade_date")],
    ]
    return pd.DataFrame(rows, columns=["step", "rows", "entity_count", "trade_date_count"])


def required_result_files(output_dir: Path) -> list[Path]:
    """主流程完整产物清单；全部存在时可直接复用，避免重复读取数据库和重算因子。"""

    return [
        output_dir / "03_strong_momentum_factor_all_stocks.csv",
        output_dir / "03_strong_momentum_screened_stock_pool.csv",
        output_dir / "03_strong_momentum_tradable_stock_pool.csv",
        output_dir / "strong_momentum_cross_section_excluded_records.csv",
        output_dir / "07_strong_momentum_factor_quantile_groups_with_forward_returns.csv",
        output_dir / "09_quantile_equal_weight_returns.csv",
        output_dir / "10_long_short_hedge_returns.csv",
        output_dir / "11_strong_momentum_portfolio_performance_table.csv",
        output_dir / "11_strong_momentum_portfolio_performance_detail.csv",
        output_dir / "12_strong_momentum_ic_series.csv",
        output_dir / "12_strong_momentum_ic_ir_summary.csv",
        output_dir / "12_strong_momentum_factor_value_statistics.csv",
        output_dir / "12_strong_momentum_factor_input_summary.csv",
        output_dir / "strong_momentum_filter_step_summary.csv",
        output_dir / "run_summary.csv",
    ]


def read_existing_run_summary(output_dir: Path) -> dict[str, str]:
    path = output_dir / "run_summary.csv"
    if not path.exists():
        return {}
    data = pd.read_csv(path, dtype=str)
    if not {"metric", "value"}.issubset(data.columns):
        return {}
    return dict(zip(data["metric"].astype(str), data["value"].astype(str)))


def has_complete_result_cache(output_dir: Path, args: argparse.Namespace, market_types: set[int]) -> bool:
    if not all(path.exists() for path in required_result_files(output_dir)):
        return False
    summary = read_existing_run_summary(output_dir)
    expected = {
        "factor_version": FACTOR_VERSION,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "holding_days": str(args.holding_days),
        "group_num": str(args.group_num),
        "min_cross_section_stock_count": str(args.min_cross_section_stock_count),
        "selected_market_types": ",".join(str(item) for item in sorted(market_types)),
        "rise_threshold": str(args.rise_threshold),
        "drawdown_threshold": str(args.drawdown_threshold),
        "min_float_market_value_dsmvosd_unit": str(args.min_float_market_value),
        "lookback_days": str(args.lookback_days),
        "max_peak_age_days": str(args.max_peak_age_days),
        "zscore_window": str(DEFAULT_ZSCORE_WINDOW),
        "zscore_limit": str(DEFAULT_ZSCORE_LIMIT),
        "zscore_ddof": str(DEFAULT_ZSCORE_DDOF),
    }
    return all(str(summary.get(key, "")) == value for key, value in expected.items())


def print_reused_result_cache(output_dir: Path) -> None:
    print("发现完整强动量因子结果缓存，直接复用并跳过本次重算。")
    print(f"结果目录：{output_dir}")
    for path in required_result_files(output_dir):
        print(f"已存在：{path.name}")



# ============================================================
# 8. Long-only 策略专用函数
# ============================================================


def build_long_only_experiment_folder_name(
    lookback_days: int,
    holding_days: int,
    market_types: set[int],
) -> str:
    """Long-only 版本不再使用 group_num；目录名只保留关键策略参数。"""

    return f"longonly_lb{lookback_days}_hd{holding_days}_mkt{market_types_to_tag(market_types)}"


def attach_long_only_execution_and_returns(
    signal_pool: pd.DataFrame,
    clean_data: pd.DataFrame,
    holding_days: int,
) -> pd.DataFrame:
    """
    给五重筛选后的股票池匹配 T+1 开盘可交易状态和未来持有期收益。

    交易口径：
    - T 日盘后产生信号；
    - T+1 日开盘价作为可交易检查和买入执行价格；
    - 持有 holding_days 个交易日区间，收益计量到 T+(holding_days+1) 收盘；
    - 不建模卖出交易，持有期末收盘价仅作为事件收益的计量价格；
    - 若 T+1 开盘价等于涨停价或跌停价，视为不可交易，不纳入当期做多组合；
    - 若缺少 T+1 或持有期末价格，也视为不可交易。
    """

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
            "buy_rank_by_factor_desc",
        ]
        return pd.DataFrame(columns=columns)

    # 将五重筛选后的 T 日信号映射到 T+1 实际买入日和持有结束日。
    signals = normalize_date_columns(signal_pool).copy()
    clean = normalize_date_columns(clean_data).copy()

    trade_calendar = clean[["trade_date"]].drop_duplicates().sort_values("trade_date").reset_index(drop=True)
    terminal_offset_days = max(int(holding_days), 1)
    trade_calendar["next_trade_date"] = trade_calendar["trade_date"].shift(-1)
    trade_calendar["holding_start_trade_date"] = trade_calendar["next_trade_date"]
    trade_calendar["holding_end_trade_date"] = trade_calendar["trade_date"].shift(-terminal_offset_days)
    signals = signals.merge(trade_calendar, on="trade_date", how="left")

    # 先计算未考虑交易限制的未来收益，再叠加 T+1 开盘涨跌停和缺失数据过滤。
    forward_returns = calculate_forward_holding_returns(clean, holding_days)
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

    # 保留收盘涨跌停字段用于诊断；真实买入过滤以 T+1 开盘是否一字涨跌停为准。
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

    # T+1 开盘缺失、未来收益不完整、或 T+1 开盘一字涨跌停，均不能进入实际可买组合。
    missing_next = signals["next_open_price"].isna() | signals["next_period_return_before_trade_filter"].isna()
    signals["is_tradable_next_open"] = (~missing_next & ~signals["is_next_open_one_word_limit"]).fillna(False).astype(bool)
    signals["is_tradable_next_close"] = signals["is_tradable_next_open"]
    # 只有 T+1 开盘可交易的信号才保留 next_period_return。
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
    signals["next_period_return_column_used"] = "compound_dretwd_tplus1_to_holding_end_open_entry"
    signals["rebalance_frequency"] = "daily"
    signals["holding_days"] = holding_days
    signals["holding_terminal_offset_days"] = terminal_offset_days

    # 同一信号日内，对所有 T+1 开盘可买股票等权。
    tradable_count = signals.groupby("trade_date")["is_tradable_next_open"].transform("sum")
    signals["long_only_weight_equal"] = np.where(
        signals["is_tradable_next_open"] & tradable_count.gt(0),
        1.0 / tradable_count.replace(0, np.nan),
        0.0,
    )
    signals = attach_buy_rank_by_factor_desc(signals, tradable_col="is_tradable_next_open")
    return sort_by_execution_date_and_factor(signals)


def build_tradable_open_buy_dates(signal_pool_with_returns: pd.DataFrame) -> pd.DataFrame:
    """汇总每个 T+1 开盘实际可买入的交易日和可买股票数量。"""

    columns = ["交易日", "可买股票数", "对应信号日"]
    if signal_pool_with_returns.empty or "holding_start_trade_date" not in signal_pool_with_returns.columns:
        return pd.DataFrame(columns=columns)

    data = normalize_date_columns(
        signal_pool_with_returns,
        columns=["trade_date", "holding_start_trade_date"],
    )
    tradable_col = "is_tradable_next_open" if "is_tradable_next_open" in data.columns else "is_tradable_next_close"
    if tradable_col not in data.columns:
        return pd.DataFrame(columns=columns)

    tradable_mask = data[tradable_col]
    if tradable_mask.dtype != bool:
        tradable_mask = tradable_mask.astype(str).str.lower().isin(["true", "1", "1.0"])
    # 输出表面向人工检查：交易日是 T+1 开盘买入日，对应信号日是 T 日。
    tradable = data.loc[tradable_mask.fillna(False)].copy()
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
        .sort_values("holding_start_trade_date")
    )
    summary["交易日"] = pd.to_datetime(summary["holding_start_trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return summary[columns]


def attach_non_compound_long_only_event_curve(portfolio: pd.DataFrame) -> pd.DataFrame:
    output = portfolio.copy()
    returns = pd.to_numeric(output["long_only_return"], errors="coerce").fillna(0.0)
    curve_start_values = []
    curve_end_values = []
    cumulative_start_values = []
    cumulative_end_values = []
    current_curve = 1.0
    current_cumulative_return = 0.0
    for period_return in returns.to_numpy(dtype="float64"):
        curve_start_values.append(current_curve)
        cumulative_start_values.append(current_cumulative_return)
        current_cumulative_return = current_cumulative_return + period_return
        current_curve = 1.0 + current_cumulative_return
        cumulative_end_values.append(current_cumulative_return)
        curve_end_values.append(current_curve)
    output["long_only_nav_start"] = curve_start_values
    output["long_only_nav"] = curve_end_values
    output["long_only_cumulative_event_return_start"] = cumulative_start_values
    output["long_only_cumulative_event_return"] = cumulative_end_values
    output["nav_update_rule"] = "long_only_event_curve_t_plus_1=long_only_event_curve_t+long_only_return_t"
    output["nav_curve_semantics"] = "non_compound_event_cumulative_return_curve_not_tradeable_nav"
    return output


def attach_additive_long_only_nav(portfolio: pd.DataFrame) -> pd.DataFrame:
    output = portfolio.copy()
    returns = pd.to_numeric(output["long_only_return"], errors="coerce").fillna(0.0)
    nav_start_values = []
    nav_end_values = []
    current_nav = 1.0
    for period_return in returns.to_numpy(dtype="float64"):
        nav_start_values.append(current_nav)
        # 用户指定的净值口径：NAV_{t+1} = NAV_t + long_only_return_t，不做重复复利。
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
    """
    构造真实市场口径的 long-only 每日 NAV。

    T 日信号在 T+1 开盘建仓；每个信号日 cohort 占用 1/holding_days 组合资金；
    cohort 内股票等权，持有 holding_days 个 close-to-close 日收益区间。
    """

    clean = normalize_date_columns(clean_data).copy()
    clean["close_price"] = pd.to_numeric(clean["close_price"], errors="coerce")
    clean["return_with_dividend"] = pd.to_numeric(clean["return_with_dividend"], errors="coerce")
    holding_days = max(int(holding_days), 1)
    terminal_offset_days = holding_days

    # 构造完整交易日历，用于把每个 T+1 开仓信号展开成持有期内每日持仓。
    calendar_all = clean[["trade_date"]].drop_duplicates().sort_values("trade_date").reset_index(drop=True)
    calendar_all["calendar_index"] = np.arange(len(calendar_all), dtype="int64")
    calendar_all["next_trade_date"] = calendar_all["trade_date"].shift(-1)
    calendar_all["holding_start_trade_date"] = calendar_all["next_trade_date"]
    calendar_all["holding_end_trade_date"] = calendar_all["trade_date"].shift(-terminal_offset_days)

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
        # 真实组合只使用 T+1 开盘可交易的信号；不可交易信号仅保留在诊断统计中。
        signals = normalize_date_columns(signal_pool_with_returns).copy()
        signals = signals.loc[signals["trade_date"].between(start_ts, end_ts)].copy()
        tradable_col = "is_tradable_next_open" if "is_tradable_next_open" in signals.columns else None
        if tradable_col is None and "is_tradable_next_close" in signals.columns:
            tradable_col = "is_tradable_next_close"
        limit_col = "is_next_open_one_word_limit" if "is_next_open_one_word_limit" in signals.columns else None
        if limit_col is None and "is_next_close_limit" in signals.columns:
            limit_col = "is_next_close_limit"
        entry_price_col = "next_open_price" if "next_open_price" in signals.columns else None
        if entry_price_col is None and "next_close_price" in signals.columns:
            entry_price_col = "next_close_price"

        tradable_mask = signals[tradable_col].fillna(False).astype(bool) if tradable_col else pd.Series(False, index=signals.index)
        limit_mask = signals[limit_col].fillna(False).astype(bool) if limit_col else pd.Series(False, index=signals.index)
        missing_next = pd.Series(False, index=signals.index)
        if entry_price_col:
            missing_next |= signals[entry_price_col].isna()
        if "next_period_return_before_trade_filter" in signals.columns:
            missing_next |= signals["next_period_return_before_trade_filter"].isna()

        if "long_only_weight_equal" not in signals.columns:
            tradable_count = tradable_mask.groupby(signals["trade_date"]).transform("sum")
            signals["long_only_weight_equal"] = np.where(tradable_mask & tradable_count.gt(0), 1.0 / tradable_count.replace(0, np.nan), 0.0)
        signals["long_only_weight_equal"] = pd.to_numeric(signals["long_only_weight_equal"], errors="coerce").fillna(0.0)

        for col in ["momentum_raw", "low_to_peak_return", "drawdown_from_peak_to_signal"]:
            if col not in signals.columns:
                signals[col] = np.nan
        for col in ["holding_start_trade_date", "holding_end_trade_date"]:
            if col not in signals.columns:
                signals[col] = pd.NaT

        signals["raw_signal_stock_count_flag"] = 1
        signals["tradable_signal_stock_count_flag"] = tradable_mask.astype("int64")
        signals["next_open_limit_excluded_flag"] = limit_mask.astype("int64")
        signals["missing_next_record_flag"] = missing_next.astype("int64")

        # 先按信号日汇总原始信号、可交易信号和被交易过滤剔除的数量。
        signal_agg = (
            signals.groupby("trade_date")
            .agg(
                raw_signal_stock_count=("raw_signal_stock_count_flag", "sum"),
                tradable_signal_stock_count=("tradable_signal_stock_count_flag", "sum"),
                next_close_limit_excluded_count=("next_open_limit_excluded_flag", "sum"),
                next_open_limit_excluded_count=("next_open_limit_excluded_flag", "sum"),
                missing_next_record_count=("missing_next_record_flag", "sum"),
                avg_signal_momentum_raw=("momentum_raw", "mean"),
                avg_low_to_peak_return=("low_to_peak_return", "mean"),
                avg_drawdown_from_peak_to_signal=("drawdown_from_peak_to_signal", "mean"),
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

        # 将每个可交易信号展开为 holding_days 个每日持仓切片，形成滚动 sleeve。
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
            # 每日组合收益 = 活跃股票日收益 * 信号日等权权重 / holding_days。
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

    # 合并每日信号统计和每日活跃持仓收益，形成最终 long-only 每日 NAV 序列。
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
    portfolio["holding_terminal_offset_days"] = terminal_offset_days
    portfolio["weighting_scheme"] = f"rolling_{holding_days}_equal_capital_sleeves_equal_weight_stocks"
    portfolio["portfolio_model"] = "rolling_sleeve_dretwd_additive_nav"

    return portfolio.sort_values("trade_date").reset_index(drop=True)


def calculate_long_only_event_returns_legacy(
    signal_pool_with_returns: pd.DataFrame,
    clean_data: pd.DataFrame,
    holding_days: int,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Build an event-test daily NAV.
    For signal day T, execution is T+1 open. Each tradable stock return is:
    product(Dretwd from T+1 through T+holding_days) - 1. The portfolio return
    is the average across stocks sharing the same T+1 execution date, then the
    event curve compounds those daily event returns.
    """

    clean = normalize_date_columns(clean_data).copy()
    holding_days = max(int(holding_days), 1)
    terminal_offset_days = holding_days

    calendar_all = clean[["trade_date"]].drop_duplicates().sort_values("trade_date").reset_index(drop=True)
    calendar_all["calendar_index"] = np.arange(len(calendar_all), dtype="int64")
    calendar_all["next_trade_date"] = calendar_all["trade_date"].shift(-1)
    calendar_all["holding_start_trade_date"] = calendar_all["trade_date"]
    calendar_all["holding_end_trade_date"] = calendar_all["trade_date"].shift(-(holding_days - 1))

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    portfolio = calendar_all.loc[calendar_all["trade_date"].between(start_ts, end_ts)].copy()

    if signal_pool_with_returns.empty:
        signal_agg = pd.DataFrame(columns=["trade_date"])
        event_agg = pd.DataFrame(columns=["trade_date"])
    else:
        signals = normalize_date_columns(signal_pool_with_returns).copy()
        if "holding_start_trade_date" not in signals.columns:
            signals["holding_start_trade_date"] = signals["next_trade_date"] if "next_trade_date" in signals.columns else pd.NaT
        if "holding_end_trade_date" not in signals.columns:
            signals["holding_end_trade_date"] = pd.NaT
        if "next_period_return" not in signals.columns:
            signals["next_period_return"] = (
                signals["next_period_return_before_trade_filter"]
                if "next_period_return_before_trade_filter" in signals.columns
                else np.nan
            )

        signals["execution_trade_date"] = pd.to_datetime(signals["holding_start_trade_date"], errors="coerce").dt.normalize()
        signals = signals.loc[signals["execution_trade_date"].between(start_ts, end_ts)].copy()
        signals["next_period_return"] = pd.to_numeric(signals["next_period_return"], errors="coerce").replace([np.inf, -np.inf], np.nan)

        tradable_col = "is_tradable_next_open" if "is_tradable_next_open" in signals.columns else None
        if tradable_col is None and "is_tradable_next_close" in signals.columns:
            tradable_col = "is_tradable_next_close"
        limit_col = "is_next_open_one_word_limit" if "is_next_open_one_word_limit" in signals.columns else None
        if limit_col is None and "is_next_close_limit" in signals.columns:
            limit_col = "is_next_close_limit"
        entry_price_col = "next_open_price" if "next_open_price" in signals.columns else None
        if entry_price_col is None and "next_close_price" in signals.columns:
            entry_price_col = "next_close_price"

        raw_tradable_mask = signals[tradable_col].fillna(False).astype(bool) if tradable_col else signals["next_period_return"].notna()
        tradable_mask = raw_tradable_mask & signals["next_period_return"].notna()
        limit_mask = signals[limit_col].fillna(False).astype(bool) if limit_col else pd.Series(False, index=signals.index)
        missing_next = pd.Series(False, index=signals.index)
        if entry_price_col:
            missing_next |= signals[entry_price_col].isna()
        if "next_period_return_before_trade_filter" in signals.columns:
            missing_next |= signals["next_period_return_before_trade_filter"].isna()

        if "long_only_weight_equal" not in signals.columns:
            tradable_count = tradable_mask.groupby(signals["execution_trade_date"]).transform("sum")
            signals["long_only_weight_equal"] = np.where(tradable_mask & tradable_count.gt(0), 1.0 / tradable_count.replace(0, np.nan), 0.0)
        signals["long_only_weight_equal"] = pd.to_numeric(signals["long_only_weight_equal"], errors="coerce").fillna(0.0)

        for col in ["momentum_raw", "low_to_peak_return", "drawdown_from_peak_to_signal"]:
            if col not in signals.columns:
                signals[col] = np.nan

        signals["raw_signal_stock_count_flag"] = 1
        signals["tradable_signal_stock_count_flag"] = tradable_mask.astype("int64")
        signals["next_open_limit_excluded_flag"] = limit_mask.astype("int64")
        signals["missing_next_record_flag"] = missing_next.astype("int64")

        if signals.empty:
            signal_agg = pd.DataFrame(columns=["trade_date"])
            event_agg = pd.DataFrame(columns=["trade_date"])
        else:
            signal_agg = (
                signals.groupby("execution_trade_date")
                .agg(
                    raw_signal_stock_count=("raw_signal_stock_count_flag", "sum"),
                    tradable_signal_stock_count=("tradable_signal_stock_count_flag", "sum"),
                    next_close_limit_excluded_count=("next_open_limit_excluded_flag", "sum"),
                    next_open_limit_excluded_count=("next_open_limit_excluded_flag", "sum"),
                    missing_next_record_count=("missing_next_record_flag", "sum"),
                    avg_signal_momentum_raw=("momentum_raw", "mean"),
                    avg_low_to_peak_return=("low_to_peak_return", "mean"),
                    avg_drawdown_from_peak_to_signal=("drawdown_from_peak_to_signal", "mean"),
                    holding_start_trade_date_min=("holding_start_trade_date", "min"),
                    holding_start_trade_date_max=("holding_start_trade_date", "max"),
                    holding_end_trade_date_min=("holding_end_trade_date", "min"),
                    holding_end_trade_date_max=("holding_end_trade_date", "max"),
                )
                .reset_index()
                .rename(columns={"execution_trade_date": "trade_date"})
            )

            tradable = signals.loc[tradable_mask].copy()
            tradable["stock_forward_return"] = pd.to_numeric(tradable["next_period_return"], errors="coerce")
            tradable = tradable.loc[tradable["stock_forward_return"].notna()].copy()
            if tradable.empty:
                event_agg = pd.DataFrame(columns=["trade_date"])
            else:
                event_agg = (
                    tradable.groupby("execution_trade_date")
                    .agg(
                        long_only_return_sum=("stock_forward_return", "sum"),
                        long_only_return=("stock_forward_return", "mean"),
                        active_stock_lot_count=("stock_code", "size"),
                        active_stock_count=("stock_code", "nunique"),
                        active_signal_sleeve_count=("trade_date", "nunique"),
                        active_missing_stock_return_count=("stock_forward_return", lambda series: int(series.isna().sum())),
                        active_signal_trade_date_min=("trade_date", "min"),
                        active_signal_trade_date_max=("trade_date", "max"),
                    )
                    .reset_index()
                    .rename(columns={"execution_trade_date": "trade_date"})
                )
                event_agg["active_sleeve_return_sum"] = event_agg["long_only_return_sum"]

    portfolio = portfolio.merge(signal_agg, on="trade_date", how="left").merge(event_agg, on="trade_date", how="left")
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

    for col in ["active_sleeve_return_sum", "long_only_return_sum", "long_only_return"]:
        if col not in portfolio.columns:
            portfolio[col] = 0.0
        portfolio[col] = pd.to_numeric(portfolio[col], errors="coerce").fillna(0.0)

    portfolio = attach_non_compound_long_only_event_curve(portfolio)
    portfolio["has_raw_signal"] = portfolio["raw_signal_stock_count"].gt(0)
    portfolio["has_tradable_signal"] = portfolio["tradable_signal_stock_count"].gt(0)
    portfolio["has_active_position"] = portfolio["has_tradable_signal"]
    portfolio["portfolio_invested_fraction"] = np.where(portfolio["has_tradable_signal"], 1.0, 0.0)
    portfolio["cash_reason"] = np.select(
        [
            ~portfolio["has_raw_signal"].to_numpy(dtype=bool),
            portfolio["has_raw_signal"].to_numpy(dtype=bool) & ~portfolio["has_tradable_signal"].to_numpy(dtype=bool),
        ],
        [
            "no_signal_hold_cash",
            "signals_filtered_by_trade_or_return_hold_cash",
        ],
        default="fully_invested_daily_event_portfolio",
    )
    portfolio["rebalance_frequency"] = "daily_signal_generation_entry_next_open"
    portfolio["holding_days"] = holding_days
    portfolio["holding_terminal_offset_days"] = terminal_offset_days
    portfolio["return_aggregation_rule"] = "average_stock_forward_compound_dretwd_T_plus_1_to_T_plus_holding_days_by_execution_day"
    portfolio["weighting_scheme"] = f"daily_equal_weight_stocks_forward_T_plus_1_to_T_plus_{holding_days}_dretwd_event_return"
    portfolio["portfolio_model"] = "daily_event_forward_dretwd_equal_weight_non_compound_curve"

    return portfolio.sort_values("trade_date").reset_index(drop=True)


def calculate_long_only_performance_attribution(
    long_only_returns: pd.DataFrame,
    holding_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    annualization_periods = TRADING_DAYS_PER_YEAR
    active_mask = (
        long_only_returns["has_active_position"].fillna(False).astype(bool)
        if "has_active_position" in long_only_returns.columns
        else long_only_returns["has_tradable_signal"].fillna(False).astype(bool)
    )
    metrics = calculate_additive_nav_metrics(
        long_only_returns["long_only_return"],
        long_only_returns.get("long_only_nav"),
        annualization_periods,
    )
    performance_summary = pd.DataFrame(
        [
            {
                "portfolio": "factor1_long_only",
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

    drawdown = calculate_drawdown_series(pd.to_numeric(long_only_returns["long_only_nav"], errors="coerce"))
    drawdown["trade_date"] = long_only_returns["trade_date"].values
    drawdown["portfolio"] = "factor1_long_only"

    yearly_rows = []
    temp = long_only_returns[["trade_date", "long_only_return"]].copy()
    temp["year"] = pd.to_datetime(temp["trade_date"]).dt.year
    for year, one_year in temp.groupby("year"):
        year_return = float(pd.to_numeric(one_year["long_only_return"], errors="coerce").dropna().sum())
        yearly_rows.append({"portfolio": "factor1_long_only", "year": year, "year_return": year_return, "observation_count": len(one_year)})
    return performance_summary, drawdown, pd.DataFrame(yearly_rows)


def build_long_only_performance_tables(performance_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if performance_summary.empty:
        detail = pd.DataFrame(
            [
                {
                    "portfolio": "factor1_long_only",
                    "cumulative_return": math.nan,
                    "annual_return": math.nan,
                    "annual_volatility": math.nan,
                    "sharpe": math.nan,
                    "max_drawdown": math.nan,
                    "win_rate": math.nan,
                    "sample_count": 0,
                    "cash_period_ratio": math.nan,
                    "avg_tradable_signal_stock_count": math.nan,
                }
            ]
        )
    else:
        row = performance_summary.iloc[0]
        detail = pd.DataFrame(
            [
                {
                    "portfolio": row.get("portfolio", "factor1_long_only"),
                    "cumulative_return": row.get("cumulative_return", math.nan),
                    "annual_return": row.get("annual_return", math.nan),
                    "annual_volatility": row.get("annual_volatility", math.nan),
                    "sharpe": row.get("sharpe_ratio", math.nan),
                    "max_drawdown": row.get("max_drawdown", math.nan),
                    "win_rate": row.get("win_rate", math.nan),
                    "sample_count": int(row.get("observation_count", 0)) if pd.notna(row.get("observation_count", pd.NA)) else 0,
                    "cash_period_ratio": row.get("cash_period_ratio", math.nan),
                    "avg_tradable_signal_stock_count": row.get("avg_tradable_signal_stock_count", math.nan),
                }
            ]
        )

    display = pd.DataFrame(
        {
            "组合": ["强势回调 Long-only 组合"],
            "累计收益": detail["cumulative_return"].map(format_percent),
            "年化收益": detail["annual_return"].map(format_percent),
            "年化波动": detail["annual_volatility"].map(format_percent),
            "Sharpe": detail["sharpe"].map(format_number),
            "最大回撤": detail["max_drawdown"].map(format_percent),
            "胜率": detail["win_rate"].map(format_percent),
            "样本期数": detail["sample_count"].astype("Int64"),
            "空仓比例": detail["cash_period_ratio"].map(format_percent),
            "平均可交易信号数": detail["avg_tradable_signal_stock_count"].map(lambda x: format_number(x, 2)),
        }
    )
    return display, detail


def configure_matplotlib_chinese_font(plt) -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def plot_long_only_nav_curve(long_only_returns: pd.DataFrame, output_path: Path) -> Path:
    import matplotlib.pyplot as plt

    configure_matplotlib_chinese_font(plt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 主程序只输出这一张 long-only NAV 图，并在图上写入 Final NAV、CumRet、Sharpe、MaxDD。
    data = normalize_date_columns(long_only_returns).sort_values("trade_date").copy()
    data["long_only_nav"] = pd.to_numeric(data.get("long_only_nav", pd.Series(dtype="float64")), errors="coerce")
    data["long_only_return"] = pd.to_numeric(data.get("long_only_return", pd.Series(dtype="float64")), errors="coerce")
    holding_days = int(pd.to_numeric(data["holding_days"], errors="coerce").dropna().iloc[0]) if "holding_days" in data.columns and data["holding_days"].notna().any() else DEFAULT_HOLDING_DAYS

    fig, ax = plt.subplots(figsize=(18, 7), dpi=200)
    if data.empty or data["long_only_nav"].dropna().empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No NAV data", ha="center", va="center", fontsize=14)
    else:
        plot_data = data.dropna(subset=["trade_date", "long_only_nav"]).copy()
        ax.plot(plot_data["trade_date"], plot_data["long_only_nav"], linewidth=1.8, label="Long-only NAV")
        ax.axhline(1.0, color="#777777", linestyle="--", linewidth=1.0)
        ax.set_title(f"Long-only NAV", fontsize=15, pad=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("NAV")
        ax.grid(True, linestyle="-", alpha=0.25)
        ax.legend(loc="best")

        metrics = calculate_additive_nav_metrics(plot_data["long_only_return"], plot_data["long_only_nav"], TRADING_DAYS_PER_YEAR)
        final_nav = float(plot_data["long_only_nav"].iloc[-1])
        summary = (
            f"Final NAV = {final_nav:.4f}    "
            f"CumRet = {final_nav - 1.0:.2%}    "
            f"Sharpe = {metrics.get('sharpe_ratio', math.nan):.4f}    "
            f"MaxDD = {metrics.get('max_drawdown', math.nan):.2%}"
        )
        fig.text(0.5, 0.975, summary, ha="center", va="top", fontsize=10.5)
        fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def parse_optional_trade_detail_date(text: str | None) -> pd.Timestamp | None:
    if text is None or str(text).strip() == "":
        return None
    parsed = pd.to_datetime(str(text).strip(), errors="coerce")
    if pd.isna(parsed):
        raise SystemExit(f"--trade-detail-date 日期格式不正确：{text}。请使用 YYYY-MM-DD。")
    return pd.Timestamp(parsed).normalize()


def build_trade_detail_message(message: str, output_path: Path, trade_date: pd.Timestamp) -> Path:
    import matplotlib.pyplot as plt

    configure_matplotlib_chinese_font(plt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 3.6), dpi=180)
    ax.axis("off")
    ax.text(
        0.5,
        0.58,
        message,
        ha="center",
        va="center",
        fontsize=15,
        fontproperties=None,
        wrap=True,
    )
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
    clean_data: pd.DataFrame,
    signal_pool_with_returns: pd.DataFrame,
    output_dir: Path,
    lookback_days: int,
) -> Path | None:
    if trade_detail_date is None:
        return None

    trade_date = pd.Timestamp(trade_detail_date).normalize()
    image_path = output_dir / f"trade_detail_{trade_date.strftime('%Y%m%d')}.png"
    calendar = normalize_date_columns(clean_data)[["trade_date"]].drop_duplicates().sort_values("trade_date")
    calendar_dates = pd.to_datetime(calendar["trade_date"]).dt.normalize()

    if not calendar_dates.eq(trade_date).any():
        message = "该日期不是清洗后主板股票池的交易日，请输入正确交易日期。"
        print(message)
        return build_trade_detail_message(message, image_path, trade_date)

    first_available_trade_date = (
        pd.Timestamp(calendar_dates.iloc[lookback_days]).normalize()
        if len(calendar_dates) > lookback_days
        else pd.NaT
    )

    if pd.notna(first_available_trade_date) and trade_date < first_available_trade_date:
        message = (
            f"该日期早于策略首个可用交易执行日 {first_available_trade_date.strftime('%Y-%m-%d')}。"
            "2018 年前约 7 个自然月因历史不足 147 个交易日不可用，请输入正确日期。"
        )
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
    display["买入开盘价"] = pd.to_numeric(display["holding_entry_open_price"], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    display["组合权重"] = pd.to_numeric(display["long_only_weight_equal"], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    display["低点到峰值涨幅"] = pd.to_numeric(display["low_to_peak_return"], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    display["峰值回撤"] = pd.to_numeric(display["drawdown_from_peak_to_signal"], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    display["峰值日期"] = pd.to_datetime(display["peak_trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    display["持有结束日期"] = pd.to_datetime(display["holding_end_trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    table_columns = ["因子排名", "买入日期", "信号日期", "股票代码", "因子值", "买入开盘价", "组合权重", "低点到峰值涨幅", "峰值回撤", "峰值日期", "持有结束日期"]
    table_data = display[table_columns]

    import matplotlib.pyplot as plt

    configure_matplotlib_chinese_font(plt)
    row_count = len(table_data)
    fig_height = min(max(3.8, 1.3 + 0.38 * (row_count + 1)), 20)
    fig, ax = plt.subplots(figsize=(15.5, fig_height), dpi=180)
    ax.axis("off")
    title = f"{trade_date.strftime('%Y-%m-%d')} 开盘买入股票明细（共 {row_count} 只）"
    ax.set_title(title, fontsize=16, pad=16)
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
    image_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"已输出交易明细图片：{image_path}")
    return image_path


def long_only_factor_step_log(
    raw_data: pd.DataFrame,
    clean_data: pd.DataFrame,
    all_factor: pd.DataFrame,
    raw_factor: pd.DataFrame,
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
        ["cleaned_mainboard_stock_pool", len(clean_data), safe_nunique(clean_data, "stock_code"), safe_nunique(clean_data, "trade_date")],
        ["all_strong_momentum_records", len(all_factor), safe_nunique(all_factor, "stock_code"), safe_nunique(all_factor, "trade_date")],
        ["screened_long_only_signal_pool", len(raw_factor), safe_nunique(raw_factor, "stock_code"), safe_nunique(raw_factor, "trade_date")],
        ["signal_pool_with_forward_returns", len(signal_pool_with_returns), safe_nunique(signal_pool_with_returns, "stock_code"), safe_nunique(signal_pool_with_returns, "trade_date")],
        ["tradable_long_only_signal_records", int(tradable_signals[tradable_col].sum()) if tradable_col in tradable_signals else 0, safe_nunique(tradable_signals, "stock_code"), safe_nunique(tradable_signals, "trade_date")],
        ["long_only_return_rows", len(long_only_returns), None, safe_nunique(long_only_returns, "trade_date")],
    ]
    return pd.DataFrame(rows, columns=["step", "rows", "entity_count", "trade_date_count"])


def required_long_only_result_files(output_dir: Path) -> list[Path]:
    return [
        output_dir / "03_strong_momentum_factor_all_stocks.csv",
        output_dir / "03_factor1_long_only_screened_stock_pool.csv",
        output_dir / "04_factor1_long_only_signal_pool_with_forward_returns.csv",
        output_dir / "05_factor1_long_only_holding_period_returns.csv",
        output_dir / "05_factor1_long_only_nav_curve.png",
        output_dir / "06_factor1_long_only_portfolio_performance_table.csv",
        output_dir / "06_factor1_long_only_portfolio_performance_detail.csv",
        output_dir / "07_factor1_long_only_drawdown_series.csv",
        output_dir / "08_factor1_long_only_yearly_performance.csv",
        output_dir / "strong_momentum_filter_step_summary.csv",
        output_dir / "run_summary.csv",
    ]


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
        "rise_threshold": str(args.rise_threshold),
        "drawdown_threshold": str(args.drawdown_threshold),
        "min_float_market_value_dsmvosd_unit": str(args.min_float_market_value),
        "lookback_days": str(args.lookback_days),
        "max_peak_age_days": str(args.max_peak_age_days),
        "zscore_window": str(DEFAULT_ZSCORE_WINDOW),
        "zscore_limit": str(DEFAULT_ZSCORE_LIMIT),
        "zscore_ddof": str(DEFAULT_ZSCORE_DDOF),
    }
    return all(str(summary.get(key, "")) == value for key, value in expected.items())


def print_reused_long_only_result_cache(output_dir: Path) -> None:
    print("发现完整 factor1_LongOnly 结果缓存，直接复用并跳过本次重算。")
    print(f"结果目录：{output_dir}")
    for path in required_long_only_result_files(output_dir):
        print(f"已存在：{path.name}")


# ============================================================
# 9. 主流程：factor1_LongOnly
# ============================================================


def main() -> None:
    args = parse_args()
    market_types = set(DEFAULT_MARKET_TYPES)
    trade_detail_date = parse_optional_trade_detail_date(args.trade_detail_date)
    output_root: Path = args.output_dir
    experiment_folder_name = build_long_only_experiment_folder_name(
        args.lookback_days,
        args.holding_days,
        market_types,
    )
    output_dir = output_root / experiment_folder_name

    global LOCAL_RAW_DATA_CACHE_DIR, LOCAL_CLEAN_DATA_CACHE_DIR
    LOCAL_RAW_DATA_CACHE_DIR = output_root / "_local_data_cache"
    LOCAL_CLEAN_DATA_CACHE_DIR = output_root / "_local_data_cache"
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    refresh_requested = args.refresh_local_data or args.refresh_clean_data or trade_detail_date is not None
    if has_complete_long_only_result_cache(output_dir, args, market_types) and not args.force_recalculate_factor and not refresh_requested:
        print_reused_long_only_result_cache(output_dir)
        return

    temp_tsv_path = output_root / "_raw_market_data_for_factor1_longonly.tsv"
    raw_cache_path = build_local_raw_data_cache_path(args.start_date, args.end_date, market_types)
    clean_cache_paths = build_local_clean_data_cache_paths(
        args.start_date,
        args.end_date,
        args.min_float_market_value,
        market_types,
    )

    print("1/7 正在读取或生成清洗后主板行情数据...")
    # 第 1 步：优先复用清洗缓存；缓存缺失时读取原始行情并重新清洗。
    raw_data = pd.DataFrame()
    raw_data_source = "not_loaded_because_clean_cache_reused"
    if clean_cache_paths["clean_data"].exists() and not args.refresh_clean_data:
        print(f"读取本地清洗行情缓存：{clean_cache_paths['clean_data']}")
        clean_data, cleaning_log, missing_summary, exclusion_summary = load_clean_market_data_from_cache(clean_cache_paths)
        clean_data_source = "local_clean_cache"
    else:
        if args.refresh_clean_data:
            print("收到 --refresh-clean-data，忽略清洗缓存并重新清洗。")
        else:
            print("未找到清洗缓存，将优先读取原始行情缓存；若原始缓存也不存在才连接 MySQL。")
        raw_data, raw_cache_path, raw_data_source = load_raw_market_data_with_cache(
            args.start_date,
            args.end_date,
            market_types,
            temp_tsv_path,
            args.refresh_local_data,
            args.keep_temp,
        )
        clean_data, cleaning_log, missing_summary, exclusion_summary = clean_market_data(
            raw_data,
            min_float_market_value=args.min_float_market_value,
            market_types=market_types,
        )
        write_clean_market_data_cache(clean_data, cleaning_log, missing_summary, exclusion_summary, clean_cache_paths)
        clean_data_source = "fresh_cleaned_and_cached"

    print("2/7 正在执行强动量五重筛选...")
    # 第 2 步：逐股票逐日计算五条筛选条件，raw_factor 只包含全部通过的信号。
    raw_factor_all_dates, missing_factor_all_dates = calculate_raw_strong_momentum(
        clean_data,
        rise_threshold=args.rise_threshold,
        drawdown_threshold=args.drawdown_threshold,
        lookback_days=args.lookback_days,
        max_peak_age_days=args.max_peak_age_days,
        zscore_window=DEFAULT_ZSCORE_WINDOW,
        zscore_limit=DEFAULT_ZSCORE_LIMIT,
        zscore_ddof=DEFAULT_ZSCORE_DDOF,
    )
    all_factor_all_dates = build_all_factor(raw_factor_all_dates, missing_factor_all_dates)

    # 删除预热期：虽然计算需要历史窗口，但正式输出只保留回测区间内且历史足够的记录。
    start_ts = pd.to_datetime(args.start_date)
    end_ts = pd.to_datetime(args.end_date)
    raw_date_window = raw_factor_all_dates.loc[raw_factor_all_dates["trade_date"].between(start_ts, end_ts)].copy()
    all_date_window = all_factor_all_dates.loc[all_factor_all_dates["trade_date"].between(start_ts, end_ts)].copy()
    missing_date_window = missing_factor_all_dates.loc[missing_factor_all_dates["trade_date"].between(start_ts, end_ts)].copy()
    raw_history_mask = pd.to_numeric(raw_date_window["full_history_valid_days"], errors="coerce").ge(args.lookback_days)
    all_history_mask = pd.to_numeric(all_date_window["full_history_valid_days"], errors="coerce").ge(args.lookback_days)
    missing_history_mask = pd.to_numeric(missing_date_window["full_history_valid_days"], errors="coerce").ge(args.lookback_days)
    raw_factor = raw_date_window.loc[raw_history_mask].copy()
    all_factor = all_date_window.loc[all_history_mask].copy()
    missing_factor = missing_date_window.loc[missing_history_mask].copy()
    insufficient_history_rows = int((~all_history_mask).sum())

    print("3/7 正在匹配 T+1 开盘可交易状态和未来持有期收益...")
    # 第 3 步：把 T 日信号转换成 T+1 开盘买入检查，并计算持有期 Dretwd 事件收益。
    signal_pool_with_returns = attach_long_only_execution_and_returns(
        raw_factor,
        clean_data=clean_data,
        holding_days=args.holding_days,
    )

    # 如果指定 trade_detail_date，则输出该实际买入日的可交易股票明细图片。
    trade_detail_image_path = output_trade_detail_image(
        trade_detail_date,
        clean_data=clean_data,
        signal_pool_with_returns=signal_pool_with_returns,
        output_dir=output_dir,
        lookback_days=args.lookback_days,
    )

    print("4/7 正在计算 long-only 真实滚动持仓每日收益和 NAV...")
    # 第 4 步：把每日新信号展开成滚动持仓 sleeve，按 Dretwd 计算每日组合收益和 NAV。
    long_only_returns = calculate_long_only_returns(
        signal_pool_with_returns,
        clean_data=clean_data,
        holding_days=args.holding_days,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    print("5/7 正在计算绩效和摘要...")
    # 第 5 步：基于每日 long-only_return/NAV 计算绩效、回撤、筛选漏斗和步骤日志。
    performance_summary, drawdown_series, yearly_performance = calculate_long_only_performance_attribution(
        long_only_returns,
        args.holding_days,
    )
    portfolio_performance_table, portfolio_performance_detail = build_long_only_performance_tables(performance_summary)
    screen_summary = screening_summary(all_factor)
    screen_filter_summary = screening_filter_step_summary(
        all_factor,
        args.lookback_days,
        args.max_peak_age_days,
        args.drawdown_threshold,
        DEFAULT_ZSCORE_WINDOW,
        DEFAULT_ZSCORE_LIMIT,
    )
    step_log = long_only_factor_step_log(
        raw_data,
        clean_data,
        all_factor,
        raw_factor,
        signal_pool_with_returns,
        long_only_returns,
    )

    print("6/7 正在输出 CSV 文件...")
    # 第 6 步：输出全量筛选、最终股票池、交易收益、可买日期、绩效和清洗诊断文件。
    write_csv(format_factor_dates(all_factor), output_dir / "03_strong_momentum_factor_all_stocks.csv")
    write_csv(format_factor_dates(raw_factor), output_dir / "03_factor1_long_only_screened_stock_pool.csv")
    write_csv(format_factor_dates(signal_pool_with_returns), output_dir / "04_factor1_long_only_signal_pool_with_forward_returns.csv")
    write_csv(
        build_tradable_open_buy_dates(signal_pool_with_returns),
        output_dir / "04_factor1_long_only_tradable_open_buy_dates.csv",
    )
    write_csv(format_factor_dates(long_only_returns), output_dir / "05_factor1_long_only_holding_period_returns.csv")
    write_csv(portfolio_performance_table, output_dir / "06_factor1_long_only_portfolio_performance_table.csv")
    write_csv(portfolio_performance_detail, output_dir / "06_factor1_long_only_portfolio_performance_detail.csv")
    write_csv(format_factor_dates(performance_summary), output_dir / "factor1_long_only_performance_summary.csv")
    write_csv(format_factor_dates(drawdown_series), output_dir / "07_factor1_long_only_drawdown_series.csv")
    write_csv(format_factor_dates(yearly_performance), output_dir / "08_factor1_long_only_yearly_performance.csv")
    write_csv(screen_summary, output_dir / "strong_momentum_screening_summary.csv")
    write_csv(screen_filter_summary, output_dir / "strong_momentum_filter_step_summary.csv")
    write_csv(cleaning_log, output_dir / "cleaning_step_log.csv")
    write_csv(exclusion_summary, output_dir / "cleaning_exclusion_reason_summary.csv")
    write_csv(missing_summary, output_dir / "missing_value_ffill_summary.csv")
    write_csv(step_log, output_dir / "factor1_long_only_step_log.csv")
    # 主策略 NAV 图只在这里生成；diagnostics 不再重复输出 NAV 图。
    nav_curve_path = plot_long_only_nav_curve(long_only_returns, output_dir / "05_factor1_long_only_nav_curve.png")
    print(f"已输出 long-only NAV 曲线：{nav_curve_path}")

    print("7/7 正在输出运行摘要...")
    # 第 7 步：记录本次运行参数、数据来源、交易口径和主要产物路径，便于复现实验。
    run_summary = pd.DataFrame(
        [
            ["factor_name", "factor1_LongOnly"],
            ["factor_version", FACTOR_VERSION],
            ["strategy_type", "long_only_dretwd_real_market_rolling_sleeve_additive_nav"],
            ["start_date", args.start_date],
            ["end_date", args.end_date],
            ["rebalance_frequency", "daily_signal_generation_entry_next_open"],
            ["lookback_days", args.lookback_days],
            ["lookback_unit", "trading_days"],
            ["peak_selection_rule", "first_occurrence_of_highest_close_in_lookback_window"],
            ["max_peak_age_days", args.max_peak_age_days],
            ["zscore_window", DEFAULT_ZSCORE_WINDOW],
            ["zscore_limit", DEFAULT_ZSCORE_LIMIT],
            ["zscore_ddof", DEFAULT_ZSCORE_DDOF],
            ["pre_peak_zscore_rule", f"max_abs_ChangeRatio_time_series_zscore_in_{DEFAULT_ZSCORE_WINDOW}_trading_days_before_peak"],
            ["holding_days", args.holding_days],
            ["holding_unit", "trading_days"],
            ["entry_rule", "next_trade_date_open"],
            ["exit_rule", f"dretwd_compound_through_T+{args.holding_days}"],
            ["return_horizon", f"T+1_Dretwd_to_T+{args.holding_days}_Dretwd"],
            ["return_calculation_source", "Dretwd_return_with_dividend_compounded"],
            ["historical_path_return_rule", "compound_Dretwd_from_pre_peak_low_exclusive_to_peak_and_peak_exclusive_to_signal"],
            ["forward_return_rule", f"compound_Dretwd_from_T+1_to_T+{args.holding_days}"],
            ["daily_nav_return_rule", "daily_active_positions_weighted_Dretwd"],
            ["nav_update_rule", "long_only_nav_t_plus_1=long_only_nav_t+daily_long_only_return_t"],
            ["nav_curve_semantics", "additive_nav_from_daily_long_only_return"],
            ["holding_terminal_offset_days", args.holding_days],
            ["weighting_scheme", f"rolling_{args.holding_days}_equal_capital_sleeves_equal_weight_stocks"],
            ["portfolio_model", "rolling_sleeve_dretwd_additive_nav"],
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
            ["rise_threshold", args.rise_threshold],
            ["drawdown_threshold", args.drawdown_threshold],
            ["min_float_market_value_dsmvosd_unit", args.min_float_market_value],
            ["cleaned_rows", len(clean_data)],
            ["insufficient_history_less_than_lookback_rows", insufficient_history_rows],
            ["all_strong_momentum_rows", len(all_factor)],
            ["long_only_signal_rows_before_trade_filter", len(raw_factor)],
            ["signal_rows_with_forward_returns", len(signal_pool_with_returns)],
            ["tradable_long_only_signal_rows", int(signal_pool_with_returns["is_tradable_next_open"].sum()) if "is_tradable_next_open" in signal_pool_with_returns else 0],
            ["long_only_return_rows", len(long_only_returns)],
            ["cash_period_count", int((~long_only_returns["has_active_position"].fillna(False)).sum()) if "has_active_position" in long_only_returns else 0],
            ["invested_period_count", int(long_only_returns["has_active_position"].fillna(False).sum()) if "has_active_position" in long_only_returns else 0],
            ["trade_detail_date", "" if trade_detail_date is None else trade_detail_date.strftime("%Y-%m-%d")],
            ["trade_detail_image_path", "" if trade_detail_image_path is None else str(trade_detail_image_path)],
            ["nav_curve_path", str(nav_curve_path)],
            ["portfolio_performance_table_rows", len(portfolio_performance_table)],
        ],
        columns=["metric", "value"],
    )
    write_csv(run_summary, output_dir / "run_summary.csv")

    print("factor1_LongOnly 处理完成。")
    print(f"输出目录：{output_dir}")
    print(f"五重筛选信号记录数：{len(raw_factor):,}")
    tradable_rows = int(signal_pool_with_returns["is_tradable_next_open"].sum()) if "is_tradable_next_open" in signal_pool_with_returns else 0
    print(f"次日开盘可交易信号记录数：{tradable_rows:,}")
    print(f"long-only 每日 NAV 记录数：{len(long_only_returns):,}")
    if trade_detail_image_path is not None:
        print(f"交易明细图片：{trade_detail_image_path}")
    print("组合绩效表：")
    print(portfolio_performance_table.to_string(index=False))


if __name__ == "__main__":
    main()
