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

# 当前项目目录。输出文件统一写入本项目 output 子目录。
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "output"
LOCAL_RAW_DATA_CACHE_DIR = DEFAULT_OUTPUT_ROOT / "_local_data_cache"
LOCAL_CLEAN_DATA_CACHE_DIR = DEFAULT_OUTPUT_ROOT / "_local_data_cache"

# MySQL 连接信息。
MYSQL_EXE = os.environ.get("MYSQL_EXE", "mysql")
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "Shx20220717")
MYSQL_DATABASE = "1626_astock"
MARKET_TABLE = "all_market_data"

# 默认研究区间和动量窗口。
DEFAULT_START_DATE = "2015-01-01"
DEFAULT_END_DATE = "2026-01-01"
DEFAULT_LOOKBACK_DAYS = 50
DEFAULT_HOLDING_DAYS = 5
DEFAULT_GROUP_NUM = 10
TRADING_DAYS_PER_YEAR = 252
MONTHS_PER_YEAR = 12
PRICE_COMPARE_TOLERANCE = 1e-6
DEFAULT_MIN_FLOAT_MARKET_VALUE = 5_000_000.0

# 调仓频率：daily=每个交易日生成信号；monthly=每月最后一个交易日生成信号。
# 月频下回看和持有按日历月数计量；日频下按交易日数计量。
DEFAULT_REBALANCE_FREQUENCY = "monthly"
VALID_REBALANCE_FREQUENCIES = {"daily", "monthly"}
DEFAULT_LOOKBACK_MONTHS = 3
DEFAULT_HOLDING_MONTHS = 1
DEFAULT_S = 0
# 月频下回看窗口内要求的最少交易日数，用于剔除数据不完整的窗口。
DEFAULT_MONTHLY_LOOKBACK_MIN_TRADING_DAYS = 21

# 01-10 编号 CSV 输出开关。
# 注意：若后续要运行 momentum_factor_diagnostics.py，至少需要保留 09、10 和主脚本自动输出的 12 号 IC 小表。
SAVE_OUTPUT_01_SQL_RAW_MARKET_DATA = False
SAVE_OUTPUT_02_CLEANED_MARKET_DATA = False
SAVE_OUTPUT_03_ALL_MOMENTUM_FACTOR = False
SAVE_OUTPUT_04_3SIGMA_FACTOR = False
SAVE_OUTPUT_05_ZSCORE_FACTOR = False
SAVE_OUTPUT_06_RANKED_FACTOR = False
SAVE_OUTPUT_07_GROUPED_FACTOR_WITH_FORWARD_RETURN = False
SAVE_OUTPUT_08_LONG_SHORT_STOCK_MEMBERS = False
SAVE_OUTPUT_09_QUANTILE_EQUAL_WEIGHT_RETURNS = True
SAVE_OUTPUT_10_LONG_SHORT_HEDGE_RETURNS = True

# 旧版兼容 CSV 会和 01-10 中的若干文件重复，占用额外磁盘空间，默认关闭。
SAVE_LEGACY_COMPATIBILITY_CSV_OUTPUTS = False

# 主计算脚本只负责当前 DEFAULT_LOOKBACK_DAYS / DEFAULT_HOLDING_DAYS / DEFAULT_GROUP_NUM 的单次实验。
# 图1、图2需要的扫描区间只放在 momentum_factor_diagnostics.py 中使用。

# A 股市场类型：1=上证A股，4=深证A股，16=创业板，32=科创板，64=北证A股。
A_SHARE_MARKET_TYPES = {1, 4, 16, 32, 64}

# Trdsta 中带 ST、*ST、SST、S*ST、GST、G*ST、UST、U*ST、NST、N*ST 以及 PT 的状态。
ST_OR_PT_STATUS_VALUES = {2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16}

# LimitStatus：1=涨停，-1=跌停，0=未涨跌停。
LIMIT_UP_OR_DOWN_VALUES = {-1, 1}

# 从 MySQL 原始表读取的字段。
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

# 将原始字段改成更便于阅读和处理的英文变量名。
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

# 数据库中数值字段存储为 varchar，读取后统一转成数值。
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
DATE_COLUMNS = [
    "trade_date",
    "capital_change_date",
    "month_end_trade_date",
    "next_month_end_trade_date",
    "momentum_start_trade_date",
    "momentum_end_trade_date",
    "momentum_start_date",
    "momentum_end_date",
    "next_trade_date",
    "holding_start_trade_date",
    "holding_end_trade_date",
    "next_trade_date_min",
    "next_trade_date_max",
    "holding_start_trade_date_min",
    "holding_start_trade_date_max",
    "holding_end_trade_date_min",
    "holding_end_trade_date_max",
    "target_end_date",
]
# limit_status 是当日涨跌停状态，不能前值填充，否则会把前一交易日涨跌停错误带入当前交易日。
NO_FORWARD_FILL_COLUMNS = {"limit_status"}
FILL_COLUMNS = [
    col
    for col in COLUMN_RENAME_MAP.values()
    if col not in KEY_COLUMNS and col not in NO_FORWARD_FILL_COLUMNS
]
RETURN_COLUMN_CHOICES = ["return_without_dividend", "return_with_dividend", "change_ratio"]


# ============================================================
# 2. 参数解析和基础工具
# ============================================================


def market_types_to_tag(market_types: set[int]) -> str:
    """把本次保留的 Markettype 集合转换成适合文件夹名称的短标签。"""

    if not market_types:
        return "na"
    return "-".join(str(item) for item in sorted(market_types))


def build_experiment_folder_name(
    lookback_days: int,
    holding_days: int,
    group_num: int,
    market_types: set[int],
    rebalance_frequency: str = DEFAULT_REBALANCE_FREQUENCY,
    lookback_months: int | None = None,
    holding_months: int | None = None,
    s: int | None = None,
) -> str:
    """生成本次实验专属文件夹名，显式区分日频和月频口径。"""

    market_tag = market_types_to_tag(market_types)
    if rebalance_frequency == "monthly":
        lb_months = int(lookback_months) if lookback_months else DEFAULT_LOOKBACK_MONTHS
        hd_months = int(holding_months) if holding_months else DEFAULT_HOLDING_MONTHS
        skip_months = int(s) if s is not None else DEFAULT_S
        return f"monthly_lb{lb_months}m_s{skip_months}_hd{hd_months}m_g{group_num}_mkt{market_tag}"
    return f"daily_lb{lookback_days}_hd{holding_days}_g{group_num}_mkt{market_tag}"


def resolve_experiment_output_dir(base_output_dir: Path, experiment_folder_name: str) -> Path:
    """在输出根目录下创建实验文件夹；若用户已经传入实验文件夹，则直接使用。"""

    if base_output_dir.name == experiment_folder_name:
        return base_output_dir
    return base_output_dir / experiment_folder_name


def parse_args() -> argparse.Namespace:
    """解析运行参数，方便调整日期、动量窗口、持有期、分组数量和输出目录。"""

    parser = argparse.ArgumentParser(
        description=(
            "从 MySQL 读取并清洗 A 股日行情数据，按参数计算动量因子，"
            "完成 3sigma 去极值、Z-score 标准化、标准化因子截面排序、"
            "分组和多空对冲组合收益计算。"
        ),
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="数据起始日期，格式 YYYY-MM-DD。")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="数据结束日期，格式 YYYY-MM-DD。")
    parser.add_argument(
        "--min-float-market-value",
        type=float,
        default=DEFAULT_MIN_FLOAT_MARKET_VALUE,
        help=(
            "流通市值下限，按数据库 Dsmvosd 原字段单位传入。"
            "字段说明中 Dsmvosd 单位为千元，因此 50 亿人民币对应 5,000,000。"
        ),
    )
    parser.add_argument(
        "--rebalance-frequency",
        choices=sorted(VALID_REBALANCE_FREQUENCIES),
        default=DEFAULT_REBALANCE_FREQUENCY,
        help=(
            "调仓频率：daily=每个交易日生成信号（回看/持有按交易日数计量）；"
            f"monthly=每月最后一个交易日生成信号（回看/持有按日历月数计量）。默认 {DEFAULT_REBALANCE_FREQUENCY}。"
        ),
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help=f"日频下动量回看交易日数量。默认 {DEFAULT_LOOKBACK_DAYS}，仅在 --rebalance-frequency=daily 时生效。",
    )
    parser.add_argument(
        "--lookback-months",
        type=int,
        default=DEFAULT_LOOKBACK_MONTHS,
        help=f"月频下动量回看日历月数。默认 {DEFAULT_LOOKBACK_MONTHS}，仅在 --rebalance-frequency=monthly 时生效。",
    )
    parser.add_argument(
        "--s",
        type=int,
        default=DEFAULT_S,
        help=(
            f"月频下动量计算跳过的最近月数。默认 {DEFAULT_S}；"
            "例如 lookback-months=6 且 s=1 时，使用 t-6 月末到 t-1 月末的累计收益。"
            "仅在 --rebalance-frequency=monthly 时生效。"
        ),
    )
    parser.add_argument(
        "--return-column",
        choices=RETURN_COLUMN_CHOICES,
        default="return_without_dividend",
        help="计算未来持有期收益使用的日收益率字段。默认 return_without_dividend。",
    )
    parser.add_argument(
        "--holding-days",
        type=int,
        default=DEFAULT_HOLDING_DAYS,
        help=(
            f"日频下分组后计算未来收益率的持有交易日数。默认 {DEFAULT_HOLDING_DAYS}；"
            "仅在 --rebalance-frequency=daily 时生效。"
        ),
    )
    parser.add_argument(
        "--holding-months",
        type=int,
        default=DEFAULT_HOLDING_MONTHS,
        help=(
            f"月频下持有日历月数。默认 {DEFAULT_HOLDING_MONTHS}；"
            "信号在月末生成，持有收益计算到下一个月末交易日。"
            "仅在 --rebalance-frequency=monthly 时生效。"
        ),
    )
    parser.add_argument(
        "--group-num",
        type=int,
        default=DEFAULT_GROUP_NUM,
        help="按标准化动量因子分成的组数。默认 10，即 10 quantile。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "输出根目录。脚本会在该目录下自动创建 "
            "daily_lb{lookback}_hd{holding}_g{group}_mkt{market_types} 或 "
            "monthly_lb{lookback}m_s{s}_hd{holding}m_g{group}_mkt{market_types} 实验文件夹；默认写入 output。"
        ),
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="保留 MySQL 导出的临时 TSV，便于排查数据问题。",
    )
    parser.add_argument(
        "--refresh-local-data",
        action="store_true",
        help="忽略本地原始数据缓存，重新从 MySQL 导出并覆盖缓存。",
    )
    parser.add_argument(
        "--refresh-clean-data",
        action="store_true",
        help="忽略本地清洗后缓存，重新执行基础清洗和前值填充并覆盖缓存。",
    )
    return parser.parse_args()


def quote_identifier(identifier: str) -> str:
    """给 MySQL 表名和字段名加反引号，避免特殊字段名导致 SQL 语法问题。"""

    return f"`{identifier.replace('`', '``')}`"


def format_date_series(series: pd.Series) -> pd.Series:
    """将日期列格式化为 YYYY-MM-DD，缺失日期保持为空。"""

    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def normalize_date_columns(
    data: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """统一日期列 dtype，避免 merge 时 object 与 datetime64 混用。"""

    output = data.copy()
    for col in columns or DATE_COLUMNS:
        if col in output.columns:
            output[col] = pd.to_datetime(output[col], errors="coerce")
    return output


def write_csv(data: pd.DataFrame, output_path: Path) -> Path:
    """统一使用 utf-8-sig 输出 CSV；若目标文件被占用，则自动写入备用文件。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data.to_csv(output_path, index=False, encoding="utf-8-sig")
        return output_path
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = output_path.with_name(f"{output_path.stem}_{timestamp}{output_path.suffix}")
        print(f"提示：目标文件被占用，已改写入备用文件：{backup_path}")
        data.to_csv(backup_path, index=False, encoding="utf-8-sig")
        return backup_path


def write_optional_csv(
    enabled: bool,
    data: pd.DataFrame,
    output_path: Path,
    skip_reason: str,
) -> Path | None:
    """根据顶部开关决定是否写出 CSV；关闭时只打印提示，不影响后续计算。"""

    if enabled:
        return write_csv(data, output_path)
    print(f"跳过输出 {output_path.name}；{skip_reason}")
    return None


# ============================================================
# 3. MySQL 数据读取
# ============================================================


def build_market_sql(start_date: str, end_date: str) -> str:
    """生成 MySQL 查询语句；这里只做日期筛选，清洗逻辑放在 pandas 中完成。"""

    selected_columns = ",\n    ".join(quote_identifier(col) for col in SOURCE_COLUMNS)
    table_name = quote_identifier(MARKET_TABLE)

    return f"""
SELECT
    {selected_columns}
FROM {table_name}
WHERE {quote_identifier("Trddt")} BETWEEN '{start_date}' AND '{end_date}';
""".strip()


def export_mysql_to_tsv(sql: str, output_path: Path) -> None:
    """调用 mysql.exe 将查询结果导出为 TSV；密码通过 MYSQL_PWD 传入，不出现在命令行参数中。"""

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
            result = subprocess.run(
                command,
                stdout=output_file,
                stderr=subprocess.PIPE,
                env=env,
                check=False,
            )
    except FileNotFoundError as exc:
        raise SystemExit(
            "没有找到 mysql 命令行客户端。\n"
            "请确认 mysql.exe 已加入 PATH，或通过 MYSQL_EXE 环境变量指定完整路径。"
        ) from exc

    if result.returncode != 0:
        error_text = result.stderr.decode("utf-8", errors="ignore")
        raise SystemExit(f"MySQL 查询失败：\n{error_text}")


def read_raw_tsv(tsv_path: Path) -> pd.DataFrame:
    """读取 MySQL 导出的 TSV，并将 NULL、空字符串等识别为缺失值。"""

    return pd.read_csv(
        tsv_path,
        sep="\t",
        dtype="string",
        na_values=["NULL", r"\N", ""],
        keep_default_na=True,
    )


def read_raw_csv(csv_path: Path) -> pd.DataFrame:
    """读取已经缓存到本地的原始行情 CSV，并保持原始字段为字符串类型。"""

    return pd.read_csv(
        csv_path,
        dtype="string",
        na_values=["NULL", r"\N", ""],
        keep_default_na=True,
    )


def build_local_raw_data_cache_path(start_date: str, end_date: str) -> Path:
    """根据研究日期区间生成本地原始数据缓存路径。"""

    start_tag = str(start_date).replace("-", "")
    end_tag = str(end_date).replace("-", "")
    cache_name = f"{MARKET_TABLE}_raw_sql_{start_tag}_{end_tag}.csv"
    return LOCAL_RAW_DATA_CACHE_DIR / cache_name


def load_raw_market_data_with_cache(
    start_date: str,
    end_date: str,
    temp_tsv_path: Path,
    refresh_local_data: bool,
    keep_temp: bool,
) -> tuple[pd.DataFrame, Path, str]:
    """优先复用本地原始数据缓存；缓存不存在或要求刷新时才从 MySQL 重新导出。"""

    cache_path = build_local_raw_data_cache_path(start_date, end_date)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not refresh_local_data:
        print(f"发现本地原始数据缓存，直接读取：{cache_path}")
        return read_raw_csv(cache_path), cache_path, "local_cache"

    print("未使用本地缓存，正在从 MySQL 导出原始行情数据...")
    sql = build_market_sql(start_date, end_date)
    export_mysql_to_tsv(sql, temp_tsv_path)

    print("正在读取 MySQL 导出的临时 TSV...")
    raw_data = read_raw_tsv(temp_tsv_path)
    if not keep_temp:
        temp_tsv_path.unlink(missing_ok=True)

    print(f"正在写入本地原始数据缓存：{cache_path}")
    write_csv(raw_data, cache_path)
    return raw_data, cache_path, "mysql_export"


def build_cache_number_tag(value: float) -> str:
    """把数值参数转换成稳定文件名标签，避免小数点等字符影响路径。"""

    numeric_value = float(value)
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return str(numeric_value).replace(".", "p").replace("-", "minus")


def build_local_clean_data_cache_paths(
    start_date: str,
    end_date: str,
    min_float_market_value: float,
    market_types: set[int],
) -> dict[str, Path]:
    """生成清洗并前值填充后数据及其清洗日志的本地缓存路径。"""

    start_tag = str(start_date).replace("-", "")
    end_tag = str(end_date).replace("-", "")
    market_tag = market_types_to_tag(market_types)
    min_value_tag = build_cache_number_tag(min_float_market_value)
    cache_prefix = (
        f"{MARKET_TABLE}_cleaned_ffill_no_limit_status_{start_tag}_{end_tag}"
        f"_mkt{market_tag}_minmv{min_value_tag}"
    )
    return {
        "clean_data": LOCAL_CLEAN_DATA_CACHE_DIR / f"{cache_prefix}.csv",
        "cleaning_log": LOCAL_CLEAN_DATA_CACHE_DIR / f"{cache_prefix}_cleaning_step_log.csv",
        "missing_summary": LOCAL_CLEAN_DATA_CACHE_DIR / f"{cache_prefix}_missing_value_ffill_summary.csv",
        "exclusion_summary": LOCAL_CLEAN_DATA_CACHE_DIR / f"{cache_prefix}_exclusion_reason_summary.csv",
    }


def clean_cache_is_complete(cache_paths: dict[str, Path]) -> bool:
    """只要清洗后数据存在，就可以直接复用并进入因子计算。"""

    return cache_paths["clean_data"].exists()


def read_clean_market_data_cache(cache_path: Path) -> pd.DataFrame:
    """读取清洗并前值填充后的行情缓存，并恢复日期和数值字段类型。"""

    data = pd.read_csv(
        cache_path,
        dtype={"stock_code": "string"},
        na_values=["NULL", r"\N", ""],
        keep_default_na=True,
    )

    data["stock_code"] = data["stock_code"].astype("string").str.strip().str.zfill(6)
    data = normalize_date_columns(data, ["trade_date", "capital_change_date"])

    for col in NUMERIC_COLUMNS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    for int_col in ["market_type", "trade_status", "limit_status"]:
        if int_col in data.columns:
            data[int_col] = data[int_col].round().astype("Int64")

    return data.sort_values(KEY_COLUMNS).reset_index(drop=True)


def write_clean_market_data_cache(
    clean_data: pd.DataFrame,
    cleaning_log: pd.DataFrame,
    missing_summary: pd.DataFrame,
    exclusion_summary: pd.DataFrame,
    cache_paths: dict[str, Path],
) -> None:
    """写入清洗并前值填充后的本地缓存，后续实验可直接从该步骤开始。"""

    cache_paths["clean_data"].parent.mkdir(parents=True, exist_ok=True)
    write_csv(format_factor_dates(clean_data), cache_paths["clean_data"])
    write_csv(cleaning_log, cache_paths["cleaning_log"])
    write_csv(missing_summary, cache_paths["missing_summary"])
    write_csv(exclusion_summary, cache_paths["exclusion_summary"])


def load_clean_market_data_from_cache(
    cache_paths: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """从本地缓存读取清洗后股票池和清洗阶段日志。"""

    clean_data = read_clean_market_data_cache(cache_paths["clean_data"])
    cleaning_log = read_optional_cache_csv(
        cache_paths["cleaning_log"],
        ["step", "before_rows", "after_rows", "removed_rows", "after_stock_count", "after_trade_date_count"],
    )
    missing_summary = read_optional_cache_csv(
        cache_paths["missing_summary"],
        ["column", "raw_missing_count", "missing_before_ffill_count", "missing_after_ffill_count", "filled_by_ffill_count"],
    )
    exclusion_summary = read_optional_cache_csv(
        cache_paths["exclusion_summary"],
        ["exclude_reason", "record_count", "stock_count", "trade_date_count"],
    )
    return clean_data, cleaning_log, missing_summary, exclusion_summary


def read_optional_cache_csv(cache_path: Path, columns: list[str]) -> pd.DataFrame:
    """读取可选缓存摘要；若不存在则返回带固定列名的空表。"""

    if cache_path.exists():
        return pd.read_csv(cache_path, encoding="utf-8-sig")
    print(f"提示：未找到缓存摘要文件 {cache_path.name}，本次将用空摘要占位。")
    return pd.DataFrame(columns=columns)


# ============================================================
# 4. 清洗逻辑
# ============================================================


def append_step_log(
    log_rows: list[dict],
    step_name: str,
    before_rows: int,
    after_rows: int,
    data_after: pd.DataFrame,
) -> None:
    """记录每个步骤前后的样本数量、股票数量和交易日数量。"""

    log_rows.append(
        {
            "step": step_name,
            "before_rows": before_rows,
            "after_rows": after_rows,
            "removed_rows": before_rows - after_rows,
            "after_stock_count": data_after["stock_code"].nunique() if "stock_code" in data_after else None,
            "after_trade_date_count": data_after["trade_date"].nunique() if "trade_date" in data_after else None,
        }
    )


def standardize_columns(raw_data: pd.DataFrame) -> pd.DataFrame:
    """统一字段命名、股票代码格式、日期格式和数值字段类型。"""

    data = raw_data.rename(columns=COLUMN_RENAME_MAP).copy()

    data["stock_code"] = data["stock_code"].astype("string").str.strip()
    valid_code_mask = data["stock_code"].notna()
    data.loc[valid_code_mask, "stock_code"] = data.loc[valid_code_mask, "stock_code"].str.zfill(6)

    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data["capital_change_date"] = pd.to_datetime(data["capital_change_date"], errors="coerce")

    for col in NUMERIC_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    return data


def build_exclusion_flags(
    data: pd.DataFrame,
    min_float_market_value: float,
) -> pd.DataFrame:
    """基于原始状态字段识别非目标市场、低流通市值、ST/PT、停牌和涨跌停记录。"""

    flags = pd.DataFrame(index=data.index)
    flags["is_not_selected_market_type"] = ~data["market_type"].isin(A_SHARE_MARKET_TYPES)
    # Dsmvosd 字段单位为千元，默认下限 5,000,000 对应 50 亿人民币。
    flags["is_below_min_float_market_value"] = (
        data["float_market_value"].isna()
        | data["float_market_value"].lt(min_float_market_value)
    )
    flags["is_st_or_pt"] = data["trade_status"].isin(ST_OR_PT_STATUS_VALUES)
    flags["is_suspended"] = (
        (data["volume"].notna() & data["volume"].le(0))
        | (data["amount"].notna() & data["amount"].le(0))
    )
    # 涨跌停只做标记，不在基础清洗阶段剔除；动量窗口仅保留诊断，交易过滤放在 T+1 开盘可交易性中执行。
    flags["is_limit_up_or_down"] = data["limit_status"].isin(LIMIT_UP_OR_DOWN_VALUES)
    flags["should_exclude"] = (
        flags["is_not_selected_market_type"]
        | flags["is_below_min_float_market_value"]
        | flags["is_st_or_pt"]
        | flags["is_suspended"]
    )
    return flags


def summarize_missing_values(
    raw_data: pd.DataFrame,
    before_fill: pd.DataFrame,
    after_fill: pd.DataFrame,
) -> pd.DataFrame:
    """汇总清洗后股票池中，前值填充前后的缺失值数量。"""

    raw_missing = raw_data.isna().sum()
    before_missing = before_fill.isna().sum()
    after_missing = after_fill.isna().sum()
    rows: list[dict] = []

    for col in before_fill.columns:
        missing_before = int(before_missing.get(col, 0))
        missing_after = int(after_missing.get(col, 0))
        rows.append(
            {
                "column": col,
                "raw_missing_count": int(raw_missing.get(col, 0)),
                "missing_before_ffill_count": missing_before,
                "missing_after_ffill_count": missing_after,
                "filled_by_ffill_count": max(missing_before - missing_after, 0),
            }
        )

    return pd.DataFrame(rows)


def summarize_exclusion_flags(data: pd.DataFrame, exclusion_flags: pd.DataFrame) -> pd.DataFrame:
    """汇总基础清洗阶段各类剔除原因命中的记录数；不同原因可能重叠。"""

    reason_columns = [
        "is_not_selected_market_type",
        "is_below_min_float_market_value",
        "is_st_or_pt",
        "is_suspended",
        "is_limit_up_or_down",
        "should_exclude",
    ]
    reason_names = {
        "is_not_selected_market_type": "not_selected_market_type",
        "is_below_min_float_market_value": "below_min_float_market_value",
        "is_st_or_pt": "st_or_pt",
        "is_suspended": "suspended",
        "is_limit_up_or_down": "limit_up_or_down_mark_only",
        "should_exclude": "overall_excluded_unique_records",
    }
    rows: list[dict] = []

    for col in reason_columns:
        mask = exclusion_flags[col].fillna(False).astype(bool)
        rows.append(
            {
                "exclude_reason": reason_names[col],
                "record_count": int(mask.sum()),
                "stock_count": data.loc[mask, "stock_code"].nunique(),
                "trade_date_count": data.loc[mask, "trade_date"].nunique(),
            }
        )

    return pd.DataFrame(rows)


def clean_market_data(
    raw_data: pd.DataFrame,
    min_float_market_value: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """执行基础清洗，返回清洗后股票池、清洗日志、缺失值填充日志和剔除原因统计。"""

    clean_log: list[dict] = []
    data = standardize_columns(raw_data)
    append_step_log(clean_log, "read_from_mysql", len(data), len(data), data)

    before_rows = len(data)
    data = data.loc[~data[KEY_COLUMNS].isna().any(axis=1)].copy()
    append_step_log(clean_log, "drop_missing_stock_code_or_trade_date", before_rows, len(data), data)

    before_rows = len(data)
    data = data.sort_values(KEY_COLUMNS)
    data = data.drop_duplicates(subset=KEY_COLUMNS, keep="last")
    append_step_log(clean_log, "drop_duplicate_stock_date_keep_last", before_rows, len(data), data)

    before_rows = len(data)
    exclusion_flags = build_exclusion_flags(
        data=data,
        min_float_market_value=min_float_market_value,
    )
    exclusion_summary = summarize_exclusion_flags(data, exclusion_flags)
    data = data.loc[~exclusion_flags["should_exclude"]].copy()
    append_step_log(
        clean_log,
        "drop_unselected_market_low_float_value_st_suspended",
        before_rows,
        len(data),
        data,
    )

    before_fill_data = data.sort_values(KEY_COLUMNS).reset_index(drop=True)
    data = before_fill_data.copy()
    data[FILL_COLUMNS] = data.groupby("stock_code", group_keys=False)[FILL_COLUMNS].ffill()

    for int_col in ["market_type", "trade_status", "limit_status"]:
        data[int_col] = data[int_col].round().astype("Int64")

    append_step_log(
        clean_log,
        "forward_fill_missing_values_by_stock_except_limit_status",
        len(before_fill_data),
        len(data),
        data,
    )
    missing_summary = summarize_missing_values(raw_data, before_fill_data, data)

    return data, pd.DataFrame(clean_log), missing_summary, exclusion_summary


# ============================================================
# 5. 动量因子、3sigma、Z-score 和标准化因子排序
# ============================================================


def build_month_end_trade_dates(trade_calendar: pd.DataFrame) -> pd.DataFrame:
    """从全市场交易日历中筛选每月最后一个交易日。

    返回含两列的 DataFrame：month_end_trade_date（每月最后一个交易日）和
    next_month_end_trade_date（下一个本月末交易日，用于月频持有期结束日）。
    """

    dates = (
        normalize_date_columns(trade_calendar[["trade_date"]], ["trade_date"])
        .dropna()
        .drop_duplicates()
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    dates = dates.dropna(subset=["trade_date"])

    # 用 (year, month) 分组取每月最大日期，避免月末当天因节假日提前到非自然月末。
    month_end = (
        dates.assign(
            _year=dates["trade_date"].dt.year,
            _month=dates["trade_date"].dt.month,
        )
        .sort_values("trade_date")
        .groupby(["_year", "_month"], as_index=False)["trade_date"]
        .max()
        .sort_values("trade_date")
        .reset_index(drop=True)
        .rename(columns={"trade_date": "month_end_trade_date"})
    )
    month_end["next_month_end_trade_date"] = month_end["month_end_trade_date"].shift(-1)
    month_end = month_end.drop(columns=[col for col in month_end.columns if col.startswith("_")], errors="ignore")
    return month_end


def build_monthly_signal_calendar(
    trade_calendar: pd.DataFrame,
    lookback_months: int,
    holding_months: int,
    s: int = DEFAULT_S,
) -> pd.DataFrame:
    """构造月频信号日历：每个本月末交易日对应回看锚点、次交易日（开仓日）和持有期末交易日。

    - momentum_start_trade_date：lookback_months 个月前的本月末交易日（回看窗口起点锚点）
    - momentum_end_trade_date：跳过最近 s 个月后的本月末交易日（回看窗口终点锚点）
    - holding_start_trade_date：信号日次交易日（实际开仓日，与日频 T+1 开盘口径一致）
    - holding_end_trade_date：信号日之后 holding_months 个月对应的本月末交易日（持有期结束日）
    """

    trade_calendar = normalize_date_columns(trade_calendar, ["trade_date"])
    month_end = build_month_end_trade_dates(trade_calendar).reset_index(drop=True)

    # 回看锚点：本月末序列向前 shift lookback_months 行。
    month_end["momentum_start_trade_date"] = month_end["month_end_trade_date"].shift(lookback_months)
    # 跳过最近 s 个月：s=1 时回看终点为上个月末；s=0 时回看终点为本月末。
    month_end["momentum_end_trade_date"] = month_end["month_end_trade_date"].shift(s)
    # 持有期结束日：本月末序列向后 shift holding_months 行。
    month_end["holding_end_trade_date"] = month_end["month_end_trade_date"].shift(-holding_months)

    # 开仓日 = 信号日次交易日（全市场口径）。
    all_dates = (
        normalize_date_columns(trade_calendar[["trade_date"]], ["trade_date"])
        .dropna()
        .drop_duplicates()
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    next_date_map = dict(zip(all_dates["trade_date"], all_dates["trade_date"].shift(-1)))
    month_end["holding_start_trade_date"] = month_end["month_end_trade_date"].map(next_date_map)

    return month_end


def resolve_annualization_periods_per_year(
    rebalance_frequency: str,
    holding_days: int | None,
    holding_months: int | None,
) -> float:
    """按调仓频率返回年化系数（每年调仓次数）。

    日频：TRADING_DAYS_PER_YEAR / holding_days（每个持有期的交易日数）。
    月频：MONTHS_PER_YEAR / holding_months。
    """

    if rebalance_frequency == "monthly":
        months = int(holding_months) if holding_months else DEFAULT_HOLDING_MONTHS
        return float(MONTHS_PER_YEAR) / float(max(months, 1))
    days = int(holding_days) if holding_days else DEFAULT_HOLDING_DAYS
    return float(TRADING_DAYS_PER_YEAR) / float(max(days, 1))


def calculate_raw_momentum(
    clean_data: pd.DataFrame,
    return_column: str,
    lookback_days: int,
    rebalance_frequency: str = DEFAULT_REBALANCE_FREQUENCY,
    lookback_months: int | None = None,
    s: int = DEFAULT_S,
    monthly_lookback_min_trading_days: int = DEFAULT_MONTHLY_LOOKBACK_MIN_TRADING_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算横截面动量因子。

    日频：第 T 日收盘后使用同一股票 T-lookback_days 至 T-1 首尾收盘价收益率。
    月频：仅在本月末交易日生成信号，使用 lookback_months 个月前月末至 T-s 月末之间的日收益率复利。
    """

    data = normalize_date_columns(clean_data)
    data = data.sort_values(KEY_COLUMNS).reset_index(drop=True).copy()
    data[return_column] = pd.to_numeric(data[return_column], errors="coerce").astype("float64")
    data["close_price"] = pd.to_numeric(data["close_price"], errors="coerce").astype("float64")
    data["limit_status"] = pd.to_numeric(data["limit_status"], errors="coerce")

    if rebalance_frequency == "monthly":
        return _calculate_raw_momentum_monthly(
            data=data,
            return_column=return_column,
            lookback_months=int(lookback_months) if lookback_months else DEFAULT_LOOKBACK_MONTHS,
            s=s,
            monthly_lookback_min_trading_days=monthly_lookback_min_trading_days,
        )

    return _calculate_raw_momentum_daily(
        data=data,
        return_column=return_column,
        lookback_days=lookback_days,
    )


def _calculate_raw_momentum_daily(
    data: pd.DataFrame,
    return_column: str,
    lookback_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """日频动量：每个交易日生成信号，回看窗口为过去 lookback_days 个交易日。"""

    grouped = data.groupby("stock_code", group_keys=False)
    window_start_offset = lookback_days
    window_end_offset = 1

    # 记录第 T 日收盘后实际使用的窗口日期；5 日窗口为 T-5 至 T-1。
    data["momentum_start_date"] = grouped["trade_date"].shift(window_start_offset)
    data["momentum_end_date"] = grouped["trade_date"].shift(window_end_offset)

    # 第 T 日信号只使用 T-1 及以前的已发生行情，涨跌停交易过滤延后到 T+1 开盘前执行。
    data["momentum_start_close_price"] = grouped["close_price"].shift(window_start_offset)
    data["momentum_end_close_price"] = grouped["close_price"].shift(window_end_offset)
    lookback_close = grouped["close_price"].shift(window_end_offset)
    lookback_limit_flag = grouped["limit_status"].shift(window_end_offset).isin(LIMIT_UP_OR_DOWN_VALUES).astype("int64")

    # 每只股票单独检查 T-5 至 T-1 的 lookback_days 个交易记录，必须凑满窗口。
    rolling_close = lookback_close.groupby(data["stock_code"]).rolling(
        window=lookback_days,
        min_periods=lookback_days,
    )
    data["lookback_valid_days"] = (
        rolling_close.count().reset_index(level=0, drop=True).astype("Int64")
    )
    data["lookback_limit_days_count"] = (
        lookback_limit_flag.groupby(data["stock_code"])
        .rolling(window=lookback_days, min_periods=lookback_days)
        .sum()
        .reset_index(level=0, drop=True)
        .astype("Int64")
    )
    data["lookback_has_limit_up_or_down"] = data["lookback_limit_days_count"].fillna(0).gt(0)
    valid_price_mask = (
        data["momentum_start_close_price"].notna()
        & data["momentum_end_close_price"].notna()
        & data["momentum_start_close_price"].gt(0)
    )
    data["momentum_raw"] = np.nan
    data.loc[valid_price_mask, "momentum_raw"] = (
        data.loc[valid_price_mask, "momentum_end_close_price"]
        / data.loc[valid_price_mask, "momentum_start_close_price"]
        - 1.0
    )
    invalid_window_mask = (
        data["lookback_valid_days"].fillna(0).lt(lookback_days)
        | ~valid_price_mask
    )
    data.loc[invalid_window_mask, "momentum_raw"] = np.nan

    return _finalize_raw_momentum(
        data=data,
        return_column=return_column,
        lookback_days=lookback_days,
    )


def _calculate_raw_momentum_monthly(
    data: pd.DataFrame,
    return_column: str,
    lookback_months: int,
    s: int,
    monthly_lookback_min_trading_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """月频动量：仅在本月末交易日生成信号。

    动量窗口：lookback_months 个月前月末交易日到 T-s 月末交易日之间的每日收益率复利。
    信号在 T 日收盘后生成，并用于 T+1 开盘，不引入未来数据。
    """

    data = data.sort_values(KEY_COLUMNS).reset_index(drop=True).copy()

    # 1) 在全市场口径下识别本月末交易日与回看锚点。
    signal_calendar = build_monthly_signal_calendar(
        data[["trade_date"]],
        lookback_months=lookback_months,
        holding_months=DEFAULT_HOLDING_MONTHS,
        s=s,
    )
    signal_calendar = signal_calendar[
        [
            "month_end_trade_date",
            "momentum_start_trade_date",
            "momentum_end_trade_date",
        ]
    ].copy()
    month_end_dates = set(signal_calendar["month_end_trade_date"].dropna().tolist())

    # 2) 标记本月末交易日行；非本月末行不参与月频信号。
    data["is_month_end_signal_date"] = data["trade_date"].isin(month_end_dates)
    signal_rows = data.loc[data["is_month_end_signal_date"]].copy()

    if signal_rows.empty:
        empty = data.iloc[0:0].copy()
        empty["momentum_raw"] = pd.Series(dtype="float64")
        return empty, empty

    # 端点日期 -> 端点收盘价查找表（每只股票单独查）。
    price_lookup = (
        data[["stock_code", "trade_date", "close_price"]]
        .rename(columns={"trade_date": "price_date", "close_price": "lookup_close_price"})
    )

    signal_rows = signal_rows.merge(
        signal_calendar,
        left_on="trade_date",
        right_on="month_end_trade_date",
        how="left",
    ).rename(
        columns={
            "momentum_start_trade_date": "momentum_start_date",
            "momentum_end_trade_date": "momentum_end_date",
        }
    )

    # 起点收盘价：通过 (stock_code, momentum_start_date) 查找。
    start_prices = price_lookup.rename(
        columns={
            "price_date": "momentum_start_date",
            "lookup_close_price": "momentum_start_close_price",
        }
    )
    signal_rows = signal_rows.merge(
        start_prices[["stock_code", "momentum_start_date", "momentum_start_close_price"]],
        on=["stock_code", "momentum_start_date"],
        how="left",
    )

    # 终点收盘价：通过 (stock_code, momentum_end_date) 查找；s=1 时为上个月末收盘价。
    end_prices = price_lookup.rename(
        columns={
            "price_date": "momentum_end_date",
            "lookup_close_price": "momentum_end_close_price",
        }
    )
    signal_rows = signal_rows.merge(
        end_prices[["stock_code", "momentum_end_date", "momentum_end_close_price"]],
        on=["stock_code", "momentum_end_date"],
        how="left",
    )

    # 回看窗口内收益、交易日数量与涨跌停天数：基于每只股票完整日行情序列统计 (start, end]。
    signal_rows = signal_rows.reset_index(drop=True)
    signal_rows["_signal_row_id"] = np.arange(len(signal_rows), dtype="int64")
    global_trade_dates = pd.to_datetime(
        data["trade_date"].dropna().drop_duplicates().sort_values()
    ).to_numpy(dtype="datetime64[ns]")
    lookback_expected_counts = np.zeros(len(signal_rows), dtype="int64")
    lookback_valid_counts = np.zeros(len(signal_rows), dtype="int64")
    lookback_limit_counts = np.zeros(len(signal_rows), dtype="int64")
    lookback_complete = np.zeros(len(signal_rows), dtype=bool)
    lookback_compound_returns = np.full(len(signal_rows), np.nan)
    signal_specs_by_stock = {
        stock_code: stock_signals[["_signal_row_id", "momentum_start_date", "momentum_end_date"]].copy()
        for stock_code, stock_signals in signal_rows.groupby("stock_code", sort=False)
    }

    for stock_code, stock_history in data.groupby("stock_code", sort=False):
        stock_signals = signal_specs_by_stock.get(stock_code)
        if stock_signals is None or stock_signals.empty:
            continue

        trade_dates = pd.to_datetime(stock_history["trade_date"]).to_numpy(dtype="datetime64[ns]")
        returns = stock_history[return_column].to_numpy(dtype="float64")
        limit_flags = (
            stock_history["limit_status"]
            .isin(LIMIT_UP_OR_DOWN_VALUES)
            .fillna(False)
            .to_numpy(dtype="int64")
        )
        limit_prefix_sum = np.concatenate([[0], np.cumsum(limit_flags)])

        start_dates = pd.to_datetime(stock_signals["momentum_start_date"]).to_numpy(dtype="datetime64[ns]")
        end_dates = pd.to_datetime(stock_signals["momentum_end_date"]).to_numpy(dtype="datetime64[ns]")
        row_ids = stock_signals["_signal_row_id"].to_numpy(dtype="int64")

        for row_id, start_dt, end_dt in zip(row_ids, start_dates, end_dates):
            if np.isnat(start_dt) or np.isnat(end_dt):
                continue
            expected_left = int(np.searchsorted(global_trade_dates, start_dt, side="right"))
            expected_right = int(np.searchsorted(global_trade_dates, end_dt, side="right"))
            expected_days = expected_right - expected_left
            if expected_days <= 0:
                continue
            lookback_expected_counts[row_id] = expected_days

            left_pos = int(np.searchsorted(trade_dates, start_dt, side="right"))
            right_pos = int(np.searchsorted(trade_dates, end_dt, side="right"))
            if right_pos <= left_pos:
                continue
            window_dates = trade_dates[left_pos:right_pos]
            window_returns = returns[left_pos:right_pos]
            valid_days = int((~np.isnan(window_returns)).sum())
            lookback_valid_counts[row_id] = valid_days
            lookback_limit_counts[row_id] = int(limit_prefix_sum[right_pos] - limit_prefix_sum[left_pos])
            has_exact_end_record = len(window_dates) > 0 and window_dates[-1] == end_dt
            if has_exact_end_record and valid_days == expected_days and len(window_returns) == expected_days:
                lookback_compound_returns[row_id] = float(np.prod(1.0 + window_returns) - 1.0)
                lookback_complete[row_id] = True

    signal_rows["lookback_expected_days"] = lookback_expected_counts
    signal_rows["lookback_valid_days"] = lookback_valid_counts
    signal_rows["lookback_limit_days_count"] = lookback_limit_counts
    signal_rows["has_complete_lookback_return"] = lookback_complete
    signal_rows["lookback_expected_days"] = signal_rows["lookback_expected_days"].astype("Int64")
    signal_rows["lookback_valid_days"] = signal_rows["lookback_valid_days"].astype("Int64")
    signal_rows["lookback_limit_days_count"] = signal_rows["lookback_limit_days_count"].astype("Int64")
    signal_rows["lookback_has_limit_up_or_down"] = signal_rows["lookback_limit_days_count"].fillna(0).gt(0)

    signal_rows["momentum_raw"] = lookback_compound_returns
    invalid_window_mask = (
        signal_rows["lookback_valid_days"].fillna(0).lt(monthly_lookback_min_trading_days)
        | ~signal_rows["has_complete_lookback_return"].fillna(False).astype(bool)
    )
    signal_rows.loc[invalid_window_mask, "momentum_raw"] = np.nan
    signal_rows = signal_rows.drop(columns=["_signal_row_id"], errors="ignore")

    # 把信号行回填到 data，非信号行 momentum_raw 留空，便于统一输出。
    data = data.drop(columns=[col for col in ["momentum_raw"] if col in data.columns], errors="ignore")
    signal_keep_cols = [
        "stock_code",
        "trade_date",
        "momentum_start_date",
        "momentum_end_date",
        "momentum_start_close_price",
        "momentum_end_close_price",
        "lookback_expected_days",
        "lookback_valid_days",
        "lookback_limit_days_count",
        "has_complete_lookback_return",
        "lookback_has_limit_up_or_down",
        "momentum_raw",
    ]
    data = data.merge(
        signal_rows[signal_keep_cols],
        on=["stock_code", "trade_date"],
        how="left",
    )

    return _finalize_raw_momentum(
        data=data,
        return_column=return_column,
        lookback_days=lookback_months,
        lookback_unit="months",
        monthly_skip_months=s,
    )


def _finalize_raw_momentum(
    data: pd.DataFrame,
    return_column: str,
    lookback_days: int,
    lookback_unit: str = "days",
    monthly_skip_months: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """整理原始动量因子输出列，日频/月频共用。"""

    base_columns = [
        "stock_code",
        "trade_date",
        "close_price",
        return_column,
        "momentum_start_date",
        "momentum_end_date",
        "momentum_start_close_price",
        "momentum_end_close_price",
        "lookback_valid_days",
        "lookback_limit_days_count",
        "lookback_has_limit_up_or_down",
        "momentum_raw",
    ]
    for optional_col in ["lookback_expected_days", "has_complete_lookback_return"]:
        if optional_col in data.columns and optional_col not in base_columns:
            insert_pos = base_columns.index("lookback_valid_days")
            base_columns.insert(insert_pos, optional_col)
    raw_factor = data[base_columns].copy()
    raw_factor = raw_factor.rename(columns={"close_price": "signal_close_price"})
    raw_factor["lookback_days"] = lookback_days
    raw_factor["lookback_unit"] = lookback_unit
    raw_factor["s"] = monthly_skip_months if monthly_skip_months is not None else pd.NA
    raw_factor["monthly_skip_months"] = monthly_skip_months if monthly_skip_months is not None else pd.NA
    raw_factor["return_column_used"] = return_column

    # 缺少完整回看窗口的记录单独输出，便于核对为什么某些股票某些日期没有因子值。
    missing_factor = raw_factor.loc[raw_factor["momentum_raw"].isna()].copy()
    valid_factor = raw_factor.loc[raw_factor["momentum_raw"].notna()].copy()

    return valid_factor, missing_factor


def apply_3sigma_winsorization(raw_factor: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按每日截面计算均值和标准差，并把动量因子控制在均值正负 3 个标准差以内。"""

    cross_section_stats = (
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
    cross_section_stats["sigma_lower_bound"] = (
        cross_section_stats["cross_section_mean"] - 3.0 * cross_section_stats["cross_section_std"]
    )
    cross_section_stats["sigma_upper_bound"] = (
        cross_section_stats["cross_section_mean"] + 3.0 * cross_section_stats["cross_section_std"]
    )

    factor = raw_factor.merge(cross_section_stats, on="trade_date", how="left")
    valid_sigma_mask = (
        factor["cross_section_std"].notna()
        & factor["cross_section_std"].ne(0)
        & factor["sigma_lower_bound"].notna()
        & factor["sigma_upper_bound"].notna()
    )

    # 这里的“去除极端值”采用金融因子处理中常用的缩尾法：不删记录，只把极端值压到边界。
    lower_extreme_mask = valid_sigma_mask & factor["momentum_raw"].lt(factor["sigma_lower_bound"])
    upper_extreme_mask = valid_sigma_mask & factor["momentum_raw"].gt(factor["sigma_upper_bound"])
    factor["is_3sigma_extreme"] = lower_extreme_mask | upper_extreme_mask
    factor["extreme_direction"] = np.select(
        [lower_extreme_mask, upper_extreme_mask],
        ["lower_than_mean_minus_3std", "higher_than_mean_plus_3std"],
        default="within_3sigma",
    )
    factor["momentum_3sigma"] = factor["momentum_raw"]
    factor.loc[lower_extreme_mask, "momentum_3sigma"] = factor.loc[
        lower_extreme_mask, "sigma_lower_bound"
    ]
    factor.loc[upper_extreme_mask, "momentum_3sigma"] = factor.loc[
        upper_extreme_mask, "sigma_upper_bound"
    ]

    extreme_records = factor.loc[factor["is_3sigma_extreme"]].copy()
    extreme_count = extreme_records.groupby("trade_date").size().rename("extreme_count").reset_index()
    date_summary = cross_section_stats.merge(extreme_count, on="trade_date", how="left")
    date_summary["extreme_count"] = date_summary["extreme_count"].fillna(0).astype("Int64")

    return factor, extreme_records, date_summary


def apply_zscore_standardization(factor_3sigma: pd.DataFrame) -> pd.DataFrame:
    """按每日截面对 3sigma 处理后的动量因子做 Z-score 标准化。"""

    zscore_stats = (
        factor_3sigma.groupby("trade_date")["momentum_3sigma"]
        .agg(
            zscore_mean="mean",
            zscore_std=lambda series: series.std(ddof=0),
            zscore_stock_count="count",
        )
        .reset_index()
    )
    factor = factor_3sigma.merge(zscore_stats, on="trade_date", how="left")

    # 当某日截面标准差为 0 时无法标准化，该日 zscore 留空并在后续排序中自动剔除。
    valid_std_mask = factor["zscore_std"].notna() & factor["zscore_std"].ne(0)
    factor["momentum_zscore"] = np.nan
    factor.loc[valid_std_mask, "momentum_zscore"] = (
        factor.loc[valid_std_mask, "momentum_3sigma"] - factor.loc[valid_std_mask, "zscore_mean"]
    ) / factor.loc[valid_std_mask, "zscore_std"]

    return factor


def rank_standardized_momentum_cross_section(factor_zscore: pd.DataFrame) -> pd.DataFrame:
    """按交易日截面将全部股票按标准化动量因子从大到小排序。"""

    # 排序必须建立在 Z-score 标准化后的因子上；股票代码只用于同分时稳定排序。
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


def calculate_forward_holding_returns(
    clean_data: pd.DataFrame,
    return_column: str,
    holding_days: int,
    rebalance_frequency: str = DEFAULT_REBALANCE_FREQUENCY,
    holding_months: int | None = None,
) -> pd.DataFrame:
    """计算第 t 日信号对应的未来持有期收益。

    日频：未来 holding_days 个交易日复利累计收益（T+1 至 T+holding_days）。
    月频：未来 holding_months 个日历月——信号日到 holding_months 个月后的本月末交易日之间的累计收益。
    """

    if rebalance_frequency == "monthly":
        return _calculate_forward_holding_returns_monthly(
            clean_data=clean_data,
            return_column=return_column,
            holding_months=int(holding_months) if holding_months else DEFAULT_HOLDING_MONTHS,
        )

    return _calculate_forward_holding_returns_daily(
        clean_data=clean_data,
        return_column=return_column,
        holding_days=holding_days,
    )


def _calculate_forward_holding_returns_daily(
    clean_data: pd.DataFrame,
    return_column: str,
    holding_days: int,
) -> pd.DataFrame:
    """日频持有收益：未来 holding_days 个交易日复利累计收益。"""

    clean_data = normalize_date_columns(clean_data)
    forward_data = clean_data.sort_values(KEY_COLUMNS)[["stock_code", "trade_date", return_column]].copy()
    forward_data[return_column] = pd.to_numeric(forward_data[return_column], errors="coerce").astype("float64")
    grouped_return = forward_data.groupby("stock_code")[return_column]

    # 第 t 日信号的未来收益从 T+1 开始，到 T+holding_days 结束。
    complete_holding_mask = pd.Series(True, index=forward_data.index)
    compound_multiplier = pd.Series(1.0, index=forward_data.index, dtype="float64")
    forward_data["future_return_valid_days"] = 0

    for day_offset in range(1, holding_days + 1):
        shifted_return = grouped_return.shift(-day_offset)
        shifted_return_valid = shifted_return.notna()
        complete_holding_mask &= shifted_return_valid
        forward_data["future_return_valid_days"] += shifted_return_valid.astype("int64")
        compound_multiplier *= 1.0 + shifted_return.fillna(0.0)

    forward_data["next_period_return_before_trade_filter"] = compound_multiplier - 1.0
    forward_data.loc[
        ~complete_holding_mask,
        "next_period_return_before_trade_filter",
    ] = np.nan
    forward_data["has_complete_holding_return"] = complete_holding_mask

    return forward_data[
        [
            "stock_code",
            "trade_date",
            "next_period_return_before_trade_filter",
            "future_return_valid_days",
            "has_complete_holding_return",
        ]
    ]


def _calculate_forward_holding_returns_monthly(
    clean_data: pd.DataFrame,
    return_column: str,
    holding_months: int,
) -> pd.DataFrame:
    """月频持有收益：信号日 T（本月末）到 holding_months 个月后本月末交易日之间的累计收益。

    收益口径与日频保持一致：复利 T+1 到持有期末交易日之间的每个交易日收益率。
    期间任一交易日收益缺失（停牌等），则该信号视为持有期不完整，next_period_return 留空。
    """

    clean_data = normalize_date_columns(clean_data)
    forward_data = clean_data.sort_values(KEY_COLUMNS)[["stock_code", "trade_date", return_column]].copy()
    forward_data[return_column] = pd.to_numeric(forward_data[return_column], errors="coerce").astype("float64")
    forward_data["future_return_valid_days"] = 0
    forward_data["next_period_return_before_trade_filter"] = np.nan
    forward_data["has_complete_holding_return"] = False
    global_trade_dates = pd.to_datetime(
        forward_data["trade_date"].dropna().drop_duplicates().sort_values()
    ).to_numpy(dtype="datetime64[ns]")

    # 信号日 = 本月末交易日；持有期末交易日 = 信号日之后 holding_months 个月的本月末交易日。
    signal_calendar = build_monthly_signal_calendar(
        forward_data[["trade_date"]],
        lookback_months=DEFAULT_LOOKBACK_MONTHS,
        holding_months=holding_months,
    )
    signal_to_end = signal_calendar.dropna(subset=["month_end_trade_date", "holding_end_trade_date"]).copy()
    signal_to_end = signal_to_end.rename(columns={"holding_end_trade_date": "target_end_date"})[
        ["month_end_trade_date", "target_end_date"]
    ]

    # 仅信号日行参与持有期计算。
    forward_data = forward_data.merge(
        signal_to_end.rename(columns={"month_end_trade_date": "trade_date"}),
        on="trade_date",
        how="left",
    )
    signal_mask = forward_data["target_end_date"].notna()
    if not signal_mask.any():
        forward_data = forward_data.drop(columns=["target_end_date"])
        return forward_data[
            [
                "stock_code",
                "trade_date",
                "next_period_return_before_trade_filter",
                "future_return_valid_days",
                "has_complete_holding_return",
        ]
    ]

    signal_rows = forward_data.loc[signal_mask, ["stock_code", "trade_date", "target_end_date"]].copy()
    signal_rows["_forward_index"] = signal_rows.index.to_numpy(dtype="int64")
    signal_rows["_signal_row_id"] = np.arange(len(signal_rows), dtype="int64")
    out_returns = np.full(len(signal_rows), np.nan)
    out_valid_days = np.zeros(len(signal_rows), dtype="int64")
    out_complete = np.zeros(len(signal_rows), dtype=bool)
    signal_specs_by_stock = {
        stock_code: stock_signals[["_signal_row_id", "trade_date", "target_end_date"]].copy()
        for stock_code, stock_signals in signal_rows.groupby("stock_code", sort=False)
    }

    for stock_code, stock_history in forward_data.groupby("stock_code", sort=False):
        stock_signals = signal_specs_by_stock.get(stock_code)
        if stock_signals is None or stock_signals.empty:
            continue

        trade_dates = pd.to_datetime(stock_history["trade_date"]).to_numpy(dtype="datetime64[ns]")
        returns = stock_history[return_column].to_numpy(dtype="float64")

        signal_dates = pd.to_datetime(stock_signals["trade_date"]).to_numpy(dtype="datetime64[ns]")
        end_dates = pd.to_datetime(stock_signals["target_end_date"]).to_numpy(dtype="datetime64[ns]")
        row_ids = stock_signals["_signal_row_id"].to_numpy(dtype="int64")

        for row_id, signal_dt, end_dt in zip(row_ids, signal_dates, end_dates):
            if np.isnat(signal_dt) or np.isnat(end_dt):
                continue

            expected_left = int(np.searchsorted(global_trade_dates, signal_dt, side="right"))
            expected_right = int(np.searchsorted(global_trade_dates, end_dt, side="right"))
            expected_days = expected_right - expected_left
            if expected_days <= 0:
                continue

            left_pos = int(np.searchsorted(trade_dates, signal_dt, side="right"))
            right_pos = int(np.searchsorted(trade_dates, end_dt, side="right"))
            if right_pos <= left_pos:
                continue

            window_dates = trade_dates[left_pos:right_pos]
            window_returns = returns[left_pos:right_pos]
            valid_mask = ~np.isnan(window_returns)
            valid_days = int(valid_mask.sum())
            out_valid_days[row_id] = valid_days

            has_exact_end_record = len(window_dates) > 0 and window_dates[-1] == end_dt
            if has_exact_end_record and valid_days == expected_days and len(window_returns) == expected_days:
                out_returns[row_id] = float(np.prod(1.0 + window_returns) - 1.0)
                out_complete[row_id] = True

    signal_rows["next_period_return_before_trade_filter"] = out_returns
    signal_rows["future_return_valid_days"] = out_valid_days
    signal_rows["has_complete_holding_return"] = out_complete

    forward_indices = signal_rows["_forward_index"].to_numpy(dtype="int64")
    forward_data.loc[forward_indices, "next_period_return_before_trade_filter"] = signal_rows[
        "next_period_return_before_trade_filter"
    ].to_numpy()
    forward_data.loc[forward_indices, "future_return_valid_days"] = signal_rows[
        "future_return_valid_days"
    ].to_numpy()
    forward_data.loc[forward_indices, "has_complete_holding_return"] = signal_rows[
        "has_complete_holding_return"
    ].to_numpy()
    forward_data["future_return_valid_days"] = forward_data["future_return_valid_days"].astype("Int64")
    forward_data = forward_data.drop(columns=["target_end_date"])

    return forward_data[
        [
            "stock_code",
            "trade_date",
            "next_period_return_before_trade_filter",
            "future_return_valid_days",
            "has_complete_holding_return",
        ]
    ]


def assign_quantile_groups(
    ranked_factor: pd.DataFrame,
    clean_data: pd.DataFrame,
    return_column: str,
    group_num: int,
    holding_days: int,
    rebalance_frequency: str = DEFAULT_REBALANCE_FREQUENCY,
    holding_months: int | None = None,
) -> pd.DataFrame:
    """按标准化动量排名分组，并合并未来持有期收益率。

    日频：未来 holding_days 个交易日收益。
    月频：未来 holding_months 个日历月收益（信号日到下个月末交易日之间）。
    """

    grouped = normalize_date_columns(ranked_factor)
    clean_data = normalize_date_columns(clean_data)

    # 第 1 组是标准化动量最高的股票，第 group_num 组是标准化动量最低的股票。
    grouped["quantile_group"] = (
        ((grouped["standardized_momentum_rank_desc"] - 1) * group_num)
        // grouped["cross_section_stock_count_after_zscore"]
        + 1
    ).clip(lower=1, upper=group_num)
    grouped["quantile_group"] = grouped["quantile_group"].astype("Int64")

    # 第 t 日的因子和分组用于未来持有期；这里用全市场交易日历严格定义开仓日和持有期结束日。
    trade_calendar = (
        clean_data[["trade_date"]]
        .drop_duplicates()
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    trade_calendar = normalize_date_columns(trade_calendar, ["trade_date"])
    trade_calendar["next_trade_date"] = trade_calendar["trade_date"].shift(-1)
    if rebalance_frequency == "monthly":
        # 月频：开仓日为信号日次交易日（与日频 T+1 口径一致），持有期结束日为下一个月末交易日。
        months = int(holding_months) if holding_months else DEFAULT_HOLDING_MONTHS
        month_end_calendar = build_monthly_signal_calendar(
            trade_calendar,
            lookback_months=DEFAULT_LOOKBACK_MONTHS,
            holding_months=months,
        )
        month_end_calendar = normalize_date_columns(month_end_calendar)
        trade_calendar = trade_calendar.merge(
            month_end_calendar[
                [
                    "month_end_trade_date",
                    "holding_start_trade_date",
                    "holding_end_trade_date",
                ]
            ].rename(columns={"month_end_trade_date": "trade_date"}),
            on="trade_date",
            how="left",
        )
        # 仅信号日行有 holding_start/holding_end；非信号日沿用 next_trade_date 作为 holding_start 占位。
        trade_calendar["holding_start_trade_date"] = trade_calendar["holding_start_trade_date"].fillna(
            trade_calendar["next_trade_date"]
        )
        trade_calendar["holding_end_trade_date"] = trade_calendar["holding_end_trade_date"]
    else:
        trade_calendar["holding_start_trade_date"] = trade_calendar["next_trade_date"]
        trade_calendar["holding_end_trade_date"] = trade_calendar["trade_date"].shift(-holding_days)
    trade_calendar = normalize_date_columns(trade_calendar)
    grouped = grouped.merge(trade_calendar, on="trade_date", how="left")

    # 先计算每个 T 日信号对应的未来 N 期收益，再在 T+1 开盘前做可交易性过滤。
    forward_returns = calculate_forward_holding_returns(
        clean_data=clean_data,
        return_column=return_column,
        holding_days=holding_days,
        rebalance_frequency=rebalance_frequency,
        holding_months=holding_months,
    )
    forward_returns = normalize_date_columns(forward_returns)
    grouped = grouped.merge(
        forward_returns,
        on=["stock_code", "trade_date"],
        how="left",
    )

    # 只匹配 T+1 当天仍在清洗后股票池中的记录，并保留开盘价/涨跌停价用于可交易性校验。
    entry_day_columns = [
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
    entry_data = clean_data.sort_values(KEY_COLUMNS)[entry_day_columns].copy()
    entry_data = normalize_date_columns(entry_data, ["trade_date"])
    for numeric_col in [
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "limit_up_price",
        "limit_down_price",
        "limit_status",
    ]:
        entry_data[numeric_col] = pd.to_numeric(entry_data[numeric_col], errors="coerce")
    entry_data = entry_data.rename(
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
    grouped = grouped.merge(
        entry_data[
            [
                "stock_code",
                "next_trade_date",
                "next_open_price",
                "next_high_price",
                "next_low_price",
                "next_close_price",
                "next_limit_up_price",
                "next_limit_down_price",
                "next_limit_status",
            ]
        ],
        on=["stock_code", "next_trade_date"],
        how="left",
    )

    grouped["next_open_to_signal_close_return"] = (
        grouped["next_open_price"] / grouped["signal_close_price"] - 1.0
    )

    # T+1 开盘涨跌停过滤：只使用开盘前/开盘时可知的价格，不使用 T+1 日内最高价/最低价。
    open_equals_limit_up = (
        grouped["next_open_price"] - grouped["next_limit_up_price"]
    ).abs().le(PRICE_COMPARE_TOLERANCE)
    open_equals_limit_down = (
        grouped["next_open_price"] - grouped["next_limit_down_price"]
    ).abs().le(PRICE_COMPARE_TOLERANCE)
    grouped["is_next_open_limit_up"] = open_equals_limit_up.fillna(False).astype(bool)
    grouped["is_next_open_limit_down"] = open_equals_limit_down.fillna(False).astype(bool)
    grouped["is_next_open_limit"] = (
        grouped["is_next_open_limit_up"] | grouped["is_next_open_limit_down"]
    ).fillna(False).astype(bool)

    # 兼容旧输出字段名；这里的含义已改为 T+1 开盘触及涨跌停价。
    grouped["is_next_open_one_word_limit_up"] = (
        grouped["is_next_open_limit_up"]
    )
    grouped["is_next_open_one_word_limit_down"] = (
        grouped["is_next_open_limit_down"]
    )
    grouped["is_next_open_one_word_limit"] = (
        grouped["is_next_open_limit"]
    ).fillna(False).astype(bool)

    missing_next_record_mask = grouped["next_open_price"].isna() | grouped[
        "next_period_return_before_trade_filter"
    ].isna()
    missing_next_record_mask = missing_next_record_mask.fillna(True).astype(bool)
    grouped["is_long_short_trade_candidate"] = (
        grouped["quantile_group"].eq(1) | grouped["quantile_group"].eq(group_num)
    ).fillna(False).astype(bool)
    grouped["is_tradable_next_open"] = (
        ~missing_next_record_mask & ~grouped["is_next_open_limit"]
    ).fillna(False).astype(bool)
    grouped["is_tradable_long_short_next_open"] = (
        grouped["is_long_short_trade_candidate"] & grouped["is_tradable_next_open"]
    ).fillna(False).astype(bool)

    # 因子和分组观察口径不做 T+1 涨跌停过滤；交易过滤只作用于已选出的多空两端。
    grouped["next_period_return"] = grouped["next_period_return_before_trade_filter"]
    grouped["next_period_return_after_long_short_trade_filter"] = grouped[
        "next_period_return_before_trade_filter"
    ].where(
        grouped["is_tradable_long_short_next_open"]
    )
    grouped["trade_filter_reason"] = np.select(
        [
            (~grouped["is_long_short_trade_candidate"]).to_numpy(dtype=bool),
            (
                grouped["is_long_short_trade_candidate"] & missing_next_record_mask
            ).to_numpy(dtype=bool),
            (
                grouped["is_long_short_trade_candidate"] & grouped["is_next_open_limit_up"]
            ).to_numpy(dtype=bool),
            (
                grouped["is_long_short_trade_candidate"] & grouped["is_next_open_limit_down"]
            ).to_numpy(dtype=bool),
        ],
        [
            "not_long_short_trade_candidate",
            "missing_next_clean_record_or_return",
            "next_open_limit_up",
            "next_open_limit_down",
        ],
        default="tradable_next_open",
    )
    grouped["next_period_return"] = pd.to_numeric(grouped["next_period_return"], errors="coerce")
    grouped["next_period_return_after_long_short_trade_filter"] = pd.to_numeric(
        grouped["next_period_return_after_long_short_trade_filter"],
        errors="coerce",
    )
    grouped["next_period_return_column_used"] = return_column
    grouped["rebalance_frequency"] = rebalance_frequency
    grouped["holding_days"] = holding_days
    grouped["holding_months"] = (
        int(holding_months) if rebalance_frequency == "monthly" and holding_months else (
            DEFAULT_HOLDING_MONTHS if rebalance_frequency == "monthly" else pd.NA
        )
    )
    grouped["group_num"] = group_num
    grouped["long_short_role"] = np.select(
        [
            grouped["quantile_group"].eq(1).fillna(False).to_numpy(dtype=bool),
            grouped["quantile_group"].eq(group_num).fillna(False).to_numpy(dtype=bool),
        ],
        [
            "long_top_group",
            "short_bottom_group",
        ],
        default="middle_group",
    )

    return grouped


def calculate_quantile_portfolio_returns(
    grouped_factor: pd.DataFrame,
    group_num: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算各分组等权收益，以及“做多最高动量组、做空最低动量组”的价差组合。"""

    portfolio_data = grouped_factor.copy()
    portfolio_data["is_group_return_record"] = portfolio_data["next_period_return"].notna()
    if "next_period_return_after_long_short_trade_filter" not in portfolio_data.columns:
        portfolio_data["next_period_return_after_long_short_trade_filter"] = portfolio_data[
            "next_period_return"
        ]
    if "is_long_short_trade_candidate" not in portfolio_data.columns:
        portfolio_data["is_long_short_trade_candidate"] = portfolio_data["quantile_group"].isin(
            [1, group_num]
        )
    portfolio_data["is_long_short_trade_candidate"] = portfolio_data[
        "is_long_short_trade_candidate"
    ].fillna(False).astype(bool)
    portfolio_data["is_tradable_long_short_return_record"] = portfolio_data[
        "next_period_return_after_long_short_trade_filter"
    ].notna()
    portfolio_data["next_open_limit_excluded_flag"] = (
        (
            portfolio_data["is_long_short_trade_candidate"]
            & portfolio_data["is_next_open_limit"].fillna(False).astype(bool)
        ).astype("int64")
        if "is_next_open_limit" in portfolio_data
        else (
            portfolio_data["is_long_short_trade_candidate"]
            & portfolio_data["is_next_open_one_word_limit"].fillna(False).astype(bool)
        ).astype("int64")
    )
    portfolio_data["next_open_one_word_limit_excluded_flag"] = portfolio_data["next_open_limit_excluded_flag"]
    portfolio_data["missing_next_record_flag"] = (
        (
            portfolio_data["is_long_short_trade_candidate"]
            & portfolio_data["trade_filter_reason"].eq("missing_next_clean_record_or_return")
        ).astype("int64")
    )

    # 每个组合内部股票等权：
    # 先在第 t 期根据动量分组；分组观察收益不做 T+1 涨跌停过滤。
    # 多空交易收益只在选出最高/最低动量组后，再剔除 T+1 开盘触及涨跌停和缺少完整持有期收益的股票。
    quantile_returns = (
        portfolio_data.groupby(["trade_date", "quantile_group"])
        .agg(
            group_return_sum=("next_period_return", lambda series: series.sum(min_count=1)),
            group_stock_count=("is_group_return_record", "sum"),
            long_short_group_return_sum=(
                "next_period_return_after_long_short_trade_filter",
                lambda series: series.sum(min_count=1),
            ),
            long_short_group_stock_count=("is_tradable_long_short_return_record", "sum"),
            signal_group_stock_count=("stock_code", "count"),
            next_open_limit_excluded_count=("next_open_limit_excluded_flag", "sum"),
            next_open_one_word_limit_excluded_count=("next_open_one_word_limit_excluded_flag", "sum"),
            missing_next_record_count=("missing_next_record_flag", "sum"),
            group_avg_momentum_zscore=("momentum_zscore", "mean"),
            group_avg_momentum_raw=("momentum_raw", "mean"),
            rebalance_frequency=("rebalance_frequency", "first"),
            s=("s", "first"),
            monthly_skip_months=("monthly_skip_months", "first"),
            holding_days=("holding_days", "first"),
            holding_months=("holding_months", "first"),
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
        quantile_returns["group_return_sum"] / quantile_returns["group_stock_count"]
    )
    quantile_returns["group_equal_weight_return_after_long_short_trade_filter"] = (
        quantile_returns["long_short_group_return_sum"]
        / quantile_returns["long_short_group_stock_count"]
    )
    quantile_returns = quantile_returns[
        [
            "trade_date",
            "quantile_group",
            "group_equal_weight_return",
            "group_return_sum",
            "group_stock_count",
            "group_equal_weight_return_after_long_short_trade_filter",
            "long_short_group_return_sum",
            "long_short_group_stock_count",
            "signal_group_stock_count",
            "next_open_limit_excluded_count",
            "next_open_one_word_limit_excluded_count",
            "missing_next_record_count",
            "group_avg_momentum_zscore",
            "group_avg_momentum_raw",
            "rebalance_frequency",
            "s",
            "monthly_skip_months",
            "holding_days",
            "holding_months",
            "next_trade_date_min",
            "next_trade_date_max",
            "holding_start_trade_date_min",
            "holding_start_trade_date_max",
            "holding_end_trade_date_min",
            "holding_end_trade_date_max",
        ]
    ]
    quantile_returns["quantile_group"] = quantile_returns["quantile_group"].astype("Int64")
    quantile_returns["group_stock_count"] = quantile_returns["group_stock_count"].astype("Int64")
    quantile_returns["long_short_group_stock_count"] = (
        quantile_returns["long_short_group_stock_count"].astype("Int64")
    )
    quantile_returns["signal_group_stock_count"] = quantile_returns["signal_group_stock_count"].astype("Int64")
    quantile_returns["next_open_limit_excluded_count"] = (
        quantile_returns["next_open_limit_excluded_count"].astype("Int64")
    )
    quantile_returns["next_open_one_word_limit_excluded_count"] = (
        quantile_returns["next_open_one_word_limit_excluded_count"].astype("Int64")
    )
    quantile_returns["missing_next_record_count"] = quantile_returns["missing_next_record_count"].astype("Int64")
    quantile_returns["s"] = quantile_returns["s"].astype("Int64")
    quantile_returns["monthly_skip_months"] = quantile_returns["monthly_skip_months"].astype("Int64")
    quantile_returns["holding_days"] = quantile_returns["holding_days"].astype("Int64")
    quantile_returns["holding_months"] = quantile_returns["holding_months"].astype("Int64")

    long_returns = quantile_returns.loc[
        quantile_returns["quantile_group"].eq(1),
        [
            "trade_date",
            "group_equal_weight_return_after_long_short_trade_filter",
            "long_short_group_return_sum",
            "long_short_group_stock_count",
            "signal_group_stock_count",
            "next_open_limit_excluded_count",
            "next_open_one_word_limit_excluded_count",
            "missing_next_record_count",
        ],
    ].rename(
        columns={
            "group_equal_weight_return_after_long_short_trade_filter": "long_top_group_return",
            "long_short_group_return_sum": "long_top_group_return_sum",
            "long_short_group_stock_count": "long_top_group_stock_count",
            "signal_group_stock_count": "long_top_group_signal_stock_count",
            "next_open_limit_excluded_count": "long_top_group_next_open_limit_excluded_count",
            "next_open_one_word_limit_excluded_count": "long_top_group_one_word_limit_excluded_count",
            "missing_next_record_count": "long_top_group_missing_next_record_count",
        }
    )
    short_returns = quantile_returns.loc[
        quantile_returns["quantile_group"].eq(group_num),
        [
            "trade_date",
            "group_equal_weight_return_after_long_short_trade_filter",
            "long_short_group_return_sum",
            "long_short_group_stock_count",
            "signal_group_stock_count",
            "next_open_limit_excluded_count",
            "next_open_one_word_limit_excluded_count",
            "missing_next_record_count",
        ],
    ].rename(
        columns={
            "group_equal_weight_return_after_long_short_trade_filter": "short_bottom_group_return",
            "long_short_group_return_sum": "short_bottom_group_return_sum",
            "long_short_group_stock_count": "short_bottom_group_stock_count",
            "signal_group_stock_count": "short_bottom_group_signal_stock_count",
            "next_open_limit_excluded_count": "short_bottom_group_next_open_limit_excluded_count",
            "next_open_one_word_limit_excluded_count": "short_bottom_group_one_word_limit_excluded_count",
            "missing_next_record_count": "short_bottom_group_missing_next_record_count",
        }
    )

    long_short = long_returns.merge(short_returns, on="trade_date", how="inner")
    holding_date_info = (
        quantile_returns.groupby("trade_date")
        .agg(
            holding_days=("holding_days", "first"),
            holding_months=("holding_months", "first"),
            rebalance_frequency=("rebalance_frequency", "first"),
            s=("s", "first"),
            monthly_skip_months=("monthly_skip_months", "first"),
            next_trade_date_min=("next_trade_date_min", "min"),
            next_trade_date_max=("next_trade_date_max", "max"),
            holding_start_trade_date_min=("holding_start_trade_date_min", "min"),
            holding_start_trade_date_max=("holding_start_trade_date_max", "max"),
            holding_end_trade_date_min=("holding_end_trade_date_min", "min"),
            holding_end_trade_date_max=("holding_end_trade_date_max", "max"),
        )
        .reset_index()
    )
    long_short = long_short.merge(holding_date_info, on="trade_date", how="left")

    # 多空对冲组合：做多最高动量组、做空最低动量组，收益为 long - short。
    long_short["long_short_spread_return"] = (
        long_short["long_top_group_return"] - long_short["short_bottom_group_return"]
    )
    long_short = long_short.sort_values("trade_date").reset_index(drop=True)
    long_short["long_top_group_nav"] = (1.0 + long_short["long_top_group_return"]).cumprod()
    long_short["short_bottom_group_nav"] = (1.0 + long_short["short_bottom_group_return"]).cumprod()
    long_short["long_short_spread_nav"] = (1.0 + long_short["long_short_spread_return"]).cumprod()

    return quantile_returns, long_short


def calculate_drawdown_series(nav_series: pd.Series) -> pd.DataFrame:
    """根据净值序列计算历史峰值、回撤和最大回撤。"""

    nav = pd.to_numeric(nav_series, errors="coerce")
    running_peak = nav.cummax()
    drawdown = nav / running_peak - 1.0
    return pd.DataFrame(
        {
            "nav": nav,
            "running_peak": running_peak,
            "drawdown": drawdown,
        }
    )


def calculate_performance_metrics(
    returns: pd.Series,
    nav: pd.Series,
    portfolio_name: str,
    holding_days: int,
    annualization_periods_per_year: float,
) -> dict:
    """计算单个组合的收益、波动、Sharpe、回撤、胜率等绩效指标。"""

    clean_returns = pd.to_numeric(returns, errors="coerce").dropna()
    clean_nav = pd.to_numeric(nav, errors="coerce").dropna()
    observation_count = len(clean_returns)

    if observation_count == 0 or clean_nav.empty:
        return {
            "portfolio": portfolio_name,
            "holding_days": holding_days,
            "annualization_periods_per_year": annualization_periods_per_year,
            "observation_count": 0,
        }

    cumulative_return = clean_nav.iloc[-1] - 1.0
    annual_return = clean_nav.iloc[-1] ** (annualization_periods_per_year / observation_count) - 1.0
    annual_volatility = clean_returns.std(ddof=1) * np.sqrt(annualization_periods_per_year)
    daily_mean_return = clean_returns.mean()
    daily_volatility = clean_returns.std(ddof=1)
    sharpe_ratio = (
        annual_return / annual_volatility
        if annual_volatility and not pd.isna(annual_volatility)
        else np.nan
    )

    drawdown_data = calculate_drawdown_series(clean_nav)
    max_drawdown = drawdown_data["drawdown"].min()
    calmar_ratio = (
        annual_return / abs(max_drawdown)
        if max_drawdown and not pd.isna(max_drawdown)
        else np.nan
    )

    positive_returns = clean_returns[clean_returns > 0]
    negative_returns = clean_returns[clean_returns < 0]
    win_rate = len(positive_returns) / observation_count
    loss_rate = len(negative_returns) / observation_count
    average_gain = positive_returns.mean() if not positive_returns.empty else np.nan
    average_loss = negative_returns.mean() if not negative_returns.empty else np.nan
    gain_loss_ratio = (
        average_gain / abs(average_loss)
        if average_loss and not pd.isna(average_loss)
        else np.nan
    )

    return {
        "portfolio": portfolio_name,
        "holding_days": holding_days,
        "annualization_periods_per_year": annualization_periods_per_year,
        "observation_count": observation_count,
        "cumulative_return": cumulative_return,
        "annual_return": annual_return,
        "daily_mean_return": daily_mean_return,
        "daily_volatility": daily_volatility,
        "annual_volatility": annual_volatility,
        "sharpe_ratio_rf0": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "average_gain": average_gain,
        "average_loss": average_loss,
        "gain_loss_ratio": gain_loss_ratio,
        "best_daily_return": clean_returns.max(),
        "worst_daily_return": clean_returns.min(),
    }


def calculate_performance_attribution(
    long_short_returns: pd.DataFrame,
    holding_days: int,
    annualization_periods_per_year: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """对多头、空头和价差组合做绩效归因，输出汇总、回撤序列和年度绩效。

    年化系数按调仓频率取值：未传时按日频口径 TRADING_DAYS_PER_YEAR / holding_days 兼容旧行为。
    """

    data = long_short_returns.sort_values("trade_date").reset_index(drop=True).copy()
    if annualization_periods_per_year is None:
        annualization_periods_per_year = float(TRADING_DAYS_PER_YEAR) / float(max(int(holding_days), 1))
    portfolio_specs = [
        ("long_top_group", "long_top_group_return", "long_top_group_nav"),
        ("short_bottom_group", "short_bottom_group_return", "short_bottom_group_nav"),
        ("high_minus_low_spread", "long_short_spread_return", "long_short_spread_nav"),
    ]

    summary_rows: list[dict] = []
    drawdown_frames: list[pd.DataFrame] = []
    yearly_rows: list[dict] = []

    for portfolio_name, return_col, nav_col in portfolio_specs:
        summary_rows.append(
            calculate_performance_metrics(
                returns=data[return_col],
                nav=data[nav_col],
                portfolio_name=portfolio_name,
                holding_days=holding_days,
                annualization_periods_per_year=annualization_periods_per_year,
            )
        )

        drawdown_data = calculate_drawdown_series(data[nav_col])
        drawdown_data.insert(0, "trade_date", data["trade_date"])
        drawdown_data.insert(1, "portfolio", portfolio_name)
        drawdown_frames.append(drawdown_data)

        yearly_data = data[["trade_date", return_col]].copy()
        yearly_data["year"] = pd.to_datetime(yearly_data["trade_date"]).dt.year
        for year, one_year in yearly_data.groupby("year"):
            year_returns = pd.to_numeric(one_year[return_col], errors="coerce").dropna()
            if year_returns.empty:
                continue
            year_nav = (1.0 + year_returns).cumprod()
            year_metrics = calculate_performance_metrics(
                returns=year_returns,
                nav=year_nav,
                portfolio_name=portfolio_name,
                holding_days=holding_days,
                annualization_periods_per_year=annualization_periods_per_year,
            )
            year_metrics["year"] = int(year)
            yearly_rows.append(year_metrics)

    performance_summary = pd.DataFrame(summary_rows)
    drawdown_series = pd.concat(drawdown_frames, ignore_index=True)
    yearly_performance = pd.DataFrame(yearly_rows)
    if not yearly_performance.empty:
        cols = ["year", "portfolio"] + [
            col for col in yearly_performance.columns if col not in {"year", "portfolio"}
        ]
        yearly_performance = yearly_performance[cols]

    return performance_summary, drawdown_series, yearly_performance


def parse_bool_series(series: pd.Series) -> pd.Series:
    """安全解析布尔字段，避免字符串 False 被 astype(bool) 误判为 True。"""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin(["true", "1", "yes", "y"])


def first_existing_series(data: pd.DataFrame, columns: list[str], default: object = pd.NA) -> pd.Series:
    """从多个兼容列名中取第一个存在的列；都不存在时返回默认值序列。"""

    for col in columns:
        if col in data.columns:
            return data[col]
    return pd.Series(default, index=data.index)


def two_sided_t_pvalue(t_value: float, degrees_of_freedom: int) -> float:
    """计算双侧 t 检验 p 值；优先使用 scipy，缺失时退化为正态近似。"""

    if pd.isna(t_value) or degrees_of_freedom <= 0:
        return math.nan

    try:
        from scipy import stats

        return float(2.0 * stats.t.sf(abs(t_value), df=degrees_of_freedom))
    except Exception:
        return float(math.erfc(abs(t_value) / math.sqrt(2.0)))


def mean_std_t_p(series: pd.Series) -> dict[str, float]:
    """对一条序列计算均值、标准差、t 值和 p 值。"""

    clean = pd.to_numeric(series, errors="coerce").dropna()
    observation_count = len(clean)
    if observation_count == 0:
        return {
            "mean": math.nan,
            "std": math.nan,
            "t_value": math.nan,
            "p_value": math.nan,
            "observation_count": 0,
        }

    mean_value = float(clean.mean())
    std_value = float(clean.std(ddof=1))
    if observation_count <= 1 or pd.isna(std_value) or std_value == 0:
        t_value = math.nan
        p_value = math.nan
    else:
        t_value = mean_value / (std_value / math.sqrt(observation_count))
        p_value = two_sided_t_pvalue(t_value, observation_count - 1)

    return {
        "mean": mean_value,
        "std": std_value,
        "t_value": t_value,
        "p_value": p_value,
        "observation_count": observation_count,
    }


def normalize_next_open_limit_columns(data: pd.DataFrame) -> pd.DataFrame:
    """统一 T+1 开盘涨跌停过滤字段，兼容旧版 one_word_limit 输出名。"""

    output = data.copy()
    output["is_next_open_limit_up"] = parse_bool_series(
        first_existing_series(output, ["is_next_open_limit_up", "is_next_open_one_word_limit_up"], False)
    )
    output["is_next_open_limit_down"] = parse_bool_series(
        first_existing_series(output, ["is_next_open_limit_down", "is_next_open_one_word_limit_down"], False)
    )
    output["is_next_open_limit"] = parse_bool_series(
        first_existing_series(output, ["is_next_open_limit", "is_next_open_one_word_limit"], False)
    )
    output["is_next_open_limit"] = (
        output["is_next_open_limit"]
        | output["is_next_open_limit_up"]
        | output["is_next_open_limit_down"]
    )
    output["is_next_open_one_word_limit_up"] = output["is_next_open_limit_up"]
    output["is_next_open_one_word_limit_down"] = output["is_next_open_limit_down"]
    output["is_next_open_one_word_limit"] = output["is_next_open_limit"]
    return output


def calculate_ic_ir(factor_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算日度 IC、RankIC 以及对应 IR。"""

    factor = normalize_date_columns(normalize_next_open_limit_columns(factor_data))
    factor["is_tradable_next_open"] = parse_bool_series(
        first_existing_series(factor, ["is_tradable_next_open"], False)
    )
    factor["is_long_short_trade_candidate"] = parse_bool_series(
        first_existing_series(factor, ["is_long_short_trade_candidate"], False)
    )
    if not factor["is_long_short_trade_candidate"].any() and {"quantile_group", "group_num"}.issubset(factor.columns):
        factor["is_long_short_trade_candidate"] = (
            factor["quantile_group"].eq(1) | factor["quantile_group"].eq(factor["group_num"])
        ).fillna(False)
    factor["is_tradable_long_short_next_open"] = parse_bool_series(
        first_existing_series(factor, ["is_tradable_long_short_next_open"], False)
    )
    if not factor["is_tradable_long_short_next_open"].any():
        factor["is_tradable_long_short_next_open"] = (
            factor["is_long_short_trade_candidate"] & factor["is_tradable_next_open"]
        ).fillna(False)
    factor["long_short_next_open_limit_flag"] = (
        factor["is_long_short_trade_candidate"] & factor["is_next_open_limit"]
    )
    factor["missing_next_record_flag"] = factor["trade_filter_reason"].eq("missing_next_clean_record_or_return")
    daily_audit = (
        factor.groupby("trade_date")
        .agg(
            signal_stock_count=("stock_code", "count"),
            tradable_next_open_count=("is_tradable_next_open", "sum"),
            next_open_limit_count=("is_next_open_limit", "sum"),
            long_short_trade_candidate_count=("is_long_short_trade_candidate", "sum"),
            tradable_long_short_next_open_count=("is_tradable_long_short_next_open", "sum"),
            long_short_next_open_limit_count=("long_short_next_open_limit_flag", "sum"),
            missing_next_record_count=("missing_next_record_flag", "sum"),
        )
        .reset_index()
    )
    daily_audit = normalize_date_columns(daily_audit, ["trade_date"])

    valid = factor[["trade_date", "momentum_zscore", "next_period_return"]].dropna().copy()
    ic_rows: list[dict] = []

    for trade_date, one_day in valid.groupby("trade_date"):
        if len(one_day) < 2:
            continue
        next_trade_date = factor.loc[factor["trade_date"].eq(trade_date), "next_trade_date"].dropna().min()
        ic_rows.append(
            {
                "trade_date": trade_date,
                "next_trade_date": next_trade_date,
                "ic": one_day["momentum_zscore"].corr(one_day["next_period_return"], method="pearson"),
                "rank_ic": one_day["momentum_zscore"].corr(one_day["next_period_return"], method="spearman"),
                "ic_stock_count": len(one_day),
            }
        )

    ic_series = pd.DataFrame(ic_rows)
    if ic_series.empty:
        ic_series = pd.DataFrame(
            columns=[
                "trade_date",
                "next_trade_date",
                "ic",
                "rank_ic",
                "ic_stock_count",
            ]
        )
    else:
        ic_series = ic_series.dropna(subset=["ic", "rank_ic"])
    ic_series = normalize_date_columns(ic_series, ["trade_date", "next_trade_date"])
    ic_series = ic_series.merge(daily_audit, on="trade_date", how="left")
    ic_series = ic_series.sort_values("trade_date").reset_index(drop=True)
    ic_series["cumulative_ic"] = ic_series["ic"].cumsum()
    ic_series["cumulative_rank_ic"] = ic_series["rank_ic"].cumsum()
    ic_series["ic_20d_rolling_mean"] = ic_series["ic"].rolling(window=20, min_periods=5).mean()
    ic_series["rank_ic_20d_rolling_mean"] = ic_series["rank_ic"].rolling(window=20, min_periods=5).mean()

    summary_rows: list[dict] = []
    for metric in ["ic", "rank_ic"]:
        stats = mean_std_t_p(ic_series[metric])
        mean_value = stats["mean"]
        std_value = stats["std"]
        ir_value = mean_value / std_value if std_value and not pd.isna(std_value) else math.nan
        summary_rows.append(
            {
                "metric": metric.upper() if metric == "ic" else "RankIC",
                "mean": mean_value,
                "std": std_value,
                "t_value": stats["t_value"],
                "p_value": stats["p_value"],
                "ir": ir_value,
                "annualized_ir_sqrt252": ir_value * math.sqrt(TRADING_DAYS_PER_YEAR)
                if not pd.isna(ir_value)
                else math.nan,
                "observation_count": stats["observation_count"],
            }
        )

    return ic_series, pd.DataFrame(summary_rows)


def calculate_factor_value_statistics(factor_data: pd.DataFrame) -> pd.DataFrame:
    """补充计算动量因子本身的均值、标准差、t 值和 p 值。"""

    rows: list[dict] = []
    for col in ["momentum_raw", "momentum_3sigma", "momentum_zscore"]:
        stats = mean_std_t_p(factor_data[col])
        rows.append({"factor_column": col, **stats})
    return pd.DataFrame(rows)


def summarize_factor_input(factor_data: pd.DataFrame) -> pd.DataFrame:
    """汇总 IC 输入数据的时间范围、收益口径和缺失情况，便于复核诊断口径。"""

    factor_data = normalize_next_open_limit_columns(factor_data)
    factor_data["is_tradable_next_open"] = parse_bool_series(
        first_existing_series(factor_data, ["is_tradable_next_open"], False)
    )
    factor_data["is_long_short_trade_candidate"] = parse_bool_series(
        first_existing_series(factor_data, ["is_long_short_trade_candidate"], False)
    )
    if not factor_data["is_long_short_trade_candidate"].any() and {"quantile_group", "group_num"}.issubset(factor_data.columns):
        factor_data["is_long_short_trade_candidate"] = (
            factor_data["quantile_group"].eq(1) | factor_data["quantile_group"].eq(factor_data["group_num"])
        ).fillna(False)
    factor_data["is_tradable_long_short_next_open"] = parse_bool_series(
        first_existing_series(factor_data, ["is_tradable_long_short_next_open"], False)
    )
    if not factor_data["is_tradable_long_short_next_open"].any():
        factor_data["is_tradable_long_short_next_open"] = (
            factor_data["is_long_short_trade_candidate"] & factor_data["is_tradable_next_open"]
        ).fillna(False)

    valid_factor = factor_data.loc[factor_data["momentum_zscore"].notna()].copy()
    valid_return = factor_data.loc[factor_data["next_period_return"].notna()].copy()
    next_open_limit = factor_data.loc[factor_data["is_next_open_limit"]].copy()
    long_short_candidate = factor_data.loc[factor_data["is_long_short_trade_candidate"]].copy()
    long_short_next_open_limit = factor_data.loc[
        factor_data["is_long_short_trade_candidate"] & factor_data["is_next_open_limit"]
    ].copy()
    tradable_next_open = factor_data.loc[factor_data["is_tradable_next_open"]].copy()
    tradable_long_short_next_open = factor_data.loc[
        factor_data["is_tradable_long_short_next_open"]
    ].copy()
    missing_next_record = factor_data.loc[
        factor_data["trade_filter_reason"].eq("missing_next_clean_record_or_return")
    ].copy()
    factor_return_column_values = (
        factor_data["return_column_used"].dropna().astype(str).unique().tolist()
        if "return_column_used" in factor_data.columns
        else []
    )
    next_return_column_values = (
        factor_data["next_period_return_column_used"].dropna().astype(str).unique().tolist()
        if "next_period_return_column_used" in factor_data.columns
        else []
    )
    lookback_days_values = (
        factor_data["lookback_days"].dropna().astype(int).astype(str).unique().tolist()
        if "lookback_days" in factor_data.columns
        else []
    )
    lookback_unit_values = (
        factor_data["lookback_unit"].dropna().astype(str).unique().tolist()
        if "lookback_unit" in factor_data.columns
        else []
    )
    holding_days_values = (
        factor_data["holding_days"].dropna().astype(int).astype(str).unique().tolist()
        if "holding_days" in factor_data.columns
        else []
    )
    holding_months_values = (
        factor_data["holding_months"].dropna().astype(int).astype(str).unique().tolist()
        if "holding_months" in factor_data.columns
        else []
    )
    s_values = (
        factor_data["s"].dropna().astype(int).astype(str).unique().tolist()
        if "s" in factor_data.columns
        else []
    )
    monthly_skip_months_values = (
        factor_data["monthly_skip_months"].dropna().astype(int).astype(str).unique().tolist()
        if "monthly_skip_months" in factor_data.columns
        else []
    )
    rebalance_frequency_values = (
        factor_data["rebalance_frequency"].dropna().astype(str).unique().tolist()
        if "rebalance_frequency" in factor_data.columns
        else []
    )
    group_num_values = (
        factor_data["group_num"].dropna().astype(int).astype(str).unique().tolist()
        if "group_num" in factor_data.columns
        else []
    )

    rows = [
        ["factor_rows", len(factor_data)],
        ["valid_zscore_rows", len(valid_factor)],
        ["valid_next_return_rows", len(valid_return)],
        ["tradable_next_open_rows", len(tradable_next_open)],
        ["next_open_limit_rows", len(next_open_limit)],
        ["long_short_trade_candidate_rows", len(long_short_candidate)],
        ["long_short_next_open_limit_rows", len(long_short_next_open_limit)],
        ["tradable_long_short_next_open_rows", len(tradable_long_short_next_open)],
        ["missing_next_clean_record_or_return_rows", len(missing_next_record)],
        ["trade_date_start", factor_data["trade_date"].min()],
        ["trade_date_end", factor_data["trade_date"].max()],
        ["momentum_start_date_min", factor_data["momentum_start_date"].min() if "momentum_start_date" in factor_data.columns else pd.NaT],
        ["momentum_start_date_max", factor_data["momentum_start_date"].max() if "momentum_start_date" in factor_data.columns else pd.NaT],
        ["momentum_end_date_min", factor_data["momentum_end_date"].min() if "momentum_end_date" in factor_data.columns else pd.NaT],
        ["momentum_end_date_max", factor_data["momentum_end_date"].max() if "momentum_end_date" in factor_data.columns else pd.NaT],
        ["holding_start_trade_date_min", factor_data["holding_start_trade_date"].min() if "holding_start_trade_date" in factor_data.columns else pd.NaT],
        ["holding_start_trade_date_max", factor_data["holding_start_trade_date"].max() if "holding_start_trade_date" in factor_data.columns else pd.NaT],
        ["holding_end_trade_date_min", factor_data["holding_end_trade_date"].min() if "holding_end_trade_date" in factor_data.columns else pd.NaT],
        ["holding_end_trade_date_max", factor_data["holding_end_trade_date"].max() if "holding_end_trade_date" in factor_data.columns else pd.NaT],
        ["next_trade_date_start", factor_data["next_trade_date"].min() if "next_trade_date" in factor_data.columns else pd.NaT],
        ["next_trade_date_end", factor_data["next_trade_date"].max() if "next_trade_date" in factor_data.columns else pd.NaT],
        ["rebalance_frequency", ",".join(rebalance_frequency_values)],
        ["lookback_days", ",".join(lookback_days_values)],
        ["lookback_unit", ",".join(lookback_unit_values)],
        ["s", ",".join(s_values)],
        ["monthly_skip_months", ",".join(monthly_skip_months_values)],
        ["holding_days", ",".join(holding_days_values)],
        ["holding_months", ",".join(holding_months_values)],
        ["group_num", ",".join(group_num_values)],
        ["factor_return_column_used", ",".join(factor_return_column_values)],
        ["next_period_return_column_used", ",".join(next_return_column_values)],
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def format_factor_dates(data: pd.DataFrame) -> pd.DataFrame:
    """统一格式化因子输出中的日期字段。"""

    output = data.copy()
    for col in [
        "trade_date",
        "momentum_start_date",
        "momentum_end_date",
        "next_trade_date",
        "holding_start_trade_date",
        "holding_end_trade_date",
        "next_trade_date_min",
        "next_trade_date_max",
        "holding_start_trade_date_min",
        "holding_start_trade_date_max",
        "holding_end_trade_date_min",
        "holding_end_trade_date_max",
    ]:
        if col in output.columns:
            output[col] = format_date_series(output[col])
    return output


def build_all_momentum_factor(raw_factor: pd.DataFrame, missing_factor: pd.DataFrame) -> pd.DataFrame:
    """合并有效和缺失动量因子记录，形成全股票横截面动量因子检查表。"""

    factor_parts = [part for part in [raw_factor, missing_factor] if not part.empty]
    if not factor_parts:
        return pd.DataFrame()
    return pd.concat(factor_parts, ignore_index=True).sort_values(
        ["trade_date", "stock_code"],
    ).reset_index(drop=True)


LONG_SHORT_MEMBER_COLUMNS = [
    "trade_date",
    "next_trade_date",
    "holding_start_trade_date",
    "holding_end_trade_date",
    "portfolio_side",
    "long_short_role",
    "stock_code",
    "quantile_group",
    "standardized_momentum_rank_desc",
    "cross_section_stock_count_after_zscore",
    "momentum_raw",
    "momentum_3sigma",
    "momentum_zscore",
    "signal_close_price",
    "next_open_price",
    "next_limit_up_price",
    "next_limit_down_price",
    "next_period_return_before_trade_filter",
    "next_period_return",
    "next_period_return_after_long_short_trade_filter",
    "future_return_valid_days",
    "has_complete_holding_return",
    "is_next_open_one_word_limit_up",
    "is_next_open_one_word_limit_down",
    "is_next_open_one_word_limit",
    "is_next_open_limit_up",
    "is_next_open_limit_down",
    "is_next_open_limit",
    "is_tradable_next_open",
    "is_long_short_trade_candidate",
    "is_tradable_long_short_next_open",
    "trade_filter_reason",
    "rebalance_frequency",
    "lookback_days",
    "lookback_unit",
    "s",
    "monthly_skip_months",
    "holding_days",
    "holding_months",
    "group_num",
]


def safe_nunique(data: pd.DataFrame, column: str) -> int:
    """安全统计唯一值数量；空表或缺列返回 0，避免日志阶段中断主流程。"""

    if column not in data.columns:
        return 0
    return int(data[column].nunique())


def safe_masked_nunique(data: pd.DataFrame, mask: pd.Series, column: str) -> int:
    """安全统计筛选后的唯一值数量；空表、缺列或空 mask 返回 0。"""

    if column not in data.columns or data.empty:
        return 0
    aligned_mask = mask.reindex(data.index, fill_value=False).astype(bool)
    return int(data.loc[aligned_mask, column].nunique())


def build_long_short_stock_members(grouped_factor: pd.DataFrame, group_num: int) -> pd.DataFrame:
    """提取每日做多最高动量组和做空最低动量组的具体股票清单。"""

    if grouped_factor.empty:
        return pd.DataFrame(columns=LONG_SHORT_MEMBER_COLUMNS)

    member_data = grouped_factor.loc[
        grouped_factor["quantile_group"].isin([1, group_num])
    ].copy()
    if member_data.empty:
        return member_data

    member_data["portfolio_side"] = np.select(
        [
            member_data["quantile_group"].eq(1).fillna(False).to_numpy(dtype=bool),
            member_data["quantile_group"].eq(group_num).fillna(False).to_numpy(dtype=bool),
        ],
        [
            "long_high_momentum_group",
            "short_low_momentum_group",
        ],
        default="other",
    )

    existing_columns = [col for col in LONG_SHORT_MEMBER_COLUMNS if col in member_data.columns]
    return member_data[existing_columns].sort_values(
        ["trade_date", "quantile_group", "standardized_momentum_rank_desc", "stock_code"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


# ============================================================
# 6. 主流程：清洗 -> 计算动量 -> 3sigma -> Z-score -> 标准化因子排序 -> 输出
# ============================================================


def main() -> None:
    args = parse_args()
    if args.lookback_days <= 0:
        raise SystemExit("lookback-days 必须为正整数。")
    if args.holding_days <= 0:
        raise SystemExit("holding-days 必须为正整数。")
    if args.lookback_months <= 0:
        raise SystemExit("lookback-months 必须为正整数。")
    if args.holding_months <= 0:
        raise SystemExit("holding-months 必须为正整数。")
    if args.s < 0:
        raise SystemExit("s 必须为非负整数。")
    if args.rebalance_frequency == "monthly" and args.s >= args.lookback_months:
        raise SystemExit("月频下 s 必须小于 lookback-months。")
    if args.group_num <= 1:
        raise SystemExit("group-num 必须大于 1。")
    if args.min_float_market_value < 0:
        raise SystemExit("min-float-market-value 不能为负数。")
    selected_market_types = set(A_SHARE_MARKET_TYPES)
    if not selected_market_types:
        raise SystemExit("A_SHARE_MARKET_TYPES 至少需要包含一个市场类型。")

    selected_market_types_text = ",".join(str(item) for item in sorted(selected_market_types))
    experiment_folder_name = build_experiment_folder_name(
        lookback_days=args.lookback_days,
        holding_days=args.holding_days,
        group_num=args.group_num,
        market_types=selected_market_types,
        rebalance_frequency=args.rebalance_frequency,
        lookback_months=args.lookback_months,
        holding_months=args.holding_months,
        s=args.s,
    )
    is_monthly = args.rebalance_frequency == "monthly"
    lookback_label = (
        f"{args.lookback_months} 个月，s={args.s}"
        if is_monthly
        else f"{args.lookback_days} 个交易日"
    )
    holding_label = f"{args.holding_months} 个月" if is_monthly else f"{args.holding_days} 个交易日"
    performance_holding_days = (
        int(round(TRADING_DAYS_PER_YEAR / MONTHS_PER_YEAR * args.holding_months))
        if is_monthly
        else args.holding_days
    )
    annualization_periods_per_year = resolve_annualization_periods_per_year(
        rebalance_frequency=args.rebalance_frequency,
        holding_days=args.holding_days,
        holding_months=args.holding_months,
    )

    output_base_dir: Path = args.output_dir
    output_dir: Path = resolve_experiment_output_dir(output_base_dir, experiment_folder_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_tsv_path = output_dir / "_raw_market_data_for_momentum.tsv"

    raw_cache_path = build_local_raw_data_cache_path(args.start_date, args.end_date)
    clean_cache_paths = build_local_clean_data_cache_paths(
        start_date=args.start_date,
        end_date=args.end_date,
        min_float_market_value=args.min_float_market_value,
        market_types=selected_market_types,
    )
    raw_tsv_data = pd.DataFrame(columns=SOURCE_COLUMNS)
    raw_data_source = "skipped_because_clean_cache_used"

    print("1/10 正在检查清洗后本地缓存和原始行情缓存...")
    if (
        clean_cache_is_complete(clean_cache_paths)
        and not args.refresh_clean_data
        and not args.refresh_local_data
    ):
        print(f"发现清洗并前值填充后的本地缓存，直接读取：{clean_cache_paths['clean_data']}")
        clean_data, cleaning_log, missing_summary, exclusion_summary = load_clean_market_data_from_cache(
            clean_cache_paths,
        )
        clean_data_source = "local_clean_cache"
        print("2/10 已复用清洗并前值填充后的本地缓存，跳过基础清洗步骤。")
    else:
        if args.refresh_clean_data:
            print("已指定 refresh-clean-data，将重新执行基础清洗并覆盖清洗缓存。")
        elif args.refresh_local_data:
            print("已指定 refresh-local-data，将重新读取原始数据并覆盖清洗缓存。")
        else:
            print("未发现清洗数据缓存，将读取原始行情数据并重新清洗。")

        raw_tsv_data, raw_cache_path, raw_data_source = load_raw_market_data_with_cache(
            start_date=args.start_date,
            end_date=args.end_date,
            temp_tsv_path=temp_tsv_path,
            refresh_local_data=args.refresh_local_data,
            keep_temp=args.keep_temp,
        )

        print("2/10 正在执行基础清洗并前值填充...")
        clean_data, cleaning_log, missing_summary, exclusion_summary = clean_market_data(
            raw_data=raw_tsv_data,
            min_float_market_value=args.min_float_market_value,
        )
        print(f"正在写入清洗并前值填充后的本地缓存：{clean_cache_paths['clean_data']}")
        write_clean_market_data_cache(
            clean_data=clean_data,
            cleaning_log=cleaning_log,
            missing_summary=missing_summary,
            exclusion_summary=exclusion_summary,
            cache_paths=clean_cache_paths,
        )
        clean_data_source = "fresh_cleaned_and_cached"

    print(f"3/10 正在计算所有股票原始 {lookback_label}横截面动量因子...")
    raw_factor, missing_factor = calculate_raw_momentum(
        clean_data=clean_data,
        return_column=args.return_column,
        lookback_days=args.lookback_days,
        rebalance_frequency=args.rebalance_frequency,
        lookback_months=args.lookback_months,
        s=args.s,
    )
    all_momentum_factor = build_all_momentum_factor(raw_factor, missing_factor)

    print("4/10 正在按交易日截面做 3sigma 去极值...")
    factor_3sigma, extreme_records, date_summary = apply_3sigma_winsorization(raw_factor)

    print("5/10 正在按交易日截面做 Z-score 标准化...")
    factor_zscore = apply_zscore_standardization(factor_3sigma)

    print("6/10 正在按标准化动量因子从大到小做截面排序...")
    ranked_factor = rank_standardized_momentum_cross_section(factor_zscore)

    print(f"7/10 正在按标准化动量排名分成 {args.group_num} 组，并匹配未来 {holding_label}收益...")
    grouped_factor = assign_quantile_groups(
        ranked_factor=ranked_factor,
        clean_data=clean_data,
        return_column=args.return_column,
        group_num=args.group_num,
        holding_days=args.holding_days,
        rebalance_frequency=args.rebalance_frequency,
        holding_months=args.holding_months,
    )
    long_short_members = build_long_short_stock_members(grouped_factor, args.group_num)

    print("8/10 正在整理做多组和做空组具体股票...")

    print(f"9/10 正在计算未来 {holding_label}等权分组收益和多空对冲组合收益...")
    quantile_returns, long_short_returns = calculate_quantile_portfolio_returns(
        grouped_factor=grouped_factor,
        group_num=args.group_num,
    )
    performance_summary, drawdown_series, yearly_performance = calculate_performance_attribution(
        long_short_returns=long_short_returns,
        holding_days=performance_holding_days,
        annualization_periods_per_year=annualization_periods_per_year,
    )
    print("正在基于内存中的分组明细生成 IC/IR 诊断小表，无需落盘 07 明细文件...")
    ic_series, ic_summary = calculate_ic_ir(grouped_factor)
    factor_value_statistics = calculate_factor_value_statistics(grouped_factor)
    factor_input_summary = summarize_factor_input(grouped_factor)

    print("10/10 正在输出结果文件...")
    if SAVE_OUTPUT_01_SQL_RAW_MARKET_DATA and raw_tsv_data.empty and raw_cache_path.exists():
        print("需要输出 01 原始数据文件，正在从本地原始缓存补充读取原始行情数据...")
        raw_tsv_data = read_raw_csv(raw_cache_path)
        raw_data_source = f"{raw_data_source}_plus_raw_cache_for_output"

    # 按用户要求的研究流程顺序输出编号文件，方便逐步检查每一阶段数据。
    write_optional_csv(
        SAVE_OUTPUT_01_SQL_RAW_MARKET_DATA,
        raw_tsv_data,
        output_dir / "01_sql_raw_market_data.csv",
        f"原始数据已复用本地缓存：{raw_cache_path}",
    )
    write_optional_csv(
        SAVE_OUTPUT_02_CLEANED_MARKET_DATA,
        format_factor_dates(clean_data),
        output_dir / "02_cleaned_market_data_after_basic_cleaning_ffill.csv",
        f"清洗后全量行情表已另存为本地缓存：{clean_cache_paths['clean_data']}",
    )
    write_optional_csv(
        SAVE_OUTPUT_03_ALL_MOMENTUM_FACTOR,
        format_factor_dates(all_momentum_factor),
        output_dir / "03_momentum_factor_all_stocks.csv",
        "全股票动量因子表较大，已仅在内存中用于后续处理。",
    )
    write_optional_csv(
        SAVE_OUTPUT_04_3SIGMA_FACTOR,
        format_factor_dates(factor_3sigma),
        output_dir / "04_momentum_factor_3sigma.csv",
        "3sigma 中间表已在内存中用于后续排序和分组。",
    )
    write_optional_csv(
        SAVE_OUTPUT_05_ZSCORE_FACTOR,
        format_factor_dates(factor_zscore),
        output_dir / "05_momentum_factor_zscore.csv",
        "Z-score 中间表已在内存中用于后续排序和分组。",
    )
    write_optional_csv(
        SAVE_OUTPUT_06_RANKED_FACTOR,
        format_factor_dates(ranked_factor),
        output_dir / "06_momentum_factor_rank_zscore.csv",
        "标准化排序明细已在内存中用于分组和收益计算。",
    )
    write_optional_csv(
        SAVE_OUTPUT_07_GROUPED_FACTOR_WITH_FORWARD_RETURN,
        format_factor_dates(grouped_factor),
        output_dir / "07_momentum_factor_quantile_groups_with_forward_returns.csv",
        "已关闭 07 明细输出；IC 诊断会使用主脚本直接输出的 12 号小表。",
    )
    write_optional_csv(
        SAVE_OUTPUT_08_LONG_SHORT_STOCK_MEMBERS,
        format_factor_dates(long_short_members),
        output_dir / "08_long_short_group_stock_members.csv",
        "做多做空股票明细较大，已仅在内存中用于组合收益计算。",
    )
    write_optional_csv(
        SAVE_OUTPUT_09_QUANTILE_EQUAL_WEIGHT_RETURNS,
        format_factor_dates(quantile_returns),
        output_dir / "09_quantile_equal_weight_returns.csv",
        "关闭后将无法用当前实验目录绘制分组收益曲线。",
    )
    write_optional_csv(
        SAVE_OUTPUT_10_LONG_SHORT_HEDGE_RETURNS,
        format_factor_dates(long_short_returns),
        output_dir / "10_long_short_hedge_returns.csv",
        "关闭后将无法用当前实验目录绘制多空组合收益曲线。",
    )
    write_csv(format_factor_dates(ic_series), output_dir / "12_momentum_ic_series.csv")
    write_csv(ic_summary, output_dir / "12_momentum_ic_ir_summary.csv")
    write_csv(factor_value_statistics, output_dir / "12_momentum_factor_value_statistics.csv")
    write_csv(factor_input_summary, output_dir / "12_momentum_factor_input_summary.csv")

    # 这些小型日志和摘要文件通常用于复核流程，体量很小，默认保留。
    write_csv(cleaning_log, output_dir / "cleaning_step_log.csv")
    write_csv(exclusion_summary, output_dir / "cleaning_exclusion_reason_summary.csv")
    write_csv(missing_summary, output_dir / "missing_value_ffill_summary.csv")

    if SAVE_LEGACY_COMPATIBILITY_CSV_OUTPUTS:
        write_csv(format_factor_dates(missing_factor), output_dir / "momentum_factor_missing_lookback.csv")
        write_csv(format_factor_dates(raw_factor), output_dir / "momentum_factor_raw.csv")
        write_csv(format_factor_dates(factor_3sigma), output_dir / "momentum_factor_3sigma.csv")
        write_csv(format_factor_dates(extreme_records), output_dir / "momentum_factor_3sigma_extreme_records.csv")
        write_csv(format_factor_dates(factor_zscore), output_dir / "momentum_factor_zscore.csv")
        write_csv(format_factor_dates(ranked_factor), output_dir / "momentum_factor_rank_zscore.csv")
        write_csv(format_factor_dates(grouped_factor), output_dir / "momentum_factor_quantile_groups.csv")
        write_csv(format_factor_dates(long_short_members), output_dir / "momentum_long_short_group_stock_members.csv")
        write_csv(format_factor_dates(quantile_returns), output_dir / "momentum_quantile_equal_weight_returns.csv")
        write_csv(format_factor_dates(long_short_returns), output_dir / "momentum_long_short_returns.csv")
    else:
        print("跳过输出旧版兼容 CSV；诊断脚本会读取 09/10 和 12 号诊断小表。")

    # 绩效归因和日期摘要体量较小，保留输出，便于复核策略表现。
    write_csv(format_factor_dates(performance_summary), output_dir / "momentum_performance_summary.csv")
    write_csv(format_factor_dates(drawdown_series), output_dir / "momentum_drawdown_series.csv")
    write_csv(format_factor_dates(yearly_performance), output_dir / "momentum_yearly_performance.csv")
    write_csv(format_factor_dates(date_summary), output_dir / "momentum_factor_date_summary.csv")

    lookback_limit_mask = (
        all_momentum_factor["lookback_has_limit_up_or_down"].fillna(False).astype(bool)
        if "lookback_has_limit_up_or_down" in all_momentum_factor
        else pd.Series(False, index=all_momentum_factor.index)
    )
    next_open_limit_mask = (
        (
            grouped_factor["is_long_short_trade_candidate"].fillna(False).astype(bool)
            & grouped_factor["is_next_open_limit"].fillna(False).astype(bool)
        )
        if {"is_long_short_trade_candidate", "is_next_open_limit"}.issubset(grouped_factor.columns)
        else pd.Series(False, index=grouped_factor.index)
    )

    factor_log = pd.DataFrame(
        [
            ["sql_raw_market_data_loaded_from_cache_or_mysql", len(raw_tsv_data), safe_nunique(raw_tsv_data, "Stkcd"), safe_nunique(raw_tsv_data, "Trddt")],
            ["cleaned_stock_pool", len(clean_data), safe_nunique(clean_data, "stock_code"), safe_nunique(clean_data, "trade_date")],
            ["all_momentum_factor_records", len(all_momentum_factor), safe_nunique(all_momentum_factor, "stock_code"), safe_nunique(all_momentum_factor, "trade_date")],
            ["raw_momentum_valid", len(raw_factor), safe_nunique(raw_factor, "stock_code"), safe_nunique(raw_factor, "trade_date")],
            ["raw_momentum_missing", len(missing_factor), safe_nunique(missing_factor, "stock_code"), safe_nunique(missing_factor, "trade_date")],
            [
                "lookback_window_limit_marked_not_excluded",
                int(lookback_limit_mask.sum()),
                safe_masked_nunique(all_momentum_factor, lookback_limit_mask, "stock_code"),
                safe_masked_nunique(all_momentum_factor, lookback_limit_mask, "trade_date"),
            ],
            ["three_sigma_processed", len(factor_3sigma), safe_nunique(factor_3sigma, "stock_code"), safe_nunique(factor_3sigma, "trade_date")],
            ["three_sigma_extreme_records", len(extreme_records), safe_nunique(extreme_records, "stock_code"), safe_nunique(extreme_records, "trade_date")],
            ["zscore_standardized", len(factor_zscore), safe_nunique(factor_zscore, "stock_code"), safe_nunique(factor_zscore, "trade_date")],
            ["standardized_momentum_ranked", len(ranked_factor), safe_nunique(ranked_factor, "stock_code"), safe_nunique(ranked_factor, "trade_date")],
            ["quantile_grouped_factor", len(grouped_factor), safe_nunique(grouped_factor, "stock_code"), safe_nunique(grouped_factor, "trade_date")],
            [
                "next_open_limit_excluded",
                int(next_open_limit_mask.sum()),
                safe_masked_nunique(grouped_factor, next_open_limit_mask, "stock_code"),
                safe_masked_nunique(grouped_factor, next_open_limit_mask, "trade_date"),
            ],
            ["quantile_equal_weight_returns", len(quantile_returns), safe_nunique(quantile_returns, "quantile_group"), safe_nunique(quantile_returns, "trade_date")],
            ["long_short_group_stock_members", len(long_short_members), safe_nunique(long_short_members, "stock_code"), safe_nunique(long_short_members, "trade_date")],
            ["long_short_returns", len(long_short_returns), None, safe_nunique(long_short_returns, "trade_date")],
            ["ic_series", len(ic_series), None, safe_nunique(ic_series, "trade_date")],
            ["ic_summary", len(ic_summary), safe_nunique(ic_summary, "metric"), None],
            ["factor_value_statistics", len(factor_value_statistics), safe_nunique(factor_value_statistics, "factor_column"), None],
            ["factor_input_summary", len(factor_input_summary), None, None],
            ["performance_summary", len(performance_summary), safe_nunique(performance_summary, "portfolio"), None],
            ["drawdown_series", len(drawdown_series), safe_nunique(drawdown_series, "portfolio"), safe_nunique(drawdown_series, "trade_date")],
            ["yearly_performance", len(yearly_performance), safe_nunique(yearly_performance, "portfolio"), None],
        ],
        columns=["step", "rows", "entity_count", "trade_date_count"],
    )
    write_csv(factor_log, output_dir / "factor_step_log.csv")

    run_summary = pd.DataFrame(
        [
            ["start_date", args.start_date],
            ["end_date", args.end_date],
            ["rebalance_frequency", args.rebalance_frequency],
            ["lookback_days", args.lookback_days],
            ["lookback_months", args.lookback_months],
            ["s", args.s],
            ["monthly_skip_months", args.s],
            ["holding_days", args.holding_days],
            ["holding_months", args.holding_months],
            ["effective_lookback_label", lookback_label],
            ["effective_holding_label", holding_label],
            ["annualization_periods_per_year", annualization_periods_per_year],
            ["group_num", args.group_num],
            ["return_column", args.return_column],
            ["selected_market_types", selected_market_types_text],
            ["market_types_folder_tag", market_types_to_tag(selected_market_types)],
            ["experiment_folder_name", experiment_folder_name],
            ["output_base_dir", str(output_base_dir)],
            ["raw_data_source", raw_data_source],
            ["local_raw_data_cache_path", str(raw_cache_path)],
            ["clean_data_source", clean_data_source],
            ["local_clean_data_cache_path", str(clean_cache_paths["clean_data"])],
            ["save_output_01_sql_raw_market_data", SAVE_OUTPUT_01_SQL_RAW_MARKET_DATA],
            ["save_output_02_cleaned_market_data", SAVE_OUTPUT_02_CLEANED_MARKET_DATA],
            ["save_output_03_all_momentum_factor", SAVE_OUTPUT_03_ALL_MOMENTUM_FACTOR],
            ["save_output_04_3sigma_factor", SAVE_OUTPUT_04_3SIGMA_FACTOR],
            ["save_output_05_zscore_factor", SAVE_OUTPUT_05_ZSCORE_FACTOR],
            ["save_output_06_ranked_factor", SAVE_OUTPUT_06_RANKED_FACTOR],
            ["save_output_07_grouped_factor_with_forward_return", SAVE_OUTPUT_07_GROUPED_FACTOR_WITH_FORWARD_RETURN],
            ["save_output_08_long_short_stock_members", SAVE_OUTPUT_08_LONG_SHORT_STOCK_MEMBERS],
            ["save_output_09_quantile_equal_weight_returns", SAVE_OUTPUT_09_QUANTILE_EQUAL_WEIGHT_RETURNS],
            ["save_output_10_long_short_hedge_returns", SAVE_OUTPUT_10_LONG_SHORT_HEDGE_RETURNS],
            ["save_legacy_compatibility_csv_outputs", SAVE_LEGACY_COMPATIBILITY_CSV_OUTPUTS],
            ["refresh_local_data", args.refresh_local_data],
            ["refresh_clean_data", args.refresh_clean_data],
            ["min_float_market_value_dsmvosd_unit", args.min_float_market_value],
            ["min_float_market_value_yuan_estimate", args.min_float_market_value * 1000.0],
            ["raw_rows_loaded", len(raw_tsv_data)],
            ["cleaned_rows", len(clean_data)],
            ["all_momentum_factor_rows", len(all_momentum_factor)],
            [
                "unselected_market_type_excluded_rows",
                int(
                    exclusion_summary.loc[
                        exclusion_summary["exclude_reason"].eq("not_selected_market_type"),
                        "record_count",
                    ].sum()
                ),
            ],
            [
                "below_min_float_market_value_rows",
                int(
                    exclusion_summary.loc[
                        exclusion_summary["exclude_reason"].eq("below_min_float_market_value"),
                        "record_count",
                    ].sum()
                ),
            ],
            ["valid_momentum_rows", len(raw_factor)],
            ["missing_momentum_rows", len(missing_factor)],
            ["lookback_window_limit_marked_not_excluded_rows", int(lookback_limit_mask.sum())],
            ["extreme_3sigma_rows", len(extreme_records)],
            ["zscore_rows", len(factor_zscore)],
            ["ranked_zscore_rows", len(ranked_factor)],
            ["quantile_grouped_rows", len(grouped_factor)],
            ["long_short_group_stock_member_rows", len(long_short_members)],
            ["next_open_limit_excluded_rows", int(next_open_limit_mask.sum())],
            ["quantile_return_rows", len(quantile_returns)],
            ["long_short_return_rows", len(long_short_returns)],
            ["ic_series_rows", len(ic_series)],
            ["ic_summary_rows", len(ic_summary)],
            ["factor_value_statistics_rows", len(factor_value_statistics)],
            ["factor_input_summary_rows", len(factor_input_summary)],
            ["performance_summary_rows", len(performance_summary)],
            ["drawdown_series_rows", len(drawdown_series)],
            ["yearly_performance_rows", len(yearly_performance)],
            ["output_dir", str(output_dir)],
        ],
        columns=["metric", "value"],
    )
    write_csv(run_summary, output_dir / "run_summary.csv")

    print("动量因子处理完成。")
    print(f"输出目录：{output_dir}")
    print(f"实验文件夹：{experiment_folder_name}")
    print(f"调仓频率：{args.rebalance_frequency}；回看：{lookback_label}；持有：{holding_label}")
    print(f"保留 Markettype：{selected_market_types_text}")
    print(f"流通市值下限 Dsmvosd：{args.min_float_market_value:,.0f}（约 {args.min_float_market_value * 1000.0:,.0f} 元）")
    print(f"清洗后股票池记录数：{len(clean_data):,}")  
    print(f"有效动量因子记录数：{len(raw_factor):,}")
    print(f"3sigma 极端值记录数：{len(extreme_records):,}")
    print(f"最终 Z-score 因子记录数：{len(factor_zscore):,}")
    print(f"最终标准化动量排序记录数：{len(ranked_factor):,}")
    print(f"分组后因子记录数：{len(grouped_factor):,}")
    print(f"多空组合收益记录数：{len(long_short_returns):,}")
    print("绩效归因摘要：")
    print(performance_summary.to_string(index=False))


if __name__ == "__main__":
    main()
