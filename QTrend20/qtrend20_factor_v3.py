from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


# 脚本所在目录即本因子的工作目录；v2 结果写入独立目录，避免覆盖其他版本的 output。
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output_v3"
DEFAULT_CACHE_DIR = PROJECT_DIR / "_cache"
# 复用当前项目已有的原始行情缓存；只扫描命名为 _local_data_cache 的目录，避免遍历无关大文件。
LOCAL_CACHE_SEARCH_ROOT = PROJECT_DIR.parent
CSV_READ_CHUNK_SIZE = 500_000

MYSQL_EXE = os.environ.get("MYSQL_EXE", "mysql")
MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.environ.get("MYSQL_PORT", "3306"))
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_DATABASE = os.environ.get("MYSQL_DATABASE", "1626_astock")
MARKET_TABLE = os.environ.get("MYSQL_MARKET_TABLE", "all_market_data")

# 回测基础参数：严格按交易日计数，每隔 N 个交易日形成信号，下一交易日开盘买入，
# 从入场日开始持有 N 个交易日，并在第 N 个交易日收盘退出。
# 默认回测区间；仍可通过命令行 --start-date / --end-date 覆盖。
DEFAULT_START_DATE = "2019-09-01"
DEFAULT_END_DATE = "2024-09-01"
Q_TREND_WINDOW = 20
DEFAULT_HOLDING_DAYS = 5
# 因子有效期：分别检验信号对未来 1/2/3/5/10/20 个交易日收益的解释能力。
DEFAULT_VALIDITY_HORIZONS = (1, 2, 3, 5, 10, 20)
DEFAULT_GROUP_NUM = 10
DEFAULT_MIN_FLOAT_MARKET_VALUE = 5_000_000.0
TRADING_DAYS_PER_YEAR = 252
PRICE_COMPARE_TOLERANCE = 1e-6
# 单日可用股票少于此阈值时，不计算 IC/RankIC，避免小样本相关系数失真。
MIN_IC_CROSS_SECTION = 10
# 仅保留过去 20 日累计收益为正的股票，以筛选“稳步上涨”而非“稳步下跌”的路径。
DEFAULT_MIN_RETURN20 = 0.0
# G1 组合的二次筛选范围：先在完整股票池中形成 G1，再仅保留该 G1 内
# Return20 位于此闭区间的股票。默认研究过去 20 日涨幅为 7%~10% 的股票。
DEFAULT_G1_RETURN20_FILTER_MIN = 0.07
DEFAULT_G1_RETURN20_FILTER_MAX = 0.10
# 主组合 G1 内需要同时回测的 Return20 范围。键用于输出文件和图像命名，
# 值为闭区间下限与上限；后续新增范围时只需扩展此配置。
DEFAULT_G1_RETURN20_RANGES = {
    "7_10": (0.07, 0.10),
    "0_10": (0.00, 0.10),
    "0_20": (0.00, 0.20),
}
# Return20 的左闭右开分箱；最后一个区间覆盖 50% 及以上的涨幅。
RETURN20_BUCKET_BREAKS = [0.0, 0.03, 0.05, 0.07, 0.10, 0.20, 0.30, 0.50, np.inf]
RETURN20_BUCKET_LABELS = [
    "0%~3%",
    "3%~5%",
    "5%~7%",
    "7%~10%",
    "10%~20%",
    "20%~30%",
    "30%~50%",
    ">=50%",
]

# A 股市场范围，以及需从股票池排除的 ST/PT 状态编码。
A_SHARE_MARKET_TYPES = {1, 4, 16, 32, 64}
ST_OR_PT_STATUS_VALUES = {2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16}

# 从 MySQL 行情表读取的最小字段集：仅保留构造、交易过滤和回测必需字段。
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
    "Markettype",
    "Trdsta",
    "LimitDown",
    "LimitUp",
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
    "Markettype": "market_type",
    "Trdsta": "trade_status",
    "LimitDown": "limit_down_price",
    "LimitUp": "limit_up_price",
}

NUMERIC_COLUMNS = [
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "volume",
    "amount",
    "float_market_value",
    "market_type",
    "trade_status",
    "limit_down_price",
    "limit_up_price",
]

# 计算脚本仅落盘供绘图脚本使用的结果表，不输出无关旧因子的中间明细。
CALCULATION_OUTPUTS = {
    "portfolio": "01_qtrend20_portfolio_returns.csv",
    "ic": "02_qtrend20_ic_rankic_series.csv",
    "quantile": "03_qtrend20_quantile_returns.csv",
    "validity": "04_qtrend20_factor_validity.csv",
    "return_bucket_long_only": "05_qtrend20_return_bucket_long_only.csv",
    "return_bucket_summary": "06_qtrend20_return_bucket_summary.csv",
    "g1_return20_filtered": "07_qtrend20_g1_return20_filtered_returns.csv",
    "all_market_return_bucket": "08_qtrend20_all_market_return_bucket_returns.csv",
}


def parse_positive_int(text: str) -> int:
    """解析必须大于 0 的整数参数，避免零天或负持有期进入回测。"""
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须输入整数。") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("必须输入大于 0 的整数。")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "构造 QTrend20，并按指定交易日间隔调仓，完成组合净值、IC/RankIC、"
            "分层回测和因子有效期所需的最小数据计算。"
        )
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="可选。直接读取行情 CSV/TSV；不传时从 MySQL 读取。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="结果目录；省略时自动输出到 output_v2/holding_Nd。",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument(
        "--holding-days",
        type=parse_positive_int,
        default=DEFAULT_HOLDING_DAYS,
        help="组合持有的交易日数量，必须为正整数；默认 20。",
    )
    parser.add_argument(
        "--validity-horizons",
        default=",".join(map(str, DEFAULT_VALIDITY_HORIZONS)),
        help="因子有效期交易日列表，例如 1,2,3,5,10,20。",
    )
    parser.add_argument("--group-num", type=int, default=DEFAULT_GROUP_NUM)
    parser.add_argument(
        "--min-return20",
        type=float,
        default=DEFAULT_MIN_RETURN20,
        help="仅保留 Return20 严格大于此阈值的股票；默认 0，即仅过去 20 日上涨股票。",
    )
    parser.add_argument(
        "--g1-return20-filter-min",
        type=float,
        default=DEFAULT_G1_RETURN20_FILTER_MIN,
        help="G1 二次筛选的 Return20 下限（含）；默认 0.07，即 7%。",
    )
    parser.add_argument(
        "--g1-return20-filter-max",
        type=float,
        default=DEFAULT_G1_RETURN20_FILTER_MAX,
        help="G1 二次筛选的 Return20 上限（含）；默认 0.10，即 10%。",
    )
    parser.add_argument(
        "--min-float-market-value",
        type=float,
        default=DEFAULT_MIN_FLOAT_MARKET_VALUE,
        help="Dsmvosd 原字段单位（千元）；默认 5,000,000，即约 50 亿元。",
    )
    args = parser.parse_args()
    # 不同持有期使用不同子目录，避免 5 日、20 日等结果相互覆盖。
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_DIR / f"holding_{args.holding_days}d"
    if args.g1_return20_filter_min > args.g1_return20_filter_max:
        parser.error("--g1-return20-filter-min 不能大于 --g1-return20-filter-max")
    return args


def parse_horizons(text: str) -> list[int]:
    """将命令行的逗号分隔持有期转成去重、升序的正整数列表。"""
    horizons = sorted({int(item.strip()) for item in text.split(",") if item.strip()})
    if not horizons or any(item <= 0 for item in horizons):
        raise ValueError("validity-horizons 必须是逗号分隔的正整数。")
    return horizons


def write_csv(data: pd.DataFrame, path: Path) -> None:
    """统一以 UTF-8-SIG 保存，确保 Excel 直接打开中文字段不乱码。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")


def quote_identifier(identifier: str) -> str:
    return f"`{identifier.replace('`', '``')}`"


def build_market_sql(query_start_date: str, end_date: str) -> str:
    """构造只按日期筛选的行情 SQL；股票池筛选统一在 pandas 中完成。"""
    columns = ", ".join(quote_identifier(column) for column in SOURCE_COLUMNS)
    return (
        f"SELECT {columns} FROM {quote_identifier(MARKET_TABLE)} "
        f"WHERE {quote_identifier('Trddt')} BETWEEN "
        f"'{query_start_date}' AND '{end_date}' ORDER BY "
        f"{quote_identifier('Stkcd')}, {quote_identifier('Trddt')};"
    )


def _get_date_column(columns: list[str]) -> str | None:
    """识别原始行情或已标准化行情中的日期字段名。"""

    if "Trddt" in columns:
        return "Trddt"
    if "trade_date" in columns:
        return "trade_date"
    return None


def inspect_local_market_file(path: Path) -> tuple[list[str], str]:
    """只读取表头检查本地缓存，避免为判断缓存可用性而加载整张行情表。"""

    if not path.exists() or not path.is_file():
        raise ValueError("文件不存在")
    if path.stat().st_size == 0:
        raise ValueError("文件为空")
    separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    try:
        columns = pd.read_csv(path, sep=separator, nrows=0, encoding="utf-8-sig").columns.tolist()
    except pd.errors.EmptyDataError as exc:
        raise ValueError("文件没有表头或数据") from exc
    date_column = _get_date_column(columns)
    required = set(SOURCE_COLUMNS)
    standardized_required = set(COLUMN_RENAME_MAP.values())
    if not date_column or not (required.issubset(columns) or standardized_required.issubset(columns)):
        raise ValueError("缺少 QTrend20 所需的行情字段")
    return columns, separator


def parse_cache_date_coverage(path: Path) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """从规范缓存文件名中提取起止日期，避免读取大文件日期列来判断覆盖范围。"""

    match = re.search(r"(\d{4}[-]?\d{2}[-]?\d{2})[_-](\d{4}[-]?\d{2}[-]?\d{2})", path.stem)
    if not match:
        return None
    start = pd.to_datetime(match.group(1).replace("-", ""), format="%Y%m%d", errors="coerce")
    end = pd.to_datetime(match.group(2).replace("-", ""), format="%Y%m%d", errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return None
    return start, end


def local_file_covers_period(path: Path, query_start: str, end_date: str) -> bool:
    """确认文件名声明的日期范围覆盖本次查询；无日期范围的文件不自动复用。"""

    coverage = parse_cache_date_coverage(path)
    if coverage is None:
        return False
    file_start, file_end = coverage
    return file_start <= pd.Timestamp(query_start) and file_end >= pd.Timestamp(end_date)


def find_local_market_file(query_start: str, end_date: str, cache_dir: Path) -> Path | None:
    """按优先级查找可复用的本地原始行情缓存，并跳过空文件/字段不完整文件。"""

    exact_cache = cache_dir / f"market_{query_start}_{end_date}.tsv"
    candidates = [exact_cache] if exact_cache.exists() else []
    if cache_dir.exists():
        candidates.extend(sorted(cache_dir.glob("market_*.tsv"), key=lambda item: item.stat().st_mtime, reverse=True))

    # 兼容项目中已有的 all_market_data_raw_sql_YYYYMMDD_YYYYMMDD.csv 原始行情缓存。
    if LOCAL_CACHE_SEARCH_ROOT.exists():
        for cache_folder in LOCAL_CACHE_SEARCH_ROOT.rglob("_local_data_cache"):
            candidates.extend(cache_folder.glob("all_market_data_raw_sql_*.csv"))

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not local_file_covers_period(candidate, query_start, end_date):
            continue
        seen.add(candidate)
        try:
            inspect_local_market_file(candidate)
        except (OSError, ValueError) as exc:
            print(f"忽略无效本地行情缓存：{candidate}（{exc}）")
            continue
        return candidate
    return None


def read_delimited_file(
    path: Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """只读取必要列，并可按日期分块过滤 CSV/TSV，以降低大缓存的内存占用。"""

    columns, separator = inspect_local_market_file(path)
    # 只读本脚本必需字段，避免把缓存中不参与回测的列一并载入内存。
    use_columns = [
        column
        for column in SOURCE_COLUMNS + list(COLUMN_RENAME_MAP.values())
        if column in columns
    ]
    use_columns = list(dict.fromkeys(use_columns))
    date_column = _get_date_column(columns)
    read_kwargs = {
        "sep": separator,
        "usecols": use_columns,
        "dtype": {"Stkcd": "string", "stock_code": "string"},
        "encoding": "utf-8-sig",
    }
    if start_date is None or end_date is None:
        return pd.read_csv(path, **read_kwargs)

    # 分块读取并立即按日期过滤：本地缓存范围更大时，可显著降低最终驻留内存。
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=CSV_READ_CHUNK_SIZE, **read_kwargs):
        chunk_dates = pd.to_datetime(chunk[date_column], errors="coerce")
        filtered = chunk.loc[chunk_dates.between(pd.Timestamp(start_date), pd.Timestamp(end_date))]
        if not filtered.empty:
            chunks.append(filtered)
    if not chunks:
        raise ValueError(f"本地行情文件在 {start_date} 至 {end_date} 内没有数据：{path}")
    return pd.concat(chunks, ignore_index=True)


def export_mysql_data(
    sql: str,
    cache_path: Path,
    query_start: str,
    end_date: str,
) -> pd.DataFrame:
    """从 MySQL 导出行情到本地 TSV 缓存；密码仅从环境变量读取。"""
    # 与项目现有动量因子脚本保持一致：环境变量优先；未设置时使用本地数据库默认密码。
    # 如用户已修改 MySQL 密码，应通过 MYSQL_PASSWORD 覆盖此默认值。
    password = os.environ.get("MYSQL_PASSWORD", "Shx20220717")
    if not password:
        raise RuntimeError(
            "未设置 MYSQL_PASSWORD。请先设置环境变量，或使用 --input-csv 指定本地行情文件。"
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MYSQL_PWD"] = password
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
    # 行情表可能很大，直接流式写入缓存，避免把完整 TSV 同时留在内存中。
    with cache_path.open("wb") as output_file:
        result = subprocess.run(
            command,
            stdout=output_file,
            stderr=subprocess.PIPE,
            env=env,
            check=False,
        )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"MySQL 导出失败：{message}")
    return read_delimited_file(cache_path, start_date=query_start, end_date=end_date)


def load_market_data(
    input_csv: Path | None,
    start_date: str,
    end_date: str,
    cache_dir: Path,
    refresh_cache: bool,
) -> pd.DataFrame:
    """优先读取用户 CSV；否则复用缓存或从 MySQL 查询，并补足因子回看期。"""
    # 为研究起点预留足够的 20 个交易日回看数据。
    query_start = (pd.Timestamp(start_date) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    if input_csv is not None:
        return read_delimited_file(input_csv, start_date=query_start, end_date=end_date)

    cache_path = cache_dir / f"market_{query_start}_{end_date}.tsv"
    if not refresh_cache:
        local_file = find_local_market_file(query_start, end_date, cache_dir)
        if local_file is not None:
            print(f"复用本地行情缓存，不读取 MySQL：{local_file}")
            return read_delimited_file(local_file, start_date=query_start, end_date=end_date)
    print("未找到可用本地行情缓存，正在从 MySQL 读取并写入缓存...")
    return export_mysql_data(
        build_market_sql(query_start, end_date),
        cache_path,
        query_start=query_start,
        end_date=end_date,
    )


def standardize_and_clean_market_data(
    raw_data: pd.DataFrame,
    min_float_market_value: float,
) -> pd.DataFrame:
    """统一字段/类型后构建可研究股票池，且绝不对价格做前值填充。"""
    data = raw_data.rename(columns=COLUMN_RENAME_MAP).copy()
    missing = [column for column in COLUMN_RENAME_MAP.values() if column not in data.columns]
    if missing:
        raise ValueError(f"行情数据缺少必要字段：{missing}")

    # 股票代码、日期和数值字段统一格式，防止后续合并时因 dtype 不一致丢失记录。
    data["stock_code"] = data["stock_code"].astype("string").str.strip().str.zfill(6)
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # 同一股票同一天若存在重复记录，保留最后一条；缺失主键的记录无法用于时序回测。
    data = (
        data.dropna(subset=["stock_code", "trade_date"])
        .sort_values(["stock_code", "trade_date"])
        .drop_duplicates(["stock_code", "trade_date"], keep="last")
    )
    # 基础股票池：A 股、满足流通市值下限、非 ST/PT、非停牌。
    selected = (
        data["market_type"].isin(A_SHARE_MARKET_TYPES)
        & data["float_market_value"].ge(min_float_market_value)
        & ~data["trade_status"].isin(ST_OR_PT_STATUS_VALUES)
        & ~(data["volume"].notna() & data["volume"].le(0))
        & ~(data["amount"].notna() & data["amount"].le(0))
    )
    # 价格不做前值填充；QTrend20 明确要求窗口内 20 个收盘价均为有效正数。
    return data.loc[selected].sort_values(["stock_code", "trade_date"]).reset_index(drop=True)


def _qtrend20_for_one_stock(close_price: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """计算纯路径偏离 QTrend 与对应的过去 20 日收益 Return20。"""
    size = len(close_price)
    qtrend_output = np.full(size, np.nan, dtype="float64")
    return20_output = np.full(size, np.nan, dtype="float64")
    if size < Q_TREND_WINDOW:
        return qtrend_output, return20_output

    # 非正收盘价视为无效；本版本直接以原始收盘价（而非对数价格）构造首尾连线。
    valid_close = np.where(close_price > 0, close_price, np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(valid_close, Q_TREND_WINDOW)
    valid = np.isfinite(windows).all(axis=1)

    # 每个窗口的首尾收盘价决定一条直线；第 i 天的理论价格在线上按等间隔插值。
    time_weight = np.arange(Q_TREND_WINDOW, dtype="float64") / (Q_TREND_WINDOW - 1)
    line_price = windows[:, [0]] + (windows[:, [-1]] - windows[:, [0]]) * time_weight
    # 用每日相对偏离消除股价绝对水平的影响，再对 20 日偏离平方取均值。
    relative_deviation = np.divide(
        windows - line_price,
        line_price,
        out=np.full_like(windows, np.nan),
        where=line_price > 0,
    )
    # QTrend20 只保留相对偏离平方均值：值越小，价格路径越贴近首尾连线。
    qtrend = np.square(relative_deviation).mean(axis=1)
    # Return20 仅用于筛选上涨股票和区间分类，不参与 QTrend20 的乘法。
    window_return = windows[:, -1] / windows[:, 0] - 1.0
    qtrend[~valid] = np.nan
    window_return[~valid] = np.nan
    start = Q_TREND_WINDOW - 1
    # 两项窗口指标均与窗口最后一个交易日对齐；前 19 日因窗口不足保留缺失值。
    qtrend_output[start:] = qtrend
    return20_output[start:] = window_return
    return qtrend_output, return20_output


def calculate_qtrend20(clean_data: pd.DataFrame, min_return20: float) -> pd.DataFrame:
    """构造 QTrend，并在每个截面仅保留 Return20 为正的股票后标准化。"""
    pieces: list[pd.DataFrame] = []
    # 因子是个股时序量，必须在每只股票独立、按日期升序的序列上滚动计算。
    for _, stock_data in clean_data.groupby("stock_code", sort=False):
        stock = stock_data.sort_values("trade_date").copy()
        close = stock["close_price"].to_numpy(dtype="float64")
        # 每只股票独立使用滚动 20 日窗口，避免不同股票之间的价格路径相互干扰。
        qtrend20_raw, return20 = _qtrend20_for_one_stock(close)
        stock["qtrend20_raw"] = qtrend20_raw
        stock["return20"] = return20
        stock["qtrend20_start_date"] = stock["trade_date"].shift(Q_TREND_WINDOW - 1)
        pieces.append(stock)

    factor = pd.concat(pieces, ignore_index=True)
    factor = factor.dropna(subset=["qtrend20_raw", "return20"]).copy()
    # 截面过滤：仅保留过去 20 日收益严格高于阈值的股票，默认即 Return20 > 0。
    factor = factor.loc[factor["return20"].gt(min_return20)].copy()
    factor["return20_bucket"] = pd.cut(
        factor["return20"],
        bins=RETURN20_BUCKET_BREAKS,
        labels=RETURN20_BUCKET_LABELS,
        right=False,
        include_lowest=True,
    )

    # 截面 3σ 缩尾：保留股票记录，只压缩极端因子值。
    stats = (
        factor.groupby("trade_date")["qtrend20_raw"]
        .agg(cs_mean="mean", cs_std=lambda values: values.std(ddof=0))
        .reset_index()
    )
    factor = factor.merge(stats, on="trade_date", how="left")
    lower = factor["cs_mean"] - 3.0 * factor["cs_std"]
    upper = factor["cs_mean"] + 3.0 * factor["cs_std"]
    factor["qtrend20_3sigma"] = factor["qtrend20_raw"].clip(lower=lower, upper=upper)

    # 截面 Z-score：将每天不同股票池的因子映射到可直接排序的统一尺度。
    zstats = (
        factor.groupby("trade_date")["qtrend20_3sigma"]
        .agg(z_mean="mean", z_std=lambda values: values.std(ddof=0))
        .reset_index()
    )
    factor = factor.merge(zstats, on="trade_date", how="left")
    factor["qtrend20_zscore"] = np.where(
        factor["z_std"].gt(0),
        (factor["qtrend20_3sigma"] - factor["z_mean"]) / factor["z_std"],
        np.nan,
    )
    return factor.dropna(subset=["qtrend20_zscore"]).reset_index(drop=True)


def keep_periodic_trading_day_signals(
    factor: pd.DataFrame,
    clean_data: pd.DataFrame,
    start_date: str,
    end_date: str,
    holding_days: int,
) -> pd.DataFrame:
    """从回测首日起仅按交易日计数，保留第 N、2N、3N…个交易日的信号。"""
    calendar = pd.DataFrame(
        {"trade_date": sorted(pd.to_datetime(clean_data["trade_date"].dropna().unique()))}
    )
    calendar = calendar.loc[
        calendar["trade_date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    ].reset_index(drop=True)
    # 不使用自然周分组：第 N 个交易日首次形成信号，之后严格每隔 N 个交易日形成信号。
    signal_dates = calendar.loc[
        (np.arange(len(calendar)) + 1) % holding_days == 0,
        ["trade_date"],
    ]
    return factor.merge(signal_dates, on="trade_date", how="inner")


def build_trade_calendar(
    clean_data: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    """基于全市场交易日历映射信号日的 T+1 入场日及各持有期收盘日。"""
    calendar = pd.DataFrame(
        {"trade_date": sorted(pd.to_datetime(clean_data["trade_date"].dropna().unique()))}
    )
    calendar["entry_date"] = calendar["trade_date"].shift(-1)
    # 只生成实际请求的持有期日期列；即使输入较大的天数，也不创建无关中间列。
    for horizon in sorted(set(horizons)):
        calendar[f"exit_date_{horizon}"] = calendar["trade_date"].shift(-horizon)
    return calendar


def calculate_forward_returns(
    factor: pd.DataFrame,
    clean_data: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    """计算 T+1 开盘至 T+h 收盘的前瞻收益，并过滤 T+1 开盘涨跌停。"""
    calendar = build_trade_calendar(clean_data, horizons)
    output = factor.merge(calendar, on="trade_date", how="left")

    # 入场价格只使用 T+1 开盘时已知字段；开盘价触及涨/跌停时视为无法成交。
    entry = clean_data[
        ["stock_code", "trade_date", "open_price", "limit_up_price", "limit_down_price"]
    ].rename(
        columns={
            "trade_date": "entry_date",
            "open_price": "entry_open_price",
            "limit_up_price": "entry_limit_up_price",
            "limit_down_price": "entry_limit_down_price",
        }
    )
    output = output.merge(entry, on=["stock_code", "entry_date"], how="left")
    output["entry_open_limit_up"] = (
        output["entry_open_price"] - output["entry_limit_up_price"]
    ).abs().le(PRICE_COMPARE_TOLERANCE)
    output["entry_open_limit_down"] = (
        output["entry_open_price"] - output["entry_limit_down_price"]
    ).abs().le(PRICE_COMPARE_TOLERANCE)
    output["is_entry_tradable"] = (
        output["entry_open_price"].gt(0)
        & ~output["entry_open_limit_up"].fillna(False)
        & ~output["entry_open_limit_down"].fillna(False)
    )

    # 对每个有效期 h，收益严格按 C_(t+h) / O_(t+1) - 1 计算。
    exit_prices = clean_data[["stock_code", "trade_date", "close_price"]]
    for horizon in horizons:
        exit_column = f"exit_date_{horizon}"
        lookup = exit_prices.rename(
            columns={"trade_date": exit_column, "close_price": f"exit_close_{horizon}"}
        )
        output = output.merge(lookup, on=["stock_code", exit_column], how="left")
        raw_return = output[f"exit_close_{horizon}"] / output["entry_open_price"] - 1.0
        output[f"forward_return_{horizon}d"] = raw_return.where(
            output["is_entry_tradable"] & output[f"exit_close_{horizon}"].gt(0)
        )
    return output


def assign_quantile_groups(factor_returns: pd.DataFrame, group_num: int) -> pd.DataFrame:
    """按每个调仓信号截面的 Z-score 从低到高排序，并划分为等数量分组（G1 为最低组）。"""
    data = factor_returns.sort_values(
        ["trade_date", "qtrend20_zscore", "stock_code"],
        ascending=[True, True, True],
    ).copy()
    # 股票代码作为同分时稳定排序的隐含次序，保证分组可复现。
    data["rank_asc"] = data.groupby("trade_date").cumcount() + 1
    data["cross_section_count"] = data.groupby("trade_date")["stock_code"].transform("count")
    data = data.loc[data["cross_section_count"].ge(group_num)].copy()
    data["quantile_group"] = (
        ((data["rank_asc"] - 1) * group_num) // data["cross_section_count"] + 1
    ).clip(1, group_num).astype("int64")
    return data


def calculate_quantile_returns(
    grouped: pd.DataFrame,
    holding_days: int,
    group_num: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算分层收益，并构造只做多最低因子组 G1 的主组合。"""
    return_column = f"forward_return_{holding_days}d"
    # 无法在 T+1 开盘成交或缺少期末价格的股票已在 forward return 中置空，此处自动排除。
    quantile = (
        grouped.dropna(subset=[return_column])
        .groupby(["trade_date", "quantile_group"])
        .agg(
            group_return=(return_column, "mean"),
            stock_count=("stock_code", "count"),
            factor_mean=("qtrend20_zscore", "mean"),
        )
        .reset_index()
        .sort_values(["trade_date", "quantile_group"])
    )
    quantile["holding_days"] = holding_days
    rebalance_frequency = f"every_{holding_days}_trading_days"
    quantile["rebalance_frequency"] = rebalance_frequency

    # 主组合保留 G1 多头；同时从同一张全截面分层表提取 G10，构造原始 G1-G10 多空组合。
    g1 = quantile.loc[quantile["quantile_group"].eq(1)].copy()
    g1 = g1.rename(
        columns={
            "group_return": "long_g1_return",
            "stock_count": "long_g1_stock_count",
            "factor_mean": "long_g1_factor_mean",
        }
    )
    g10 = quantile.loc[quantile["quantile_group"].eq(group_num)].copy()
    g10 = g10.rename(
        columns={
            "group_return": "long_g10_return",
            "stock_count": "long_g10_stock_count",
            "factor_mean": "long_g10_factor_mean",
        }
    )
    g10 = g10[
        [
            "trade_date",
            "long_g10_return",
            "long_g10_stock_count",
            "long_g10_factor_mean",
        ]
    ]
    portfolio = g1.merge(g10, on="trade_date", how="inner")
    portfolio = portfolio.sort_values("trade_date").reset_index(drop=True)
    portfolio["long_g1_nav"] = (1.0 + portfolio["long_g1_return"]).cumprod()
    # 多空组合按用户指定的 G1-G10 原始价差收益计算：G1 做多、G10 做空。
    portfolio["long_short_return"] = (
        portfolio["long_g1_return"] - portfolio["long_g10_return"]
    )
    portfolio["long_short_nav"] = (1.0 + portfolio["long_short_return"]).cumprod()
    portfolio["holding_days"] = holding_days
    portfolio["rebalance_frequency"] = rebalance_frequency
    portfolio["periods_per_year"] = TRADING_DAYS_PER_YEAR / holding_days
    return quantile, portfolio


def calculate_all_a_equal_weight_benchmark(
    clean_data: pd.DataFrame,
    signal_dates: pd.Series,
    holding_days: int,
) -> pd.DataFrame:
    """按主组合相同的调仓日、入场与出场规则计算全 A 股可交易股票等权基准。"""
    return_column = f"forward_return_{holding_days}d"
    # 基准不施加 QTrend 或 Return20 条件：在每个主组合信号日，使用基础 A 股股票池中
    # 所有可交易股票等权持有。calculate_forward_returns 会沿用 T+1 开盘买入、持有期末收盘卖出
    # 以及开盘涨跌停不可成交的统一处理，确保与主组合收益定义一致。
    signal_calendar = pd.DataFrame({"trade_date": pd.to_datetime(signal_dates).unique()})
    market_snapshot = clean_data.merge(signal_calendar, on="trade_date", how="inner")
    market_forward = calculate_forward_returns(
        market_snapshot,
        clean_data=clean_data,
        horizons=[holding_days],
    )
    benchmark = (
        market_forward.dropna(subset=[return_column])
        .groupby("trade_date", sort=True)
        .agg(
            benchmark_return=(return_column, "mean"),
            benchmark_stock_count=("stock_code", "count"),
        )
        .reset_index()
    )
    # 与主组合调仓日历对齐；理论上每期均有全 A 股可交易成分，若异常缺失则保留 NaN 供校验发现。
    benchmark["benchmark_nav"] = (1.0 + benchmark["benchmark_return"]).cumprod()
    benchmark["benchmark_name"] = "全A市场等权基准"
    return benchmark


def calculate_g1_return20_filtered_portfolio(
    grouped: pd.DataFrame,
    holding_days: int,
    return20_min: float,
    return20_max: float,
) -> pd.DataFrame:
    """从原始 G1 中筛选指定 Return20 区间，不重新进行分组排序。"""
    return_column = f"forward_return_{holding_days}d"
    # 关键顺序：grouped 已经基于完整 Return20>0 的截面按 QTrend 升序完成 G1-G10 分组。
    # 此处只在既有 G1 内做 Return20 范围过滤，因此严格等价于从当前 G1 组合中剔除
    # Return20 不在指定范围内的股票，而不是在剩余股票中重新定义 G1。
    filtered_g1 = grouped.loc[
        grouped["quantile_group"].eq(1)
        & grouped["return20"].ge(return20_min)
        & grouped["return20"].le(return20_max)
    ].dropna(subset=[return_column]).copy()

    portfolio = (
        filtered_g1.groupby("trade_date", sort=True)
        .agg(
            long_g1_return=(return_column, "mean"),
            long_g1_stock_count=("stock_code", "count"),
            long_g1_factor_mean=("qtrend20_zscore", "mean"),
            long_g1_return20_mean=("return20", "mean"),
        )
        .reset_index()
    )
    # 使用原始主组合 G1 的完整调仓日历。若某个调仓期没有股票落入指定 Return20
    # 区间，则该期保持空仓现金，组合收益记为 0，避免删除日期后高估年化收益和 Sharpe。
    all_g1_dates = (
        grouped.loc[grouped["quantile_group"].eq(1)]
        .dropna(subset=[return_column])[["trade_date"]]
        .drop_duplicates()
        .sort_values("trade_date")
    )
    portfolio = all_g1_dates.merge(portfolio, on="trade_date", how="left")
    portfolio["long_g1_return"] = portfolio["long_g1_return"].fillna(0.0)
    portfolio["long_g1_stock_count"] = (
        portfolio["long_g1_stock_count"].fillna(0).astype("int64")
    )

    # 有股票时按筛选后的 G1 成分股等权持有；无股票时净值保持不变。
    portfolio["long_g1_nav"] = (1.0 + portfolio["long_g1_return"]).cumprod()
    portfolio["holding_days"] = holding_days
    portfolio["periods_per_year"] = TRADING_DAYS_PER_YEAR / holding_days
    portfolio["rebalance_frequency"] = f"every_{holding_days}_trading_days"
    portfolio["return20_filter_min"] = return20_min
    portfolio["return20_filter_max"] = return20_max
    portfolio["selection_rule"] = "full_universe_g1_then_return20_filter"
    return portfolio


def calculate_g1_return20_range_portfolios(
    grouped: pd.DataFrame,
    holding_days: int,
    return20_ranges: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """按配置批量计算主组合 G1 的多个 Return20 子区间组合。"""
    pieces: list[pd.DataFrame] = []
    for range_key, (return20_min, return20_max) in return20_ranges.items():
        one_range = calculate_g1_return20_filtered_portfolio(
            grouped,
            holding_days=holding_days,
            return20_min=return20_min,
            return20_max=return20_max,
        )
        one_range.insert(1, "range_key", range_key)
        one_range.insert(2, "range_label", f"{return20_min:.0%}~{return20_max:.0%}")
        pieces.append(one_range)
    if not pieces:
        raise ValueError("至少需要配置一个 G1 Return20 区间。")
    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["range_key", "trade_date"])
        .reset_index(drop=True)
    )


def calculate_return_bucket_long_only(
    grouped: pd.DataFrame,
    holding_days: int,
) -> pd.DataFrame:
    """先确定全截面主组合 G1，再按 Return20 区间拆分该 G1 的股票。"""
    return_column = f"forward_return_{holding_days}d"
    bucket_keys = ["trade_date", "return20_bucket"]
    # grouped 已经在全部 Return20>0 股票中按 QTrend 升序完成 G1-G10 分组。
    # 这里先锁定主组合 G1，再按 Return20 区间区分其成分股；区间内不重新排名、
    # 不重新选择 G1，从而保证所有区间都是主组合 G1 的互斥子集。
    main_g1 = grouped.loc[
        grouped["quantile_group"].eq(1)
        & grouped["return20_bucket"].notna()
    ].copy()

    # 每个调仓日、每个 Return20 区间内对主组合 G1 成分股等权计算未来持有期收益。
    long_only = (
        main_g1.dropna(subset=[return_column])
        .groupby(bucket_keys, observed=True)
        .agg(
            long_only_return=(return_column, "mean"),
            stock_count=("stock_code", "count"),
            qtrend_mean=("qtrend20_raw", "mean"),
            return20_mean=("return20", "mean"),
        )
        .reset_index()
        .sort_values(bucket_keys)
    )
    long_only["long_only_nav"] = long_only.groupby("return20_bucket", observed=True)[
        "long_only_return"
    ].transform(lambda values: (1.0 + values).cumprod())
    long_only["holding_days"] = holding_days
    long_only["position_side"] = "main_g1_return20_subgroup"
    long_only["rebalance_frequency"] = f"every_{holding_days}_trading_days"
    long_only["selection_rule"] = "full_universe_g1_then_return20_bucket"
    return long_only


def calculate_all_market_return_bucket_returns(
    grouped: pd.DataFrame,
    holding_days: int,
) -> pd.DataFrame:
    """在每个调仓截面对全部合格股票按 Return20 分桶，并计算各桶等权未来收益。"""
    return_column = f"forward_return_{holding_days}d"
    # grouped 已是当期可交易、且 Return20>0 的全市场股票池；这里不使用 QTrend 分组，
    # 仅按过去 20 日涨幅进行区间划分。因此该结果用于回答“全市场不同涨幅段”的后续收益，
    # 不能与“先选 G1 再分桶”的主组合诊断图混为同一口径。
    eligible = grouped.loc[
        grouped["return20_bucket"].notna()
    ].dropna(subset=[return_column]).copy()
    bucket_columns = ["trade_date", "return20_bucket"]
    bucket_returns = (
        eligible.groupby(bucket_columns, observed=True)
        .agg(
            long_only_return=(return_column, "mean"),
            stock_count=("stock_code", "count"),
            return20_mean=("return20", "mean"),
        )
        .reset_index()
    )

    # 以所有调仓日和所有 Return20 区间构造完整网格：某桶当期没有股票时按空仓处理，
    # 收益记为 0，净值保持不变。这样高涨幅桶的时间轴完整，不会把稀疏观测跨期连成误导性直线。
    all_dates = eligible[["trade_date"]].drop_duplicates().sort_values("trade_date")
    bucket_order = pd.Index(RETURN20_BUCKET_LABELS, name="return20_bucket")
    calendar = pd.MultiIndex.from_product(
        [all_dates["trade_date"].tolist(), bucket_order.tolist()],
        names=bucket_columns,
    ).to_frame(index=False)
    result = calendar.merge(bucket_returns, on=bucket_columns, how="left")
    result["long_only_return"] = result["long_only_return"].fillna(0.0)
    result["stock_count"] = result["stock_count"].fillna(0).astype("int64")
    result["long_only_nav"] = result.groupby("return20_bucket", sort=False)[
        "long_only_return"
    ].transform(lambda values: (1.0 + values).cumprod())
    result["holding_days"] = holding_days
    result["periods_per_year"] = TRADING_DAYS_PER_YEAR / holding_days
    result["position_side"] = "all_market_return20_bucket_equal_weight"
    result["rebalance_frequency"] = f"every_{holding_days}_trading_days"
    result["selection_rule"] = "all_eligible_stocks_then_return20_bucket"
    return result.sort_values(bucket_columns).reset_index(drop=True)


def calculate_return_bucket_summary(return_bucket_long_only: pd.DataFrame) -> pd.DataFrame:
    """汇总各 Return20 区间的样本量、G1 多头收益与复利表现。"""
    rows: list[dict[str, float | int | str]] = []
    for bucket, data in return_bucket_long_only.groupby("return20_bucket", observed=True):
        returns = data["long_only_return"].dropna()
        nav = (1.0 + returns).cumprod()
        rows.append(
            {
                "return20_bucket": str(bucket),
                "observation_days": int(len(data)),
                "average_stock_count": float(data["stock_count"].mean()),
                "average_return20": float(data["return20_mean"].mean()),
                "average_qtrend": float(data["qtrend_mean"].mean()),
                "average_forward_return": float(returns.mean()),
                "cumulative_return": float(nav.iloc[-1] - 1.0) if not nav.empty else math.nan,
                "win_rate": float((returns > 0).mean()) if not returns.empty else math.nan,
            }
        )
    return pd.DataFrame(rows)


def _daily_correlation(one_day: pd.DataFrame, return_column: str) -> pd.Series:
    """计算单一调仓信号截面的 Pearson IC 与 Spearman RankIC。"""
    valid = one_day[["qtrend20_zscore", return_column]].dropna()
    # 相关系数的横截面样本不足时留空，后续不会被均值或累计值使用。
    if len(valid) < MIN_IC_CROSS_SECTION:
        return pd.Series({"ic": np.nan, "rank_ic": np.nan, "stock_count": len(valid)})
    ic = valid["qtrend20_zscore"].corr(valid[return_column], method="pearson")
    # Spearman 等价于秩变量的 Pearson 相关；显式排名可避免把 scipy 变成必需依赖。
    factor_rank = valid["qtrend20_zscore"].rank(method="average")
    return_rank = valid[return_column].rank(method="average")
    rank_ic = factor_rank.corr(return_rank, method="pearson")
    return pd.Series({"ic": ic, "rank_ic": rank_ic, "stock_count": len(valid)})


def calculate_ic_series(factor_returns: pd.DataFrame, holding_days: int) -> pd.DataFrame:
    """按调仓信号日汇总 IC/RankIC，并生成两条累计 IC 曲线。"""
    return_column = f"forward_return_{holding_days}d"
    ic = (
        factor_returns.groupby("trade_date", sort=True)
        .apply(_daily_correlation, return_column=return_column, include_groups=False)
        .reset_index()
        .dropna(subset=["ic", "rank_ic"])
    )
    # 累计 IC 为日度 IC 的逐日求和，不是复利收益。
    ic["cumulative_ic"] = ic["ic"].cumsum()
    ic["cumulative_rank_ic"] = ic["rank_ic"].cumsum()
    ic["holding_days"] = holding_days
    ic["rebalance_frequency"] = f"every_{holding_days}_trading_days"
    ic["periods_per_year"] = TRADING_DAYS_PER_YEAR / holding_days
    return ic


def _mean_test(values: pd.Series) -> dict[str, float]:
    """计算均值、标准差、IR、t 值和双侧 p 值，供有效期横向比较。"""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    count = len(clean)
    mean = float(clean.mean()) if count else math.nan
    std = float(clean.std(ddof=1)) if count > 1 else math.nan
    t_value = mean / (std / math.sqrt(count)) if count > 1 and std > 0 else math.nan
    try:
        from scipy import stats

        p_value = float(2.0 * stats.t.sf(abs(t_value), df=count - 1))
    except Exception:
        p_value = float(math.erfc(abs(t_value) / math.sqrt(2.0))) if not math.isnan(t_value) else math.nan
    return {
        "mean": mean,
        "std": std,
        "ir": mean / std if std > 0 else math.nan,
        "t_value": t_value,
        "p_value": p_value,
        "observation_count": count,
    }


def calculate_factor_validity(
    factor_returns: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    """在固定调仓信号截面上逐个检验前瞻期 IC/RankIC，并汇总均值、IR 和显著性。"""
    rows: list[dict[str, float | int]] = []
    # 因子有效期只改变前瞻收益终点，不改变同一日截面的因子暴露。
    for horizon in horizons:
        daily = (
            factor_returns.groupby("trade_date", sort=True)
            .apply(
                _daily_correlation,
                return_column=f"forward_return_{horizon}d",
                include_groups=False,
            )
            .reset_index()
        )
        ic_stats = _mean_test(daily["ic"])
        rank_stats = _mean_test(daily["rank_ic"])
        rows.append(
            {
                "holding_days": horizon,
                "ic_mean": ic_stats["mean"],
                "ic_ir": ic_stats["ir"],
                "ic_t_value": ic_stats["t_value"],
                "ic_p_value": ic_stats["p_value"],
                "rank_ic_mean": rank_stats["mean"],
                "rank_ic_ir": rank_stats["ir"],
                "rank_ic_t_value": rank_stats["t_value"],
                "rank_ic_p_value": rank_stats["p_value"],
                "observation_count": min(
                    int(ic_stats["observation_count"]),
                    int(rank_stats["observation_count"]),
                ),
            }
        )
    return pd.DataFrame(rows)


def run_backtest(
    clean_data: pd.DataFrame,
    start_date: str,
    end_date: str,
    holding_days: int,
    validity_horizons: list[int],
    group_num: int,
    min_return20: float,
    g1_return20_ranges: dict[str, tuple[float, float]],
) -> dict[str, pd.DataFrame]:
    """串联 Return20 过滤、交易日周期分组、多空回测、区间多头、IC 与有效期计算。"""
    # 除命令行校验外再次检查，确保其他 Python 程序直接调用本函数时也不能传入非法持有期。
    if not isinstance(holding_days, (int, np.integer)) or holding_days <= 0:
        raise ValueError("holding_days 必须是大于 0 的整数。")
    # 主回测持有期也纳入收益计算列表，避免和有效期配置重复计算。
    horizons = sorted(set(validity_horizons + [holding_days]))
    factor = calculate_qtrend20(clean_data, min_return20=min_return20)
    factor = factor.loc[
        factor["trade_date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    ].copy()
    # 不划分自然周：第 N、2N、3N…个交易日得到信号，下一交易日开盘建仓。
    factor = keep_periodic_trading_day_signals(
        factor,
        clean_data,
        start_date=start_date,
        end_date=end_date,
        holding_days=holding_days,
    )
    factor_returns = calculate_forward_returns(factor, clean_data, horizons)
    grouped = assign_quantile_groups(factor_returns, group_num)
    quantile, portfolio = calculate_quantile_returns(grouped, holding_days, group_num)
    # 仅以主组合实际存在的调仓日为基准日，避免因因子窗口不足造成基准与组合日历不一致。
    benchmark = calculate_all_a_equal_weight_benchmark(
        clean_data=clean_data,
        signal_dates=portfolio["trade_date"],
        holding_days=holding_days,
    )
    portfolio = portfolio.merge(benchmark, on="trade_date", how="left", validate="one_to_one")
    if portfolio["benchmark_return"].isna().any():
        raise RuntimeError("全A市场等权基准存在缺失调仓期，请检查行情或交易限制字段。")
    g1_return20_filtered = calculate_g1_return20_range_portfolios(
        grouped,
        holding_days=holding_days,
        return20_ranges=g1_return20_ranges,
    )
    return_bucket_long_only = calculate_return_bucket_long_only(
        grouped,
        holding_days=holding_days,
    )
    return_bucket_summary = calculate_return_bucket_summary(return_bucket_long_only)
    all_market_return_bucket = calculate_all_market_return_bucket_returns(
        grouped,
        holding_days=holding_days,
    )
    ic = calculate_ic_series(grouped, holding_days)
    validity = calculate_factor_validity(grouped, validity_horizons)
    return {
        "portfolio": portfolio,
        "g1_return20_filtered": g1_return20_filtered,
        "ic": ic,
        "quantile": quantile,
        "validity": validity,
        "return_bucket_long_only": return_bucket_long_only,
        "return_bucket_summary": return_bucket_summary,
        "all_market_return_bucket": all_market_return_bucket,
    }


def validate_outputs(outputs: dict[str, pd.DataFrame], group_num: int) -> None:
    """在写文件前检查结果非空、两端分组齐全且日期顺序正确。"""
    for name, data in outputs.items():
        if data.empty:
            raise RuntimeError(f"{name} 结果为空，请检查日期、股票池或行情字段。")
    groups = set(outputs["quantile"]["quantile_group"].astype(int).unique())
    if not {1, group_num}.issubset(groups):
        raise RuntimeError("分层回测未形成 G1（最小因子组）和 G10（最大因子组）。")
    if not outputs["portfolio"]["trade_date"].is_monotonic_increasing:
        raise RuntimeError("组合收益日期未按升序排列。")
    benchmark_required = {"benchmark_return", "benchmark_nav", "benchmark_stock_count"}
    if not benchmark_required.issubset(outputs["portfolio"].columns):
        raise RuntimeError("主组合结果缺少全A市场等权基准字段。")
    if outputs["portfolio"]["benchmark_return"].isna().any():
        raise RuntimeError("全A市场等权基准收益存在缺失值。")


def main() -> None:
    """命令行入口：读取数据→清洗→回测→校验→写出主回测和区间分析 CSV。"""
    args = parse_args()
    validity_horizons = parse_horizons(args.validity_horizons)
    # 1. 获取原始行情（本地文件、缓存或 MySQL）。
    raw_data = load_market_data(
        input_csv=args.input_csv,
        start_date=args.start_date,
        end_date=args.end_date,
        cache_dir=args.cache_dir,
        refresh_cache=args.refresh_cache,
    )
    # 2. 建立研究股票池并统一数据格式。
    clean_data = standardize_and_clean_market_data(
        raw_data,
        min_float_market_value=args.min_float_market_value,
    )
    # 3. 完成因子、收益、分组、IC 和有效期计算。
    g1_return20_ranges = dict(DEFAULT_G1_RETURN20_RANGES)
    # 保留原有命令行参数作为 7%~10% 主筛选区间的快捷覆盖方式。
    g1_return20_ranges["7_10"] = (
        args.g1_return20_filter_min,
        args.g1_return20_filter_max,
    )
    outputs = run_backtest(
        clean_data=clean_data,
        start_date=args.start_date,
        end_date=args.end_date,
        holding_days=args.holding_days,
        validity_horizons=validity_horizons,
        group_num=args.group_num,
        min_return20=args.min_return20,
        g1_return20_ranges=g1_return20_ranges,
    )
    # 4. 结果校验通过后，才写出供诊断脚本使用的文件。
    validate_outputs(outputs, args.group_num)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, filename in CALCULATION_OUTPUTS.items():
        write_csv(outputs[name], args.output_dir / filename)

    print("QTrend20 v3 G1 多头交易日周期回测计算完成。")
    print(f"输出目录：{args.output_dir.resolve()}")
    print(f"组合持有期：{args.holding_days} 个交易日；分组数：{args.group_num}")
    print(f"Return20 截面过滤：严格大于 {args.min_return20:.2%}")
    print(f"G1 Return20 区间组合：{g1_return20_ranges}")
    print(f"因子有效期：{validity_horizons}")


if __name__ == "__main__":
    main()
