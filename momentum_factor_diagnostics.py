from __future__ import annotations

import argparse
import ast
import math
import re
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 1. 基础配置：读取前序脚本输出，生成因子检测结果
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
MAIN_SCRIPT_PATH = PROJECT_DIR / "calculate_momentum_factor_sql.py"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "output"
TRADING_DAYS_PER_YEAR = 252
MONTHS_PER_YEAR = 12
MAIN_DEFAULTS_FALLBACK = {
    "DEFAULT_REBALANCE_FREQUENCY": "daily",
    "DEFAULT_LOOKBACK_DAYS": 50,
    "DEFAULT_HOLDING_DAYS": 5,
    "DEFAULT_LOOKBACK_MONTHS": 3,
    "DEFAULT_HOLDING_MONTHS": 1,
    "DEFAULT_S": 0,
    "DEFAULT_GROUP_NUM": 10,
}
LEGACY_EXPERIMENT_DIR_RE = re.compile(
    r"^lb(?P<lookback>\d+)_hd(?P<holding>\d+)_g(?P<group>\d+)_mkt(?P<market>.+)$"
)
DAILY_EXPERIMENT_DIR_RE = re.compile(
    r"^daily_lb(?P<lookback>\d+)_hd(?P<holding>\d+)_g(?P<group>\d+)_mkt(?P<market>.+)$"
)
MONTHLY_EXPERIMENT_DIR_RE = re.compile(
    r"^monthly_lb(?P<lookback_months>\d+)m_s(?P<s>\d+)_hd(?P<holding_months>\d+)m_g(?P<group>\d+)_mkt(?P<market>.+)$"
)
RETURN_COLUMN_TAGS = {
    "return_without_dividend": "retnd",
    "return_with_dividend": "retwd",
    "change_ratio": "chg",
}


# ============================================================
# 2. 参数与通用工具
# ============================================================


def parse_args() -> argparse.Namespace:
    """解析命令行参数，默认读取上一阶段生成的动量分组结果。"""

    parser = argparse.ArgumentParser(
        description="基于参数化动量因子结果，计算排序检验、IC/IR、分组收益曲线和多空 NAV 曲线。",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "上一阶段因子分组结果目录。若传入 output 根目录，脚本会自动读取其中最新的 "
            "lb{lookback}_hd{holding}_g{group}_mkt{market_types} 实验文件夹。"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="因子检测结果输出目录。不填时直接写入 input-dir 对应的实验文件夹，使 CSV 和 PNG 放在同一目录。",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        help="动量因子回看交易日数；默认从 input-dir/run_summary.csv 或因子文件自动识别。",
    )
    parser.add_argument(
        "--rebalance-frequency",
        choices=["auto", "daily", "monthly"],
        default="auto",
        help="调仓频率；auto 时跟随 calculate_momentum_factor_sql.py 的 DEFAULT_REBALANCE_FREQUENCY。",
    )
    parser.add_argument(
        "--lookback-months",
        type=int,
        default=None,
        help="月频回看月数；默认跟随主脚本或 input-dir/run_summary.csv。",
    )
    parser.add_argument(
        "--holding-months",
        type=int,
        default=None,
        help="月频持有月数；默认跟随主脚本或 input-dir/run_summary.csv。",
    )
    parser.add_argument(
        "--s",
        type=int,
        default=None,
        help="月频动量跳过的最近月数；默认跟随主脚本或 input-dir/run_summary.csv。",
    )
    parser.add_argument(
        "--holding-days",
        type=int,
        default=None,
        help="未来持有收益交易日数；默认从 input-dir/run_summary.csv 或收益文件自动识别。",
    )
    parser.add_argument(
        "--group-num",
        type=int,
        default=None,
        help="分组数量；默认从分组收益文件自动识别。",
    )
    parser.add_argument(
        "--return-column",
        default=None,
        help="未来收益率字段名；默认从 input-dir/run_summary.csv 或因子文件自动识别。",
    )
    parser.add_argument(
        "--output-tag",
        default=None,
        help="自定义输出文件标签；不填则自动使用 lb{回看天数}_hd{持有天数}_g{分组数}_mkt{市场类型}。",
    )
    return parser.parse_args()


def load_main_script_defaults() -> dict[str, object]:
    """Read DEFAULT_* constants from the main script without importing or executing it."""

    defaults = MAIN_DEFAULTS_FALLBACK.copy()
    if not MAIN_SCRIPT_PATH.exists():
        return defaults

    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            source = MAIN_SCRIPT_PATH.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        return defaults

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return defaults

    wanted = set(defaults)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in wanted:
                try:
                    defaults[target.id] = ast.literal_eval(node.value)
                except Exception:
                    pass
    return defaults


def to_int_or_default(value: object, default: int) -> int:
    if value is None or pd.isna(value):
        return int(default)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "na", "<na>"}:
        return int(default)
    return int(float(text))


def parse_experiment_dir_name(path: Path) -> dict[str, object] | None:
    daily_match = DAILY_EXPERIMENT_DIR_RE.match(path.name)
    if daily_match:
        return {
            "rebalance_frequency": "daily",
            "lookback_days": int(daily_match.group("lookback")),
            "holding_days": int(daily_match.group("holding")),
            "group_num": int(daily_match.group("group")),
            "market_tag": daily_match.group("market"),
        }

    monthly_match = MONTHLY_EXPERIMENT_DIR_RE.match(path.name)
    if monthly_match:
        return {
            "rebalance_frequency": "monthly",
            "lookback_months": int(monthly_match.group("lookback_months")),
            "s": int(monthly_match.group("s")),
            "holding_months": int(monthly_match.group("holding_months")),
            "group_num": int(monthly_match.group("group")),
            "market_tag": monthly_match.group("market"),
        }

    legacy_match = LEGACY_EXPERIMENT_DIR_RE.match(path.name)
    if legacy_match:
        return {
            "rebalance_frequency": "daily",
            "lookback_days": int(legacy_match.group("lookback")),
            "holding_days": int(legacy_match.group("holding")),
            "group_num": int(legacy_match.group("group")),
            "market_tag": legacy_match.group("market"),
            "legacy_directory": True,
        }

    return None


def experiment_matches_main_defaults(info: dict[str, object], defaults: dict[str, object]) -> bool:
    frequency = str(defaults.get("DEFAULT_REBALANCE_FREQUENCY", "daily")).lower()
    if info.get("rebalance_frequency") != frequency:
        return False
    if frequency == "monthly":
        return (
            info.get("lookback_months") == int(defaults.get("DEFAULT_LOOKBACK_MONTHS", 3))
            and info.get("holding_months") == int(defaults.get("DEFAULT_HOLDING_MONTHS", 1))
            and info.get("s") == int(defaults.get("DEFAULT_S", 0))
            and info.get("group_num") == int(defaults.get("DEFAULT_GROUP_NUM", 10))
        )
    return (
        info.get("lookback_days") == int(defaults.get("DEFAULT_LOOKBACK_DAYS", 50))
        and info.get("holding_days") == int(defaults.get("DEFAULT_HOLDING_DAYS", 5))
        and info.get("group_num") == int(defaults.get("DEFAULT_GROUP_NUM", 10))
    )


def ensure_file_exists(path: Path) -> None:
    """检查输入文件是否存在，避免后续报错不清楚。"""

    if not path.exists():
        raise SystemExit(f"缺少必要输入文件：{path}")


def first_existing_input_file(input_dir: Path, candidate_names: list[str]) -> Path:
    """按优先级寻找输入文件；优先读取主脚本新生成的编号文件，兼容旧文件名。"""

    for file_name in candidate_names:
        path = input_dir / file_name
        if path.exists():
            return path
    candidate_text = "\n".join(str(input_dir / file_name) for file_name in candidate_names)
    raise SystemExit(f"缺少必要输入文件，已尝试以下路径：\n{candidate_text}")


def optional_input_file(input_dir: Path, candidate_names: list[str]) -> Path | None:
    """按优先级寻找可选输入文件，找不到时返回 None。"""

    for file_name in candidate_names:
        path = input_dir / file_name
        if path.exists():
            return path
    return None


def write_csv(data: pd.DataFrame, output_path: Path, index: bool = False) -> Path:
    """统一用 utf-8-sig 输出，便于 Windows Excel 正确显示中文。"""

    try:
        data.to_csv(output_path, index=index, encoding="utf-8-sig")
        return output_path
    except PermissionError:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = output_path.with_name(f"{output_path.stem}_{timestamp}{output_path.suffix}")
        data.to_csv(fallback_path, index=index, encoding="utf-8-sig")
        print(f"无法覆盖 {output_path.name}，可能文件正被打开；已改写入 {fallback_path.name}")
        return fallback_path


def set_chinese_font() -> None:
    """设置中文字体，确保 PNG 图表中的中文标题和行名能正常显示。"""

    candidates = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    available_fonts = {font.name for font in plt.matplotlib.font_manager.fontManager.ttflist}
    for font_name in candidates:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def set_yearly_xaxis(ax: plt.Axes, dates: pd.Series) -> None:
    """Set one x-axis major tick for every calendar year covered by dates."""

    clean_dates = pd.to_datetime(dates, errors="coerce").dropna()
    if clean_dates.empty:
        return

    start_year = int(clean_dates.min().year)
    end_year = int(clean_dates.max().year)
    ax.set_xlim(pd.Timestamp(start_year, 1, 1), pd.Timestamp(end_year, 12, 31))
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelrotation=0)


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
    """对一条日收益率序列计算均值、标准差、t 值和 p 值。"""

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


def parse_bool_series(series: pd.Series) -> pd.Series:
    """安全解析 CSV 中的布尔字段，避免字符串 False 被 astype(bool) 误判为 True。"""

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

    # 旧列名保留为兼容别名，但诊断统计统一使用 is_next_open_limit。
    output["is_next_open_one_word_limit_up"] = output["is_next_open_limit_up"]
    output["is_next_open_one_word_limit_down"] = output["is_next_open_limit_down"]
    output["is_next_open_one_word_limit"] = output["is_next_open_limit"]
    return output


def sanitize_filename_part(value: object) -> str:
    """把参数值转成适合放入文件名的短标签。"""

    text = str(value).strip().lower()
    text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff-]+", "_", text)
    return text.strip("_") or "na"


def normalize_market_types_tag(value: object) -> str:
    """把 Markettype 参数或摘要值转换成 mkt 后面的标签，例如 1,4 -> 1-4。"""

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "na"}:
        return "na"
    text = text.strip("{}[]()")
    tokens = [token for token in re.split(r"[,\s，、_-]+", text) if token]
    numeric_tokens: list[str] = []
    for token in tokens:
        try:
            numeric_tokens.append(str(int(float(token))))
        except ValueError:
            numeric_tokens.append(sanitize_filename_part(token))
    return "-".join(sorted(set(numeric_tokens), key=lambda item: int(item) if item.isdigit() else item))


def build_experiment_folder_name(
    lookback_days: object,
    holding_days: object,
    group_num: object,
    market_types_tag: str,
    rebalance_frequency: str = "daily",
    lookback_months: object | None = None,
    holding_months: object | None = None,
    s: object | None = None,
) -> str:
    """生成本次实验文件夹名：lb{回看}_hd{持有}_g{分组}_mkt{市场类型}。"""

    if str(rebalance_frequency).lower() == "monthly":
        return f"monthly_lb{lookback_months}m_s{s}_hd{holding_months}m_g{group_num}_mkt{market_types_tag}"
    return f"daily_lb{lookback_days}_hd{holding_days}_g{group_num}_mkt{market_types_tag}"


def resolve_experiment_output_dir(base_output_dir: Path, experiment_folder_name: str) -> Path:
    """如果用户传入 output 根目录，则在其中创建实验文件夹；若已是该文件夹则直接使用。"""

    if base_output_dir.name == experiment_folder_name:
        return base_output_dir
    return base_output_dir / experiment_folder_name


def has_calculation_outputs(path: Path) -> bool:
    """判断目录是否包含主计算脚本生成的必要文件。"""

    required_file_groups = [
        ["09_quantile_equal_weight_returns.csv", "momentum_quantile_equal_weight_returns.csv"],
        ["10_long_short_hedge_returns.csv", "momentum_long_short_returns.csv"],
        ["run_summary.csv"],
    ]
    factor_detail_file_group = [
        "07_momentum_factor_quantile_groups_with_forward_returns.csv",
        "momentum_factor_quantile_groups.csv",
    ]
    precomputed_diagnostic_file_groups = [
        ["12_momentum_ic_series.csv"],
        ["12_momentum_ic_ir_summary.csv"],
        ["12_momentum_factor_value_statistics.csv"],
        ["12_momentum_factor_input_summary.csv"],
    ]
    has_required = path.is_dir() and all(
        any((path / file_name).exists() for file_name in candidate_names)
        for candidate_names in required_file_groups
    )
    has_factor_detail = any((path / file_name).exists() for file_name in factor_detail_file_group)
    has_precomputed_diagnostics = all(
        any((path / file_name).exists() for file_name in candidate_names)
        for candidate_names in precomputed_diagnostic_file_groups
    )
    return has_required and (has_factor_detail or has_precomputed_diagnostics)


def find_latest_experiment_dir(output_root: Path) -> Path:
    """从 output 根目录中寻找最新的实验文件夹。"""

    if not output_root.exists():
        raise SystemExit(f"输出根目录不存在：{output_root}")

    main_defaults = load_main_script_defaults()
    target_frequency = str(main_defaults.get("DEFAULT_REBALANCE_FREQUENCY", "daily")).lower()
    candidates = []
    for path in output_root.iterdir():
        if not path.is_dir() or not has_calculation_outputs(path):
            continue
        info = parse_experiment_dir_name(path)
        if info is None:
            continue
        candidates.append((path, info))
    if not candidates:
        raise SystemExit(
            f"没有在 {output_root} 下找到包含主计算结果的 lb*_hd*_g*_mkt* 实验文件夹。"
        )
    preferred = [path for path, info in candidates if experiment_matches_main_defaults(info, main_defaults)]
    if preferred:
        return max(preferred, key=lambda path: (path.stat().st_mtime, path.name))

    same_frequency = [path for path, info in candidates if info.get("rebalance_frequency") == target_frequency]
    if same_frequency:
        return max(same_frequency, key=lambda path: (path.stat().st_mtime, path.name))

    return max((path for path, _ in candidates), key=lambda path: (path.stat().st_mtime, path.name))


def resolve_input_dir(input_dir: Path) -> Path:
    """解析诊断输入目录；支持直接传实验目录，也支持传 output 根目录自动选择最新实验。"""

    if has_calculation_outputs(input_dir):
        return input_dir
    return find_latest_experiment_dir(input_dir)


def numbered_path(output_dir: Path, number: int, base_name: str, suffix: str) -> Path:
    """生成诊断阶段编号输出路径，与主脚本 01-10 文件顺序衔接。"""

    return output_dir / f"{number:02d}_{base_name}{suffix}"


def load_upstream_run_summary(input_dir: Path) -> dict[str, str]:
    """读取上游计算脚本输出的 run_summary.csv，用于自动识别研究参数。"""

    path = input_dir / "run_summary.csv"
    if not path.exists():
        return {}
    summary = pd.read_csv(path, dtype=str)
    if "metric" not in summary.columns or "value" not in summary.columns:
        return {}
    return dict(zip(summary["metric"].astype(str), summary["value"].astype(str)))


def first_valid_value(series: pd.Series, default: object = None) -> object:
    """从一列中取第一个非空值，取不到时返回默认值。"""

    clean = series.dropna()
    if clean.empty:
        return default
    return clean.iloc[0]


def infer_research_params(
    args: argparse.Namespace,
    input_dir: Path,
    quantile_returns: pd.DataFrame,
    long_short_returns: pd.DataFrame,
    factor_data: pd.DataFrame,
) -> dict[str, object]:
    """从命令行参数、上游 run_summary 和数据列中推断本次诊断对应的研究参数。"""

    main_defaults = load_main_script_defaults()
    upstream_summary = load_upstream_run_summary(input_dir)
    requested_frequency = str(getattr(args, "rebalance_frequency", "auto")).lower()
    if requested_frequency != "auto":
        rebalance_frequency = requested_frequency
    else:
        rebalance_frequency = str(
            upstream_summary.get("rebalance_frequency")
            or main_defaults.get("DEFAULT_REBALANCE_FREQUENCY", "daily")
        ).lower()

    lookback_days = (
        args.lookback_days
        or upstream_summary.get("lookback_days")
        or main_defaults.get("DEFAULT_LOOKBACK_DAYS")
        or first_valid_value(factor_data.get("lookback_days", pd.Series(dtype="float64")), 50)
    )
    lookback_months = (
        getattr(args, "lookback_months", None)
        or upstream_summary.get("lookback_months")
        or main_defaults.get("DEFAULT_LOOKBACK_MONTHS")
        or first_valid_value(factor_data.get("lookback_days", pd.Series(dtype="float64")), 3)
    )
    monthly_skip_months = (
        getattr(args, "s", None)
        if getattr(args, "s", None) is not None
        else upstream_summary.get("s")
        or upstream_summary.get("monthly_skip_months")
        or main_defaults.get("DEFAULT_S")
        or first_valid_value(factor_data.get("monthly_skip_months", pd.Series(dtype="float64")), 0)
    )
    holding_days = (
        args.holding_days
        or upstream_summary.get("holding_days")
        or main_defaults.get("DEFAULT_HOLDING_DAYS")
        or first_valid_value(long_short_returns.get("holding_days", pd.Series(dtype="float64")), None)
        or first_valid_value(quantile_returns.get("holding_days", pd.Series(dtype="float64")), None)
        or first_valid_value(factor_data.get("holding_days", pd.Series(dtype="float64")), 1)
    )
    holding_months = (
        getattr(args, "holding_months", None)
        or upstream_summary.get("holding_months")
        or main_defaults.get("DEFAULT_HOLDING_MONTHS")
        or first_valid_value(long_short_returns.get("holding_months", pd.Series(dtype="float64")), None)
        or first_valid_value(quantile_returns.get("holding_months", pd.Series(dtype="float64")), None)
        or first_valid_value(factor_data.get("holding_months", pd.Series(dtype="float64")), 1)
    )
    group_num = (
        args.group_num
        or upstream_summary.get("group_num")
        or main_defaults.get("DEFAULT_GROUP_NUM")
        or first_valid_value(factor_data.get("group_num", pd.Series(dtype="float64")), None)
        or quantile_returns["quantile_group"].dropna().max()
    )
    return_column = (
        args.return_column
        or upstream_summary.get("return_column")
        or first_valid_value(factor_data.get("next_period_return_column_used", pd.Series(dtype="string")), None)
        or first_valid_value(factor_data.get("return_column_used", pd.Series(dtype="string")), "unknown_return")
    )
    market_types_value = (
        upstream_summary.get("selected_market_types")
        or upstream_summary.get("market_types_folder_tag")
        or "na"
    )

    lookback_days = to_int_or_default(lookback_days, int(main_defaults.get("DEFAULT_LOOKBACK_DAYS", 50)))
    holding_days = to_int_or_default(holding_days, int(main_defaults.get("DEFAULT_HOLDING_DAYS", 5)))
    lookback_months = to_int_or_default(lookback_months, int(main_defaults.get("DEFAULT_LOOKBACK_MONTHS", 3)))
    holding_months = to_int_or_default(holding_months, int(main_defaults.get("DEFAULT_HOLDING_MONTHS", 1)))
    monthly_skip_months = to_int_or_default(monthly_skip_months, int(main_defaults.get("DEFAULT_S", 0)))
    group_num = to_int_or_default(group_num, int(main_defaults.get("DEFAULT_GROUP_NUM", 10)))
    return_column = str(return_column)
    return_column_tag = RETURN_COLUMN_TAGS.get(return_column, sanitize_filename_part(return_column))
    market_types_tag = normalize_market_types_tag(market_types_value)
    experiment_folder_name = build_experiment_folder_name(
        lookback_days=lookback_days,
        holding_days=holding_days,
        group_num=group_num,
        market_types_tag=market_types_tag,
        rebalance_frequency=rebalance_frequency,
        lookback_months=lookback_months,
        holding_months=holding_months,
        s=monthly_skip_months,
    )

    output_tag = args.output_tag or experiment_folder_name

    return {
        "rebalance_frequency": rebalance_frequency,
        "lookback_days": lookback_days,
        "lookback_months": lookback_months,
        "s": monthly_skip_months,
        "monthly_skip_months": monthly_skip_months,
        "holding_days": holding_days,
        "holding_months": holding_months,
        "group_num": group_num,
        "return_column": return_column,
        "return_column_tag": return_column_tag,
        "selected_market_types": str(market_types_value),
        "market_types_tag": market_types_tag,
        "experiment_folder_name": experiment_folder_name,
        "output_tag": sanitize_filename_part(output_tag),
    }


# ============================================================
# 3. 数据读取
# ============================================================


def load_quantile_returns(input_dir: Path) -> pd.DataFrame:
    """读取分组等权收益率数据，并保留未来持有期和 T+1 可交易过滤后的组内股票数。"""

    path = first_existing_input_file(
        input_dir,
        [
            "09_quantile_equal_weight_returns.csv",
            "momentum_quantile_equal_weight_returns.csv",
        ],
    )
    data = pd.read_csv(path)
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data["quantile_group"] = pd.to_numeric(data["quantile_group"], errors="coerce").astype("Int64")
    for col in [
        "group_equal_weight_return",
        "group_return_sum",
        "group_stock_count",
        "signal_group_stock_count",
        "next_open_limit_excluded_count",
        "next_open_one_word_limit_excluded_count",
        "missing_next_record_count",
        "group_avg_momentum_zscore",
        "group_avg_momentum_raw",
        "holding_days",
        "holding_months",
        "monthly_skip_months",
    ]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    if "next_open_limit_excluded_count" not in data.columns:
        data["next_open_limit_excluded_count"] = pd.to_numeric(
            first_existing_series(data, ["next_open_one_word_limit_excluded_count"], 0),
            errors="coerce",
        )
    if "next_open_one_word_limit_excluded_count" not in data.columns:
        data["next_open_one_word_limit_excluded_count"] = data["next_open_limit_excluded_count"]
    for col in [
        "next_trade_date_min",
        "next_trade_date_max",
        "holding_start_trade_date_min",
        "holding_start_trade_date_max",
        "holding_end_trade_date_min",
        "holding_end_trade_date_max",
    ]:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce")
    return data


def load_long_short_returns(input_dir: Path) -> pd.DataFrame:
    """读取多空价差组合收益率、NAV、持有期和高低组 T+1 可交易股票数。"""

    path = first_existing_input_file(
        input_dir,
        [
            "10_long_short_hedge_returns.csv",
            "momentum_long_short_returns.csv",
        ],
    )
    data = pd.read_csv(path)
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    for col in [
        "long_top_group_return",
        "short_bottom_group_return",
        "long_short_spread_return",
        "long_top_group_nav",
        "short_bottom_group_nav",
        "long_short_spread_nav",
        "long_top_group_return_sum",
        "long_top_group_stock_count",
        "long_top_group_signal_stock_count",
        "long_top_group_next_open_limit_excluded_count",
        "long_top_group_one_word_limit_excluded_count",
        "long_top_group_missing_next_record_count",
        "short_bottom_group_return_sum",
        "short_bottom_group_stock_count",
        "short_bottom_group_signal_stock_count",
        "short_bottom_group_next_open_limit_excluded_count",
        "short_bottom_group_one_word_limit_excluded_count",
        "short_bottom_group_missing_next_record_count",
        "holding_days",
        "holding_months",
        "monthly_skip_months",
    ]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    if "long_top_group_next_open_limit_excluded_count" not in data.columns:
        data["long_top_group_next_open_limit_excluded_count"] = pd.to_numeric(
            first_existing_series(data, ["long_top_group_one_word_limit_excluded_count"], 0),
            errors="coerce",
        )
    if "short_bottom_group_next_open_limit_excluded_count" not in data.columns:
        data["short_bottom_group_next_open_limit_excluded_count"] = pd.to_numeric(
            first_existing_series(data, ["short_bottom_group_one_word_limit_excluded_count"], 0),
            errors="coerce",
        )
    if "long_top_group_one_word_limit_excluded_count" not in data.columns:
        data["long_top_group_one_word_limit_excluded_count"] = data[
            "long_top_group_next_open_limit_excluded_count"
        ]
    if "short_bottom_group_one_word_limit_excluded_count" not in data.columns:
        data["short_bottom_group_one_word_limit_excluded_count"] = data[
            "short_bottom_group_next_open_limit_excluded_count"
        ]
    for col in [
        "next_trade_date_min",
        "next_trade_date_max",
        "holding_start_trade_date_min",
        "holding_start_trade_date_max",
        "holding_end_trade_date_min",
        "holding_end_trade_date_max",
    ]:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce")
    return data


def load_factor_for_ic(input_dir: Path) -> pd.DataFrame:
    """读取计算 IC 所需的标准化因子、下一期收益和 T+1 可交易过滤标记。"""

    path = first_existing_input_file(
        input_dir,
        [
            "07_momentum_factor_quantile_groups_with_forward_returns.csv",
            "momentum_factor_quantile_groups.csv",
        ],
    )
    desired_usecols = [
        "stock_code",
        "trade_date",
        "momentum_start_date",
        "momentum_end_date",
        "holding_start_trade_date",
        "holding_end_trade_date",
        "momentum_raw",
        "momentum_3sigma",
        "momentum_zscore",
        "lookback_days",
        "lookback_unit",
        "holding_days",
        "holding_months",
        "lookback_valid_days",
        "lookback_limit_days_count",
        "lookback_has_limit_up_or_down",
        "return_column_used",
        "rebalance_frequency",
        "s",
        "monthly_skip_months",
        "next_period_return",
        "next_period_return_before_trade_filter",
        "next_period_return_after_long_short_trade_filter",
        "next_trade_date",
        "next_period_return_column_used",
        "is_next_open_limit_up",
        "is_next_open_limit_down",
        "is_next_open_limit",
        "is_next_open_one_word_limit",
        "is_tradable_next_open",
        "is_long_short_trade_candidate",
        "is_tradable_long_short_next_open",
        "trade_filter_reason",
        "quantile_group",
        "group_num",
        "long_short_role",
    ]
    available_columns = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [col for col in desired_usecols if col in available_columns]
    data = pd.read_csv(
        path,
        usecols=usecols,
        dtype={
            "stock_code": "string",
            "lookback_unit": "string",
            "rebalance_frequency": "string",
            "return_column_used": "string",
            "next_period_return_column_used": "string",
            "trade_filter_reason": "string",
            "long_short_role": "string",
        },
        low_memory=False,
    )
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    for col in ["momentum_start_date", "momentum_end_date", "holding_start_trade_date", "holding_end_trade_date", "next_trade_date"]:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce")
    for col in [
        "momentum_raw",
        "momentum_3sigma",
        "momentum_zscore",
        "next_period_return",
        "next_period_return_before_trade_filter",
        "next_period_return_after_long_short_trade_filter",
        "lookback_days",
        "holding_days",
        "holding_months",
        "s",
        "monthly_skip_months",
        "lookback_valid_days",
        "lookback_limit_days_count",
        "quantile_group",
        "group_num",
    ]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in ["holding_days"]:
        if col not in data.columns:
            data[col] = 1
    if "holding_months" not in data.columns:
        data["holding_months"] = pd.NA
    if "monthly_skip_months" not in data.columns:
        data["monthly_skip_months"] = pd.NA
    if "rebalance_frequency" not in data.columns:
        data["rebalance_frequency"] = pd.NA
    if "next_trade_date" not in data.columns:
        data["next_trade_date"] = pd.NaT
    if "trade_filter_reason" not in data.columns:
        data["trade_filter_reason"] = ""
    if "next_period_return_column_used" not in data.columns:
        data["next_period_return_column_used"] = pd.NA
    if "return_column_used" not in data.columns:
        data["return_column_used"] = pd.NA
    if "group_num" not in data.columns:
        data["group_num"] = pd.NA
    data = normalize_next_open_limit_columns(data)
    for col in ["lookback_has_limit_up_or_down", "is_tradable_next_open"]:
        if col in data.columns:
            data[col] = parse_bool_series(data[col])
        else:
            data[col] = False
    if "is_long_short_trade_candidate" in data.columns:
        data["is_long_short_trade_candidate"] = parse_bool_series(data["is_long_short_trade_candidate"])
    else:
        data["is_long_short_trade_candidate"] = (
            data["quantile_group"].eq(1) | data["quantile_group"].eq(data["group_num"])
        ).fillna(False)
    if "is_tradable_long_short_next_open" in data.columns:
        data["is_tradable_long_short_next_open"] = parse_bool_series(data["is_tradable_long_short_next_open"])
    else:
        data["is_tradable_long_short_next_open"] = (
            data["is_long_short_trade_candidate"] & data["is_tradable_next_open"]
        ).fillna(False)
    return data


def find_factor_detail_file(input_dir: Path) -> Path | None:
    """寻找 07 因子明细文件；没有该大文件时允许改用 12 号预计算小表。"""

    return optional_input_file(
        input_dir,
        [
            "07_momentum_factor_quantile_groups_with_forward_returns.csv",
            "momentum_factor_quantile_groups.csv",
        ],
    )


def load_precomputed_factor_diagnostics(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读取主脚本直接输出的 12 号 IC/摘要小表，用于跳过 07 明细文件。"""

    ic_series_path = first_existing_input_file(input_dir, ["12_momentum_ic_series.csv"])
    ic_summary_path = first_existing_input_file(input_dir, ["12_momentum_ic_ir_summary.csv"])
    factor_stats_path = first_existing_input_file(input_dir, ["12_momentum_factor_value_statistics.csv"])
    factor_input_summary_path = first_existing_input_file(input_dir, ["12_momentum_factor_input_summary.csv"])

    ic_series = pd.read_csv(ic_series_path)
    for col in ["trade_date", "next_trade_date"]:
        if col in ic_series.columns:
            ic_series[col] = pd.to_datetime(ic_series[col], errors="coerce")
    for col in [
        "ic",
        "rank_ic",
        "ic_stock_count",
        "signal_stock_count",
        "tradable_next_open_count",
        "next_open_limit_count",
        "long_short_trade_candidate_count",
        "tradable_long_short_next_open_count",
        "long_short_next_open_limit_count",
        "missing_next_record_count",
        "cumulative_ic",
        "cumulative_rank_ic",
        "ic_20d_rolling_mean",
        "rank_ic_20d_rolling_mean",
    ]:
        if col in ic_series.columns:
            ic_series[col] = pd.to_numeric(ic_series[col], errors="coerce")

    ic_summary = pd.read_csv(ic_summary_path)
    factor_stats = pd.read_csv(factor_stats_path)
    factor_input_summary = pd.read_csv(factor_input_summary_path)
    return ic_series, ic_summary, factor_stats, factor_input_summary


def load_long_short_members(input_dir: Path) -> pd.DataFrame:
    """读取主脚本输出的做多组和做空组股票明细；旧实验目录不存在该文件时返回空表。"""

    candidate_names = [
        "08_long_short_group_stock_members.csv",
        "momentum_long_short_group_stock_members.csv",
    ]
    existing_paths = [input_dir / file_name for file_name in candidate_names if (input_dir / file_name).exists()]
    if not existing_paths:
        return pd.DataFrame()

    data = pd.read_csv(existing_paths[0])
    for col in ["trade_date", "next_trade_date", "holding_start_trade_date", "holding_end_trade_date"]:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce")
    for col in [
        "quantile_group",
        "standardized_momentum_rank_desc",
        "momentum_raw",
        "momentum_3sigma",
        "momentum_zscore",
        "next_period_return",
        "future_return_valid_days",
    ]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    if "is_tradable_next_open" in data.columns:
        data["is_tradable_next_open"] = parse_bool_series(data["is_tradable_next_open"])
    if "is_long_short_trade_candidate" in data.columns:
        data["is_long_short_trade_candidate"] = parse_bool_series(data["is_long_short_trade_candidate"])
    if "is_tradable_long_short_next_open" in data.columns:
        data["is_tradable_long_short_next_open"] = parse_bool_series(data["is_tradable_long_short_next_open"])
    data = normalize_next_open_limit_columns(data)
    return data


def summarize_long_short_members(long_short_members: pd.DataFrame) -> pd.DataFrame:
    """汇总做多组和做空组股票明细，用于检查组合两端样本覆盖和可交易情况。"""

    columns = [
        "portfolio_side",
        "record_count",
        "stock_count",
        "trade_date_count",
        "avg_daily_member_count",
        "avg_next_period_return",
        "tradable_next_open_ratio",
    ]
    if long_short_members.empty or "portfolio_side" not in long_short_members.columns:
        return pd.DataFrame(columns=columns)

    data = long_short_members.copy()
    data["tradable_flag"] = (
        parse_bool_series(data["is_tradable_long_short_next_open"])
        if "is_tradable_long_short_next_open" in data.columns
        else parse_bool_series(data["is_tradable_next_open"])
        if "is_tradable_next_open" in data.columns
        else False
    )
    summary = (
        data.groupby("portfolio_side")
        .agg(
            record_count=("stock_code", "count"),
            stock_count=("stock_code", "nunique"),
            trade_date_count=("trade_date", "nunique"),
            avg_next_period_return=("next_period_return", "mean"),
            tradable_next_open_ratio=("tradable_flag", "mean"),
        )
        .reset_index()
    )
    summary["avg_daily_member_count"] = summary["record_count"] / summary["trade_date_count"].replace(0, pd.NA)
    return summary[columns]


# ============================================================
# 4. 排序检验表：均值、标准差、t 值、p 值
# ============================================================


def build_sort_test_table(
    quantile_returns: pd.DataFrame,
    long_short_returns: pd.DataFrame,
    group_num: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造与示例图类似的分组排序检验表。"""

    # 展示顺序按图例习惯从低因子到高因子：Low 是最后一组，High 是第 1 组。
    group_order = list(range(group_num, 0, -1))
    display_labels = ["Low"] + [str(i) for i in range(1, group_num - 1)] + ["High"]

    return_wide = quantile_returns.pivot(
        index="trade_date",
        columns="quantile_group",
        values="group_equal_weight_return",
    ).sort_index().reindex(columns=range(1, group_num + 1))

    raw_stats_rows: list[dict] = []
    table_values = {
        "均值（%）": [],
        "标准差（%）": [],
        "t值": [],
        "p值": [],
    }

    for group_id, label in zip(group_order, display_labels, strict=True):
        stats = mean_std_t_p(return_wide[group_id])
        raw_stats_rows.append({"portfolio": label, "source": f"group_{group_id}", **stats})
        table_values["均值（%）"].append(stats["mean"] * 100.0)
        table_values["标准差（%）"].append(stats["std"] * 100.0)
        table_values["t值"].append(stats["t_value"])
        table_values["p值"].append(stats["p_value"])

    spread_stats = mean_std_t_p(long_short_returns["long_short_spread_return"])
    raw_stats_rows.append({"portfolio": "High-Low", "source": "group_1_minus_group_10", **spread_stats})
    table_values["均值（%）"].append(spread_stats["mean"] * 100.0)
    table_values["标准差（%）"].append(spread_stats["std"] * 100.0)
    table_values["t值"].append(spread_stats["t_value"])
    table_values["p值"].append(spread_stats["p_value"])

    columns = display_labels + ["High-Low"]
    table = pd.DataFrame(table_values, index=columns).T
    raw_stats = pd.DataFrame(raw_stats_rows)
    return table, raw_stats


def save_sort_test_table_image(table: pd.DataFrame, output_path: Path) -> None:
    """把排序检验表保存成与示例图风格相近的 PNG。"""

    display_table = table.astype(object).copy()
    for row_name in display_table.index:
        if row_name in {"均值（%）", "标准差（%）", "t值"}:
            display_table.loc[row_name] = display_table.loc[row_name].map(lambda value: f"{value:.2f}")
        else:
            display_table.loc[row_name] = display_table.loc[row_name].map(lambda value: f"{value:.2f}")

    fig, ax = plt.subplots(figsize=(12.5, 3.1), dpi=220)
    ax.axis("off")
    ax.set_title("动量排序检验结果", fontsize=13, pad=12)

    table_artist = ax.table(
        cellText=display_table.values,
        rowLabels=display_table.index,
        colLabels=display_table.columns,
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(9.5)
    table_artist.scale(1.0, 1.35)

    for (row, col), cell in table_artist.get_celld().items():
        cell.set_edgecolor("#333333")
        cell.set_linewidth(0.25)
        cell.set_facecolor("white")
        if row == 0 or col == -1:
            cell.set_text_props(weight="normal")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 5. IC 和 IR
# ============================================================


def calculate_ic_ir(factor_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算日度 IC、RankIC 以及对应 IR。"""

    factor = normalize_next_open_limit_columns(factor_data)
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

    # IC 使用第 t 日因子与第 t+1 日收益率，不做 T+1 交易过滤；交易过滤只用于多空两端收益。
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

    ic_series = pd.DataFrame(ic_rows).dropna(subset=["ic", "rank_ic"])
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


def plot_daily_ic_curve(ic_series: pd.DataFrame, output_path: Path) -> None:
    """绘制每日 IC 序列，横轴为时间，纵轴为当日截面 IC 值。"""

    plot_data = ic_series.sort_values("trade_date").copy()
    if plot_data.empty:
        return

    fig, ax = plt.subplots(figsize=(13, 5.4), dpi=170)
    ax.plot(
        plot_data["trade_date"],
        plot_data["ic"],
        linewidth=0.9,
        color="#1f77b4",
        label="Daily IC",
    )
    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=0.8)
    ax.set_title("动量因子每日 IC 值")
    ax.set_xlabel("时间")
    ax.set_ylabel("IC")
    ax.legend()
    ax.grid(alpha=0.22)
    set_yearly_xaxis(ax, plot_data["trade_date"])
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_ic_curve(ic_series: pd.DataFrame, output_path: Path) -> None:
    """绘制累计 IC 曲线，用类似 NAV 的走势展示因子预测能力的长期方向。"""

    plot_data = ic_series.sort_values("trade_date").copy()
    if plot_data.empty:
        return

    # 统计每日 IC 序列，用于在图表顶部直接展示因子方向和稳定性。
    clean_ic = pd.to_numeric(plot_data["ic"], errors="coerce").dropna()
    ic_mean = clean_ic.mean()
    ic_std = clean_ic.std(ddof=1)
    ic_ir = ic_mean / ic_std if ic_std and not pd.isna(ic_std) else math.nan
    ic_positive_ratio = (clean_ic > 0).mean() if not clean_ic.empty else math.nan
    stats_text = (
        f"IC Mean = {ic_mean:.4f}    "
        f"IC Std = {ic_std:.4f}    "
        f"IR = {ic_ir:.4f}    "
        f"IC > 0 = {ic_positive_ratio:.2%}"
    )

    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=170)
    ax.plot(
        plot_data["trade_date"],
        plot_data["cumulative_ic"],
        linewidth=1.25,
        color="#1f77b4",
        label="Cumulative IC",
    )
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=0.8)
    ax.set_title("动量因子累计 IC 走势")
    fig.text(0.5, 0.965, stats_text, ha="center", va="top", fontsize=10.5)
    ax.set_xlabel("时间")
    ax.set_ylabel("累计 IC")
    ax.legend()
    ax.grid(alpha=0.22)
    set_yearly_xaxis(ax, plot_data["trade_date"])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


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
        ["lookback_days", ",".join(lookback_days_values)],
        ["holding_days", ",".join(holding_days_values)],
        ["rebalance_frequency", ",".join(rebalance_frequency_values)],
        ["monthly_skip_months", ",".join(monthly_skip_months_values)],
        ["holding_months", ",".join(holding_months_values)],
        ["group_num", ",".join(group_num_values)],
        ["factor_return_column_used", ",".join(factor_return_column_values)],
        ["next_period_return_column_used", ",".join(next_return_column_values)],
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


# ============================================================
# 6. 收益曲线和 NAV 曲线
# ============================================================


def build_quantile_nav(
    quantile_returns: pd.DataFrame,
    group_num: int | None = None,
    holding_days: int = 1,
    rebalance_frequency: str = "daily",
    holding_months: int | None = None,
) -> pd.DataFrame:
    """整理各分组收益率，并用非重叠持有期收益计算各组 NAV 曲线。"""

    return_wide = quantile_returns.pivot(
        index="trade_date",
        columns="quantile_group",
        values="group_equal_weight_return",
    ).sort_index()
    if group_num is not None:
        return_wide = return_wide.reindex(columns=range(1, group_num + 1))
    return_wide.columns = [f"G{int(col)}" for col in return_wide.columns]

    # group_equal_weight_return 是未来 holding_days 日收益率；若逐日 cumprod，会把重叠持有期收益重复复利。
    # 这里按 holding_days 间隔抽取非重叠调仓点，只用于展示更合理的分组 NAV 曲线。
    sample_step = max(int(holding_months or 1), 1) if rebalance_frequency == "monthly" else max(int(holding_days), 1)
    non_overlapping_return_wide = return_wide.iloc[::sample_step].copy()
    nav_wide = (1.0 + non_overlapping_return_wide).cumprod()
    nav_wide.columns = [f"{col}_NAV" for col in nav_wide.columns]

    combined = pd.concat([non_overlapping_return_wide, nav_wide], axis=1).reset_index()
    return combined


def summarize_quantile_returns(quantile_returns: pd.DataFrame, quantile_nav: pd.DataFrame) -> pd.DataFrame:
    """汇总各组收益、最终净值和 T+1 可交易过滤情况。"""

    rows: list[dict] = []
    group_num = int(quantile_returns["quantile_group"].dropna().max())
    for group_id in range(1, group_num + 1):
        group_data = quantile_returns.loc[quantile_returns["quantile_group"].eq(group_id)].copy()
        group_return = group_data["group_equal_weight_return"].dropna()
        final_nav_series = quantile_nav[f"G{group_id}_NAV"].dropna()
        signal_count = group_data.get("signal_group_stock_count", pd.Series(dtype="float64"))
        tradable_count = group_data.get("group_stock_count", pd.Series(dtype="float64"))
        next_open_limit_excluded = group_data.get(
            "next_open_limit_excluded_count",
            group_data.get("next_open_one_word_limit_excluded_count", pd.Series(dtype="float64")),
        )
        missing_next_record = group_data.get("missing_next_record_count", pd.Series(dtype="float64"))
        tradable_ratio = tradable_count / signal_count.replace(0, pd.NA)
        rows.append(
            {
                "quantile_group": group_id,
                "label": "G1 High" if group_id == 1 else f"G{group_num} Low" if group_id == group_num else f"G{group_id}",
                "observation_count": len(group_return),
                "daily_mean_return": group_return.mean(),
                "daily_std_return": group_return.std(ddof=1),
                "daily_positive_ratio": (group_return > 0).mean(),
                "daily_min_return": group_return.min(),
                "daily_max_return": group_return.max(),
                "final_nav": final_nav_series.iloc[-1] if not final_nav_series.empty else math.nan,
                "avg_signal_group_stock_count": signal_count.mean(),
                "avg_tradable_group_stock_count": tradable_count.mean(),
                "avg_tradable_ratio_after_t1_filter": tradable_ratio.mean(),
                "total_next_open_limit_excluded": next_open_limit_excluded.sum(),
                "total_missing_next_record": missing_next_record.sum(),
            }
        )

    summary = pd.DataFrame(rows)
    summary["final_cumulative_return"] = summary["final_nav"] - 1.0
    summary["final_cumulative_return_percent"] = summary["final_cumulative_return"] * 100.0
    summary["daily_mean_return_percent"] = summary["daily_mean_return"] * 100.0
    summary["daily_std_return_percent"] = summary["daily_std_return"] * 100.0
    return summary


def summarize_long_short_returns(long_short_returns: pd.DataFrame) -> pd.DataFrame:
    """汇总多空价差组合的日度收益分布，用于判断曲线是否被坐标轴压缩。"""

    spread_return = pd.to_numeric(long_short_returns["long_short_spread_return"], errors="coerce").dropna()
    summary = spread_return.describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    rows = summary.rename_axis("metric").reset_index(name="value")
    rows = pd.concat(
        [
            rows,
            pd.DataFrame(
                [
                    ["positive_ratio", float((spread_return > 0).mean())],
                    ["negative_ratio", float((spread_return < 0).mean())],
                    ["mean_abs_return", float(spread_return.abs().mean())],
                    ["annualized_volatility", float(spread_return.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))],
                ],
                columns=["metric", "value"],
            ),
        ],
        ignore_index=True,
    )
    rows["value_percent"] = rows["value"] * 100.0
    return rows


def plot_quantile_return_curves(quantile_nav: pd.DataFrame, output_path: Path, group_num: int) -> None:
    """绘制所有分组的原始 NAV 曲线。"""

    plot_data = quantile_nav.copy()
    plt.figure(figsize=(13, 7), dpi=170)
    for group_id in range(1, group_num + 1):
        label = f"G{group_id}"
        if group_id == 1:
            label = "G1 High"
        elif group_id == group_num:
            label = f"G{group_num} Low"
        plt.plot(plot_data["trade_date"], plot_data[f"G{group_id}_NAV"], linewidth=1.05, label=label)

    plt.axhline(1.0, color="#666666", linestyle="--", linewidth=0.8)
    plt.title(f"动量因子{group_num}组等权收益率曲线")
    plt.xlabel("时间")
    plt.ylabel("NAV")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(alpha=0.22)
    set_yearly_xaxis(plt.gca(), plot_data["trade_date"])
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_long_short_nav(long_short_returns: pd.DataFrame, output_path: Path) -> None:
    """绘制多空价差组合原始 NAV 走势。"""

    plot_data = long_short_returns.sort_values("trade_date").copy()
    if plot_data.empty:
        return

    if "holding_end_trade_date_max" in plot_data.columns:
        plot_data["nav_plot_date"] = pd.to_datetime(plot_data["holding_end_trade_date_max"], errors="coerce")
    elif "holding_end_trade_date_min" in plot_data.columns:
        plot_data["nav_plot_date"] = pd.to_datetime(plot_data["holding_end_trade_date_min"], errors="coerce")
    else:
        plot_data["nav_plot_date"] = pd.to_datetime(plot_data["trade_date"], errors="coerce")

    plot_data = plot_data.dropna(subset=["nav_plot_date", "long_short_spread_nav"]).sort_values("nav_plot_date")
    if plot_data.empty:
        return

    first_row = plot_data.iloc[0]
    if "holding_start_trade_date_min" in plot_data.columns and pd.notna(first_row.get("holding_start_trade_date_min")):
        initial_date = pd.to_datetime(first_row["holding_start_trade_date_min"], errors="coerce")
    else:
        initial_date = pd.to_datetime(first_row["nav_plot_date"], errors="coerce")
    if pd.isna(initial_date) or initial_date > first_row["nav_plot_date"]:
        initial_date = first_row["nav_plot_date"]

    nav_plot = pd.concat(
        [
            pd.DataFrame(
                {
                    "nav_plot_date": [initial_date],
                    "long_short_spread_nav": [1.0],
                }
            ),
            plot_data[["nav_plot_date", "long_short_spread_nav"]],
        ],
        ignore_index=True,
    )

    fig, nav_ax = plt.subplots(figsize=(13, 5.4), dpi=170)

    nav_ax.plot(
        nav_plot["nav_plot_date"],
        nav_plot["long_short_spread_nav"],
        linewidth=1.15,
        color="#1f77b4",
        label="High-Low NAV",
    )
    nav_ax.axhline(1.0, color="#666666", linestyle="--", linewidth=0.8)
    nav_ax.set_title("动量因子多空价差组合净值走势")
    nav_ax.set_xlabel("时间")
    nav_ax.set_ylabel("NAV")
    nav_ax.legend()
    nav_ax.grid(alpha=0.22)
    set_yearly_xaxis(nav_ax, nav_plot["nav_plot_date"])

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_long_short_volatility(
    long_short_returns: pd.DataFrame,
    output_path: Path,
    holding_days: int,
    rebalance_frequency: str = "daily",
    holding_months: int | None = None,
    windows: tuple = (10, 20, 60),
) -> None:
    """绘制多空价差组合日收益率的滚动年化波动率。"""

    plot_data = long_short_returns.sort_values("trade_date").copy()
    spread_return = pd.to_numeric(plot_data["long_short_spread_return"], errors="coerce")
    if rebalance_frequency == "monthly":
        annualization_periods_per_year = MONTHS_PER_YEAR / max(int(holding_months or 1), 1)
    else:
        annualization_periods_per_year = TRADING_DAYS_PER_YEAR / max(int(holding_days), 1)

    plt.figure(figsize=(13, 5.5), dpi=170)

    for window in windows:
        rolling_vol = (
            spread_return.rolling(window=window, min_periods=window)
            .std(ddof=1)
            * (annualization_periods_per_year ** 0.5)
        )
        plt.plot(
            plot_data["trade_date"],
            rolling_vol * 100.0,
            linewidth=1.05,
            label=f"{window}日滚动年化波动率",
        )

    plt.axhline(
        spread_return.std(ddof=1) * (annualization_periods_per_year ** 0.5) * 100.0,
        color="#d62728",
        linestyle="--",
        linewidth=1.0,
        label="全样本年化波动率",
    )

    plt.title("动量因子多空价差组合 — 滚动年化波动率")
    plt.xlabel("时间")
    plt.ylabel("年化波动率（%）")
    plt.legend()
    plt.grid(alpha=0.22)
    set_yearly_xaxis(plt.gca(), plot_data["trade_date"])
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


# ============================================================
# 7. 主流程
# ============================================================


def main() -> None:
    args = parse_args()
    input_dir: Path = resolve_input_dir(args.input_dir)
    set_chinese_font()

    print("1/6 正在读取分组收益、多空收益和因子数据...")
    quantile_returns = load_quantile_returns(input_dir)
    long_short_returns = load_long_short_returns(input_dir)
    factor_detail_path = find_factor_detail_file(input_dir)
    if factor_detail_path is not None:
        factor_data = load_factor_for_ic(input_dir)
    else:
        print("未找到 07 因子明细文件，将使用主脚本预计算的 12 号 IC/摘要小表。")
        factor_data = pd.DataFrame()
    long_short_members = load_long_short_members(input_dir)

    print("2/6 正在识别本次诊断对应的研究参数...")
    research_params = infer_research_params(
        args=args,
        input_dir=input_dir,
        quantile_returns=quantile_returns,
        long_short_returns=long_short_returns,
        factor_data=factor_data,
    )
    output_tag = research_params["output_tag"]
    rebalance_frequency = str(research_params["rebalance_frequency"])
    group_num = int(research_params["group_num"])
    holding_days = int(research_params["holding_days"])
    holding_months = int(research_params["holding_months"])
    experiment_folder_name = str(research_params["experiment_folder_name"])
    output_dir: Path = (
        input_dir
        if args.output_dir is None
        else resolve_experiment_output_dir(args.output_dir, experiment_folder_name)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("3/6 正在生成排序检验统计表...")
    sort_test_table, sort_test_stats = build_sort_test_table(
        quantile_returns,
        long_short_returns,
        group_num=group_num,
    )
    long_short_member_summary = summarize_long_short_members(long_short_members)
    sort_test_table_csv = numbered_path(output_dir, 11, "momentum_sort_test_table", ".csv")
    sort_test_stats_csv = numbered_path(output_dir, 11, "momentum_sort_test_stats_detail", ".csv")
    long_short_member_summary_csv = numbered_path(output_dir, 11, "momentum_long_short_group_member_summary", ".csv")
    sort_test_png = numbered_path(output_dir, 11, "momentum_sort_test_table", ".png")
    write_csv(sort_test_table, sort_test_table_csv, index=True)
    write_csv(sort_test_stats, sort_test_stats_csv)
    write_csv(long_short_member_summary, long_short_member_summary_csv)
    save_sort_test_table_image(sort_test_table, sort_test_png)

    print("4/6 正在计算 IC/IR 并绘制 IC 曲线...")
    if factor_data.empty:
        ic_series, ic_summary, factor_stats, factor_input_summary = load_precomputed_factor_diagnostics(input_dir)
    else:
        ic_series, ic_summary = calculate_ic_ir(factor_data)
        factor_stats = calculate_factor_value_statistics(factor_data)
        factor_input_summary = summarize_factor_input(factor_data)
    ic_series_csv = numbered_path(output_dir, 12, "momentum_ic_series", ".csv")
    ic_summary_csv = numbered_path(output_dir, 12, "momentum_ic_ir_summary", ".csv")
    factor_stats_csv = numbered_path(output_dir, 12, "momentum_factor_value_statistics", ".csv")
    factor_input_summary_csv = numbered_path(output_dir, 12, "momentum_factor_input_summary", ".csv")
    daily_ic_png = numbered_path(output_dir, 12, "momentum_daily_ic_curve", ".png")
    cumulative_ic_png = numbered_path(output_dir, 12, "momentum_cumulative_ic_curve", ".png")
    write_csv(ic_series, ic_series_csv)
    write_csv(ic_summary, ic_summary_csv)
    write_csv(factor_stats, factor_stats_csv)
    write_csv(factor_input_summary, factor_input_summary_csv)
    plot_daily_ic_curve(ic_series, daily_ic_png)
    plot_cumulative_ic_curve(ic_series, cumulative_ic_png)

    print("5/6 正在整理分组收益率、NAV 数据并绘图...")
    quantile_nav = build_quantile_nav(
        quantile_returns,
        group_num=group_num,
        holding_days=holding_days,
        rebalance_frequency=rebalance_frequency,
        holding_months=holding_months,
    )
    quantile_return_summary = summarize_quantile_returns(quantile_returns, quantile_nav)
    long_short_return_summary = summarize_long_short_returns(long_short_returns)
    quantile_returns_csv = numbered_path(output_dir, 13, "momentum_quantile_returns_for_diagnostics", ".csv")
    quantile_nav_csv = numbered_path(output_dir, 13, "momentum_quantile_nav", ".csv")
    quantile_summary_csv = numbered_path(output_dir, 13, "momentum_quantile_return_summary", ".csv")
    long_short_distribution_csv = numbered_path(output_dir, 14, "momentum_long_short_return_distribution", ".csv")
    quantile_nav_png = numbered_path(output_dir, 13, "momentum_quantile_nav_curve", ".png")
    long_short_nav_png = numbered_path(output_dir, 14, "momentum_long_short_nav_curve", ".png")
    long_short_volatility_png = numbered_path(output_dir, 14, "momentum_long_short_volatility", ".png")
    write_csv(quantile_returns, quantile_returns_csv)
    write_csv(quantile_nav, quantile_nav_csv)
    write_csv(quantile_return_summary, quantile_summary_csv)
    write_csv(long_short_return_summary, long_short_distribution_csv)
    plot_quantile_return_curves(quantile_nav, quantile_nav_png, group_num=group_num)
    plot_long_short_nav(long_short_returns, long_short_nav_png)
    plot_long_short_volatility(
        long_short_returns,
        long_short_volatility_png,
        holding_days=holding_days,
        rebalance_frequency=rebalance_frequency,
        holding_months=holding_months,
    )

    print("6/6 正在输出运行汇总...")
    run_summary = pd.DataFrame(
        [
            ["input_dir", str(input_dir)],
            ["output_dir", str(output_dir)],
            ["output_tag", output_tag],
            ["experiment_folder_name", experiment_folder_name],
            ["rebalance_frequency", research_params["rebalance_frequency"]],
            ["lookback_days", research_params["lookback_days"]],
            ["lookback_months", research_params["lookback_months"]],
            ["s", research_params["s"]],
            ["monthly_skip_months", research_params["monthly_skip_months"]],
            ["holding_days", research_params["holding_days"]],
            ["holding_months", research_params["holding_months"]],
            ["group_num", research_params["group_num"]],
            ["selected_market_types", research_params["selected_market_types"]],
            ["market_types_tag", research_params["market_types_tag"]],
            ["return_column", research_params["return_column"]],
            ["quantile_return_rows", len(quantile_returns)],
            ["long_short_return_rows", len(long_short_returns)],
            ["factor_rows_for_ic", len(factor_data)],
            ["long_short_group_member_rows", len(long_short_members)],
            ["ic_observation_count", len(ic_series)],
            ["quantile_return_start", quantile_returns["trade_date"].min()],
            ["quantile_return_end", quantile_returns["trade_date"].max()],
            ["long_short_return_start", long_short_returns["trade_date"].min()],
            ["long_short_return_end", long_short_returns["trade_date"].max()],
            [
                "next_period_return_column_used",
                ",".join(
                    factor_data["next_period_return_column_used"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
                if "next_period_return_column_used" in factor_data.columns
                else str(
                    factor_input_summary.loc[
                        factor_input_summary["metric"].eq("next_period_return_column_used"),
                        "value",
                    ].iloc[0]
                )
                if {"metric", "value"}.issubset(factor_input_summary.columns)
                and factor_input_summary["metric"].eq("next_period_return_column_used").any()
                else "",
            ],
            ["sort_test_csv", str(sort_test_table_csv)],
            ["sort_test_detail_csv", str(sort_test_stats_csv)],
            ["long_short_group_member_summary_csv", str(long_short_member_summary_csv)],
            ["sort_test_png", str(sort_test_png)],
            ["ic_series_csv", str(ic_series_csv)],
            ["ic_summary_csv", str(ic_summary_csv)],
            ["daily_ic_png", str(daily_ic_png)],
            ["cumulative_ic_png", str(cumulative_ic_png)],
            ["quantile_returns_csv", str(quantile_returns_csv)],
            ["quantile_nav_csv", str(quantile_nav_csv)],
            ["quantile_return_summary_csv", str(quantile_summary_csv)],
            ["long_short_return_distribution_csv", str(long_short_distribution_csv)],
            ["quantile_nav_curve_png", str(quantile_nav_png)],
            ["long_short_nav_curve_png", str(long_short_nav_png)],
            ["long_short_volatility_png", str(long_short_volatility_png)],
        ],
        columns=["metric", "value"],
    )
    diagnostic_summary_csv = numbered_path(output_dir, 15, "diagnostic_run_summary", ".csv")
    write_csv(run_summary, diagnostic_summary_csv)

    print("因子检测完成。")
    print(f"输出目录：{output_dir}")
    print(f"输出标签：{output_tag}")
    print(f"IC/IR 摘要：\n{ic_summary.to_string(index=False)}")


if __name__ == "__main__":
    main()
