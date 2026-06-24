from __future__ import annotations

import argparse
import ast
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd


# ============================================================
# 1. 配置
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
MAIN_SCRIPT_PATH = PROJECT_DIR / "calculate_momentum_factor_sql.py"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "output"
DEFAULT_PLOT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "momentum_factor_figures"
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
    r"^monthly_lb(?P<lookback_months>\d+)m(?:_s(?P<s>\d+))?_hd(?P<holding_months>\d+)m_g(?P<group>\d+)_mkt(?P<market>.+)$"
)

TRADING_DAYS_PER_YEAR = 252
MONTHS_PER_YEAR = 12

FACTOR_FILE_CANDIDATES = [
    "07_momentum_factor_quantile_groups_with_forward_returns.csv",
    "momentum_factor_quantile_groups.csv",
]
QUANTILE_RETURN_FILE_CANDIDATES = [
    "09_quantile_equal_weight_returns.csv",
    "momentum_quantile_equal_weight_returns.csv",
]
QUANTILE_RETURN_SUMMARY_FILE_CANDIDATES = [
    "13_momentum_quantile_return_summary.csv",
]
IC_SUMMARY_FILE_CANDIDATES = [
    "12_momentum_ic_ir_summary.csv",
    "momentum_ic_ir_summary.csv",
]
IC_SUMMARY_FILE_GLOBS = [
    "12_momentum_ic_ir_summary*.csv",
    "*momentum_ic_ir_summary*.csv",
]


@dataclass(frozen=True)
class ExperimentInfo:
    path: Path
    rebalance_frequency: str
    lookback_days: int | None
    holding_days: int | None
    group_num: int
    market_tag: str
    lookback_months: int | None = None
    holding_months: int | None = None
    s: int | None = None


@dataclass
class MetricResult:
    value: float
    source_file: Path | None
    status: str
    observation_count: int | None = None


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


def main_default_rebalance_frequency(defaults: dict[str, object]) -> str:
    value = str(defaults.get("DEFAULT_REBALANCE_FREQUENCY", "daily")).lower()
    return value if value in {"daily", "monthly"} else "daily"


def resolve_auto_frequency(requested_frequency: str, defaults: dict[str, object]) -> str:
    requested = str(requested_frequency).lower()
    if requested != "auto":
        return requested
    return main_default_rebalance_frequency(defaults)


def resolve_figure_frequency(figure_frequency: str, global_frequency: str) -> str:
    requested = str(figure_frequency).lower()
    if requested != "auto":
        return requested
    return str(global_frequency).lower()


# ============================================================
# 2. 参数和通用工具
# ============================================================


def parse_args() -> argparse.Namespace:
    main_defaults = load_main_script_defaults()
    default_group_num = int(main_defaults.get("DEFAULT_GROUP_NUM", 10))
    default_lookback_days = int(main_defaults.get("DEFAULT_LOOKBACK_DAYS", 50))
    default_holding_days = int(main_defaults.get("DEFAULT_HOLDING_DAYS", 5))
    default_lookback_months = int(main_defaults.get("DEFAULT_LOOKBACK_MONTHS", 3))
    default_holding_months = int(main_defaults.get("DEFAULT_HOLDING_MONTHS", 1))
    default_s = int(main_defaults.get("DEFAULT_S", 0))

    parser = argparse.ArgumentParser(
        description=(
            "基于 calculate_momentum_factor_sql.py 的输出目录绘制动量因子参数图和分组绩效表。"
            "脚本只读取已有 CSV，不重新连接数据库或重跑主计算。"
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="主脚本 output 根目录。")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_OUTPUT_DIR, help="PNG 和汇总 CSV 输出目录。")
    parser.add_argument("--market-tag", default=None, help="可选市场标签过滤，例如 1-4-16-32-64。")
    parser.add_argument(
        "--rebalance-frequency",
        choices=["auto", "daily", "monthly"],
        default="auto",
        help="全局画图频率；auto 时跟随 calculate_momentum_factor_sql.py 的 DEFAULT_REBALANCE_FREQUENCY。",
    )
    parser.add_argument("--group-num", type=int, default=default_group_num, help="图1/图2使用的分组数，默认跟随主脚本。")
    parser.add_argument("--lookback-days", default="20,40,60", help="图1横轴 lookback days，逗号分隔。")
    parser.add_argument(
        "--figure1-experiments",
        default="lb10_hd5,lb20_hd5,lb30_hd5,lb40_hd5,lb50_hd5,lb60_hd5",
        help="图1横轴实验标签，逗号分隔；支持 lb40_hd20 或 40:20 写法。",
    )
    parser.add_argument("--holding-days", default="1,5,10", help="图2横轴 holding days，逗号分隔。")
    parser.add_argument("--holding-months", default="1,2,3,4,5", help="图2月频横轴 holding months，逗号分隔。")
    parser.add_argument("--fixed-holding-days", type=int, default=default_holding_days, help="图1固定 holding days，默认跟随主脚本。")
    parser.add_argument("--fixed-lookback-days", type=int, default=default_lookback_days, help="图2固定 lookback days，默认跟随主脚本。")
    parser.add_argument("--fixed-lookback-months", type=int, default=default_lookback_months, help="图2月频固定 lookback months，默认跟随主脚本。")
    parser.add_argument("--figure2-s", type=int, default=default_s, help="图2月频 skip 月数，默认跟随主脚本。")
    parser.add_argument("--figure3-lookback-days", type=int, default=default_lookback_days, help="图3日频实验 lookback days，默认跟随主脚本。")
    parser.add_argument("--figure3-holding-days", type=int, default=default_holding_days, help="图3日频实验 holding days，默认跟随主脚本。")
    parser.add_argument("--figure3-group-num", type=int, default=default_group_num, help="图3分组数量，默认跟随主脚本。")
    parser.add_argument(
        "--figure3-rebalance-frequency",
        choices=["auto", "daily", "monthly"],
        default="auto",
        help="图3选择的实验频率；auto 时跟随 --rebalance-frequency。",
    )
    parser.add_argument("--figure3-lookback-months", type=int, default=default_lookback_months, help="图3月频实验 lookback months，默认跟随主脚本。")
    parser.add_argument("--figure3-holding-months", type=int, default=default_holding_months, help="图3月频实验 holding months，默认跟随主脚本。")
    parser.add_argument("--figure3-s", type=int, default=default_s, help="图3月频实验 skip 月数，默认跟随主脚本。")
    parser.add_argument(
        "--figure4-rebalance-frequency",
        choices=["auto", "daily", "monthly"],
        default="auto",
        help="图4分组绩效表选择的实验频率；auto 时跟随 --rebalance-frequency。",
    )
    parser.add_argument("--figure4-lookback-days", type=int, default=None, help="图4日频实验 lookback days；默认沿用图3。")
    parser.add_argument("--figure4-holding-days", type=int, default=None, help="图4日频实验 holding days；默认沿用图3。")
    parser.add_argument("--figure4-lookback-months", type=int, default=default_lookback_months, help="图4月频实验 lookback months，默认跟随主脚本。")
    parser.add_argument("--figure4-holding-months", type=int, default=default_holding_months, help="图4月频实验 holding months，默认跟随主脚本。")
    parser.add_argument("--figure4-s", type=int, default=default_s, help="图4月频实验 skip 月数，默认跟随主脚本。")
    parser.add_argument("--figure4-group-num", type=int, default=default_group_num, help="图4分组数量，默认跟随主脚本。")
    parser.add_argument(
        "--figure4-overlap-mode",
        choices=["non-overlap", "all"],
        default="non-overlap",
        help="图4绩效计算使用非重叠调仓点或全部重叠截面；默认 non-overlap。",
    )
    parser.add_argument(
        "--validity-metric",
        choices=["ic_mean", "rank_ic_mean"],
        default="ic_mean",
        help="图2“因子有效期”的柱状值，默认使用 IC Mean。",
    )
    parser.add_argument(
        "--figure3-return-mode",
        choices=["mean-holding", "last-holding", "cumulative"],
        default="mean-holding",
        help=(
            "图3收益率口径：mean-holding 为各组 holding horizon 收益的时间序列均值；"
            "last-holding 为最后一个截面的 holding horizon 收益；"
            "cumulative 为旧版跨期复利口径。"
        ),
    )
    parser.add_argument("--dpi", type=int, default=180, help="输出图片 DPI。")
    parser.add_argument("--strict", action="store_true", help="缺少任一输入实验或 CSV 时直接报错。")
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for token in re.split(r"[,，\s]+", str(text).strip()):
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise SystemExit(f"参数列表为空：{text}")
    return values


def parse_figure1_experiments(text: str) -> list[dict[str, int | str]]:
    """解析图1实验标签，支持 lb40_hd20 或 40:20。"""

    specs: list[dict[str, int | str]] = []
    for token in re.split(r"[,，\s]+", str(text).strip()):
        if not token:
            continue

        normalized = token.strip()
        match = re.match(r"^lb(?P<lookback>\d+)_hd(?P<holding>\d+)$", normalized, flags=re.IGNORECASE)
        if match is None:
            match = re.match(r"^(?P<lookback>\d+)[:/_-](?P<holding>\d+)$", normalized)
        if match is None:
            raise SystemExit(f"无法解析图1实验标签：{token}；请使用 lb40_hd20 或 40:20。")

        lookback_days = int(match.group("lookback"))
        holding_days = int(match.group("holding"))
        specs.append(
            {
                "lookback_days": lookback_days,
                "holding_days": holding_days,
                "label": f"lb{lookback_days}_hd{holding_days}",
            }
        )
    if not specs:
        raise SystemExit(f"图1实验标签为空：{text}")
    return sorted(specs, key=lambda item: (int(item["lookback_days"]), int(item["holding_days"])))


def set_chinese_font() -> None:
    candidates = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    available_fonts = {font.name for font in plt.matplotlib.font_manager.fontManager.ttflist}
    for font_name in candidates:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def write_csv(data: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def discover_experiments(output_root: Path) -> list[ExperimentInfo]:
    if not output_root.exists():
        return []

    experiments: list[ExperimentInfo] = []
    for path in output_root.iterdir():
        if not path.is_dir():
            continue
        daily_match = DAILY_EXPERIMENT_DIR_RE.match(path.name)
        legacy_match = LEGACY_EXPERIMENT_DIR_RE.match(path.name)
        monthly_match = MONTHLY_EXPERIMENT_DIR_RE.match(path.name)
        if daily_match or legacy_match:
            match = daily_match or legacy_match
            experiments.append(
                ExperimentInfo(
                    path=path,
                    rebalance_frequency="daily",
                    lookback_days=int(match.group("lookback")),
                    holding_days=int(match.group("holding")),
                    group_num=int(match.group("group")),
                    market_tag=match.group("market"),
                )
            )
        elif monthly_match:
            experiments.append(
                ExperimentInfo(
                    path=path,
                    rebalance_frequency="monthly",
                    lookback_days=None,
                    holding_days=None,
                    group_num=int(monthly_match.group("group")),
                    market_tag=monthly_match.group("market"),
                    lookback_months=int(monthly_match.group("lookback_months")),
                    holding_months=int(monthly_match.group("holding_months")),
                    s=int(monthly_match.group("s") or 0),
                )
            )
    return experiments


def select_experiment(
    experiments: list[ExperimentInfo],
    lookback_days: int,
    holding_days: int,
    group_num: int,
    market_tag: str | None,
) -> ExperimentInfo | None:
    candidates = [
        item
        for item in experiments
        if item.rebalance_frequency == "daily"
        and item.lookback_days == lookback_days
        and item.holding_days == holding_days
        and item.group_num == group_num
        and (market_tag is None or item.market_tag == market_tag)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item.path.stat().st_mtime, item.path.name))


def select_monthly_experiment(
    experiments: list[ExperimentInfo],
    lookback_months: int,
    holding_months: int,
    s: int,
    group_num: int,
    market_tag: str | None,
) -> ExperimentInfo | None:
    candidates = [
        item
        for item in experiments
        if item.rebalance_frequency == "monthly"
        and item.lookback_months == lookback_months
        and item.holding_months == holding_months
        and (item.s or 0) == s
        and item.group_num == group_num
        and (market_tag is None or item.market_tag == market_tag)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item.path.stat().st_mtime, item.path.name))


def find_input_file(experiment_dir: Path, exact_names: list[str], glob_patterns: list[str] | None = None) -> Path | None:
    for name in exact_names:
        path = experiment_dir / name
        if path.exists():
            return path

    for pattern in glob_patterns or []:
        matches = [path for path in experiment_dir.glob(pattern) if path.is_file()]
        if matches:
            return max(matches, key=lambda path: (path.stat().st_mtime, path.name))
    return None


def empty_metric(status: str) -> MetricResult:
    return MetricResult(value=math.nan, source_file=None, status=status, observation_count=None)


# ============================================================
# 3. 指标读取和计算
# ============================================================


def read_ic_mean_from_summary(experiment_dir: Path, metric: str) -> MetricResult | None:
    path = find_input_file(experiment_dir, IC_SUMMARY_FILE_CANDIDATES, IC_SUMMARY_FILE_GLOBS)
    if path is None:
        return None

    data = pd.read_csv(path)
    if not {"metric", "mean"}.issubset(data.columns):
        return None

    target = "IC" if metric == "ic_mean" else "RankIC"
    matched = data.loc[data["metric"].astype(str).str.upper().eq(target.upper())].copy()
    if matched.empty:
        return None

    value = pd.to_numeric(matched["mean"], errors="coerce").dropna()
    if value.empty:
        return None

    obs = None
    if "observation_count" in matched.columns:
        obs_values = pd.to_numeric(matched["observation_count"], errors="coerce").dropna()
        obs = int(obs_values.iloc[0]) if not obs_values.empty else None
    return MetricResult(value=float(value.iloc[0]), source_file=path, status="read_ic_summary", observation_count=obs)


def compute_ic_mean_from_factor_file(experiment_dir: Path, metric: str) -> MetricResult:
    path = find_input_file(experiment_dir, FACTOR_FILE_CANDIDATES)
    if path is None:
        return empty_metric("missing_factor_file_07")

    available_columns = pd.read_csv(path, nrows=0).columns.tolist()
    required_columns = ["trade_date", "momentum_zscore", "next_period_return"]
    if any(col not in available_columns for col in required_columns):
        return MetricResult(math.nan, path, "missing_required_factor_columns", None)

    data = pd.read_csv(path, usecols=required_columns, low_memory=False)
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data["momentum_zscore"] = pd.to_numeric(data["momentum_zscore"], errors="coerce")
    data["next_period_return"] = pd.to_numeric(data["next_period_return"], errors="coerce")

    valid = data.dropna(subset=required_columns).copy()
    method = "pearson" if metric == "ic_mean" else "spearman"
    values: list[float] = []
    for _, one_day in valid.groupby("trade_date"):
        if len(one_day) < 2:
            continue
        corr_value = one_day["momentum_zscore"].corr(one_day["next_period_return"], method=method)
        if not pd.isna(corr_value):
            values.append(float(corr_value))

    if not values:
        return MetricResult(math.nan, path, "no_valid_daily_ic", 0)
    return MetricResult(float(pd.Series(values).mean()), path, "computed_from_factor_07", len(values))


def get_ic_metric(experiment: ExperimentInfo | None, metric: str) -> MetricResult:
    if experiment is None:
        return empty_metric("missing_experiment_dir")

    summary_result = read_ic_mean_from_summary(experiment.path, metric)
    if summary_result is not None:
        return summary_result
    return compute_ic_mean_from_factor_file(experiment.path, metric)


def compute_group_final_returns(
    experiment: ExperimentInfo | None,
    group_num: int,
    return_mode: str,
) -> tuple[pd.DataFrame, str, Path | None]:
    columns = ["quantile_group", "label", "final_return", "status", "source_file"]
    if experiment is None:
        return pd.DataFrame(
            [
                {
                    "quantile_group": group_id,
                    "label": f"group{group_id}",
                    "final_return": math.nan,
                    "status": "missing_experiment_dir",
                    "source_file": "",
                }
                for group_id in range(1, group_num + 1)
            ],
            columns=columns,
        ), "missing_experiment_dir", None

    path = find_input_file(experiment.path, QUANTILE_RETURN_FILE_CANDIDATES)
    if path is None:
        return pd.DataFrame(
            [
                {
                    "quantile_group": group_id,
                    "label": f"group{group_id}",
                    "final_return": math.nan,
                    "status": "missing_quantile_return_file_09",
                    "source_file": "",
                }
                for group_id in range(1, group_num + 1)
            ],
            columns=columns,
        ), "missing_quantile_return_file_09", None

    available_columns = pd.read_csv(path, nrows=0).columns.tolist()
    required_columns = ["trade_date", "quantile_group", "group_equal_weight_return"]
    if any(col not in available_columns for col in required_columns):
        return pd.DataFrame(
            [
                {
                    "quantile_group": group_id,
                    "label": f"group{group_id}",
                    "final_return": math.nan,
                    "status": "missing_required_quantile_columns",
                    "source_file": str(path),
                }
                for group_id in range(1, group_num + 1)
            ],
            columns=columns,
        ), "missing_required_quantile_columns", path

    data = pd.read_csv(path, usecols=required_columns)
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data["quantile_group"] = pd.to_numeric(data["quantile_group"], errors="coerce")
    data["group_equal_weight_return"] = pd.to_numeric(data["group_equal_weight_return"], errors="coerce")
    data = data.dropna(subset=required_columns).sort_values("trade_date")

    rows: list[dict] = []
    for group_id in range(1, group_num + 1):
        one_group = data.loc[data["quantile_group"].eq(group_id), "group_equal_weight_return"].dropna()
        if one_group.empty:
            final_return = math.nan
            status = "missing_group_return"
        elif return_mode == "last-holding":
            final_return = float(one_group.iloc[-1])
            status = "last_holding_return"
        elif return_mode == "mean-holding":
            final_return = float(one_group.mean())
            status = "mean_holding_return"
        else:
            final_return = float((1.0 + one_group).cumprod().iloc[-1] - 1.0)
            status = "final_cumulative_return"

        rows.append(
            {
                "quantile_group": group_id,
                "label": f"group{group_id}",
                "final_return": final_return,
                "status": status,
                "source_file": str(path),
            }
        )
    return pd.DataFrame(rows, columns=columns), "ok", path


def read_group_final_nav(
    experiment: ExperimentInfo | None,
    group_num: int,
) -> tuple[pd.DataFrame, str, Path | None]:
    columns = ["quantile_group", "label", "final_nav", "status", "source_file"]
    if experiment is None:
        return pd.DataFrame(
            [
                {
                    "quantile_group": group_id,
                    "label": f"group{group_id}",
                    "final_nav": math.nan,
                    "status": "missing_experiment_dir",
                    "source_file": "",
                }
                for group_id in range(1, group_num + 1)
            ],
            columns=columns,
        ), "missing_experiment_dir", None

    path = find_input_file(experiment.path, QUANTILE_RETURN_SUMMARY_FILE_CANDIDATES)
    if path is None:
        return pd.DataFrame(
            [
                {
                    "quantile_group": group_id,
                    "label": f"group{group_id}",
                    "final_nav": math.nan,
                    "status": "missing_quantile_return_summary_13",
                    "source_file": "",
                }
                for group_id in range(1, group_num + 1)
            ],
            columns=columns,
        ), "missing_quantile_return_summary_13", None

    data = pd.read_csv(path)
    required_columns = ["quantile_group", "final_nav"]
    if any(col not in data.columns for col in required_columns):
        return pd.DataFrame(
            [
                {
                    "quantile_group": group_id,
                    "label": f"group{group_id}",
                    "final_nav": math.nan,
                    "status": "missing_required_nav_summary_columns",
                    "source_file": str(path),
                }
                for group_id in range(1, group_num + 1)
            ],
            columns=columns,
        ), "missing_required_nav_summary_columns", path

    data["quantile_group"] = pd.to_numeric(data["quantile_group"], errors="coerce")
    data["final_nav"] = pd.to_numeric(data["final_nav"], errors="coerce")
    label_values = data["label"].astype(str) if "label" in data.columns else pd.Series([""] * len(data))
    data = data.assign(label=label_values)
    data = data.dropna(subset=["quantile_group"]).copy()

    rows: list[dict] = []
    for group_id in range(1, group_num + 1):
        matched = data.loc[data["quantile_group"].eq(group_id)]
        if matched.empty:
            final_nav = math.nan
            label = f"group{group_id}"
            status = "missing_group_nav"
        else:
            final_nav = float(matched["final_nav"].iloc[0])
            raw_label = str(matched["label"].iloc[0])
            label = raw_label if raw_label and raw_label.lower() != "nan" else f"group{group_id}"
            status = "read_quantile_return_summary_13"
        rows.append(
            {
                "quantile_group": group_id,
                "label": label,
                "final_nav": final_nav,
                "status": status,
                "source_file": str(path),
            }
        )
    return pd.DataFrame(rows, columns=columns), "ok", path


def first_non_null_value(data: pd.DataFrame, column: str, default: object = None) -> object:
    if column not in data.columns:
        return default
    values = data[column].dropna()
    if values.empty:
        return default
    return values.iloc[0]


def infer_annualization_and_step(
    data: pd.DataFrame,
    experiment: ExperimentInfo,
    overlap_mode: str,
) -> tuple[float, int, str, int | None, int | None]:
    frequency = str(first_non_null_value(data, "rebalance_frequency", experiment.rebalance_frequency))
    holding_days_value = first_non_null_value(data, "holding_days", experiment.holding_days)
    holding_months_value = first_non_null_value(data, "holding_months", experiment.holding_months)

    holding_days = int(float(holding_days_value)) if holding_days_value not in [None, ""] and not pd.isna(holding_days_value) else None
    holding_months = int(float(holding_months_value)) if holding_months_value not in [None, ""] and not pd.isna(holding_months_value) else None

    if frequency == "monthly":
        months = max(int(holding_months or 1), 1)
        return float(MONTHS_PER_YEAR) / float(months), (1 if overlap_mode == "all" else months), frequency, holding_days, months

    days = max(int(holding_days or 1), 1)
    return float(TRADING_DAYS_PER_YEAR) / float(days), (1 if overlap_mode == "all" else days), "daily", days, holding_months


def calculate_group_performance_metrics(
    returns: pd.Series,
    annualization_periods_per_year: float,
) -> dict[str, float | int]:
    clean_returns = pd.to_numeric(returns, errors="coerce").dropna()
    observation_count = int(len(clean_returns))
    if observation_count == 0:
        return {
            "observation_count": 0,
            "annual_return_net": math.nan,
            "annual_volatility": math.nan,
            "sharpe_ratio": math.nan,
            "max_drawdown": math.nan,
            "win_rate": math.nan,
        }

    nav = (1.0 + clean_returns).cumprod()
    final_nav = float(nav.iloc[-1])
    annual_return = (
        final_nav ** (annualization_periods_per_year / observation_count) - 1.0
        if final_nav > 0
        else math.nan
    )
    annual_volatility = (
        float(clean_returns.std(ddof=1) * math.sqrt(annualization_periods_per_year))
        if observation_count > 1
        else math.nan
    )
    sharpe_ratio = (
        annual_return / annual_volatility
        if annual_volatility and not pd.isna(annual_volatility)
        else math.nan
    )
    max_drawdown = float((nav / nav.cummax() - 1.0).min())
    win_rate = float(clean_returns.gt(0).mean())

    return {
        "observation_count": observation_count,
        "annual_return_net": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
    }


def compute_group_performance_table(
    experiment: ExperimentInfo | None,
    group_num: int,
    overlap_mode: str,
) -> tuple[pd.DataFrame, str, Path | None]:
    columns = [
        "quantile_group",
        "label",
        "annual_return_net",
        "annual_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
        "observation_count",
        "annualization_periods_per_year",
        "rebalance_frequency",
        "holding_days",
        "holding_months",
        "overlap_mode",
        "status",
        "source_file",
    ]
    if experiment is None:
        rows = [
            {
                "quantile_group": group_id,
                "label": f"G{group_id}",
                "status": "missing_experiment_dir",
                "source_file": "",
                "overlap_mode": overlap_mode,
            }
            for group_id in range(1, group_num + 1)
        ]
        return pd.DataFrame(rows, columns=columns), "missing_experiment_dir", None

    path = find_input_file(experiment.path, QUANTILE_RETURN_FILE_CANDIDATES)
    if path is None:
        rows = [
            {
                "quantile_group": group_id,
                "label": f"G{group_id}",
                "status": "missing_quantile_return_file_09",
                "source_file": "",
                "overlap_mode": overlap_mode,
            }
            for group_id in range(1, group_num + 1)
        ]
        return pd.DataFrame(rows, columns=columns), "missing_quantile_return_file_09", None

    available_columns = pd.read_csv(path, nrows=0).columns.tolist()
    required_columns = ["trade_date", "quantile_group", "group_equal_weight_return"]
    if any(col not in available_columns for col in required_columns):
        rows = [
            {
                "quantile_group": group_id,
                "label": f"G{group_id}",
                "status": "missing_required_quantile_columns",
                "source_file": str(path),
                "overlap_mode": overlap_mode,
            }
            for group_id in range(1, group_num + 1)
        ]
        return pd.DataFrame(rows, columns=columns), "missing_required_quantile_columns", path

    optional_columns = [
        "holding_days",
        "holding_months",
        "rebalance_frequency",
        "s",
        "monthly_skip_months",
    ]
    usecols = required_columns + [col for col in optional_columns if col in available_columns]
    data = pd.read_csv(path, usecols=usecols)
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data["quantile_group"] = pd.to_numeric(data["quantile_group"], errors="coerce")
    data["group_equal_weight_return"] = pd.to_numeric(data["group_equal_weight_return"], errors="coerce")
    data = data.dropna(subset=required_columns).sort_values(["trade_date", "quantile_group"]).reset_index(drop=True)

    annualization_periods, sample_step, frequency, holding_days, holding_months = infer_annualization_and_step(
        data,
        experiment,
        overlap_mode,
    )

    rows: list[dict] = []
    for group_id in range(1, group_num + 1):
        one_group = (
            data.loc[data["quantile_group"].eq(group_id)]
            .sort_values("trade_date")
            .reset_index(drop=True)
        )
        sampled_returns = one_group["group_equal_weight_return"].iloc[::sample_step]
        metrics = calculate_group_performance_metrics(sampled_returns, annualization_periods)
        label = f"G{group_id}"
        if group_id == 1:
            label = f"{label}（最高动量）"
        elif group_id == group_num:
            label = f"{label}（最低动量）"
        rows.append(
            {
                "quantile_group": group_id,
                "label": label,
                **metrics,
                "annualization_periods_per_year": annualization_periods,
                "rebalance_frequency": frequency,
                "holding_days": holding_days,
                "holding_months": holding_months,
                "overlap_mode": overlap_mode,
                "status": "ok" if metrics["observation_count"] else "missing_group_return",
                "source_file": str(path),
            }
        )
    return pd.DataFrame(rows, columns=columns), "ok", path


# ============================================================
# 4. 绘图
# ============================================================


def annotate_points(ax: plt.Axes, x_values: list[int], y_values: pd.Series) -> None:
    for x_value, y_value in zip(x_values, y_values, strict=False):
        if pd.isna(y_value):
            continue
        ax.annotate(
            f"{y_value:.4f}",
            (x_value, y_value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )


def annotate_bars(ax: plt.Axes, values: pd.Series, percent: bool = False) -> None:
    for patch, value in zip(ax.patches, values, strict=False):
        if pd.isna(value):
            continue
        label = f"{value:.2%}" if percent else f"{value:.4f}"
        va = "bottom" if value >= 0 else "top"
        offset = 3 if value >= 0 else -3
        ax.annotate(
            label,
            (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=9,
        )


def plot_empty_figure(title: str, output_path: Path, message: str, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=dpi)
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=16)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_figure1(data: pd.DataFrame, output_path: Path, dpi: int) -> None:
    title = "图1：不同参数组合的 IC Mean"
    if not data["value"].notna().any():
        plot_empty_figure(title, output_path, "没有找到可用 IC Mean 数据", dpi)
        return

    fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=dpi)
    plot_data = data.sort_values("x_order").copy()
    x_values = plot_data["x_order"].tolist()
    ax.plot(x_values, plot_data["value"], marker="o", linewidth=2.0, color="#2f6f9f")
    ax.set_title(title, fontsize=14, pad=14)
    ax.set_xlabel("参数组合")
    ax.set_ylabel("IC Mean")
    ax.set_xticks(x_values)
    ax.set_xticklabels(plot_data["label"].tolist())
    ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle="--")
    ax.grid(alpha=0.24)
    annotate_points(ax, x_values, plot_data["value"])
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_figure2(
    data: pd.DataFrame,
    output_path: Path,
    frequency: str,
    fixed_lookback_days: int,
    fixed_lookback_months: int,
    s: int,
    metric: str,
    dpi: int,
) -> None:
    metric_label = "IC Mean" if metric == "ic_mean" else "RankIC Mean"
    if frequency == "monthly":
        title = f"Lookback Months = {fixed_lookback_months}, s = {s}"
        x_column = "holding_months"
        x_label = "Holding Months"
    else:
        title = f"Lookback Days = {fixed_lookback_days}"
        x_column = "holding_days"
        x_label = "Holding Days"

    if not data["value"].notna().any():
        plot_empty_figure(title, output_path, f"没有找到可用 {metric_label} 数据", dpi)
        return

    fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=dpi)
    plot_data = data.sort_values(x_column).copy()
    ax.bar(plot_data[x_column].astype(str), plot_data["value"], color="#579c87", width=0.58)
    ax.set_title(title, fontsize=14, pad=14)
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"因子有效期（{metric_label}）")
    ax.axhline(0.0, color="#666666", linewidth=0.8)
    ax.grid(axis="y", alpha=0.24)
    annotate_bars(ax, plot_data["value"], percent=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_figure3(data: pd.DataFrame, output_path: Path, return_mode: str, holding_label: str, dpi: int) -> None:
    if return_mode == "mean-holding":
        y_label = f"最终收益率（{holding_label}）"
    elif return_mode == "last-holding":
        y_label = f"最终收益率（{holding_label}，最后截面）"
    else:
        y_label = "最终累计收益率（跨期复利）"
    title = "各分组最终收益率"
    if not data["final_return"].notna().any():
        plot_empty_figure(title, output_path, "没有找到可用分组收益数据", dpi)
        return

    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=dpi)
    colors = ["#b95050" if group_id == 1 else "#3c76a6" if group_id == len(data) else "#8096a8" for group_id in data["quantile_group"]]
    ax.bar(data["label"], data["final_return"], color=colors, width=0.62)
    ax.set_title(title, fontsize=14, pad=14)
    ax.set_xlabel("分组数")
    ax.set_ylabel(y_label)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.axhline(0.0, color="#666666", linewidth=0.8)
    ax.grid(axis="y", alpha=0.24)
    annotate_bars(ax, data["final_return"], percent=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_figure5_final_nav(data: pd.DataFrame, output_path: Path, dpi: int) -> None:
    title = "各分组最终NAV"
    if data.empty or not data["final_nav"].notna().any():
        plot_empty_figure(title, output_path, "没有找到可用最终NAV数据", dpi)
        return

    plot_data = data.sort_values("quantile_group").copy()
    plot_data["plot_label"] = plot_data["quantile_group"].apply(lambda value: f"G{int(value)}")

    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=dpi)
    colors = [
        "#b95050" if group_id == 1 else "#3c76a6" if group_id == len(plot_data) else "#8096a8"
        for group_id in plot_data["quantile_group"]
    ]
    ax.bar(plot_data["plot_label"], plot_data["final_nav"], color=colors, width=0.62)
    ax.set_title(title, fontsize=14, pad=14)
    ax.set_xlabel("分组数")
    ax.set_ylabel("最终NAV")
    ax.axhline(1.0, color="#666666", linewidth=0.9, linestyle="--")
    ax.grid(axis="y", alpha=0.24)
    annotate_bars(ax, plot_data["final_nav"], percent=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def format_percent_value(value: object) -> str:
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value):
        return ""
    return f"{float(numeric_value):.2%}"


def format_float_value(value: object, digits: int = 3) -> str:
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value):
        return ""
    return f"{float(numeric_value):.{digits}f}"


def build_experiment_subtitle(experiment: ExperimentInfo | None, frequency: str, group_num: int) -> str:
    if experiment is not None and experiment.rebalance_frequency == "monthly":
        return (
            f"频率: monthly | lookback={experiment.lookback_months}m | "
            f"s={experiment.s or 0} | holding={experiment.holding_months}m | "
            f"group={experiment.group_num} | market={experiment.market_tag}"
        )
    if experiment is not None and experiment.rebalance_frequency == "daily":
        return (
            f"频率: daily | lookback={experiment.lookback_days}d | "
            f"holding={experiment.holding_days}d | group={experiment.group_num} | "
            f"market={experiment.market_tag}"
        )
    return f"频率: {frequency} | group={group_num}"


def plot_figure4_group_performance_table(
    data: pd.DataFrame,
    output_path: Path,
    dpi: int,
    group_num: int,
    subtitle: str = "",
) -> None:
    observed_group_num = pd.to_numeric(data.get("quantile_group", pd.Series(dtype="float64")), errors="coerce").max()
    group_num = int(observed_group_num) if pd.notna(observed_group_num) else int(group_num)
    title = f"Group1 至 Group{group_num} 分组绩效"
    metric_columns = [
        "annual_return_net",
        "annual_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
    ]
    if data.empty or not data[metric_columns].notna().any().any():
        plot_empty_figure(title, output_path, "没有找到可用分组绩效数据", dpi)
        return

    table_data = data.sort_values("quantile_group").copy()
    cell_rows = []
    for _, row in table_data.iterrows():
        cell_rows.append(
            [
                row["label"],
                format_percent_value(row["annual_return_net"]),
                format_percent_value(row["annual_volatility"]),
                format_float_value(row["sharpe_ratio"], digits=3),
                format_percent_value(row["max_drawdown"]),
                format_percent_value(row["win_rate"]),
            ]
        )

    headers = ["分组", "年化收益", "年化波动", "夏普比率", "最大回撤", "胜率"]
    fig_height = max(4.8 if subtitle else 4.4, 0.40 * (len(cell_rows) + 2) + (0.3 if subtitle else 0.0))
    fig, ax = plt.subplots(figsize=(11.5, fig_height), dpi=dpi)
    ax.axis("off")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=10 if subtitle else 4)
    if subtitle:
        ax.text(
            0.5,
            0.955,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.5,
            color="#4f5963",
        )

    table = ax.table(
        cellText=cell_rows,
        colLabels=headers,
        cellLoc="left",
        colLoc="left",
        loc="upper center",
        colWidths=[0.23, 0.19, 0.15, 0.15, 0.15, 0.13],
        bbox=[0.0, 0.01, 1.0, 0.89 if subtitle else 0.94],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.45)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#dfe5eb")
        cell.set_linewidth(0.7)
        if row_idx == 0:
            cell.set_facecolor("#f4f6f8")
            cell.set_text_props(weight="bold", color="#2f3437")
        else:
            cell.set_facecolor("#ffffff" if row_idx % 2 else "#fbfcfd")
            if col_idx in [1, 2, 3]:
                cell.set_text_props(weight="bold", color="#2f3437")
            else:
                cell.set_text_props(color="#2f3437")

    fig.tight_layout(pad=0.6)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 5. 主流程
# ============================================================


def build_ic_scan_rows(
    experiments: list[ExperimentInfo],
    lookback_days_values: list[int],
    fixed_holding_days: int,
    group_num: int,
    market_tag: str | None,
    metric: str,
    figure_name: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for lookback_days in lookback_days_values:
        experiment = select_experiment(experiments, lookback_days, fixed_holding_days, group_num, market_tag)
        result = get_ic_metric(experiment, metric)
        rows.append(
            {
                "figure": figure_name,
                "lookback_days": lookback_days,
                "holding_days": fixed_holding_days,
                "group_num": group_num,
                "market_tag": market_tag or (experiment.market_tag if experiment else ""),
                "metric": metric,
                "value": result.value,
                "observation_count": result.observation_count,
                "status": result.status,
                "experiment_dir": str(experiment.path) if experiment else "",
                "source_file": str(result.source_file) if result.source_file else "",
            }
        )
    return pd.DataFrame(rows)


def build_ic_experiment_rows(
    experiments: list[ExperimentInfo],
    experiment_specs: list[dict[str, int | str]],
    group_num: int,
    market_tag: str | None,
    metric: str,
    figure_name: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for x_order, spec in enumerate(experiment_specs, start=1):
        lookback_days = int(spec["lookback_days"])
        holding_days = int(spec["holding_days"])
        label = str(spec["label"])
        experiment = select_experiment(experiments, lookback_days, holding_days, group_num, market_tag)
        result = get_ic_metric(experiment, metric)
        rows.append(
            {
                "figure": figure_name,
                "x_order": x_order,
                "label": label,
                "lookback_days": lookback_days,
                "holding_days": holding_days,
                "group_num": group_num,
                "market_tag": market_tag or (experiment.market_tag if experiment else ""),
                "metric": metric,
                "value": result.value,
                "observation_count": result.observation_count,
                "status": result.status,
                "experiment_dir": str(experiment.path) if experiment else "",
                "source_file": str(result.source_file) if result.source_file else "",
            }
        )
    return pd.DataFrame(rows)


def build_holding_scan_rows(
    experiments: list[ExperimentInfo],
    fixed_lookback_days: int,
    holding_days_values: list[int],
    group_num: int,
    market_tag: str | None,
    metric: str,
    figure_name: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for holding_days in holding_days_values:
        experiment = select_experiment(experiments, fixed_lookback_days, holding_days, group_num, market_tag)
        result = get_ic_metric(experiment, metric)
        rows.append(
            {
                "figure": figure_name,
                "lookback_days": fixed_lookback_days,
                "holding_days": holding_days,
                "group_num": group_num,
                "market_tag": market_tag or (experiment.market_tag if experiment else ""),
                "metric": metric,
                "value": result.value,
                "observation_count": result.observation_count,
                "status": result.status,
                "experiment_dir": str(experiment.path) if experiment else "",
                "source_file": str(result.source_file) if result.source_file else "",
            }
        )
    return pd.DataFrame(rows)


def build_monthly_holding_scan_rows(
    experiments: list[ExperimentInfo],
    fixed_lookback_months: int,
    holding_months_values: list[int],
    s: int,
    group_num: int,
    market_tag: str | None,
    metric: str,
    figure_name: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for holding_months in holding_months_values:
        experiment = select_monthly_experiment(
            experiments,
            lookback_months=fixed_lookback_months,
            holding_months=holding_months,
            s=s,
            group_num=group_num,
            market_tag=market_tag,
        )
        result = get_ic_metric(experiment, metric)
        rows.append(
            {
                "figure": figure_name,
                "rebalance_frequency": "monthly",
                "lookback_days": pd.NA,
                "lookback_months": fixed_lookback_months,
                "s": s,
                "holding_days": pd.NA,
                "holding_months": holding_months,
                "group_num": group_num,
                "market_tag": market_tag or (experiment.market_tag if experiment else ""),
                "metric": metric,
                "value": result.value,
                "observation_count": result.observation_count,
                "status": result.status,
                "experiment_dir": str(experiment.path) if experiment else "",
                "source_file": str(result.source_file) if result.source_file else "",
            }
        )
    return pd.DataFrame(rows)


def fail_if_missing(strict: bool, summary: pd.DataFrame) -> None:
    if not strict:
        return
    bad = summary.loc[summary["value"].isna()]
    if bad.empty:
        return
    columns = [
        col
        for col in ["figure", "lookback_days", "lookback_months", "holding_days", "holding_months", "group_num", "status", "experiment_dir"]
        if col in bad.columns
    ]
    detail = bad[columns].to_string(index=False)
    raise SystemExit(f"存在缺失或不可用数据：\n{detail}")


def select_figure3_experiment(
    args: argparse.Namespace,
    experiments: list[ExperimentInfo],
    frequency: str,
) -> ExperimentInfo | None:
    if frequency == "monthly":
        return select_monthly_experiment(
            experiments,
            lookback_months=args.figure3_lookback_months,
            holding_months=args.figure3_holding_months,
            s=args.figure3_s,
            group_num=args.figure3_group_num,
            market_tag=args.market_tag,
        )

    return select_experiment(
        experiments,
        lookback_days=args.figure3_lookback_days,
        holding_days=args.figure3_holding_days,
        group_num=args.figure3_group_num,
        market_tag=args.market_tag,
    )


def select_figure4_experiment(
    args: argparse.Namespace,
    experiments: list[ExperimentInfo],
    frequency: str,
) -> ExperimentInfo | None:
    figure4_lookback_days = args.figure4_lookback_days or args.figure3_lookback_days
    figure4_holding_days = args.figure4_holding_days or args.figure3_holding_days

    if frequency == "monthly":
        return select_monthly_experiment(
            experiments,
            lookback_months=args.figure4_lookback_months,
            holding_months=args.figure4_holding_months,
            s=args.figure4_s,
            group_num=args.figure4_group_num,
            market_tag=args.market_tag,
        )

    return select_experiment(
        experiments,
        lookback_days=figure4_lookback_days,
        holding_days=figure4_holding_days,
        group_num=args.figure4_group_num,
        market_tag=args.market_tag,
    )


def main() -> None:
    args = parse_args()
    main_defaults = load_main_script_defaults()
    global_frequency = resolve_auto_frequency(args.rebalance_frequency, main_defaults)
    figure3_frequency = resolve_figure_frequency(args.figure3_rebalance_frequency, global_frequency)
    figure4_frequency = resolve_figure_frequency(args.figure4_rebalance_frequency, global_frequency)
    set_chinese_font()

    figure1_experiment_specs = parse_figure1_experiments(args.figure1_experiments)
    holding_days_values = parse_int_list(args.holding_days)
    holding_months_values = parse_int_list(args.holding_months)
    experiments = discover_experiments(args.output_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig1_data = build_ic_experiment_rows(
        experiments=experiments,
        experiment_specs=figure1_experiment_specs,
        group_num=args.group_num,
        market_tag=args.market_tag,
        metric="ic_mean",
        figure_name="figure1_ic_mean_by_experiment",
    )
    if global_frequency == "monthly":
        fig2_data = build_monthly_holding_scan_rows(
            experiments=experiments,
            fixed_lookback_months=args.fixed_lookback_months,
            holding_months_values=holding_months_values,
            s=args.figure2_s,
            group_num=args.group_num,
            market_tag=args.market_tag,
            metric=args.validity_metric,
            figure_name="figure2_factor_validity_by_holding",
        )
    else:
        fig2_data = build_holding_scan_rows(
            experiments=experiments,
            fixed_lookback_days=args.fixed_lookback_days,
            holding_days_values=holding_days_values,
            group_num=args.group_num,
            market_tag=args.market_tag,
            metric=args.validity_metric,
            figure_name="figure2_factor_validity_by_holding",
        )

    figure3_experiment = select_figure3_experiment(args, experiments, figure3_frequency)
    figure3_holding_label = (
        f"holding months = {args.figure3_holding_months}"
        if figure3_frequency == "monthly"
        else f"holding days = {args.figure3_holding_days}"
    )
    fig3_data, fig3_status, fig3_source = compute_group_final_returns(
        figure3_experiment,
        group_num=args.figure3_group_num,
        return_mode=args.figure3_return_mode,
    )
    fig3_summary = fig3_data.assign(
        figure="figure3_final_return_by_group",
        rebalance_frequency=figure3_frequency,
        lookback_days=args.figure3_lookback_days,
        lookback_months=args.figure3_lookback_months,
        s=args.figure3_s,
        holding_days=args.figure3_holding_days,
        holding_months=args.figure3_holding_months,
        group_num=args.figure3_group_num,
        market_tag=args.market_tag or (figure3_experiment.market_tag if figure3_experiment else ""),
        metric=args.figure3_return_mode,
        value=fig3_data["final_return"],
        observation_count=pd.NA,
        experiment_dir=str(figure3_experiment.path) if figure3_experiment else "",
        source_file=str(fig3_source) if fig3_source else "",
    )[
        [
            "figure",
            "rebalance_frequency",
            "lookback_days",
            "lookback_months",
            "s",
            "holding_days",
            "holding_months",
            "group_num",
            "market_tag",
            "metric",
            "quantile_group",
            "label",
            "value",
            "observation_count",
            "status",
            "experiment_dir",
            "source_file",
        ]
    ]
    if fig3_status != "ok":
        fig3_summary["status"] = fig3_status

    fig5_data, fig5_status, fig5_source = read_group_final_nav(
        figure3_experiment,
        group_num=args.figure3_group_num,
    )
    fig5_summary = fig5_data.assign(
        figure="figure5_final_nav_by_group",
        rebalance_frequency=figure3_frequency,
        lookback_days=args.figure3_lookback_days,
        lookback_months=args.figure3_lookback_months,
        s=args.figure3_s,
        holding_days=args.figure3_holding_days,
        holding_months=args.figure3_holding_months,
        group_num=args.figure3_group_num,
        market_tag=args.market_tag or (figure3_experiment.market_tag if figure3_experiment else ""),
        metric="final_nav",
        value=fig5_data["final_nav"],
        observation_count=pd.NA,
        experiment_dir=str(figure3_experiment.path) if figure3_experiment else "",
        source_file=str(fig5_source) if fig5_source else "",
    )[
        [
            "figure",
            "rebalance_frequency",
            "lookback_days",
            "lookback_months",
            "s",
            "holding_days",
            "holding_months",
            "group_num",
            "market_tag",
            "metric",
            "quantile_group",
            "label",
            "value",
            "observation_count",
            "status",
            "experiment_dir",
            "source_file",
        ]
    ]
    if fig5_status != "ok":
        fig5_summary["status"] = fig5_status

    figure4_experiment = select_figure4_experiment(args, experiments, figure4_frequency)
    fig4_data, fig4_status, fig4_source = compute_group_performance_table(
        figure4_experiment,
        group_num=args.figure4_group_num,
        overlap_mode=args.figure4_overlap_mode,
    )
    if fig4_status != "ok":
        fig4_data["status"] = fig4_status

    fig1_path = args.output_dir / "figure1_ic_mean_by_lookback.png"
    fig2_path = args.output_dir / "figure2_factor_validity_by_holding.png"
    fig3_path = args.output_dir / "figure3_final_return_by_group.png"
    fig4_path = args.output_dir / "figure4_group_performance_table.png"
    fig5_path = args.output_dir / "figure5_final_nav_by_group.png"
    fig4_summary_path = args.output_dir / "figure4_group_performance_summary.csv"
    summary_path = args.output_dir / "momentum_factor_figure_metrics.csv"

    summary = pd.concat(
        [
            fig1_data.assign(quantile_group=pd.NA),
            fig2_data.assign(quantile_group=pd.NA, label=pd.NA),
            fig3_summary,
            fig5_summary,
        ],
        ignore_index=True,
        sort=False,
    )
    fail_if_missing(args.strict, summary)
    write_csv(summary, summary_path)
    write_csv(fig4_data, fig4_summary_path)

    plot_figure1(fig1_data, fig1_path, dpi=args.dpi)
    plot_figure2(
        fig2_data,
        fig2_path,
        frequency=global_frequency,
        fixed_lookback_days=args.fixed_lookback_days,
        fixed_lookback_months=args.fixed_lookback_months,
        s=args.figure2_s,
        metric=args.validity_metric,
        dpi=args.dpi,
    )
    plot_figure3(
        fig3_data,
        fig3_path,
        return_mode=args.figure3_return_mode,
        holding_label=figure3_holding_label,
        dpi=args.dpi,
    )
    plot_figure5_final_nav(fig5_data, fig5_path, dpi=args.dpi)
    figure4_subtitle = build_experiment_subtitle(
        figure4_experiment,
        frequency=figure4_frequency,
        group_num=args.figure4_group_num,
    )
    plot_figure4_group_performance_table(
        fig4_data,
        fig4_path,
        dpi=args.dpi,
        group_num=args.figure4_group_num,
        subtitle=figure4_subtitle,
    )

    print("画图完成。")
    print(f"指标汇总：{summary_path}")
    print(f"分组绩效表汇总：{fig4_summary_path}")
    print(f"{fig1_path}")
    print(f"{fig2_path}")
    print(f"{fig3_path}")
    print(f"{fig4_path}")
    print(f"{fig5_path}")
    missing_columns = [
        col
        for col in ["figure", "lookback_days", "lookback_months", "holding_days", "holding_months", "group_num", "status"]
        if col in summary.columns
    ]
    missing = summary.loc[summary["value"].isna(), missing_columns]
    if not missing.empty:
        print("提示：以下组合缺少可用数据，图片中对应位置会留空：")
        print(missing.to_string(index=False))


if __name__ == "__main__":
    main()
