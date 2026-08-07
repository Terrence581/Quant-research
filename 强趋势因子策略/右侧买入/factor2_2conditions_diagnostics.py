from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd


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


CONFIG = load_sibling_module("factor2_2conditions_config", "factor2_2conditions.py")
BASE_DIAG = load_sibling_module("factor2_longonly_diagnostics_base", "factor2_LongOnly_diagnostics.py")
REFERENCE_DIAG = load_sibling_module("factor1_reference_diagnostics", "factor1_LongOnly_diagnostics.py")

DEFAULT_OUTPUT_ROOT = CONFIG.DEFAULT_OUTPUT_ROOT
DEFAULT_HIGH_LOOKBACK_DAYS = CONFIG.DEFAULT_HIGH_LOOKBACK_DAYS
DEFAULT_DRAWDOWN_LOOKBACK_DAYS = CONFIG.DEFAULT_DRAWDOWN_LOOKBACK_DAYS
DEFAULT_DRAWDOWN_THRESHOLD = CONFIG.DEFAULT_DRAWDOWN_THRESHOLD
DEFAULT_HOLDING_DAYS = CONFIG.DEFAULT_HOLDING_DAYS
DEFAULT_MARKET_TYPES = CONFIG.DEFAULT_MARKET_TYPES
DEFAULT_TRADE_DETAIL_DATE = None
TRADING_DAYS_PER_YEAR = CONFIG.TRADING_DAYS_PER_YEAR

DATE_COLUMNS = list(
    dict.fromkeys(
        CONFIG.DATE_COLUMNS
        + [
            "next_trade_date",
            "holding_start_trade_date",
            "holding_end_trade_date",
            "active_signal_trade_date_min",
            "active_signal_trade_date_max",
        ]
    )
)


def build_default_input_dir(output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    return output_root / CONFIG.build_long_only_experiment_folder_name(
        DEFAULT_HIGH_LOOKBACK_DAYS,
        DEFAULT_DRAWDOWN_LOOKBACK_DAYS,
        DEFAULT_HOLDING_DAYS,
        DEFAULT_MARKET_TYPES,
    )


DEFAULT_INPUT_DIR = build_default_input_dir(DEFAULT_OUTPUT_ROOT)


# ============================================================
# 2. 参数和文件工具
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "factor2_2conditions 诊断：检查两条件 long-only 信号池、T+1 开盘交易过滤、"
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


def numbered_path(output_dir: Path, number: int, base_name: str, suffix: str) -> Path:
    return output_dir / f"{number:02d}_{base_name}{suffix}"


def write_csv(data: pd.DataFrame, output_path: Path, index: bool = False) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=index, encoding="utf-8-sig")
    return output_path


def read_csv_with_dates(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={"stock_code": str})
    for col in DATE_COLUMNS:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce")
    return data


def first_existing_file(input_dir: Path, candidates: list[str]) -> Path:
    for name in candidates:
        path = input_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"缺少必要文件：{', '.join(candidates)}。请先运行 factor2_2conditions.py 生成完整回测输出。"
    )


def optional_file(input_dir: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        path = input_dir / name
        if path.exists():
            return path
    return None


def load_optional_summary(input_dir: Path, candidates: list[str]) -> pd.DataFrame:
    path = optional_file(input_dir, candidates)
    return read_csv_with_dates(path) if path is not None else pd.DataFrame()


def read_run_summary(input_dir: Path) -> dict[str, str]:
    path = first_existing_file(input_dir, ["run_summary.csv"])
    data = pd.read_csv(path, dtype=str)
    if not {"metric", "value"}.issubset(data.columns):
        return {}
    return dict(zip(data["metric"], data["value"]))


def load_screened_signal_pool(input_dir: Path) -> pd.DataFrame:
    return read_csv_with_dates(first_existing_file(input_dir, ["03_factor2_2conditions_long_only_signal_pool.csv"]))


def load_signal_pool(input_dir: Path) -> pd.DataFrame:
    return read_csv_with_dates(first_existing_file(input_dir, ["04_factor2_2conditions_long_only_signal_pool_with_forward_returns.csv"]))


def load_long_only_returns(input_dir: Path) -> pd.DataFrame:
    return read_csv_with_dates(first_existing_file(input_dir, ["05_factor2_2conditions_long_only_holding_period_returns.csv"]))


def load_drawdown_series(input_dir: Path) -> pd.DataFrame:
    path = optional_file(input_dir, ["07_factor2_2conditions_long_only_drawdown_series.csv"])
    return read_csv_with_dates(path) if path is not None else pd.DataFrame()


def to_int(value: object, default: int) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def parse_optional_trade_detail_date(text: str | None) -> pd.Timestamp | None:
    if text is None or str(text).strip() == "":
        return None
    parsed = pd.to_datetime(str(text).strip(), errors="coerce")
    if pd.isna(parsed):
        raise SystemExit(f"--trade-detail-date 日期格式不正确：{text}。请使用 YYYY-MM-DD。")
    return pd.Timestamp(parsed).normalize()


# ============================================================
# 3. 诊断表和图
# ============================================================


def build_daily_signal_diagnostics(long_only_returns: pd.DataFrame) -> pd.DataFrame:
    daily = long_only_returns.copy()
    limit_count_col = "next_open_limit_excluded_count" if "next_open_limit_excluded_count" in daily else "next_close_limit_excluded_count"
    daily["raw_to_tradable_ratio"] = daily.apply(
        lambda row: BASE_DIAG.safe_ratio(row.get("tradable_signal_stock_count", math.nan), row.get("raw_signal_stock_count", math.nan)),
        axis=1,
    )
    daily["next_open_limit_exclusion_ratio"] = daily.apply(
        lambda row: BASE_DIAG.safe_ratio(row.get(limit_count_col, math.nan), row.get("raw_signal_stock_count", math.nan)),
        axis=1,
    )
    daily["missing_next_record_ratio"] = daily.apply(
        lambda row: BASE_DIAG.safe_ratio(row.get("missing_next_record_count", math.nan), row.get("raw_signal_stock_count", math.nan)),
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
        "avg_breakout_return",
        "avg_rolling_max_drawdown",
        "long_only_return",
        "long_only_nav",
        "has_raw_signal",
        "has_tradable_signal",
        "has_active_position",
        "cash_reason",
    ]
    return daily[[col for col in keep_cols if col in daily.columns]].sort_values("trade_date").reset_index(drop=True)


def save_trade_detail_message_image(message: str, trade_date: pd.Timestamp, output_path: Path) -> Path:
    import matplotlib.pyplot as plt

    BASE_DIAG.set_chinese_font()
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
    signal_pool: pd.DataFrame,
    long_only_returns: pd.DataFrame,
    output_dir: Path,
) -> Path | None:
    if trade_detail_date is None:
        return None
    trade_date = pd.Timestamp(trade_detail_date).normalize()
    output_path = numbered_path(output_dir, 15, f"factor2_2conditions_long_only_trade_detail_{trade_date.strftime('%Y%m%d')}", ".png")

    calendar_dates = pd.to_datetime(long_only_returns.get("trade_date", pd.Series(dtype="datetime64[ns]")), errors="coerce").dt.normalize()
    if not calendar_dates.eq(trade_date).any():
        message = "该日期不在 long-only 回测交易日序列中，请输入正确交易日期。"
        print(message)
        return save_trade_detail_message_image(message, trade_date, output_path)

    signals = signal_pool.copy()
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
        return save_trade_detail_message_image(message, trade_date, output_path)

    display = CONFIG.sort_by_execution_date_and_stock(day_trades)
    display["买入日期"] = display["holding_start_trade_date"].dt.strftime("%Y-%m-%d")
    display["信号日期"] = display["trade_date"].dt.strftime("%Y-%m-%d")
    display["股票代码"] = display["stock_code"].astype(str)
    display["买入开盘价"] = pd.to_numeric(display.get("holding_entry_open_price", pd.Series(dtype="float64")), errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    display["组合权重"] = pd.to_numeric(display.get("long_only_weight_equal", pd.Series(dtype="float64")), errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    display["T日收盘价"] = pd.to_numeric(display.get("signal_close_price", pd.Series(dtype="float64")), errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    display["前高日期"] = pd.to_datetime(display.get("prior_high_trade_date", pd.Series(dtype="datetime64[ns]")), errors="coerce").dt.strftime("%Y-%m-%d")
    display["前高收盘价"] = pd.to_numeric(display.get("prior_high_close_price", pd.Series(dtype="float64")), errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    display["突破幅度"] = pd.to_numeric(display.get("breakout_return", pd.Series(dtype="float64")), errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    display["回撤窗口起点"] = pd.to_datetime(display.get("drawdown_window_start", pd.Series(dtype="datetime64[ns]")), errors="coerce").dt.strftime("%Y-%m-%d")
    display["最大回撤"] = pd.to_numeric(display.get("rolling_max_drawdown", pd.Series(dtype="float64")), errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    display["回撤高点日期"] = pd.to_datetime(display.get("drawdown_peak_trade_date", pd.Series(dtype="datetime64[ns]")), errors="coerce").dt.strftime("%Y-%m-%d")
    display["回撤低点日期"] = pd.to_datetime(display.get("drawdown_trough_trade_date", pd.Series(dtype="datetime64[ns]")), errors="coerce").dt.strftime("%Y-%m-%d")
    display["持有结束日期"] = pd.to_datetime(display.get("holding_end_trade_date", pd.Series(dtype="datetime64[ns]")), errors="coerce").dt.strftime("%Y-%m-%d")
    table_columns = ["买入日期", "信号日期", "股票代码", "买入开盘价", "组合权重", "T日收盘价", "前高日期", "前高收盘价", "突破幅度", "回撤窗口起点", "最大回撤", "回撤高点日期", "回撤低点日期", "持有结束日期"]
    table_data = display[table_columns]

    import matplotlib.pyplot as plt

    BASE_DIAG.set_chinese_font()
    row_count = len(table_data)
    fig_height = min(max(3.8, 1.3 + 0.38 * (row_count + 1)), 20)
    fig, ax = plt.subplots(figsize=(20.5, fig_height), dpi=170)
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
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"已输出交易明细图片：{output_path}")
    return output_path


def format_dates(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    for col in DATE_COLUMNS:
        if col in output.columns:
            output[col] = pd.to_datetime(output[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return output


# ============================================================
# 4. 主流程
# ============================================================


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    BASE_DIAG.set_chinese_font()
    trade_detail_date = parse_optional_trade_detail_date(args.trade_detail_date)

    run_summary = read_run_summary(input_dir)
    holding_days = args.holding_days if args.holding_days is not None else to_int(run_summary.get("holding_days"), DEFAULT_HOLDING_DAYS)

    print("1/5 正在读取 factor2_2conditions 回测输出...")
    screened_signal_pool = load_screened_signal_pool(input_dir)
    signal_pool = load_signal_pool(input_dir)
    long_only_returns = load_long_only_returns(input_dir)
    drawdown_series = load_drawdown_series(input_dir)
    portfolio_detail = load_optional_summary(input_dir, ["06_factor2_2conditions_long_only_portfolio_performance_detail.csv"])
    yearly_performance = load_optional_summary(input_dir, ["08_factor2_2conditions_long_only_yearly_performance.csv"])
    filter_step_summary = load_optional_summary(input_dir, ["factor2_2conditions_filter_step_summary.csv"])
    screening_reason_summary = load_optional_summary(input_dir, ["factor2_2conditions_screening_summary.csv"])

    print("2/5 正在生成信号池和交易过滤诊断...")
    signal_quality_summary = BASE_DIAG.build_signal_quality_summary(signal_pool, long_only_returns)
    trade_filter_summary = BASE_DIAG.build_trade_filter_reason_summary(signal_pool)
    daily_signal = build_daily_signal_diagnostics(long_only_returns)
    cash_reason_summary = BASE_DIAG.build_cash_reason_summary(long_only_returns)
    write_csv(signal_quality_summary, numbered_path(output_dir, 9, "factor2_2conditions_long_only_signal_quality_summary", ".csv"))
    write_csv(trade_filter_summary, numbered_path(output_dir, 9, "factor2_2conditions_long_only_trade_filter_reason_summary", ".csv"))
    write_csv(daily_signal, numbered_path(output_dir, 9, "factor2_2conditions_long_only_daily_signal_diagnostics", ".csv"))
    write_csv(cash_reason_summary, numbered_path(output_dir, 9, "factor2_2conditions_long_only_cash_reason_summary", ".csv"))
    BASE_DIAG.save_metric_table_image(
        signal_quality_summary,
        "factor2_2conditions Long-only 信号质量摘要",
        numbered_path(output_dir, 9, "factor2_2conditions_long_only_signal_quality_summary", ".png"),
    )

    print("3/5 正在生成组合绩效、NAV 和收益分布诊断...")
    nav_drawdown = BASE_DIAG.build_nav_drawdown_table(long_only_returns, drawdown_series)
    monthly_performance = BASE_DIAG.build_monthly_performance(long_only_returns)
    performance_recheck = BASE_DIAG.build_performance_recheck(long_only_returns, holding_days)
    if not performance_recheck.empty:
        performance_recheck["portfolio"] = "factor2_2conditions_long_only_recheck"
    return_distribution = BASE_DIAG.build_return_distribution(signal_pool, long_only_returns, holding_days)
    write_csv(nav_drawdown, numbered_path(output_dir, 10, "factor2_2conditions_long_only_nav_drawdown_series", ".csv"))
    write_csv(monthly_performance, numbered_path(output_dir, 10, "factor2_2conditions_long_only_monthly_performance", ".csv"))
    write_csv(performance_recheck, numbered_path(output_dir, 10, "factor2_2conditions_long_only_performance_recheck", ".csv"))
    write_csv(return_distribution, numbered_path(output_dir, 11, "factor2_2conditions_long_only_return_distribution", ".csv"))
    if not portfolio_detail.empty:
        write_csv(portfolio_detail, numbered_path(output_dir, 10, "factor2_2conditions_long_only_portfolio_performance_detail_copy", ".csv"))
    if not yearly_performance.empty:
        write_csv(yearly_performance, numbered_path(output_dir, 10, "factor2_2conditions_long_only_yearly_performance_copy", ".csv"))

    print("4/5 正在绘制 long-only 诊断图表...")
    REFERENCE_DIAG.plot_reference_set(
        signal_pool=signal_pool,
        long_only_returns=long_only_returns,
        run_summary=run_summary,
        output_dir=output_dir,
        factor_column="breakout_return",
        factor_label="factor2_2conditions",
        holding_days=holding_days,
        return_mode="sum",
        strategy_module=CONFIG,
    )
    trade_detail_image_path = output_trade_detail_image(
        trade_detail_date,
        signal_pool=signal_pool,
        long_only_returns=long_only_returns,
        output_dir=output_dir,
    )

    print("5/5 正在输出筛选摘要和诊断运行摘要...")
    write_csv(filter_step_summary, numbered_path(output_dir, 14, "factor2_2conditions_long_only_filter_step_summary", ".csv"))
    write_csv(screening_reason_summary, numbered_path(output_dir, 14, "factor2_2conditions_long_only_screening_reason_summary", ".csv"))
    diagnostic_summary = pd.DataFrame(
        [
            ["factor_name", run_summary.get("factor_name", "factor2_2conditions")],
            ["input_dir", str(input_dir)],
            ["output_dir", str(output_dir)],
            ["screened_signal_records", len(screened_signal_pool)],
            ["signal_pool_records", len(signal_pool)],
            ["long_only_return_rows", len(long_only_returns)],
            ["holding_days", holding_days],
            ["high_lookback_days", run_summary.get("high_lookback_days", DEFAULT_HIGH_LOOKBACK_DAYS)],
            ["drawdown_lookback_days", run_summary.get("drawdown_lookback_days", DEFAULT_DRAWDOWN_LOOKBACK_DAYS)],
            ["drawdown_threshold", run_summary.get("drawdown_threshold", DEFAULT_DRAWDOWN_THRESHOLD)],
            ["trade_detail_date", "" if trade_detail_date is None else trade_detail_date.strftime("%Y-%m-%d")],
            ["trade_detail_image_path", "" if trade_detail_image_path is None else str(trade_detail_image_path)],
        ],
        columns=["metric", "value"],
    )
    write_csv(diagnostic_summary, numbered_path(output_dir, 14, "factor2_2conditions_diagnostic_run_summary", ".csv"))

    print("factor2_2conditions 诊断完成。")
    print(f"输出目录：{output_dir}")
    if trade_detail_image_path is not None:
        print(f"交易明细图片：{trade_detail_image_path}")


if __name__ == "__main__":
    main()
