from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



# 图表脚本默认读取同目录 output_v2 下的计算结果表，并将诊断图输出回该目录。
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_DIR / "output_v3"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output_v3"
TRADING_DAYS_PER_YEAR = 252
DEFAULT_HOLDING_DAYS = 5

# 计算脚本与绘图脚本的固定文件契约，避免混入不相关的旧因子输出。
INPUT_FILES = {
    "portfolio": "01_qtrend20_portfolio_returns.csv",
    "ic": "02_qtrend20_ic_rankic_series.csv",
    "quantile": "03_qtrend20_quantile_returns.csv",
    "validity": "04_qtrend20_factor_validity.csv",
    "return_bucket_long_only": "05_qtrend20_return_bucket_long_only.csv",
    "return_bucket_summary": "06_qtrend20_return_bucket_summary.csv",
    "g1_return20_filtered": "07_qtrend20_g1_return20_filtered_returns.csv",
    "all_market_return_bucket": "08_qtrend20_all_market_return_bucket_returns.csv",
}

# 按用户指定的五类结果固定图名和输出顺序。
FIGURE_FILES = {
    "long_short": "00_qtrend20_v3_g1_minus_g10_nav.png",
    "portfolio": "01_qtrend20_v3_g1_portfolio_nav.png",
    "ic": "02_qtrend20_v3_ic_curve.png",
    "rank_ic": "03_qtrend20_v3_rank_ic_curve.png",
    "quantile": "04_qtrend20_v3_quantile_backtest.png",
    "validity": "05_qtrend20_v3_factor_validity.png",
    "return_bucket_long_only": "06_qtrend20_v3_return_bucket_long_only_nav.png",
    "return20_0_3": "07_qtrend20_v3_return20_0_3_nav.png",
    "return20_3_5": "08_qtrend20_v3_return20_3_5_nav.png",
    "return20_5_7": "09_qtrend20_v3_return20_5_7_nav.png",
    "g1_return20_7_10": "10_qtrend20_v3_g1_return20_7_10_nav.png",
    "g1_return20_0_10": "11_qtrend20_v3_g1_return20_0_10_nav.png",
    "g1_return20_0_20": "12_qtrend20_v3_g1_return20_0_20_nav.png",
    "all_market_return_bucket": "13_qtrend20_v3_all_market_return_bucket_nav.png",
}

SELECTED_G1_RETURN20_RANGES = {
    "7_10": "7%~10%",
    "0_10": "0%~10%",
    "0_20": "0%~20%",
}

SELECTED_RETURN20_BUCKETS = {
    "return20_0_3": "0%~3%",
    "return20_3_5": "3%~5%",
    "return20_5_7": "5%~7%",
}


def parse_args() -> argparse.Namespace:
    """解析输入目录、输出目录和图片分辨率。"""
    parser = argparse.ArgumentParser(
        description="绘制 QTrend20 的组合净值、IC、RankIC、分层回测、有效期和 Return20 分区间多头走势。"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="计算结果目录；省略时读取 output_v2/holding_Nd。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="图片输出目录；省略时输出到与输入相同的持有期目录。",
    )
    parser.add_argument(
        "--holding-days",
        type=int,
        default=DEFAULT_HOLDING_DAYS,
        help=(
            "用于自动定位 holding_Nd 输入目录；默认值与计算脚本的 "
            f"DEFAULT_HOLDING_DAYS 保持一致（当前为 {DEFAULT_HOLDING_DAYS}）。"
        ),
    )
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    if args.holding_days <= 0:
        parser.error("--holding-days 必须是大于 0 的整数。")
    # 绘图脚本默认跟随持有期读取和写入对应子目录。
    default_period_dir = DEFAULT_OUTPUT_DIR / f"holding_{args.holding_days}d"
    if args.input_dir is None:
        args.input_dir = default_period_dir
    return args


def set_chinese_font() -> None:
    """依次尝试常见中文字体，确保标题、坐标轴和指标说明能正确显示。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_inputs(input_dir: Path) -> dict[str, pd.DataFrame]:
    """读取计算脚本生成的四张 CSV，并将交易日转换为日期类型。"""
    outputs: dict[str, pd.DataFrame] = {}
    for name, filename in INPUT_FILES.items():
        path = input_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"缺少计算结果：{path}")
        outputs[name] = pd.read_csv(path, encoding="utf-8-sig")
    # 统一日期 dtype，保证时间轴排序和 Matplotlib 日期刻度正确。
    for data in outputs.values():
        if "trade_date" in data.columns:
            data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    return outputs


def infer_result_holding_days(outputs: dict[str, pd.DataFrame]) -> int:
    """从回测 CSV 读取唯一持有期，避免文件夹名称和图中文字使用另一套参数。"""
    holding_values: set[int] = set()
    # 有效期表本来就包含多个前瞻期，因此只检查主回测相关结果表。
    for name in ("portfolio", "g1_return20_filtered", "ic", "quantile", "return_bucket_long_only", "all_market_return_bucket"):
        data = outputs[name]
        if "holding_days" not in data.columns:
            raise ValueError(f"{INPUT_FILES[name]} 缺少 holding_days 字段。")
        values = pd.to_numeric(data["holding_days"], errors="coerce").dropna().astype(int)
        holding_values.update(values.tolist())
    if len(holding_values) != 1:
        raise ValueError(f"回测结果中的持有期不唯一：{sorted(holding_values)}")
    holding_days = holding_values.pop()
    if holding_days <= 0:
        raise ValueError("回测结果中的 holding_days 必须大于 0。")
    return holding_days


def set_date_axis(ax: plt.Axes, dates: pd.Series) -> None:
    """根据样本跨度自动选择按月或按年显示横轴刻度。"""
    clean = pd.to_datetime(dates, errors="coerce").dropna()
    if clean.empty:
        return
    span_years = max((clean.max() - clean.min()).days / 365.25, 0)
    if span_years < 2:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    else:
        interval = 2 if span_years > 8 else 1
        ax.xaxis.set_major_locator(mdates.YearLocator(interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def performance_metrics_from_nav(nav: pd.Series, periods_per_year: float) -> dict[str, float]:
    """在净值曲线已经生成后，由同一条净值序列计算全部绩效指标。"""
    nav_values = pd.to_numeric(nav, errors="coerce").dropna().reset_index(drop=True)
    period_returns = nav_values.pct_change().dropna()
    period_count = len(period_returns)
    if period_count == 0:
        return {
            key: math.nan
            for key in (
                "cumulative_return",
                "annual_return",
                "annual_vol",
                "sharpe",
                "max_dd",
                "win_rate",
            )
        }
    cumulative_return = nav_values.iloc[-1] / nav_values.iloc[0] - 1.0
    annual_return = (nav_values.iloc[-1] / nav_values.iloc[0]) ** (
        periods_per_year / period_count
    ) - 1.0
    annual_vol = period_returns.std(ddof=1) * math.sqrt(periods_per_year)
    return_std = period_returns.std(ddof=1)
    sharpe = (
        period_returns.mean() / return_std * math.sqrt(periods_per_year)
        if return_std > 0
        else math.nan
    )
    max_dd = float((nav_values / nav_values.cummax() - 1.0).min())
    return {
        "cumulative_return": float(cumulative_return),
        "annual_return": float(annual_return),
        "annual_vol": float(annual_vol),
        "sharpe": float(sharpe),
        "max_dd": max_dd,
        "win_rate": float((period_returns > 0).mean()),
    }


def _metric_line(metrics: dict[str, float]) -> str:
    """将绩效指标格式化为放在净值图标题下方的一行摘要文本。"""
    return (
        f"累计收益 {metrics['cumulative_return']:.2%}   |   "
        f"年化收益 {metrics['annual_return']:.2%}   |   "
        f"年化波动 {metrics['annual_vol']:.2%}   |   "
        f"Sharpe {metrics['sharpe']:.2f}   |   "
        f"最大回撤 {metrics['max_dd']:.2%}   |   "
        f"胜率 {metrics['win_rate']:.2%}"
    )


def plot_portfolio_nav(data: pd.DataFrame, path: Path, dpi: int) -> None:
    """先绘制 G1 多头交易日周期复利净值，再由已绘制净值计算绩效指标。"""
    plot_data = data.sort_values("trade_date").dropna(
        subset=["trade_date", "long_g1_return"]
    ).copy()
    period_values = (
        pd.to_numeric(plot_data["periods_per_year"], errors="coerce").dropna()
        if "periods_per_year" in plot_data.columns
        else pd.Series([TRADING_DAYS_PER_YEAR])
    )
    periods_per_year = float(period_values.iloc[0])
    holding_values = pd.to_numeric(plot_data["holding_days"], errors="coerce").dropna()
    holding_days = int(holding_values.iloc[0]) if not holding_values.empty else 1
    # 第一步：由每个交易日持有周期的组合收益复利生成唯一净值，图和指标均使用该序列。
    plot_data["plotted_nav"] = (
        1.0 + pd.to_numeric(plot_data["long_g1_return"], errors="coerce")
    ).cumprod()

    # 第二步：先把组合净值曲线画到坐标轴上。
    fig, ax = plt.subplots(figsize=(14, 7.8), dpi=dpi)
    ax.plot(
        plot_data["trade_date"],
        plot_data["plotted_nav"],
        label="G1 Long NAV",
        color="#4c78a8",
        linewidth=1.45,
    )
    # 全A市场等权基准与 G1 使用同一调仓日、T+1 开盘买入和持有期末收盘卖出规则；
    # 仅不施加 QTrend/Return20 因子筛选，用于观察多头组合相对于全市场的表现。
    if {"benchmark_return", "benchmark_nav"}.issubset(plot_data.columns):
        plot_data["benchmark_plotted_nav"] = (
            1.0 + pd.to_numeric(plot_data["benchmark_return"], errors="coerce")
        ).cumprod()
        ax.plot(
            plot_data["trade_date"],
            plot_data["benchmark_plotted_nav"],
            label="全A市场等权基准 NAV",
            color="#333333",
            linestyle="--",
            linewidth=1.25,
        )
    else:
        raise ValueError("主组合结果缺少全A市场等权基准字段，请先运行计算脚本。")

    # 第三步：在曲线画出后补入初始净值 1，并由同一条净值序列计算绩效指标。
    nav_with_initial = pd.concat(
        [pd.Series([1.0]), plot_data["plotted_nav"].reset_index(drop=True)],
        ignore_index=True,
    )
    metrics = performance_metrics_from_nav(nav_with_initial, periods_per_year)
    metric_text = _metric_line(metrics)
    fig.suptitle(
        f"QTrend20 v3 因子 G1 多头每 {holding_days} 个交易日调仓复利净值走势",
        fontsize=16,
        y=0.975,
    )
    fig.text(0.5, 0.925, metric_text, ha="center", va="top", fontsize=11.5)
    ax.axhline(1.0, color="#666666", linestyle="--", linewidth=0.8)
    ax.set_xlabel("时间")
    ax.set_ylabel("NAV")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.22)
    set_date_axis(ax, plot_data["trade_date"])
    fig.subplots_adjust(top=0.80, bottom=0.11, left=0.07, right=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_long_short_nav(data: pd.DataFrame, path: Path, dpi: int) -> None:
    """绘制全截面 G1-G10 原始多空组合复利净值。"""
    plot_data = data.sort_values("trade_date").dropna(
        subset=["trade_date", "long_short_return"]
    ).copy()
    if plot_data.empty:
        raise ValueError("G1-G10 多空组合没有可绘制的收益数据。")
    periods_per_year = float(
        pd.to_numeric(plot_data["periods_per_year"], errors="coerce").dropna().iloc[0]
    )
    holding_days = int(
        pd.to_numeric(plot_data["holding_days"], errors="coerce").dropna().iloc[0]
    )

    # 先由每个调仓周期的 G1-G10 价差收益生成多空净值，再由同一序列计算绩效指标。
    plot_data["plotted_nav"] = (
        1.0 + pd.to_numeric(plot_data["long_short_return"], errors="coerce")
    ).cumprod()
    fig, ax = plt.subplots(figsize=(14, 7.8), dpi=dpi)
    ax.plot(
        plot_data["trade_date"],
        plot_data["plotted_nav"],
        label="G1-G10 Long-Short NAV",
        color="#4c78a8",
        linewidth=1.45,
    )
    nav_with_initial = pd.concat(
        [pd.Series([1.0]), plot_data["plotted_nav"].reset_index(drop=True)],
        ignore_index=True,
    )
    metric_text = _metric_line(
        performance_metrics_from_nav(nav_with_initial, periods_per_year)
    )
    fig.suptitle(
        f"QTrend20 v3 G1-G10 原始多空每 {holding_days} 个交易日调仓复利净值走势",
        fontsize=16,
        y=0.975,
    )
    fig.text(0.5, 0.925, metric_text, ha="center", va="top", fontsize=11.5)
    ax.axhline(1.0, color="#666666", linestyle="--", linewidth=0.8)
    ax.set_xlabel("时间")
    ax.set_ylabel("NAV")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.22)
    set_date_axis(ax, plot_data["trade_date"])
    fig.subplots_adjust(top=0.80, bottom=0.11, left=0.07, right=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_g1_return20_filtered_nav(
    data: pd.DataFrame,
    range_key: str,
    path: Path,
    dpi: int,
) -> None:
    """绘制指定主组合 G1 Return20 区间的复利净值，版式与主组合保持一致。"""
    plot_data = data.loc[data["range_key"].eq(range_key)].copy()
    plot_data = plot_data.sort_values("trade_date").dropna(
        subset=["trade_date", "long_g1_return"]
    )
    if plot_data.empty:
        raise ValueError("G1 的 Return20 筛选组合没有可绘制的收益数据。")

    holding_days = int(pd.to_numeric(plot_data["holding_days"], errors="coerce").dropna().iloc[0])
    periods_per_year = float(
        pd.to_numeric(plot_data["periods_per_year"], errors="coerce").dropna().iloc[0]
    )
    return20_min = float(
        pd.to_numeric(plot_data["return20_filter_min"], errors="coerce").dropna().iloc[0]
    )
    return20_max = float(
        pd.to_numeric(plot_data["return20_filter_max"], errors="coerce").dropna().iloc[0]
    )
    range_text = f"{return20_min:.0%}~{return20_max:.0%}"

    # 先按每个调仓持有期的等权收益复利得到净值；绩效指标也只使用这条净值序列。
    plot_data["plotted_nav"] = (
        1.0 + pd.to_numeric(plot_data["long_g1_return"], errors="coerce")
    ).cumprod()
    fig, ax = plt.subplots(figsize=(14, 7.8), dpi=dpi)
    ax.plot(
        plot_data["trade_date"],
        plot_data["plotted_nav"],
        label=f"G1 Return20 {range_text} Filter NAV",
        color="#4c78a8",
        linewidth=1.45,
    )

    # 先画净值曲线，再以同一曲线计算标题下方的收益、波动、夏普和回撤指标。
    nav_with_initial = pd.concat(
        [pd.Series([1.0]), plot_data["plotted_nav"].reset_index(drop=True)],
        ignore_index=True,
    )
    metric_text = _metric_line(
        performance_metrics_from_nav(nav_with_initial, periods_per_year)
    )
    fig.suptitle(
        f"QTrend20 v3 因子 G1（Return20 {range_text}）每 {holding_days} 个交易日调仓复利净值走势",
        fontsize=16,
        y=0.975,
    )
    fig.text(0.5, 0.925, metric_text, ha="center", va="top", fontsize=11.5)
    ax.axhline(1.0, color="#666666", linestyle="--", linewidth=0.8)
    ax.set_xlabel("时间")
    ax.set_ylabel("NAV")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.22)
    set_date_axis(ax, plot_data["trade_date"])
    fig.subplots_adjust(top=0.80, bottom=0.11, left=0.07, right=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_ic_with_cumulative(
    data: pd.DataFrame,
    daily_column: str,
    cumulative_column: str,
    label: str,
    path: Path,
    dpi: int,
) -> None:
    """以左轴单期柱状图、右轴累计曲线的形式绘制 IC 或 RankIC。"""
    plot_data = data.sort_values("trade_date").dropna(
        subset=["trade_date", daily_column, cumulative_column]
    )
    fig, left = plt.subplots(figsize=(14, 6.3), dpi=dpi)
    # 创建共享横轴的右侧坐标轴，避免累计 IC 与日频 IC 因量纲不同相互压缩。
    right = left.twinx()
    is_rank_ic = label == "RankIC"
    bar_color = "#8fb3e6" if is_rank_ic else "#d08a5b"
    line_color = "#c53929" if is_rank_ic else "#3b3b6d"
    holding_values = pd.to_numeric(plot_data["holding_days"], errors="coerce").dropna()
    holding_days = int(holding_values.iloc[0]) if not holding_values.empty else 1
    frequency_text = f"每 {holding_days} 个交易日"
    daily_legend = f"{frequency_text} Rank IC" if is_rank_ic else "IC"
    cumulative_legend = "累计 Rank IC（右轴）" if is_rank_ic else "累计 IC（右轴）"

    # 左轴细柱表示每个调仓周期的截面相关系数，颜色分别参照 IC/RankIC 样图。
    left.bar(
        plot_data["trade_date"],
        plot_data[daily_column],
        width=1.0,
        color=bar_color,
        alpha=0.90,
        label=daily_legend,
    )
    mean_value = plot_data[daily_column].mean()
    left.axhline(0.0, color="#555555", linewidth=0.7)
    # 右轴粗线表示从回测起点逐日累加的累计 IC。
    right.plot(
        plot_data["trade_date"],
        plot_data[cumulative_column],
        color=line_color,
        linewidth=1.65,
        label=cumulative_legend,
    )
    title_label = "Rank IC" if is_rank_ic else "IC"
    left.set_title(f"QTrend20 v3 因子{frequency_text} {title_label}（mean {mean_value:.2%}）")
    left.set_xlabel("时间")
    left.set_ylabel(f"{frequency_text} {title_label}" if is_rank_ic else "IC")
    right.set_ylabel(f"累计 {title_label}")
    left.grid(alpha=0.22)
    set_date_axis(left, plot_data["trade_date"])
    handles_left, labels_left = left.get_legend_handles_labels()
    handles_right, labels_right = right.get_legend_handles_labels()
    # IC 图图例置于底部，RankIC 图图例置左上，分别对齐参考图版式。
    legend_location = "upper left" if is_rank_ic else "upper center"
    legend_kwargs = (
        {}
        if is_rank_ic
        else {"bbox_to_anchor": (0.5, -0.13), "ncol": 2, "frameon": True}
    )
    left.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc=legend_location,
        **legend_kwargs,
    )
    fig.tight_layout(rect=(0, 0.05 if not is_rank_ic else 0, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_quantile_backtest(data: pd.DataFrame, path: Path, dpi: int) -> None:
    """按 G1（最低因子值）至 G10（最高因子值）展示分层平均前瞻收益。"""
    # 分组计算已固定为 G1 最低、G10 最高；绘图顺序必须与回测组合方向保持一致。
    group_mean = data.groupby("quantile_group")["group_return"].mean().sort_index()
    group_num = int(group_mean.index.max())
    ordered_groups = list(range(1, group_num + 1))
    values = group_mean.reindex(ordered_groups).to_numpy(dtype="float64") * 100.0
    labels = [
        "G1（最低）" if group == 1 else f"G{group_num}（最高）" if group == group_num else f"G{group}"
        for group in ordered_groups
    ]
    holding_days = int(pd.to_numeric(data["holding_days"], errors="coerce").dropna().iloc[0])

    fig, ax = plt.subplots(figsize=(14, 6.5), dpi=dpi)
    bars = ax.bar(labels, values, color="#4c78a8", width=0.78)
    span = max(float(np.nanmax(values) - np.nanmin(values)), 0.01)
    padding = span * 0.035
    # 在柱顶（负收益则柱底）写出精确的日均收益率。
    for bar, value in zip(bars, values):
        vertical_alignment = "bottom" if value >= 0 else "top"
        y = value + padding if value >= 0 else value - padding
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{value:.3f}%",
            ha="center",
            va=vertical_alignment,
            fontsize=9,
        )
    ax.margins(y=0.14)
    ax.axhline(0.0, color="#555555", linewidth=0.75)
    ax.set_title(f"QTrend20 v3 因子排序检验（每 {holding_days} 个交易日调仓）")
    ax.set_xlabel(f"QTrend20 v3 因子分组（G1 最低，G{group_num} 最高）")
    ax.set_ylabel(f"前瞻 {holding_days} 日收益率（%）")
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_factor_validity(data: pd.DataFrame, path: Path, dpi: int) -> None:
    """比较不同持有期的 IC Mean 与 RankIC Mean，以识别信号衰减速度。"""
    plot_data = data.sort_values("holding_days").copy()
    x_labels = plot_data["holding_days"].astype(int).astype(str)
    positions = np.arange(len(plot_data))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.5, 6.3), dpi=dpi)
    # 同一持有期采用成对柱状图，便于直观看出线性相关与排序相关的一致性。
    ic_bars = ax.bar(
        positions - width / 2,
        plot_data["ic_mean"] * 100.0,
        width=width,
        color="#d08a5b",
        label="IC Mean",
    )
    rank_bars = ax.bar(
        positions + width / 2,
        plot_data["rank_ic_mean"] * 100.0,
        width=width,
        color="#8fb3e6",
        label="RankIC Mean",
    )
    ax.bar_label(ic_bars, fmt="%.2f%%", padding=3, fontsize=8)
    ax.bar_label(rank_bars, fmt="%.2f%%", padding=3, fontsize=8)
    ax.margins(y=0.12)
    ax.axhline(0.0, color="#555555", linewidth=0.8)
    ax.set_xticks(positions, x_labels)
    ax.set_xlabel("持有期（交易日）")
    ax.set_ylabel("截面相关系数均值（%）")
    ax.set_title("QTrend20 v3 因子有效期")
    ax.legend()
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_return_bucket_long_only_nav(data: pd.DataFrame, path: Path, dpi: int) -> None:
    """把全截面已确定的主组合 G1 按 Return20 区间拆分并绘制独立复利净值。"""
    plot_data = data.sort_values(["return20_bucket", "trade_date"]).dropna(
        subset=["trade_date", "long_only_nav"]
    )
    fig, ax = plt.subplots(figsize=(14, 7.0), dpi=dpi)
    # 各区间均为主组合 G1 的互斥子集，并分别从 1 开始复利比较不同历史涨幅段表现。
    for bucket, one_bucket in plot_data.groupby("return20_bucket", sort=False):
        ax.plot(
            one_bucket["trade_date"],
            one_bucket["long_only_nav"],
            linewidth=1.35,
            label=str(bucket),
        )
    holding_values = pd.to_numeric(plot_data["holding_days"], errors="coerce").dropna()
    holding_days = int(holding_values.iloc[0]) if not holding_values.empty else 1
    ax.axhline(1.0, color="#666666", linestyle="--", linewidth=0.8)
    ax.set_title("主组合 G1 按不同 Return20 区间拆分的多头净值走势")
    ax.set_xlabel("时间")
    ax.set_ylabel("NAV")
    ax.legend(title="Return20 区间", loc="best", ncol=2)
    ax.grid(alpha=0.22)
    set_date_axis(ax, plot_data["trade_date"])
    ax.text(
        0.5,
        -0.16,
        f"先在全部 Return20 > 0 股票中按 QTrend 升序确定主组合 G1，再按 Return20 区间拆分；每 {holding_days} 个交易日调仓并持有 {holding_days} 日。",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_all_market_return_bucket_nav(data: pd.DataFrame, path: Path, dpi: int) -> None:
    """绘制全市场股票按 Return20 区间划分后的等权多头净值走势。"""
    plot_data = data.sort_values(["return20_bucket", "trade_date"]).dropna(
        subset=["trade_date", "long_only_nav"]
    )
    if plot_data.empty:
        raise ValueError("全市场 Return20 分桶结果没有可绘制的数据。")
    fig, ax = plt.subplots(figsize=(14, 7.0), dpi=dpi)
    # 每条线代表一个固定 Return20 区间内、当期全部合格股票的等权组合；
    # 某期没有该区间股票时计算脚本已记为 0 收益、净值保持不变，避免稀疏样本跨期直连。
    for bucket, one_bucket in plot_data.groupby("return20_bucket", sort=False):
        ax.plot(
            one_bucket["trade_date"],
            one_bucket["long_only_nav"],
            linewidth=1.35,
            label=str(bucket),
        )
    holding_days = int(
        pd.to_numeric(plot_data["holding_days"], errors="coerce").dropna().iloc[0]
    )
    ax.axhline(1.0, color="#666666", linestyle="--", linewidth=0.8)
    ax.set_title("全市场股票按不同 Return20 区间划分的等权多头净值走势")
    ax.set_xlabel("时间")
    ax.set_ylabel("NAV")
    ax.legend(title="Return20 区间", loc="best", ncol=2)
    ax.grid(alpha=0.22)
    set_date_axis(ax, plot_data["trade_date"])
    ax.text(
        0.5,
        -0.16,
        f"每个调仓截面使用全部可交易且 Return20 > 0 的股票，按 Return20 区间分桶并在桶内等权；每 {holding_days} 个交易日调仓并持有 {holding_days} 日。",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_single_return20_bucket_nav(
    data: pd.DataFrame,
    bucket_label: str,
    path: Path,
    dpi: int,
) -> None:
    """按原组合净值图版式，单独绘制一个 Return20 区间的 G1 多头净值。"""
    plot_data = data.loc[data["return20_bucket"].eq(bucket_label)].copy()
    plot_data = plot_data.sort_values("trade_date").dropna(
        subset=["trade_date", "long_only_return"]
    )
    if plot_data.empty:
        raise ValueError(f"Return20 区间 {bucket_label} 没有可绘制的组合收益。")

    holding_days = int(pd.to_numeric(plot_data["holding_days"], errors="coerce").dropna().iloc[0])
    periods_per_year = TRADING_DAYS_PER_YEAR / holding_days

    # 先使用该区间每期 G1 收益生成净值，并画出与主组合相同的净值曲线。
    plot_data["plotted_nav"] = (
        1.0 + pd.to_numeric(plot_data["long_only_return"], errors="coerce")
    ).cumprod()
    fig, ax = plt.subplots(figsize=(14, 7.8), dpi=dpi)
    ax.plot(
        plot_data["trade_date"],
        plot_data["plotted_nav"],
        label=f"Main G1 - Return20 {bucket_label} NAV",
        color="#4c78a8",
        linewidth=1.45,
    )

    # 净值画出后，再由同一条净值序列计算标题下方的绩效指标。
    nav_with_initial = pd.concat(
        [pd.Series([1.0]), plot_data["plotted_nav"].reset_index(drop=True)],
        ignore_index=True,
    )
    metric_text = _metric_line(
        performance_metrics_from_nav(nav_with_initial, periods_per_year)
    )
    fig.suptitle(
        f"QTrend20 v3 主组合 G1（Return20 {bucket_label}）每 {holding_days} 个交易日调仓复利净值走势",
        fontsize=16,
        y=0.975,
    )
    fig.text(0.5, 0.925, metric_text, ha="center", va="top", fontsize=11.5)
    ax.axhline(1.0, color="#666666", linestyle="--", linewidth=0.8)
    ax.set_xlabel("时间")
    ax.set_ylabel("NAV")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.22)
    set_date_axis(ax, plot_data["trade_date"])
    fig.subplots_adjust(top=0.80, bottom=0.11, left=0.07, right=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """命令行入口：加载结果表，依序生成主回测和 Return20 分区间诊断图。"""
    args = parse_args()
    set_chinese_font()
    data = load_inputs(args.input_dir)
    # 输出目录以 CSV 中实际参与回测的持有期为准，而不是依赖绘图脚本的默认参数。
    actual_holding_days = infer_result_holding_days(data)
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_DIR / f"holding_{actual_holding_days}d"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 0. 原始 G1-G10 多空净值；1. G1 多头净值；2. IC；3. RankIC；4. 分层排序；
    # 5. 因子有效期；6. 主组合 G1 的 Return20 区间拆分；7. G1 筛选区间净值。
    plot_long_short_nav(
        data["portfolio"],
        args.output_dir / FIGURE_FILES["long_short"],
        args.dpi,
    )
    plot_portfolio_nav(
        data["portfolio"],
        args.output_dir / FIGURE_FILES["portfolio"],
        args.dpi,
    )
    plot_ic_with_cumulative(
        data["ic"],
        daily_column="ic",
        cumulative_column="cumulative_ic",
        label="IC",
        path=args.output_dir / FIGURE_FILES["ic"],
        dpi=args.dpi,
    )
    plot_ic_with_cumulative(
        data["ic"],
        daily_column="rank_ic",
        cumulative_column="cumulative_rank_ic",
        label="RankIC",
        path=args.output_dir / FIGURE_FILES["rank_ic"],
        dpi=args.dpi,
    )
    plot_quantile_backtest(
        data["quantile"],
        args.output_dir / FIGURE_FILES["quantile"],
        args.dpi,
    )
    plot_factor_validity(
        data["validity"],
        args.output_dir / FIGURE_FILES["validity"],
        args.dpi,
    )
    plot_return_bucket_long_only_nav(
        data["return_bucket_long_only"],
        args.output_dir / FIGURE_FILES["return_bucket_long_only"],
        args.dpi,
    )
    # 7. 全市场按 Return20 分桶的等权多头净值；与上图的“主组合 G1 分桶”口径独立。
    plot_all_market_return_bucket_nav(
        data["all_market_return_bucket"],
        args.output_dir / FIGURE_FILES["all_market_return_bucket"],
        args.dpi,
    )
    # 7-9. 在已确定的主组合 G1 内分别按配置区间筛选，不重新分组。
    range_figure_keys = {
        "7_10": "g1_return20_7_10",
        "0_10": "g1_return20_0_10",
        "0_20": "g1_return20_0_20",
    }
    for range_key, figure_key in range_figure_keys.items():
        plot_g1_return20_filtered_nav(
            data["g1_return20_filtered"],
            range_key=range_key,
            path=args.output_dir / FIGURE_FILES[figure_key],
            dpi=args.dpi,
        )
    for figure_key, bucket_label in SELECTED_RETURN20_BUCKETS.items():
        plot_single_return20_bucket_nav(
            data["return_bucket_long_only"],
            bucket_label=bucket_label,
            path=args.output_dir / FIGURE_FILES[figure_key],
            dpi=args.dpi,
        )

    print(f"QTrend20 v3 G1 多头诊断图完成；实际持有期：{actual_holding_days} 个交易日。")
    for filename in FIGURE_FILES.values():
        print(args.output_dir / filename)


if __name__ == "__main__":
    main()
