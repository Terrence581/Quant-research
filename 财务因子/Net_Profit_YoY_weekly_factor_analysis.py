# -*- coding: utf-8 -*-
"""
单季度净利润同比因子(NetProfitYOY)复现——图表脚本。
读取 Net_Profit_YoY_weekly_factor.py 的输出，绘制对应研报"原始因子"的图：
- 多空组合净值（对应图18左 / 图26左的"原始因子"曲线）
- 分组净值曲线 group_1..group_10（对应图19左）
- 周频 Rank IC 柱状 + 累计 Rank IC（沿用图20 原始因子的展示形式）

运行：  D:/Anaconda/python.exe 财务因子/Net_Profit_YoY_weekly_factor_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# 与因子脚本参数区保持同一来源，避免两处手工同步。
from Net_Profit_YoY_weekly_factor import (  # noqa: E402
    OUTPUT_VARIANT_NAME as FACTOR_VARIANT_NAME,
    ROE_FILTER_ENABLED,
    ROE_MIN_PCT,
)

OUTPUT_DIR = PROJECT_DIR / "net_profit_yoy_weekly_output"

# 输出目录名与因子脚本 run() 中的命名规则一致。
if ROE_FILTER_ENABLED:
    OUTPUT_VARIANT_NAME = f"{FACTOR_VARIANT_NAME}_roe_gt_{ROE_MIN_PCT:g}"
    FILTER_TAG = f"（ROE TTM > {ROE_MIN_PCT:g}% 筛选）"
else:
    OUTPUT_VARIANT_NAME = FACTOR_VARIANT_NAME
    FILTER_TAG = ""


def set_chinese_font() -> None:
    candidates = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    available = {f.name for f in plt.matplotlib.font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_long_short_nav(ls: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ls["signal_date"], ls["long_short_nav"], color="#c0392b", lw=1.6,
            label="原始因子 多空净值")
    ax.set_title(f"单季度净利润同比因子(NetProfitYOY) 原始因子 多空组合净值（周频）{FILTER_TAG}")
    ax.set_ylabel("净值")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_group_nav(groups: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    nav_cols = sorted([c for c in groups.columns if c.endswith("_nav")],
                      key=lambda x: int(x.split("_")[1]))
    cmap = plt.cm.RdYlGn_r
    for i, c in enumerate(nav_cols):
        g = int(c.split("_")[1])
        ax.plot(groups["signal_date"], groups[c], lw=1.1,
                color=cmap(i / (len(nav_cols) - 1)), label=f"group_{g}")
    ax.set_title(f"NetProfitYOY 原始因子 分组净值{FILTER_TAG}（group_1=因子最高, group_10=因子最低）")
    ax.set_ylabel("净值")
    ax.legend(ncol=2, fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rank_ic(ic: pd.DataFrame, path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(ic["signal_date"], ic["rank_ic"], width=5, color="#5b8bd0",
            alpha=0.6, label="周频 Rank IC")
    ax1.set_ylabel("周频 Rank IC")
    ax1.axhline(0, color="black", lw=0.6)
    m = float(np.nanmax(np.abs(pd.to_numeric(ic["rank_ic"], errors="coerce")))) if len(ic) else 0.0
    if m > 0:
        ax1.set_ylim(-m * 1.18, m * 1.18)
    ax2 = ax1.twinx()
    ax2.plot(ic["signal_date"], ic["cumulative_rank_ic"], color="#c0392b",
             lw=1.6, label="累计 Rank IC（右轴）")
    ax2.set_ylabel("累计 Rank IC")
    mean_ic = ic["rank_ic"].mean()
    ax1.set_title(f"NetProfitYOY 原始因子 周频 Rank IC（mean {mean_ic*100:.2f}%）{FILTER_TAG}")
    ax1.grid(alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ============================================================
# 参照上传图样式的绩效诊断图
# 分组升序展示：Low=因子最低(group_10) … High=因子最高(group_1)；High-Low=多空。
# ============================================================
WEEKS_PER_YEAR = 52.0
SORT_LABELS = ["Low", "1", "2", "3", "4", "5", "6", "7", "8", "High"]
SORT_GROUP_COLS = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]  # 升序因子对应的组号


def _t_p(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = len(s)
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan
    mean, std = s.mean(), s.std(ddof=1)
    t = mean / (std / np.sqrt(n)) if std > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), n - 1) if np.isfinite(t) else np.nan
    return mean, std, t, p


def plot_quantile_sort_bar(groups: pd.DataFrame, path: Path) -> None:
    """排序检验柱状图：各分组前瞻一周平均收益（升序 Low..High）。"""
    means = [groups[f"group_{g}_return"].mean() * 100 for g in SORT_GROUP_COLS]
    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(range(10), means, color="#4a78a8", edgecolor="white", width=0.8)
    ax.axhline(0, color="black", lw=0.8)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + (0.004 if m >= 0 else -0.004),
                f"{m:.3f}%", ha="center", va="bottom" if m >= 0 else "top", fontsize=10)
    ax.set_xticks(range(10))
    ax.set_xticklabels(SORT_LABELS)
    ax.set_title(f"单季度净利润同比排序检验{FILTER_TAG}")
    ax.set_xlabel("单季度净利润同比因子分组")
    ax.set_ylabel("前瞻一周收益率 (%)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sort_test_table(groups: pd.DataFrame, ls: pd.DataFrame, path: Path) -> None:
    """排序检验结果表：各分组 + High-Low 的均值/标准差/t值/p值。"""
    series = [groups[f"group_{g}_return"] for g in SORT_GROUP_COLS] + [ls["long_short_return"]]
    labels = SORT_LABELS + ["High-Low"]
    means, stds, ts, ps = [], [], [], []
    for s in series:
        m, sd, t, p = _t_p(s)
        means.append(f"{m*100:.2f}")
        stds.append(f"{sd*100:.2f}")
        ts.append(f"{t:.2f}")
        ps.append(f"{p:.2f}")
    fig, ax = plt.subplots(figsize=(17, 3))
    ax.axis("off")
    table = ax.table(
        cellText=[means, stds, ts, ps],
        rowLabels=["均值（%）", "标准差（%）", "t值", "p值"],
        colLabels=labels, loc="center", cellLoc="center", rowLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        if r == 0 or c == -1:
            cell.set_text_props(weight="bold")
    ax.set_title(f"单季度净利润同比排序检验结果{FILTER_TAG}", pad=20)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_high_low_nav(ls: pd.DataFrame, path: Path) -> None:
    """High-Low 周频复利净值 + 顶部绩效指标。"""
    r = pd.to_numeric(ls["long_short_return"], errors="coerce").dropna()
    nav = pd.to_numeric(ls["long_short_nav"], errors="coerce")
    n = len(r)
    cum = nav.iloc[-1] - 1.0
    ann = nav.iloc[-1] ** (WEEKS_PER_YEAR / n) - 1.0
    vol = r.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR)
    sharpe = ann / vol if vol else 0.0
    max_dd = (nav / nav.cummax() - 1.0).min()
    win = (r > 0).mean()
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(ls["signal_date"], nav, color="#4a78a8", lw=1.4, label="High-Low NAV")
    ax.axhline(1.0, ls="--", color="grey", lw=0.8)
    fig.suptitle(f"单季度净利润同比因子 High-Low 周频复利净值走势{FILTER_TAG}", y=0.985, fontsize=13)
    fig.text(0.5, 0.925, f"累计收益 {cum*100:.2f}%   |   年化收益 {ann*100:.2f}%   |   "
             f"年化波动 {vol*100:.2f}%   |   Sharpe {sharpe:.2f}   |   "
             f"最大回撤 {max_dd*100:.2f}%   |   胜率 {win*100:.2f}%",
             ha="center", fontsize=11)
    ax.set_xlabel("时间")
    ax.set_ylabel("NAV")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ic(ic: pd.DataFrame, path: Path) -> None:
    """周频 IC 柱状 + 累计 IC 线（右轴）。"""
    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax1.bar(ic["signal_date"], ic["ic"], width=5, color="#c0733a", alpha=0.85, label="IC")
    ax1.axhline(0, color="black", lw=0.6)
    ax1.set_ylabel("IC")
    # 按 IC 幅度设对称上下限并留白，避免最下方/最上方的柱被轴边截断。
    m = float(np.nanmax(np.abs(pd.to_numeric(ic["ic"], errors="coerce")))) if len(ic) else 0.0
    if m > 0:
        ax1.set_ylim(-m * 1.18, m * 1.18)
    ax2 = ax1.twinx()
    ax2.plot(ic["signal_date"], ic["cumulative_ic"], color="#3b3b6d", lw=1.8,
             label="累计 IC（右轴）")
    ax2.set_ylabel("累计 IC")
    mean_ic = ic["ic"].mean()
    ax1.set_title(f"单季度净利润同比因子 IC（mean {mean_ic*100:.2f}%）{FILTER_TAG}")
    ax1.set_xlabel("时间")
    ax1.grid(alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    fig.legend(lines, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    set_chinese_font()
    vdir = OUTPUT_DIR / OUTPUT_VARIANT_NAME
    if not (vdir / "03_weekly_long_short.csv").exists():
        print(f"未找到 {OUTPUT_VARIANT_NAME} 输出，请先运行主脚本")
        return
    ls = pd.read_csv(vdir / "03_weekly_long_short.csv", parse_dates=["signal_date"])
    groups = pd.read_csv(vdir / "04_weekly_group_nav.csv", parse_dates=["signal_date"])
    ic = pd.read_csv(vdir / "05_weekly_rank_ic.csv", parse_dates=["signal_date"])

    plot_long_short_nav(ls, vdir / "fig1_long_short_nav.png")
    plot_group_nav(groups, vdir / "fig2_group_nav.png")
    plot_rank_ic(ic, vdir / "fig3_rank_ic.png")
    plot_quantile_sort_bar(groups, vdir / "fig4_quantile_sort_bar.png")
    plot_sort_test_table(groups, ls, vdir / "fig5_sort_test_table.png")
    plot_high_low_nav(ls, vdir / "fig6_high_low_nav.png")
    plot_ic(ic, vdir / "fig7_ic.png")
    print(f"[{OUTPUT_VARIANT_NAME}] 图已保存至：{vdir}")
    for filename in ["fig1_long_short_nav.png", "fig2_group_nav.png", "fig3_rank_ic.png",
                     "fig4_quantile_sort_bar.png", "fig5_sort_test_table.png",
                     "fig6_high_low_nav.png", "fig7_ic.png"]:
        print("  -", filename)


if __name__ == "__main__":
    main()
