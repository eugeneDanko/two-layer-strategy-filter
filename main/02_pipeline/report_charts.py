from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FINAL_ROOT = Path(r"D:\Filter_2_option_v2\final_result")
OUT_DIR = FINAL_ROOT / "07_report_charts"


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_plot(path: Path) -> str:
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def barh(df: pd.DataFrame, x: str, y: str, title: str, path: Path, color: str = "#3f6c8f") -> str:
    data = df.copy()
    plt.figure(figsize=(11, max(4, 0.35 * len(data))))
    plt.barh(data[y], data[x], color=color)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel("")
    plt.gca().invert_yaxis()
    return save_plot(path)


def plot_group_pnl() -> str:
    df = pd.read_csv(FINAL_ROOT / "03_test_portfolio" / "tables" / "02_test_group_summary.csv")
    df["group"] = df["basket"].astype(str) + " | " + df["filter_group"].astype(str)
    df = df.sort_values("mean_test_total", ascending=True)
    return barh(
        df,
        "mean_test_total",
        "group",
        "Mean test PnL by strategy group",
        OUT_DIR / "01_group_mean_test_pnl.png",
        "#4c78a8",
    )


def plot_group_counts() -> str:
    df = pd.read_csv(FINAL_ROOT / "03_test_portfolio" / "tables" / "02_test_group_summary.csv")
    df["group"] = df["basket"].astype(str) + " | " + df["filter_group"].astype(str)
    df = df.sort_values("n_strategies", ascending=True)
    return barh(
        df,
        "n_strategies",
        "group",
        "Number of strategies by group",
        OUT_DIR / "02_group_strategy_counts.png",
        "#59a14f",
    )


def plot_instrument_contribution(top_n: int = 25) -> str:
    df = pd.read_csv(FINAL_ROOT / "03_test_portfolio" / "tables" / "03_test_instrument_summary.csv")
    df = df.sort_values("total_test_pnl", ascending=False).head(top_n).sort_values("total_test_pnl", ascending=True)
    return barh(
        df,
        "total_test_pnl",
        "instrument",
        f"Top {top_n} instruments by total test PnL",
        OUT_DIR / "03_top_instruments_total_pnl.png",
        "#f28e2b",
    )


def plot_negative_instruments(bottom_n: int = 20) -> str:
    df = pd.read_csv(FINAL_ROOT / "03_test_portfolio" / "tables" / "03_test_instrument_summary.csv")
    df = df.sort_values("total_test_pnl", ascending=True).head(bottom_n).sort_values("total_test_pnl", ascending=True)
    return barh(
        df,
        "total_test_pnl",
        "instrument",
        f"Bottom {bottom_n} instruments by total test PnL",
        OUT_DIR / "04_bottom_instruments_total_pnl.png",
        "#e15759",
    )


def plot_strategy_scatter() -> str:
    df = pd.read_csv(FINAL_ROOT / "03_test_portfolio" / "tables" / "01_test_leaderboard.csv")
    df["abs_drawdown"] = df["test_max_drawdown"].abs()
    colors = {"two_option_improved": "#4c78a8", "self_sufficient_core": "#f28e2b"}
    plt.figure(figsize=(10, 7))
    for basket, g in df.groupby("basket"):
        plt.scatter(g["abs_drawdown"], g["test_total"], s=18, alpha=0.65, label=basket, color=colors.get(basket, None))
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Strategy test PnL vs max drawdown")
    plt.xlabel("abs(max drawdown)")
    plt.ylabel("test PnL")
    plt.legend()
    return save_plot(OUT_DIR / "05_strategy_pnl_vs_drawdown.png")


def plot_strategy_histogram() -> str:
    df = pd.read_csv(FINAL_ROOT / "03_test_portfolio" / "tables" / "01_test_leaderboard.csv")
    plt.figure(figsize=(10, 6))
    for basket, g in df.groupby("basket"):
        plt.hist(g["test_total"], bins=40, alpha=0.55, label=basket)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Distribution of strategy test PnL")
    plt.xlabel("test PnL")
    plt.ylabel("count")
    plt.legend()
    return save_plot(OUT_DIR / "06_strategy_pnl_distribution.png")


def plot_filter_heatmap() -> str:
    df = pd.read_csv(FINAL_ROOT / "03_test_portfolio" / "tables" / "01_test_leaderboard.csv")
    pivot = pd.pivot_table(
        df,
        index="calibration_group",
        columns="filter_group",
        values="test_total",
        aggfunc="sum",
        fill_value=0,
    )
    plt.figure(figsize=(10, max(4, 0.8 * len(pivot))))
    im = plt.imshow(pivot.values, aspect="auto", cmap="RdYlGn")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(im, label="sum test PnL")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            plt.text(j, i, f"{pivot.iloc[i, j]:.1f}", ha="center", va="center", fontsize=9)
    plt.title("Filter group heatmap by total test PnL")
    return save_plot(OUT_DIR / "07_filter_group_heatmap.png")


def plot_diagnostic_bar() -> str:
    df = pd.read_csv(FINAL_ROOT / "04_diagnostic_portfolios" / "tables" / "09_diagnostic_portfolios.csv")
    df = df.sort_values("portfolio_total", ascending=True)
    return barh(
        df,
        "portfolio_total",
        "diagnostic_portfolio",
        "Diagnostic portfolios by total PnL",
        OUT_DIR / "08_diagnostic_portfolio_total_pnl.png",
        "#76b7b2",
    )


def plot_diagnostic_risk_return() -> str:
    df = pd.read_csv(FINAL_ROOT / "04_diagnostic_portfolios" / "tables" / "09_diagnostic_portfolios.csv")
    df["abs_drawdown"] = df["portfolio_max_drawdown"].abs()
    plt.figure(figsize=(9, 6))
    plt.scatter(df["abs_drawdown"], df["portfolio_total"], s=80, color="#76b7b2")
    for _, r in df.iterrows():
        plt.text(r["abs_drawdown"], r["portfolio_total"], str(r["diagnostic_portfolio"]), fontsize=8, ha="left", va="bottom")
    plt.title("Diagnostic portfolios: PnL vs drawdown")
    plt.xlabel("abs(max drawdown)")
    plt.ylabel("portfolio PnL")
    return save_plot(OUT_DIR / "09_diagnostic_risk_return.png")


def plot_capital_efficiency() -> str:
    df = pd.read_csv(FINAL_ROOT / "03_test_portfolio" / "tables" / "05_return_on_capital_scenarios.csv")
    plt.figure(figsize=(8, 5))
    plt.bar(df["risk_limit"].astype(str), df["return_on_capital"], color="#edc948")
    plt.title("Return on capital by risk limit")
    plt.xlabel("risk limit")
    plt.ylabel("return on capital")
    return save_plot(OUT_DIR / "10_return_on_capital.png")


def plot_final_overlay_comparison_bar() -> str:
    df = pd.read_csv(FINAL_ROOT / "06_final_overlay_comparison" / "tables" / "04_portfolio_comparison.csv")
    cols = ["portfolio_total", "portfolio_max_drawdown", "portfolio_sharpe", "portfolio_sortino"]
    melted = df[["portfolio_name"] + cols].melt(id_vars="portfolio_name", var_name="metric", value_name="value")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    for ax, metric in zip(axes, cols):
        part = melted[melted["metric"] == metric]
        ax.bar(part["portfolio_name"], part["value"], color=["#4c78a8", "#f28e2b"])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)
        ax.axhline(0, color="black", linewidth=0.8)
    return save_plot(OUT_DIR / "11_final_overlay_metric_comparison.png")


def run_report_charts() -> dict:
    ensure_dirs()
    plots = [
        plot_group_pnl(),
        plot_group_counts(),
        plot_instrument_contribution(),
        plot_negative_instruments(),
        plot_strategy_scatter(),
        plot_strategy_histogram(),
        plot_filter_heatmap(),
        plot_diagnostic_bar(),
        plot_diagnostic_risk_return(),
        plot_capital_efficiency(),
        plot_final_overlay_comparison_bar(),
    ]
    manifest = {"output_dir": str(OUT_DIR), "n_plots": len(plots), "plots": plots}
    with (OUT_DIR / "00_report_charts_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


if __name__ == "__main__":
    print(json.dumps(run_report_charts(), ensure_ascii=False, indent=2))
