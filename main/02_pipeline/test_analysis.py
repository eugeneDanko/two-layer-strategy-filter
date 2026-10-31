from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_ROOT = Path(r"D:\Filter_2_option_v2")
TEST_DIR = OUTPUT_ROOT / "08_test_run_output"
OUTPUT_DIR = OUTPUT_ROOT / "09_test_analysis_output"
PLOT_DIR = OUTPUT_DIR / "plots"
DIAG_DIR = OUTPUT_DIR / "diagnostic_portfolios"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)


def drawdown(x: pd.Series) -> pd.Series:
    return x - x.cummax()


def metrics_from_path(x: pd.Series, prefix: str) -> dict:
    if len(x) < 2:
        return {}
    step = x.diff().fillna(0.0)
    downside = step[step < 0]
    std = step.std()
    dstd = downside.std()
    total = float(x.iloc[-1] - x.iloc[0])
    dd = float(drawdown(x).min())
    pos = float(step[step > 0].sum())
    neg = float(step[step < 0].sum())
    return {
        f"{prefix}_total": total,
        f"{prefix}_max_drawdown": dd,
        f"{prefix}_sharpe": float(step.mean() / std * np.sqrt(44 * 252)) if std and std > 1e-12 else np.nan,
        f"{prefix}_sortino": float(step.mean() / dstd * np.sqrt(44 * 252)) if dstd and dstd > 1e-12 else np.nan,
        f"{prefix}_hit_rate": float((step > 0).mean()),
        f"{prefix}_profit_factor": float(pos / abs(neg)) if neg < 0 else np.nan,
    }


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_portfolio(port: pd.DataFrame) -> list[str]:
    files = []
    if port.empty:
        return files
    port = port.copy()
    port["timestamp"] = pd.to_datetime(port["timestamp"])

    plt.figure(figsize=(13, 6))
    plt.plot(port["timestamp"], port["portfolio_net_pnl"], label="portfolio net PnL")
    plt.plot(port["timestamp"], port["portfolio_core_net_pnl"], label="core net PnL", alpha=0.8)
    plt.plot(port["timestamp"], port["portfolio_core_gross_pnl"], label="core gross PnL", alpha=0.7)
    plt.title("Portfolio PnL")
    plt.xlabel("time")
    plt.ylabel("PnL")
    plt.legend()
    p = PLOT_DIR / "01_portfolio_pnl.png"
    save_plot(p)
    files.append(str(p))

    plt.figure(figsize=(13, 4))
    plt.fill_between(port["timestamp"], port["portfolio_drawdown"], 0, alpha=0.35)
    plt.plot(port["timestamp"], port["portfolio_drawdown"])
    plt.title("Portfolio drawdown")
    plt.xlabel("time")
    plt.ylabel("drawdown")
    p = PLOT_DIR / "02_portfolio_drawdown.png"
    save_plot(p)
    files.append(str(p))

    gross = port["portfolio_core_gross_pnl"].replace(0, np.nan)
    cost_gap = port["portfolio_core_gross_pnl"] - port["portfolio_core_net_pnl"]
    plt.figure(figsize=(13, 4))
    plt.plot(port["timestamp"], cost_gap, label="gross minus net")
    plt.title("Core cost gap")
    plt.xlabel("time")
    plt.ylabel("PnL gap")
    plt.legend()
    p = PLOT_DIR / "03_core_cost_gap.png"
    save_plot(p)
    files.append(str(p))

    final = float(port["portfolio_net_pnl"].iloc[-1] - port["portfolio_net_pnl"].iloc[0])
    max_dd = abs(float(port["portfolio_drawdown"].min()))
    risk_limits = [0.1, 0.2, 0.3]
    roc_rows = []
    for risk in risk_limits:
        capital = max_dd / risk if risk > 0 else np.nan
        roc_rows.append({"risk_limit": risk, "capital_base": capital, "return_on_capital": final / capital if capital else np.nan})
    pd.DataFrame(roc_rows).to_csv(OUTPUT_DIR / "05_return_on_capital_scenarios.csv", index=False)
    return files


def portfolio_from_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    strategy_sum = None
    core_sum = None
    gross_sum = None
    for _, row in catalog.iterrows():
        p = pd.read_parquet(row["test_path"], columns=["timestamp", "strategy_net_pnl", "core_net_pnl", "core_gross_pnl"])
        p = p.sort_values("timestamp")
        idx = pd.to_datetime(p["timestamp"])
        strategy_step = pd.Series(p["strategy_net_pnl"].diff().fillna(0.0).to_numpy(dtype=float), index=idx)
        core_step = pd.Series(p["core_net_pnl"].diff().fillna(0.0).to_numpy(dtype=float), index=idx)
        gross_step = pd.Series(p["core_gross_pnl"].diff().fillna(0.0).to_numpy(dtype=float), index=idx)
        strategy_sum = strategy_step if strategy_sum is None else strategy_sum.add(strategy_step, fill_value=0.0)
        core_sum = core_step if core_sum is None else core_sum.add(core_step, fill_value=0.0)
        gross_sum = gross_step if gross_sum is None else gross_sum.add(gross_step, fill_value=0.0)
    if strategy_sum is None:
        return pd.DataFrame()
    port = pd.DataFrame(
        {
            "timestamp": strategy_sum.index,
            "strategy_step": strategy_sum.to_numpy(dtype=float),
            "core_step": core_sum.reindex(strategy_sum.index, fill_value=0.0).to_numpy(dtype=float),
            "gross_step": gross_sum.reindex(strategy_sum.index, fill_value=0.0).to_numpy(dtype=float),
        }
    ).sort_values("timestamp")
    port["portfolio_net_pnl"] = port["strategy_step"].cumsum()
    port["portfolio_core_net_pnl"] = port["core_step"].cumsum()
    port["portfolio_core_gross_pnl"] = port["gross_step"].cumsum()
    port["portfolio_drawdown"] = drawdown(port["portfolio_net_pnl"])
    return port


def plot_diagnostic_portfolio(port: pd.DataFrame, slug: str, title: str) -> list[str]:
    files = []
    if port.empty:
        return files
    port = port.copy()
    port["timestamp"] = pd.to_datetime(port["timestamp"])
    plt.figure(figsize=(13, 5))
    plt.plot(port["timestamp"], port["portfolio_net_pnl"], label="strategy net PnL")
    plt.plot(port["timestamp"], port["portfolio_core_net_pnl"], label="core net PnL", alpha=0.75)
    plt.title(title)
    plt.xlabel("time")
    plt.ylabel("PnL")
    plt.legend()
    p = DIAG_DIR / f"{slug}_pnl.png"
    save_plot(p)
    files.append(str(p))

    plt.figure(figsize=(13, 3.5))
    plt.fill_between(port["timestamp"], port["portfolio_drawdown"], 0, alpha=0.35)
    plt.plot(port["timestamp"], port["portfolio_drawdown"])
    plt.title(f"{title} drawdown")
    plt.xlabel("time")
    plt.ylabel("drawdown")
    p = DIAG_DIR / f"{slug}_drawdown.png"
    save_plot(p)
    files.append(str(p))
    return files


def build_diagnostic_portfolios(leaderboard: pd.DataFrame, paths_catalog: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    diagnostics = [
        ("all", "All strategies", leaderboard),
        ("top_50", "Top 50 strategies by test PnL", leaderboard.head(50)),
        ("top_100", "Top 100 strategies by test PnL", leaderboard.head(100)),
        ("without_zs_rs", "All strategies without ZS/RS", leaderboard[leaderboard["instrument"] != "ZS/RS"]),
        ("two_option_only", "Two option strategies only", leaderboard[leaderboard["basket"] == "two_option_improved"]),
        ("core_only", "Self sufficient core only", leaderboard[leaderboard["basket"] == "self_sufficient_core"]),
        ("positive_only", "Positive test strategies only", leaderboard[leaderboard["test_total"] > 0]),
        (
            "robust_positive",
            "Positive strategies with drawdown below PnL",
            leaderboard[(leaderboard["test_total"] > 0) & (leaderboard["test_drawdown_to_pnl"] <= 1.0)],
        ),
    ]
    rows = []
    plot_files = []
    catalog = paths_catalog.copy()
    catalog["spec_id"] = catalog["spec_id"].astype(int)
    for slug, title, subset in diagnostics:
        ids = set(subset["spec_id"].astype(int))
        cat = catalog[catalog["spec_id"].isin(ids)].copy()
        port = portfolio_from_catalog(cat)
        if port.empty:
            rows.append({"diagnostic_portfolio": slug, "description": title, "n_strategies": 0})
            continue
        port.to_parquet(DIAG_DIR / f"{slug}_portfolio_path.parquet", index=False)
        m = metrics_from_path(port["portfolio_net_pnl"], "portfolio")
        m.update(
            {
                "diagnostic_portfolio": slug,
                "description": title,
                "n_strategies": int(len(cat)),
                "n_instruments": int(subset["instrument"].nunique()),
                "core_total": float(port["portfolio_core_net_pnl"].iloc[-1] - port["portfolio_core_net_pnl"].iloc[0]),
                "gross_total": float(port["portfolio_core_gross_pnl"].iloc[-1] - port["portfolio_core_gross_pnl"].iloc[0]),
            }
        )
        m["cost_gap"] = m["gross_total"] - m["core_total"]
        rows.append(m)
        plot_files.extend(plot_diagnostic_portfolio(port, slug, title))
    out = pd.DataFrame(rows)
    first = ["diagnostic_portfolio", "description", "n_strategies", "n_instruments"]
    rest = [c for c in out.columns if c not in first]
    return out[first + rest], plot_files


def build_tables() -> dict:
    metrics = pd.read_csv(TEST_DIR / "01_test_strategy_results.csv")
    paths_catalog = pd.read_csv(TEST_DIR / "02_test_strategy_paths_catalog.csv")
    port = pd.read_parquet(TEST_DIR / "03_test_portfolio_path.parquet")
    usage = pd.read_csv(TEST_DIR / "04_test_instruments_usage.csv")

    metrics["test_drawdown_to_pnl"] = metrics["test_max_drawdown"].abs() / metrics["test_total"].abs().replace(0, np.nan)
    metrics["core_test_drawdown_to_pnl"] = metrics["core_test_max_drawdown"].abs() / metrics["core_test_total"].abs().replace(0, np.nan)
    metrics["test_positive"] = metrics["test_total"] > 0
    metrics["improved_vs_core"] = metrics["test_improvement_vs_core"] > 0
    leaderboard = metrics.sort_values(
        ["test_positive", "improved_vs_core", "test_total", "test_sharpe", "test_sortino"],
        ascending=[False, False, False, False, False],
    )

    group_summary = (
        metrics.groupby(["basket", "calibration_group", "filter_group"], dropna=False)
        .agg(
            n_strategies=("spec_id", "count"),
            n_positive=("test_positive", "sum"),
            n_improved=("improved_vs_core", "sum"),
            median_test_total=("test_total", "median"),
            mean_test_total=("test_total", "mean"),
            median_improvement=("test_improvement_vs_core", "median"),
            median_sharpe=("test_sharpe", "median"),
            median_sortino=("test_sortino", "median"),
            median_drawdown_to_pnl=("test_drawdown_to_pnl", "median"),
        )
        .reset_index()
        .sort_values(["n_positive", "median_test_total"], ascending=[False, False])
    )

    instrument_summary = (
        metrics.groupby("instrument", dropna=False)
        .agg(
            n_strategies=("spec_id", "count"),
            total_test_pnl=("test_total", "sum"),
            median_test_pnl=("test_total", "median"),
            n_positive=("test_positive", "sum"),
        )
        .reset_index()
        .sort_values(["total_test_pnl", "n_positive"], ascending=[False, False])
    )

    port_metrics = pd.DataFrame([metrics_from_path(port["portfolio_net_pnl"], "portfolio")])
    port_metrics["n_strategies"] = len(metrics)
    port_metrics["n_instruments"] = metrics["instrument"].nunique()
    port_metrics["core_total"] = float(port["portfolio_core_net_pnl"].iloc[-1] - port["portfolio_core_net_pnl"].iloc[0]) if len(port) else np.nan
    port_metrics["gross_total"] = float(port["portfolio_core_gross_pnl"].iloc[-1] - port["portfolio_core_gross_pnl"].iloc[0]) if len(port) else np.nan
    port_metrics["cost_gap"] = port_metrics["gross_total"] - port_metrics["core_total"]

    leaderboard.to_csv(OUTPUT_DIR / "01_test_leaderboard.csv", index=False)
    group_summary.to_csv(OUTPUT_DIR / "02_test_group_summary.csv", index=False)
    instrument_summary.to_csv(OUTPUT_DIR / "03_test_instrument_summary.csv", index=False)
    usage.to_csv(OUTPUT_DIR / "04_test_instruments_usage_copy.csv", index=False)
    port_metrics.to_csv(OUTPUT_DIR / "06_portfolio_metrics.csv", index=False)
    paths_catalog.to_csv(OUTPUT_DIR / "07_test_paths_catalog_copy.csv", index=False)

    plots = plot_portfolio(port)
    diagnostic_summary, diagnostic_plots = build_diagnostic_portfolios(leaderboard, paths_catalog)
    diagnostic_summary.to_csv(OUTPUT_DIR / "09_diagnostic_portfolios.csv", index=False)
    plots.extend(diagnostic_plots)
    manifest = {
        "test_dir": str(TEST_DIR),
        "analysis_dir": str(OUTPUT_DIR),
        "n_strategies": int(len(metrics)),
        "n_positive": int((metrics["test_total"] > 0).sum()),
        "n_improved_vs_core": int((metrics["test_improvement_vs_core"] > 0).sum()),
        "n_instruments": int(metrics["instrument"].nunique()),
        "plots": plots,
        "saved_files": [
            "01_test_leaderboard.csv",
            "02_test_group_summary.csv",
            "03_test_instrument_summary.csv",
            "04_test_instruments_usage_copy.csv",
            "05_return_on_capital_scenarios.csv",
            "06_portfolio_metrics.csv",
            "07_test_paths_catalog_copy.csv",
            "08_manifest.json",
            "09_diagnostic_portfolios.csv",
        ],
    }
    with (OUTPUT_DIR / "08_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    return manifest


def run_analysis() -> dict:
    ensure_dirs()
    return build_tables()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        print("Use --run to analyze final test results.")
        return
    print(json.dumps(run_analysis(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
