from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fine_tuning_batch import metric_block, option_layer
from test_analysis import drawdown, metrics_from_path, save_plot
from test_run import load_core_paths, load_self_sufficient_core, metric_with_extras, read_protocol_dates, safe_name


OUTPUT_ROOT = Path(r"D:\Filter_2_option_v2")
OLD_TEST_DIR = OUTPUT_ROOT / "08_test_run_output"
OLD_ANALYSIS_DIR = OUTPUT_ROOT / "09_test_analysis_output"
OVERLAY_TRAIN_DIR = OUTPUT_ROOT / "11_core_put_overlay_train_output"
OUTPUT_DIR = OUTPUT_ROOT / "12_final_overlay_test_output"
PATH_DIR = OUTPUT_DIR / "paths"
PLOT_DIR = OUTPUT_DIR / "plots"

OVERLAY_RULE = {
    "drawdown_reduction_min": 0.0,
    "pnl_retention_min": 0.85,
    "require_better_drawdown_to_pnl": True,
}


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PATH_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def select_overlay_decisions() -> pd.DataFrame:
    train = pd.read_csv(OVERLAY_TRAIN_DIR / "02_put_overlay_train_best.csv")
    train["use_put_overlay"] = (
        (train["drawdown_reduction"] > OVERLAY_RULE["drawdown_reduction_min"])
        & (train["pnl_retention"] >= OVERLAY_RULE["pnl_retention_min"])
        & (train["overlay_drawdown_to_pnl"] < train["core_drawdown_to_pnl"])
    )
    return train


def load_two_option_catalog() -> pd.DataFrame:
    catalog = pd.read_csv(OLD_TEST_DIR / "02_test_strategy_paths_catalog.csv")
    return catalog[catalog["basket"].eq("two_option_improved")].copy()


def compute_core_or_overlay(row: pd.Series, test_start: pd.Timestamp, test_end: pd.Timestamp) -> tuple[dict, pd.DataFrame]:
    path = pd.read_parquet(row["path"])
    path["timestamp"] = pd.to_datetime(path["timestamp"])
    test = path[(path["timestamp"] >= test_start) & (path["timestamp"] <= test_end)].copy()
    core_net = test["net_pnl"].to_numpy(dtype=float)
    core_gross = test["gross_pnl"].to_numpy(dtype=float) if "gross_pnl" in test else core_net.copy()
    use_overlay = bool(row.get("use_put_overlay", False))
    if use_overlay:
        put_pnl, delta, strike = option_layer(
            core_net,
            "put",
            int(row["t_bars"]),
            int(row["std_span"]),
            float(row["k_shift_sigma"]),
            row["center_scale"],
        )
        final = core_net + float(row["overlay_weight"]) * put_pnl
        basket = "core_put_overlay"
        decision_type = "core_with_train_selected_put_overlay"
    else:
        put_pnl = np.zeros_like(core_net)
        delta = np.full_like(core_net, np.nan)
        strike = np.full_like(core_net, np.nan)
        final = core_net
        basket = "core_only"
        decision_type = "core_only_after_overlay_rule"
    metrics = {
        "spec_id": int(row["spec_id"]),
        "strategy_id": row["strategy_id"],
        "family": row.get("family", ""),
        "instrument": row.get("instrument", ""),
        "basket": basket,
        "filter_group": "core_only" if not use_overlay else "put_overlay",
        "decision_type": decision_type,
        "use_put_overlay": use_overlay,
        "t_bars": int(row.get("t_bars", 0)) if use_overlay else 0,
        "std_span": int(row.get("std_span", 0)) if use_overlay else 0,
        "k_shift_sigma": float(row.get("k_shift_sigma", 0.0)) if use_overlay else 0.0,
        "center_scale": row.get("center_scale", "none") if use_overlay else "none",
        "overlay_weight": float(row.get("overlay_weight", 0.0)) if use_overlay else 0.0,
        "test_n_obs": int(len(test)),
        "test_start": test["timestamp"].min() if len(test) else pd.NaT,
        "test_end": test["timestamp"].max() if len(test) else pd.NaT,
    }
    metrics.update(metric_with_extras(core_net, "core_test"))
    metrics.update(metric_with_extras(final, "test"))
    metrics["test_improvement_vs_core"] = metrics["test_total"] - metrics["core_test_total"]
    metrics["test_gross_total"] = float(core_gross[-1] - core_gross[0]) if len(core_gross) else np.nan
    metrics["test_cost_gap_core"] = metrics["test_gross_total"] - metrics["core_test_total"]
    out_path = pd.DataFrame(
        {
            "timestamp": test["timestamp"].to_numpy(),
            "spec_id": int(row["spec_id"]),
            "strategy_id": row["strategy_id"],
            "basket": basket,
            "instrument": row.get("instrument", ""),
            "core_gross_pnl": core_gross,
            "core_net_pnl": core_net,
            "put_overlay_pnl": put_pnl,
            "strategy_net_pnl": final,
            "put_delta": delta,
            "put_strike": strike,
        }
    )
    return metrics, out_path


def portfolio_from_files(catalog: pd.DataFrame) -> pd.DataFrame:
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


def make_group_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    m = metrics.copy()
    m["test_positive"] = m["test_total"] > 0
    m["improved_vs_core"] = m["test_improvement_vs_core"] > 0
    m["test_drawdown_to_pnl"] = m["test_max_drawdown"].abs() / m["test_total"].abs().replace(0, np.nan)
    return (
        m.groupby(["basket", "filter_group"], dropna=False)
        .agg(
            n_strategies=("spec_id", "count"),
            n_positive=("test_positive", "sum"),
            n_improved=("improved_vs_core", "sum"),
            total_test_pnl=("test_total", "sum"),
            median_test_pnl=("test_total", "median"),
            median_sharpe=("test_sharpe", "median"),
            median_sortino=("test_sortino", "median"),
            median_drawdown_to_pnl=("test_drawdown_to_pnl", "median"),
        )
        .reset_index()
        .sort_values("total_test_pnl", ascending=False)
    )


def old_portfolio_metrics() -> tuple[pd.DataFrame, float]:
    old = pd.read_csv(OLD_ANALYSIS_DIR / "06_portfolio_metrics.csv")
    old_total = float(old["portfolio_total"].iloc[0])
    return old, old_total


def new_portfolio_metrics(port: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame([metrics_from_path(port["portfolio_net_pnl"], "portfolio")])
    out["n_strategies"] = np.nan
    out["core_total"] = float(port["portfolio_core_net_pnl"].iloc[-1] - port["portfolio_core_net_pnl"].iloc[0])
    out["gross_total"] = float(port["portfolio_core_gross_pnl"].iloc[-1] - port["portfolio_core_gross_pnl"].iloc[0])
    out["cost_gap"] = out["gross_total"] - out["core_total"]
    return out


def plot_comparison(new_port: pd.DataFrame, old_total: float) -> list[str]:
    files = []
    old_port = pd.read_parquet(OLD_TEST_DIR / "03_test_portfolio_path.parquet")
    old_port["timestamp"] = pd.to_datetime(old_port["timestamp"])
    new_port = new_port.copy()
    new_port["timestamp"] = pd.to_datetime(new_port["timestamp"])
    plt.figure(figsize=(13, 6))
    plt.plot(old_port["timestamp"], old_port["portfolio_net_pnl"], label="old portfolio")
    plt.plot(new_port["timestamp"], new_port["portfolio_net_pnl"], label="final overlay portfolio")
    plt.axhline(old_total, linestyle="--", color="black", alpha=0.6, label="old final PnL level")
    plt.title("Final overlay portfolio vs old portfolio")
    plt.xlabel("time")
    plt.ylabel("PnL")
    plt.legend()
    p = PLOT_DIR / "01_final_overlay_vs_old_pnl.png"
    save_plot(p)
    files.append(str(p))

    plt.figure(figsize=(13, 4))
    plt.plot(old_port["timestamp"], old_port["portfolio_drawdown"], label="old drawdown")
    plt.plot(new_port["timestamp"], new_port["portfolio_drawdown"], label="final overlay drawdown")
    plt.title("Drawdown comparison")
    plt.xlabel("time")
    plt.ylabel("drawdown")
    plt.legend()
    p = PLOT_DIR / "02_final_overlay_vs_old_drawdown.png"
    save_plot(p)
    files.append(str(p))
    return files


def run_final_overlay_test() -> dict:
    ensure_dirs()
    test_start, test_end = read_protocol_dates()
    core_paths = load_core_paths()
    core = load_self_sufficient_core().merge(core_paths, on="spec_id", how="left")
    decisions = select_overlay_decisions()
    decision_cols = [
        "spec_id",
        "use_put_overlay",
        "t_bars",
        "std_span",
        "k_shift_sigma",
        "center_scale",
        "overlay_weight",
        "drawdown_reduction",
        "pnl_retention",
        "core_drawdown_to_pnl",
        "overlay_drawdown_to_pnl",
    ]
    core = core.merge(decisions[decision_cols], on="spec_id", how="left")
    core["use_put_overlay"] = core["use_put_overlay"].fillna(False).astype(bool)

    two_option_catalog = load_two_option_catalog()
    metrics_rows = []
    saved_paths = []
    errors = []

    for _, row in two_option_catalog.iterrows():
        try:
            p = Path(row["test_path"])
            path_df = pd.read_parquet(p)
            metrics = {
                "spec_id": int(row["spec_id"]),
                "strategy_id": row["strategy_id"],
                "basket": "two_option_improved",
                "filter_group": "two_option",
                "instrument": row.get("instrument", ""),
                "use_put_overlay": False,
            }
            x = path_df["strategy_net_pnl"].to_numpy(dtype=float)
            core_x = path_df["core_net_pnl"].to_numpy(dtype=float)
            metrics.update(metric_with_extras(core_x, "core_test"))
            metrics.update(metric_with_extras(x, "test"))
            metrics["test_improvement_vs_core"] = metrics["test_total"] - metrics["core_test_total"]
            metrics_rows.append(metrics)
            saved_paths.append({"spec_id": int(row["spec_id"]), "strategy_id": row["strategy_id"], "basket": "two_option_improved", "test_path": str(p)})
        except Exception as exc:
            errors.append({"spec_id": row.get("spec_id"), "strategy_id": row.get("strategy_id"), "basket": "two_option_improved", "error": repr(exc)})

    for _, row in core.iterrows():
        try:
            metrics, out_path = compute_core_or_overlay(row, test_start, test_end)
            metrics_rows.append(metrics)
            p = PATH_DIR / f"{metrics['basket']}__{int(metrics['spec_id']):05d}__{safe_name(metrics['strategy_id'])}.parquet"
            out_path.to_parquet(p, index=False)
            saved_paths.append({"spec_id": metrics["spec_id"], "strategy_id": metrics["strategy_id"], "basket": metrics["basket"], "test_path": str(p)})
        except Exception as exc:
            errors.append({"spec_id": row.get("spec_id"), "strategy_id": row.get("strategy_id"), "basket": "core", "error": repr(exc)})

    metrics = pd.DataFrame(metrics_rows)
    paths = pd.DataFrame(saved_paths)
    errors_df = pd.DataFrame(errors, columns=["spec_id", "strategy_id", "basket", "error"])
    port = portfolio_from_files(paths)
    old_metrics, old_total = old_portfolio_metrics()
    new_metrics = new_portfolio_metrics(port)
    new_metrics["n_strategies"] = len(metrics)
    comparison = pd.concat(
        [
            old_metrics.assign(portfolio_name="old_664_portfolio"),
            new_metrics.assign(portfolio_name="final_overlay_portfolio"),
        ],
        ignore_index=True,
        sort=False,
    )
    first = ["portfolio_name"]
    comparison = comparison[first + [c for c in comparison.columns if c not in first]]
    group_summary = make_group_summary(metrics)
    plots = plot_comparison(port, old_total)

    metrics.to_csv(OUTPUT_DIR / "01_final_overlay_strategy_results.csv", index=False)
    paths.to_csv(OUTPUT_DIR / "02_final_overlay_paths_catalog.csv", index=False)
    port.to_parquet(OUTPUT_DIR / "03_final_overlay_portfolio_path.parquet", index=False)
    comparison.to_csv(OUTPUT_DIR / "04_portfolio_comparison.csv", index=False)
    group_summary.to_csv(OUTPUT_DIR / "05_group_summary.csv", index=False)
    decisions.to_csv(OUTPUT_DIR / "06_overlay_train_decisions_snapshot.csv", index=False)
    errors_df.to_csv(OUTPUT_DIR / "07_errors.csv", index=False)
    manifest = {
        "test_start": str(test_start),
        "test_end": str(test_end),
        "n_strategies": int(len(metrics)),
        "n_two_option": int((metrics["basket"] == "two_option_improved").sum()),
        "n_core_put_overlay": int((metrics["basket"] == "core_put_overlay").sum()),
        "n_core_only": int((metrics["basket"] == "core_only").sum()),
        "n_errors": int(len(errors_df)),
        "old_portfolio_total": old_total,
        "new_portfolio_total": float(new_metrics["portfolio_total"].iloc[0]),
        "plots": plots,
        "output_dir": str(OUTPUT_DIR),
    }
    with (OUTPUT_DIR / "08_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        print("Use --run to build final overlay test portfolio.")
        return
    print(json.dumps(run_final_overlay_test(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
