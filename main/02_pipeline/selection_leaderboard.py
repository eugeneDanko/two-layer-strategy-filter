from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

OUTPUT_ROOT = Path(r"D:\Filter_2_option_v2")
FACTORY_OUTPUT = OUTPUT_ROOT / "01_core_strategy_factory_output"
FILTER_OUTPUT = OUTPUT_ROOT / "02_two_option_filter_output"
OUTPUT_DIR = OUTPUT_ROOT / "03_selection_leaderboard_output"

CORE_METRICS_CSV = FACTORY_OUTPUT / "03_core_strategy_train_metrics.csv"
FILTER_BEST_CSV = FILTER_OUTPUT / "03_two_option_best_by_strategy.csv"
FILTER_PATHS_CSV = FILTER_OUTPUT / "04_two_option_best_paths.csv"

SELECT_FRACTION = 0.60


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_div(a, b):
    if b is None or not np.isfinite(b) or abs(b) < 1e-12:
        return np.nan
    return float(a / b)


def path_metrics(path_file):
    p = pd.read_parquet(path_file)
    core = p["core_net_pnl"].to_numpy(dtype=float)
    opt = p["two_option_pnl"].to_numpy(dtype=float)
    step = np.diff(opt, prepend=opt[0])
    downside = step[step < 0]
    total = float(opt[-1] - opt[0]) if len(opt) else np.nan
    mean = float(np.nanmean(step)) if len(step) else np.nan
    std = float(np.nanstd(step)) if len(step) else np.nan
    down_std = float(np.nanstd(downside)) if len(downside) else np.nan
    sharpe = safe_div(mean, std) * np.sqrt(44 * 252) if np.isfinite(safe_div(mean, std)) else np.nan
    sortino = safe_div(mean, down_std) * np.sqrt(44 * 252) if np.isfinite(safe_div(mean, down_std)) else np.nan
    peak = np.maximum.accumulate(opt)
    dd = float(np.nanmin(opt - peak)) if len(opt) else np.nan
    core_total = float(core[-1] - core[0]) if len(core) else np.nan
    return {
        "two_option_path_total": total,
        "two_option_path_sharpe": sharpe,
        "two_option_path_sortino": sortino,
        "two_option_path_max_drawdown": dd,
        "core_path_total": core_total,
        "path_improvement_vs_core": total - core_total if np.isfinite(total) and np.isfinite(core_total) else np.nan,
    }


def load_inputs(compute_path_metrics=True):
    best = pd.read_csv(FILTER_BEST_CSV)
    core = pd.read_csv(CORE_METRICS_CSV)
    paths = pd.read_csv(FILTER_PATHS_CSV)
    keep_cols = [
        "spec_id", "strategy_id", "train_n_obs", "train_gross_total", "train_net_total",
        "train_cost_gap", "train_cost_gap_to_abs_gross", "train_max_drawdown",
        "train_trend_r2", "train_step_mean", "train_step_std", "train_half_life_bars",
        "calibration_net_total", "test_net_total",
    ]
    core = core[[c for c in keep_cols if c in core.columns]].drop_duplicates("spec_id")
    df = best.merge(core, on=["spec_id", "strategy_id"], how="left", suffixes=("", "_core"))
    df = df.merge(paths[["spec_id", "best_path"]], on="spec_id", how="left")
    if compute_path_metrics:
        rows = []
        for _, row in df.iterrows():
            try:
                rows.append(path_metrics(row["best_path"]))
            except Exception:
                rows.append({})
        df = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    return df


def add_selection_fields(df):
    df = df.copy()
    df["has_enough_data"] = df.get("train_n_obs", 0) >= 2000
    df["shape_stability_score"] = 0.0
    df.loc[df.get("train_trend_r2", 0).fillna(0) >= 0.50, "shape_stability_score"] += 1.0
    df.loc[df.get("train_half_life_bars", np.nan).notna(), "shape_stability_score"] += 1.0
    df["drawdown_to_avg_pnl"] = df.apply(
        lambda r: abs(r.get("two_option_train_max_drawdown", np.nan)) / max(abs(r.get("two_option_train_total", np.nan)), 1e-12),
        axis=1,
    )
    df["self_sufficient_core"] = (
        (df.get("train_net_total", np.nan) > 0) &
        (
            (df.get("improvement_vs_core", np.nan) <= 0) |
            (df.get("improvement_vs_core", np.nan).abs() <= 0.10 * df.get("train_net_total", np.nan).abs()) |
            (df["drawdown_to_avg_pnl"] >= 1.0)
        )
    )
    df["filter_group"] = np.where(
        df["self_sufficient_core"],
        "core_only",
        df["opt1"].astype(str) + "/" + df["opt2"].astype(str),
    )
    df["mean_reversion_priority"] = (
        df["filter_group"].isin(["call/put", "put/put"]) |
        df.get("geometry_label", "").astype(str).str.contains("mean_reversion", na=False)
    )
    df["calibration_candidate"] = (
        df["has_enough_data"] &
        (df["shape_stability_score"] > 0) &
        (
            (df.get("two_option_train_total", np.nan) > 0) |
            (df.get("improvement_vs_core", np.nan) > 0) |
            (df["self_sufficient_core"])
        )
    )
    return df


def rank_and_select(df):
    df = df.copy()
    sort_cols = [
        "has_enough_data",
        "shape_stability_score",
        "improvement_vs_core",
        "two_option_train_total",
        "two_option_train_max_drawdown",
        "two_option_path_sharpe",
        "two_option_path_sortino",
    ]
    ascending = [False, False, False, False, False, False, False]
    ranked_parts = []
    for group, part in df.groupby("filter_group", dropna=False):
        part = part.sort_values(sort_cols, ascending=ascending).copy()
        part["rank_in_group"] = np.arange(1, len(part) + 1)
        n_select = int(np.ceil(len(part) * SELECT_FRACTION))
        part["selected_top_60pct_in_group"] = part["rank_in_group"] <= n_select
        ranked_parts.append(part)
    ranked = pd.concat(ranked_parts, ignore_index=True)
    selected = ranked[ranked["selected_top_60pct_in_group"] & ranked["calibration_candidate"]].copy()
    return ranked, selected


def build_leaderboard():
    ensure_dirs()
    df = load_inputs(compute_path_metrics=True)
    df = add_selection_fields(df)
    ranked, selected = rank_and_select(df)
    ranked.to_csv(OUTPUT_DIR / "01_full_selection_leaderboard.csv", index=False)
    selected.to_csv(OUTPUT_DIR / "02_selected_for_calibration.csv", index=False)
    group_summary = ranked.groupby("filter_group").agg(
        n=("spec_id", "size"),
        selected_60pct=("selected_top_60pct_in_group", "sum"),
        calibration_candidates=("calibration_candidate", "sum"),
        median_improvement=("improvement_vs_core", "median"),
        median_two_option_pnl=("two_option_train_total", "median"),
        median_core_pnl=("core_train_total", "median"),
        median_sharpe=("two_option_path_sharpe", "median"),
        median_sortino=("two_option_path_sortino", "median"),
    ).reset_index()
    group_summary.to_csv(OUTPUT_DIR / "03_group_summary.csv", index=False)
    manifest = {
        "n_input_strategies": int(len(df)),
        "n_ranked": int(len(ranked)),
        "n_selected_for_calibration": int(len(selected)),
        "select_fraction_per_group": SELECT_FRACTION,
        "output_dir": str(OUTPUT_DIR),
    }
    (OUTPUT_DIR / "04_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return ranked, selected, group_summary, manifest


def audit():
    ensure_dirs()
    files = {
        "full_leaderboard": OUTPUT_DIR / "01_full_selection_leaderboard.csv",
        "selected": OUTPUT_DIR / "02_selected_for_calibration.csv",
        "group_summary": OUTPUT_DIR / "03_group_summary.csv",
        "manifest": OUTPUT_DIR / "04_manifest.json",
    }
    report = {k: {"exists": v.exists(), "size_bytes": v.stat().st_size if v.exists() else 0} for k, v in files.items()}
    if files["full_leaderboard"].exists():
        full = pd.read_csv(files["full_leaderboard"])
        report["full_rows"] = int(len(full))
        report["full_unique_spec_ids"] = int(full["spec_id"].nunique())
    if files["selected"].exists():
        sel = pd.read_csv(files["selected"])
        report["selected_rows"] = int(len(sel))
        report["selected_by_group"] = sel.groupby("filter_group").size().to_dict()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["build", "audit"], default="build")
    args = parser.parse_args()
    if args.mode == "build":
        build_leaderboard()
    else:
        audit()
