from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_ROOT = Path(r"D:\Filter_2_option_v2")
SELECTED_CSV = OUTPUT_ROOT / "05_select_for_fine_calibration_output" / "02d_selected_for_fine_with_rescue.csv"
FINE_DIR = OUTPUT_ROOT / "06_fine_tuning_output"
BATCH_DIR = FINE_DIR / "batches"
ANALYSIS_DIR = OUTPUT_ROOT / "07_fine_tuning_analysis_output"


def ensure_dirs() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def read_csvs(pattern: str) -> pd.DataFrame:
    frames = []
    for path in sorted(BATCH_DIR.glob(pattern)):
        if path.stat().st_size == 0:
            continue
        df = pd.read_csv(path)
        df["source_file"] = str(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def read_manifests() -> pd.DataFrame:
    rows = []
    for path in sorted(BATCH_DIR.glob("manifest_batch_*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                row = json.load(f)
            row["manifest_file"] = str(path)
            rows.append(row)
        except Exception as exc:
            rows.append({"manifest_file": str(path), "manifest_error": repr(exc)})
    return pd.DataFrame(rows)


def drawdown_to_pnl(drawdown: pd.Series, pnl: pd.Series) -> pd.Series:
    denom = pnl.abs().replace(0, np.nan)
    return drawdown.abs() / denom


def make_leaderboard(best: pd.DataFrame) -> pd.DataFrame:
    if best.empty:
        return best
    out = best.copy()
    out["validation_improvement_positive"] = out["improvement_validation"] > 0
    out["validation_positive"] = out["validation_total"] > 0
    out["calibration_positive"] = out["calibration_total"] > 0
    out["core_validation_positive"] = out["core_validation_total"] > 0
    out["validation_drawdown_to_pnl"] = drawdown_to_pnl(out["validation_max_drawdown"], out["validation_total"])
    out["core_validation_drawdown_to_pnl"] = drawdown_to_pnl(
        out["core_validation_max_drawdown"], out["core_validation_total"]
    )
    out["overfit_gap"] = out["calibration_total"] - out["validation_total"]
    out["sharpe_improvement_validation"] = out["validation_sharpe"] - out["core_validation_sharpe"]
    out["sortino_improvement_validation"] = out["validation_sortino"] - out["core_validation_sortino"]
    sort_cols = [
        "validation_positive",
        "validation_improvement_positive",
        "selection_score",
        "validation_total",
        "validation_sharpe",
        "validation_sortino",
    ]
    return out.sort_values(sort_cols, ascending=[False, False, False, False, False, False])


def group_summary(leaderboard: pd.DataFrame) -> pd.DataFrame:
    if leaderboard.empty:
        return pd.DataFrame()
    keys = ["calibration_group", "filter_group"]
    rows = []
    for key, g in leaderboard.groupby(keys, dropna=False):
        rows.append(
            {
                "calibration_group": key[0],
                "filter_group": key[1],
                "n_strategies": int(len(g)),
                "n_validation_positive": int((g["validation_total"] > 0).sum()),
                "n_improved_validation": int((g["improvement_validation"] > 0).sum()),
                "median_validation_total": float(g["validation_total"].median()),
                "mean_validation_total": float(g["validation_total"].mean()),
                "median_improvement_validation": float(g["improvement_validation"].median()),
                "median_validation_sharpe": float(g["validation_sharpe"].median()),
                "median_validation_sortino": float(g["validation_sortino"].median()),
                "median_validation_drawdown_to_pnl": float(g["validation_drawdown_to_pnl"].median()),
                "top_selection_score": float(g["selection_score"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["n_validation_positive", "median_improvement_validation", "median_validation_total"],
        ascending=[False, False, False],
    )


def batch_audit(selected: pd.DataFrame, grid: pd.DataFrame, best: pd.DataFrame, paths: pd.DataFrame, manifests: pd.DataFrame) -> pd.DataFrame:
    batch_labels = sorted(
        set(Path(p).stem.split("_")[-1] for p in BATCH_DIR.glob("*_batch_*.csv"))
        | set(Path(p).stem.split("_")[-1] for p in BATCH_DIR.glob("manifest_batch_*.json"))
    )
    rows = []
    for label in batch_labels:
        grid_file = BATCH_DIR / f"grid_batch_{label}.csv"
        best_file = BATCH_DIR / f"best_batch_{label}.csv"
        paths_file = BATCH_DIR / f"best_paths_batch_{label}.csv"
        manifest_file = BATCH_DIR / f"manifest_batch_{label}.json"
        rows.append(
            {
                "batch_label": label,
                "grid_file_exists": grid_file.exists(),
                "best_file_exists": best_file.exists(),
                "paths_file_exists": paths_file.exists(),
                "manifest_file_exists": manifest_file.exists(),
                "grid_file_mb": round(grid_file.stat().st_size / 1024**2, 3) if grid_file.exists() else np.nan,
                "best_file_mb": round(best_file.stat().st_size / 1024**2, 3) if best_file.exists() else np.nan,
                "paths_file_mb": round(paths_file.stat().st_size / 1024**2, 3) if paths_file.exists() else np.nan,
                "n_grid_rows": int((grid["source_file"].astype(str) == str(grid_file)).sum()) if not grid.empty else 0,
                "n_best_rows": int((best["source_file"].astype(str) == str(best_file)).sum()) if not best.empty else 0,
                "n_path_rows": int((paths["source_file"].astype(str) == str(paths_file)).sum()) if not paths.empty else 0,
            }
        )
    audit = pd.DataFrame(rows)
    if not manifests.empty and "batch_label" in manifests.columns:
        audit = audit.merge(
            manifests.drop(columns=[c for c in ["source_file"] if c in manifests.columns]),
            on="batch_label",
            how="left",
            suffixes=("", "_manifest"),
        )
    return audit


def run_analysis() -> dict:
    ensure_dirs()
    selected = pd.read_csv(SELECTED_CSV)
    grid = read_csvs("grid_batch_*.csv")
    best = read_csvs("best_batch_*.csv")
    paths = read_csvs("best_paths_batch_*.csv")
    manifests = read_manifests()

    selected_ids = set(selected["spec_id"].astype(int))
    best_ids = set(best["spec_id"].astype(int)) if "spec_id" in best else set()
    path_ids = set(paths["spec_id"].astype(int)) if "spec_id" in paths else set()
    missing_best = sorted(selected_ids - best_ids)
    extra_best = sorted(best_ids - selected_ids)
    missing_paths = sorted(best_ids - path_ids)

    leaderboard = make_leaderboard(best)
    summary = group_summary(leaderboard)
    audit = batch_audit(selected, grid, best, paths, manifests)

    selected.to_csv(ANALYSIS_DIR / "00_selected_input_snapshot.csv", index=False)
    audit.to_csv(ANALYSIS_DIR / "01_batch_audit.csv", index=False)
    leaderboard.to_csv(ANALYSIS_DIR / "02_fine_tuning_leaderboard.csv", index=False)
    summary.to_csv(ANALYSIS_DIR / "03_group_summary.csv", index=False)
    pd.DataFrame({"spec_id": missing_best}).to_csv(ANALYSIS_DIR / "04_missing_best_spec_ids.csv", index=False)
    pd.DataFrame({"spec_id": extra_best}).to_csv(ANALYSIS_DIR / "05_extra_best_spec_ids.csv", index=False)
    pd.DataFrame({"spec_id": missing_paths}).to_csv(ANALYSIS_DIR / "06_missing_best_path_spec_ids.csv", index=False)

    manifest = {
        "selected_file": str(SELECTED_CSV),
        "fine_output_dir": str(FINE_DIR),
        "analysis_output_dir": str(ANALYSIS_DIR),
        "n_selected": int(len(selected)),
        "n_unique_selected_spec_ids": int(len(selected_ids)),
        "n_grid_rows": int(len(grid)),
        "n_best_rows": int(len(best)),
        "n_unique_best_spec_ids": int(len(best_ids)),
        "n_best_path_rows": int(len(paths)),
        "n_unique_best_path_spec_ids": int(len(path_ids)),
        "n_missing_best_spec_ids": int(len(missing_best)),
        "n_extra_best_spec_ids": int(len(extra_best)),
        "n_missing_best_path_spec_ids": int(len(missing_paths)),
        "saved_files": [
            "00_selected_input_snapshot.csv",
            "01_batch_audit.csv",
            "02_fine_tuning_leaderboard.csv",
            "03_group_summary.csv",
            "04_missing_best_spec_ids.csv",
            "05_extra_best_spec_ids.csv",
            "06_missing_best_path_spec_ids.csv",
            "07_manifest.json",
        ],
    }
    with (ANALYSIS_DIR / "07_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run fine tuning analysis")
    args = parser.parse_args()
    if not args.run:
        print("Use --run to aggregate and audit fine tuning results.")
        return
    manifest = run_analysis()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
