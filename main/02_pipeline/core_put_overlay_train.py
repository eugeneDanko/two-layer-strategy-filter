from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fine_tuning_batch import metric_block, option_layer


OUTPUT_ROOT = Path(r"D:\Filter_2_option_v2")
WORK_ROOT = Path(r"./workspace")
PROTOCOL_PATH = WORK_ROOT / "backtest_protocol.yaml"
FULL_SELECTION_CSV = OUTPUT_ROOT / "03_selection_leaderboard_output" / "01_full_selection_leaderboard.csv"
CORE_PATHS_CSV = OUTPUT_ROOT / "01_core_strategy_factory_output" / "06_all_core_strategy_paths.csv"
OUTPUT_DIR = OUTPUT_ROOT / "11_core_put_overlay_train_output"
BATCH_DIR = OUTPUT_DIR / "batches"
BEST_PATH_DIR = OUTPUT_DIR / "best_paths"
DEFAULT_BATCH_SIZE = 100

T_GRID = [44, 220, 968]
STD_GRID = [44, 88]
K_GRID = [0.5, 1.0, 1.5]
CENTER_MODES = ["period_start", "median_mad"]
WEIGHT_GRID = [0.25, 0.5, 1.0, 1.5]


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    BEST_PATH_DIR.mkdir(parents=True, exist_ok=True)


def read_train_dates() -> tuple[pd.Timestamp, pd.Timestamp]:
    text = PROTOCOL_PATH.read_text(encoding="utf-8")
    start = "2008-05-01"
    end = "2016-12-31 23:59:59"
    in_train = False
    for raw in text.splitlines():
        line = raw.strip()
        if line == "train:":
            in_train = True
            continue
        if in_train and line.startswith(("calibration:", "test:")):
            in_train = False
        if in_train and line.startswith("start:"):
            start = line.split(":", 1)[1].strip().strip('"').strip("'")
        if in_train and line.startswith("end:"):
            end = line.split(":", 1)[1].strip().strip('"').strip("'")
    return pd.Timestamp(start), pd.Timestamp(end)


def load_core_candidates() -> pd.DataFrame:
    full = pd.read_csv(FULL_SELECTION_CSV)
    core = full[full["self_sufficient_core"].astype(bool)].copy()
    core = core.sort_values("selection_score", ascending=False).drop_duplicates("spec_id")
    paths = pd.read_csv(CORE_PATHS_CSV)[["spec_id", "path"]]
    paths["spec_id"] = paths["spec_id"].astype(int)
    core["spec_id"] = core["spec_id"].astype(int)
    return core.merge(paths, on="spec_id", how="left")


def drawdown_ratio(total: float, max_drawdown: float) -> float:
    return float(abs(max_drawdown) / max(abs(total), 1e-12)) if np.isfinite(total) and np.isfinite(max_drawdown) else np.nan


def overlay_score(core_m: dict, overlay_m: dict, weight: float) -> float:
    core_total = core_m["core_train_total"]
    overlay_total = overlay_m["overlay_train_total"]
    dd_reduction = abs(core_m["core_train_max_drawdown"]) - abs(overlay_m["overlay_train_max_drawdown"])
    pnl_retention = overlay_total / max(abs(core_total), 1e-12)
    improvement = overlay_total - core_total
    sharpe_gain = np.nan_to_num(overlay_m["overlay_train_sharpe"]) - np.nan_to_num(core_m["core_train_sharpe"])
    sortino_gain = np.nan_to_num(overlay_m["overlay_train_sortino"]) - np.nan_to_num(core_m["core_train_sortino"])
    penalty = 0.15 * abs(weight)
    return float(2.0 * dd_reduction + 0.7 * improvement + 0.5 * sharpe_gain + 0.5 * sortino_gain + 0.4 * pnl_retention - penalty)


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))[:170]


def evaluate_one(row: pd.Series, train_start: pd.Timestamp, train_end: pd.Timestamp) -> tuple[list[dict], dict | None, pd.DataFrame | None]:
    path = pd.read_parquet(row["path"])
    path["timestamp"] = pd.to_datetime(path["timestamp"])
    train = path[(path["timestamp"] >= train_start) & (path["timestamp"] <= train_end)].copy()
    if len(train) < 200:
        return [], None, None
    core = train["net_pnl"].to_numpy(dtype=float)
    gross = train["gross_pnl"].to_numpy(dtype=float) if "gross_pnl" in train else core.copy()
    core_m = metric_block(core, "core_train")
    rows = []
    best = None
    best_path = None
    for t_bars in T_GRID:
        for std_span in STD_GRID:
            for k_shift in K_GRID:
                for center_mode in CENTER_MODES:
                    put_pnl, delta, strike = option_layer(core, "put", t_bars, std_span, k_shift, center_mode)
                    for weight in WEIGHT_GRID:
                        overlay = core + weight * put_pnl
                        overlay_m = metric_block(overlay, "overlay_train")
                        out = {
                            "spec_id": int(row["spec_id"]),
                            "strategy_id": row["strategy_id"],
                            "family": row.get("family", ""),
                            "instrument": row.get("instrument", ""),
                            "t_bars": int(t_bars),
                            "std_span": int(std_span),
                            "k_shift_sigma": float(k_shift),
                            "center_scale": center_mode,
                            "overlay_weight": float(weight),
                            "train_n_obs": int(len(train)),
                            "train_start": train["timestamp"].min(),
                            "train_end": train["timestamp"].max(),
                        }
                        out.update(core_m)
                        out.update(overlay_m)
                        out["overlay_improvement"] = out["overlay_train_total"] - out["core_train_total"]
                        out["drawdown_reduction"] = abs(out["core_train_max_drawdown"]) - abs(out["overlay_train_max_drawdown"])
                        out["core_drawdown_to_pnl"] = drawdown_ratio(out["core_train_total"], out["core_train_max_drawdown"])
                        out["overlay_drawdown_to_pnl"] = drawdown_ratio(out["overlay_train_total"], out["overlay_train_max_drawdown"])
                        out["pnl_retention"] = out["overlay_train_total"] / max(abs(out["core_train_total"]), 1e-12)
                        out["overlay_score"] = overlay_score(core_m, overlay_m, weight)
                        rows.append(out)
                        if best is None or out["overlay_score"] > best["overlay_score"]:
                            best = out
                            best_path = pd.DataFrame(
                                {
                                    "timestamp": train["timestamp"].to_numpy(),
                                    "spec_id": int(row["spec_id"]),
                                    "strategy_id": row["strategy_id"],
                                    "instrument": row.get("instrument", ""),
                                    "core_gross_pnl": gross,
                                    "core_net_pnl": core,
                                    "put_overlay_pnl": put_pnl,
                                    "final_overlay_pnl": overlay,
                                    "put_delta": delta,
                                    "put_strike": strike,
                                }
                            )
    return rows, best, best_path


def summarize_best(best: pd.DataFrame) -> pd.DataFrame:
    if best.empty:
        return pd.DataFrame()
    return (
        best.assign(
            improved_pnl=best["overlay_improvement"] > 0,
            reduced_drawdown=best["drawdown_reduction"] > 0,
            better_ratio=best["overlay_drawdown_to_pnl"] < best["core_drawdown_to_pnl"],
        )
        .groupby("instrument", dropna=False)
        .agg(
            n=("spec_id", "count"),
            n_reduced_drawdown=("reduced_drawdown", "sum"),
            n_improved_pnl=("improved_pnl", "sum"),
            median_drawdown_reduction=("drawdown_reduction", "median"),
            median_overlay_improvement=("overlay_improvement", "median"),
            median_overlay_total=("overlay_train_total", "median"),
        )
        .reset_index()
        .sort_values(["n_reduced_drawdown", "median_drawdown_reduction"], ascending=[False, False])
    )


def run_train_overlay_batch(batch_id: int = 0, batch_size: int = DEFAULT_BATCH_SIZE) -> dict:
    ensure_dirs()
    train_start, train_end = read_train_dates()
    candidates = load_core_candidates()
    start = int(batch_id) * int(batch_size)
    end = min(start + int(batch_size), len(candidates))
    label = f"{int(batch_id):04d}"
    part = candidates.iloc[start:end].copy()
    grid_rows = []
    best_rows = []
    saved_paths = []
    errors = []
    batch_path_dir = BEST_PATH_DIR / f"batch_{label}"
    batch_path_dir.mkdir(parents=True, exist_ok=True)
    for _, row in part.iterrows():
        try:
            rows, best, best_path = evaluate_one(row, train_start, train_end)
            grid_rows.extend(rows)
            if best is not None:
                best_rows.append(best)
                p = batch_path_dir / f"{int(best['spec_id']):05d}__{safe_name(best['strategy_id'])}.parquet"
                best_path.to_parquet(p, index=False)
                saved_paths.append({"spec_id": best["spec_id"], "strategy_id": best["strategy_id"], "best_path": str(p)})
        except Exception as exc:
            errors.append({"spec_id": row.get("spec_id"), "strategy_id": row.get("strategy_id"), "error": repr(exc)})

    grid = pd.DataFrame(grid_rows)
    best = pd.DataFrame(best_rows)
    paths = pd.DataFrame(saved_paths)
    err = pd.DataFrame(errors, columns=["spec_id", "strategy_id", "error"])
    if not best.empty:
        best = best.sort_values(
            ["drawdown_reduction", "overlay_score", "overlay_train_total", "overlay_train_sharpe"],
            ascending=[False, False, False, False],
        )
    grid.to_csv(BATCH_DIR / f"grid_batch_{label}.csv", index=False)
    best.to_csv(BATCH_DIR / f"best_batch_{label}.csv", index=False)
    paths.to_csv(BATCH_DIR / f"best_paths_batch_{label}.csv", index=False)
    err.to_csv(BATCH_DIR / f"errors_batch_{label}.csv", index=False)
    manifest = {
        "batch_id": int(batch_id),
        "batch_label": label,
        "batch_size": int(batch_size),
        "train_start": str(train_start),
        "train_end": str(train_end),
        "n_candidates": int(len(candidates)),
        "start_row": int(start),
        "end_row_exclusive": int(end),
        "n_batch_candidates": int(len(part)),
        "n_grid_rows": int(len(grid)),
        "n_best_rows": int(len(best)),
        "n_errors": int(len(errors)),
        "output_dir": str(OUTPUT_DIR),
        "saved_files": [
            f"batches/grid_batch_{label}.csv",
            f"batches/best_batch_{label}.csv",
            f"batches/best_paths_batch_{label}.csv",
            f"batches/errors_batch_{label}.csv",
            f"best_paths/batch_{label}/",
            f"batches/manifest_batch_{label}.json",
        ],
    }
    with (BATCH_DIR / f"manifest_batch_{label}.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    return manifest


def read_batch_csvs(pattern: str) -> pd.DataFrame:
    frames = []
    for path in sorted(BATCH_DIR.glob(pattern)):
        if path.exists() and path.stat().st_size:
            df = pd.read_csv(path)
            df["source_file"] = str(path)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate_batches() -> dict:
    ensure_dirs()
    candidates = load_core_candidates()
    grid = read_batch_csvs("grid_batch_*.csv")
    best = read_batch_csvs("best_batch_*.csv")
    paths = read_batch_csvs("best_paths_batch_*.csv")
    errors = read_batch_csvs("errors_batch_*.csv")
    if not best.empty:
        best = best.sort_values(
            ["drawdown_reduction", "overlay_score", "overlay_train_total", "overlay_train_sharpe"],
            ascending=[False, False, False, False],
        )
    summary = summarize_best(best)
    grid.to_csv(OUTPUT_DIR / "01_put_overlay_train_grid.csv", index=False)
    best.to_csv(OUTPUT_DIR / "02_put_overlay_train_best.csv", index=False)
    paths.to_csv(OUTPUT_DIR / "03_put_overlay_train_best_paths.csv", index=False)
    pd.DataFrame(errors, columns=list(errors.columns) if not errors.empty else ["spec_id", "strategy_id", "error"]).to_csv(
        OUTPUT_DIR / "04_put_overlay_train_errors.csv", index=False
    )
    summary.to_csv(OUTPUT_DIR / "05_put_overlay_train_instrument_summary.csv", index=False)
    done_ids = set(best["spec_id"].astype(int)) if "spec_id" in best else set()
    all_ids = set(candidates["spec_id"].astype(int))
    missing = sorted(all_ids - done_ids)
    pd.DataFrame({"spec_id": missing}).to_csv(OUTPUT_DIR / "07_missing_spec_ids.csv", index=False)
    manifest = {
        "n_candidates": int(len(candidates)),
        "n_grid_rows": int(len(grid)),
        "n_best_rows": int(len(best)),
        "n_errors": int(len(errors)),
        "n_missing_spec_ids": int(len(missing)),
        "output_dir": str(OUTPUT_DIR),
        "saved_files": [
            "01_put_overlay_train_grid.csv",
            "02_put_overlay_train_best.csv",
            "03_put_overlay_train_best_paths.csv",
            "04_put_overlay_train_errors.csv",
            "05_put_overlay_train_instrument_summary.csv",
            "06_manifest.json",
            "07_missing_spec_ids.csv",
        ],
    }
    with (OUTPUT_DIR / "06_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--batch-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    if args.aggregate:
        print(json.dumps(aggregate_batches(), ensure_ascii=False, indent=2))
        return
    if not args.run:
        print("Use --run --batch-id N to test one put overlay train batch, or --aggregate to combine batches.")
        return
    print(json.dumps(run_train_overlay_batch(batch_id=args.batch_id, batch_size=args.batch_size), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

