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

CORE_PATHS_CSV = OUTPUT_ROOT / "01_core_strategy_factory_output" / "06_all_core_strategy_paths.csv"
FULL_SELECTION_CSV = OUTPUT_ROOT / "03_selection_leaderboard_output" / "01_full_selection_leaderboard.csv"
FINE_LEADERBOARD_CSV = OUTPUT_ROOT / "07_fine_tuning_analysis_output" / "02_fine_tuning_leaderboard.csv"
FINE_BATCH_DIR = OUTPUT_ROOT / "06_fine_tuning_output" / "batches"

OUTPUT_DIR = OUTPUT_ROOT / "08_test_run_output"
PATH_DIR = OUTPUT_DIR / "paths"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PATH_DIR.mkdir(parents=True, exist_ok=True)


def read_protocol_dates() -> tuple[pd.Timestamp, pd.Timestamp]:
    text = PROTOCOL_PATH.read_text(encoding="utf-8")
    start = "2020-01-01"
    end = "2023-08-10 23:59:59"
    in_test = False
    for raw in text.splitlines():
        line = raw.strip()
        if line == "test:":
            in_test = True
            continue
        if in_test and line.startswith(("train:", "calibration:")):
            in_test = False
        if in_test and line.startswith("start:"):
            start = line.split(":", 1)[1].strip().strip('"').strip("'")
        if in_test and line.startswith("end:"):
            end = line.split(":", 1)[1].strip().strip('"').strip("'")
    return pd.Timestamp(start), pd.Timestamp(end)


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))[:180]


def load_core_paths() -> pd.DataFrame:
    paths = pd.read_csv(CORE_PATHS_CSV)
    paths["spec_id"] = paths["spec_id"].astype(int)
    return paths[["spec_id", "path"]]


def load_fine_strategies() -> pd.DataFrame:
    fine = pd.read_csv(FINE_LEADERBOARD_CSV)
    fine = fine.sort_values("selection_score", ascending=False).drop_duplicates("spec_id")
    fine["basket"] = "two_option_improved"
    return fine


def load_self_sufficient_core() -> pd.DataFrame:
    full = pd.read_csv(FULL_SELECTION_CSV)
    core = full[full["self_sufficient_core"].astype(bool)].copy()
    core = core.sort_values("selection_score", ascending=False).drop_duplicates("spec_id")
    core["basket"] = "self_sufficient_core"
    core["calibration_group"] = "core_only"
    core["filter_group"] = "core_only"
    core["decision_type"] = "core_only_test"
    return core


def load_fine_grid_candidates() -> pd.DataFrame:
    frames = []
    for path in sorted(FINE_BATCH_DIR.glob("grid_batch_*.csv")):
        if path.stat().st_size:
            df = pd.read_csv(path)
            df["source_file"] = str(path)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_merged_mean_reversion_rows(grid: pd.DataFrame) -> pd.DataFrame:
    if grid.empty:
        return pd.DataFrame()
    mr = grid[
        (grid["calibration_group"] == "mean_reversion")
        & (grid["filter_group"].isin(["call/put", "put/put"]))
    ].copy()
    if mr.empty:
        return pd.DataFrame()
    mr = mr.sort_values(["spec_id", "filter_group", "selection_score"], ascending=[True, True, False])
    best_by_side = mr.drop_duplicates(["spec_id", "filter_group"])
    sides = best_by_side.pivot_table(index="spec_id", columns="filter_group", values="selection_score", aggfunc="first")
    merge_ids = sides.dropna(subset=["call/put", "put/put"], how="any").index.astype(int).tolist()
    rows = []
    for spec_id in merge_ids:
        pair = best_by_side[best_by_side["spec_id"].astype(int).eq(spec_id)]
        up = pair[pair["filter_group"].eq("call/put")].iloc[0].to_dict()
        down = pair[pair["filter_group"].eq("put/put")].iloc[0].to_dict()
        base = up.copy()
        base["basket"] = "merged_mean_reversion"
        base["decision_type"] = "merged_two_sided_mean_reversion"
        base["filter_group"] = "call/put+put/put"
        base["leg_up_json"] = json.dumps(up, ensure_ascii=False, default=str)
        base["leg_down_json"] = json.dumps(down, ensure_ascii=False, default=str)
        rows.append(base)
    return pd.DataFrame(rows)


def metric_with_extras(x: np.ndarray, prefix: str) -> dict:
    out = metric_block(x, prefix)
    step = np.diff(x, prepend=x[0]) if len(x) else np.array([])
    pos = step[step > 0].sum() if len(step) else np.nan
    neg = step[step < 0].sum() if len(step) else np.nan
    out[f"{prefix}_hit_rate"] = float((step > 0).mean()) if len(step) else np.nan
    out[f"{prefix}_profit_factor"] = float(pos / abs(neg)) if np.isfinite(neg) and neg < 0 else np.nan
    out[f"{prefix}_step_std"] = float(np.nanstd(step)) if len(step) else np.nan
    return out


def apply_two_option(row: pd.Series, core_net: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    layer1, delta1, strike1 = option_layer(
        core_net,
        row["opt1"],
        int(row["t1_bars"]),
        int(row["std1"]),
        float(row["k1_shift_sigma"]),
        row["center_scale1"],
    )
    layer2, delta2, strike2 = option_layer(
        layer1,
        row["opt2"],
        int(row["t2_bars"]),
        int(row["std2"]),
        float(row["k2_shift_sigma"]),
        row["center_scale2"],
    )
    return layer1, layer2, delta1, delta2, strike1, strike2


def evaluate_two_option(row: pd.Series, path: pd.DataFrame, test_start: pd.Timestamp, test_end: pd.Timestamp) -> tuple[dict, pd.DataFrame]:
    test = path[(path["timestamp"] >= test_start) & (path["timestamp"] <= test_end)].copy()
    core_net = test["net_pnl"].to_numpy(dtype=float)
    core_gross = test["gross_pnl"].to_numpy(dtype=float) if "gross_pnl" in test else core_net.copy()
    layer1, layer2, delta1, delta2, strike1, strike2 = apply_two_option(row, core_net)
    metrics = {
        "spec_id": int(row["spec_id"]),
        "strategy_id": row["strategy_id"],
        "family": row.get("family", ""),
        "instrument": row.get("instrument", ""),
        "basket": row.get("basket", "two_option_improved"),
        "filter_group": row.get("filter_group", ""),
        "calibration_group": row.get("calibration_group", ""),
        "decision_type": row.get("decision_type", ""),
        "opt1": row.get("opt1", ""),
        "opt2": row.get("opt2", ""),
        "t1_bars": int(row.get("t1_bars", 0)),
        "t2_bars": int(row.get("t2_bars", 0)),
        "std1": int(row.get("std1", 0)),
        "std2": int(row.get("std2", 0)),
        "k1_shift_sigma": float(row.get("k1_shift_sigma", 0.0)),
        "k2_shift_sigma": float(row.get("k2_shift_sigma", 0.0)),
        "center_scale1": row.get("center_scale1", ""),
        "center_scale2": row.get("center_scale2", ""),
        "test_n_obs": int(len(test)),
        "test_start": test["timestamp"].min() if len(test) else pd.NaT,
        "test_end": test["timestamp"].max() if len(test) else pd.NaT,
    }
    metrics.update(metric_with_extras(core_net, "core_test"))
    metrics.update(metric_with_extras(layer2, "test"))
    metrics["test_improvement_vs_core"] = metrics["test_total"] - metrics["core_test_total"]
    metrics["test_gross_total"] = float(core_gross[-1] - core_gross[0]) if len(core_gross) else np.nan
    metrics["test_cost_gap_core"] = metrics["test_gross_total"] - metrics["core_test_total"]
    out_path = pd.DataFrame(
        {
            "timestamp": test["timestamp"].to_numpy(),
            "spec_id": int(row["spec_id"]),
            "strategy_id": row["strategy_id"],
            "basket": row.get("basket", "two_option_improved"),
            "instrument": row.get("instrument", ""),
            "core_gross_pnl": core_gross,
            "core_net_pnl": core_net,
            "layer1_pnl": layer1,
            "strategy_net_pnl": layer2,
            "delta1": delta1,
            "delta2": delta2,
            "strike1": strike1,
            "strike2": strike2,
        }
    )
    return metrics, out_path


def evaluate_core(row: pd.Series, path: pd.DataFrame, test_start: pd.Timestamp, test_end: pd.Timestamp) -> tuple[dict, pd.DataFrame]:
    test = path[(path["timestamp"] >= test_start) & (path["timestamp"] <= test_end)].copy()
    core_net = test["net_pnl"].to_numpy(dtype=float)
    core_gross = test["gross_pnl"].to_numpy(dtype=float) if "gross_pnl" in test else core_net.copy()
    metrics = {
        "spec_id": int(row["spec_id"]),
        "strategy_id": row["strategy_id"],
        "family": row.get("family", ""),
        "instrument": row.get("instrument", ""),
        "basket": "self_sufficient_core",
        "filter_group": "core_only",
        "calibration_group": "core_only",
        "decision_type": "core_only_test",
        "opt1": "core",
        "opt2": "core",
        "test_n_obs": int(len(test)),
        "test_start": test["timestamp"].min() if len(test) else pd.NaT,
        "test_end": test["timestamp"].max() if len(test) else pd.NaT,
    }
    metrics.update(metric_with_extras(core_net, "core_test"))
    metrics.update(metric_with_extras(core_net, "test"))
    metrics["test_improvement_vs_core"] = 0.0
    metrics["test_gross_total"] = float(core_gross[-1] - core_gross[0]) if len(core_gross) else np.nan
    metrics["test_cost_gap_core"] = metrics["test_gross_total"] - metrics["core_test_total"]
    out_path = pd.DataFrame(
        {
            "timestamp": test["timestamp"].to_numpy(),
            "spec_id": int(row["spec_id"]),
            "strategy_id": row["strategy_id"],
            "basket": "self_sufficient_core",
            "instrument": row.get("instrument", ""),
            "core_gross_pnl": core_gross,
            "core_net_pnl": core_net,
            "layer1_pnl": core_net,
            "strategy_net_pnl": core_net,
            "delta1": np.nan,
            "delta2": np.nan,
            "strike1": np.nan,
            "strike2": np.nan,
        }
    )
    return metrics, out_path


def evaluate_merged(row: pd.Series, path: pd.DataFrame, test_start: pd.Timestamp, test_end: pd.Timestamp) -> tuple[dict, pd.DataFrame]:
    up = pd.Series(json.loads(row["leg_up_json"]))
    down = pd.Series(json.loads(row["leg_down_json"]))
    test = path[(path["timestamp"] >= test_start) & (path["timestamp"] <= test_end)].copy()
    core_net = test["net_pnl"].to_numpy(dtype=float)
    core_gross = test["gross_pnl"].to_numpy(dtype=float) if "gross_pnl" in test else core_net.copy()
    up_l1, up_l2, up_d1, up_d2, up_k1, up_k2 = apply_two_option(up, core_net)
    dn_l1, dn_l2, dn_d1, dn_d2, dn_k1, dn_k2 = apply_two_option(down, core_net)
    merged = up_l2 + dn_l2
    metrics = {
        "spec_id": int(row["spec_id"]),
        "strategy_id": row["strategy_id"],
        "family": row.get("family", ""),
        "instrument": row.get("instrument", ""),
        "basket": "merged_mean_reversion",
        "filter_group": "call/put+put/put",
        "calibration_group": "mean_reversion",
        "decision_type": "merged_two_sided_mean_reversion",
        "opt1": "merged",
        "opt2": "merged",
        "test_n_obs": int(len(test)),
        "test_start": test["timestamp"].min() if len(test) else pd.NaT,
        "test_end": test["timestamp"].max() if len(test) else pd.NaT,
    }
    metrics.update(metric_with_extras(core_net, "core_test"))
    metrics.update(metric_with_extras(merged, "test"))
    metrics["test_improvement_vs_core"] = metrics["test_total"] - metrics["core_test_total"]
    metrics["test_gross_total"] = float(core_gross[-1] - core_gross[0]) if len(core_gross) else np.nan
    metrics["test_cost_gap_core"] = metrics["test_gross_total"] - metrics["core_test_total"]
    out_path = pd.DataFrame(
        {
            "timestamp": test["timestamp"].to_numpy(),
            "spec_id": int(row["spec_id"]),
            "strategy_id": row["strategy_id"],
            "basket": "merged_mean_reversion",
            "instrument": row.get("instrument", ""),
            "core_gross_pnl": core_gross,
            "core_net_pnl": core_net,
            "layer1_pnl": up_l1 + dn_l1,
            "strategy_net_pnl": merged,
            "delta1": up_d1 + dn_d1,
            "delta2": up_d2 + dn_d2,
            "strike1": np.nan,
            "strike2": np.nan,
        }
    )
    return metrics, out_path


def portfolio_path_from_files(saved_paths: list[dict]) -> pd.DataFrame:
    if not saved_paths:
        return pd.DataFrame()
    strategy_sum = None
    core_sum = None
    gross_sum = None
    for row in saved_paths:
        p = pd.read_parquet(row["test_path"], columns=["timestamp", "strategy_net_pnl", "core_net_pnl", "core_gross_pnl"])
        p = p.sort_values("timestamp")
        idx = pd.to_datetime(p["timestamp"])
        strategy_step = pd.Series(p["strategy_net_pnl"].diff().fillna(0.0).to_numpy(dtype=float), index=idx)
        core_step = pd.Series(p["core_net_pnl"].diff().fillna(0.0).to_numpy(dtype=float), index=idx)
        gross_step = pd.Series(p["core_gross_pnl"].diff().fillna(0.0).to_numpy(dtype=float), index=idx)
        strategy_sum = strategy_step if strategy_sum is None else strategy_sum.add(strategy_step, fill_value=0.0)
        core_sum = core_step if core_sum is None else core_sum.add(core_step, fill_value=0.0)
        gross_sum = gross_step if gross_sum is None else gross_sum.add(gross_step, fill_value=0.0)
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
    port["portfolio_drawdown"] = port["portfolio_net_pnl"] - port["portfolio_net_pnl"].cummax()
    return port


def run_test(include_merged: bool = True) -> dict:
    ensure_dirs()
    test_start, test_end = read_protocol_dates()
    core_paths = load_core_paths()
    fine = load_fine_strategies()
    core = load_self_sufficient_core()
    rows = [fine, core]
    if include_merged:
        rows.append(build_merged_mean_reversion_rows(load_fine_grid_candidates()))
    candidates = pd.concat([x for x in rows if not x.empty], ignore_index=True)
    candidates = candidates.merge(core_paths, on="spec_id", how="left", suffixes=("", "_core"))

    metrics_rows = []
    saved_paths = []
    errors = []
    for _, row in candidates.iterrows():
        try:
            path_file = Path(row["path"])
            path = pd.read_parquet(path_file)
            path["timestamp"] = pd.to_datetime(path["timestamp"])
            if row["basket"] == "self_sufficient_core":
                metrics, out_path = evaluate_core(row, path, test_start, test_end)
            elif row["basket"] == "merged_mean_reversion":
                metrics, out_path = evaluate_merged(row, path, test_start, test_end)
            else:
                metrics, out_path = evaluate_two_option(row, path, test_start, test_end)
            metrics_rows.append(metrics)
            p = PATH_DIR / f"{metrics['basket']}__{int(metrics['spec_id']):05d}__{safe_name(metrics['strategy_id'])}.parquet"
            out_path.to_parquet(p, index=False)
            saved_paths.append({"spec_id": metrics["spec_id"], "strategy_id": metrics["strategy_id"], "basket": metrics["basket"], "test_path": str(p)})
        except Exception as exc:
            errors.append({"spec_id": row.get("spec_id"), "strategy_id": row.get("strategy_id"), "basket": row.get("basket"), "error": repr(exc)})

    metrics = pd.DataFrame(metrics_rows)
    paths = pd.DataFrame(saved_paths)
    port = portfolio_path_from_files(saved_paths)
    instrument_usage = metrics.groupby(["basket", "instrument"], dropna=False).size().reset_index(name="n_strategies") if not metrics.empty else pd.DataFrame()

    metrics.to_csv(OUTPUT_DIR / "01_test_strategy_results.csv", index=False)
    paths.to_csv(OUTPUT_DIR / "02_test_strategy_paths_catalog.csv", index=False)
    port.to_parquet(OUTPUT_DIR / "03_test_portfolio_path.parquet", index=False)
    instrument_usage.to_csv(OUTPUT_DIR / "04_test_instruments_usage.csv", index=False)
    error_columns = ["spec_id", "strategy_id", "basket", "error"]
    pd.DataFrame(errors, columns=error_columns).to_csv(OUTPUT_DIR / "05_test_errors.csv", index=False)

    manifest = {
        "test_start": str(test_start),
        "test_end": str(test_end),
        "n_candidates": int(len(candidates)),
        "n_tested": int(len(metrics)),
        "n_errors": int(len(errors)),
        "n_two_option_improved": int((metrics["basket"] == "two_option_improved").sum()) if not metrics.empty else 0,
        "n_self_sufficient_core": int((metrics["basket"] == "self_sufficient_core").sum()) if not metrics.empty else 0,
        "n_merged_mean_reversion": int((metrics["basket"] == "merged_mean_reversion").sum()) if not metrics.empty else 0,
        "output_dir": str(OUTPUT_DIR),
        "saved_files": [
            "01_test_strategy_results.csv",
            "02_test_strategy_paths_catalog.csv",
            "03_test_portfolio_path.parquet",
            "04_test_instruments_usage.csv",
            "05_test_errors.csv",
            "06_manifest.json",
        ],
    }
    with (OUTPUT_DIR / "06_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--no-merged", action="store_true")
    args = parser.parse_args()
    if not args.run:
        print("Use --run to execute the final test sample pass.")
        return
    manifest = run_test(include_merged=not args.no_merged)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

