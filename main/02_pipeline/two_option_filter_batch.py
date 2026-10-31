from pathlib import Path
import argparse
import json
import math
import numpy as np
import pandas as pd

WORK_ROOT = Path(r"./workspace")
OUTPUT_ROOT = Path(r"D:\Filter_2_option_v2")
FACTORY_OUTPUT = OUTPUT_ROOT / "01_core_strategy_factory_output"
OUTPUT_DIR = OUTPUT_ROOT / "02_two_option_filter_output"
BATCH_DIR = OUTPUT_DIR / "batches"
BEST_PATH_DIR = OUTPUT_DIR / "best_paths"

CORE_PATHS_CSV = FACTORY_OUTPUT / "06_all_core_strategy_paths.csv"
CORE_METRICS_CSV = FACTORY_OUTPUT / "03_core_strategy_train_metrics.csv"

DEFAULT_BATCH_SIZE = 150
COMBOS = [("call", "call"), ("call", "put"), ("put", "call"), ("put", "put")]
T1_GRID = [44 * 22]
T2_GRID = [44 * 5, 44 * 22]
STD_SPANS = [44]
K_SHIFT_SIGMAS = [0.0]


def ensure_dirs():
    for p in [OUTPUT_DIR, BATCH_DIR, BEST_PATH_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def load_catalog():
    paths = pd.read_csv(CORE_PATHS_CSV)
    metrics = pd.read_csv(CORE_METRICS_CSV)
    keep = ["spec_id", "strategy_id", "family", "instrument", "geometry_label", "train_net_total", "calibration_net_total", "test_net_total"]
    metrics = metrics[[c for c in keep if c in metrics.columns]].drop_duplicates("spec_id")
    cat = paths.merge(metrics, on=["spec_id", "strategy_id"], how="left", suffixes=("", "_metric"))
    cat = cat.sort_values("spec_id").reset_index(drop=True)
    return cat


def period_anchor(values, period):
    n = len(values)
    out = np.empty(n, dtype=float)
    last = np.nan
    for i in range(n):
        if i % period == 0 or not np.isfinite(last):
            last = values[i]
        out[i] = last
    return out


def rolling_std(x, window):
    return pd.Series(x).rolling(window, min_periods=max(5, window // 3)).std().to_numpy()


def option_layer(base, opt_type, t_bars, std_span, k_shift_sigma):
    base = np.asarray(base, dtype=float)
    inc = np.diff(base, prepend=base[0])
    vol = rolling_std(inc, std_span)
    anchor = period_anchor(base, int(t_bars))
    shift = k_shift_sigma * np.nan_to_num(vol, nan=0.0)
    if opt_type == "call":
        strike = anchor + shift
    else:
        strike = anchor - shift
    spread = base - strike
    scale = pd.Series(spread).ewm(span=int(std_span), adjust=False, min_periods=max(5, int(std_span)//3)).std().replace(0, np.nan).to_numpy()
    z = np.divide(spread, scale, out=np.zeros_like(spread), where=np.isfinite(scale) & (scale != 0))
    delta = np.tanh(z)
    if opt_type == "call":
        delta = np.clip(delta, 0.0, 1.0)
    else:
        delta = np.clip(delta, -1.0, 0.0)
    pnl = np.cumsum(np.roll(delta, 1) * inc)
    pnl[0] = 0.0
    return pnl, delta, strike


def drawdown(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmin(x - np.maximum.accumulate(x))) if len(x) else np.nan


def slope_r2(x):
    arr = np.asarray(x, dtype=float)
    ok = np.isfinite(arr)
    if ok.sum() < 20:
        return np.nan, np.nan
    y = arr[ok]
    t = np.arange(len(y), dtype=float)
    t -= t.mean()
    yc = y - y.mean()
    den = np.dot(t, t)
    b = float(np.dot(t, yc) / den) if den else np.nan
    fit = y.mean() + b * t
    ssr = float(np.sum((y - fit) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return b, 1.0 - ssr / sst if sst > 0 else np.nan


def score_path(final_pnl, core_pnl):
    final_total = float(final_pnl[-1] - final_pnl[0])
    core_total = float(core_pnl[-1] - core_pnl[0])
    slope, r2 = slope_r2(final_pnl)
    max_dd = drawdown(final_pnl - final_pnl[0])
    improvement = final_total - core_total
    # Фильтр ценится за положительный результат, улучшение и меньшую просадку.
    score = final_total + 0.5 * improvement + 0.25 * abs(core_total) - 0.25 * abs(max_dd)
    return {
        "two_option_train_total": final_total,
        "core_train_total": core_total,
        "improvement_vs_core": improvement,
        "two_option_train_slope": slope,
        "two_option_train_trend_r2": r2,
        "two_option_train_max_drawdown": max_dd,
        "selection_score": float(score),
    }


def evaluate_one(row):
    path = pd.read_parquet(row["path"])
    if "timestamp" in path.columns:
        path["timestamp"] = pd.to_datetime(path["timestamp"])
    train = path[path["timestamp"] <= pd.Timestamp("2016-12-31 23:59:59")].copy()
    if len(train) < 200:
        return [], None
    core = train["net_pnl"].to_numpy(dtype=float)
    rows = []
    best = None
    best_payload = None
    for opt1, opt2 in COMBOS:
        for t1 in T1_GRID:
            for t2 in T2_GRID:
                for std1 in STD_SPANS:
                    for std2 in STD_SPANS:
                        for k1 in K_SHIFT_SIGMAS:
                            for k2 in K_SHIFT_SIGMAS:
                                layer1, delta1, strike1 = option_layer(core, opt1, t1, std1, k1)
                                final, delta2, strike2 = option_layer(layer1, opt2, t2, std2, k2)
                                m = score_path(final, core)
                                out = {
                                    "spec_id": int(row["spec_id"]),
                                    "strategy_id": row["strategy_id"],
                                    "family": row.get("family", ""),
                                    "instrument": row.get("instrument", ""),
                                    "geometry_label": row.get("geometry_label", ""),
                                    "opt1": opt1,
                                    "opt2": opt2,
                                    "t1_bars": t1,
                                    "t2_bars": t2,
                                    "std1": std1,
                                    "std2": std2,
                                    "k1_shift_sigma": k1,
                                    "k2_shift_sigma": k2,
                                }
                                out.update(m)
                                rows.append(out)
                                if best is None or out["selection_score"] > best["selection_score"]:
                                    best = out
                                    best_payload = (train, layer1, final, delta1, delta2, strike1, strike2)
    if best_payload is not None:
        train, layer1, final, delta1, delta2, strike1, strike2 = best_payload
        best_path = pd.DataFrame({
            "timestamp": train["timestamp"].to_numpy(),
            "strategy_id": row["strategy_id"],
            "core_net_pnl": core,
            "layer1_pnl": layer1,
            "two_option_pnl": final,
            "delta1": delta1,
            "delta2": delta2,
            "strike1": strike1,
            "strike2": strike2,
        })
    else:
        best_path = None
    return rows, best_path


def run_batch(batch_id=0, batch_size=DEFAULT_BATCH_SIZE, start_id=None, end_id=None, batch_label=None):
    ensure_dirs()
    cat = load_catalog()
    if start_id is None:
        start = int(batch_id) * int(batch_size)
    else:
        start = int(start_id)
    if end_id is None:
        end = min(start + int(batch_size), len(cat))
    else:
        end = min(int(end_id), len(cat))
    label = batch_label or f"{int(batch_id):04d}"
    part = cat.iloc[start:end]
    all_rows = []
    best_rows = []
    saved = []
    best_dir = BEST_PATH_DIR / f"batch_{label}"
    best_dir.mkdir(parents=True, exist_ok=True)
    for _, row in part.iterrows():
        rows, best_path = evaluate_one(row)
        all_rows.extend(rows)
        if rows:
            best = max(rows, key=lambda x: x["selection_score"])
            best_rows.append(best)
            if best_path is not None:
                safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(row["strategy_id"]))
                p = best_dir / f"{int(row['spec_id']):05d}__{safe}.parquet"
                best_path.to_parquet(p, index=False)
                saved.append({"spec_id": int(row["spec_id"]), "strategy_id": row["strategy_id"], "best_path": str(p)})
    grid = pd.DataFrame(all_rows)
    best = pd.DataFrame(best_rows)
    paths = pd.DataFrame(saved)
    grid.to_csv(BATCH_DIR / f"grid_batch_{label}.csv", index=False)
    best.to_csv(BATCH_DIR / f"best_batch_{label}.csv", index=False)
    paths.to_csv(BATCH_DIR / f"best_paths_batch_{label}.csv", index=False)
    manifest = {
        "batch_label": label,
        "start_spec_id": start,
        "end_spec_id_exclusive": end,
        "n_catalog_total": int(len(cat)),
        "n_strategies": int(len(part)),
        "n_grid_rows": int(len(grid)),
        "n_best_rows": int(len(best)),
        "n_best_paths_saved": int(len(paths)),
        "output_dir": str(OUTPUT_DIR),
    }
    (BATCH_DIR / f"manifest_batch_{label}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return grid, best, paths, manifest


def combine_batches():
    ensure_dirs()
    best_files = sorted(BATCH_DIR.glob("best_batch_*.csv"))
    grid_files = sorted(BATCH_DIR.glob("grid_batch_*.csv"))
    path_files = sorted(BATCH_DIR.glob("best_paths_batch_*.csv"))
    best = pd.concat([pd.read_csv(p) for p in best_files], ignore_index=True) if best_files else pd.DataFrame()
    paths = pd.concat([pd.read_csv(p) for p in path_files], ignore_index=True) if path_files else pd.DataFrame()
    if not best.empty:
        best = best.drop_duplicates("spec_id", keep="last").sort_values("selection_score", ascending=False)
    if not paths.empty:
        paths = paths.drop_duplicates("spec_id", keep="last")
    best.to_csv(OUTPUT_DIR / "03_two_option_best_by_strategy.csv", index=False)
    paths.to_csv(OUTPUT_DIR / "04_two_option_best_paths.csv", index=False)
    summary = {
        "n_grid_batch_files": len(grid_files),
        "n_best_batch_files": len(best_files),
        "n_unique_strategies": int(len(best)),
        "n_best_paths": int(len(paths)),
        "output_dir": str(OUTPUT_DIR),
    }
    (OUTPUT_DIR / "05_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return best, paths, summary


def audit_batches():
    cat = load_catalog()
    best_files = sorted(BATCH_DIR.glob("best_batch_*.csv"))
    raw = pd.concat([pd.read_csv(p) for p in best_files], ignore_index=True) if best_files else pd.DataFrame()
    if raw.empty:
        report = {"catalog_rows": int(len(cat)), "processed_unique": 0, "missing_count": int(len(cat)), "duplicate_count": 0}
    else:
        missing = sorted(set(cat["spec_id"]) - set(raw["spec_id"]))
        duplicate = sorted(raw.loc[raw["spec_id"].duplicated(), "spec_id"].astype(int).unique().tolist())
        report = {
            "catalog_rows": int(len(cat)),
            "processed_rows": int(len(raw)),
            "processed_unique": int(raw["spec_id"].nunique()),
            "missing_count": int(len(missing)),
            "duplicate_count": int(len(duplicate)),
            "missing_ranges": ranges(missing),
            "duplicate_head": duplicate[:20],
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def ranges(vals):
    if not vals:
        return []
    out = []
    s = p = int(vals[0])
    for v in vals[1:]:
        v = int(v)
        if v == p + 1:
            p = v
        else:
            out.append([s, p + 1])
            s = p = v
    out.append([s, p + 1])
    return out


def summary():
    cat = load_catalog()
    n = len(cat)
    print("Total core strategies (всего базовых стратегий):", n)
    print("Batch size (размер пакета):", DEFAULT_BATCH_SIZE)
    print("Batch ids (номера пакетов): 0 ..", math.ceil(n / DEFAULT_BATCH_SIZE) - 1)
    print(cat.groupby("family").size().sort_values(ascending=False).to_string())
    return cat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["summary", "batch", "combine", "audit"], default="summary")
    parser.add_argument("--batch-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--start-id", type=int, default=None)
    parser.add_argument("--end-id", type=int, default=None)
    parser.add_argument("--batch-label", type=str, default=None)
    args = parser.parse_args()
    if args.mode == "summary":
        summary()
    elif args.mode == "batch":
        run_batch(args.batch_id, args.batch_size, args.start_id, args.end_id, args.batch_label)
    elif args.mode == "combine":
        combine_batches()
    elif args.mode == "audit":
        audit_batches()

