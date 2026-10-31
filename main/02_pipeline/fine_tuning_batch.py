from pathlib import Path
import argparse
import json
import math
import numpy as np
import pandas as pd

WORK_ROOT = Path(r"./workspace")
OUTPUT_ROOT = Path(r"D:\Filter_2_option_v2")
FACTORY_OUTPUT = OUTPUT_ROOT / "01_core_strategy_factory_output"
LEADERBOARD_OUTPUT = OUTPUT_ROOT / "05_select_for_fine_calibration_output"
OUTPUT_DIR = OUTPUT_ROOT / "06_fine_tuning_output"
BATCH_DIR = OUTPUT_DIR / "batches"
BEST_PATH_DIR = OUTPUT_DIR / "best_paths"

SELECTED_CSV = LEADERBOARD_OUTPUT / "02d_selected_for_fine_with_rescue.csv"
CORE_PATHS_CSV = FACTORY_OUTPUT / "06_all_core_strategy_paths.csv"

DEFAULT_BATCH_SIZE = 80
CALIBRATION_START = pd.Timestamp("2017-01-01")
CALIBRATION_END = pd.Timestamp("2018-06-30 23:59:59")
VALIDATION_START = pd.Timestamp("2018-07-01")
VALIDATION_END = pd.Timestamp("2019-12-31 23:59:59")

FIXED_T_BARS_TREND_COARSE = [44, 44 * 5, 44 * 22]
FIXED_T_BARS_TREND_FINE = [44, 132, 220, 440, 968]
FIXED_T_BARS_MR_COARSE = [44, 44 * 5]
FIXED_T_BARS_MR_FINE = [44, 220]
HALF_LIFE_MULTS_COARSE = [0.5, 1.0, 1.5]
HALF_LIFE_MULTS_FINE = [0.5, 1.0, 1.5, 2.0]
MIN_T_BARS = 22
MAX_T_BARS = 44 * 66

STD_SPANS_TREND_COARSE = [44, 88]
STD_SPANS_TREND_FINE = [44, 88]
K1_SHIFT_TREND_COARSE = [0.5, 1.0, 1.5]
K2_SHIFT_TREND_COARSE = [0.5, 1.5]
K1_SHIFT_TREND_FINE = [0.5, 1.0, 1.5]
K2_SHIFT_TREND_FINE = [0.5, 1.5]
K_SHIFT_TREND_FINE = K1_SHIFT_TREND_FINE
K_SHIFT_MR_COARSE = [0.5, 1.0, 1.5]
K1_SHIFT_MR_FINE = [0.5, 1.0, 1.5, 2.0]
K2_SHIFT_MR_FINE = [0.5, 1.0, 1.5]
K_SHIFT_MR_FINE = K1_SHIFT_MR_FINE

TREND_ANCHORS = ["period_start"]
MR_CENTER_SCALE_PRIMARY = ["median_mad"]
MR_CENTER_SCALE_FALLBACK = ["mean_std"]

TREND_T_ALLOWED = sorted(set([22, 44, 66, 88, 110, 154, 220, 330, 440, 660, 968, 1452, 1936]))
TREND_STD_ALLOWED = sorted(set([22, 33, 44, 66, 88, 132, 176]))
TREND_K_ALLOWED = sorted(set([0.17, 0.33, 0.5, 0.67, 0.83, 1.0, 1.17, 1.33, 1.5, 1.67, 1.83, 2.0]))
MR_K_ALLOWED = sorted(set([0.17, 0.33, 0.5, 0.67, 0.83, 1.0, 1.17, 1.33, 1.5, 1.67, 1.83, 2.0, 2.33]))


def ensure_dirs():
    for p in [OUTPUT_DIR, BATCH_DIR, BEST_PATH_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def load_selected():
    selected = pd.read_csv(SELECTED_CSV)
    paths = pd.read_csv(CORE_PATHS_CSV)[["spec_id", "path"]]
    selected = selected.merge(paths, on="spec_id", how="left", suffixes=("", "_core"))
    selected = selected.sort_values("spec_id").reset_index(drop=True)
    return selected


def ranges(vals):
    if not vals:
        return []
    out = []
    start = prev = int(vals[0])
    for v in vals[1:]:
        v = int(v)
        if v == prev + 1:
            prev = v
        else:
            out.append([start, prev + 1])
            start = prev = v
    out.append([start, prev + 1])
    return out


def rolling_std(x, window):
    return pd.Series(x).rolling(int(window), min_periods=max(5, int(window) // 3)).std().to_numpy()


def rolling_mad(x, window):
    s = pd.Series(x)
    window = int(window)
    med = s.rolling(window, min_periods=max(5, window // 3)).median()
    mad = (s - med).abs().rolling(window, min_periods=max(5, window // 3)).median()
    return (1.4826 * mad).to_numpy()


def center_and_scale(x, mode, window):
    s = pd.Series(x)
    window = int(window)
    if mode == "period_start":
        out = np.empty(len(x), dtype=float)
        last = np.nan
        for i, v in enumerate(x):
            if i % window == 0 or not np.isfinite(last):
                last = v
            out[i] = last
        scale = rolling_std(np.diff(x, prepend=x[0]), max(22, min(window, 176)))
        return out, scale
    if mode == "median_mad":
        center = s.rolling(window, min_periods=max(5, window // 3)).median().bfill().ffill().to_numpy()
        scale = rolling_mad(x, window)
        return center, scale
    if mode == "mean_std":
        center = s.rolling(window, min_periods=max(5, window // 3)).mean().bfill().ffill().to_numpy()
        scale = rolling_std(x - center, window)
        return center, scale
    raise ValueError(f"unknown center_scale mode: {mode}")


def infer_half_life_bars(path_part):
    x = path_part["core_net_pnl"].to_numpy(dtype=float) if "core_net_pnl" in path_part else path_part["net_pnl"].to_numpy(dtype=float)
    if len(x) < 200:
        return np.nan
    trend = np.linspace(x[0], x[-1], len(x))
    s = pd.Series(x - trend).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 200:
        return np.nan
    y = s.diff().dropna()
    lag = s.shift(1).reindex(y.index)
    if lag.std() == 0:
        return np.nan
    beta = np.polyfit(lag.to_numpy(), y.to_numpy(), 1)[0]
    if beta >= 0:
        return np.nan
    return float(-np.log(2) / beta)


def build_t_grid(group, half_life, calibration_mode="coarse"):
    if group == "mean_reversion":
        vals = list(FIXED_T_BARS_MR_FINE if calibration_mode == "fine" else FIXED_T_BARS_MR_COARSE)
        if np.isfinite(half_life):
            mults = HALF_LIFE_MULTS_FINE if calibration_mode == "fine" else HALF_LIFE_MULTS_COARSE
            vals += [int(round(float(half_life) * m)) for m in mults]
        vals = [int(np.clip(v, MIN_T_BARS, MAX_T_BARS)) for v in vals]
        return sorted(set(vals))
    return sorted(set(FIXED_T_BARS_TREND_FINE if calibration_mode == "fine" else FIXED_T_BARS_TREND_COARSE))


def calibration_group(row):
    fg = str(row.get("filter_group", ""))
    geom = str(row.get("geometry_label", ""))
    if fg == "core_only":
        return "core_only"
    if "mean_reversion" in geom or fg in {"call/put", "put/put"}:
        return "mean_reversion"
    return "trend"


def option_layer(base, opt_type, t_bars, std_span, k_shift_sigma, center_scale_mode):
    base = np.asarray(base, dtype=float)
    inc = np.diff(base, prepend=base[0])
    t_bars = int(t_bars)
    std_span = int(std_span)
    center_window = t_bars if center_scale_mode == "period_start" else t_bars
    center, scale = center_and_scale(base, center_scale_mode, center_window)
    if np.nanmedian(np.abs(scale)) < 1e-12 or not np.isfinite(np.nanmedian(scale)):
        scale = rolling_std(inc, std_span)
    scale = np.nan_to_num(scale, nan=0.0)
    strike = center + k_shift_sigma * scale if opt_type == "call" else center - k_shift_sigma * scale
    spread = base - strike
    norm = pd.Series(spread).ewm(span=std_span, adjust=False, min_periods=max(5, std_span // 3)).std().replace(0, np.nan).to_numpy()
    z = np.divide(spread, norm, out=np.zeros_like(spread), where=np.isfinite(norm) & (norm != 0))
    delta = np.tanh(z)
    delta = np.clip(delta, 0.0, 1.0) if opt_type == "call" else np.clip(delta, -1.0, 0.0)
    pnl = np.cumsum(np.roll(delta, 1) * inc)
    pnl[0] = 0.0
    return pnl, delta, strike


def metric_block(x, prefix):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return {f"{prefix}_total": np.nan, f"{prefix}_sharpe": np.nan, f"{prefix}_sortino": np.nan, f"{prefix}_max_drawdown": np.nan}
    step = np.diff(x, prepend=x[0])
    downside = step[step < 0]
    mean = float(np.nanmean(step))
    std = float(np.nanstd(step))
    dstd = float(np.nanstd(downside)) if len(downside) else np.nan
    sharpe = mean / std * np.sqrt(44 * 252) if std > 1e-12 else np.nan
    sortino = mean / dstd * np.sqrt(44 * 252) if np.isfinite(dstd) and dstd > 1e-12 else np.nan
    dd = float(np.nanmin(x - np.maximum.accumulate(x)))
    return {f"{prefix}_total": float(x[-1] - x[0]), f"{prefix}_sharpe": sharpe, f"{prefix}_sortino": sortino, f"{prefix}_max_drawdown": dd}


def score(cal, val, core_cal, core_val):
    improvement_cal = cal["calibration_total"] - core_cal["core_calibration_total"]
    improvement_val = val["validation_total"] - core_val["core_validation_total"]
    dd_penalty = abs(cal["calibration_max_drawdown"]) + abs(val["validation_max_drawdown"])
    return float(
        cal["calibration_total"]
        + val["validation_total"]
        + 0.7 * improvement_cal
        + 1.2 * improvement_val
        + 0.25 * np.nan_to_num(cal["calibration_sharpe"])
        + 0.25 * np.nan_to_num(val["validation_sharpe"])
        - 0.25 * dd_penalty
    )


def evaluate_core_only(row, path):
    cal = path[(path["timestamp"] >= CALIBRATION_START) & (path["timestamp"] <= CALIBRATION_END)]
    val = path[(path["timestamp"] >= VALIDATION_START) & (path["timestamp"] <= VALIDATION_END)]
    core_cal = metric_block(cal["net_pnl"].to_numpy(dtype=float), "core_calibration")
    core_val = metric_block(val["net_pnl"].to_numpy(dtype=float), "core_validation")
    out = {
        "spec_id": int(row["spec_id"]),
        "strategy_id": row["strategy_id"],
        "filter_group": "core_only",
        "calibration_group": "core_only",
        "decision_type": "core_only_baseline",
        "opt1": "core",
        "opt2": "core",
        "t1_bars": 0,
        "t2_bars": 0,
        "std1": 0,
        "std2": 0,
        "k1_shift_sigma": 0.0,
        "k2_shift_sigma": 0.0,
        "center_scale1": "none",
        "center_scale2": "none",
        "half_life_bars": np.nan,
    }
    out.update(core_cal)
    out.update(core_val)
    out["calibration_total"] = out["core_calibration_total"]
    out["validation_total"] = out["core_validation_total"]
    out["improvement_calibration"] = 0.0
    out["improvement_validation"] = 0.0
    out["selection_score"] = out["calibration_total"] + out["validation_total"]
    return [out], None


def nearest_allowed(value, allowed):
    if value is None or not np.isfinite(value):
        return allowed[len(allowed) // 2]
    return min(allowed, key=lambda x: abs(float(x) - float(value)))


def local_numeric_grid(best_value, allowed, radius=1, edge_extend=True):
    center = nearest_allowed(best_value, allowed)
    idx = allowed.index(center)
    lo = max(0, idx - radius)
    hi = min(len(allowed), idx + radius + 1)
    vals = list(allowed[lo:hi])
    if edge_extend and idx == 0 and len(allowed) > hi:
        vals.append(allowed[hi])
    if edge_extend and idx == len(allowed) - 1 and lo > 0:
        vals.insert(0, allowed[lo - 1])
    return sorted(set(vals))


def local_numeric_grid_count(best_value, allowed, count=4):
    if count >= len(allowed):
        return list(allowed)
    center = nearest_allowed(best_value, allowed)
    idx = allowed.index(center)
    left = idx
    right = idx
    while right - left + 1 < count:
        can_left = left > 0
        can_right = right < len(allowed) - 1
        if can_left and can_right:
            left_gap = abs(float(center) - float(allowed[left - 1]))
            right_gap = abs(float(allowed[right + 1]) - float(center))
            if left_gap <= right_gap:
                left -= 1
            else:
                right += 1
        elif can_left:
            left -= 1
        elif can_right:
            right += 1
        else:
            break
    return sorted(set(allowed[left:right + 1]))


def mean_reversion_t_candidates(half_life):
    vals = [22, 44, 66, 88, 110, 154, 220, 330, 440]
    if np.isfinite(half_life):
        for mult in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]:
            vals.append(int(round(float(half_life) * mult)))
    return sorted(set(int(np.clip(v, MIN_T_BARS, MAX_T_BARS)) for v in vals))


def local_t_grid(best_t, group, half_life):
    if group == "mean_reversion":
        return local_numeric_grid(best_t, mean_reversion_t_candidates(half_life), radius=2, edge_extend=True)
    return local_numeric_grid_count(best_t, TREND_T_ALLOWED, count=4)


def choose_grids(group, half_life, calibration_mode, row=None):
    if calibration_mode == "fine" and row is not None:
        if group == "mean_reversion":
            mr_t = mean_reversion_t_candidates(half_life)
            t1_grid = local_numeric_grid_count(row.get("t1_bars", np.nan), mr_t, count=2)
            t2_grid = local_numeric_grid_count(row.get("t2_bars", np.nan), mr_t, count=7)
        else:
            t1_grid = local_t_grid(row.get("t1_bars", np.nan), group, half_life)
            t2_grid = local_t_grid(row.get("t2_bars", np.nan), group, half_life)
    else:
        t_grid = build_t_grid(group, half_life, calibration_mode)
        t1_grid = t_grid
        t2_grid = t_grid
    if group == "mean_reversion":
        std1_grid = [44]
        std2_grid = [44]
        if calibration_mode == "fine" and row is not None:
            k1_grid = local_numeric_grid_count(row.get("k1_shift_sigma", np.nan), MR_K_ALLOWED, count=4)
            k2_grid = local_numeric_grid(row.get("k2_shift_sigma", np.nan), MR_K_ALLOWED, radius=1, edge_extend=True)
        else:
            k1_grid = K1_SHIFT_MR_FINE if calibration_mode == "fine" else K_SHIFT_MR_COARSE
            k2_grid = K2_SHIFT_MR_FINE if calibration_mode == "fine" else K_SHIFT_MR_COARSE
        center_scale_modes = list(MR_CENTER_SCALE_PRIMARY)
    else:
        if calibration_mode == "fine" and row is not None:
            std1_grid = local_numeric_grid(row.get("std1", np.nan), TREND_STD_ALLOWED, radius=1, edge_extend=True)
            std2_grid = local_numeric_grid(row.get("std2", np.nan), TREND_STD_ALLOWED, radius=1, edge_extend=True)
            k1_grid = local_numeric_grid(row.get("k1_shift_sigma", np.nan), TREND_K_ALLOWED, radius=1, edge_extend=True)
            k2_grid = local_numeric_grid(row.get("k2_shift_sigma", np.nan), TREND_K_ALLOWED, radius=1, edge_extend=True)
        elif calibration_mode == "fine":
            std1_grid = STD_SPANS_TREND_FINE
            std2_grid = STD_SPANS_TREND_FINE
            k1_grid = K1_SHIFT_TREND_FINE
            k2_grid = K2_SHIFT_TREND_FINE
        else:
            std1_grid = STD_SPANS_TREND_COARSE
            std2_grid = STD_SPANS_TREND_COARSE
            k1_grid = K1_SHIFT_TREND_COARSE
            k2_grid = K2_SHIFT_TREND_COARSE
        center_scale_modes = list(TREND_ANCHORS)
    return t1_grid, t2_grid, std1_grid, std2_grid, k1_grid, k2_grid, center_scale_modes


def weak_median_mad_result(best):
    if best is None:
        return True
    cal = best.get("calibration_total", np.nan)
    val = best.get("validation_total", np.nan)
    dd = abs(best.get("validation_max_drawdown", np.nan))
    pnl_abs = max(abs(val), 1e-12)
    return (not np.isfinite(cal)) or (not np.isfinite(val)) or cal <= 0 or val <= 0 or dd / pnl_abs > 1.0


def evaluate_grid(row, core_cal_arr, core_val_arr, core_cal, core_val, group, half_life, calibration_mode, center_scale_modes):
    t1_grid, t2_grid, std1_grid, std2_grid, k1_grid, k2_grid, _ = choose_grids(group, half_life, calibration_mode, row=row)
    opt1 = row["opt1"]
    opt2 = row["opt2"]
    rows = []
    best = None
    best_payload = None
    for t1 in t1_grid:
        for t2 in t2_grid:
            for std1 in std1_grid:
                for std2 in std2_grid:
                    for k1 in k1_grid:
                        for k2 in k2_grid:
                            for mode1 in center_scale_modes:
                                for mode2 in center_scale_modes:
                                    l1_cal, d1_cal, k1_cal = option_layer(core_cal_arr, opt1, t1, std1, k1, mode1)
                                    final_cal, d2_cal, k2_cal = option_layer(l1_cal, opt2, t2, std2, k2, mode2)
                                    l1_val, d1_val, k1_val = option_layer(core_val_arr, opt1, t1, std1, k1, mode1)
                                    final_val, d2_val, k2_val = option_layer(l1_val, opt2, t2, std2, k2, mode2)
                                    cal_m = metric_block(final_cal, "calibration")
                                    val_m = metric_block(final_val, "validation")
                                    s = score(cal_m, val_m, core_cal, core_val)
                                    out = {
                                        "spec_id": int(row["spec_id"]),
                                        "strategy_id": row["strategy_id"],
                                        "family": row.get("family", ""),
                                        "instrument": row.get("instrument", ""),
                                        "filter_group": row.get("filter_group", ""),
                                        "geometry_label": row.get("geometry_label", ""),
                                        "calibration_group": group,
                                        "calibration_mode": calibration_mode,
                                        "decision_type": "two_option_calibrated",
                                        "opt1": opt1,
                                        "opt2": opt2,
                                        "t1_bars": int(t1),
                                        "t2_bars": int(t2),
                                        "std1": int(std1),
                                        "std2": int(std2),
                                        "k1_shift_sigma": float(k1),
                                        "k2_shift_sigma": float(k2),
                                        "center_scale1": mode1,
                                        "center_scale2": mode2,
                                        "half_life_bars": half_life,
                                        "improvement_calibration": cal_m["calibration_total"] - core_cal["core_calibration_total"],
                                        "improvement_validation": val_m["validation_total"] - core_val["core_validation_total"],
                                        "selection_score": s,
                                    }
                                    out.update(core_cal)
                                    out.update(core_val)
                                    out.update(cal_m)
                                    out.update(val_m)
                                    rows.append(out)
                                    if best is None or s > best["selection_score"]:
                                        best = out
                                        best_payload = (l1_cal, final_cal, d1_cal, d2_cal, k1_cal, k2_cal, l1_val, final_val, d1_val, d2_val, k1_val, k2_val)
    return rows, best, best_payload


def evaluate_one(row, calibration_mode="coarse"):
    path = pd.read_parquet(row["path"])
    path["timestamp"] = pd.to_datetime(path["timestamp"])
    group = calibration_group(row)
    if group == "core_only":
        return evaluate_core_only(row, path)
    train_part = path[path["timestamp"] <= pd.Timestamp("2016-12-31 23:59:59")]
    half_life = infer_half_life_bars(train_part)

    cal = path[(path["timestamp"] >= CALIBRATION_START) & (path["timestamp"] <= CALIBRATION_END)].copy()
    val = path[(path["timestamp"] >= VALIDATION_START) & (path["timestamp"] <= VALIDATION_END)].copy()
    if len(cal) < 200 or len(val) < 200:
        return [], None
    core_cal_arr = cal["net_pnl"].to_numpy(dtype=float)
    core_val_arr = val["net_pnl"].to_numpy(dtype=float)
    core_cal = metric_block(core_cal_arr, "core_calibration")
    core_val = metric_block(core_val_arr, "core_validation")
    _, _, _, _, _, _, primary_modes = choose_grids(group, half_life, calibration_mode)
    rows, best, payload = evaluate_grid(row, core_cal_arr, core_val_arr, core_cal, core_val, group, half_life, calibration_mode, primary_modes)
    if group == "mean_reversion" and calibration_mode == "coarse" and weak_median_mad_result(best):
        fallback_rows, fallback_best, fallback_payload = evaluate_grid(row, core_cal_arr, core_val_arr, core_cal, core_val, group, half_life, calibration_mode, MR_CENTER_SCALE_FALLBACK)
        rows.extend(fallback_rows)
        if best is None or (fallback_best is not None and fallback_best["selection_score"] > best["selection_score"]):
            best = fallback_best
            payload = fallback_payload
    best_path = None
    if payload is not None:
        l1c, fc, d1c, d2c, k1c, k2c, l1v, fv, d1v, d2v, k1v, k2v = payload
        p1 = pd.DataFrame({
            "timestamp": cal["timestamp"].to_numpy(), "split": "calibration", "strategy_id": row["strategy_id"],
            "core_net_pnl": core_cal_arr, "layer1_pnl": l1c, "two_option_pnl": fc,
            "delta1": d1c, "delta2": d2c, "strike1": k1c, "strike2": k2c,
        })
        p2 = pd.DataFrame({
            "timestamp": val["timestamp"].to_numpy(), "split": "validation", "strategy_id": row["strategy_id"],
            "core_net_pnl": core_val_arr, "layer1_pnl": l1v, "two_option_pnl": fv,
            "delta1": d1v, "delta2": d2v, "strike1": k1v, "strike2": k2v,
        })
        best_path = pd.concat([p1, p2], ignore_index=True)
    return rows, best_path


def run_batch(batch_id=0, batch_size=DEFAULT_BATCH_SIZE, start_id=None, end_id=None, batch_label=None, calibration_mode="coarse"):
    ensure_dirs()
    selected = load_selected()
    start = int(batch_id) * int(batch_size) if start_id is None else int(start_id)
    end = min(start + int(batch_size), len(selected)) if end_id is None else min(int(end_id), len(selected))
    label = batch_label or f"{int(batch_id):04d}"
    part = selected.iloc[start:end].copy()
    grid_rows = []
    best_rows = []
    saved = []
    path_dir = BEST_PATH_DIR / f"batch_{label}"
    path_dir.mkdir(parents=True, exist_ok=True)
    for _, row in part.iterrows():
        rows, best_path = evaluate_one(row, calibration_mode=calibration_mode)
        grid_rows.extend(rows)
        if rows:
            best = max(rows, key=lambda x: x["selection_score"])
            best_rows.append(best)
            if best_path is not None:
                safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(row["strategy_id"]))
                p = path_dir / f"{int(row['spec_id']):05d}__{safe}.parquet"
                best_path.to_parquet(p, index=False)
                saved.append({"spec_id": int(row["spec_id"]), "strategy_id": row["strategy_id"], "best_path": str(p)})
    grid = pd.DataFrame(grid_rows)
    best = pd.DataFrame(best_rows)
    paths = pd.DataFrame(saved)
    grid.to_csv(BATCH_DIR / f"grid_batch_{label}.csv", index=False)
    best.to_csv(BATCH_DIR / f"best_batch_{label}.csv", index=False)
    paths.to_csv(BATCH_DIR / f"best_paths_batch_{label}.csv", index=False)
    manifest = {
        "batch_label": label,
        "start_row": start,
        "end_row_exclusive": end,
        "n_selected_total": int(len(selected)),
        "n_strategies": int(len(part)),
        "n_grid_rows": int(len(grid)),
        "n_best_rows": int(len(best)),
        "n_best_paths_saved": int(len(paths)),
        "batch_size": int(batch_size),
        "calibration_mode": calibration_mode,
    }
    (BATCH_DIR / f"manifest_batch_{label}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return grid, best, paths, manifest


def combine_batches():
    ensure_dirs()
    best_files = sorted(BATCH_DIR.glob("best_batch_*.csv"))
    path_files = sorted(BATCH_DIR.glob("best_paths_batch_*.csv"))
    best = pd.concat([pd.read_csv(p) for p in best_files], ignore_index=True) if best_files else pd.DataFrame()
    paths = pd.concat([pd.read_csv(p) for p in path_files], ignore_index=True) if path_files else pd.DataFrame()
    if not best.empty:
        best = best.drop_duplicates("spec_id", keep="last").sort_values("selection_score", ascending=False)
    if not paths.empty:
        paths = paths.drop_duplicates("spec_id", keep="last")
    best.to_csv(OUTPUT_DIR / "03_calibrated_best_by_strategy.csv", index=False)
    paths.to_csv(OUTPUT_DIR / "04_calibrated_best_paths.csv", index=False)
    summary = {
        "n_best_batch_files": len(best_files),
        "n_unique_strategies": int(len(best)),
        "n_best_paths": int(len(paths)),
        "output_dir": str(OUTPUT_DIR),
    }
    (OUTPUT_DIR / "05_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return best, paths, summary


def audit_batches():
    selected = load_selected()
    best_files = sorted(BATCH_DIR.glob("best_batch_*.csv"))
    raw = pd.concat([pd.read_csv(p) for p in best_files], ignore_index=True) if best_files else pd.DataFrame()
    if raw.empty:
        report = {"selected_rows": int(len(selected)), "processed_unique": 0, "missing_count": int(len(selected))}
    else:
        missing = sorted(set(selected["spec_id"]) - set(raw["spec_id"]))
        dup = sorted(raw.loc[raw["spec_id"].duplicated(), "spec_id"].astype(int).unique().tolist())
        report = {
            "selected_rows": int(len(selected)),
            "processed_rows": int(len(raw)),
            "processed_unique": int(raw["spec_id"].nunique()),
            "missing_count": int(len(missing)),
            "duplicate_count": int(len(dup)),
            "missing_spec_id_ranges": ranges(missing),
            "duplicate_head": dup[:20],
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def summary():
    selected = load_selected()
    print("Selected for calibration (отобрано для калибровки):", len(selected))
    print("Batch size (размер пакета):", DEFAULT_BATCH_SIZE)
    print("Batch ids (номера пакетов): 0 ..", math.ceil(len(selected) / DEFAULT_BATCH_SIZE) - 1)
    print("Coarse trend tuples per strategy (грубые трендовые кортежи на стратегию):", len(FIXED_T_BARS_TREND_COARSE) ** 2 * len(STD_SPANS_TREND_COARSE) ** 2 * len(K1_SHIFT_TREND_COARSE) * len(K2_SHIFT_TREND_COARSE))
    print("Coarse mean reversion primary tuples per strategy (грубые основные кортежи возврата к среднему на стратегию): about", "5*5*3*3 = 225 before unique half-life clipping")
    print(selected.groupby("filter_group").size().sort_values(ascending=False).to_string())
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["summary", "batch", "combine", "audit"], default="summary")
    parser.add_argument("--batch-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--start-id", type=int, default=None)
    parser.add_argument("--end-id", type=int, default=None)
    parser.add_argument("--batch-label", type=str, default=None)
    parser.add_argument("--calibration-mode", choices=["coarse", "fine"], default="fine")
    args = parser.parse_args()
    if args.mode == "summary":
        summary()
    elif args.mode == "batch":
        run_batch(args.batch_id, args.batch_size, args.start_id, args.end_id, args.batch_label, args.calibration_mode)
    elif args.mode == "combine":
        combine_batches()
    elif args.mode == "audit":
        audit_batches()

