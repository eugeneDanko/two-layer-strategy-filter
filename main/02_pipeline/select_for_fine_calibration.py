from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

OUTPUT_ROOT = Path(r"D:\Filter_2_option_v2")
CALIBRATION_OUTPUT = OUTPUT_ROOT / "04_two_option_calibration_output"
OUTPUT_DIR = OUTPUT_ROOT / "05_select_for_fine_calibration_output"

CALIBRATED_BEST_CSV = CALIBRATION_OUTPUT / "03_calibrated_best_by_strategy.csv"
CALIBRATED_PATHS_CSV = CALIBRATION_OUTPUT / "04_calibrated_best_paths.csv"

SELECT_FRACTION_PER_GROUP = 0.60
MAX_FINE_PER_GROUP = {
    "put/call": 120,
    "call/call": 80,
    "call/put": 80,
    "put/put": 60,
    "mean_reversion": 120,
}
MAX_RESCUE_MEAN_REVERSION = 120


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_ratio(num, den):
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-12:
        return np.nan
    return float(num / den)


def load_inputs():
    if not CALIBRATED_BEST_CSV.exists():
        raise FileNotFoundError(f"Сначала нужен coarse calibration result (результат грубой калибровки): {CALIBRATED_BEST_CSV}")
    best = pd.read_csv(CALIBRATED_BEST_CSV)
    paths = pd.read_csv(CALIBRATED_PATHS_CSV) if CALIBRATED_PATHS_CSV.exists() else pd.DataFrame(columns=["spec_id", "best_path"])
    if "spec_id" in paths.columns:
        paths = paths.drop_duplicates("spec_id")
        best = best.merge(paths[["spec_id", "best_path"]], on="spec_id", how="left")
    return best


def add_scores(df):
    df = df.copy()
    df = df[df.get("calibration_group", "") != "core_only"].copy()
    df["validation_positive"] = df["validation_total"] > 0
    df["calibration_positive"] = df["calibration_total"] > 0
    df["improvement_positive"] = df["improvement_validation"] > 0
    df["validation_drawdown_to_pnl"] = [
        safe_ratio(abs(dd), abs(pnl)) for dd, pnl in zip(df["validation_max_drawdown"], df["validation_total"])
    ]
    df["calibration_drawdown_to_pnl"] = [
        safe_ratio(abs(dd), abs(pnl)) for dd, pnl in zip(df["calibration_max_drawdown"], df["calibration_total"])
    ]
    df["drawdown_ok"] = df["validation_drawdown_to_pnl"].fillna(np.inf) <= 1.5
    df["shape_ok"] = (
        (df.get("calibration_group", "").astype(str) == "mean_reversion") |
        (df.get("validation_total", 0) * df.get("calibration_total", 0) > 0)
    )
    df["fine_group"] = np.where(
        df.get("calibration_group", "").astype(str) == "mean_reversion",
        "mean_reversion",
        df["filter_group"].astype(str),
    )
    df["mean_reversion_core_pool"] = (
        (df.get("calibration_group", "").astype(str) == "mean_reversion")
        | df.get("geometry_label", "").astype(str).str.contains("mean_reversion", na=False)
        | df["filter_group"].astype(str).isin(["call/put", "put/put"])
    )
    df["fine_rank_score"] = (
        2.0 * df["validation_positive"].astype(float)
        + 1.5 * df["improvement_positive"].astype(float)
        + 1.0 * df["drawdown_ok"].astype(float)
        + 1.0 * df["shape_ok"].astype(float)
        + 0.8 * df["validation_total"].rank(pct=True)
        + 0.8 * df["improvement_validation"].rank(pct=True)
        + 0.5 * df["validation_sharpe"].replace([np.inf, -np.inf], np.nan).fillna(-999).rank(pct=True)
        + 0.5 * df["validation_sortino"].replace([np.inf, -np.inf], np.nan).fillna(-999).rank(pct=True)
        - 0.8 * df["validation_drawdown_to_pnl"].replace([np.inf, -np.inf], np.nan).fillna(999).rank(pct=True)
    )
    df["fine_candidate"] = (
        df["calibration_positive"]
        & (df["validation_positive"] | df["improvement_positive"])
        & df["shape_ok"]
    )
    # Rescue (спасение) нужно для mean reversion (возврата к среднему):
    # coarse calibration (грубая калибровка) могла промахнуться по T (экспирации)
    # или center/scale (центру/масштабу), поэтому не выбрасываем устойчивые формы сразу.
    df["near_break_even_validation"] = df["validation_total"].fillna(-np.inf) >= -0.05
    df["near_break_even_improvement"] = df["improvement_validation"].fillna(-np.inf) >= -0.05
    df["has_half_life"] = df.get("half_life_bars", pd.Series(index=df.index, dtype=float)).notna()
    df["rescue_candidate"] = (
        df["mean_reversion_core_pool"]
        & ~df["fine_candidate"]
        & (
            df["has_half_life"]
            | df["shape_ok"]
            | df["near_break_even_validation"]
            | df["near_break_even_improvement"]
        )
        & (df["validation_drawdown_to_pnl"].fillna(np.inf) <= 3.0)
    )
    return df


def select_for_fine():
    ensure_dirs()
    df = add_scores(load_inputs())
    ranked_parts = []
    selected_parts = []
    trend_selected_parts = []
    mean_reversion_selected_parts = []
    for group, part in df.groupby("fine_group", dropna=False):
        part = part.sort_values(
            [
                "fine_candidate",
                "validation_positive",
                "improvement_positive",
                "drawdown_ok",
                "fine_rank_score",
                "validation_total",
                "improvement_validation",
            ],
            ascending=[False, False, False, False, False, False, False],
        ).copy()
        part["rank_in_fine_group"] = np.arange(1, len(part) + 1)
        top_n = int(np.ceil(len(part) * SELECT_FRACTION_PER_GROUP))
        top_n = min(top_n, MAX_FINE_PER_GROUP.get(str(group), top_n))
        part["selected_for_fine"] = part["fine_candidate"] & (part["rank_in_fine_group"] <= top_n)
        ranked_parts.append(part)
        chosen = part[part["selected_for_fine"]].copy()
        selected_parts.append(chosen)
        if str(group) == "mean_reversion":
            mean_reversion_selected_parts.append(chosen)
        else:
            trend_selected_parts.append(chosen)

    ranked = pd.concat(ranked_parts, ignore_index=True) if ranked_parts else pd.DataFrame()
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    trend_selected = pd.concat(trend_selected_parts, ignore_index=True) if trend_selected_parts else pd.DataFrame()
    mean_reversion_selected = pd.concat(mean_reversion_selected_parts, ignore_index=True) if mean_reversion_selected_parts else pd.DataFrame()
    rescue = ranked[ranked["rescue_candidate"]].sort_values(
        ["has_half_life", "shape_ok", "near_break_even_validation", "near_break_even_improvement", "fine_rank_score"],
        ascending=[False, False, False, False, False],
    ).head(MAX_RESCUE_MEAN_REVERSION).copy() if not ranked.empty else pd.DataFrame()
    selected_with_rescue = pd.concat([selected, rescue], ignore_index=True).drop_duplicates("spec_id", keep="first") if not selected.empty or not rescue.empty else pd.DataFrame()
    summary = ranked.groupby("fine_group").agg(
        n=("spec_id", "size"),
        fine_candidates=("fine_candidate", "sum"),
        rescue_candidates=("rescue_candidate", "sum"),
        selected_for_fine=("selected_for_fine", "sum"),
        median_validation_total=("validation_total", "median"),
        median_improvement_validation=("improvement_validation", "median"),
        median_validation_sharpe=("validation_sharpe", "median"),
        median_validation_sortino=("validation_sortino", "median"),
    ).reset_index() if not ranked.empty else pd.DataFrame()

    ranked.to_csv(OUTPUT_DIR / "01_fine_selection_leaderboard.csv", index=False)
    selected.to_csv(OUTPUT_DIR / "02_selected_for_fine.csv", index=False)
    trend_selected.to_csv(OUTPUT_DIR / "02a_selected_for_fine_trend.csv", index=False)
    mean_reversion_selected.to_csv(OUTPUT_DIR / "02b_selected_for_fine_mean_reversion.csv", index=False)
    rescue.to_csv(OUTPUT_DIR / "02c_mean_reversion_rescue_candidates.csv", index=False)
    selected_with_rescue.to_csv(OUTPUT_DIR / "02d_selected_for_fine_with_rescue.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "03_fine_selection_group_summary.csv", index=False)
    manifest = {
        "input_file": str(CALIBRATED_BEST_CSV),
        "n_ranked": int(len(ranked)),
        "n_selected_for_fine": int(len(selected)),
        "n_selected_trend": int(len(trend_selected)),
        "n_selected_mean_reversion": int(len(mean_reversion_selected)),
        "n_mean_reversion_rescue": int(len(rescue)),
        "n_selected_with_rescue": int(len(selected_with_rescue)),
        "select_fraction_per_group": SELECT_FRACTION_PER_GROUP,
        "max_fine_per_group": MAX_FINE_PER_GROUP,
        "max_rescue_mean_reversion": MAX_RESCUE_MEAN_REVERSION,
        "output_dir": str(OUTPUT_DIR),
    }
    (OUTPUT_DIR / "04_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return ranked, selected, summary, manifest


def audit():
    ensure_dirs()
    files = {
        "input_calibrated_best": CALIBRATED_BEST_CSV,
        "input_calibrated_paths": CALIBRATED_PATHS_CSV,
        "leaderboard": OUTPUT_DIR / "01_fine_selection_leaderboard.csv",
        "selected": OUTPUT_DIR / "02_selected_for_fine.csv",
        "selected_trend": OUTPUT_DIR / "02a_selected_for_fine_trend.csv",
        "selected_mean_reversion": OUTPUT_DIR / "02b_selected_for_fine_mean_reversion.csv",
        "mean_reversion_rescue": OUTPUT_DIR / "02c_mean_reversion_rescue_candidates.csv",
        "selected_with_rescue": OUTPUT_DIR / "02d_selected_for_fine_with_rescue.csv",
        "summary": OUTPUT_DIR / "03_fine_selection_group_summary.csv",
        "manifest": OUTPUT_DIR / "04_manifest.json",
    }
    report = {k: {"exists": v.exists(), "size_bytes": v.stat().st_size if v.exists() else 0} for k, v in files.items()}
    if files["leaderboard"].exists():
        lb = pd.read_csv(files["leaderboard"])
        report["leaderboard_rows"] = int(len(lb))
        report["leaderboard_unique_spec_ids"] = int(lb["spec_id"].nunique()) if "spec_id" in lb else 0
    if files["selected"].exists():
        sel = pd.read_csv(files["selected"])
        report["selected_rows"] = int(len(sel))
        report["selected_unique_spec_ids"] = int(sel["spec_id"].nunique()) if "spec_id" in sel else 0
        report["selected_by_group"] = sel.groupby("fine_group").size().to_dict() if "fine_group" in sel else {}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["select", "audit"], default="select")
    args = parser.parse_args()
    if args.mode == "select":
        select_for_fine()
    else:
        audit()
