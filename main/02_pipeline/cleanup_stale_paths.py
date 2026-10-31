from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


OUTPUT_ROOT = Path(r"D:\Filter_2_option_v2")
FACTORY_DIR = OUTPUT_ROOT / "01_core_strategy_factory_output"
CORE_PATHS_DIR = FACTORY_DIR / "paths_selected"
CORE_PATHS_CSV = FACTORY_DIR / "06_all_core_strategy_paths.csv"
FINE_SELECTED_CSV = OUTPUT_ROOT / "05_select_for_fine_calibration_output" / "02d_selected_for_fine_with_rescue.csv"
FULL_SELECTION_CSV = OUTPUT_ROOT / "03_selection_leaderboard_output" / "01_full_selection_leaderboard.csv"
TEST_RESULTS_CSV = OUTPUT_ROOT / "08_test_run_output" / "01_test_strategy_results.csv"
TEST_PATHS_DIR = OUTPUT_ROOT / "08_test_run_output" / "paths"
OUTPUT_DIR = OUTPUT_ROOT / "10_cleanup_stale_paths_output"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_ids(path: Path, column: str = "spec_id") -> set[int]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    df = pd.read_csv(path, usecols=[column])
    return set(df[column].dropna().astype(int))


def keep_spec_ids() -> set[int]:
    keep = set()
    keep |= read_ids(FINE_SELECTED_CSV)
    if FULL_SELECTION_CSV.exists():
        full = pd.read_csv(FULL_SELECTION_CSV, usecols=["spec_id", "self_sufficient_core"])
        keep |= set(full.loc[full["self_sufficient_core"].astype(bool), "spec_id"].astype(int))
    keep |= read_ids(TEST_RESULTS_CSV)
    return keep


def audit_core_paths(keep_ids: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    catalog = pd.read_csv(CORE_PATHS_CSV)
    catalog["spec_id"] = catalog["spec_id"].astype(int)
    rows = []
    for _, row in catalog.iterrows():
        p = Path(row["path"])
        exists = p.exists()
        size = p.stat().st_size if exists else 0
        rows.append(
            {
                "category": "core_strategy_path",
                "spec_id": int(row["spec_id"]),
                "strategy_id": row.get("strategy_id", ""),
                "path": str(p),
                "exists": bool(exists),
                "size_bytes": int(size),
                "keep": int(row["spec_id"]) in keep_ids,
                "delete_candidate": exists and int(row["spec_id"]) not in keep_ids,
            }
        )
    audit = pd.DataFrame(rows)
    delete_plan = audit[audit["delete_candidate"]].copy()
    return audit, delete_plan


def audit_old_best_paths() -> pd.DataFrame:
    candidates = [
        OUTPUT_ROOT / "02_two_option_filter_output" / "best_paths",
        OUTPUT_ROOT / "04_two_option_calibration_output" / "best_paths",
    ]
    rows = []
    for root in candidates:
        for p in root.rglob("*.parquet") if root.exists() else []:
            rows.append(
                {
                    "category": root.parent.name + "_best_paths",
                    "spec_id": None,
                    "strategy_id": "",
                    "path": str(p),
                    "exists": True,
                    "size_bytes": int(p.stat().st_size),
                    "keep": False,
                    "delete_candidate": True,
                }
            )
    return pd.DataFrame(rows)


def audit_empty_dirs() -> pd.DataFrame:
    rows = []
    for root in [CORE_PATHS_DIR, OUTPUT_ROOT / "02_two_option_filter_output" / "best_paths", OUTPUT_ROOT / "04_two_option_calibration_output" / "best_paths"]:
        if not root.exists():
            continue
        for d in sorted([x for x in root.rglob("*") if x.is_dir()], key=lambda x: len(x.parts), reverse=True):
            rows.append({"path": str(d)})
    return pd.DataFrame(rows)


def summarize(plan: pd.DataFrame, keep_ids: set[int]) -> dict:
    return {
        "keep_spec_ids": int(len(keep_ids)),
        "delete_files": int(len(plan)),
        "delete_size_gb": round(float(plan["size_bytes"].sum()) / 1024**3, 3) if not plan.empty else 0.0,
        "core_paths_dir": str(CORE_PATHS_DIR),
        "test_paths_dir": str(TEST_PATHS_DIR),
    }


def run_cleanup(include_old_best_paths: bool = True, execute: bool = False) -> dict:
    ensure_dirs()
    keep_ids = keep_spec_ids()
    core_audit, core_delete = audit_core_paths(keep_ids)
    parts = [core_delete]
    old_best = audit_old_best_paths() if include_old_best_paths else pd.DataFrame()
    if include_old_best_paths and not old_best.empty:
        parts.append(old_best)
    plan = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    core_audit.to_csv(OUTPUT_DIR / "01_core_paths_audit.csv", index=False)
    plan.to_csv(OUTPUT_DIR / "02_delete_plan.csv", index=False)
    pd.DataFrame({"spec_id": sorted(keep_ids)}).to_csv(OUTPUT_DIR / "03_keep_spec_ids.csv", index=False)

    deleted = []
    failed = []
    if execute and not plan.empty:
        for _, row in plan.iterrows():
            p = Path(row["path"])
            try:
                if p.exists() and p.is_file():
                    size = p.stat().st_size
                    p.unlink()
                    deleted.append({"path": str(p), "size_bytes": int(size)})
            except Exception as exc:
                failed.append({"path": str(p), "error": repr(exc)})
        for _, row in audit_empty_dirs().iterrows():
            d = Path(row["path"])
            try:
                if d.exists() and d.is_dir() and not any(d.iterdir()):
                    d.rmdir()
            except Exception:
                pass

    pd.DataFrame(deleted, columns=["path", "size_bytes"]).to_csv(OUTPUT_DIR / "04_deleted_files.csv", index=False)
    pd.DataFrame(failed, columns=["path", "error"]).to_csv(OUTPUT_DIR / "05_delete_failures.csv", index=False)

    manifest = summarize(plan, keep_ids)
    manifest.update(
        {
            "execute": bool(execute),
            "include_old_best_paths": bool(include_old_best_paths),
            "deleted_files": int(len(deleted)),
            "deleted_size_gb": round(sum(x["size_bytes"] for x in deleted) / 1024**3, 3) if deleted else 0.0,
            "failed_deletes": int(len(failed)),
            "output_dir": str(OUTPUT_DIR),
            "saved_files": [
                "01_core_paths_audit.csv",
                "02_delete_plan.csv",
                "03_keep_spec_ids.csv",
                "04_deleted_files.csv",
                "05_delete_failures.csv",
                "06_manifest.json",
            ],
        }
    )
    with (OUTPUT_DIR / "06_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--keep-old-best-paths", action="store_true")
    args = parser.parse_args()
    manifest = run_cleanup(include_old_best_paths=not args.keep_old_best_paths, execute=args.execute)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
