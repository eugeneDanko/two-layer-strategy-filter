from pathlib import Path
import json
import math
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yaml
except ImportError:
    yaml = None

try:
    import numba as nb
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False


DATA_ROOT = Path(r"./data")
WORK_ROOT = Path(r"./workspace")
OUTPUT_ROOT = Path(r"D:\Filter_2_option_v2")
OUTPUT_DIR = OUTPUT_ROOT / "01_core_strategy_factory_output"
PATH_DIR = OUTPUT_DIR / "paths_selected"

DATA_PATH = DATA_ROOT / "dfs_5min.pickle"
PANEL_CACHE = OUTPUT_DIR / "01_active_close_panel_30min.parquet"
PROTOCOL_PATH = WORK_ROOT / "backtest_protocol.yaml"
if not PROTOCOL_PATH.exists():
    PROTOCOL_PATH = DATA_ROOT / "backtest_protocol.yaml"

COST_PER_UNIT_TURNOVER = 0.00005

SECTOR_MAP = {
    "energy": ["CL", "RB", "HO", "NG"],
    "grains_oilseeds": ["ZC", "ZS", "ZM", "ZL", "ZW", "KE", "ZO", "RS"],
    "livestock": ["LE", "GF", "HE"],
    "fx": ["E6", "B6", "A6", "J6", "S6", "N6", "D6", "DX"],
    "softs": ["SB", "CC"],
}

PRIORITY_PAIRS = [
    ("RB", "CL"), ("HO", "CL"), ("RB", "HO"), ("NG", "CL"),
    ("ZW", "KE"), ("ZM", "ZS"), ("ZL", "ZS"), ("RS", "ZS"), ("RS", "ZL"), ("ZC", "ZW"), ("ZC", "ZS"),
    ("LE", "GF"), ("HE", "LE"),
    ("DX", "E6"), ("DX", "B6"), ("E6", "B6"), ("A6", "N6"), ("S6", "E6"), ("J6", "DX"),
]

LOOKBACKS = [22, 44, 88, 176, 352]
FAST_SLOW = [(22, 88), (44, 176), (88, 352)]
ZSCORE_WINDOWS = [88, 176, 352]
ZSCORE_THRESHOLDS = [0.5, 1.0, 1.5]


def ensure_dirs():
    for p in [WORK_ROOT, OUTPUT_DIR, PATH_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def read_protocol():
    global COST_PER_UNIT_TURNOVER
    if PROTOCOL_PATH.exists() and yaml is not None:
        with PROTOCOL_PATH.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {
            "splits": {
                "train": {"start": "2008-05-01", "end": "2016-12-31 23:59:59"},
                "calibration": {"start": "2017-01-01", "end": "2019-12-31 23:59:59"},
                "test": {"start": "2020-01-01", "end": "2023-08-10 23:59:59"},
            },
            "cost_per_unit_turnover": 0.00005,
        }
    COST_PER_UNIT_TURNOVER = float(cfg.get("cost_per_unit_turnover", 0.00005))
    splits_raw = cfg.get("splits", cfg)
    splits = {k: {"start": pd.Timestamp(v["start"]), "end": pd.Timestamp(v["end"])} for k, v in splits_raw.items()}
    return cfg, splits


PROTOCOL, SPLITS = read_protocol()


if HAS_NUMBA:
    @nb.njit(cache=True)
    def pnl_from_signal(price, signal, cost):
        n = price.shape[0]
        gross = np.zeros(n, dtype=np.float64)
        net = np.zeros(n, dtype=np.float64)
        pos = np.zeros(n, dtype=np.float64)
        last_sig = 0.0
        for i in range(1, n):
            sig = signal[i - 1]
            if not np.isfinite(sig):
                sig = 0.0
            if sig > 1.0:
                sig = 1.0
            elif sig < -1.0:
                sig = -1.0
            ret = 0.0
            if np.isfinite(price[i]) and np.isfinite(price[i - 1]) and price[i - 1] != 0.0:
                ret = math.log(price[i] / price[i - 1])
            turnover = abs(sig - last_sig)
            gross[i] = gross[i - 1] + sig * ret
            net[i] = net[i - 1] + sig * ret - cost * turnover
            pos[i] = sig
            last_sig = sig
        return gross, net, pos
else:
    def pnl_from_signal(price, signal, cost):
        price_s = pd.Series(price)
        sig = pd.Series(signal).clip(-1, 1).fillna(0.0)
        ret = np.log(price_s / price_s.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        pos = sig.shift(1).fillna(0.0)
        turnover = pos.diff().abs().fillna(0.0)
        gross = (pos * ret).cumsum().to_numpy()
        net = (pos * ret - cost * turnover).cumsum().to_numpy()
        return gross, net, pos.to_numpy()


def load_or_build_panel(freq="30min"):
    ensure_dirs()
    if PANEL_CACHE.exists():
        panel = pd.read_parquet(PANEL_CACHE)
        panel.index = pd.to_datetime(panel.index)
        return panel.sort_index()
    old_panel = DATA_ROOT / "priority_groups_stat_arb_audit" / "04_active_close_panel_30min.parquet"
    if freq == "30min" and old_panel.exists():
        panel = pd.read_parquet(old_panel)
        panel.index = pd.to_datetime(panel.index)
        panel.sort_index().to_parquet(PANEL_CACHE)
        return panel.sort_index()
    with DATA_PATH.open("rb") as f:
        dfs = pickle.load(f)
    series = []
    for market, df in dfs.items():
        tmp = df.copy()
        tmp.columns = [str(c).strip() for c in tmp.columns]
        if "Date" not in tmp.columns or "Close" not in tmp.columns:
            continue
        tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce")
        s = tmp.dropna(subset=["Date"]).set_index("Date")["Close"].astype(float).sort_index()
        series.append(s.resample(freq).last().rename(market))
    panel = pd.concat(series, axis=1).sort_index()
    panel.to_parquet(PANEL_CACHE)
    return panel


def safe_zscore(s, window):
    m = s.rolling(window, min_periods=max(10, window // 3)).mean()
    sd = s.rolling(window, min_periods=max(10, window // 3)).std()
    return (s - m) / sd.replace(0, np.nan)


def make_strategy_path(strategy_id, family, instrument, price_like, signal):
    idx = price_like.index
    gross, net, pos = pnl_from_signal(
        price_like.to_numpy(dtype=np.float64),
        signal.reindex(idx).to_numpy(dtype=np.float64),
        COST_PER_UNIT_TURNOVER,
    )
    return pd.DataFrame({
        "timestamp": idx,
        "strategy_id": strategy_id,
        "family": family,
        "instrument": instrument,
        "price_like": price_like.to_numpy(dtype=np.float64),
        "signal": signal.reindex(idx).to_numpy(dtype=np.float64),
        "position": pos,
        "gross_pnl": gross,
        "net_pnl": net,
    })


def max_drawdown(x):
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return np.nan
    peak = np.maximum.accumulate(arr)
    return float(np.nanmin(arr - peak))


def slope_r2(y):
    arr = np.asarray(y, dtype=float)
    ok = np.isfinite(arr)
    if ok.sum() < 20:
        return np.nan, np.nan
    yy = arr[ok]
    xx = np.arange(yy.size, dtype=float)
    xx -= xx.mean()
    yy0 = yy - yy.mean()
    den = float(np.dot(xx, xx))
    if den == 0:
        return np.nan, np.nan
    b = float(np.dot(xx, yy0) / den)
    fit = yy.mean() + b * xx
    ss_res = float(np.sum((yy - fit) ** 2))
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    return b, 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def half_life_bars(series):
    s = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 200:
        return np.nan
    y = s.diff().dropna()
    x = s.shift(1).reindex(y.index)
    if x.std() == 0:
        return np.nan
    beta = np.polyfit(x.to_numpy(), y.to_numpy(), 1)[0]
    if beta >= 0:
        return np.nan
    return float(-np.log(2) / beta)


def metrics_for_path(path):
    out = {}
    for split_name, bounds in SPLITS.items():
        part = path[(path["timestamp"] >= bounds["start"]) & (path["timestamp"] <= bounds["end"])]
        if part.empty:
            out[f"{split_name}_n_obs"] = 0
            continue
        gross = part["gross_pnl"].to_numpy()
        net = part["net_pnl"].to_numpy()
        pos = part["position"].to_numpy()
        step = np.diff(net, prepend=net[0])
        slope, r2 = slope_r2(net)
        gross_total = float(gross[-1] - gross[0])
        net_total = float(net[-1] - net[0])
        cost_gap = gross_total - net_total
        out.update({
            f"{split_name}_n_obs": int(part.shape[0]),
            f"{split_name}_gross_total": gross_total,
            f"{split_name}_net_total": net_total,
            f"{split_name}_cost_gap": cost_gap,
            f"{split_name}_cost_gap_to_abs_gross": float(cost_gap / max(abs(gross_total), 1e-12)),
            f"{split_name}_max_drawdown": max_drawdown(net - net[0]),
            f"{split_name}_slope": slope,
            f"{split_name}_trend_r2": r2,
            f"{split_name}_step_mean": float(np.nanmean(step)),
            f"{split_name}_step_std": float(np.nanstd(step)),
            f"{split_name}_turnover": float(np.nansum(np.abs(np.diff(pos, prepend=0.0)))),
            f"{split_name}_half_life_bars": half_life_bars(net - np.linspace(net[0], net[-1], len(net))),
        })
    return out


def beta_ols(y, x):
    tmp = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    bounds = SPLITS["train"]
    tmp = tmp[(tmp.index >= bounds["start"]) & (tmp.index <= bounds["end"])]
    if len(tmp) < 500 or tmp["x"].std() == 0:
        return np.nan
    return float(np.polyfit(tmp["x"].to_numpy(), tmp["y"].to_numpy(), 1)[0])


def candidate_pairs(panel):
    pairs = set()
    for a, b in PRIORITY_PAIRS:
        if a in panel.columns and b in panel.columns:
            pairs.add((a, b))
    daily = np.log(panel).resample("1D").last().diff()
    corr = daily.corr(min_periods=250)
    cols = list(panel.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            c = corr.loc[a, b]
            if np.isfinite(c) and abs(c) >= 0.35:
                pairs.add((a, b))
    return sorted(pairs)


def generate_single_asset_strategies(panel):
    for market in panel.columns:
        px = panel[market].dropna()
        if len(px) < 1000:
            continue
        log_px = np.log(px)
        for lb in LOOKBACKS:
            mom = log_px.diff(lb)
            yield make_strategy_path(f"{market}__ts_momentum__lb_{lb}", "time_series_momentum", market, px, np.sign(mom))
            yield make_strategy_path(f"{market}__ts_reversal__lb_{lb}", "time_series_reversal", market, px, -np.sign(mom))
        for fast, slow in FAST_SLOW:
            sig = np.sign(px.rolling(fast, min_periods=fast).mean() - px.rolling(slow, min_periods=slow).mean())
            yield make_strategy_path(f"{market}__ma_cross__fast_{fast}__slow_{slow}", "moving_average_crossover", market, px, sig)
        for window in ZSCORE_WINDOWS:
            high = px.rolling(window, min_periods=window).max().shift(1)
            low = px.rolling(window, min_periods=window).min().shift(1)
            breakout = pd.Series(0.0, index=px.index)
            breakout[px > high] = 1.0
            breakout[px < low] = -1.0
            yield make_strategy_path(f"{market}__channel_breakout__window_{window}", "channel_breakout", market, px, breakout)
            yield make_strategy_path(f"{market}__channel_reversal__window_{window}", "channel_reversal", market, px, -breakout)


def generate_pair_strategies(panel):
    for y_m, x_m in candidate_pairs(panel):
        px = panel[[y_m, x_m]].dropna()
        if len(px) < 1500:
            continue
        log_y = np.log(px[y_m])
        log_x = np.log(px[x_m])
        beta = beta_ols(log_y, log_x)
        if not np.isfinite(beta):
            continue
        spread = (log_y - beta * log_x).rename(f"{y_m}_{x_m}_spread")
        instrument = f"{y_m}/{x_m}"
        for window in ZSCORE_WINDOWS:
            z = safe_zscore(spread, window)
            for threshold in ZSCORE_THRESHOLDS:
                mr = pd.Series(0.0, index=spread.index)
                mr[z > threshold] = -1.0
                mr[z < -threshold] = 1.0
                yield make_strategy_path(f"{y_m}_{x_m}__spread_mean_reversion__window_{window}__z_{threshold}", "pair_spread_mean_reversion", instrument, spread, mr)
                yield make_strategy_path(f"{y_m}_{x_m}__spread_momentum__window_{window}__z_{threshold}", "pair_spread_momentum", instrument, spread, -mr)


def generate_sector_strategies(panel):
    for sector, markets in SECTOR_MAP.items():
        cols = [m for m in markets if m in panel.columns]
        if len(cols) < 2:
            continue
        sub = panel[cols].dropna(how="all")
        returns = np.log(sub).diff()
        price_like = returns.mean(axis=1).fillna(0.0).cumsum()
        for lb in LOOKBACKS:
            past = np.log(sub).diff(lb)
            trend = np.sign(past).mean(axis=1)
            yield make_strategy_path(f"{sector}__sector_trend__lb_{lb}", "sector_trend_portfolio", sector, price_like, trend)
            yield make_strategy_path(f"{sector}__sector_reversal__lb_{lb}", "sector_reversal_portfolio", sector, price_like, -trend)
            rank = past.rank(axis=1, pct=True)
            cs_mom = ((rank > 0.67).astype(float) - (rank < 0.33).astype(float)).mean(axis=1)
            yield make_strategy_path(f"{sector}__cross_sectional_momentum__lb_{lb}", "sector_cross_sectional_momentum", sector, price_like, cs_mom)
            yield make_strategy_path(f"{sector}__cross_sectional_reversal__lb_{lb}", "sector_cross_sectional_reversal", sector, price_like, -cs_mom)


def geometry_label(row):
    if row.get("train_net_total", 0) > 0 and row.get("train_trend_r2", 0) >= 0.5:
        return "positive_trend_candidate"
    if row.get("train_net_total", 0) < 0 and row.get("train_trend_r2", 0) >= 0.5:
        return "negative_trend_candidate"
    if np.isfinite(row.get("train_half_life_bars", np.nan)):
        return "mean_reversion_shape_candidate"
    return "noisy_or_optional_candidate"


def run_factory(save_paths=True):
    ensure_dirs()
    panel = load_or_build_panel("30min")
    panel = panel.loc[:, panel.notna().sum().sort_values(ascending=False).index]
    panel.to_parquet(PANEL_CACHE)

    market_rows = []
    for col in panel.columns:
        s = panel[col].dropna()
        market_rows.append({
            "market": col,
            "n_obs": int(s.shape[0]),
            "start": s.index.min(),
            "end": s.index.max(),
            "sector": next((sector for sector, xs in SECTOR_MAP.items() if col in xs), "other"),
        })
    pd.DataFrame(market_rows).sort_values(["sector", "market"]).to_csv(OUTPUT_DIR / "02_market_universe.csv", index=False)

    all_metrics = []
    cache = {}
    for generator_name, generator in [
        ("single_asset", generate_single_asset_strategies(panel)),
        ("pair", generate_pair_strategies(panel)),
        ("sector", generate_sector_strategies(panel)),
    ]:
        before = len(all_metrics)
        for path in generator:
            row = {
                "strategy_id": path["strategy_id"].iloc[0],
                "family": path["family"].iloc[0],
                "instrument": path["instrument"].iloc[0],
            }
            row.update(metrics_for_path(path))
            all_metrics.append(row)
            v = row.get("train_net_total", np.nan)
            if np.isfinite(v):
                cache[row["strategy_id"]] = (abs(v), path)
        print(f"{generator_name}: {len(all_metrics) - before} strategies")

    metrics = pd.DataFrame(all_metrics)
    metrics["train_sharpe_like"] = metrics["train_step_mean"] / metrics["train_step_std"].replace(0, np.nan) * np.sqrt(44 * 252)
    metrics["same_sign_train_calibration"] = np.sign(metrics["train_net_total"]) == np.sign(metrics["calibration_net_total"])
    metrics["same_sign_all_splits"] = (
        (np.sign(metrics["train_net_total"]) == np.sign(metrics["calibration_net_total"])) &
        (np.sign(metrics["train_net_total"]) == np.sign(metrics["test_net_total"]))
    )
    metrics["cost_dominated_train"] = metrics["train_cost_gap_to_abs_gross"] > 0.6
    metrics["abs_train_net_total"] = metrics["train_net_total"].abs()
    metrics["geometry_label"] = metrics.apply(geometry_label, axis=1)
    metrics = metrics.sort_values(["abs_train_net_total", "train_trend_r2"], ascending=[False, False])

    metrics.to_csv(OUTPUT_DIR / "03_core_strategy_train_metrics.csv", index=False)
    metrics[["strategy_id", "family", "instrument", "geometry_label"]].to_csv(OUTPUT_DIR / "04_core_strategy_catalog.csv", index=False)

    limits = {
        "positive_trend_candidate": 80,
        "negative_trend_candidate": 80,
        "mean_reversion_shape_candidate": 80,
        "noisy_or_optional_candidate": 40,
    }
    selected_ids = []
    for label, limit in limits.items():
        part = metrics[metrics["geometry_label"] == label].sort_values(
            ["cost_dominated_train", "abs_train_net_total", "train_trend_r2"],
            ascending=[True, False, False],
        )
        selected_ids.extend(part["strategy_id"].head(limit).tolist())
    selected = metrics[metrics["strategy_id"].isin(selected_ids)].copy()
    selected.to_csv(OUTPUT_DIR / "05_selected_core_strategies_for_two_option.csv", index=False)

    saved = []
    if save_paths:
        for strategy_id in selected["strategy_id"]:
            item = cache.get(strategy_id)
            if item is None:
                continue
            path = item[1]
            family_dir = PATH_DIR / path["family"].iloc[0]
            family_dir.mkdir(parents=True, exist_ok=True)
            safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in strategy_id)
            out_path = family_dir / f"{safe_id}.parquet"
            path.to_parquet(out_path, index=False)
            saved.append({"strategy_id": strategy_id, "path": str(out_path)})
    pd.DataFrame(saved).to_csv(OUTPUT_DIR / "06_selected_core_strategy_paths.csv", index=False)

    manifest = {
        "data_path": str(DATA_PATH),
        "work_root": str(WORK_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "output_dir": str(OUTPUT_DIR),
        "has_numba": HAS_NUMBA,
        "n_markets": int(panel.shape[1]),
        "n_strategies_total": int(metrics.shape[0]),
        "n_strategies_selected": int(selected.shape[0]),
        "n_paths_saved": int(len(saved)),
        "families": sorted(metrics["family"].unique().tolist()),
        "note": "Notebook is lightweight; programs are on C, outputs are on D.",
    }
    with (OUTPUT_DIR / "07_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    return metrics, selected, manifest


def describe_strategy_families():
    return pd.DataFrame([
        {"family": "time_series_momentum", "meaning_ru": "моментум временного ряда", "role_ru": "трендовые кандидаты"},
        {"family": "time_series_reversal", "meaning_ru": "реверс временного ряда", "role_ru": "контртрендовые кандидаты"},
        {"family": "moving_average_crossover", "meaning_ru": "пересечение скользящих средних", "role_ru": "классический тренд"},
        {"family": "channel_breakout", "meaning_ru": "пробой канала", "role_ru": "сильные движения"},
        {"family": "channel_reversal", "meaning_ru": "реверс от границ канала", "role_ru": "возврат в диапазон"},
        {"family": "pair_spread_mean_reversion", "meaning_ru": "возврат парного спреда к среднему", "role_ru": "relative value"},
        {"family": "pair_spread_momentum", "meaning_ru": "моментум парного спреда", "role_ru": "альтернатива возврату"},
        {"family": "sector_trend_portfolio", "meaning_ru": "секторный трендовый портфель", "role_ru": "портфельный тренд"},
        {"family": "sector_reversal_portfolio", "meaning_ru": "секторный реверсивный портфель", "role_ru": "портфельный реверс"},
        {"family": "sector_cross_sectional_momentum", "meaning_ru": "секторный поперечный моментум", "role_ru": "relative strength"},
        {"family": "sector_cross_sectional_reversal", "meaning_ru": "секторный поперечный реверс", "role_ru": "relative weakness"},
    ])


def build_strategy_specs(panel=None):
    if panel is None:
        panel = load_or_build_panel("30min")
    specs = []
    for market in panel.columns:
        if panel[market].dropna().shape[0] < 1000:
            continue
        for lb in LOOKBACKS:
            specs.append({"family": "time_series_momentum", "instrument": market, "market": market, "lookback": lb})
            specs.append({"family": "time_series_reversal", "instrument": market, "market": market, "lookback": lb})
        for fast, slow in FAST_SLOW:
            specs.append({"family": "moving_average_crossover", "instrument": market, "market": market, "fast": fast, "slow": slow})
        for window in ZSCORE_WINDOWS:
            specs.append({"family": "channel_breakout", "instrument": market, "market": market, "window": window})
            specs.append({"family": "channel_reversal", "instrument": market, "market": market, "window": window})

    for y_m, x_m in candidate_pairs(panel):
        px = panel[[y_m, x_m]].dropna()
        if len(px) < 1500:
            continue
        for window in ZSCORE_WINDOWS:
            for threshold in ZSCORE_THRESHOLDS:
                specs.append({"family": "pair_spread_mean_reversion", "instrument": f"{y_m}/{x_m}", "y": y_m, "x": x_m, "window": window, "threshold": threshold})
                specs.append({"family": "pair_spread_momentum", "instrument": f"{y_m}/{x_m}", "y": y_m, "x": x_m, "window": window, "threshold": threshold})

    for sector, markets in SECTOR_MAP.items():
        cols = [m for m in markets if m in panel.columns]
        if len(cols) < 2:
            continue
        for lb in LOOKBACKS:
            specs.append({"family": "sector_trend_portfolio", "instrument": sector, "sector": sector, "lookback": lb})
            specs.append({"family": "sector_reversal_portfolio", "instrument": sector, "sector": sector, "lookback": lb})
            specs.append({"family": "sector_cross_sectional_momentum", "instrument": sector, "sector": sector, "lookback": lb})
            specs.append({"family": "sector_cross_sectional_reversal", "instrument": sector, "sector": sector, "lookback": lb})

    for i, spec in enumerate(specs):
        spec["spec_id"] = i
        spec["strategy_id"] = strategy_id_from_spec(spec)
    return specs


def strategy_id_from_spec(spec):
    f = spec["family"]
    if f in {"time_series_momentum", "time_series_reversal"}:
        return f"{spec['market']}__{f}__lb_{spec['lookback']}"
    if f == "moving_average_crossover":
        return f"{spec['market']}__ma_cross__fast_{spec['fast']}__slow_{spec['slow']}"
    if f in {"channel_breakout", "channel_reversal"}:
        return f"{spec['market']}__{f}__window_{spec['window']}"
    if f in {"pair_spread_mean_reversion", "pair_spread_momentum"}:
        return f"{spec['y']}_{spec['x']}__{f}__window_{spec['window']}__z_{spec['threshold']}"
    if f.startswith("sector_"):
        return f"{spec['sector']}__{f}__lb_{spec['lookback']}"
    raise ValueError(f"Unknown family: {f}")


def path_from_spec(spec, panel):
    f = spec["family"]
    if f in {"time_series_momentum", "time_series_reversal"}:
        px = panel[spec["market"]].dropna()
        mom = np.log(px).diff(int(spec["lookback"]))
        sig = np.sign(mom)
        if f == "time_series_reversal":
            sig = -sig
        return make_strategy_path(spec["strategy_id"], f, spec["instrument"], px, sig)
    if f == "moving_average_crossover":
        px = panel[spec["market"]].dropna()
        sig = np.sign(px.rolling(int(spec["fast"]), min_periods=int(spec["fast"])).mean() - px.rolling(int(spec["slow"]), min_periods=int(spec["slow"])).mean())
        return make_strategy_path(spec["strategy_id"], f, spec["instrument"], px, sig)
    if f in {"channel_breakout", "channel_reversal"}:
        px = panel[spec["market"]].dropna()
        window = int(spec["window"])
        high = px.rolling(window, min_periods=window).max().shift(1)
        low = px.rolling(window, min_periods=window).min().shift(1)
        sig = pd.Series(0.0, index=px.index)
        sig[px > high] = 1.0
        sig[px < low] = -1.0
        if f == "channel_reversal":
            sig = -sig
        return make_strategy_path(spec["strategy_id"], f, spec["instrument"], px, sig)
    if f in {"pair_spread_mean_reversion", "pair_spread_momentum"}:
        px = panel[[spec["y"], spec["x"]]].dropna()
        log_y = np.log(px[spec["y"]])
        log_x = np.log(px[spec["x"]])
        beta = beta_ols(log_y, log_x)
        spread = (log_y - beta * log_x).rename(f"{spec['y']}_{spec['x']}_spread")
        z = safe_zscore(spread, int(spec["window"]))
        threshold = float(spec["threshold"])
        sig = pd.Series(0.0, index=spread.index)
        sig[z > threshold] = -1.0
        sig[z < -threshold] = 1.0
        if f == "pair_spread_momentum":
            sig = -sig
        return make_strategy_path(spec["strategy_id"], f, spec["instrument"], spread, sig)
    if f.startswith("sector_"):
        cols = [m for m in SECTOR_MAP[spec["sector"]] if m in panel.columns]
        sub = panel[cols].dropna(how="all")
        returns = np.log(sub).diff()
        price_like = returns.mean(axis=1).fillna(0.0).cumsum()
        past = np.log(sub).diff(int(spec["lookback"]))
        if f in {"sector_trend_portfolio", "sector_reversal_portfolio"}:
            sig = np.sign(past).mean(axis=1)
            if f == "sector_reversal_portfolio":
                sig = -sig
        else:
            rank = past.rank(axis=1, pct=True)
            sig = ((rank > 0.67).astype(float) - (rank < 0.33).astype(float)).mean(axis=1)
            if f == "sector_cross_sectional_reversal":
                sig = -sig
        return make_strategy_path(spec["strategy_id"], f, spec["instrument"], price_like, sig)
    raise ValueError(f"Unknown family: {f}")


def save_strategy_catalog(panel=None):
    ensure_dirs()
    if panel is None:
        panel = load_or_build_panel("30min")
    specs = build_strategy_specs(panel)
    catalog = pd.DataFrame(specs)
    catalog.to_csv(OUTPUT_DIR / "02_strategy_spec_catalog.csv", index=False)
    return catalog


def run_batch(batch_id=0, batch_size=100, save_paths=True, start_id=None, end_id=None, batch_label=None):
    ensure_dirs()
    batch_id = int(batch_id)
    batch_size = int(batch_size)
    panel = load_or_build_panel("30min")
    catalog = save_strategy_catalog(panel)
    if start_id is None:
        start = batch_id * batch_size
    else:
        start = int(start_id)
    if end_id is None:
        end = min(start + batch_size, len(catalog))
    else:
        end = min(int(end_id), len(catalog))
    label = batch_label if batch_label is not None else f"{batch_id:04d}"
    batch = catalog.iloc[start:end].copy()
    if batch.empty:
        print(f"Batch (пакет) {batch_id}: empty range {start}:{end}")
        return pd.DataFrame(), {"batch_id": batch_id, "status": "empty"}

    rows = []
    saved = []
    batch_path_dir = PATH_DIR / f"batch_{label}"
    if save_paths:
        batch_path_dir.mkdir(parents=True, exist_ok=True)

    for _, spec_row in batch.iterrows():
        spec = spec_row.dropna().to_dict()
        path = path_from_spec(spec, panel)
        row = {
            "batch_id": batch_id,
            "spec_id": int(spec["spec_id"]),
            "strategy_id": spec["strategy_id"],
            "family": spec["family"],
            "instrument": spec["instrument"],
        }
        row.update(metrics_for_path(path))
        row["train_sharpe_like"] = row.get("train_step_mean", np.nan) / row.get("train_step_std", np.nan) * np.sqrt(44 * 252) if row.get("train_step_std", 0) else np.nan
        row["cost_dominated_train"] = row.get("train_cost_gap_to_abs_gross", np.nan) > 0.6
        row["abs_train_net_total"] = abs(row.get("train_net_total", np.nan))
        row["geometry_label"] = geometry_label(row)
        rows.append(row)
        if save_paths:
            safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in spec["strategy_id"])
            out_path = batch_path_dir / f"{int(spec['spec_id']):05d}__{safe_id}.parquet"
            path.to_parquet(out_path, index=False)
            saved.append({"batch_id": batch_id, "spec_id": int(spec["spec_id"]), "strategy_id": spec["strategy_id"], "path": str(out_path)})
        del path

    metrics = pd.DataFrame(rows)
    batch_dir = OUTPUT_DIR / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(batch_dir / f"metrics_batch_{label}.csv", index=False)
    pd.DataFrame(saved).to_csv(batch_dir / f"paths_batch_{label}.csv", index=False)
    manifest = {
        "batch_id": batch_id,
        "batch_label": label,
        "batch_size": batch_size,
        "start_spec_id": int(start),
        "end_spec_id_exclusive": int(end),
        "n_catalog_total": int(len(catalog)),
        "n_processed": int(len(metrics)),
        "n_paths_saved": int(len(saved)),
        "output_dir": str(OUTPUT_DIR),
        "has_numba": HAS_NUMBA,
    }
    with (batch_dir / f"manifest_batch_{label}.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return metrics, manifest


def combine_batches():
    ensure_dirs()
    batch_dir = OUTPUT_DIR / "batches"
    metric_files = sorted(batch_dir.glob("metrics_batch_*.csv"))
    path_files = sorted(batch_dir.glob("paths_batch_*.csv"))
    metrics = pd.concat([pd.read_csv(p) for p in metric_files], ignore_index=True) if metric_files else pd.DataFrame()
    paths = pd.concat([pd.read_csv(p) for p in path_files], ignore_index=True) if path_files else pd.DataFrame()
    if not metrics.empty:
        metrics = metrics.drop_duplicates(subset=["spec_id"], keep="last")
        metrics = metrics.sort_values(["abs_train_net_total", "train_trend_r2"], ascending=[False, False])
    if not paths.empty:
        paths = paths.drop_duplicates(subset=["spec_id"], keep="last")
    metrics.to_csv(OUTPUT_DIR / "03_core_strategy_train_metrics.csv", index=False)
    if not metrics.empty:
        metrics[["strategy_id", "family", "instrument", "geometry_label"]].to_csv(OUTPUT_DIR / "04_core_strategy_catalog.csv", index=False)
    paths.to_csv(OUTPUT_DIR / "06_all_core_strategy_paths.csv", index=False)
    manifest = {
        "n_batches": len(metric_files),
        "n_strategies_total_processed": int(len(metrics)),
        "n_paths_saved": int(len(paths)),
        "output_dir": str(OUTPUT_DIR),
    }
    with (OUTPUT_DIR / "07_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return metrics, paths, manifest


def audit_batches():
    ensure_dirs()
    catalog_path = OUTPUT_DIR / "02_strategy_spec_catalog.csv"
    if catalog_path.exists():
        catalog = pd.read_csv(catalog_path)
    else:
        catalog = save_strategy_catalog(load_or_build_panel("30min"))
    batch_dir = OUTPUT_DIR / "batches"
    metric_files = sorted(batch_dir.glob("metrics_batch_*.csv"))
    raw = pd.concat([pd.read_csv(p) for p in metric_files], ignore_index=True) if metric_files else pd.DataFrame()
    if raw.empty:
        report = {"catalog_rows": int(len(catalog)), "processed_rows": 0, "missing_count": int(len(catalog)), "duplicate_count": 0}
    else:
        missing = sorted(set(catalog["spec_id"]) - set(raw["spec_id"]))
        duplicate_ids = sorted(raw.loc[raw["spec_id"].duplicated(), "spec_id"].astype(int).unique().tolist())
        report = {
            "catalog_rows": int(len(catalog)),
            "processed_rows": int(len(raw)),
            "unique_processed_spec_ids": int(raw["spec_id"].nunique()),
            "missing_count": int(len(missing)),
            "duplicate_count": int(len(duplicate_ids)),
            "missing_ranges": contiguous_ranges(missing),
            "duplicate_head": duplicate_ids[:30],
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def contiguous_ranges(values):
    if not values:
        return []
    ranges = []
    start = prev = int(values[0])
    for value in values[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
        else:
            ranges.append([start, prev + 1])
            start = prev = value
    ranges.append([start, prev + 1])
    return ranges


def print_catalog_summary():
    panel = load_or_build_panel("30min")
    catalog = save_strategy_catalog(panel)
    summary = catalog.groupby("family").size().sort_values(ascending=False).reset_index(name="n_strategies")
    print("Total strategies (всего стратегий):", len(catalog))
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["summary", "batch", "combine", "audit"], default="summary")
    parser.add_argument("--batch-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--start-id", type=int, default=None)
    parser.add_argument("--end-id", type=int, default=None)
    parser.add_argument("--batch-label", type=str, default=None)
    parser.add_argument("--no-save-paths", action="store_true")
    args = parser.parse_args()
    if args.mode == "summary":
        print_catalog_summary()
    elif args.mode == "batch":
        run_batch(args.batch_id, args.batch_size, save_paths=not args.no_save_paths, start_id=args.start_id, end_id=args.end_id, batch_label=args.batch_label)
    elif args.mode == "combine":
        combine_batches()
    elif args.mode == "audit":
        audit_batches()

