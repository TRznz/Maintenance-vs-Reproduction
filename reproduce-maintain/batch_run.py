"""
batch_run.py
============
Headless batch runner for the Reproduce-Maintain model.

  • runs a parameter grid × several random seeds in parallel (multiprocessing)
  • stopping rule: once EITHER type goes extinct, the run continues for
    another `grace` steps (default 500) and then stops; it also stops at T
    (default 1500) or when both types are extinct.  A type still alive at
    the stop is right-censored (recorded as "alive").
  • population cap (max_agents, default None = no cap): if the total number
    of agents exceeds it, the run stops and `cap_hit_step` records when.
    Only Reproducers can multiply, so hitting the cap means Reproducers have
    survived the opening and are exploding; Maintainer dominance up to that
    point is still measured.
  • parameter points can be a full factorial (dict) or an explicit list of
    dicts, combined with factorial() — see the bottom of this file
  • writes to the output folder, saving each run as soon as it finishes
    (so a crash or a sync problem can never lose finished runs):
        summary.csv                       one row per run (parameters + metrics)
        timeseries/run_0000.csv.gz ...    every collected variable, every step
        maintainer_deaths/run_0000.csv .. one row per Maintainer death
    load_timeseries(out_dir) / load_deaths(out_dir) read them back as one table.
  • Tip: write results to a folder that is NOT synced by OneDrive / iCloud /
    Dropbox; sync clients can lock files mid-write and cause timeouts.

Usage:
    python batch_run.py                      # results next to this file
    python batch_run.py ~/mesa_results       # results under another folder
Requirements: mesa, numpy, pandas; environment.py, agents.py, model.py and
this file in the same folder.  Results go to results/<round>/ next to it.

Metric definitions
------------------
share            Maintainer biomass / total biomass, each step
dom_steps        number of steps with share > 0.5 (counted from step 1)
dom_steps_skip50 same, ignoring steps 1–50 (checks the opening transient)
dom_longest      longest uninterrupted run of steps with share > 0.5
share_mean       mean share over the run (until Maintainer extinction or T)
peak_share / peak_step
ext_R / ext_M    first extinction step of each type (None = still alive)
censored         True if Maintainers were still alive at the stop
first_extinct    "Reproducer", "Maintainer", "both" (same step) or None
outlasting       type that outlived the other: "Reproducer", "Maintainer",
                 "tie" (same step) or "coexist" (both alive at the stop)
R_censored       True if Reproducers were still alive at the stop
Death classes (per type, health deaths split by dominant damage source):
  energy                    — starved (energy < 0)
  health_age                — health ≤ 0, lifetime age damage > hazard damage
  health_hazard             — health ≤ 0, lifetime hazard damage ≥ age damage
  age_exceeds_repair        — health deaths where age damage alone was
                              already above the repair cap (true "old age")
"""

import itertools
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _dominance(share, thr=0.5, skip=0):
    s = np.nan_to_num(np.asarray(share[1:], dtype=float), nan=0.0)  # drop step 0
    s = s[skip:]
    above = s > thr
    longest = cur = 0
    for v in above:
        cur = cur + 1 if v else 0
        longest = max(longest, cur)
    return int(above.sum()), int(longest)


def _death_classes(log, typ):
    rows = [d for d in log if d["type"] == typ]
    out = {"energy": 0, "health_age": 0, "health_hazard": 0, "age_exceeds_repair": 0}
    for d in rows:
        if d["cause"] == "energy":
            out["energy"] += 1
        else:
            if d["cum_age_damage"] > d["cum_hazard_damage"]:
                out["health_age"] += 1
            else:
                out["health_hazard"] += 1
            if d["age_exceeds_repair"]:
                out["age_exceeds_repair"] += 1
    return {f"{typ[0]}_death_{k}": v for k, v in out.items()}


def run_one(job):
    """job = (run_id, params_dict, seed, T)"""
    warnings.filterwarnings("ignore")
    from model import ReproduceMaintainModel, ReproduceMaintainScenario

    run_id, params, seed, T, grace, max_agents = job
    t0 = time.time()
    # keys starting with "_" are labels for the summary, not model parameters
    model_params = {k: v for k, v in params.items() if not k.startswith("_")}
    m = ReproduceMaintainModel(ReproduceMaintainScenario(rng=seed, **model_params))
    cap_hit = None
    while m.running and m.steps < T:
        firsts = [x for x in (m.reproducer_extinction_step,
                              m.maintainer_extinction_step) if x is not None]
        if firsts and m.steps >= min(firsts) + grace:
            break
        if max_agents is not None and len(m.agents) > max_agents:
            cap_hit = int(m.steps)
            break
        m.step()

    eR, eM = m.reproducer_extinction_step, m.maintainer_extinction_step
    if eR is None and eM is None:
        first, outlast = None, "coexist"
    elif eR is not None and eM is not None and eR == eM:
        first, outlast = "both", "tie"
    elif eR is None and eM is None:
        first, outlast = None, "coexist"
    elif eM is None or (eR is not None and eR < eM):
        first, outlast = "Reproducer", "Maintainer"
    else:
        first, outlast = "Maintainer", "Reproducer"

    ts = m.datacollector.get_model_vars_dataframe()
    ts.insert(0, "step", np.arange(len(ts)))
    ts.insert(0, "run_id", run_id)

    share = ts["MaintainerBiomassShare"].to_numpy()
    dom, longest = _dominance(share)
    dom50, _ = _dominance(share, skip=50)
    sh = np.nan_to_num(share[1:], nan=0.0)
    peak = int(np.argmax(sh)) + 1 if len(sh) else 0

    m_deaths = [d for d in m.death_log if d["type"] == "Maintainer"]
    m_life = [d["age"] for d in m_deaths]

    # fixed-window biomass metric: mean Maintainer share over steps 51–400,
    # padded with 1 (Reproducers extinct) or 0 (Maintainers extinct) after an
    # extinction, or with the last value if the run stopped early otherwise
    win = pd.Series(share, index=np.arange(len(share))).reindex(range(51, 401))
    if eR is not None and (eM is None or eR < eM):
        pad = 1.0
    elif eM is not None and (eR is None or eM < eR):
        pad = 0.0
    else:
        full = pd.Series(share).dropna()          # e.g. stopped at the population cap
        pad = full.iloc[-1] if len(full) else np.nan
    share_51_400 = float(win.fillna(pad).mean())

    # mean Maintainer share from step 31 until the first extinction (or the
    # end of the run if neither type went extinct)
    firsts = [x for x in (eR, eM) if x is not None]
    end = min(firsts) if firsts else len(share) - 1
    seg = np.asarray(share[31:end + 1], dtype=float)
    if end < 31:          # one type was gone before step 31: count the winner fully
        share_31_first = pad if np.isfinite(pad) else np.nan
    else:
        share_31_first = float(np.nanmean(seg)) if np.isfinite(seg).any() else np.nan

    summary = {
        "run_id": run_id, "seed": seed, **params,
        "share_51_400": share_51_400,
        "share_31_first": share_31_first,
        "steps_run": int(m.steps),
        "dom_steps": dom, "dom_steps_skip50": dom50, "dom_longest": longest,
        "share_mean": float(sh.mean()) if len(sh) else float("nan"),
        "peak_share": float(sh.max()) if len(sh) else float("nan"),
        "peak_step": peak,
        "ext_R": m.reproducer_extinction_step,
        "ext_M": m.maintainer_extinction_step,
        "censored": m.maintainer_extinction_step is None,
        "R_censored": m.reproducer_extinction_step is None,
        "first_extinct": first,
        "cap_hit_step": cap_hit,
        "outlasting": outlast,
        "M_lifespan_median": float(np.median(m_life)) if m_life else float("nan"),
        "M_max_size": float(np.nanmax(ts["MaxMaintainerSize"].to_numpy()))
                      if ts["MaxMaintainerSize"].notna().any() else float("nan"),
        "R_count_end": int(ts["Reproducers"].iloc[-1]),
        "R_mean_age_mean": float(np.nanmean(ts["MeanReproducerAge"].to_numpy())),
        **_death_classes(m.death_log, "Maintainer"),
        **_death_classes(m.death_log, "Reproducer"),
        "seconds": round(time.time() - t0, 1),
    }
    deaths = pd.DataFrame(m_deaths)
    if len(deaths):
        deaths.insert(0, "run_id", run_id)
    return summary, ts, deaths


def factorial(*factors):
    """
    Full factorial of several factors.  Each factor is either
      • ("param_name", [v1, v2, ...])   — one parameter, several values, or
      • [ {..}, {..}, ... ]             — a list of parameter dicts
        (use this when one "level" sets several parameters at once,
         e.g. a level that sets several parameters at once)
    Returns a list of parameter dicts.
    """
    levels = []
    for f in factors:
        if isinstance(f, tuple):
            name, vals = f
            levels.append([{name: v} for v in vals])
        else:
            levels.append(list(f))
    out = []
    for combo in itertools.product(*levels):
        d = {}
        for part in combo:
            d.update(part)
        out.append(d)
    return out


def run_grid(grid, seeds, T, out_dir, processes=None, grace=500, max_agents=None):
    """
    grid: dict name → list of values (full factorial), or a list of
          parameter dicts (e.g. from factorial()).
    Returns the summary DataFrame.
    """
    os.makedirs(out_dir, exist_ok=True)
    if isinstance(grid, dict):
        names = list(grid)
        combos = [dict(zip(names, vals)) for vals in itertools.product(*grid.values())]
    else:
        combos = list(grid)
    # make every point list the same parameters (missing ones = model defaults)
    from model import ReproduceMaintainScenario as _D
    keys = sorted({k for c in combos for k in c})
    combos = [{k: c.get(k, getattr(_D, k, None)) for k in keys} for c in combos]
    jobs = [(i, p, s, T, grace, max_agents)
            for i, (p, s) in enumerate(itertools.product(combos, seeds))]
    print(f"{len(combos)} parameter points × {len(seeds)} seeds = {len(jobs)} runs", flush=True)

    ts_dir = os.path.join(out_dir, "timeseries")
    d_dir = os.path.join(out_dir, "maintainer_deaths")
    os.makedirs(ts_dir, exist_ok=True)
    os.makedirs(d_dir, exist_ok=True)

    summaries = []
    t0 = time.time()
    with Pool(processes) as pool:
        for k, (smry, ts, d) in enumerate(pool.imap_unordered(run_one, jobs), 1):
            summaries.append(smry)
            rid = smry["run_id"]
            ts.to_csv(os.path.join(ts_dir, f"run_{rid:04d}.csv.gz"), index=False)
            if len(d):
                d.to_csv(os.path.join(d_dir, f"run_{rid:04d}.csv"), index=False)
            print(f"[{k}/{len(jobs)}] run {smry['run_id']} {smry['seconds']}s "
                  f"dom={smry['dom_steps']} extR={smry['ext_R']} extM={smry['ext_M']} "
                  f"(elapsed {time.time()-t0:.0f}s)", flush=True)
            if k % 10 == 0 or k == len(jobs):           # checkpoint
                pd.DataFrame(summaries).sort_values("run_id").to_csv(
                    os.path.join(out_dir, "summary.csv"), index=False)

    summary = pd.DataFrame(summaries).sort_values("run_id")
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    return summary


def _load_dir(folder):
    import glob
    files = sorted(glob.glob(os.path.join(folder, "run_*.csv*")))
    return pd.concat([pd.read_csv(f) for f in files]) if files else pd.DataFrame()


def load_timeseries(out_dir):
    """All per-run time series in out_dir as one DataFrame."""
    return _load_dir(os.path.join(out_dir, "timeseries"))


def load_deaths(out_dir):
    """All per-run Maintainer death records in out_dir as one DataFrame."""
    return _load_dir(os.path.join(out_dir, "maintainer_deaths"))


if __name__ == "__main__":
    # Figure 1 rerun under the uniform-hazard rule:
    # damage axes — age-damage slope b × hazard per cell; resource at default,
    # environment static (all amplitudes 0), age reset on.
    # 5 hazard levels × 4 age slopes = 20 points × 10 seeds = 200 runs.
    # grace = 50: the win metric is fixed at the first extinction.
    HERE = os.path.dirname(os.path.abspath(__file__))
    BASE = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else HERE
    points = factorial(
        ("hazard_level", [0.0, 0.0002, 0.0004, 0.0006, 0.001]),   # 50% point ≈ 0.0006 under uniform hazard
        ("age_damage_slope", [0.0, 1e-4, 3e-4, 1e-3]),
        ("offspring_age_reset", [True]),
    )
    run_grid(points, seeds=list(range(1, 11)), T=1500, grace=50,
             max_agents=30000,
             out_dir=os.path.join(BASE, "results", "fig1"))
