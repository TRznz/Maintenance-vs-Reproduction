"""
run_rounds.py — production rounds for the poster (plan: run_plan.md).

Usage:
    python run_rounds.py A                  # results next to this file
    python run_rounds.py A ~/mesa_results   # results under another folder

Each round writes results/<round>/summary.csv (+ per-run time series and
Maintainer death records).  Key outputs per run:
    first_extinct   "Reproducer" / "Maintainer" / "both" / empty (both alive)
    share_51_400    mean Maintainer biomass share, steps 51–400 (padded)
Seeds start at 101 so production runs never reuse the coarse-scan seeds.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import batch_run as B

SEEDS20 = list(range(101, 121))

# common settings: static environment, age reset on, defaults otherwise
BASE = {"offspring_age_reset": True}


def round_A():
    """Fig 1 · damage: hazard per cell × age-damage slope b."""
    pts = B.factorial(
        ("hazard_level", [0.0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0006, 0.001]),  # transition is at 0–0.0004 (coarse check)
        ("age_damage_slope", [0.0, 1e-4, 3e-4, 5e-4, 1e-3]),
        [BASE],
    )
    return dict(grid=pts, seeds=SEEDS20, T=800, grace=0, max_agents=30000)


def round_B():
    """Fig 2 · resource: flux (0.2 × K) × exclusivity, at the hazard boundary."""
    fluxes = [0.5, 1, 2, 5, 10, 25]
    pts = B.factorial(
        [{"resource_capacity": f / 0.2} for f in fluxes],
        ("exclusivity", [0.0, 0.5, 0.75, 1.0, 3.0]),   # persistence flips between 0.5 and 1 (coarse check)
        [dict(BASE, hazard_level=0.0004, age_damage_slope=1e-4)],   # on the Fig 1 boundary (45%)
    )
    return dict(grid=pts, seeds=SEEDS20, T=1500, grace=0, max_agents=30000)


def round_D():
    """
    Fig 3 · environmental fluctuation and its synchrony.
    Baseline = the Fig 2 default cell (hazard 4e-4, b 1e-4, resource 2,
    exclusivity 0.5; 45% Reproducers-first when static).
      single field : resource OR hazard fluctuates (the other stays constant)
                     × period {100, 400} × amplitude {0.5, 0.9} × patch size
      joint        : both fluctuate, period 100, amplitude 0.9, linked
                     independent / positive / negative × patch size
    """
    base = dict(BASE, hazard_level=0.0004, age_damage_slope=1e-4)
    Ls = [1, 5, 10, 25, 50]
    pts = [dict(base, _cond="static")]
    for f in ("resource", "hazard"):
        for P in (100, 400):
            for A in (0.5, 0.9):
                for L in Ls:
                    pts.append(dict(base, _cond=f"{f}", **{
                        f"{f}_temporal_amp": A, f"{f}_period": P, f"{f}_patch_size": L}))
    for link in ("independent", "positive", "negative"):
        for L in Ls:
            pts.append(dict(base, _cond=f"both-{link}", hazard_link=link,
                            resource_temporal_amp=0.9, resource_period=100, resource_patch_size=L,
                            hazard_temporal_amp=0.9, hazard_period=100, hazard_patch_size=L))
    return dict(grid=pts, seeds=SEEDS20, T=1500, grace=0, max_agents=30000)


# ---------------------------------------------------------------------------
# Final rounds for Fig 3 (environmental dynamics), planned 2026-11-18
# ---------------------------------------------------------------------------
SEEDS60 = list(range(101, 161))     # 101–120 reproduce round D exactly
SEEDS40 = list(range(101, 141))
DYN_BASE = dict(BASE, hazard_level=0.0004, age_damage_slope=1e-4)   # Fig 2 default cell
PATCHES = [1, 2, 5, 10, 25, 50]


def _temporal(base, field, P, L):
    return dict(base, **{f"{field}_temporal_amp": 0.9, f"{field}_period": P, f"{field}_patch_size": L})


def _joint(base, link, L, P=100):
    return dict(base, hazard_link=link,
                resource_temporal_amp=0.9, resource_period=P, resource_patch_size=L,
                hazard_temporal_amp=0.9, hazard_period=P, hazard_patch_size=L)


def round_D2():
    """Fig 3 main: field × patch size (× period), amplitude 0.9; 50 extra steps after first extinction."""
    pts = [dict(DYN_BASE, _cond="static")]
    for f in ("resource", "hazard"):
        for P in (100, 400):
            for L in PATCHES:
                pts.append(dict(_temporal(DYN_BASE, f, P, L), _cond=f))
    for link in ("independent", "positive", "negative"):
        for L in PATCHES:
            pts.append(dict(_joint(DYN_BASE, link, L), _cond=f"both-{link}"))
    return dict(grid=pts, seeds=SEEDS60, T=1500, grace=50, max_agents=30000)


def round_G():
    """Fig 3 inset: whole-map resource seasons, period scan."""
    pts = [dict(DYN_BASE, _cond="static")]
    for P in (50, 100, 150, 200, 300, 400):
        pts.append(dict(_temporal(DYN_BASE, "resource", P, 50), _cond="resource-sync"))
    return dict(grid=pts, seeds=SEEDS60, T=1500, grace=0, max_agents=30000)


def round_I():
    """Fixed spatial patches (no change over time), spatial amplitude 0.9."""
    pts = [dict(DYN_BASE, _cond="static")]
    for f in ("resource", "hazard"):
        for L in (1, 5, 10, 25, 50):
            pts.append(dict(DYN_BASE, _cond=f"{f}-spatial",
                            **{f"{f}_spatial_amp": 0.9, f"{f}_patch_size": L}))
    return dict(grid=pts, seeds=SEEDS60, T=1500, grace=0, max_agents=30000)


def round_J():
    """Robustness: key cells at a second boundary point (hazard 3e-4, b 3e-4; static ≈ 65%)."""
    alt = dict(BASE, hazard_level=0.0003, age_damage_slope=3e-4)
    pts = [dict(alt, _cond="static"),
           dict(_temporal(alt, "hazard", 100, 50), _cond="hazard-sync-P100"),
           dict(_temporal(alt, "resource", 100, 5), _cond="resource-L5-P100"),
           dict(_temporal(alt, "resource", 100, 50), _cond="resource-sync-P100"),
           dict(_temporal(alt, "resource", 400, 50), _cond="resource-sync-P400"),
           dict(_joint(alt, "positive", 25), _cond="both-positive-L25"),
           dict(_joint(alt, "negative", 25), _cond="both-negative-L25")]
    return dict(grid=pts, seeds=SEEDS40, T=1500, grace=0, max_agents=30000)


SEEDS_EXTRA = list(range(141, 221))   # adds 80 seeds to the 40 of round J → 120 per cell
ALT2 = dict(BASE, hazard_level=0.0003, age_damage_slope=3e-4)
ALT3 = dict(BASE, hazard_level=0.0002, age_damage_slope=5e-4)   # age-heavy boundary point (round A: 70%)


def _key_cells(base):
    return [dict(base, _cond="static"),
            dict(_temporal(base, "hazard", 100, 50), _cond="hazard-sync-P100"),
            dict(_temporal(base, "resource", 100, 5), _cond="resource-L5-P100"),
            dict(_temporal(base, "resource", 100, 50), _cond="resource-sync-P100"),
            dict(_temporal(base, "resource", 400, 50), _cond="resource-sync-P400"),
            dict(_joint(base, "positive", 25), _cond="both-positive-L25"),
            dict(_joint(base, "negative", 25), _cond="both-negative-L25")]


def round_K1():
    """More seeds for the disputed pair at the second baseline (static vs short whole-map resource seasons)."""
    pts = [dict(ALT2, _cond="static"), dict(_temporal(ALT2, "resource", 100, 50), _cond="resource-sync-P100")]
    return dict(grid=pts, seeds=SEEDS_EXTRA, T=1500, grace=0, max_agents=30000)


def round_K2():
    """Third baseline (hazard 2e-4, b 5e-4): the 7 key cells, 40 seeds."""
    return dict(grid=_key_cells(ALT3), seeds=SEEDS40, T=1500, grace=0, max_agents=30000)


def round_K3():
    """Third baseline: more seeds for static vs short whole-map resource seasons."""
    pts = [dict(ALT3, _cond="static"), dict(_temporal(ALT3, "resource", 100, 50), _cond="resource-sync-P100")]
    return dict(grid=pts, seeds=SEEDS_EXTRA, T=1500, grace=0, max_agents=30000)


ROUNDS = {"A": round_A, "B": round_B, "D": round_D,
          "D2": round_D2, "G": round_G, "I": round_I, "J": round_J,
          "K1": round_K1, "K2": round_K2, "K3": round_K3,
          "C": round_D2}   # poster name: Round C = D2

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ROUNDS:
        sys.exit(f"usage: python run_rounds.py <{'|'.join(ROUNDS)}> [output folder]")
    name = sys.argv[1]
    base = os.path.expanduser(sys.argv[2]) if len(sys.argv) > 2 else HERE
    cfg = ROUNDS[name]()
    B.run_grid(cfg["grid"], cfg["seeds"], cfg["T"],
               out_dir=os.path.join(base, "results", f"round_{name}"),
               grace=cfg["grace"], max_agents=cfg["max_agents"])
