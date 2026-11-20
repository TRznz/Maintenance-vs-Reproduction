"""
model.py
========
ReproduceMaintainScenario  — all parameters in one place
allocate_resources()       — two-phase synchronous allocation (vectorised):
                             claims → proportional payout → zero the cell;
                             also records each agent's hazard exposure
ReproduceMaintainModel     — main Model class

KEY ARCHITECTURE NOTE — two-phase synchronous allocation:
  If each agent fed inside its own step(), execution order would decide who
  eats first (first-come-first-served).  Instead all agents first submit
  claims, then every cell is shared out at once, so the weighted share
  formula always has a complete denominator.
"""

import numpy as np
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import ContinuousSpace
from mesa.experimental.scenarios import Scenario

from environment import EnvironmentGrid
from agents import Reproducer, Maintainer


# ======================================================================
# Scenario
# ======================================================================

class ReproduceMaintainScenario(Scenario):
    # ── World (fixed) ──────────────────────────────────────────────────
    width:  float = 500.0
    height: float = 500.0
    torus:  bool  = True
    grid_resolution: float = 10.0

    # ── Initial population (fixed) ─────────────────────────────────────
    initial_reproducer_count: int   = 100
    initial_maintainer_count: int   = 100
    initial_energy:           float = 5.0
    initial_size:             float = 1.0

    # ── Movement (fixed in UI; kept here for later batch sweeps) ───────
    move_step_size: float = 5.0

    # ── Size-derived functions ─────────────────────────────────────────
    # capture_radius   = r0 + k_r * sqrt(size)   (foraging + hazard exposure)
    # energy_capacity  = E0_cap + k_cap * size
    # maintenance_cost = c0 + c1 * size          ← SLIDERS (const vs linear)
    r0:     float = 2.0
    k_r:    float = 3.0
    E0_cap: float = 20.0
    k_cap:  float = 10.0
    c0:     float = 0.2
    c1:     float = 0.1

    # ── Growth / split ─────────────────────────────────────────────────
    energy_buffer:              float = 2.0
    growth_invest_rate:         float = 0.3   # fixed
    growth_efficiency:          float = 0.5   # fixed
    # Growth gate (MAINTAINER ONLY; Reproducer grows unconditionally):
    # grow only if health ≥ growth_min_health
    # AND energy ≥ growth_min_energy_frac * energy_capacity   ← SLIDERS
    growth_min_health:      float = 0.99
    growth_min_energy_frac: float = 0.1
    reproducer_split_threshold: float = 10.0  # slider
    reproduction_efficiency:    float = 0.7   # slider
    # Experimental switch (not in UI): True = offspring starts at age 0;
    # False = offspring inherits the parent's age (knock-out of age reset)
    offspring_age_reset: bool = True

    # ── Repair (fixed, identical for both types) ───────────────────────
    # health restored per step ≤ repair_max_per_step
    # energy cost = health restored / repair_efficiency
    repair_max_per_step: float = 0.05
    repair_efficiency:   float = 0.5    # 0.5 health per energy → 2 energy per unit health

    # ── Resource competition ───────────────────────────────────────────
    # weight = size ^ exclusivity  (0 = equal split, 1 = proportional, …)
    exclusivity: float = 0.5

    # ── Environment fields: resource and hazard follow the SAME rule ──
    #   value(cell, t) = level × (1 + S·u[patch]) × (1 + A·sin(2πt/P + φ[patch]))
    #   S = spatial amplitude, A = temporal amplitude, P = period (0 = static),
    #   L = patch size in cells (must divide the 50-cell grid: 1,2,5,10,25,50;
    #   L = 50 → one patch → the whole map changes in sync).
    #   See environment.py for details.  All amplitudes 0 → uniform, constant.

    # Resource: `resource_capacity` = K, the level the field modulates.
    # A grazed cell regrows about regrowth_rate × K per step.  Round 2 showed
    # that only this product matters (pairs with equal product but K = 10 vs 50
    # gave the same results), so the regrowth rate is FIXED at 0.2 — the value
    # every earlier round used — and the amount is set through K alone.
    # Resource follows capacity changes with a lag of about 1/0.2 = 5 steps,
    # so use periods P ≥ 20 for temporal fluctuation.
    resource_capacity:      float = 10.0
    resource_regrowth_rate: float = 0.2     # fixed (not a slider)
    resource_spatial_amp:   float = 0.0     # S
    resource_temporal_amp:  float = 0.0     # A
    resource_period:        float = 0.0     # P (steps; 0 = constant in time)
    resource_patch_size:    int   = 5       # L (cells)
    # initial resource is 0 everywhere; the first update + regrowth fills it

    # ── Age damage: a + b * age  (sliders) ─────────────────────────────
    age_damage_const: float = 0.0
    age_damage_slope: float = 0.0001

    # ── Hazard: uniform base level × the same dynamic rule as resource ─
    # Damage to an agent per step = sum of hazard over the cells inside its
    # capture radius (bigger body → more exposure).  0.0005 per cell equals
    # the map mean of the old patchy hazard at density 0.1, severity 0.005.
    hazard_level:        float = 0.0005
    hazard_spatial_amp:  float = 0.0        # S
    hazard_temporal_amp: float = 0.0        # A
    hazard_period:       float = 0.0        # P
    hazard_patch_size:   int   = 5          # L
    # Relation between the two fields' patterns: "independent" (default),
    # "positive" (hazard high where/when resource is high) or "negative"
    # (hazard low where/when resource is high).  Linked fields must share
    # patch size and period.
    hazard_link: str = "independent"


# ======================================================================
# Synchronous resource allocation  (vectorised, numpy)
# ======================================================================

def _claim_pairs(model, agents):
    """
    Find every (agent, cell) claim pair for all agents at once.

    Same rule as EnvironmentGrid.cells_within_radius():
      • candidate window: ±(ceil(radius / res) + 1) cells around the agent
      • a cell is claimed if its centre is within `radius` (torus distance)
      • the agent's own cell is always claimed
      • each cell is claimed at most once per agent (only matters when a
        body is so large that its window wraps around the whole world)

    Returns (agent_idx, cell_flat) as two equal-length int arrays.
    """
    env = model.environment
    s = model.scenario
    res, nr, nc = env.resolution, env.n_rows, env.n_cols

    pos = np.array([a.pos for a in agents], dtype=float)
    size = np.array([a.size for a in agents], dtype=float)
    x, y = pos[:, 0], pos[:, 1]
    radius = s.r0 + s.k_r * np.sqrt(size)

    col0 = np.floor_divide(x, res).astype(np.int64)
    row0 = np.floor_divide(y, res).astype(np.int64)
    k = np.ceil(radius / res).astype(np.int64) + 1

    ai_parts, cell_parts = [], []
    for kv in np.unique(k):
        idx = np.nonzero(k == kv)[0]
        d = np.arange(-kv, kv + 1)
        DR, DC = np.meshgrid(d, d, indexing="ij")
        DR, DC = DR.ravel(), DC.ravel()

        rows = (row0[idx, None] + DR[None, :]) % nr
        cols = (col0[idx, None] + DC[None, :]) % nc
        dx = np.abs(x[idx, None] - (cols + 0.5) * res)
        dy = np.abs(y[idx, None] - (rows + 0.5) * res)
        dx = np.minimum(dx, env.width - dx)
        dy = np.minimum(dy, env.height - dy)
        mask = (dx**2 + dy**2) ** 0.5 <= radius[idx, None]
        mask[:, (DR == 0) & (DC == 0)] = True          # own cell always counts

        ai_parts.append(np.broadcast_to(idx[:, None], mask.shape)[mask])
        cell_parts.append((rows * nc + cols)[mask])

    ai = np.concatenate(ai_parts)
    cell = np.concatenate(cell_parts)

    # de-duplicate (agent, cell) pairs
    key = np.unique(ai * (nr * nc) + cell)
    return key // (nr * nc), key % (nr * nc)


def allocate_resources(model):
    """
    Two-phase synchronous allocation, done in one vectorised pass.

    Phase 1 (claims): every agent claims each cell in its capture radius
        with weight = max(size, 0.01) ** exclusivity.
    Phase 2 (payout): every claimed cell gives its whole content to its
        claimants in proportion to weight, then is set to 0.

    Also stores each agent's hazard exposure for this step
    (= summed hazard over the same cells), applied later in individual_step.

    Mathematically identical to the earlier per-agent loop version; only
    the order of floating-point additions differs.
    """
    agents = list(model.agents)
    if not agents:
        return
    env = model.environment
    n_cells = env.n_rows * env.n_cols

    ai, cell = _claim_pairs(model, agents)

    size = np.array([a.size for a in agents], dtype=float)
    w = np.maximum(size, 0.01) ** model.scenario.exclusivity
    w_pair = w[ai]

    R = env.resource.reshape(-1)                       # view, writes go through
    total_w = np.bincount(cell, weights=w_pair, minlength=n_cells)
    avail = R[cell]
    gain_pair = np.where(avail > 0, avail * (w_pair / total_w[cell]), 0.0)
    gain = np.bincount(ai, weights=gain_pair, minlength=len(agents))

    exposure = np.bincount(
        ai, weights=env.hazard.reshape(-1)[cell], minlength=len(agents)
    )

    R[np.unique(cell)] = 0.0                           # claimed cells are emptied

    for a, g, h in zip(agents, gain, exposure):
        a.energy += float(g)
        a.hazard_exposure = float(h)


# ======================================================================
# Main Model
# ======================================================================

def _biomass(m, cls):
    return float(sum(a.size for a in m.agents_by_type.get(cls, [])))


def _mean_age(m, cls):
    ags = m.agents_by_type.get(cls, [])
    return float(np.mean([a.age for a in ags])) if len(ags) else float("nan")


def _maintainer_size(m, fn):
    ags = m.agents_by_type.get(Maintainer, [])
    return float(fn([a.size for a in ags])) if len(ags) else float("nan")


def _maintainer_share(m):
    r = _biomass(m, Reproducer)
    mt = _biomass(m, Maintainer)
    total = r + mt
    return mt / total if total > 0 else float("nan")


class ReproduceMaintainModel(Model):
    """
    Two life-history strategies competing in one world:
      Reproducer — splits when large enough; many small bodies
      Maintainer — never splits; one large, long-lived body
    """

    def __init__(self, scenario=None):
        if scenario is None:
            scenario = ReproduceMaintainScenario()
        super().__init__(scenario=scenario)

        self.space = ContinuousSpace(scenario.width, scenario.height, scenario.torus)

        self.environment = EnvironmentGrid(
            scenario.width,
            scenario.height,
            scenario.grid_resolution,
            resource_params=dict(
                level=scenario.resource_capacity,
                spatial_amp=scenario.resource_spatial_amp,
                temporal_amp=scenario.resource_temporal_amp,
                period=scenario.resource_period,
                patch_size=scenario.resource_patch_size,
            ),
            hazard_params=dict(
                level=scenario.hazard_level,
                spatial_amp=scenario.hazard_spatial_amp,
                temporal_amp=scenario.hazard_temporal_amp,
                period=scenario.hazard_period,
                patch_size=scenario.hazard_patch_size,
            ),
            rng=self.rng,
            hazard_link=scenario.hazard_link,
        )

        for cls, n in (
            (Reproducer, scenario.initial_reproducer_count),
            (Maintainer, scenario.initial_maintainer_count),
        ):
            for _ in range(n):
                pos = (
                    self.random.uniform(0, scenario.width),
                    self.random.uniform(0, scenario.height),
                )
                agent = cls(self, scenario.initial_energy, scenario.initial_size, pos)
                agent.pos = None
                self.space.place_agent(agent, pos)

        self.reproducer_extinction_step = None
        self.maintainer_extinction_step = None

        # One row per death (see Animal._die); diagnostic, for later analysis
        self.death_log = []

        self.datacollector = DataCollector(
            model_reporters={
                "Reproducers": lambda m: len(m.agents_by_type.get(Reproducer, [])),
                "Maintainers": lambda m: len(m.agents_by_type.get(Maintainer, [])),
                "TotalReproducerBiomass": lambda m: _biomass(m, Reproducer),
                "TotalMaintainerBiomass": lambda m: _biomass(m, Maintainer),
                "MaintainerBiomassShare": _maintainer_share,
                "MeanReproducerAge": lambda m: _mean_age(m, Reproducer),
                "MeanMaintainerAge": lambda m: _mean_age(m, Maintainer),
                "MedianMaintainerSize": lambda m: _maintainer_size(m, np.median),
                "MaxMaintainerSize": lambda m: _maintainer_size(m, np.max),
                "TotalResource": lambda m: float(m.environment.resource.sum()),
                "ReproducerExtinctionStep": lambda m: m.reproducer_extinction_step,
                "MaintainerExtinctionStep": lambda m: m.maintainer_extinction_step,
            }
        )

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        """
          0. Environment update: this step's resource capacity and hazard map
          1. Resource regrowth toward capacity
          2. All agents move (shuffled)
          3–4. Resource claims + payout (two-phase, vectorised);
               hazard exposure recorded for each agent
          5. Individual steps: maintenance / damage / repair / grow / split
          6. Extinction check (stop when both types are gone)
          7. Data collection
        """
        self.environment.update(self.steps)
        self.environment.regrow(self.scenario.resource_regrowth_rate)
        self.agents.shuffle_do("move")

        allocate_resources(self)

        self.agents.shuffle_do("individual_step")

        n_r = len(self.agents_by_type.get(Reproducer, []))
        n_m = len(self.agents_by_type.get(Maintainer, []))
        if self.reproducer_extinction_step is None and n_r == 0:
            self.reproducer_extinction_step = self.steps
        if self.maintainer_extinction_step is None and n_m == 0:
            self.maintainer_extinction_step = self.steps

        self.datacollector.collect(self)

        if n_r == 0 and n_m == 0:
            self.running = False
