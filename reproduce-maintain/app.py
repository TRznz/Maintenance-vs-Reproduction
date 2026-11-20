"""
app.py
======
Solara-based interactive visualisation for the Reproduce-Maintain model.

Run with:
    solara run app.py

Three visual panels:
  1. Space view   — resource density, hazard zones, agents drawn at capture_radius
  2. Population   — Reproducer vs Maintainer counts over time
  3. Biomass      — total biomass of each type over time

Slider defaults are read from ReproduceMaintainScenario, so the UI and the
model always start from the same values.
"""

import math
import numpy as np
import solara
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

from mesa.visualization import SolaraViz, Slider, make_plot_component
from mesa.visualization.utils import update_counter

from model import ReproduceMaintainModel, ReproduceMaintainScenario
from agents import Reproducer, Maintainer

D = ReproduceMaintainScenario   # class-level defaults
PATCH_SIZES = [1, 2, 5, 10, 25, 50]   # must divide the 50 × 50 grid


# ======================================================================
# Space visualisation component
# ======================================================================

@solara.component
def SpaceView(model):
    """
    Layers:
      1. Green heatmap  — resource currently in each cell (grazed cells are 0)
      2. Red overlay    — current hazard map (uniform tint when the hazard
                          has no spatial or temporal variation)
      3. Circles        — each agent at its capture_radius (= foraging AND
                          hazard-exposure footprint)
    """
    update_counter.get()
    s = model.scenario
    fig = Figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    ax.imshow(
        model.environment.resource,
        extent=[0, s.width, 0, s.height],
        origin="lower",
        cmap="YlGn",
        vmin=0,
        vmax=max(model.environment.resource_field.max_value(), 1e-9),
        aspect="auto",
        alpha=0.75,
    )

    hazard = model.environment.hazard
    if hazard.max() > 0:
        ax.imshow(
            hazard,
            extent=[0, s.width, 0, s.height],
            origin="lower",
            cmap="Reds",
            vmin=0,
            vmax=model.environment.hazard_field.max_value(),
            aspect="auto",
            alpha=0.35,
        )

    def make_circles(agents, color, label):
        if not agents:
            return
        patches = [Circle(a.pos, radius=s.r0 + s.k_r * math.sqrt(a.size)) for a in agents]
        col = PatchCollection(
            patches, facecolor=color, alpha=0.45, edgecolor=color,
            linewidths=0.5, zorder=3,
        )
        ax.add_collection(col)
        ax.scatter([], [], c=color, alpha=0.7, label=label)

    reps   = list(model.agents_by_type.get(Reproducer, []))
    maints = list(model.agents_by_type.get(Maintainer, []))

    make_circles(reps,   "royalblue",  f"Reproducer (n={len(reps)})")
    make_circles(maints, "darkorange", f"Maintainer (n={len(maints)})")

    ax.set_xlim(0, s.width)
    ax.set_ylim(0, s.height)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    ax.set_title(f"Step {model.steps}", fontsize=9)

    solara.FigureMatplotlib(fig)


# ======================================================================
# Line charts
# ======================================================================

def post_process_legend(ax):
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)


population_chart = make_plot_component(
    {"Reproducers": "royalblue", "Maintainers": "darkorange"},
    post_process=post_process_legend,
)

biomass_chart = make_plot_component(
    {"TotalReproducerBiomass": "royalblue", "TotalMaintainerBiomass": "darkorange"},
    post_process=post_process_legend,
)


# ======================================================================
# Model parameters (20 controls)
# ======================================================================

model_params = {
    # ── Competition ─────────────────────────────────────────────────────
    "exclusivity": {
        "type": "SliderFloat", "label": "Resource Exclusivity",
        "value": D.exclusivity, "min": 0.0, "max": 3.0, "step": 0.05,
    },

    # ── Resource field: level × (1 + S·u) × (1 + A·sin(2πt/P + φ)) ──────
    "resource_capacity":     Slider("Resource capacity K (regrowth ≈ 0.2·K per step)", D.resource_capacity, 1.0, 100.0, 1.0),
    "resource_spatial_amp":  Slider("Resource spatial amplitude S", D.resource_spatial_amp, 0.0, 1.0, 0.05),
    "resource_temporal_amp": Slider("Resource temporal amplitude A", D.resource_temporal_amp, 0.0, 1.0, 0.05),
    "resource_period":       Slider("Resource period P (0 = static)", D.resource_period, 0, 500, 10),
    "resource_patch_size": {
        "type": "Select", "label": "Resource patch size L (cells; 50 = whole map)",
        "value": D.resource_patch_size, "values": PATCH_SIZES,
    },

    # ── Reproduction ────────────────────────────────────────────────────
    "reproducer_split_threshold": Slider("Reproducer Split Threshold", D.reproducer_split_threshold, 2.0, 50.0, 1.0),
    "reproduction_efficiency":    Slider("Reproduction Efficiency",    D.reproduction_efficiency,    0.1,  1.0, 0.05),

    # ── Growth gate (Maintainer only) ───────────────────────────────────
    "growth_min_health":      Slider("Maintainer grows only if health ≥",         D.growth_min_health,      0.0, 1.0, 0.01),
    "growth_min_energy_frac": Slider("Maintainer grows only if energy ≥ (× cap)", D.growth_min_energy_frac, 0.0, 1.0, 0.05),

    # ── Age damage: a + b * age ─────────────────────────────────────────
    "age_damage_const": Slider("Age Damage a (constant)", D.age_damage_const, 0.0, 0.05,  0.001),
    "age_damage_slope": Slider("Age Damage b (per age)",  D.age_damage_slope, 0.0, 0.001, 0.00005),

    # ── Size energy cost: c0 + c1 * size ────────────────────────────────
    "c0": Slider("Upkeep c0 (constant)", D.c0, 0.0, 1.0, 0.05),
    "c1": Slider("Upkeep c1 (per size)", D.c1, 0.0, 0.5, 0.01),

    # ── Hazard field: same rule; damage = sum over capture radius ───────
    "hazard_level":        Slider("Hazard per cell", D.hazard_level, 0.0, 0.003, 0.0001),
    "hazard_spatial_amp":  Slider("Hazard spatial amplitude S", D.hazard_spatial_amp, 0.0, 1.0, 0.05),
    "hazard_temporal_amp": Slider("Hazard temporal amplitude A", D.hazard_temporal_amp, 0.0, 1.0, 0.05),
    "hazard_period":       Slider("Hazard period P (0 = static)", D.hazard_period, 0, 500, 10),
    "hazard_link": {
        "type": "Select", "label": "Hazard pattern vs resource (linked: same patch size & period)",
        "value": D.hazard_link, "values": ["independent", "positive", "negative"],
    },
    "hazard_patch_size": {
        "type": "Select", "label": "Hazard patch size L (cells; 50 = whole map)",
        "value": D.hazard_patch_size, "values": PATCH_SIZES,
    },
}


# ======================================================================
# Assemble page
# ======================================================================

model = ReproduceMaintainModel(scenario=ReproduceMaintainScenario())

page = SolaraViz(
    model,
    components=[SpaceView, population_chart, biomass_chart],
    model_params=model_params,
    name="Reproduce vs. Maintain",
)

page  # noqa: B018
