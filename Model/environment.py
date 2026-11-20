"""
environment.py
==============
EnvironmentGrid: numpy-backed resource and hazard fields.  Neither is a Mesa
Agent — each field is updated with one vectorised call per step.

Both fields follow the SAME rule (Nov 2026 redesign):

    value(cell, t) = level × (1 + S · u[patch]) × (1 + A · sin(2π t / P + φ[patch]))

    level  total amount            resource: capacity K (a grazed cell regrows
                                             about 0.2 × K per step)
                                   hazard:   per-cell hazard per step
    S      spatial amplitude       how different patches are from each other
                                   (0 = spatially uniform)
    A      temporal amplitude      how much each patch rises and falls over time
                                   (0 = constant in time)
    P      period in steps         how fast it changes (0 = constant in time)
    L      patch size in cells     grain of both the spatial pattern and the
                                   temporal asynchrony; L = whole grid → one
                                   patch → the whole map rises and falls together

    u[patch]  fixed spatial offsets, evenly spaced in [-1, 1] and shuffled
              → their mean over patches is exactly 0
    φ[patch]  fixed random phases in [0, 2π)

Linking (hazard_link): by default the two fields draw their own patches and
phases independently.  "positive" makes hazard reuse the resource field's
offsets and phases (rich places/times are also dangerous); "negative" negates
the offsets and shifts the phases by π (rich places/times are safe).  Both
fields must then use the same patch size and period.

Means are preserved: the spatial factor averages to 1 over the map and the
temporal factor averages to 1 over a full period, so the long-run, map-wide
mean of each field equals `level`.  Both factors stay ≥ 0 when S, A ≤ 1.
"""

import numpy as np


class DynamicField:
    """Patch structure + per-step multiplier for one field (resource or hazard)."""

    def __init__(self, n_rows, n_cols, level, spatial_amp, temporal_amp,
                 period, patch_size, rng):
        L = int(patch_size)
        if L < 1 or n_rows % L or n_cols % L:
            raise ValueError(
                f"patch_size {L} must divide the grid ({n_rows} x {n_cols} cells)"
            )
        self.level = float(level)
        self.S = float(spatial_amp)
        self.A = float(temporal_amp)
        self.P = float(period)

        pr, pc = n_rows // L, n_cols // L
        n_patches = pr * pc

        # patch index of every cell
        rows = np.arange(n_rows)[:, None] // L
        cols = np.arange(n_cols)[None, :] // L
        patch_of_cell = rows * pc + cols

        # spatial offsets: evenly spaced in [-1, 1], shuffled (mean exactly 0)
        if n_patches == 1:
            u = np.zeros(1)
        else:
            u = np.linspace(-1.0, 1.0, n_patches)
            rng.shuffle(u)
        phase = rng.uniform(0.0, 2.0 * np.pi, n_patches)

        self.patch_size = L
        self.u = u[patch_of_cell]                           # spatial offset per cell
        self.spatial = 1.0 + self.S * self.u                # fixed for the run
        self.phase = phase[patch_of_cell]                   # fixed for the run

    def link_to(self, other, sign):
        """
        Make this field's pattern follow another field's patches.
          sign = +1 (positive): same spatial offsets and phases — where/when
                                `other` is high, this field is high too
          sign = -1 (negative): offsets negated, phases shifted by π — where/
                                when `other` is high, this field is low
        Requires the same patch size and period.  This field's own random
        draws were already made (so the random-number stream is unchanged);
        they are simply overwritten.
        """
        if self.patch_size != other.patch_size:
            raise ValueError("linked fields need the same patch size "
                             f"({self.patch_size} vs {other.patch_size})")
        if self.A > 0 and other.A > 0 and self.P != other.P:
            raise ValueError("linked fields need the same period "
                             f"({self.P} vs {other.P})")
        self.u = sign * other.u
        self.spatial = 1.0 + self.S * self.u
        self.phase = other.phase + (0.0 if sign > 0 else np.pi)

    def value(self, t):
        """Current field values (n_rows × n_cols) at step t."""
        if self.A == 0.0 or self.P <= 0.0:
            temporal = 1.0
        else:
            temporal = 1.0 + self.A * np.sin(2.0 * np.pi * t / self.P + self.phase)
        return self.level * self.spatial * temporal

    def max_value(self):
        """Largest value the field can ever take (for colour scales)."""
        return self.level * (1.0 + self.S) * (1.0 + self.A)


class EnvironmentGrid:
    """
    Two 2-D arrays over the world:
      • resource[row, col]  — resource currently in the cell
      • hazard[row, col]    — hazard dealt this step to every agent whose
                              capture radius covers the cell
    plus `capacity[row, col]`, the resource level the cell is regrowing toward
    this step.

    Each step, update(t) recomputes capacity and hazard from their dynamic
    fields, then regrow() moves resource toward capacity.
    """

    def __init__(self, width, height, grid_resolution, resource_params,
                 hazard_params, rng, hazard_link="independent"):
        """
        resource_params / hazard_params: dicts with keys
            level, spatial_amp, temporal_amp, period, patch_size
        hazard_link: "independent" (default), "positive" (hazard high where
            and when resource is high) or "negative" (hazard low where and
            when resource is high)
        """
        self.width = width
        self.height = height
        self.resolution = grid_resolution
        self.n_cols = int(np.ceil(width / grid_resolution))
        self.n_rows = int(np.ceil(height / grid_resolution))
        self.rng = rng

        self.resource_field = DynamicField(self.n_rows, self.n_cols, rng=rng, **resource_params)
        self.hazard_field = DynamicField(self.n_rows, self.n_cols, rng=rng, **hazard_params)
        if hazard_link == "positive":
            self.hazard_field.link_to(self.resource_field, +1)
        elif hazard_link == "negative":
            self.hazard_field.link_to(self.resource_field, -1)
        elif hazard_link != "independent":
            raise ValueError(f"unknown hazard_link {hazard_link!r}")

        # Resource starts EMPTY; after the first regrowth each cell holds
        # regrowth_rate × capacity, the level of a cell grazed every step.
        self.resource = np.zeros((self.n_rows, self.n_cols))
        self.capacity = self.resource_field.value(0)
        self.hazard = self.hazard_field.value(0)

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    def update(self, t):
        """Recompute this step's resource capacity and hazard map."""
        self.capacity = self.resource_field.value(t)
        self.hazard = self.hazard_field.value(t)

    def regrow(self, regrowth_rate):
        """
        Linear move toward the current capacity:
            R ← R + regrowth_rate × (capacity − R)
        A grazed (emptied) cell regrows about regrowth_rate × capacity per
        step; an ungrazed cell fills up to capacity.  If capacity drops below
        R, R falls toward it at the same rate (lag ≈ 1 / regrowth_rate steps).
        """
        self.resource += regrowth_rate * (self.capacity - self.resource)
        np.clip(self.resource, 0, None, out=self.resource)

    # ------------------------------------------------------------------
    # Spatial queries
    # ------------------------------------------------------------------

    def cell_index(self, pos):
        """Convert continuous (x, y) to (row, col) with torus wrapping."""
        x, y = pos
        col = int(x // self.resolution) % self.n_cols
        row = int(y // self.resolution) % self.n_rows
        return row, col

    def cells_within_radius(self, pos, radius):
        """
        (row, col) cells whose centres lie within `radius` (torus distance),
        plus the agent's own cell.  Reference implementation; the model uses
        the vectorised version in model._claim_pairs().
        """
        x, y = pos
        r_cells = int(np.ceil(radius / self.resolution)) + 1
        col0 = int(x // self.resolution)
        row0 = int(y // self.resolution)
        own = (row0 % self.n_rows, col0 % self.n_cols)

        result = [own]
        for dr in range(-r_cells, r_cells + 1):
            for dc in range(-r_cells, r_cells + 1):
                r = (row0 + dr) % self.n_rows
                c = (col0 + dc) % self.n_cols
                if (r, c) == own or (r, c) in result:
                    continue
                dx = abs(x - (c + 0.5) * self.resolution)
                dy = abs(y - (r + 0.5) * self.resolution)
                dx = min(dx, self.width - dx)
                dy = min(dy, self.height - dy)
                if (dx**2 + dy**2) ** 0.5 <= radius:
                    result.append((r, c))
        return result

    def hazard_sum(self, cells):
        """Summed hazard over a list of (row, col) cells."""
        if not cells:
            return 0.0
        rows, cols = zip(*cells)
        return float(self.hazard[list(rows), list(cols)].sum())
