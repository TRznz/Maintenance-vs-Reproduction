"""
agents.py
=========
Animal     — shared base class (energy, health, age, size; move/damage/repair/grow)
Reproducer — splits into two when size >= split_threshold AND energy is sufficient
             Priority order: try_grow() first, then repair()
Maintainer — never splits (split_threshold = inf)
             Priority order: repair() first, then try_grow()

The ONLY structural differences between the two types are:
  1. split_threshold value
  2. allocate_surplus() priority order
  3. growth gate: Maintainer grows only if health and energy are high enough;
     Reproducer grows unconditionally (its size is already capped by the
     split threshold, and its strategy is to reproduce, not to maintain)
Everything else — including repair efficiency and the per-step repair cap —
is identical.

Changes (Nov 2026):
  • Repair: at most `repair_max_per_step` health per step, bought at
    `repair_efficiency` health per unit energy (same for both types).
  • Intrinsic damage: age only, linear form  a + b * age.
    The old size-based intrinsic damage is removed.
  • Extrinsic damage: SUM of hazard over every cell inside the capture
    radius (the same cells the agent forages from) → bigger body, more
    exposure.
"""

import math
from mesa import Agent


class Animal(Agent):
    """
    Base class for all organisms.

    State variables:
        energy        — metabolic resource; < 0 → death
        health        — [0, 1]; ≤ 0 → death
        age           — steps survived
        size          — body mass / biomass
        hazard_exposure — summed hazard over the cells inside the capture
                          radius this step (set by the model during
                          resource allocation)
    """

    def __init__(self, model, energy, size, pos):
        super().__init__(model)
        self.energy = energy
        self.health = 1.0
        self.age = 0
        self.size = size
        self.pos = pos
        self.hazard_exposure = 0.0
        # lifetime damage bookkeeping (diagnostic only — does not affect dynamics)
        self.cum_age_damage = 0.0
        self.cum_hazard_damage = 0.0
        self.last_age_damage = 0.0

    # ------------------------------------------------------------------
    # Size-derived properties
    # ------------------------------------------------------------------

    @property
    def capture_radius(self):
        """Foraging AND hazard-exposure radius: r0 + k_r * sqrt(size)."""
        s = self.model.scenario
        return s.r0 + s.k_r * math.sqrt(self.size)

    @property
    def maintenance_cost(self):
        """Per-step energy upkeep: c0 (constant part) + c1 * size (linear part)."""
        s = self.model.scenario
        return s.c0 + s.c1 * self.size

    @property
    def energy_capacity(self):
        """Maximum energy reserve: E0_cap + k_cap * size."""
        s = self.model.scenario
        return s.E0_cap + s.k_cap * self.size

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def move(self):
        """Random walk, step uniformly in [-step, +step] on each axis (torus)."""
        s = self.model.scenario
        dx = self.random.uniform(-s.move_step_size, s.move_step_size)
        dy = self.random.uniform(-s.move_step_size, s.move_step_size)
        x, y = self.pos
        new_x = (x + dx) % s.width
        new_y = (y + dy) % s.height
        self.model.space.move_agent(self, (new_x, new_y))
        self.pos = (new_x, new_y)

    # ------------------------------------------------------------------
    # Damage
    # ------------------------------------------------------------------

    def apply_intrinsic_damage(self):
        """
        Age damage per step = a + b * age
          a > 0, b = 0  → constant damage
          a = 0, b > 0  → linearly increasing damage
          a = b = 0     → no ageing
        """
        s = self.model.scenario
        d = s.age_damage_const + s.age_damage_slope * self.age
        self.health -= d
        self.last_age_damage = d
        self.cum_age_damage += d

    def apply_extrinsic_damage(self):
        """
        Hazard damage = sum of hazard values over all cells inside the
        capture radius (the same cells it forages from).  A larger body
        covers more cells and therefore accumulates more hazard.
        The sum is computed by the model in allocate_resources().
        """
        self.health -= self.hazard_exposure
        self.cum_hazard_damage += self.hazard_exposure

    # ------------------------------------------------------------------
    # Repair  (identical for both types; only call-order differs)
    # ------------------------------------------------------------------

    def repair(self):
        """
        Restore health using surplus energy, limited by:
          • missing health           (1 - health)
          • per-step cap             repair_max_per_step
          • what surplus can buy     surplus * repair_efficiency

        Energy spent = health_restored / repair_efficiency.
        """
        if self.health >= 1.0:
            return
        s = self.model.scenario
        if s.repair_efficiency <= 0 or s.repair_max_per_step <= 0:
            return

        surplus = max(0.0, self.energy - s.energy_buffer)
        amount = min(
            1.0 - self.health,
            s.repair_max_per_step,
            surplus * s.repair_efficiency,
        )
        if amount <= 0:
            return
        self.health += amount
        self.energy -= amount / s.repair_efficiency

    # ------------------------------------------------------------------
    # Growth
    # ------------------------------------------------------------------

    def can_grow(self):
        """
        Growth gate (used by Maintainer; Reproducer overrides it).
        Grow only when BOTH hold:
          • health ≥ growth_min_health
            (repair is keeping up with damage)
          • energy ≥ growth_min_energy_frac * energy_capacity
            (reserves are reasonably full)
        Without this gate a Maintainer grows until its summed hazard
        exposure exceeds the repair cap, and then dies of its own size.
        """
        s = self.model.scenario
        return (
            self.health >= s.growth_min_health
            and self.energy >= s.growth_min_energy_frac * self.energy_capacity
        )

    def try_grow(self):
        """
        Invest a fixed fraction (growth_invest_rate) of surplus into size at
        growth_efficiency.  Skipped if size ≥ split_threshold or the growth
        gate (can_grow) is closed.
        """
        s = self.model.scenario
        if self.size >= self.split_threshold or not self.can_grow():
            return
        surplus = self.energy - s.energy_buffer
        if surplus > 0:
            invest = surplus * s.growth_invest_rate
            self.size += invest * s.growth_efficiency
            self.energy -= invest

    # ------------------------------------------------------------------
    # Split trigger
    # ------------------------------------------------------------------

    def ready_to_split(self):
        """size ≥ threshold AND energy ≥ 2 * energy_buffer (Maintainer: never)."""
        s = self.model.scenario
        return (
            self.size >= self.split_threshold
            and self.energy >= 2 * s.energy_buffer
        )

    # ------------------------------------------------------------------
    # Surplus allocation  (subclasses define priority order)
    # ------------------------------------------------------------------

    def allocate_surplus(self):
        raise NotImplementedError

    def split(self):
        """Default split does nothing. Only Reproducer overrides this."""
        pass

    # ------------------------------------------------------------------
    # Per-step logic  (after the synchronous resource phase)
    # ------------------------------------------------------------------

    def individual_step(self):
        """
          1. Age
          2. Pay maintenance cost
          3. Intrinsic (age) + extrinsic (hazard) damage
          4. Death check
          5. Cap energy at capacity
          6. Allocate surplus (repair / grow — subclass decides order)
          7. Split check
        """
        self.age += 1
        self.energy -= self.maintenance_cost
        self.apply_intrinsic_damage()
        self.apply_extrinsic_damage()

        if self.health <= 0 or self.energy < 0:
            self._die()
            return

        self.energy = min(self.energy, self.energy_capacity)

        self.allocate_surplus()

        if self.ready_to_split():
            self.split()

    def _die(self):
        """Record the death (diagnostic log), then remove from space and model."""
        s = self.model.scenario
        self.model.death_log.append({
            "step": self.model.steps,
            "type": type(self).__name__,
            "age": self.age,
            "size": self.size,
            "cause": "health" if self.health <= 0 else "energy",
            "cum_age_damage": self.cum_age_damage,
            "cum_hazard_damage": self.cum_hazard_damage,
            # True = age damage alone already exceeded the repair cap:
            # this agent could not have survived even with zero hazard
            "age_exceeds_repair": self.last_age_damage >= s.repair_max_per_step,
        })
        try:
            self.model.space.remove_agent(self)
        except Exception:
            pass
        self.remove()


# ======================================================================
# Reproducer
# ======================================================================

class Reproducer(Animal):
    """Grow toward split_threshold first; repair with what is left."""

    def __init__(self, model, energy, size, pos):
        super().__init__(model, energy, size, pos)
        self.split_threshold = model.scenario.reproducer_split_threshold

    def can_grow(self):
        """
        Reproducer grows unconditionally.  Its size is capped by the split
        threshold anyway, and gating growth on post-damage health (checked
        before this type repairs) would block reproduction whenever per-step
        damage exceeds ~0.01.
        """
        return True

    def allocate_surplus(self):
        self.try_grow()
        self.repair()

    def split(self):
        """
        Asexual budding:
          • Parent gives half its energy; offspring keeps
            that half × reproduction_efficiency
          • Size is halved; offspring gets the same size (biomass conserved)
          • Offspring: health 1; age 0 if offspring_age_reset is True,
            otherwise it inherits the parent's age.
            Parent keeps its own age and health.
        """
        s = self.model.scenario
        transfer = self.energy / 2
        self.energy -= transfer
        offspring_energy = transfer * s.reproduction_efficiency
        self.size /= 2

        offspring = Reproducer(self.model, offspring_energy, self.size, self.pos)
        if not s.offspring_age_reset:
            offspring.age = self.age              # knock-out: no age reset
        offspring.pos = None                      # clear so place_agent doesn't warn
        self.model.space.place_agent(offspring, self.pos)


# ======================================================================
# Maintainer
# ======================================================================

class Maintainer(Animal):
    """Repair first; grow with what is left.  Never splits."""

    def __init__(self, model, energy, size, pos):
        super().__init__(model, energy, size, pos)
        self.split_threshold = math.inf

    def allocate_surplus(self):
        self.repair()
        self.try_grow()
