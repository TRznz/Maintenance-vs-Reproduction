# Reproduce–Maintain: reproduction as one strategy for persistence

Agent-based model (Python, [Mesa](https://mesa.readthedocs.io)) accompanying the poster
**"Reproduction as One Strategy for Persistence: Rethinking the Boundaries of Life"**
(Tianqin Ren, Department of Philosophy, University of Missouri; PSA 2026).

Two types of agents compete for resource on a map that also carries environmental hazard:

- **Reproducers** grow to a size threshold and split; offspring start at age 0.
- **Maintainers** never split; they repair first and keep growing.

Both types follow identical rules for movement, foraging, costs and damage. They differ only in how
they spend surplus energy. Each run ends when one type goes extinct; the outcome is which type dies out first.

## Run the interactive model

```bash
pip install -r requirements.txt
solara run app.py
```

A browser window opens at `http://localhost:8765`. On a Mac, if you see
`[Errno 8] nodename nor servname provided`, run `solara run app.py --host 127.0.0.1` instead.

## Files

| File | Contents |
|---|---|
| `model.py` | Model and default parameters (`ReproduceMaintainScenario`) |
| `agents.py` | Reproducer and Maintainer: step order, growth, repair, splitting |
| `environment.py` | Resource and hazard fields (shared rule for patchiness and fluctuation) |
| `app.py` | Interactive interface (SolaraViz) |
| `batch_run.py` | Batch runner and per-run summary statistics |
| `run_rounds.py` | Parameter grids for the rounds shown on the poster |
| `make_fig1_A.py`, `make_fig2_B.py`, `make_fig3_final.py` | Poster figures 1–3 |
| `results/round_*/summary.csv` | One row per run (parameters and outcomes) |

## Reproduce the poster figures

The summary files are included, so the figures can be drawn without re-running the model.
Run these from this folder; output goes to `figs/`.

```bash
python make_fig1_A.py      # Fig 1 · Damage   (Round A)
python make_fig2_B.py      # Fig 2 · Resource (Round B)
python make_fig3_final.py  # Fig 3 · Fluctuating environments (Round C)
```

To re-run a round from scratch (this takes a while; it uses all CPU cores):

```bash
python run_rounds.py A
python run_rounds.py B
python run_rounds.py D2    # = Round C on the poster (alias: python run_rounds.py C)
```

Each round writes `results/round_<name>/summary.csv`, plus per-run time series and Maintainer death
records (not included here because of their size).

| Poster | Code | Content | Runs |
|---|---|---|---|
| Round A | `A` | hazard per cell × age-damage slope, static environment | 35 cells × 20 seeds |
| Round B | `B` | resource per step × exclusivity, static environment | 30 cells × 20 seeds |
| Round C | `D2` | resource and/or hazard fluctuation × patch size (Fig 3 uses period 100) | 60 seeds per cell |

Robustness checks at two further baselines on the Fig 1 boundary are in `results/round_J`
(hazard 3×10⁻⁴, aging slope 3×10⁻⁴) and `results/round_K1`–`K3` (adds hazard 2×10⁻⁴, aging slope 5×10⁻⁴).

## Key outcome columns in `summary.csv`

- `first_extinct`: `Reproducer`, `Maintainer`, `both`, or empty (neither died out before the step limit)
- `ext_R`, `ext_M`: step at which each type went extinct (empty if it did not)
- `cap_hit_step`: step at which the Reproducer population hit the 30,000-agent cap and the run was stopped
- Columns starting with `_` (e.g. `_cond`) are labels for grouping, not model parameters

## Notes

The model is conceptual: parameter values (repair cap, damage rates, resource levels) are not calibrated
to any organism. Outcomes depend on damage relative to the per-step repair cap, so the axes of Fig 1 are
effectively in units of that cap.

Code implementation, data processing and visualization were assisted by Claude (Anthropic); the author is
responsible for model design, conceptual analysis and any remaining errors.

Contact: rentianqin@gmail.com · www.rentianqin.com
