## Simulator

This project provides an experiment-to-hssm pipeline. Demos are currently made as `marimo` apps.

It is organized as a modular workflow that combines:

- a motion coherence experiment simulator jsPsych demo (embedded in the app)
- simulation and HSSM fitting controls for iterative workflow
- an HSSM analysis stage with behavior summaries and plots.

### Run

From this directory:

**Prior predictive actor** (Gaussian subject sampling + prior predictive forward model):

```bash
uv run marimo run experiments/coherence_demo/priorPredictive_coherence_demo.py
```

**Forward DDM actor** (ssm-simulators, SSM parameters):

```bash
uv run marimo run experiments/coherence_demo/ssm_coherence_demo.py
```

Or open in the editor:

```bash
uv run marimo edit experiments/coherence_demo/ssm_coherence_demo.py
```

**Probabilistic Selection Task** (a shortened reinforcement-learning + DDM teaching variant based on Pedersen, Frank & Biele 2017): play the task, simulate players, fit the RL-DDM with HSSM, and inspect a single-dataset recovery illustration:

```bash
uv run marimo run experiments/pst_demo/pst_app.py
```

The app starts with a quick 30–60-choice learning game and no final round. Select **Thorough** to
restore the original demo schedule of 120–240 learning choices; the optional all-pairs final round
can be enabled independently.

**Probabilistic Approach-Avoidance Task** (a pedagogical PAAT based on Cheng et al. 2026): play a count-matched synthetic design, simulate the stock `angle` model, fit its HSSM drift regression, and inspect support- and deadline-aware checks:

```bash
uv run marimo run experiments/paat_demo/paat_app.py
```

### Tests

```bash
uv run pytest            # fast tests (Python + browser scripts via node)
uv run pytest -m slow    # computationally intensive real HSSM fits
```

### Notes

- The tutorial simulates a **binary left/right motion task** for multiple actors and trials using **jsPsych-style** trial objects in Python; motion previews use **Canvas** in the browser.
- HSSM fitting is triggered with a dedicated **Run hssm fit** button after simulation.
- The end-of-pipeline model visualization uses `hssm.plotting.plot_model_cartoon`.

### Project docs

- Detailed file/directory responsibilities: `DOCUMENTATION.md`
- PST app design, decisions and validation results: `PST_PLAN.md`
- PAAT scientific contract, design limitations and validation gate: `PAAT_PLAN.md`
