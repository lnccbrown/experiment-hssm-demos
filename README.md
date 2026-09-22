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

### Notes

- The tutorial simulates a **binary left/right motion task** for multiple actors and trials using **jsPsych-style** trial objects in Python; motion previews use **Canvas** in the browser.
- HSSM fitting is triggered with a dedicated **Run hssm fit** button after simulation.
- The end-of-pipeline model visualization uses `hssm.plotting.plot_model_cartoon`.

### Project docs

- Detailed file/directory responsibilities: `DOCUMENTATION.md`
