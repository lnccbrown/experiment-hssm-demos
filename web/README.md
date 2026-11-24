# Motion coherence demo (web)

Standalone Vite+ React host for the jsPsych motion-coherence experiment from
`experiments/coherence_demo/` (same timeline as the marimo demo, run without Python).

```bash
cd web
vp install
vp dev      # http://localhost:5173
vp build    # static site in web/dist
```

- `src/experiment.ts` is a TS port of `coherence_timeline.py`. Change both together.
- The dot stimulus (`renderers/motion_coherence/`) and the results charts
  (`runtime/demo_results_charts.js`, `runtime/jspsych_runner.css`) are imported straight from the
  Python side, so there is one copy of each.
- **Download data (JSON)** saves jsPsych rows. `motion_trials_dataframe()` in
  `experiments/coherence_demo/motion_coherence_export.py` reads that file for HSSM fitting.
