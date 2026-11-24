# Motion coherence demo (web)

Standalone Vite+ React host for the jsPsych motion-coherence experiment from
`experiments/coherence_demo/` (same timeline as the marimo demo, run without Python).

```bash
cd web
vp install
vp dev      # http://localhost:5173
vp build    # static site in web/dist
```

## Native app (Tauri 2: macOS + iOS)

Needs Rust and full Xcode (with the iOS platform from Xcode > Settings > Components).
`xcode-select -p` must point at Xcode, not CommandLineTools: `tauri ios build` starts
`xcodebuild` without passing `DEVELOPER_DIR` through, so run
`sudo xcode-select -s /Applications/Xcode.app` once.

```bash
pnpm tauri dev                      # macOS window
pnpm tauri build                    # macOS .app/.dmg in src-tauri/target/release/bundle
pnpm tauri ios dev                  # iPhone simulator (pick one when prompted)
pnpm tauri ios build                # device build; set bundle.iOS.developmentTeam first
```

In the app, **Save data (JSON)** writes `motion_coherence_<timestamp>.json` to Documents
(`~/Documents` on macOS; the app's folder in the Files app on iOS). WKWebView ignores browser
downloads, hence the difference from the web build.

- `src/experiment.ts` is a TS port of `coherence_timeline.py`. Change both together.
- The dot stimulus (`renderers/motion_coherence/`) and the results charts
  (`runtime/demo_results_charts.js`, `runtime/jspsych_runner.css`) are imported straight from the
  Python side, so there is one copy of each.
- **Save data (JSON)** saves jsPsych rows. `motion_trials_dataframe()` in
  `experiments/coherence_demo/motion_coherence_export.py` reads that file for HSSM fitting.
