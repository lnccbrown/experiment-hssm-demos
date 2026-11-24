import { useEffect, useState } from "react";
import { initJsPsych, type JsPsych } from "jspsych";
import { isTauri } from "@tauri-apps/api/core";
import { BaseDirectory, writeTextFile } from "@tauri-apps/plugin-fs";
import vegaEmbed from "vega-embed";
import "jspsych/css/jspsych.css";
// Shared with the marimo runner; plain scripts that install window globals.
import "../../runtime/jspsych_runner.css";
import "../../renderers/motion_coherence/motion_coherence.css";
import "../../renderers/motion_coherence/motion_coherence.js";
import "../../runtime/demo_results_charts.js";
import { buildCoherenceTimeline, DEFAULT_DISPLAY } from "./experiment";

window.vegaEmbed = vegaEmbed;

// WKWebView (the Tauri webview on macOS/iOS) ignores <a download>, so the app writes to
// Documents instead (shown in the iOS Files app via src-tauri/Info.ios.plist).
async function saveData(jsPsych: JsPsych): Promise<string> {
  const name = `motion_coherence_${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
  if (!isTauri()) {
    jsPsych.data.get().localSave("json", name);
    return `Downloaded ${name}`;
  }
  await writeTextFile(name, jsPsych.data.get().json(), { baseDir: BaseDirectory.Document });
  return `Saved to Documents/${name}`;
}

type Settings = { a: number; b: number; c: number; reps: number; lifetime: number };

const clamp = (v: number, lo: number, hi: number, fallback: number) =>
  Number.isFinite(v) ? Math.max(lo, Math.min(hi, v)) : fallback;

function NumberField(props: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <label>
      {props.label}
      <input
        type="number"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        onChange={(e) => props.onChange(e.target.valueAsNumber)}
      />
    </label>
  );
}

export default function App() {
  const [settings, setSettings] = useState<Settings>({
    a: 0.2,
    b: 0.5,
    c: 0.8,
    reps: 5,
    lifetime: 0.1,
  });
  const [runId, setRunId] = useState(0);
  const [finished, setFinished] = useState<JsPsych | null>(null);
  const [saveStatus, setSaveStatus] = useState("");
  const set = (key: keyof Settings) => (v: number) => setSettings((s) => ({ ...s, [key]: v }));

  useEffect(() => {
    if (runId === 0) return;
    let stale = false;
    const jsPsych = initJsPsych({
      display_element: "jspsych-target",
      on_finish: () => {
        if (stale) return;
        const rows = JSON.parse(jsPsych.data.get().json()) as Record<string, unknown>[];
        window.JsPsychDemoCharts.mount(rows.filter((r) => r.task === "motion_coherence"));
        setFinished(jsPsych);
      },
    });
    const levels: [number, number, number] = [
      clamp(settings.a, 0, 1, 0.2),
      clamp(settings.b, 0, 1, 0.5),
      clamp(settings.c, 0, 1, 0.8),
    ];
    const display = { ...DEFAULT_DISPLAY, dot_lifetime_s: clamp(settings.lifetime, 0.01, 10, 0.1) };
    void jsPsych.run(
      buildCoherenceTimeline(jsPsych, levels, Math.round(clamp(settings.reps, 1, 20, 5)), display),
    );
    return () => {
      stale = true;
      jsPsych.pluginAPI.clearAllTimeouts();
    };
    // Settings are read once per run; only a (re)start rebuilds the timeline.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);

  const restart = () => {
    setFinished(null);
    setSaveStatus("");
    setRunId((n) => n + 1);
  };

  return (
    <main>
      <h1>Motion coherence demo</h1>
      <p>
        Binary left/right random-dot motion task at three coherence levels. Settings apply when you
        (re)start the demo.
      </p>
      <div className="controls">
        <NumberField
          label="Coherence A (proportion)"
          value={settings.a}
          min={0}
          max={1}
          step={0.05}
          onChange={set("a")}
        />
        <NumberField
          label="Coherence B (proportion)"
          value={settings.b}
          min={0}
          max={1}
          step={0.05}
          onChange={set("b")}
        />
        <NumberField
          label="Coherence C (proportion)"
          value={settings.c}
          min={0}
          max={1}
          step={0.05}
          onChange={set("c")}
        />
        <NumberField
          label="Trials per level (count)"
          value={settings.reps}
          min={1}
          max={20}
          step={1}
          onChange={set("reps")}
        />
        <NumberField
          label="Dot lifetime (s)"
          value={settings.lifetime}
          min={0.01}
          max={10}
          step={0.01}
          onChange={set("lifetime")}
        />
      </div>
      {/* New key per run gives each jsPsych instance a fresh element, dropping the old one's listeners. */}
      <div id="jspsych-target" key={runId} />
      <div className="actions">
        <button type="button" onClick={restart}>
          {runId === 0 ? "Start demo" : "Restart demo"}
        </button>
        {finished && (
          <button
            type="button"
            onClick={() =>
              saveData(finished).then(setSaveStatus, (e: unknown) =>
                setSaveStatus(`Save failed: ${String(e)}`),
              )
            }
          >
            Save data (JSON)
          </button>
        )}
        <span role="status">{saveStatus}</span>
      </div>
    </main>
  );
}
