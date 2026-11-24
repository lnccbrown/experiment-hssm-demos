// TS port of experiments/coherence_demo/coherence_timeline.py (+ the notebook's
// make_motion_coherence_trials and constant_stimuli_afc_timeline). Keep the two in sync.
import type { JsPsych } from "jspsych";
import htmlButtonResponse from "@jspsych/plugin-html-button-response";
import htmlKeyboardResponse from "@jspsych/plugin-html-keyboard-response";

export type DisplayParams = {
  n_dots: number;
  speed_px_s: number;
  dot_lifetime_s: number;
  canvas_width: number;
  canvas_height: number;
  seed: number;
};

// Same constants as the marimo notebook's MOTION_* cell.
export const DEFAULT_DISPLAY: DisplayParams = {
  n_dots: 100,
  speed_px_s: 120,
  dot_lifetime_s: 0.1,
  canvas_width: 500,
  canvas_height: 260,
  seed: 42,
};

const COUNTDOWN_SECONDS = 3;

// Same as Python's sum(ord(ch) for ch in s) & 0x7FFFFFFF, so trial seeds match.
const sumCharCodes = (s: string) =>
  Array.from(s, (ch) => ch.codePointAt(0) ?? 0).reduce((a, b) => a + b, 0) & 0x7fffffff;

/** Mirrors renderers/motion_coherence/motion_coherence_stimulus.py::motion_trial_stimulus_html. */
export function motionTrialStimulusHtml(
  level: number,
  direction: "left" | "right",
  trialId: string,
  d: DisplayParams,
) {
  const c = Math.max(0, Math.min(1, level));
  const w = Math.round(d.canvas_width);
  const h = Math.round(d.canvas_height);
  return `
<div class="motion-coherence-stimulus">
  <canvas
    id="rdk-${encodeURIComponent(trialId)}"
    class="motion-coherence-canvas"
    width="${w}"
    height="${h}"
    data-rdk="1"
    data-stim-level="${c}"
    data-dir-sign="${direction === "left" ? -1 : 1}"
    data-seed="${Math.round(d.seed) + sumCharCodes(trialId)}"
    data-n-dots="${Math.round(d.n_dots)}"
    data-speed-px-s="${d.speed_px_s}"
    data-dot-lifetime-s="${Math.max(0.001, d.dot_lifetime_s)}"
    style="width:${w}px;height:${h}px;max-width:${w}px;max-height:${h}px;"
  ></canvas>
</div>
`;
}

/** Intro -> countdown -> (motion trial, feedback) x levels*reps, as in build_coherence_timeline. */
export function buildCoherenceTimeline(
  jsPsych: JsPsych,
  levels: [number, number, number],
  repsPerLevel: number,
  display: DisplayParams,
) {
  const labelled = (["A", "B", "C"] as const).flatMap((label, i) =>
    Array.from({ length: repsPerLevel }, () => ({ label, level: levels[i] })),
  );

  const intro = {
    type: htmlButtonResponse,
    stimulus:
      '<div style="font-size:20px;line-height:1.6;text-align:center;">' +
      "Press left arrow for motion to the left, and right arrow for motion to the right</div>",
    choices: ["click here to begin."],
    button_layout: "grid",
    grid_rows: 1,
    grid_columns: 1,
  };

  const countdown = Array.from({ length: COUNTDOWN_SECONDS }, (_, i) => ({
    type: htmlKeyboardResponse,
    stimulus: `<div style="font-size:72px;font-weight:700;text-align:center;">${COUNTDOWN_SECONDS - i}</div>`,
    choices: "NO_KEYS",
    trial_duration: 1000,
    data: { task: "countdown" },
  }));

  const feedback = {
    type: htmlKeyboardResponse,
    stimulus: () => {
      const ok = !!jsPsych.data.get().last(1).values()[0]?.correct;
      return `<div style="font-size:30px;font-weight:700;color:${ok ? "#15803d" : "#dc2626"};text-align:center;">${ok ? "Correct" : "Incorrect"}</div>`;
    },
    choices: "NO_KEYS",
    trial_duration: 1000,
  };

  const trials = labelled.flatMap(({ label, level }, i) => {
    const motion = Math.random() < 0.5 ? "left" : "right";
    const correctIndex = motion === "right" ? 1 : 0;
    const correctKey = correctIndex ? "ArrowRight" : "ArrowLeft";
    const trial = {
      type: htmlKeyboardResponse,
      stimulus: motionTrialStimulusHtml(level, motion, `trial-${i}`, display),
      choices: ["ArrowLeft", "ArrowRight"],
      response_ends_trial: true,
      data: {
        stim_level: level,
        motion_direction: motion,
        correct_response: correctKey,
        task: "motion_coherence",
        label,
        trial_num: i + 1,
        correct_key: correctKey,
        correct_index: correctIndex,
        trial_index_py: i,
        stimulus_plugin: "motion_rdk",
      },
      on_load: () => window.__startAllMotionCanvases?.(),
      on_finish: (data: Record<string, unknown>) => {
        data.correct = jsPsych.pluginAPI.compareKeys(data.response as string, correctKey);
      },
    };
    return [trial, feedback];
  });

  return [intro, ...countdown, ...trials];
}
