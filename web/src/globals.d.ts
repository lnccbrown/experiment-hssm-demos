// Globals set by the shared plain-JS scripts in renderers/ and runtime/.
interface Window {
  __startAllMotionCanvases?: (selector?: string) => void;
  JsPsychDemoCharts: { mount: (rows: Record<string, unknown>[]) => void };
  vegaEmbed: unknown;
}
