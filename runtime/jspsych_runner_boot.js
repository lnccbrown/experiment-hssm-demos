/** Boot script: reads runner config + timeline payloads and starts jsPsych. */
(() => {
  const boot = () => {
    const config = JsPsychRunnerCore.decodeB64Json("__RUNNER_CONFIG_B64__");
    JsPsychRunnerCore.start(config, "__TIMELINE_B64__");
  };
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    boot();
  }
})();
