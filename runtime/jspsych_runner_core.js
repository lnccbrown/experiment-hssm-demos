/** Generic jsPsych timeline runner (plugin-agnostic). */
window.JsPsychRunnerCore = (() => {
  const PLUGIN_CTORS = {
    "html-keyboard-response": () =>
      typeof jsPsychHtmlKeyboardResponse !== "undefined" ? jsPsychHtmlKeyboardResponse : null,
    "html-button-response": () =>
      typeof jsPsychHtmlButtonResponse !== "undefined" ? jsPsychHtmlButtonResponse : null,
  };

  const REVIVE_KEYS = [
    "on_finish",
    "on_start",
    "on_load",
    "stimulus",
    "conditional_function",
    "loop_function",
    "on_timeline_start",
    "on_timeline_finish",
  ];

  function decodeB64Json(b64) {
    return JSON.parse(atob(b64));
  }

  function reviveCallback(obj, key) {
    if (typeof obj[key] === "string" && obj[key].trim().startsWith("function")) {
      obj[key] = eval("(" + obj[key] + ")");
    }
  }

  function buildTypeMap(pluginNames) {
    const typeMap = {};
    for (const name of pluginNames) {
      const ctorFactory = PLUGIN_CTORS[name];
      if (ctorFactory) {
        typeMap[name] = ctorFactory();
      }
    }
    return typeMap;
  }

  /** Map plugin names to constructors and revive callbacks, recursing into nested timelines. */
  function prepareTimeline(rawTimeline, pluginNames, reviveKeys) {
    const typeMap = buildTypeMap(pluginNames);
    const keys = reviveKeys || REVIVE_KEYS;
    const prepareNode = (node) => {
      const t = { ...node };
      if (typeof t.type === "string" && typeMap[t.type]) {
        t.type = typeMap[t.type];
      }
      for (const key of keys) {
        reviveCallback(t, key);
      }
      if (Array.isArray(t.timeline)) {
        t.timeline = t.timeline.map(prepareNode);
      }
      return t;
    };
    return rawTimeline.map(prepareNode);
  }

  function assertJsPsychLoaded() {
    if (typeof initJsPsych !== "function") {
      const err = document.createElement("div");
      err.style.color = "#b91c1c";
      err.style.fontWeight = "600";
      err.textContent = "Failed to load jsPsych runtime scripts.";
      document.body.appendChild(err);
      throw new Error("initJsPsych is unavailable; script CDN load failed.");
    }
  }

  /** jsPsych's ``.json()`` export (do not use ``.values()`` + JSON.stringify). */
  function collectResultsJson(jsPsych) {
    return jsPsych.data.get().json();
  }

  function renderResultsCharts(rows, config) {
    const view = window[config.results_view || "JsPsychDemoCharts"];
    if (!config.show_results_charts || !view) return;
    const task = config.results_task_filter;
    const filtered = task
      ? rows.filter((r) => r && r.task === task)
      : rows;
    view.mount(filtered);
  }

  function createJsPsych(config) {
    const jsPsych = initJsPsych({
      display_element: config.display_element || "jspsych-target",
      on_finish: () => {
        const rowsJson = collectResultsJson(jsPsych);
        const rows = JSON.parse(rowsJson);
        postResultsToParent(
          rowsJson,
          config.results_message_type || "jspsych-results",
          config.results_session_id || null,
        );
        renderResultsCharts(rows, config);
      },
    });
    // jsPsych v7: no global `jsPsych`; expose instance for eval'd trial callbacks.
    window.__jsPsychInstance = jsPsych;
    return jsPsych;
  }

  function postResultsToParent(rowsJson, messageType, sessionId) {
    window.parent.postMessage(
      { type: messageType, rows_json: rowsJson, session_id: sessionId },
      "*",
    );
  }

  function armDisplayFocus(root) {
    if (!root) return;
    root.setAttribute("tabindex", "0");
    // jsPsych focuses the display element at every trial start; that must not scroll the host page.
    const nativeFocus = root.focus;
    root.focus = (options) => nativeFocus.call(root, { ...options, preventScroll: true });
    let inputArmed = false;
    window.addEventListener("pointerdown", () => {
      inputArmed = true;
      root.focus();
    });
    return { root, getArmed: () => inputArmed, setArmed: (v) => { inputArmed = v; } };
  }

  function installArrowKeyPolicy(focusState) {
    const isArrow = (k) => k === "ArrowLeft" || k === "ArrowRight";
    const handler = (e) => {
      if (!focusState.getArmed()) return;
      if (isArrow(e.key)) {
        e.stopPropagation();
        e.preventDefault();
      }
    };
    window.addEventListener("keydown", handler, { passive: false });
    window.addEventListener("keyup", handler, { passive: false });
  }

  /**
   * While the frame cannot receive key presses (nothing clicked yet, or the participant clicked the
   * host page), dim the task with a "click here" message and pause between trials. The overlay lets
   * clicks through, so any click on the task focuses it again (armDisplayFocus).
   */
  function installFocusGuard(jsPsych, doc) {
    const d = doc || document;
    const view = d.defaultView || window;
    const guard = d.createElement("div");
    guard.className = "runner-focus-guard";
    guard.hidden = true;
    guard.innerHTML =
      '<div class="runner-focus-guard__title">Click here to play</div>' +
      '<div class="runner-focus-guard__hint">The game only hears your keys after you click on it.</div>';
    d.body.appendChild(guard);
    let active = true;
    const sync = () => {
      const away = active && !d.hasFocus();
      if (away === !guard.hidden) return;
      guard.hidden = !away;
      if (away) jsPsych.pauseExperiment();
      else jsPsych.resumeExperiment();
    };
    view.addEventListener("focus", sync);
    view.addEventListener("blur", sync);
    const timer = view.setInterval(sync, 300); // focus/blur events are not reliable across frames
    return {
      element: guard,
      sync,
      stop() {
        active = false;
        view.clearInterval(timer);
        sync();
      },
    };
  }

  function runPrepared(jsPsych, timeline) {
    return jsPsych.run(timeline);
  }

  function start(config, timelineB64) {
    assertJsPsychLoaded();
    const rawTimeline = decodeB64Json(timelineB64);
    const timeline = prepareTimeline(rawTimeline, config.plugins || [], config.revive_keys);
    const jsPsych = createJsPsych(config);
    const root = document.getElementById(config.display_element || "jspsych-target");
    const focusState = armDisplayFocus(root);
    if (config.input_arrow_keys) {
      installArrowKeyPolicy(focusState);
    }
    const finished = runPrepared(jsPsych, timeline);
    if (config.focus_guard) {
      const guard = installFocusGuard(jsPsych);
      finished.then(guard.stop);
    }
  }

  return {
    decodeB64Json,
    prepareTimeline,
    assertJsPsychLoaded,
    collectResultsJson,
    createJsPsych,
    postResultsToParent,
    renderResultsCharts,
    armDisplayFocus,
    installArrowKeyPolicy,
    installFocusGuard,
    runPrepared,
    start,
  };
})();
