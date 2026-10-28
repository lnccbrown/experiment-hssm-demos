/** Browser-side PAAT scoring, wheel animation, outcomes, and end screen. */
(function (global) {
  "use strict";

  const state = { pendingMethod: null };

  function paatRows() {
    const j = global.__jsPsychInstance;
    if (!j) return [];
    return j.data.get().values().filter((r) => r && r.task === "paat");
  }

  function sideFromResponse(response) {
    const key = response == null ? "" : String(response).toLowerCase();
    if (key === "arrowleft") return "left";
    if (key === "arrowright") return "right";
    return null;
  }

  function scoreTrial(data) {
    const side = sideFromResponse(data.response);
    data.choice_side = side;
    data.timed_out = side === null;
    data.response_method = side === null ? null : state.pendingMethod || "key";
    if (side === null) {
      data.chosen_option = null;
      data.response = null;
      data.selected_wheel = "aversive";
      data.spin_value = null;
      data.outcome = 1;
      data.reward_value = 0;
      data.aversive_outcome = true;
      return data;
    }

    const option = side === data.risky_side ? "risky" : "safe";
    const wheel = data.selected_wheel;
    const spin = Number(option === "risky" ? data.spin_risky : data.spin_safe);
    const probability = Number(data[`p_${wheel}_${option}`]);
    const outcome = spin < probability ? 1 : 0;
    data.chosen_option = option;
    data.response = option === "risky" ? 1 : -1;
    data.spin_value = spin;
    data.outcome = outcome;
    data.reward_value = wheel === "reward" && outcome ? Number(data.reward_amount) : 0;
    data.aversive_outcome = wheel === "aversive" && outcome === 1;
    return data;
  }

  function armOptions() {
    const root = global.document && global.document.getElementById("jspsych-target");
    if (!root) return;
    for (const option of root.querySelectorAll(".paat-option[data-side]")) {
      const choose = () => {
        const key = option.dataset.side === "left" ? "ArrowLeft" : "ArrowRight";
        state.pendingMethod = "click";
        try {
          root.dispatchEvent(new global.KeyboardEvent("keydown", { key, bubbles: true }));
          root.dispatchEvent(new global.KeyboardEvent("keyup", { key, bubbles: true }));
        } finally {
          state.pendingMethod = null;
        }
      };
      option.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return;
        choose();
      });
      option.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") choose();
      });
    }
  }

  function totalReward(rows) {
    return rows.reduce((sum, row) => sum + Number(row.reward_value || 0), 0);
  }

  function formatReward(value, unit) {
    return unit === "dollars" ? `$${Number(value).toFixed(2)}` : `${Math.round(Number(value))} points`;
  }

  function updateReward() {
    const el = global.document && global.document.querySelector(".paat-status__reward");
    const rows = paatRows();
    if (el && rows.length) el.textContent = `Reward: ${formatReward(totalReward(rows), rows[0].reward_unit)}`;
  }

  function lastChoice() {
    const rows = paatRows();
    return rows[rows.length - 1] || {};
  }

  function wheelMarkup(row) {
    const wheel = row.selected_wheel || "aversive";
    const option = row.chosen_option || "safe";
    const probability = row.timed_out ? 1 : Number(row[`p_${wheel}_${option}`]);
    const pct = Math.round(100 * probability);
    const label = wheel === "reward" ? "Reward" : "Aversive";
    const spin = row.spin_value == null ? 0 : Number(row.spin_value);
    const duration = Math.max(1, Number(row.spin_duration_ms || 4000));
    return (
      `<div class="paat-spin-wrap" style="--spin-angle:${360 * spin}deg;--spin-duration:${duration}ms">` +
      `<div class="paat-wheel paat-wheel--${wheel}" style="--paat-arc:${360 * probability}deg">` +
      '<div class="paat-wheel__disc"></div>' +
      `<div class="paat-wheel__number">${pct}%</div></div></div>` +
      `<div class="paat-wheel__label">${label} wheel</div>`
    );
  }

  function spinHTML() {
    const row = lastChoice();
    if (row.timed_out) {
      return '<div class="paat-spin-stage"><div class="paat-spin-card"><strong>Response deadline passed</strong>' +
        '<p class="paat-muted">Aversive outcome shown by the task rule.</p></div></div>';
    }
    return '<div class="paat-spin-stage"><div class="paat-spin-card"><div class="paat-muted">One wheel was selected</div>' +
      wheelMarkup(row) + "</div></div>";
  }

  function outcomeHTML() {
    const row = lastChoice();
    let kind;
    let icon;
    let title;
    let detail = "";
    if (row.timed_out || row.aversive_outcome) {
      kind = "aversive";
      icon = "&#9888;";
      title = "Aversive outcome";
      detail = "Public demo placeholder — no disturbing image is bundled.";
    } else if (row.selected_wheel === "aversive") {
      kind = "neutral";
      icon = "&#9675;";
      title = "Neutral outcome";
    } else if (row.outcome === 1) {
      kind = "reward";
      icon = "+";
      title = formatReward(row.reward_amount, row.reward_unit);
    } else {
      kind = "none";
      icon = "0";
      title = "No reward";
    }
    return `<div class="paat-outcome-stage"><div class="paat-outcome paat-outcome--${kind}">` +
      `<div class="paat-outcome__icon">${icon}</div><div class="paat-outcome__title">${title}</div>` +
      (detail ? `<div class="paat-muted">${detail}</div>` : "") + "</div></div>";
  }

  function summarize(rows) {
    const answered = rows.filter((row) => !row.timed_out);
    const risky = answered.filter((row) => row.response === 1).length;
    const rts = answered.map((row) => Number(row.rt)).filter(Number.isFinite).sort((a, b) => a - b);
    const middle = Math.floor(rts.length / 2);
    const median = !rts.length
      ? null
      : rts.length % 2
        ? rts[middle]
        : (rts[middle - 1] + rts[middle]) / 2;
    return {
      trials: rows.length,
      riskyRate: answered.length ? risky / answered.length : null,
      medianRtMs: median,
      reward: totalReward(rows),
      rewardUnit: rows.length ? rows[0].reward_unit : "points",
    };
  }

  function mount(rows) {
    const root = global.document && global.document.getElementById("jspsych-target");
    if (!root) return;
    const taskRows = rows.filter((row) => row && row.task === "paat");
    const summary = summarize(taskRows);
    const pct = summary.riskyRate == null ? "–" : `${Math.round(100 * summary.riskyRate)}%`;
    const rt = summary.medianRtMs == null ? "–" : `${(summary.medianRtMs / 1000).toFixed(2)} s`;
    root.innerHTML = '<div class="paat-results-stage"><div class="paat-panel">' +
      '<h2>Session complete</h2><div class="paat-results-grid">' +
      `<div class="paat-result"><strong>${pct}</strong>risky choices</div>` +
      `<div class="paat-result"><strong>${rt}</strong>median response</div>` +
      `<div class="paat-result"><strong>${formatReward(summary.reward, summary.rewardUnit)}</strong>reward</div>` +
      '</div><p class="paat-muted">Scroll down to explore the choices and connect them to the angle model.</p>' +
      "</div></div>";
  }

  global.PAATSession = { scoreTrial, armOptions, updateReward, spinHTML, outcomeHTML, summarize };
  global.PAATResults = { mount, summarize };
})(typeof window !== "undefined" ? window : globalThis);
