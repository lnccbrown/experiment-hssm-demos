/** PST browser helpers: scoring, points, feedback, adaptive blocks and the end-of-game view. */
(function (global) {
  "use strict";

  const state = { stopLearning: false, pendingMethod: null };

  function pstRows(phase) {
    const j = global.__jsPsychInstance;
    if (!j) return [];
    return j.data
      .get()
      .values()
      .filter((r) => r && r.task === "pst" && (!phase || r.phase === phase));
  }

  function points(rows) {
    return rows.reduce((sum, r) => sum + (r.feedback === 1 ? 1 : 0), 0);
  }

  function choseBetter(r) {
    return !r.timed_out && !r.anticipated && r.chosen_symbol !== null &&
      r.chosen_symbol !== undefined && r.chosen_symbol === r.better_symbol;
  }

  /** on_finish of every choice trial: side, symbol, and the pre-drawn outcome of that choice. */
  function scoreTrial(data) {
    const key = data.response == null ? "" : String(data.response).toLowerCase();
    const side = key === "arrowleft" ? "left" : key === "arrowright" ? "right" : null;
    data.choice_side = side;
    data.timed_out = side === null;
    data.response_method = side === null ? null : state.pendingMethod || "key";
    const minimumRt = Number(data.minimum_rt_ms == null ? 200 : data.minimum_rt_ms);
    data.anticipated = side !== null && Number.isFinite(Number(data.rt)) && Number(data.rt) < minimumRt;
    if (side === null) {
      data.chosen_symbol = null;
      data.feedback = null;
      data.correct = false;
      return data;
    }
    data.chosen_symbol = side === "left" ? data.left_symbol : data.right_symbol;
    data.correct = data.chosen_symbol === data.better_symbol;
    if (data.anticipated) {
      // Do not show an outcome or update learning for trials excluded as anticipations.
      data.feedback = null;
      data.correct = false;
      return data;
    }
    const reward = side === "left" ? data.reward_left : data.reward_right;
    data.feedback = reward === null || reward === undefined ? null : Number(reward);
    return data;
  }

  /** on_load of choice screens: fill in the running score (learning phase only). */
  function updatePoints() {
    const el = global.document && global.document.querySelector(".pst-status__points");
    if (el) el.textContent = `Points: ${points(pstRows("learning"))}`;
  }

  /**
   * on_load of choice screens: pressing a card counts as pressing that side's arrow key, so clicks
   * share jsPsych's RT clock and scoring. Rows record response_method "click" or "key".
   */
  function armCards() {
    const doc = global.document;
    const root = doc && doc.getElementById("jspsych-target");
    if (!root) return;
    for (const card of root.querySelectorAll(".pst-card[data-side]")) {
      card.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return;
        const key = card.dataset.side === "left" ? "ArrowLeft" : "ArrowRight";
        state.pendingMethod = "click";
        try {
          // keyup too: jsPsych ignores a key it still considers held down
          root.dispatchEvent(new global.KeyboardEvent("keydown", { key, bubbles: true }));
          root.dispatchEvent(new global.KeyboardEvent("keyup", { key, bubbles: true }));
        } finally {
          state.pendingMethod = null;
        }
      });
    }
  }

  function feedbackHTML() {
    const rows = pstRows();
    const last = rows[rows.length - 1] || {};
    const learning = last.phase === "learning";
    const score = learning ? `<span class="pst-status__points">Points: ${points(pstRows("learning"))}</span>` : "";
    let body;
    if (last.timed_out) {
      body = '<div class="pst-feedback pst-feedback--slow">Too slow! Try to answer a bit faster.</div>';
    } else if (last.anticipated) {
      body = '<div class="pst-feedback pst-feedback--slow">Too fast! Wait until the symbols appear.</div>';
    } else if (last.feedback === 1) {
      const delta = learning ? ' <span class="pst-feedback__delta">+1</span>' : "";
      body = `<div class="pst-feedback pst-feedback--win">Correct!${delta}</div>`;
    } else {
      body = '<div class="pst-feedback pst-feedback--loss">Incorrect</div>';
    }
    return (
      `<div class="pst-screen"><div class="pst-status"><span class="pst-status__label"></span>${score}</div>${body}` +
      '<div class="pst-keys pst-keys--placeholder" aria-hidden="true"><span><kbd>&larr;</kbd></span></div></div>'
    );
  }

  function blockSummaryHTML(block) {
    const learning = pstRows("learning");
    const inBlock = learning.filter((r) => r.block === block);
    return (
      '<div class="pst-panel">' +
      `<h2>Block ${block} done</h2>` +
      `<p>You earned <strong>${points(inBlock)}</strong> points in this block ` +
      `(<strong>${points(learning)}</strong> in total).</p>` +
      '<p class="pst-muted">Take a short breather, then continue when you are ready.</p>' +
      "</div>"
    );
  }

  /** Proportion of better-symbol choices per pair; timeouts count as incorrect. */
  function blockAccuracy(rows) {
    const byPair = {};
    for (const r of rows) {
      const p = byPair[r.pair] || (byPair[r.pair] = { n: 0, better: 0 });
      p.n += 1;
      if (choseBetter(r)) p.better += 1;
    }
    const out = {};
    for (const [pair, p] of Object.entries(byPair)) out[pair] = p.better / p.n;
    return out;
  }

  /** conditional_function of blocks after the minimum: stop once a block met every criterion. */
  function shouldRunBlock(previousBlock, criterion) {
    if (state.stopLearning) return false;
    const accuracy = blockAccuracy(pstRows("learning").filter((r) => r.block === previousBlock));
    const met = Object.entries(criterion).every(([pair, level]) => (accuracy[pair] || 0) >= level);
    if (met) state.stopLearning = true;
    return !met;
  }

  function testCategory(pair) {
    const hasA = pair.includes("A");
    const hasB = pair.includes("B");
    if (hasA && !hasB) return "choose_A";
    if (hasB && !hasA) return "avoid_B";
    return null;
  }

  /** Numbers for the end screen: points, last-block accuracy per pair, choose-A / avoid-B. */
  function summarize(rows) {
    const learning = rows.filter((r) => r.phase === "learning");
    const lastBlock = learning.reduce((m, r) => Math.max(m, r.block), 0);
    const labels = {};
    for (const r of learning) labels[r.pair] = r.pair_label || r.pair;
    const test = rows.filter((r) => r.phase === "test" && !r.timed_out && !r.anticipated);
    const rate = (rs) => (rs.length ? rs.filter(choseBetter).length / rs.length : null);
    return {
      points: points(learning),
      blocks: lastBlock,
      labels,
      lastBlockAccuracy: blockAccuracy(learning.filter((r) => r.block === lastBlock)),
      chooseA: rate(test.filter((r) => testCategory(r.pair) === "choose_A")),
      avoidB: rate(test.filter((r) => testCategory(r.pair) === "avoid_B")),
    };
  }

  function mount(rows) {
    const root = global.document && global.document.getElementById("jspsych-target");
    if (!root) return;
    const s = summarize(rows);
    const pct = (x) => (x === null || x === undefined ? "–" : `${Math.round(100 * x)}%`);
    const tiles = Object.keys(s.lastBlockAccuracy)
      .sort()
      .map(
        (pair) =>
          `<div class="pst-tile"><div class="pst-tile__value">${pct(s.lastBlockAccuracy[pair])}</div>` +
          `<div class="pst-tile__label">${s.labels[pair]}</div></div>`
      )
      .join("");
    const test =
      s.chooseA === null
        ? ""
        : `<p>Final round: you picked the best symbol ${pct(s.chooseA)} of the time when it was on screen, ` +
          `and avoided the worst one ${pct(s.avoidB)} of the time.</p>`;
    root.innerHTML =
      '<div class="pst-results-stage"><div class="pst-panel pst-results">' +
      `<h2>All done: ${s.points} points!</h2>` +
      `<p>How often you chose the better symbol in your last block (block ${s.blocks}):</p>` +
      `<div class="pst-results__tiles">${tiles}</div>` +
      test +
      '<p class="pst-muted">Scroll down: the notebook now shows your learning curves and what a ' +
      "learning model says about how you learned.</p>" +
      "</div></div>";
  }

  global.PSTSession = {
    scoreTrial,
    updatePoints,
    armCards,
    feedbackHTML,
    blockSummaryHTML,
    blockAccuracy,
    shouldRunBlock,
    summarize,
  };
  global.PSTResults = { mount, summarize };
})(typeof window !== "undefined" ? window : globalThis);
