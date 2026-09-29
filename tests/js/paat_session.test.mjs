import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const repo = new URL("../../", import.meta.url);

function sandbox(rows = []) {
  const context = vm.createContext({ console });
  context.window = context;
  context.__jsPsychInstance = { data: { get: () => ({ values: () => rows }) } };
  vm.runInContext(readFileSync(new URL("renderers/paat_wheels/paat_session.js", repo), "utf8"), context);
  return context;
}

const choice = (overrides = {}) => ({
  task: "paat",
  profile: "conference",
  block: 1,
  trial: 1,
  risky_side: "left",
  selected_wheel: "reward",
  p_reward_risky: 0.7,
  p_reward_safe: 0.3,
  p_aversive_risky: 0.8,
  p_aversive_safe: 0.2,
  spin_risky: 0.5,
  spin_safe: 0.5,
  reward_amount: 10,
  reward_unit: "points",
  ...overrides,
});

// Boundary coding and scheduled outcomes.
{
  const { PAATSession } = sandbox();
  const risky = PAATSession.scoreTrial(choice({ response: "arrowleft" }));
  assert.equal(risky.chosen_option, "risky");
  assert.equal(risky.response, 1);
  assert.equal(risky.outcome, 1);
  assert.equal(risky.reward_value, 10);

  const safe = PAATSession.scoreTrial(choice({ response: "ArrowRight" }));
  assert.equal(safe.chosen_option, "safe");
  assert.equal(safe.response, -1);
  assert.equal(safe.outcome, 0);
  assert.equal(safe.reward_value, 0);

  const timeout = PAATSession.scoreTrial(choice({ response: null }));
  assert.equal(timeout.timed_out, true);
  assert.equal(timeout.selected_wheel, "aversive");
  assert.equal(timeout.aversive_outcome, true);
}

// Clicking an option dispatches the same key event while preserving input method.
{
  const rows = [];
  const context = sandbox(rows);
  const handlers = {};
  const options = ["left", "right"].map((side) => ({
    dataset: { side },
    addEventListener: (type, fn) => { handlers[`${side}:${type}`] = fn; },
  }));
  const root = {
    querySelectorAll: () => options,
    dispatchEvent: (event) => {
      if (event.type === "keydown") rows.push(context.PAATSession.scoreTrial(choice({ response: event.key })));
    },
  };
  context.document = { getElementById: () => root };
  context.KeyboardEvent = class {
    constructor(type, init) { this.type = type; Object.assign(this, init); }
  };
  context.PAATSession.armOptions();
  handlers["right:pointerdown"]({ button: 0 });
  assert.equal(rows[0].choice_side, "right");
  assert.equal(rows[0].response_method, "click");
}

// End summary uses only task rows and reports risky choices, RT, and reward.
{
  const rows = [];
  const { PAATSession } = sandbox(rows);
  rows.push(PAATSession.scoreTrial(choice({ response: "arrowleft", rt: 900 })));
  rows.push(PAATSession.scoreTrial(choice({ response: "arrowright", rt: 700 })));
  const summary = PAATSession.summarize(rows);
  assert.equal(summary.trials, 2);
  assert.equal(summary.riskyRate, 0.5);
  assert.equal(summary.medianRtMs, 800);
  assert.equal(summary.reward, 10);
  assert.match(PAATSession.spinHTML(), /Reward wheel/);
  assert.match(PAATSession.outcomeHTML(), /No reward/);
}

console.log("paat_session.js: all checks passed");
