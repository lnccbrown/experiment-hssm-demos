// Node checks for renderers/pst_symbols/pst_session.js against a fake jsPsych data store.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const repo = new URL("../../", import.meta.url);

function sandbox(rows = []) {
  const context = vm.createContext({ console });
  context.window = context;
  context.__jsPsychInstance = { data: { get: () => ({ values: () => rows }) } };
  vm.runInContext(readFileSync(new URL("renderers/pst_symbols/pst_session.js", repo), "utf8"), context);
  return context;
}

const choice = (overrides = {}) => ({
  task: "pst", phase: "learning", block: 1, pair: "AB", pair_label: "80/20 pair",
  left_symbol: 0, right_symbol: 1, better_symbol: 0, reward_left: 1, reward_right: 0,
  ...overrides,
});

// scoreTrial: side, symbol, pre-drawn outcome, timeouts, and no feedback in the test phase
{
  const { PSTSession } = sandbox();
  const left = PSTSession.scoreTrial(choice({ response: "arrowleft" }));
  assert.equal(left.choice_side, "left");
  assert.equal(left.chosen_symbol, 0);
  assert.equal(left.feedback, 1);
  assert.equal(left.correct, true);
  const right = PSTSession.scoreTrial(choice({ response: "ArrowRight" }));
  assert.equal(right.chosen_symbol, 1);
  assert.equal(right.feedback, 0);
  assert.equal(right.correct, false);
  const slow = PSTSession.scoreTrial(choice({ response: null }));
  assert.equal(slow.timed_out, true);
  assert.equal(slow.feedback, null);
  const test = PSTSession.scoreTrial(choice({ phase: "test", reward_left: null, reward_right: null, response: "arrowleft" }));
  assert.equal(test.feedback, null);
  const anticipated = PSTSession.scoreTrial(choice({ response: "arrowleft", rt: 150, minimum_rt_ms: 200 }));
  assert.equal(anticipated.anticipated, true);
  assert.equal(anticipated.feedback, null, "excluded anticipations must not deliver a learning outcome");
}

// armCards: pressing a card is the same as pressing that side's arrow key, recorded as a click
{
  const rows = [];
  const context = sandbox(rows);
  const { PSTSession } = context;
  const handlers = {};
  const cards = ["left", "right"].map((side) => ({
    dataset: { side },
    addEventListener: (type, fn) => { handlers[side] = fn; },
  }));
  const events = [];
  const root = {
    querySelectorAll: () => cards,
    dispatchEvent: (e) => {
      events.push(`${e.type}:${e.key}`);
      if (e.type === "keydown") rows.push(PSTSession.scoreTrial(choice({ response: e.key })));
    },
  };
  context.document = { getElementById: () => root };
  context.KeyboardEvent = class {
    constructor(type, init) {
      this.type = type;
      Object.assign(this, init);
    }
  };
  PSTSession.armCards();
  handlers.right({ button: 0 });
  assert.deepEqual(events, ["keydown:ArrowRight", "keyup:ArrowRight"], "keyup too, or jsPsych treats the key as held");
  assert.equal(rows[0].choice_side, "right");
  assert.equal(rows[0].chosen_symbol, 1);
  assert.equal(rows[0].response_method, "click");
  handlers.left({ button: 2 });
  assert.equal(events.length, 2, "other mouse buttons are ignored");
  assert.equal(PSTSession.scoreTrial(choice({ response: "arrowleft" })).response_method, "key");
  assert.equal(PSTSession.scoreTrial(choice({ response: null })).response_method, null);
}

// feedback screen follows the last scored trial and shows the running score
{
  const rows = [];
  const { PSTSession } = sandbox(rows);
  rows.push(PSTSession.scoreTrial(choice({ response: "arrowleft" })));
  assert.match(PSTSession.feedbackHTML(), /Correct!/);
  assert.match(PSTSession.feedbackHTML(), /Points: 1/);
  rows.push(PSTSession.scoreTrial(choice({ response: "arrowright" })));
  assert.match(PSTSession.feedbackHTML(), /Incorrect/);
  rows.push(PSTSession.scoreTrial(choice({ response: null })));
  assert.match(PSTSession.feedbackHTML(), /Too slow/);
  rows.push(PSTSession.scoreTrial(choice({ response: "arrowleft", rt: 100 })));
  assert.match(PSTSession.feedbackHTML(), /Too fast/);
  assert.match(PSTSession.blockSummaryHTML(1), /<strong>1<\/strong> points in this block/);
}

// adaptive length: keep going until a block meets every pair's criterion, then stay stopped
{
  const rows = [];
  const { PSTSession } = sandbox(rows);
  const add = (block, pair, better, n) => {
    for (let i = 0; i < n; i += 1) rows.push({ ...choice({ block, pair }), chosen_symbol: i < better ? 0 : 1 });
  };
  const criterion = { AB: 0.65, CD: 0.6, EF: 0.5 };
  add(2, "AB", 7, 10); add(2, "CD", 6, 10); add(2, "EF", 4, 10);
  assert.equal(PSTSession.shouldRunBlock(2, criterion), true);
  add(3, "AB", 7, 10); add(3, "CD", 6, 10); add(3, "EF", 5, 10);
  assert.equal(PSTSession.shouldRunBlock(3, criterion), false);
  assert.equal(PSTSession.shouldRunBlock(2, criterion), false);
}

// end-screen numbers
{
  const rows = [
    { ...choice({ block: 1 }), feedback: 1, chosen_symbol: 0 },
    { ...choice({ block: 2 }), feedback: 1, chosen_symbol: 0 },
    { ...choice({ block: 2 }), feedback: 0, chosen_symbol: 1 },
    { task: "pst", phase: "test", pair: "AC", better_symbol: 0, chosen_symbol: 0, timed_out: false },
    { task: "pst", phase: "test", pair: "BD", better_symbol: 3, chosen_symbol: 1, timed_out: false },
    { task: "pst", phase: "test", pair: "AC", better_symbol: 0, chosen_symbol: 1, timed_out: false, anticipated: true },
  ];
  const { PSTResults } = sandbox(rows);
  const summary = PSTResults.summarize(rows);
  assert.equal(summary.points, 2);
  assert.equal(summary.blocks, 2);
  assert.deepEqual({ ...summary.lastBlockAccuracy }, { AB: 0.5 });
  assert.equal(summary.chooseA, 1);
  assert.equal(summary.avoidB, 0);
}

console.log("pst_session.js: all checks passed");
