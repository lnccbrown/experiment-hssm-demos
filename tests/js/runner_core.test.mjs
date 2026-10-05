// Node checks for runtime/jspsych_runner_core.js: nested timelines and configurable revive keys.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const repo = new URL("../../", import.meta.url);
const context = vm.createContext({ console });
context.window = context;
vm.runInContext(readFileSync(new URL("runtime/jspsych_runner_core.js", repo), "utf8"), context);
const core = context.JsPsychRunnerCore;

const raw = [
  { type: "html-button-response", stimulus: "<p>hello</p>" },
  {
    timeline: [
      {
        type: "html-keyboard-response",
        stimulus: "function(){ return 'nested'; }",
        on_finish: "function(data){ data.seen = true; }",
      },
    ],
    conditional_function: "function(){ return false; }",
  },
];

const prepared = core.prepareTimeline(raw, []);
assert.equal(prepared[0].stimulus, "<p>hello</p>");
assert.equal(typeof prepared[1].conditional_function, "function");
assert.equal(prepared[1].conditional_function(), false);
assert.equal(prepared[1].timeline[0].stimulus(), "nested");
assert.equal(typeof prepared[1].timeline[0].on_finish, "function");

const limited = core.prepareTimeline(raw, [], ["stimulus"]);
assert.equal(typeof limited[1].conditional_function, "string");
assert.equal(typeof limited[1].timeline[0].stimulus, "function");

assert.equal(typeof raw[1].conditional_function, "string", "the raw timeline must not be mutated");
assert.equal(typeof raw[1].timeline[0].stimulus, "string", "the raw timeline must not be mutated");

// result messages carry the per-run nonce used by the parent bridge
{
  const messages = [];
  context.parent = { postMessage: (payload, target) => messages.push({ payload, target }) };
  core.postResultsToParent("[]", "pst-results", "nonce-1");
  assert.deepEqual(JSON.parse(JSON.stringify(messages)), [
    { payload: { type: "pst-results", rows_json: "[]", session_id: "nonce-1" }, target: "*" },
  ]);
}

// armDisplayFocus: jsPsych's per-trial focus() must not scroll the host page
{
  const calls = [];
  const root = { setAttribute() {}, focus(options) { calls.push(options); } };
  context.addEventListener = () => {};
  core.armDisplayFocus(root);
  root.focus();
  root.focus({ focusVisible: false });
  assert.deepEqual(JSON.parse(JSON.stringify(calls)), [
    { preventScroll: true },
    { focusVisible: false, preventScroll: true },
  ]);
}

// installFocusGuard: overlay + pause while the frame cannot hear keys, resume on focus, off after the run
{
  let focused = false;
  const listeners = {};
  const view = { addEventListener: (type, fn) => { listeners[type] = fn; }, setInterval: () => 1, clearInterval() {} };
  const appended = [];
  const doc = {
    defaultView: view,
    body: { appendChild: (el) => appended.push(el) },
    createElement: () => ({ hidden: false }),
    hasFocus: () => focused,
  };
  const log = [];
  const jsPsych = { pauseExperiment: () => log.push("pause"), resumeExperiment: () => log.push("resume") };
  const guard = core.installFocusGuard(jsPsych, doc);
  assert.equal(appended.length, 1);
  assert.equal(guard.element.hidden, true);
  guard.sync();
  assert.equal(guard.element.hidden, false, "no focus yet: overlay shown");
  guard.sync();
  assert.deepEqual(log, ["pause"], "pauses once, not on every check");
  focused = true;
  listeners.focus();
  assert.equal(guard.element.hidden, true);
  assert.deepEqual(log, ["pause", "resume"]);
  focused = false;
  listeners.blur();
  assert.deepEqual(log, ["pause", "resume", "pause"]);
  guard.stop();
  assert.equal(guard.element.hidden, true, "finished run: overlay off");
  assert.deepEqual(log, ["pause", "resume", "pause", "resume"]);
  listeners.blur();
  assert.equal(guard.element.hidden, true, "stays off after the run");
}

console.log("jspsych_runner_core.js: all checks passed");
