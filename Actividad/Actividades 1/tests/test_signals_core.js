// Run: node "Actividad/Actividades 1/tests/test_signals_core.js"  (after python export_web.py)
"use strict";
const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const docs = path.resolve(__dirname, "..", "..", "..", "docs", "mnist");
const core = require(path.join(docs, "signals_core.js"));
const sandbox = { window: {} };
vm.runInNewContext(fs.readFileSync(path.join(docs, "signals_data.js"), "utf8"), sandbox);
const data = sandbox.window.SIGNALS_DATA;
let passed = 0;

for (const [task, r] of Object.entries(data.regression)) {
  const layers = core.loadModel(r.layers);
  let maxDiff = 0;
  r.verify_x.forEach((x, i) => { maxDiff = Math.max(maxDiff, Math.abs(core.predict(layers, Float32Array.of(x))[0] - r.verify_y[i])); });
  assert(maxDiff < 1e-4, `${task}: max |JS - Keras| = ${maxDiff}`);
  console.log(`ok - regression ${task} matches Keras (max diff ${maxDiff.toExponential(2)})`);
  passed++;
}

const ts = data.timeseries;
if (ts) {
  const test = core.decodeFloat32(ts.test_series);
  const w = ts.config.window, window = new Float32Array(w);
  for (let i = 0; i < w; i++) window[i] = (test[i] - ts.mean) / ts.std;
  for (const [kind, m] of Object.entries(ts.models)) {
    const pred = core.predict(core.loadModel(m.layers), window);
    let maxDiff = 0;
    pred.forEach((v, i) => { maxDiff = Math.max(maxDiff, Math.abs(v - m.verify_direct[i])); });
    assert.strictEqual(pred.length, ts.config.horizon);
    assert(maxDiff < 1e-3, `${kind}: max |JS - Keras| = ${maxDiff}`);
    console.log(`ok - time series ${kind} matches Keras (max diff ${maxDiff.toExponential(2)})`);
    passed++;
  }
  const auto = core.autoregressive(core.loadModel(ts.models.DNN.layers), window, ts.config.auto_steps);
  assert.strictEqual(auto.length, ts.config.horizon * ts.config.auto_steps);
  console.log("ok - autoregressive forecast has horizon x steps samples");
  passed++;
} else {
  console.log("skip - time-series models not exported yet");
}
console.log(`${passed} tests passed`);
