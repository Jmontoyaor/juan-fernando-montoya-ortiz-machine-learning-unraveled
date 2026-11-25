/*
 * Browser inference for the regression and time-series models (no libraries).
 * Supports the Keras layers used in scratchpad_General.ipynb:
 *   Dense (relu / linear), SimpleRNN (tanh), LSTM and GRU (reset_after=true).
 * Weight layouts follow Keras exactly:
 *   Dense      [kernel (in, out), bias (out)]
 *   SimpleRNN  [kernel (f, u), recurrent (u, u), bias (u)]
 *   LSTM       [kernel (f, 4u), recurrent (u, 4u), bias (4u)]        gates i, f, c, o
 *   GRU        [kernel (f, 3u), recurrent (u, 3u), bias (2, 3u)]     gates z, r, h
 * Works in browsers (window.SignalsCore) and Node (module.exports) for tests.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.SignalsCore = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  function decodeFloat32(text) {
    var bytes;
    if (typeof Buffer !== "undefined") bytes = Uint8Array.from(Buffer.from(text, "base64"));
    else { var bin = atob(text); bytes = new Uint8Array(bin.length); for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i); }
    return new Float32Array(bytes.buffer);
  }

  function loadModel(layers) {
    return layers.map(function (l) {
      return Object.assign({}, l, { w: l.weights.map(function (w) { return { shape: w.shape, data: decodeFloat32(w.data) }; }) });
    });
  }

  var sigmoid = function (v) { return 1 / (1 + Math.exp(-v)); };

  function act(name, v) {
    if (name === "relu") return v > 0 ? v : 0;
    if (name === "tanh") return Math.tanh(v);
    if (name === "sigmoid") return sigmoid(v);
    if (name === "linear" || name == null) return v;
    throw new Error("Unsupported activation: " + name);
  }

  function dense(x, layer) {
    var k = layer.w[0], b = layer.w[1].data, nin = k.shape[0], nout = k.shape[1], y = new Float32Array(nout), i, j;
    for (j = 0; j < nout; j++) y[j] = b[j];
    for (i = 0; i < nin; i++) {
      var xi = x[i];
      if (xi === 0) continue;
      for (j = 0; j < nout; j++) y[j] += xi * k.data[i * nout + j];
    }
    for (j = 0; j < nout; j++) y[j] = act(layer.activation, y[j]);
    return y;
  }

  /* h·R for a recurrent kernel of shape (u, m). */
  function matvec(h, r, u, m, out) {
    for (var j = 0; j < m; j++) out[j] = 0;
    for (var i = 0; i < u; i++) {
      var hi = h[i];
      if (hi === 0) continue;
      for (j = 0; j < m; j++) out[j] += hi * r[i * m + j];
    }
    return out;
  }

  /* seq: Float32Array of length T with one feature per step. Returns the last hidden state. */
  function recurrent(seq, layer) {
    var u = layer.units, K = layer.w[0].data, R = layer.w[1].data, B = layer.w[2].data, T = seq.length, t, j;
    var h = new Float32Array(u), ra = layer.recurrent_activation || "sigmoid", ac = layer.activation || "tanh";
    if (layer.type === "SimpleRNN") {
      var rh = new Float32Array(u);
      for (t = 0; t < T; t++) {
        matvec(h, R, u, u, rh);
        var hn = new Float32Array(u);
        for (j = 0; j < u; j++) hn[j] = act(ac, seq[t] * K[j] + rh[j] + B[j]);
        h = hn;
      }
      return h;
    }
    if (layer.type === "LSTM") {
      var c = new Float32Array(u), z = new Float32Array(4 * u);
      for (t = 0; t < T; t++) {
        matvec(h, R, u, 4 * u, z);
        for (j = 0; j < 4 * u; j++) z[j] += seq[t] * K[j] + B[j];
        for (j = 0; j < u; j++) {
          var ig = act(ra, z[j]), fg = act(ra, z[u + j]), cc = act(ac, z[2 * u + j]), og = act(ra, z[3 * u + j]);
          c[j] = fg * c[j] + ig * cc;
          h[j] = og * act(ac, c[j]);
        }
      }
      return h;
    }
    if (layer.type === "GRU") {
      if (layer.reset_after === false) throw new Error("Only reset_after=true GRU is supported");
      var bIn = B.subarray(0, 3 * u), bRec = B.subarray(3 * u, 6 * u), hr = new Float32Array(3 * u);
      for (t = 0; t < T; t++) {
        matvec(h, R, u, 3 * u, hr);
        var next = new Float32Array(u);
        for (j = 0; j < u; j++) {
          var xz = seq[t] * K[j] + bIn[j], xr = seq[t] * K[u + j] + bIn[u + j], xh = seq[t] * K[2 * u + j] + bIn[2 * u + j];
          var zg = act(ra, xz + hr[j] + bRec[j]);
          var rg = act(ra, xr + hr[u + j] + bRec[u + j]);
          var hh = act(ac, xh + rg * (hr[2 * u + j] + bRec[2 * u + j]));
          next[j] = zg * h[j] + (1 - zg) * hh;
        }
        h = next;
      }
      return h;
    }
    throw new Error("Unsupported recurrent layer: " + layer.type);
  }

  /* Runs a Sequential stack. Recurrent layers consume the whole sequence; Dense layers a vector. */
  function predict(layers, input) {
    var x = input;
    for (var i = 0; i < layers.length; i++) {
      var l = layers[i];
      if (l.type === "Dense") x = dense(x, l);
      else if (l.type === "SimpleRNN" || l.type === "LSTM" || l.type === "GRU") x = recurrent(x, l);
      else if (l.type !== "InputLayer") throw new Error("Unsupported layer: " + l.type);
    }
    return x;
  }

  /* Autoregressive forecast: predict p samples, slide the window by p, repeat. */
  function autoregressive(layers, window, steps) {
    var w = Float32Array.from(window), out = [];
    for (var s = 0; s < steps; s++) {
      var pred = predict(layers, w), p = pred.length, nw = new Float32Array(w.length);
      nw.set(w.subarray(p));
      nw.set(pred, w.length - p);
      w = nw;
      for (var k = 0; k < p; k++) out.push(pred[k]);
    }
    return Float32Array.from(out);
  }

  return { decodeFloat32: decodeFloat32, loadModel: loadModel, predict: predict, autoregressive: autoregressive };
});
