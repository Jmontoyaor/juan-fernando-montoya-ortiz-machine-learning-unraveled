"""Fair comparison of recurrent networks on a synthetic time series (scratchpad_General).

All models see the same windows, use 64 units, the same optimizer, epochs and callbacks.
RNN, LSTM and GRU are the compared networks; the DNN is the reference baseline.
"""
import json
import time

import numpy as np
import tensorflow as tf

from mnist_app.config import ARTIFACTS_DIR

SEED = 42
N = 40_000
WINDOW = 1000          # w: past samples the model sees
HORIZON = 200          # p: future samples predicted at once
AUTO_STEPS = 10        # autoregressive steps -> 2000 samples
STRIDE = 50            # step between consecutive windows
UNITS = 64
EPOCHS = 100
BATCH = 32
LEARNING_RATE = 1e-3
KINDS = ("DNN", "RNN", "LSTM", "GRU")
OUT_DIR = ARTIFACTS_DIR / "timeseries"


def make_series(seed=SEED):
    """Synthetic sensor signal: level + trend + slow and fast cycles + noise."""
    rng = np.random.RandomState(seed)
    t = np.arange(N)
    series = (20 + 0.000025 * t + 0.6 * np.sin(2 * np.pi * t / 2400)
              + 0.15 * np.sin(2 * np.pi * t / 450) + 0.025 * rng.randn(N))
    return series.astype(np.float32)


def split_and_normalize(series):
    """Chronological 70/15/15 split; mean and std come from training only."""
    a, b = int(N * 0.70), int(N * 0.85)
    train, val, test = series[:a], series[a:b], series[b:]
    mean, std = float(train.mean()), float(train.std())
    norm = lambda s: ((s - mean) / std).astype(np.float32)
    return norm(train), norm(val), norm(test), mean, std, (a, b)


def make_windows(signal, window=WINDOW, horizon=HORIZON, stride=STRIDE):
    x, y = [], []
    for i in range(0, len(signal) - window - horizon + 1, stride):
        x.append(signal[i:i + window])
        y.append(signal[i + window:i + window + horizon])
    return np.asarray(x, np.float32), np.asarray(y, np.float32)


def build(kind):
    recurrent = {"RNN": tf.keras.layers.SimpleRNN, "LSTM": tf.keras.layers.LSTM, "GRU": tf.keras.layers.GRU}
    if kind == "DNN":
        layers = [tf.keras.layers.Input(shape=(WINDOW,), name="window"),
                  tf.keras.layers.Dense(UNITS, activation="relu", name="hidden")]
    else:
        layers = [tf.keras.layers.Input(shape=(WINDOW, 1), name="window"),
                  recurrent[kind](UNITS, name="recurrent")]
    layers.append(tf.keras.layers.Dense(HORIZON, name="forecast"))
    model = tf.keras.Sequential(layers, name=kind)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse", metrics=["mae"])
    return model


def shape_for(kind, x):
    return x if kind == "DNN" else x[..., None]


def predict(model, kind, window):
    return np.asarray(model(shape_for(kind, np.asarray(window, np.float32).reshape(1, WINDOW)), training=False))[0]


def autoregressive(model, kind, window, steps=AUTO_STEPS):
    window, out = np.asarray(window, np.float32).copy(), []
    for _ in range(steps):
        pred = predict(model, kind, window)
        out.append(pred)
        window = np.concatenate([window[HORIZON:], pred])  # drop the oldest p samples, append the prediction
    return np.concatenate(out)


def train_all(epochs=EPOCHS):
    train, val, test, mean, std, limits = split_and_normalize(make_series())
    (xt, yt), (xv, yv), (xs, ys) = (make_windows(s) for s in (train, val, test))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start_window, future = test[:WINDOW], test[WINDOW:WINDOW + HORIZON * AUTO_STEPS]
    summary = {"config": {"n": N, "window": WINDOW, "horizon": HORIZON, "auto_steps": AUTO_STEPS,
                          "stride": STRIDE, "units": UNITS, "epochs_max": epochs, "batch_size": BATCH,
                          "learning_rate": LEARNING_RATE, "seed": SEED, "split": [0.70, 0.15, 0.15],
                          "windows": {"train": len(xt), "val": len(xv), "test": len(xs)}},
               "mean": mean, "std": std, "limits": list(limits), "models": {}}
    for kind in KINDS:
        tf.keras.utils.set_random_seed(SEED)  # same initial seed for every model
        model = build(kind)
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-5)]
        t0 = time.perf_counter()
        hist = model.fit(shape_for(kind, xt), yt, validation_data=(shape_for(kind, xv), yv), epochs=epochs,
                         batch_size=BATCH, callbacks=callbacks, shuffle=False, verbose=2)
        seconds = time.perf_counter() - t0
        test_mse, test_mae = model.evaluate(shape_for(kind, xs), ys, verbose=0)
        direct = predict(model, kind, start_window)
        auto = autoregressive(model, kind, start_window)
        denorm = lambda v: np.asarray(v) * std + mean
        real_d, real_a = denorm(future[:HORIZON]), denorm(future)
        summary["models"][kind] = {
            "parameters": int(model.count_params()),
            "epochs_run": len(hist.history["loss"]),
            "training_seconds": seconds,
            "history": {k: [float(v) for v in vals] for k, vals in hist.history.items() if k in ("loss", "val_loss")},
            "test_windows_mse_norm": float(test_mse), "test_windows_mae_norm": float(test_mae),
            "direct_mse": float(np.mean((real_d - denorm(direct)) ** 2)),
            "direct_mae": float(np.mean(np.abs(real_d - denorm(direct)))),
            "auto_mse": float(np.mean((real_a - denorm(auto)) ** 2)),
            "auto_mae": float(np.mean(np.abs(real_a - denorm(auto)))),
        }
        model.save(OUT_DIR / f"{kind.lower()}.keras")
        print(f"{kind}: params={model.count_params()} epochs={len(hist.history['loss'])} "
              f"test_mse={test_mse:.5f} auto_mse={summary['models'][kind]['auto_mse']:.5f} ({seconds:.0f}s)", flush=True)
    (OUT_DIR / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
