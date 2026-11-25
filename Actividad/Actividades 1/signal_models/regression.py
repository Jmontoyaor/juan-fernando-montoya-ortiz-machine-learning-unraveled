"""Regression examples from scratchpad_General: a straight line and a cosine."""
import json

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from mnist_app.config import ARTIFACTS_DIR

SEED = 42
OUT_DIR = ARTIFACTS_DIR / "regression"

TASKS = {
    # y = 3x + 1 on [-10, 10) with a single neuron: Dense(1)
    "line": {"x": (-10.0, 10.0, 0.01), "hidden": [], "epochs": 100, "extrapolate": (-15.0, 15.0)},
    # y = cos(2*pi*2*x) on [-2, 2) with Dense(64) x3 + Dense(1)
    "cosine": {"x": (-2.0, 2.0, 1 / 1000), "hidden": [64, 64, 64], "epochs": 300, "extrapolate": (-3.0, 3.0)},
}


def target(task, x):
    return 3 * x + 1 if task == "line" else np.cos(2 * np.pi * 2 * x)


def build(task):
    layers = [tf.keras.layers.Input(shape=(1,), name="x")]
    layers += [tf.keras.layers.Dense(u, activation="relu", name=f"hidden_{i + 1}") for i, u in enumerate(TASKS[task]["hidden"])]
    layers.append(tf.keras.layers.Dense(1, name="y"))
    model = tf.keras.Sequential(layers, name=f"regression_{task}")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train(task):
    lo, hi, step = TASKS[task]["x"]
    x = np.arange(lo, hi, step).reshape(-1, 1).astype(np.float32)
    y = target(task, x).astype(np.float32)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=SEED)
    tf.keras.utils.set_random_seed(SEED)
    model = build(task)
    hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=TASKS[task]["epochs"],
                     batch_size=32, verbose=0)
    mse, mae = model.evaluate(x_test, y_test, verbose=0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(OUT_DIR / f"{task}.keras")
    result = {
        "task": task, "epochs": TASKS[task]["epochs"], "batch_size": 32, "seed": SEED,
        "train_samples": len(x_train), "test_samples": len(x_test), "parameters": int(model.count_params()),
        "test_mse": float(mse), "test_mae": float(mae),
        "history": {k: [float(v) for v in vals] for k, vals in hist.history.items() if k in ("loss", "val_loss")},
        "test_points": {"x": x_test[:, 0].tolist(), "y": y_test[:, 0].tolist()},
        "x_range": [lo, hi], "extrapolate": list(TASKS[task]["extrapolate"]),
    }
    if task == "line":
        w, b = model.get_layer("y").get_weights()
        result["learned"] = {"m": float(w[0, 0]), "b": float(b[0])}
    (OUT_DIR / f"{task}_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{task}: test_mse={mse:.6f} test_mae={mae:.6f} params={model.count_params()}"
          + (f" learned m={result['learned']['m']:.4f} b={result['learned']['b']:.4f}" if task == "line" else ""), flush=True)
    return result
