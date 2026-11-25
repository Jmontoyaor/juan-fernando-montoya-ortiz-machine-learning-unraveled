"""Export trained models and saved results for the static web page (GitHub Pages).

Writes docs/mnist/model_data.js, a plain script that defines window.MNIST_DATA so the
page works both from GitHub Pages and when index.html is opened directly from disk.

Usage:  python export_web.py
"""
import base64
import json

import numpy as np
import tensorflow as tf

from mnist_app.config import ACTIVITY_DIR, ARTIFACTS_DIR, MODEL_FILES
from mnist_app.data import for_model, load_mnist

OUTPUT = ACTIVITY_DIR.parents[1] / "docs" / "mnist" / "model_data.js"
VERIFY_COUNT = 32


def b64(array, dtype):
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


def export_model(kind, test_images):
    model = tf.keras.models.load_model(ARTIFACTS_DIR / MODEL_FILES[kind])
    results = json.loads((ARTIFACTS_DIR / f"{kind}_results.json").read_text(encoding="utf-8"))
    layers = []
    for layer in model.layers:
        weights = layer.get_weights()
        entry = {"name": layer.name, "type": type(layer).__name__,
                 "output_shape": list(layer.output.shape[1:]), "params": layer.count_params(),
                 "activation": layer.get_config().get("activation")}
        config = layer.get_config()
        if "padding" in config:
            entry["padding"] = config["padding"]
        if "pool_size" in config:
            entry["pool_size"] = list(config["pool_size"])
            entry["strides"] = list(config.get("strides") or config["pool_size"])
        if weights:
            kernel, bias = weights
            entry.update(kernel_shape=list(kernel.shape), kernel=b64(kernel, "<f4"), bias=b64(bias, "<f4"))
        layers.append(entry)
    verify_x = test_images[:VERIFY_COUNT]
    expected = model.predict(for_model(verify_x, kind), verbose=0)
    errors = results["metrics"]["errors"][:8]
    return {
        "layers": layers,
        "parameters": results["parameters"],
        "config": {k: results[k] for k in ("epochs", "batch_size", "seed", "train_samples",
                                           "validation_samples", "test_samples", "training_seconds")},
        "history": results["history"],
        "metrics": {
            "accuracy": results["metrics"]["accuracy"],
            "loss": results["metrics"]["loss"],
            "per_class": {str(i): results["metrics"]["classification_report"][str(i)] for i in range(10)},
            "macro_f1": results["metrics"]["classification_report"]["macro avg"]["f1-score"],
            "confusion_matrix": results["metrics"]["confusion_matrix"],
        },
        "errors": [dict(e, image=b64(np.round(test_images[e["index"]] * 255), "u1")) for e in errors],
        "verify_expected": b64(expected, "<f4"),
    }


def export_weights(model):
    """All layers with their weights as base64 float32 plus the config needed for inference."""
    layers = []
    for layer in model.layers:
        cfg = layer.get_config()
        entry = {"name": layer.name, "type": type(layer).__name__, "activation": cfg.get("activation"),
                 "units": cfg.get("units"), "reset_after": cfg.get("reset_after"),
                 "recurrent_activation": cfg.get("recurrent_activation"),
                 "weights": [{"shape": list(w.shape), "data": b64(w, "<f4")} for w in layer.get_weights()]}
        layers.append(entry)
    return layers


def export_signals():
    from signal_models import regression, timeseries  # imported lazily: optional part of the export

    data = {"regression": {}, "timeseries": None}
    for task in regression.TASKS:
        model_path = regression.OUT_DIR / f"{task}.keras"
        if not model_path.exists():
            continue
        model = tf.keras.models.load_model(model_path)
        result = json.loads((regression.OUT_DIR / f"{task}_results.json").read_text(encoding="utf-8"))
        probe = np.linspace(*result["extrapolate"], 7, dtype=np.float32).reshape(-1, 1)
        result.update(layers=export_weights(model), verify_x=probe[:, 0].tolist(),
                      verify_y=model.predict(probe, verbose=0)[:, 0].tolist())
        data["regression"][task] = result

    results_path = timeseries.OUT_DIR / "results.json"
    if results_path.exists():
        summary = json.loads(results_path.read_text(encoding="utf-8"))
        series = timeseries.make_series()
        _, _, test, mean, std, limits = timeseries.split_and_normalize(series)
        probe = test[:timeseries.WINDOW]
        for kind in timeseries.KINDS:
            model = tf.keras.models.load_model(timeseries.OUT_DIR / f"{kind.lower()}.keras")
            summary["models"][kind].update(layers=export_weights(model),
                                           verify_direct=timeseries.predict(model, kind, probe).tolist())
        step = 10  # thinned full series, only for the overview chart
        summary.update(test_series=b64(test * std + mean, "<f4"), test_start=limits[1],
                       overview={"step": step, "values": b64(series[::step], "<f4")})
        data["timeseries"] = summary

    out = OUTPUT.parent / "signals_data.js"
    out.write_text("// Generated by Actividad/Actividades 1/export_web.py - do not edit by hand.\n"
                   "window.SIGNALS_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Export trained models to docs/mnist/.")
    parser.add_argument("--part", choices=("all", "mnist", "signals"), default="all")
    part = parser.parse_args().part
    if part in ("all", "signals"):
        export_signals()
    if part == "signals":
        return
    _, _, (test_images, test_labels) = load_mnist()
    data = {
        "samples": [{"label": int(test_labels[i]), "image": b64(np.round(test_images[i] * 255), "u1")}
                    for i in range(6)],
        "verify_images": b64(test_images[:VERIFY_COUNT], "<f4"),
        "verify_count": VERIFY_COUNT,
        "models": {kind: export_model(kind, test_images) for kind in MODEL_FILES},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("// Generated by Actividad/Actividades 1/export_web.py - do not edit by hand.\n"
                      "window.MNIST_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n",
                      encoding="utf-8")
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
