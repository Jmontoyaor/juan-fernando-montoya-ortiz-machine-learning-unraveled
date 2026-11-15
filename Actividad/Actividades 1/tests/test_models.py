import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from mnist_app.data import for_model
from mnist_app.models import build_model


class ModelTests(unittest.TestCase):
    def test_reference_architectures(self):
        dense, cnn = build_model("dense"), build_model("cnn")
        self.assertEqual(dense.input_shape, (None, 784))
        self.assertEqual(cnn.input_shape, (None, 28, 28, 1))
        self.assertEqual(dense.output_shape, (None, 10))
        self.assertEqual(cnn.output_shape, (None, 10))
        # 784*64+64 + 64*64+64 + 64*10+10
        self.assertEqual(dense.count_params(), 55_050)
        # (3*3*1*32+32) + (3*3*32*64+64) + (24*24*64*10+10)
        self.assertEqual(cnn.count_params(), 387_466)

    def test_pooling_architecture_from_scratchpad_3(self):
        pool = build_model("cnn_pool")
        self.assertEqual(pool.output_shape, (None, 10))
        self.assertEqual(pool.get_layer("pool_1").output.shape[1:], (14, 14, 32))
        self.assertEqual(pool.get_layer("pool_2").output.shape[1:], (7, 7, 64))
        # (9*1*32+32) + (9*32*64+64) + (7*7*64*10+10)
        self.assertEqual(pool.count_params(), 50_186)

    def test_input_shapes(self):
        images = np.random.default_rng(0).random((5, 28, 28), dtype=np.float32)
        self.assertEqual(for_model(images, "dense").shape, (5, 784))
        self.assertEqual(for_model(images, "cnn").shape, (5, 28, 28, 1))

    def test_short_training_save_reload_predict(self):
        rng = np.random.default_rng(0)
        images = rng.random((64, 28, 28), dtype=np.float32)
        labels = rng.integers(0, 10, 64)
        for kind in ("dense", "cnn", "cnn_pool"):
            model = build_model(kind)
            model.fit(for_model(images, kind), labels, epochs=1, batch_size=32, verbose=0)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / f"{kind}.keras"
                model.save(path)
                reloaded = tf.keras.models.load_model(path)
            probs = reloaded.predict(for_model(images[:2], kind), verbose=0)
            self.assertEqual(probs.shape, (2, 10))
            np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
            np.testing.assert_allclose(probs, model.predict(for_model(images[:2], kind), verbose=0), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
