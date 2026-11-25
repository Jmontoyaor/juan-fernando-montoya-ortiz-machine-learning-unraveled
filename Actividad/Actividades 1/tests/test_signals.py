import unittest

import numpy as np

from signal_models import regression, timeseries as ts


class TimeSeriesTests(unittest.TestCase):
    def test_series_is_reproducible(self):
        np.testing.assert_array_equal(ts.make_series(), ts.make_series())
        self.assertEqual(ts.make_series().shape, (ts.N,))

    def test_chronological_split_and_train_only_normalization(self):
        series = ts.make_series()
        train, val, test, mean, std, (a, b) = ts.split_and_normalize(series)
        self.assertEqual((len(train), len(val), len(test)), (28_000, 6_000, 6_000))
        self.assertAlmostEqual(mean, float(series[:a].mean()), places=4)
        self.assertAlmostEqual(std, float(series[:a].std()), places=4)
        np.testing.assert_allclose(test, (series[b:] - mean) / std, atol=1e-5)  # test is the most recent part

    def test_windows_target_follows_input(self):
        signal = np.arange(3000, dtype=np.float32)
        x, y = ts.make_windows(signal)
        self.assertEqual(x.shape[1:], (ts.WINDOW,))
        self.assertEqual(y.shape[1:], (ts.HORIZON,))
        np.testing.assert_array_equal(y[:, 0], x[:, -1] + 1)  # target starts right after the window

    def test_fair_architectures(self):
        expected = {"DNN": 77_064, "RNN": 17_224, "LSTM": 29_896, "GRU": 25_864}
        for kind, params in expected.items():
            model = ts.build(kind)
            self.assertEqual(model.count_params(), params)
            self.assertEqual(model.output_shape, (None, ts.HORIZON))

    def test_autoregressive_length(self):
        model = ts.build("DNN")
        out = ts.autoregressive(model, "DNN", np.zeros(ts.WINDOW, np.float32), steps=3)
        self.assertEqual(out.shape, (3 * ts.HORIZON,))


class RegressionTests(unittest.TestCase):
    def test_architectures(self):
        self.assertEqual(regression.build("line").count_params(), 2)
        self.assertEqual(regression.build("cosine").count_params(), 2 * 64 + 2 * (64 * 64 + 64) + 65)

    def test_targets(self):
        np.testing.assert_allclose(regression.target("line", np.array([0.0, 1.0])), [1.0, 4.0])
        np.testing.assert_allclose(regression.target("cosine", np.array([0.0, 0.25, 0.5])), [1.0, -1.0, 1.0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
