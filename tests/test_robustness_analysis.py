import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from robustness_analysis import interval_score, diagnostics


class RobustnessTests(unittest.TestCase):
    def test_interval_score_inside_and_outside(self):
        scores = interval_score([5., -2., 12.], [0., 0., 0.],
                                [10., 10., 10.], alpha=.2)
        np.testing.assert_allclose(scores, [10., 30., 30.])

    def test_paired_diagnostics(self):
        df = pd.DataFrame({
            'delivery_utc': ['2023-01-01T00:00:00Z', '2023-01-01T00:30:00Z'],
            'actual': [5., 12.],
            'q0.1': [0., 0.], 'q0.9': [10., 10.],
            'lower_calibrated': [-2., -2.],
            'upper_calibrated': [12., 12.]
        })
        daily, summary = diagnostics(df)
        self.assertEqual(summary['observations'], 2)
        self.assertEqual(summary['original_coverage'], .5)
        self.assertEqual(summary['calibrated_coverage'], 1.)
        self.assertEqual(summary['original_above_rate'], .5)
        self.assertEqual(summary['calibrated_above_rate'], 0.)
        self.assertEqual(len(daily), 1)

    def test_invalid_bounds_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Lower interval'):
            interval_score([2.], [5.], [1.])


if __name__ == '__main__':
    unittest.main()
