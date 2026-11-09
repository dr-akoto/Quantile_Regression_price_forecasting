import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from uncertainty_analysis import paired_day_bootstrap


def sample(days=30):
    rows = []
    for day in pd.date_range('2023-01-01', periods=days, tz='UTC'):
        for h in range(48):
            rows.append({
                'delivery_utc': str(day + pd.Timedelta(minutes=30*h)),
                'forecast_issue_utc': str(day - pd.Timedelta(hours=12)),
                'actual': float(h),
                'q0.1': 4., 'q0.9': 40.
            })
    return pd.DataFrame(rows)


class UncertaintyAnalysisTests(unittest.TestCase):
    def test_reproducible_paired_bootstrap(self):
        data = sample()
        a = paired_day_bootstrap(data, 20, 10, 10, (7, 14),
                                 replicates=100, seed=42)
        b = paired_day_bootstrap(data, 20, 10, 10, (7, 14),
                                 replicates=100, seed=42)
        self.assertEqual(a, b)
        self.assertEqual(a['held_out_observations'], 480)
        self.assertEqual(a['difference_definition'], 'selected minus original')
        for key in ('interval_score_difference', 'coverage_difference',
                    'mean_width_difference'):
            self.assertEqual(len(a[key]['bootstrap_95pct_interval']), 2)

    def test_too_few_replicates_rejected(self):
        with self.assertRaisesRegex(ValueError, '100'):
            paired_day_bootstrap(sample(), 20, 10, 10, (7, 14),
                                 replicates=50)


if __name__ == '__main__':
    unittest.main()
