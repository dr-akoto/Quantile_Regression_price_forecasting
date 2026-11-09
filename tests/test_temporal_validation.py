import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from temporal_validation import evaluate


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


class TemporalValidationTests(unittest.TestCase):
    def test_split_and_candidate_count(self):
        report = evaluate(sample(), development_days=20, test_days=10,
                          warmup_days=10, windows=(7, 14))
        self.assertEqual(report['held_out_selected']['observations'], 480)
        self.assertEqual(len(report['development_candidates']), 5)
        self.assertEqual(report['held_out_start_utc'][:10], '2023-01-21')

    def test_future_actuals_do_not_change_selection(self):
        data = sample()
        first = evaluate(data, 20, 10, 10, (7, 14))
        altered = data.copy()
        altered.loc[20*48:, 'actual'] = 10000.
        second = evaluate(altered, 20, 10, 10, (7, 14))
        self.assertEqual(first['selected_method'], second['selected_method'])
        self.assertEqual(first['selected_window_days'], second['selected_window_days'])
        self.assertEqual(first['development_candidates'], second['development_candidates'])

    def test_incorrect_day_count_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exactly'):
            evaluate(sample(), 20, 9, 10, (7,))


if __name__ == '__main__':
    unittest.main()
