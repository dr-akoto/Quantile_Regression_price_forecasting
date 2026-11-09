import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from research_pipeline import features
from daily_backtest import issue_time, eligible_training


class DailyBacktestTests(unittest.TestCase):
    def test_issue_is_noon_previous_utc_day(self):
        day = pd.Timestamp('2023-10-29', tz='UTC')
        self.assertEqual(issue_time(day), pd.Timestamp('2023-10-28 12:00', tz='UTC'))

    def test_training_targets_are_published_by_issue(self):
        idx = pd.date_range('2023-01-01', periods=1200, freq='30min', tz='UTC')
        frame = features(pd.Series(np.arange(len(idx), dtype=float), index=idx))
        issue = pd.Timestamp('2023-01-24 12:00', tz='UTC')
        train = eligible_training(frame, issue, delay_hours=48)
        self.assertTrue((train.index + pd.Timedelta(hours=48) <= issue).all())
        self.assertEqual(train.index.max(), pd.Timestamp('2023-01-22 12:00', tz='UTC'))
        self.assertNotIn(pd.Timestamp('2023-01-22 12:30', tz='UTC'), train.index)

    def test_weekly_lags_available_for_full_delivery_day(self):
        idx = pd.date_range('2023-01-01', periods=2000, freq='30min', tz='UTC')
        frame = features(pd.Series(np.arange(len(idx), dtype=float), index=idx))
        day = pd.Timestamp('2023-02-01', tz='UTC')
        issue = issue_time(day)
        slots = frame.loc[frame.index.normalize() == day]
        self.assertEqual(len(slots), 48)
        for lag in (336, 672):
            sources = slots.index - pd.Timedelta(minutes=lag * 30)
            self.assertTrue(((sources + pd.Timedelta(hours=48)) <= issue).all())


if __name__ == '__main__':
    unittest.main()
