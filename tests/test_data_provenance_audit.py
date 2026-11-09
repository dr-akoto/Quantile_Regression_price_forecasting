import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from data_provenance_audit import audit


class DataProvenanceAuditTests(unittest.TestCase):
    def write_csv(self, folder, timestamps, prices):
        path = Path(folder) / 'prices.csv'
        pd.DataFrame({'dtm': timestamps, 'DA_Price': prices}).to_csv(path, index=False)
        return path

    def test_clean_utc_grid_and_unverified_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_csv(tmp, ['2023-01-01 00:00', '2023-01-01 00:30'],
                                  [10, -5])
            result = audit(path)
            self.assertEqual(result['missing_30min_utc_grid_timestamps'], 0)
            self.assertEqual(result['negative_price_count'], 1)
            self.assertFalse(result['publication_timing_verified'])
            self.assertFalse(result['source_provenance_verified'])

    def test_gap_and_duplicate_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_csv(tmp, ['2023-01-01 00:00', '2023-01-01 00:00',
                                        '2023-01-01 01:00'], [1, 2, 3])
            result = audit(path)
            self.assertEqual(result['duplicate_utc_timestamps'], 1)
            self.assertEqual(result['missing_30min_utc_grid_timestamps'], 1)

    def test_invalid_timestamp_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_csv(tmp, ['not a date'], [1])
            with self.assertRaisesRegex(ValueError, 'Invalid timestamps'):
                audit(path)

    def test_negative_delay_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_csv(tmp, ['2023-01-01'], [1])
            with self.assertRaisesRegex(ValueError, 'nonnegative'):
                audit(path, assumed_delay_hours=-1)


if __name__ == '__main__':
    unittest.main()
