import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from research_pipeline import features,pinball,load_csv
class ResearchTests(unittest.TestCase):
 def test_no_future_features(self):
  idx=pd.date_range('2020-01-01',periods=800,freq='30min',tz='UTC')
  s=pd.Series(np.arange(800,dtype=float),index=idx)
  f=features(s)
  self.assertNotIn('lag_96',f.columns)
  self.assertNotIn('lag_144',f.columns)
  self.assertTrue(np.isnan(f.iloc[335].lag_336))
  self.assertEqual(f.iloc[336].lag_336,0)
  self.assertEqual(f.iloc[672].lag_672,0)
  changed=s.copy();changed.iloc[700:]=99999
  before=features(s);after=features(changed)
  pd.testing.assert_frame_equal(before.iloc[:700].drop(columns=['target']),after.iloc[:700].drop(columns=['target']))
 def test_issue_time_availability(self):
  # Every lagged actual must be at least 48 hours old when the
  # day-ahead forecast is issued at 12:00 UTC on the previous day.
  idx=pd.date_range('2023-01-01',periods=1000,freq='30min',tz='UTC')
  s=pd.Series(np.arange(1000,dtype=float),index=idx)
  f=features(s)
  for delivery in idx[672:]:
   issue=delivery.normalize()-pd.Timedelta(days=1)+pd.Timedelta(hours=12)
   for lag in (336,672):
    source=delivery-pd.Timedelta(minutes=30*lag)
    self.assertLessEqual(source+pd.Timedelta(hours=48),issue)
  self.assertNotIn('lag_96',f.columns)
  self.assertNotIn('lag_144',f.columns)
 def test_pinball(self):
  self.assertAlmostEqual(pinball([10],[0],.1),1)
  self.assertAlmostEqual(pinball([0],[10],.9),1)
 def test_missing_data_not_backfilled(self):
  idx=pd.date_range('2020-01-01',periods=800,freq='30min',tz='UTC')
  s=pd.Series(np.ones(800),index=idx);s.iloc[0]=np.nan
  self.assertTrue(np.isnan(features(s).iloc[336].lag_336))
  self.assertTrue(np.isnan(features(s).iloc[672].lag_672))
if __name__=='__main__':unittest.main()
