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
  self.assertTrue(np.isnan(f.iloc[95].lag_96))
  self.assertEqual(f.iloc[96].lag_96,0)
  self.assertEqual(f.iloc[336].lag_336,0)
  changed=s.copy();changed.iloc[700:]=99999
  before=features(s);after=features(changed)
  pd.testing.assert_frame_equal(before.iloc[:700].drop(columns=['target']),after.iloc[:700].drop(columns=['target']))
 def test_pinball(self):
  self.assertAlmostEqual(pinball([10],[0],.1),1)
  self.assertAlmostEqual(pinball([0],[10],.9),1)
 def test_missing_data_not_backfilled(self):
  idx=pd.date_range('2020-01-01',periods=800,freq='30min',tz='UTC')
  s=pd.Series(np.ones(800),index=idx);s.iloc[0]=np.nan
  self.assertTrue(np.isnan(features(s).iloc[96].lag_96))
if __name__=='__main__':unittest.main()
