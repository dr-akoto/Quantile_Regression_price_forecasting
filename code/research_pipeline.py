"""Auditable, market-agnostic, day-ahead probabilistic price forecasting.

Strict assumption: forecasts for delivery day D issued at 12:00 UTC on D-1.
Assume historical actual prices become available 48 hours after delivery.
Use only 7-day and 14-day historical lags, which are available by the
12:00 UTC D-1 forecast issue under this assumption.
Actual publication and revision timing must still be independently verified.
External forecast covariates are excluded unless point-in-time vintages are supplied.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

QS = (0.1, 0.25, 0.5, 0.75, 0.9)

def pinball(y, p, q):
    e = np.asarray(y) - np.asarray(p)
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))

def load_csv(path, time_col, target_col, timezone='UTC', frequency='30min'):
    raw = pd.read_csv(path)
    if time_col not in raw or target_col not in raw:
        raise ValueError('Missing required columns: '+time_col+', '+target_col)
    dt = pd.to_datetime(raw[time_col], errors='raise')
    if dt.dt.tz is None:
        # Explicit ambiguous/nonexistent policy: reject rather than silently shift.
        dt = dt.dt.tz_localize(timezone, ambiguous='raise', nonexistent='raise')
    dt = dt.dt.tz_convert('UTC')
    if dt.duplicated().any():
        raise ValueError('Duplicate UTC delivery timestamps')
    s = pd.Series(pd.to_numeric(raw[target_col], errors='coerce').to_numpy(), index=pd.DatetimeIndex(dt)).sort_index()
    full = pd.date_range(s.index.min(), s.index.max(), freq=frequency, tz='UTC')
    if not s.index.isin(full).all():
        raise ValueError('Timestamps not aligned with expected frequency')
    return s.reindex(full).rename('price'), {'source':str(path), 'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(), 'observed_rows':len(s), 'missing_grid_periods':len(full)-len(s), 'missing_target_values':int(s.isna().sum())}

def features(s, freq='30min'):
    if freq != '30min':
        raise ValueError('Only 30-minute settlement periods supported; adapt explicitly for other grids')
    idx = s.index
    df = pd.DataFrame({'target':s}, index=idx)
    hr = idx.hour + idx.minute/60
    day = idx.day_of_week
    year_day = idx.day_of_year
    for key, angle in [('hour', 2*np.pi*hr/24), ('dow',2*np.pi*day/7), ('year',2*np.pi*year_day/365.25)]:
        df[key+'_sin'] = np.sin(angle)
        df[key+'_cos'] = np.cos(angle)
    df['weekend'] = (day>=5).astype(int)
    # Conservative lags for forecasts issued at 12:00 UTC on D-1.
    # Under a 48-hour publication delay, 2- and 3-day lags are not
    # guaranteed to be available for every delivery slot.
    for lag in (336, 672):
        df[f'lag_{lag}'] = s.shift(lag)
    df['weekly_change'] = df['lag_336']-df['lag_672']
    return df

def evaluate(s, out, train_days=500, test_days=90, folds=3, model='boosting', iterations=75):
    df = features(s).dropna()
    if len(df)<(train_days+test_days)*48:
        raise ValueError('Not enough complete data for requested windows')
    dates = pd.DatetimeIndex(df.index).normalize().unique()
    # Nonoverlapping forward test windows, expanding training history.
    total = train_days + folds*test_days
    if len(dates)<total:
        raise ValueError(f'Need {total} complete/usable days; got {len(dates)}')
    start = len(dates)-total
    allpred=[]; rows=[]
    feature_cols=[c for c in df if c!='target']
    for fold in range(folds):
        cutoff=dates[start+train_days+fold*test_days]
        end=dates[start+train_days+(fold+1)*test_days] if fold<folds-1 else dates[-1]+pd.Timedelta(days=1)
        train=df[df.index<cutoff-pd.Timedelta(hours=48)]
        test=df[(df.index>=cutoff)&(df.index<end)]
        if train.empty or test.empty: raise ValueError('Empty fold')
        # Every train target must precede the forecast issue for the first test day.
        assert train.index.max()<cutoff-pd.Timedelta(hours=48)
        frame=pd.DataFrame({'delivery_utc':test.index.astype(str),'actual':test.target.to_numpy(), 'fold':fold,'baseline_weekly':test.lag_336.to_numpy()})
        preds=[]
        for q in QS:
            if model=='linear':
                estimator=make_pipeline(StandardScaler(),QuantileRegressor(quantile=q,alpha=0.1,solver='highs'))
            else:
                estimator=HistGradientBoostingRegressor(loss='quantile',quantile=q,max_iter=iterations,max_leaf_nodes=15,min_samples_leaf=35,l2_regularization=5,random_state=42)
            estimator.fit(train[feature_cols],train.target)
            preds.append(estimator.predict(test[feature_cols]))
        # Quantile rearrangement ensures non-crossing quantiles.
        preds=np.sort(np.vstack(preds),axis=0)
        for j,q in enumerate(QS):frame[f'q{q:g}']=preds[j]
        frame['interval_80_covered']=((frame.actual>=frame['q0.1'])&(frame.actual<=frame['q0.9'])).astype(int)
        frame['interval_80_width']=frame['q0.9']-frame['q0.1']
        for q in QS:
            rows.append({'fold':fold,'target':s.name,'quantile':q,'model_pinball':pinball(frame.actual,frame[f'q{q:g}'],q), 'baseline_pinball':pinball(frame.actual,frame.baseline_weekly,q),'train_rows':len(train),'test_rows':len(test),'train_end_utc':str(train.index.max()),'test_start_utc':str(test.index.min()),'test_end_utc':str(test.index.max())})
        allpred.append(frame)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    predictions=pd.concat(allpred,ignore_index=True)
    scores=pd.DataFrame(rows)
    predictions.to_csv(out/'predictions.csv',index=False)
    scores.to_csv(out/'fold_metrics.csv',index=False)
    aggregate={'target':s.name,'model':model,'forecast_issue':'12:00 UTC on preceding UTC day','assumed_publication_delay':'48 hours (conservative historical actuals)', 'feature_columns':feature_cols,'folds':folds,'mean_pinball_model':float(scores.model_pinball.mean()),'mean_pinball_weekly_baseline':float(scores.baseline_pinball.mean()),'relative_pinball_improvement_pct':float(100*(1-scores.model_pinball.mean()/scores.baseline_pinball.mean())),'empirical_80pct_coverage':float(predictions.interval_80_covered.mean()),'mean_80pct_width':float(predictions.interval_80_width.mean()),'warning':'Historical data only; actual publication/revision timing and data licensing require independent verification. Not a reproduction of original HEFTCom2024 paper.'}
    (out/'summary.json').write_text(json.dumps(aggregate,indent=2)+'\n')
    return aggregate

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,default=Path(__file__).parent/'data'/'Energy_Data_20200920_20231027.csv')
    p.add_argument('--time-column',default='dtm');p.add_argument('--target',default='DA_Price')
    p.add_argument('--timezone',default='UTC');p.add_argument('--output',type=Path,default=Path('results/research_uk'))
    p.add_argument('--model',choices=['boosting','linear'],default='boosting')
    p.add_argument('--train-days',type=int,default=500);p.add_argument('--test-days',type=int,default=90)
    p.add_argument('--folds',type=int,default=3);p.add_argument('--iterations',type=int,default=75)
    a=p.parse_args()
    s,meta=load_csv(a.input,a.time_column,a.target,a.timezone);s.name=a.target
    result=evaluate(s,a.output,a.train_days,a.test_days,a.folds,a.model,a.iterations)
    (a.output/'source_audit.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
