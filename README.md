## What are these codes?
This is the offical implementation of the published paper. Please cite this work:  


### Abstract
In this study, a probabilistic electricity price forecasting scheme is proposed for the volatile UK market. The proposed scheme utilizes Quantile Regression (QR) with general predictors and excludes market-specific variables to ensure adaptability. When forecasting day-ahead electricity prices, the scheme employs a set of quantile models that integrate lagged price data, demand forecasts, and renewable energy generation forecasts (solar and wind) to train the regression models. Additionally, the proposed approach incorporates real-time data retrieval via an API to enable automated day-ahead forecasting, ensuring accuracy and operational efficiency.


## About the author
- [Yuki Osone], s2420851@u.tsukuba.ac.jp

## Structure of the code
![image](https://github.com/user-attachments/assets/d46b175b-3ffc-4197-94f6-f7e72b5fdd63)

The proposed model consists of multiple code files. By running api_DAP.py and api_SSP.py first, followed by predict_price.py, it is possible to perform real-time predictions while retrieving data via the APIs.

## How to run
### 1. Quantile Regression model (including solar power generation, electricity demand, and wind power generation forecasts as predictors)
Run "api_DAP.py", "api_SSP.py" and "predict_price.py" 

### 2. Quantile Regression model (not including solar power generation, electricity demand, and wind power generation forecasts as predictors)
Run "predict_price_api.py".  


## Reproducible UK research baseline (experimental)

The original implementation is retained. A separate, leakage-conscious research pipeline evaluates historical half-hourly UK price targets with forward-chaining tests, quantile regression and a weekly-naive benchmark. It deliberately excludes realized demand, wind and solar covariates that cannot be verified as available at forecast issuance.

Install dependencies and run from the repository root:

```bash
pip install pandas numpy scikit-learn
python code/research_pipeline.py --target DA_Price --folds 2 --test-days 60 --iterations 35 --output results/research_uk_da
python code/research_pipeline.py --target SS_Price --folds 2 --test-days 60 --iterations 35 --output results/research_uk_ss
python -m unittest discover -s tests -v
```

The default dataset is `code/data/Energy_Data_20200920_20231027.csv`. Results are generated locally in the chosen output directory and include `predictions.csv`, `fold_metrics.csv`, `summary.json` and `source_audit.json`.

**Research limitations:** This is not yet a reproduction of the complete HEFTCom2024 published methodology. New UK market sources and Tokyo/Japan data have not been integrated or validated. The assumed preceding-day issue time and 48-hour availability delay must be verified against each market's publication schedule. See [research status](docs/PROJECT_STATUS.md).
