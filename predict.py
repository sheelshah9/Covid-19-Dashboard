from regressors import Regressor
import pickle
from helper import fetch_data, preprocess_data
from graphs import Graphs
import pandas as pd

df = fetch_data()
grouped_df, daily_df = preprocess_data(df)
pred_path = "data/"

reg = Regressor()
# forecasted_days_arima, real_arima, intervals_arima = reg.ARIMA(daily_df, interval_forecast=10)
# with open(pred_path+'forecast_daily_cases_arima', 'wb') as fp:
#     pickle.dump([forecasted_days_arima, real_arima, intervals_arima], fp)
#
# forecasted_days_xg, real_xg, intervals_xg = reg.XGBoost(daily_df, interval_forecast=10)
# with open(pred_path+'forecast_daily_cases_xgboost', 'wb') as fp:
#     pickle.dump([forecasted_days_xg, real_xg, intervals_xg], fp)

forecasted_days_lstm, real_lstm, intervals_lstm = reg.LSTM(daily_df, interval_forecast=7)
with open(pred_path+'forecast_daily_cases_lstm', 'wb') as fp:
    pickle.dump([forecasted_days_lstm, real_lstm, intervals_lstm], fp)

# forecasted_days_arima, real_arima, intervals_arima = reg.ARIMA(grouped_df, interval_forecast=10)
# with open(pred_path+'forecast_total_cases_arima', 'wb') as fp:
#     pickle.dump([forecasted_days_arima, real_arima, intervals_arima], fp)
#
# forecasted_days_xg, real_xg, intervals_xg = reg.XGBoost(grouped_df, interval_forecast=10)
# with open(pred_path+'forecast_total_cases_xgboost', 'wb') as fp:
#     pickle.dump([forecasted_days_xg, real_xg, intervals_xg], fp)

forecasted_days_lstm, real_lstm, intervals_lstm = reg.LSTM(grouped_df, interval_forecast=7)
with open(pred_path+'forecast_total_cases_lstm', 'wb') as fp:
    pickle.dump([forecasted_days_lstm, real_lstm, intervals_lstm], fp)

for state,abbvr in Graphs.us_state_abbrev.items():
    tempdf = pd.DataFrame(daily_df.loc[state])
    tempdf = tempdf.dropna()
    tempdf = tempdf[tempdf[state]>0]

    if tempdf[tempdf[state]>0].shape[0]==0:
        continue

    forecasted_days_arima, real_arima, intervals_arima = reg.ARIMA(tempdf.T, interval_forecast=10, row=state)
    with open(pred_path+'forecast_'+state+'_cases_arima', 'wb') as fp:
        pickle.dump([forecasted_days_arima, real_arima, intervals_arima], fp)