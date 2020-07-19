from regressors import Regressor
import pickle
from helper import Data
from graphs import Graphs
import pandas as pd

d = Data()

d.fetch_data()
total_df = d.preprocess_data(d.df_us_cases)
daily_df = d.daily_data(total_df)
daily_reg = Regressor(daily_df, 7)

pred_path = "data/"

reg = Regressor()
arima_daily_data = daily_reg.ARIMA()
arima_daily_data.to_csv("Total_ARIMA.csv")

xg_daily_data = reg.XGBoost()
xg_daily_data.to_csv("Total_XGBoost.csv")

lstm_daily_data = reg.LSTM()
lstm_daily_data.to_csv("Total_LSTM.csv")

total_reg = Regressor(total_df, 7)
arima_total_data = total_reg.ARIMA()
arima_total_data.to_csv("T")
forecasted_days_arima, real_arima, intervals_arima = reg.ARIMA(grouped_df, interval_forecast=10)
with open(pred_path+'forecast_total_cases_arima', 'wb') as fp:
    pickle.dump([forecasted_days_arima, real_arima, intervals_arima], fp)

forecasted_days_xg, real_xg, intervals_xg = reg.XGBoost(grouped_df, interval_forecast=10)
with open(pred_path+'forecast_total_cases_xgboost', 'wb') as fp:
    pickle.dump([forecasted_days_xg, real_xg, intervals_xg], fp)

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