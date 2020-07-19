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

arima_daily_data = daily_reg.ARIMA()
arima_daily_data.to_csv(pred_path+"Daily_ARIMA.csv")

xg_daily_data = daily_reg.XGBoost()
xg_daily_data.to_csv(pred_path+"Daily_XGBoost.csv")

lstm_daily_data = daily_reg.LSTM()
lstm_daily_data.to_csv(pred_path+"Daily_LSTM.csv")

total_reg = Regressor(total_df, 7)

arima_total_data = total_reg.ARIMA()
arima_total_data.to_csv(pred_path+"Total_ARIMA.csv")

xg_total_data = total_reg.XGBoost()
xg_total_data.to_csv(pred_path+"Total_XGBoost.csv")

lstm_total_data = total_reg.LSTM()
lstm_total_data.to_csv(pred_path+"Total_LSTM.csv")


for state in daily_df.columns.tolist()[:-1]:
    tempdf = pd.DataFrame(daily_df[state], index=daily_df.index)
    # tempdf = tempdf.dropna()
    state_reg = Regressor(tempdf, 7)

    arima_state_data = state_reg.ARIMA(row=state)
    arima_state_data.to_csv(pred_path+"daily_{}_ARIMA.csv".format(state))

    xg_state_data = state_reg.XGBoost(row=state)
    xg_state_data.to_csv(pred_path+"daily_{}_XGBoost.csv".format(state))

    lstm_state_data = state_reg.LSTM(row=state, num_estimators=3)
    lstm_state_data.to_csv(pred_path+"daily_{}_LSTM.csv".format(state))

for state in total_df.columns.tolist()[:-1]:
    tempdf = pd.DataFrame(total_df[state], index=total_df.index)
    tempdf = tempdf.dropna()
    state_reg = Regressor(tempdf, 7)

    arima_state_data = state_reg.ARIMA(row=state)
    arima_state_data.to_csv(pred_path+"total_{}_ARIMA.csv".format(state))

    xg_state_data = state_reg.XGBoost(row=state)
    xg_state_data.to_csv(pred_path+"total_{}_XGBoost.csv".format(state))

    lstm_state_data = state_reg.LSTM(row=state, num_estimators=3)
    lstm_state_data.to_csv(pred_path+"total_{}_LSTM.csv".format(state))