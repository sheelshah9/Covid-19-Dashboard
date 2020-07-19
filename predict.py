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


for state in daily_df.columns.tolist():
    tempdf = pd.DataFrame(daily_df[state], index=daily_df.index)
    # tempdf = tempdf.dropna()
    state_reg = Regressor(tempdf, 7)

    arima_state_data = state_reg.ARIMA(row=state)
    arima_state_data.to_csv(pred_path+"daily_{}_ARIMA.csv".format(state))

    xg_state_data = state_reg.XGBoost(row=state)
    xg_state_data.to_csv(pred_path+"daily_{}_XGBoost.csv".format(state))

    lstm_state_data = state_reg.LSTM(row=state, num_estimators=3)
    lstm_state_data.to_csv(pred_path+"daily_{}_LSTM.csv".format(state))
