from regressors import Regressor
from helper import Data
import pandas as pd


def run_predictions():
    data = Data()

    data.fetch_data()
    total_df = data.preprocess_cases_data(data.df_us_cases)
    daily_df = data.daily_data(total_df)
    daily_reg = Regressor(daily_df, 7)

    pred_path = "data/"
    for state in daily_df.columns.tolist():
        tempdf = pd.DataFrame(daily_df[state], index=daily_df.index)
        state_reg = Regressor(tempdf, 7)

        arima_state_data = state_reg.ARIMA(row=state)
        arima_state_data.to_csv(pred_path + "daily_{}_ARIMA.csv".format(state))

        xg_state_data = state_reg.XGBoost(row=state)
        xg_state_data.to_csv(pred_path + "daily_{}_XGBoost.csv".format(state))

        lstm_state_data = state_reg.LSTM(row=state, num_estimators=1)
        lstm_state_data.to_csv(pred_path + "daily_{}_LSTM.csv".format(state))

    total_df_death = data.preprocess_death_data(data.df_us_deaths)
    daily_df_death = data.daily_data(total_df_death)

    for state in daily_df_death.columns.tolist():
        tempdf = pd.DataFrame(daily_df_death[state], index=daily_df_death.index)
        death_reg = Regressor(tempdf, 7)

        arima_state_death = death_reg.ARIMA(row=state)
        arima_state_death.to_csv(pred_path + "death_{}_ARIMA.csv".format(state))

        xg_state_death = death_reg.XGBoost(row=state)
        xg_state_death.to_csv(pred_path + "death_{}_XGBoost.csv".format(state))

        lstm_state_death = death_reg.LSTM(row=state, num_estimators=1)
        lstm_state_death.to_csv(pred_path + "death_{}_LSTM.csv".format(state))
