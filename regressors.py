from statsmodels.tsa.arima_model import ARIMA
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

class Regressor:
    def __init__(self, daily_df, forecast_interval):
        self.daily_df = daily_df
        self.forecast_interval = forecast_interval

    @staticmethod
    def generate_dates(start, interval_forecast):
        index = pd.date_range(start, periods=interval_forecast+1, freq='D')
        return index[1:]

    def ARIMA(self, row='Total'):
        daily_df = self.daily_df.copy()
        params = [(i,j,k) for i in range(5,7) for j in range(3) for k in range(3)]
        min_aic = float("inf")
        final_model = None
        final_param = (6,2,2)
        for param in params:
            try:
                model = ARIMA(np.array(daily_df[row].tolist(), dtype=np.float32), order=param)
                model = model.fit()
                if model.aic<min_aic:
                    min_aic=model.aic
                    final_model = model
                    final_param = param
                # break
            except Exception as e:
                continue

        model = final_model
        preds = model.predict(start = final_param[1], end = len(daily_df.index)-1)
        preds = np.array(preds) + np.array(daily_df[row].tolist())[final_param[1]:]

        real, _, intervals = model.forecast(self.forecast_interval)
        preds = np.append(preds, real)
        preds[preds < 0] = 0

        forecasted_days = self.generate_dates(daily_df.index[-1], self.forecast_interval)
        preds = pd.DataFrame(data=preds, columns=["forecast"], index=daily_df.index[final_param[1]:].union(forecasted_days))
        interval_low = pd.DataFrame(data=intervals[:,0], columns=["interval_low"], index=forecasted_days)
        interval_high = pd.DataFrame(data=intervals[:,1], columns=["interval_high"], index=forecasted_days)

        daily_df = pd.concat([daily_df, preds, interval_low, interval_high], axis=1)
        daily_df['interval_low'].fillna(daily_df['forecast'], inplace=True)
        daily_df['interval_high'].fillna(daily_df['forecast'], inplace=True)
        return daily_df

    def XGBoost(self, row='Total'):
        daily_df = self.daily_df.copy()
        X = daily_df.index.tolist()
        # Day of Week, Day of month, Month, Day of Year, Week of Year
        X = [[x.isoweekday(), x.day, x.month, int(x.strftime("%j")), int(x.strftime("%W"))] for i,x in enumerate(X)]
        Y = daily_df[row].tolist()
        y_hat = []

        forecasted_days = self.generate_dates(daily_df.index[-1], self.forecast_interval)
        forecasted_days_ret = forecasted_days.copy()

        # Day of Week, Day of month, Month, Day of Year, Week of Year
        forecasted_days = [[x.isoweekday(), x.day, x.month, int(x.strftime("%j")), int(x.strftime("%W"))] for i, x in
                           enumerate(forecasted_days)]
        forecasted_days = np.concatenate([X, forecasted_days])

        for _ in range(5):
            x_train, y_train, x_test, y_test = train_test_split(X, Y, test_size=0.2)
            model = xgb.XGBRegressor(n_estimators=100, early_stopping_rounds=50, verbosity=0)
            x_train, x_test, y_train, y_test = pd.DataFrame(x_train), pd.DataFrame(y_train), pd.DataFrame(x_test), pd.DataFrame(y_test)
            model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)])
            pred = np.array(model.predict(pd.DataFrame(forecasted_days)))
            y_hat.append(pred)

        low, high = [], []
        preds = []
        y_hat = np.array(y_hat)
        for i in range(len(forecasted_days)):
            low.append(y_hat[:, i].min())
            high.append(y_hat[:, i].max())
            preds.append(y_hat[:, i].mean())

        preds = pd.DataFrame(data=preds, columns=["forecast"],
                             index=daily_df.index.union(forecasted_days_ret))
        interval_low = pd.DataFrame(data=low, columns=["interval_low"], index=daily_df.index.union(forecasted_days_ret))
        interval_high = pd.DataFrame(data=high, columns=["interval_high"], index=daily_df.index.union(forecasted_days_ret))

        daily_df = pd.concat([daily_df, preds, interval_low, interval_high], axis=1)

        return daily_df

    def XGBoost_mod(self, daily_df, interval_forecast):
        test_df = daily_df.loc['Total'].T
        final_df = test_df.copy()
        fixed_interval = 5
        for i in range(fixed_interval+interval_forecast):
            final_df = pd.concat([test_df.shift(i+1), final_df], axis=1)
        final_df = final_df.iloc[fixed_interval+interval_forecast:,1:]
        final_df.columns = [i for i in range(fixed_interval+interval_forecast)]

        model = xgb.XGBRegressor(n_estimators=300, early_stopping_rounds=50, verbosity=0)
        x, y = final_df.iloc[:,:-interval_forecast], final_df.iloc[:,-interval_forecast:]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

        multi_model = MultiOutputRegressor(model).fit(x_train, y_train)

        x_forecast = pd.DataFrame(final_df.iloc[-1,interval_forecast:].tolist(), index=x_train.columns).T
        forecasted_days = []
        pred = multi_model.predict(x_forecast)
        # print(pred)
        return pred[0]

    def LSTM(self, lookback=7, row='Total', num_estimators=10):
        daily_df = self.daily_df.copy()
        scaler = MinMaxScaler()
        train = daily_df[row].tolist()
        train = np.array(train).reshape((-1, 1))
        train = scaler.fit_transform(train)
        train = train.ravel()

        x_train, y_train = [], []
        for i in range(len(train) - lookback - self.forecast_interval + 1):
            x = train[i:i + lookback]
            x_train.append(x)
            y_train.append(train[i + lookback:i + lookback + self.forecast_interval])

        train_pred = []
        for i in range(len(train)-lookback+1):
            train_pred.append(train[i:i+lookback])

        train_pred = np.array(train_pred)
        train_pred = train_pred.reshape((train_pred.shape[0], train_pred.shape[1], 1))

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        print(x_train.shape)
        model = Sequential()
        model.add(LSTM(50, input_shape=(lookback, 1)))
        # model.add(LSTM(50, activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(self.forecast_interval))

        model.compile(loss='mean_squared_error', optimizer='adam')
        y_hat = []
        for _ in range(num_estimators):
            model.fit(x_train, y_train, epochs=100, verbose=0, shuffle=True)
            x_test = np.array(y_train[-1]).reshape((1, x_train.shape[1], 1))
            x_test = np.concatenate([train_pred, x_test])
            y_hat.append(scaler.inverse_transform(model.predict(x_test)))

        y_hat = np.array(y_hat)
        ans = []
        for i in range(num_estimators):
            ans.append(y_hat[i, 0:y_hat.shape[1]:lookback].ravel())
        ans = np.array(ans)
        print(ans.shape)
        low, high = [], []
        pred = []
        for i in range(ans.shape[1]):
            low.append(max(ans[:, i].min(), 0))
            high.append(ans[:, i].max())
            pred.append(max(ans[:, i].mean(), 0))


        forecasted_days = self.generate_dates(daily_df.index[-1], self.forecast_interval)
        preds = pd.DataFrame(data=pred, columns=["forecast"],
                             index=daily_df.index[lookback+(train_pred.shape[0])%lookback-1:].union(forecasted_days))
        interval_low = pd.DataFrame(data=low, columns=["interval_low"], index=daily_df.index[lookback+(train_pred.shape[0])%lookback-1:].union(forecasted_days))
        interval_high = pd.DataFrame(data=high, columns=["interval_high"], index=daily_df.index[lookback+(train_pred.shape[0])%lookback-1:].union(forecasted_days))

        daily_df = pd.concat([daily_df, preds, interval_low, interval_high], axis=1)
        return daily_df


if __name__ == "__main__":

    import pandas as pd, datetime
    from helper import Data
    import matplotlib.pyplot as plt

    d = Data()

    d.fetch_data()
    df = d.preprocess_cases_data(d.df_us_cases)
    df = d.daily_data(df)
    reg = Regressor(df,7)
    # df = reg.ARIMA()
    # # print(i)
    xgm = reg.ARIMA()
    # xgm['lstm_forecast'].plot('g')
    # xgm['lstm_interval_low'].plot('r')
    # plt.plot(xgm.index, xgm['lstm_forecast'], 'g')
    # plt.plot(xgm.index, xgm['lstm_interval_low'], 'r')
    # plt.plot(xgm.index, xgm['lstm_interval_high'], 'y')
    # plt.bar(xgm.index, xgm['Total'])
    # plt.show()