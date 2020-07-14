from statsmodels.tsa.arima_model import ARIMA
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
import datetime
import pandas as pd

class Regressor:
    def __int__(self):
        pass

    def ARIMA(self, daily_df, interval_forecast, row='Total'):
        # print(np.array(daily_df.loc[row].tolist(), dt)
        params = [(i,j,k) for i in range(5,7) for j in range(3) for k in range(2)]
        min_aic = float("inf")
        final_model = None
        for param in params:
            try:
                model = ARIMA(np.array(daily_df.loc[row].tolist(), dtype=np.float32), order=param)
                model = model.fit()
                if model.aic<min_aic:
                    min_aic=model.aic
                    final_model = model
            except Exception as e:
                continue
        model = final_model
        last_day = daily_df.columns[-1]
        forecasted_day = datetime.datetime.strptime(last_day, '%d, %b %Y')
        forecasted_days = []
        for i in range(1, interval_forecast + 1):
            temp = forecasted_day + datetime.timedelta(i)
            forecasted_days.append(temp.strftime("%d, %b %Y"))

        real, _, intervals = model.forecast(interval_forecast)
        print(type(intervals))
        return forecasted_days, real, np.array([intervals[:,0].tolist(), intervals[:,1].tolist()])

    def XGBoost(self, daily_df, interval_forecast, row='Total'):
        X = daily_df.columns.tolist()
        X = [datetime.datetime.strptime(x, '%d, %b %Y') for x in X]
        # Day of Week, Day of month, Month, Day of Year, Week of Year
        X = [[x.isoweekday(), x.day, x.month, int(x.strftime("%j")), int(x.strftime("%W"))] for i,x in enumerate(X)]
        Y = daily_df.loc[row].tolist()
        x_train, y_train, x_test, y_test = train_test_split(X,Y, test_size=0.1, random_state=10)
        model = xgb.XGBRegressor(n_estimators=300, early_stopping_rounds=50, verbosity=0)
        x_train, x_test, y_train, y_test = pd.DataFrame(x_train), pd.DataFrame(y_train), pd.DataFrame(x_test), pd.DataFrame(y_test)
        model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)])

        last_day = daily_df.columns[-1]
        forecasted_day = datetime.datetime.strptime(last_day, '%d, %b %Y')
        forecasted_days = []
        for i in range(1, interval_forecast + 1):
            temp = forecasted_day + datetime.timedelta(i)
            forecasted_days.append(temp.strftime("%d, %b %Y"))

        forecasted_days_ret = forecasted_days
        forecasted_days = [datetime.datetime.strptime(x, '%d, %b %Y') for x in forecasted_days]
        # Day of Week, Day of month, Month, Day of Year, Week of Year
        forecasted_days = [[x.isoweekday(), x.day, x.month, int(x.strftime("%j")), int(x.strftime("%W"))] for i, x in enumerate(forecasted_days)]
        preds = np.array(model.predict(pd.DataFrame(forecasted_days)))
        rmse = model.evals_result()['validation_1']['rmse'][-1]
        return forecasted_days_ret, preds, np.array([preds-rmse, preds+rmse])

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

if __name__ == "__main__":
    reg = Regressor()
    import pandas as pd, datetime

    def fetch_data():
        url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
        df = pd.read_csv(url)
        return df


    def preprocess_data(df):
        grouped_df = df.drop(['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Country_Region', 'Lat', 'Long_'],
                             axis=1)
        grouped_df = grouped_df.groupby(['Province_State']).sum()
        grouped_df.loc['Total'] = grouped_df.sum()

        daily_df = grouped_df.copy()
        daily_df.columns = [datetime.datetime.strptime(x, '%m/%d/%y').strftime("%d, %b %Y") for x in daily_df.columns]

        for x, y in enumerate(daily_df.columns[::-1]):
            if x == len(daily_df.columns) - 1:
                continue
            else:
                daily_df[y] = daily_df[y] - daily_df[daily_df.columns[~x - 1]]

        return daily_df

    df = preprocess_data(fetch_data())
    f, r, i = reg.ARIMA(df, 7)
    # print(i)
    xgm = reg.XGBoost_mod(df, 7)