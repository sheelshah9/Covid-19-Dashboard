import pandas as pd


class Data():

    def fetch_data(self):
        url_us_cases = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
        url_us_deaths = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv"
        self.df_us_cases = pd.read_csv(url_us_cases)
        self.df_us_deaths = pd.read_csv(url_us_deaths)

    @staticmethod
    def preprocess_cases_data(df):
        grouped_df = df.drop(['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Country_Region', 'Lat', 'Long_'],
                             axis=1)
        grouped_df = grouped_df.groupby(['Province_State']).sum()
        grouped_df.loc['Total'] = grouped_df.sum()
        grouped_df = grouped_df.T
        # If there are less than 10 new cases in last 7 days, we omit that state
        grouped_df = grouped_df.loc[:, (grouped_df.iloc[-7:, :] > 10).all(axis=0)]
        grouped_df.index = pd.to_datetime(grouped_df.index, infer_datetime_format=True)
        return grouped_df

    @staticmethod
    def preprocess_death_data(df):
        grouped_df = df.drop(
            ['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Country_Region', 'Lat', 'Long_', 'Combined_Key',
             'Population'],
            axis=1)
        grouped_df = grouped_df.groupby(['Province_State']).sum()
        grouped_df.loc['Total'] = grouped_df.sum()
        grouped_df = grouped_df.T
        # If there are less than 5 new deaths in last 7 days, we omit that state
        grouped_df = grouped_df.loc[:, (grouped_df.iloc[-7:, :] > 5).all(axis=0)]
        grouped_df.index = pd.to_datetime(grouped_df.index, infer_datetime_format=True)
        return grouped_df

    @staticmethod
    def daily_data(df):
        new_df = df.diff()
        new_df = new_df.iloc[1:, :]
        return new_df


if __name__ == "__main__":
    d = Data()
    d.fetch_data()
    x = d.preprocess_cases_data(d.df_us_cases)
    print(x)
