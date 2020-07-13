import pandas as pd
import datetime

def fetch_data():
    url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
    df = pd.read_csv(url)
    return df

def preprocess_data(df):
    grouped_df = df.drop(['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Country_Region', 'Lat', 'Long_'], axis=1)
    grouped_df = grouped_df.groupby(['Province_State']).sum()
    grouped_df.loc['Total'] = grouped_df.sum()
    grouped_df.columns = [datetime.datetime.strptime(x, '%m/%d/%y').strftime("%d, %b %Y") for x in grouped_df.columns]

    daily_df = grouped_df.copy()

    for x, y in enumerate(daily_df.columns[::-1]):
        if x == len(daily_df.columns) - 1:
            continue
        else:
            daily_df[y] = daily_df[y] - daily_df[daily_df.columns[~x - 1]]

    return grouped_df, daily_df