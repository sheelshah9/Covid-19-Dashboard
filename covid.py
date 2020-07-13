from regressors import Regressor
import plotly.graph_objects as go
from graphs import Graphs
from helper import fetch_data, preprocess_data
import pandas as pd

def forecast_daily_cases(grouped_df, daily_df):

    ts = daily_df
    reg = Regressor()
    print(daily_df)
    forecasted_days_arima, real_arima, intervals_arima = reg.ARIMA(daily_df, interval_forecast=10)
    forecasted_days_xg, real_xg, intervals_xg = reg.XGBoost(daily_df, interval_forecast=10)
    ts = ts.drop(ts.columns[1:39], axis=1)

    data = []
    flag = True
    for forecasted_days, real, intervals in zip([forecasted_days_arima, forecasted_days_xg], [real_arima, real_xg], [intervals_arima, intervals_xg]):
        upper_bound = go.Scatter(
            name="Upper Bound",
            x=ts.columns.to_list() + forecasted_days,
            y=ts.loc['Total'].tolist() + intervals[1].tolist(),
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            visible=flag
        )

        trace = go.Scatter(
            name="Prediction",
            x=ts.columns.to_list() + forecasted_days,
            y=ts.loc['Total'].tolist() + real.tolist(),
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            visible=flag
        )

        lower_bound = go.Scatter(
            name="Lower Bound",
            x=ts.columns.to_list() + forecasted_days,
            y=ts.loc['Total'].tolist() + intervals[0].tolist(),
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            visible=flag
        )
        data.append(lower_bound)
        data.append(trace)
        data.append(upper_bound)
        flag=False

    return data

def forecast_total_cases(grouped_df, daily_df):

    ts = grouped_df
    reg = Regressor()
    forecasted_days_arima, real_arima, intervals_arima = reg.ARIMA(grouped_df, interval_forecast=10)
    forecasted_days_xg, real_xg, intervals_xg = reg.XGBoost(grouped_df, interval_forecast=10)
    ts = ts.drop(ts.columns[1:39], axis=1)

    data = []
    flag = True
    for forecasted_days, real, intervals in zip([forecasted_days_arima, forecasted_days_xg], [real_arima, real_xg],
                                                [intervals_arima, intervals_xg]):
        upper_bound = go.Scatter(
            name="Upper Bound",
            x=ts.columns.to_list() + forecasted_days,
            y=ts.loc['Total'].tolist() + intervals[1].tolist(),
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            visible=flag
        )

        trace = go.Scatter(
            name="Prediction",
            x=ts.columns.to_list() + forecasted_days,
            y=ts.loc['Total'].tolist() + real.tolist(),
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            visible=flag
        )

        lower_bound = go.Scatter(
            name="Lower Bound",
            x=ts.columns.to_list() + forecasted_days,
            y=ts.loc['Total'].tolist() + intervals[0].tolist(),
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            visible=flag
        )
        data.append(lower_bound)
        data.append(trace)
        data.append(upper_bound)
        flag = False

    return data

def forecast_state_wise(grouped_df, daily_df):
    ts = daily_df
    reg = Regressor()
    forecast = {}
    for state,abbvr in Graphs.us_state_abbrev.items():
        tempdf = pd.DataFrame(daily_df.loc[state])
        tempdf = tempdf.dropna()
        # tempdf = (tempdf-tempdf.mean())/tempdf.std()
        tempdf = tempdf[tempdf[state]>0]

        # tempdf = tempdf.dropna()
        if tempdf[tempdf[state]>0].shape[0]==0:
            continue
        print(state)
        forecasted_days_arima, real_arima, intervals_arima = reg.ARIMA(tempdf.T, interval_forecast=10, row=state)
        forecast[state] = [forecasted_days_arima, real_arima, intervals_arima, ]
    ts = ts.drop(ts.columns[1:39], axis=1)

    data = []
    flag = True
    states = []
    for key, val in forecast.items():
        states.append(key)
        forecasted_days, real, intervals = val
        upper_bound = go.Scatter(
            name="Upper Bound",
            x=ts.columns.to_list() + forecasted_days,
            y=ts.loc[key].tolist() + intervals[1].tolist(),
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            visible=flag
        )

        trace = go.Scatter(
            name="Prediction",
            x=ts.columns.to_list() + forecasted_days,
            y=ts.loc[key].tolist() + real.tolist(),
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            visible=flag
        )

        lower_bound = go.Scatter(
            name="Lower Bound",
            x=ts.columns.to_list() + forecasted_days,
            y=ts.loc[key].tolist() + intervals[0].tolist(),
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            visible=flag
        )
        data.append(lower_bound)
        data.append(trace)
        data.append(upper_bound)
        flag = False

    return states, data


import dash_core_components as dcc
import dash_html_components as html
import dash
from dash.dependencies import Input, Output

df = fetch_data()
grouped_df, daily_df = preprocess_data(df)

data = forecast_daily_cases(grouped_df, daily_df)
total_data = forecast_total_cases(grouped_df, daily_df)
states, state_wise_data = forecast_state_wise(grouped_df, daily_df)
graph = Graphs()

app = dash.Dash(__name__)
server = app.server
app.layout = html.Div([
    html.H1('COVID-19 Dashboard'),
    dcc.Graph(id="Data", figure=graph.draw_graph_daily_increase(data)),
    dcc.Interval(
        id="12hrinterval",
        interval=43200000,
        n_intervals=0
    ),
    dcc.Graph(id="State", figure=graph.draw_total_state_map(df)),
    dcc.Interval(
        id="12hrinterval_state",
        interval=43200000,
        n_intervals=0
    ),
    dcc.Graph(id="Total_US", figure=graph.draw_graph_daily_increase(total_data)),
    dcc.Interval(
        id="12hrinterval_total",
        interval=43200000,
        n_intervals=0
    ),
    dcc.Graph(id="State_wise", figure=graph.draw_graph_state_wise(states, state_wise_data)),
    dcc.Interval(
        id="12hrinterval_statewise",
        interval=43200000,
        n_intervals=0
    )
])

@app.callback(Output("Data", "figure"),
        [Input("12hrinterval", "n_intervals")])
def draw_figure(n):
    df = fetch_data()
    grouped_df, daily_df = preprocess_data(df)
    data = forecast_daily_cases(grouped_df, daily_df)
    fig = graph.draw_graph_daily_increase(data)
    return fig

@app.callback(Output("Data", "figure"),
        [Input("12hrinterval_state", "n_intervals")])
def draw_figure_state(n):
    df = fetch_data()
    fig = graph.draw_total_state_map(df)
    return fig

@app.callback(Output("Data", "figure"),
        [Input("12hrinterval_total", "n_intervals")])
def draw_figure_total(n):
    df = fetch_data()
    grouped_df, daily_df = preprocess_data(df)
    data = forecast_total_cases(grouped_df, daily_df)
    fig = graph.draw_graph_total_increase(data)
    return fig

@app.callback(Output("Data", "figure"),
        [Input("12hrinterval_statewise", "n_intervals")])
def draw_figure_total(n):
    df = fetch_data()
    grouped_df, daily_df = preprocess_data(df)
    state_wise_forecast = forecast_state_wise(grouped_df, daily_df)
    fig = graph.draw_graph_total_increase(data)
    return fig


if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0')