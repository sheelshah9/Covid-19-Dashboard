from regressors import Regressor
import plotly.graph_objects as go
from graphs import Graphs
from helper import fetch_data, preprocess_data
import pandas as pd
import pickle

pred_path = "data/"

def forecast_daily_cases(grouped_df, daily_df):

    ts = daily_df
    reg = Regressor()
    # forecasted_days_arima, real_arima, intervals_arima = reg.ARIMA(daily_df, interval_forecast=10)

    with open(pred_path+'forecast_daily_cases_arima', 'rb') as fp:
        forecasted_days_arima, real_arima, intervals_arima = pickle.load(fp)
    # with open('forecast_daily_cases_arima', 'wb') as fp:
    #     pickle.dump([forecasted_days_arima, real_arima, intervals_arima], fp)

    # forecasted_days_xg, real_xg, intervals_xg = reg.XGBoost(daily_df, interval_forecast=10)
    # with open('forecast_daily_cases_xgboost', 'wb') as fp:
    #     pickle.dump([forecasted_days_xg, real_xg, intervals_xg], fp)
    with open(pred_path+'forecast_daily_cases_xgboost', 'rb') as fp:
        forecasted_days_xg, real_xg, intervals_xg = pickle.load(fp)

    with open(pred_path+'forecast_daily_cases_lstm', 'rb') as fp:
        forecasted_days_lstm, real_lstm, intervals_lstm = pickle.load(fp)

    ts = ts.drop(ts.columns[1:39], axis=1)

    data = []
    flag = True
    forecast_all, real_all, interval_all = [forecasted_days_arima, forecasted_days_xg, forecasted_days_lstm], [real_arima, real_xg, real_lstm], [intervals_arima, intervals_xg, intervals_lstm]
    for forecasted_days, real, intervals in zip(forecast_all, real_all, interval_all):
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
    # forecasted_days_arima, real_arima, intervals_arima = reg.ARIMA(grouped_df, interval_forecast=10)
    # with open(pred_path+'forecast_total_cases_arima', 'wb') as fp:
    #     pickle.dump([forecasted_days_arima, real_arima, intervals_arima], fp)
    with open(pred_path+'forecast_total_cases_arima', 'rb') as fp:
        forecasted_days_arima, real_arima, intervals_arima = pickle.load(fp)

    # forecasted_days_xg, real_xg, intervals_xg = reg.XGBoost(grouped_df, interval_forecast=10)
    # with open(pred_path+'forecast_total_cases_xgboost', 'wb') as fp:
    #     pickle.dump([forecasted_days_xg, real_xg, intervals_xg], fp)
    with open(pred_path+'forecast_total_cases_xgboost', 'rb') as fp:
        forecasted_days_xg, real_xg, intervals_xg = pickle.load(fp)

    with open(pred_path+'forecast_total_cases_lstm', 'rb') as fp:
        forecasted_days_lstm, real_lstm, intervals_lstm = pickle.load(fp)

    ts = ts.drop(ts.columns[1:39], axis=1)

    data = []
    flag = True
    forecast_all, real_all, interval_all = [forecasted_days_arima, forecasted_days_xg, forecasted_days_lstm], [
        real_arima, real_xg, real_lstm], [intervals_arima, intervals_xg, intervals_lstm]
    for forecasted_days, real, intervals in zip(forecast_all, real_all, interval_all):
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
        # print(state)
        # forecasted_days_arima, real_arima, intervals_arima = reg.ARIMA(tempdf.T, interval_forecast=10, row=state)
        # with open(pred_path+'forecast_'+state+'_cases_arima', 'wb') as fp:
        #     pickle.dump([forecasted_days_arima, real_arima, intervals_arima], fp)
        with open(pred_path+'forecast_'+state+'_cases_arima', 'rb') as fp:
            forecasted_days_arima, real_arima, intervals_arima = pickle.load(fp)
        forecast[state] = [forecasted_days_arima, real_arima, intervals_arima]
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
            marker=dict(color="#F58518"),
            line=dict(width=0),
            fillcolor='rgba(230, 131, 60, 0.5)',
            fill='tonexty',
            visible=flag
        )

        trace = go.Scatter(
            name="Prediction",
            x=ts.columns.to_list() + forecasted_days,
            y=ts.loc[key].tolist() + real.tolist(),
            mode='lines',
            line=dict(color='rgb(10, 10, 10)'),
            fillcolor='rgba(230, 131, 60, 0.5)',
            fill='tonexty',
            visible=flag
        )

        lower_bound = go.Scatter(
            name="Lower Bound",
            x=ts.columns.to_list() + forecasted_days,
            y=ts.loc[key].tolist() + intervals[0].tolist(),
            marker=dict(color="#F58518"),
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
import dash_bootstrap_components as dbc
import dash
from dash.dependencies import Input, Output

df = fetch_data()
grouped_df, daily_df = preprocess_data(df)

data = forecast_daily_cases(grouped_df, daily_df)
total_data = forecast_total_cases(grouped_df, daily_df)
states, state_wise_data = forecast_state_wise(grouped_df, daily_df)
graph = Graphs()

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.LUX])
server = app.server

navbar = dbc.Nav(className="nav nav-pills", children=[
    dbc.NavItem(dbc.NavLink([html.I(className="fa fa-github"), "  GitHub"], href="https://github.com/sheelshah9", active=True)),
    dbc.NavItem(dbc.NavLink([html.I(className="fa fa-linkedin"), "  LinkedIn"], href="https://www.linkedin.com/in/sheelshah09/", active=True))
])

app.layout = dbc.Container(fluid=True, children=[
    #Header
    html.Br(),
    dbc.Row([html.H1("Covid-19 Dashboard",id="nav-pills")], justify="center", align="center", className="h-50", style={"height": "100h"}),
    navbar,
    html.Br(),html.Br(),

    #Body
    dbc.Row([
            dbc.Col(md=9, children=[
            dbc.Col(html.H4("Forecast 10 days from today"), width={"size":6,"offset":3}),
            dbc.Tabs([
                dbc.Tab([dcc.Graph(id="graph_daily_increase_US", figure=graph.draw_graph_daily_increase(data)),
                         dcc.Graph(id="Total_US", figure=graph.draw_graph_daily_increase(total_data))],
                        label="US projected cases", ),
                dbc.Tab([dcc.Graph(id="State_wise", figure=graph.draw_graph_state_wise(states, state_wise_data)),
                         dcc.Graph(id="State_map", figure=graph.draw_total_state_map(df))],
                        label="State projections")
                    ])
                ])
            ])
]   )



# app.layout = html.Div([
#     html.H1('COVID-19 Dashboard'),
#     dcc.Graph(id="Data", figure=graph.draw_graph_daily_increase(data)),
#     dcc.Interval(
#         id="12hrinterval",
#         interval=43200000,
#         n_intervals=0
#     ),
#     dcc.Graph(id="State", figure=graph.draw_total_state_map(df)),
#     dcc.Interval(
#         id="12hrinterval_state",
#         interval=43200000,
#         n_intervals=0
#     ),
#     dcc.Graph(id="Total_US", figure=graph.draw_graph_daily_increase(total_data)),
#     dcc.Interval(
#         id="12hrinterval_total",
#         interval=43200000,
#         n_intervals=0
#     ),
#     dcc.Graph(id="State_wise", figure=graph.draw_graph_state_wise(states, state_wise_data)),
#     dcc.Interval(
#         id="12hrinterval_statewise",
#         interval=43200000,
#         n_intervals=0
#     )
# ])

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
    states, data = forecast_state_wise(grouped_df, daily_df)
    fig = graph.draw_graph_total_increase(data)
    return fig


if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0')