from regressors import Regressor
import plotly.graph_objects as go
from graphs import Graphs
from helper import Data
import pandas as pd
import pickle

pred_path = "data/"

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash
from dash.dependencies import Input, Output

d = Data()

d.fetch_data()
pre_df = d.preprocess_data(d.df_us_cases)
df = d.daily_data(pre_df)
daily_reg = Regressor(df, 7)
arima_data = daily_reg.ARIMA()
total_reg = Regressor(pre_df, 7)
total_arima_data = total_reg.ARIMA()
g = Graphs.draw_graph_daily_increase(arima_data)
# total_data, y_max_total = forecast_total_cases(grouped_df, daily_df)
# states, state_wise_data = forecast_state_wise(grouped_df, daily_df)
# graph = Graphs()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
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
            dbc.Col(html.H4("Forecast 10 days from today"), width={"size":6, "offset":3}),
            dbc.Tabs([
                dbc.Tab([dcc.Graph(id="graph_daily_increase_US", figure=Graphs.draw_graph_daily_increase(arima_data)),
                         dcc.Graph(id="Total_US", figure=Graphs.draw_graph_daily_increase(total_arima_data))],
                        label="US projected cases", ),
                dbc.Tab([dcc.Graph(id="State_wise", figure=Graphs.draw_graph_daily_increase(arima_data)),
                         dcc.Graph(id="State_map", figure=Graphs.draw_total_state_map(pre_df))],
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



if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0')