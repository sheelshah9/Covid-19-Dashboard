from regressors import Regressor
import plotly.graph_objects as go
from graphs import Graphs
from helper import Data
import pandas as pd
import pickle

pred_path = "data/"
models = ['ARIMA', 'XGBoost', 'LSTM']

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash
from dash.dependencies import Input, Output

d = Data()

d.fetch_data()
pre_df = d.preprocess_data(d.df_us_cases)
# df = d.daily_data(pre_df)
# daily_reg = Regressor(df, 7)
# arima_data = daily_reg.ARIMA()
# total_reg = Regressor(pre_df, 7)
# total_arima_data = total_reg.ARIMA()
# g = Graphs.draw_graph(arima_data)
# total_data, y_max_total = forecast_total_cases(grouped_df, daily_df)
# states, state_wise_data = forecast_state_wise(grouped_df, daily_df)
# graph = Graphs()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server

navbar = dbc.Nav(className="nav nav-pills", children=[
    dbc.NavItem(dbc.NavLink([html.I(className="fa fa-github"), "  GitHub"], href="https://github.com/sheelshah9", active=True, target="_blank")),
    dbc.NavItem(dbc.NavLink([html.I(className="fa fa-linkedin"), "  LinkedIn"], href="https://www.linkedin.com/in/sheelshah09/", active=True, target="_blank"))
])

dropdown_state = dbc.FormGroup([
    html.H4("Select State"),
    dcc.Dropdown(id="state", options=[{'label':x, 'value':x} for x in pre_df.columns.tolist()], value='Total')
])

dropdown_model = dbc.FormGroup([
    html.H4("Select Forecast Method"),
    dcc.Dropdown(id="method", options=[{'label':x, 'value':x} for x in models], value='ARIMA')
])

app.layout = dbc.Container(fluid=True, children=[
    #Header
    html.Br(),
    dbc.Row([html.H1("Covid-19 Dashboard",id="nav-pills")], justify="center", align="center", className="h-50", style={"height": "100h"}),
    navbar,
    html.Br(),html.Br(),

    #Body
    dbc.Row([
            dbc.Col(md=3, children=[
                dropdown_state,
                html.Br(), html.Br(),
                dropdown_model,
                html.Br(), html.Br(),
                html.Div(id='out-panel')
            ]),
            dbc.Col(md=9, children=[
            dbc.Col(html.H4("Forecast 10 days from today"), width={"size":6, "offset":3}),
            dbc.Tabs([
                dbc.Tab([dcc.Graph(id="graph_daily_cases"),
                         dcc.Graph(id="graph_total_cases")
                         ],
                        label="Projected Cases", ),
                dbc.Tab([dcc.Graph(id="graph_daily_deaths"),
                         # dcc.Graph(id="graph_total_deaths")
                         ],
                        label="Projected Deaths", ),
                dbc.Tab([dcc.Graph(id="State_map", figure=Graphs.draw_total_state_map(pre_df))],
                        label="State Maps")
                    ])
                ])
            ])
]   )

@app.callback(output=Output("graph_daily_cases","figure"), inputs=[Input("state","value"), Input("method","value")])
def plot_cases(state, method):
    data = pd.read_csv(pred_path+"daily_{}_{}.csv".format(state, method), index_col=0)
    return Graphs.draw_graph(data, row=state)

@app.callback(output=Output("graph_total_cases","figure"), inputs=[Input("state","value"), Input("method","value")])
def plot_cases(state, method):
    data = pd.read_csv(pred_path+"daily_{}_{}.csv".format(state, method), index_col=0)
    data = data.cumsum()
    return Graphs.draw_graph(data, row=state)

@app.callback(output=Output("out-panel", "children"), inputs=[Input("state", "value"), Input("method", "value")])
def render_panel(state, method):
    data = pd.read_csv(pred_path + "daily_{}_{}.csv".format(state, method), index_col=0)
    return Graphs.draw_panel(data[state], state)

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