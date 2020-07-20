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

data = Data()

data.fetch_data()
preprocessed_df = data.preprocess_cases_data(data.df_us_cases)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server

navbar = dbc.Nav(className="nav nav-pills", children=[
    dbc.NavItem(dbc.NavLink([html.I(className="fa fa-github"), "  GitHub"], href="https://github.com/sheelshah9", active=True, target="_blank")),
    dbc.NavItem(dbc.NavLink([html.I(className="fa fa-linkedin"), "  LinkedIn"], href="https://www.linkedin.com/in/sheelshah09/", active=True, target="_blank"))
])

dropdown_state = dbc.FormGroup([
    html.H4("Select State"),
    dcc.Dropdown(id="state", options=[{'label':x, 'value':x} for x in preprocessed_df.columns.tolist()], value='Total')
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
            dbc.Col(html.H4("Forecast 7 days from today"), width={"size":6, "offset":3}),
            dbc.Tabs([
                dbc.Tab([dcc.Graph(id="graph_daily_cases"),
                         dcc.Graph(id="graph_total_cases")
                         ],
                        label="Projected Cases", ),
                dbc.Tab([dcc.Graph(id="graph_daily_deaths"),
                         dcc.Graph(id="graph_total_deaths")
                         ],
                        label="Projected Deaths", ),
                dbc.Tab([dcc.Graph(id="State_map", figure=Graphs.draw_total_state_map(preprocessed_df))],
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
    data = pd.Series(data[state], index=data.index)
    return Graphs.draw_panel(data, state)

@app.callback(output=Output("graph_daily_deaths","figure"), inputs=[Input("state","value"), Input("method","value")])
def plot_cases(state, method):
    data = pd.read_csv(pred_path+"death_{}_{}.csv".format(state, method), index_col=0)
    return Graphs.draw_graph(data, row=state)

@app.callback(output=Output("graph_total_deaths","figure"), inputs=[Input("state","value"), Input("method","value")])
def plot_cases(state, method):
    data = pd.read_csv(pred_path+"death_{}_{}.csv".format(state, method), index_col=0)
    data = data.cumsum()
    return Graphs.draw_graph(data, row=state)


if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0')