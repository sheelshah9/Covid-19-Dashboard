import plotly.graph_objects as go
import datetime
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.express as px
from urllib.request import urlopen
import json


class Graphs:
    us_state_abbrev = {
        'Alabama': 'AL',
        'Alaska': 'AK',
        'American Samoa': 'AS',
        'Arizona': 'AZ',
        'Arkansas': 'AR',
        'California': 'CA',
        'Colorado': 'CO',
        'Connecticut': 'CT',
        'Delaware': 'DE',
        'District of Columbia': 'DC',
        'Florida': 'FL',
        'Georgia': 'GA',
        'Guam': 'GU',
        'Hawaii': 'HI',
        'Idaho': 'ID',
        'Illinois': 'IL',
        'Indiana': 'IN',
        'Iowa': 'IA',
        'Kansas': 'KS',
        'Kentucky': 'KY',
        'Louisiana': 'LA',
        'Maine': 'ME',
        'Maryland': 'MD',
        'Massachusetts': 'MA',
        'Michigan': 'MI',
        'Minnesota': 'MN',
        'Mississippi': 'MS',
        'Missouri': 'MO',
        'Montana': 'MT',
        'Nebraska': 'NE',
        'Nevada': 'NV',
        'New Hampshire': 'NH',
        'New Jersey': 'NJ',
        'New Mexico': 'NM',
        'New York': 'NY',
        'North Carolina': 'NC',
        'North Dakota': 'ND',
        'Northern Mariana Islands': 'MP',
        'Ohio': 'OH',
        'Oklahoma': 'OK',
        'Oregon': 'OR',
        'Pennsylvania': 'PA',
        'Puerto Rico': 'PR',
        'Rhode Island': 'RI',
        'South Carolina': 'SC',
        'South Dakota': 'SD',
        'Tennessee': 'TN',
        'Texas': 'TX',
        'Utah': 'UT',
        'Vermont': 'VT',
        'Virgin Islands': 'VI',
        'Virginia': 'VA',
        'Washington': 'WA',
        'West Virginia': 'WV',
        'Wisconsin': 'WI',
        'Wyoming': 'WY'
    }

    def __init__(self):
        with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
            self.counties = json.load(response)
        self.fips = set()
        for feat in self.counties["features"]:
            prop = feat["properties"]
            self.fips.add(prop["COUNTY"])

    @staticmethod
    def draw_graph(df, row='Total'):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            name="Lower Bound",
            x=df.index,
            y=df['interval_low'],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines'
        ))
        fig.add_trace(go.Scatter(
            name="Prediction",
            x=df.index,
            y=df['forecast'],
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty'
        ))
        fig.add_trace(go.Scatter(
            name="Upper Bound",
            x=df.index,
            y=df['interval_high'],
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty'
        ))
        fig.add_trace(go.Bar(x=df.index, y=df[row], name='Actual Cases', marker_color='red'))

        fig.add_shape({"x0": datetime.date.today(), "x1": datetime.date.today(), "y0": 0, "y1": df["forecast"].max(),
                       "type": "line", "line": {"width": 2, "dash": "dot"}})
        fig.add_trace(
            go.Scatter(x=[datetime.date.today()], y=[df["forecast"].max()], text=["today"], mode="text",
                       line={"color": "green"},
                       showlegend=False))

        return fig

    @staticmethod
    def draw_total_state_map(df):
        grouped_df = df.drop([x for x in df.columns.tolist() if x not in Graphs.us_state_abbrev], axis=1)
        fig = go.Figure(data=go.Choropleth(
            locations=[Graphs.us_state_abbrev[x] for x in grouped_df.columns.tolist()],  # Spatial coordinates
            z=grouped_df.iloc[-1, :].astype(float),  # Data to be color-coded
            locationmode='USA-states',  # set of locations match entries in `locations`
            colorscale='Reds',
            colorbar_title="Cases",
        ))

        fig.update_layout(
            title_text='COVID-19 cases by states',
            geo_scope='usa',  # limite map scope to USA
        )

        return fig

    @staticmethod
    def draw_panel(df, row, type):
        df = df.dropna()
        total_cases = df.sum()
        new_case_today = df.iloc[-1]
        peak_date, peak_cases = df.idxmax(), df.max()
        panel = html.Div([
            html.H4(row),
            dbc.Card(body=True, className="text-white bg-primary", children=[
                html.H6("Total {}:".format(type), style={'color': 'white'}),
                html.H3("{:,.0f}".format(total_cases), className='text-danger'),
                html.H6("New {} Today:".format(type), style={'color': 'white'}),
                html.H3("{:,.0f}".format(new_case_today), className='text-danger'),
                html.H6("Peak Day:", style={'color': 'white'}),
                html.H3(peak_date, className='text-danger'),
                html.H6("With {:,.0f} {}".format(peak_cases, type), style={'color': 'white'})
            ])
        ])
        return panel

    def draw_statewise_map(self, df, row):
        df = df.iloc[:, [4, 5, 6, -1]]
        df = df[df["Province_State"] == row]
        df.dropna(inplace=True)
        df["FIPS"] = df.FIPS.map(int).map("{:05}".format)
        df.rename(columns={df.columns[-1]: "Cases"}, inplace=True)
        fig = px.choropleth(df, geojson=self.counties, locations="FIPS", color=df.columns[-1],
                            projection="mercator", color_continuous_scale="Reds", hover_name=df.columns[1],
                            hover_data={
                                'FIPS': False,
                                df.columns[-1]: True
                            }
                            )
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(margin={"r": 0, "l": 0, "b": 0, "t": 0})
        return fig
