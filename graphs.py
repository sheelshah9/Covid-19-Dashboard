import plotly.graph_objects as go
import datetime
import dash_html_components as html
import dash_bootstrap_components as dbc


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

    def __int__(self):
        pass

    @staticmethod
    def draw_graph_daily_increase(df):
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
        fig.add_trace(go.Bar(x=df.index, y=df['Total'], name='Actual Cases', marker_color='red'))

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
        # print(grouped_df)
        # print(grouped_df.loc[-1].astype(float).tolist())
        fig = go.Figure(data=go.Choropleth(
            locations=[Graphs.us_state_abbrev[x] for x in grouped_df.columns.tolist()],  # Spatial coordinates
            z=grouped_df.iloc[-1, :].astype(float),  # Data to be color-coded
            locationmode='USA-states',  # set of locations match entries in `locations`
            colorscale='Reds',
            colorbar_title="Millions USD",
        ))

        fig.update_layout(
            title_text='COVID-19 cases by states',
            geo_scope='usa',  # limite map scope to USA
        )

        return fig

    # TODO
    def draw_graph_state_wise(self, states, data):
        layout = go.Layout(
            yaxis=dict(title='No. of cases reported daily - Statewise'),
            title='Covid-19 Daily increase projection - Statewise',
            showlegend=False)

        fig = go.Figure(data=data, layout=layout)

        # Dynamically generate dropdowns
        dropdowns = []
        visible_array = [False] * len(states) * 3
        for i, s in enumerate(states):
            vis_arr = visible_array.copy()
            vis_arr[i * 3:i * 3 + 3] = [True] * 3
            temp = dict(label=s,
                        method="update",
                        args=[{"visible": vis_arr}]
                        )
            dropdowns.append(temp)

        # Add dropdown
        fig.update_layout(
            updatemenus=[
                dict(
                    active=0,
                    buttons=list(dropdowns),
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.09,
                    xanchor="left",
                    y=1.2,
                    yanchor="top"
                ),
            ]
        )

        # Add annotation
        fig.update_layout(
            annotations=[
                dict(text="State:", showarrow=False,
                     x=0, y=1.13, yref="paper", align="left")
            ], plot_bgcolor='rgb(255,255,255)'
        )

        return fig

    def draw_panel(self, df):
        total_cases = df.sum()
        new_case_today = df.iloc[-1]
        peak_date, peak_cases = df.argmax(), df.max()
        panel = html.Div([
            html.H4(df.columns.tolist()),
            dbc.Card(body=True, className="text-white bg-primary", children=[
                html.H6("Total Cases:", style={'color':'white'})

            ])
        ])