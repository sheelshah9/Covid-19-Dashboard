import plotly.graph_objects as go
import datetime


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

    def draw_graph_daily_increase(self, data, y_max):
        layout = go.Layout(
            yaxis=dict(title='No. of cases reported daily'),
            title='Covid-19 Daily increase projection',
            showlegend=False)

        fig = go.Figure(data=data, layout=layout)

        # Add dropdown
        fig.update_layout(
            updatemenus=[
                dict(
                    active=0,
                    buttons=list([
                        dict(label="Arima",
                             method="update",
                             args=[{"visible": [True, True, True, False, False, False, False, False, False]}]
                             ),
                        dict(
                            label="XGBoost",
                            method="update",
                            args=[{"visible": [False, False, False, True, True, True, False, False, False]}]
                        ),
                        dict(
                            label="LSTM",
                            method="update",
                            args=[{"visible": [False, False, False, False, False, False, True, True, True]}]
                        )
                    ]),
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.15,
                    xanchor="left",
                    y=1.2,
                    yanchor="top"
                ),
            ]
        )

        # Add annotation
        fig.update_layout(
            annotations=[
                dict(text="Forecast Method:", showarrow=False,
                     x=0, y=1.13, yref="paper", align="left")
            ]
        )

        fig.add_shape(
            {"x0": datetime.date.today().strftime("%d, %b %Y"), "x1": datetime.date.today().strftime("%d, %b %Y"),
             "y0": 0, "y1": y_max,
             "type": "line", "line": {"width": 2, "dash": "dot"}})
        fig.add_trace(
            go.Scatter(x=[datetime.date.today().strftime("%d, %b %Y")], y=[y_max], text=["today"], mode="text",
                       line={"color": "green"}, showlegend=False))

        return fig


    def draw_total_state_map(self, df):
        grouped_df = df.drop(['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Country_Region', 'Lat', 'Long_'],
                             axis=1)
        grouped_df = grouped_df.groupby(['Province_State']).sum()
        grouped_df = grouped_df.drop(grouped_df.columns[:-1], axis=1)
        grouped_df = grouped_df.drop([x for x in grouped_df.index.tolist() if x not in Graphs.us_state_abbrev])
        # print(grouped_df)
        # print(grouped_df.loc[-1].astype(float).tolist())
        fig = go.Figure(data=go.Choropleth(
            locations=[Graphs.us_state_abbrev[x] for x in grouped_df.index.tolist()],  # Spatial coordinates
            z=grouped_df.iloc[:, -1].astype(float),  # Data to be color-coded
            locationmode='USA-states',  # set of locations match entries in `locations`
            colorscale='Reds',
            colorbar_title="Millions USD",
        ))

        fig.update_layout(
            title_text='COVID-19 cases by states',
            geo_scope='usa',  # limite map scope to USA
        )

        return fig

    #TODO
    def draw_graph_state_wise(self, states, data):
        layout = go.Layout(
            yaxis=dict(title='No. of cases reported daily - Statewise'),
            title='Covid-19 Daily increase projection - Statewise',
            showlegend=False)

        fig = go.Figure(data=data, layout=layout)

        # Dynamically generate dropdowns
        dropdowns = []
        visible_array = [False]*len(states)*3
        for i,s in enumerate(states):
            vis_arr = visible_array.copy()
            vis_arr[i*3:i*3+3] = [True]*3
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