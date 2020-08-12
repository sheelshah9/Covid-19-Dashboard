# Covid-19-Dashboard

A COVID-19 dashboard which forecasts cases and deaths for each state in the US. The dashboard also provides with a county-wise map for each state. The data and predictions are updated every day when the data at [JHU CSSE COVID-19 Data](https://github.com/CSSEGISandData/COVID-19) is updated.  
Project website: http://www.sheel.ml

## Forecast Model
The dashboard currently uses three different types of models: ARIMA, XGBoost and LSTM.

## Features
#### 1) Cases  
The graph shows the original number of cases reported (as a bar graph of orange color) and the number of cases predicted by different methods (as a line graph of blue color with margin of error).
![Alt Text](https://github.com/sheelshah9/Covid-19-Dashboard/blob/master/images/cases_graph.gif)

#### 2) Deaths  
The graph shows the original number of deaths reported (as a bar graph of orange color) and the number of deaths predicted by different methods (as a line graph of blue color with margin of error).
![Alt Text](https://github.com/sheelshah9/Covid-19-Dashboard/blob/master/images/deaths_graph.gif)

#### 3) Panel  
The panel shows the total number of cases/deaths, daily cases/deaths, peak date, and peak cases/deaths for each state.

#### 4) Map  
The map shows county-wise cases for each state.
![Alt Text](https://github.com/sheelshah9/Covid-19-Dashboard/blob/master/images/maps.gif)
