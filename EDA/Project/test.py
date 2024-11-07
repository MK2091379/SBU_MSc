# Import required libraries
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# Load and preprocess dataset (dummy data creation for illustration)
df = pd.read_csv('bitcoin.csv')

df.columns = ['date','open','high','low','close','adj close','volume']

df['date'] = pd.to_datetime(df['date'])
df['day'] = df['date'].dt.day.astype('int')
df['month'] = df['date'].dt.month.astype('int')
df['year'] = df['date'].dt.year.astype('int')
df['weekday'] = df['date'].dt.day_name()

# Initialize Dash app
app = Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("Bitcoin Price Analysis Dashboard", style={'text-align': 'center'}),
    
    # Date range selector
    dcc.DatePickerRange(
        id='date-picker',
        start_date=df['date'].min(),
        end_date=df['date'].max(),
        display_format='YYYY-MM-DD'
    ),

    # Dropdown for selecting price type
    dcc.Dropdown(
        id='price-type',
        options=[{'label': col.capitalize(), 'value': col} for col in ['open', 'high', 'low', 'close', 'adj close']],
        value='close',
        style={'width': '40%'}
    ),
    
    # Line plot for selected price type
    dcc.Graph(id='price-line-chart'),
    
    # Candlestick chart
    dcc.Graph(id='candlestick-chart'),
    
    # Volume chart
    dcc.Graph(id='volume-chart'),

    # Correlation heatmap
    dcc.Graph(id='correlation-heatmap'),
    
    # Technical Indicators - Moving Average, RSI, Bollinger Bands
    html.Div([
        dcc.Graph(id='moving-average-chart'),
        dcc.Graph(id='rsi-chart'),
        dcc.Graph(id='bollinger-bands-chart')
    ])
])

# Callbacks for interactivity
@app.callback(
    [Output('price-line-chart', 'figure'),
     Output('candlestick-chart', 'figure'),
     Output('volume-chart', 'figure'),
     Output('correlation-heatmap', 'figure'),
     Output('moving-average-chart', 'figure'),
     Output('rsi-chart', 'figure'),
     Output('bollinger-bands-chart', 'figure')],
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('price-type', 'value')]
)
def update_dashboard(start_date, end_date, price_type):
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    # Line chart for selected price type
    line_chart = go.Figure([go.Scatter(x=filtered_df['date'], y=filtered_df[price_type], mode='lines', name=price_type)])
    
    # Candlestick chart
    candlestick_chart = go.Figure(data=[go.Candlestick(
        x=filtered_df['date'],
        open=filtered_df['open'],
        high=filtered_df['high'],
        low=filtered_df['low'],
        close=filtered_df['close'])])
    
    # Volume chart
    volume_chart = go.Figure([go.Bar(x=filtered_df['date'], y=filtered_df['volume'], name='Volume')])
    
    # Correlation heatmap
    corr = filtered_df[['open', 'high', 'low', 'close', 'adj close', 'volume']].corr()
    heatmap = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns))
    
    # Moving average
    filtered_df['SMA'] = filtered_df[price_type].rolling(window=20).mean()
    moving_avg_chart = go.Figure([go.Scatter(x=filtered_df['date'], y=filtered_df['SMA'], mode='lines', name='SMA')])
    
    # RSI calculation
    delta = filtered_df['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_chart = go.Figure([go.Scatter(x=filtered_df['date'], y=rsi, mode='lines', name='RSI')])
    
    # Bollinger Bands
    filtered_df['MA20'] = filtered_df[price_type].rolling(window=20).mean()
    filtered_df['Upper'] = filtered_df['MA20'] + 2 * filtered_df[price_type].rolling(window=20).std()
    filtered_df['Lower'] = filtered_df['MA20'] - 2 * filtered_df[price_type].rolling(window=20).std()
    bollinger_bands_chart = go.Figure([
        go.Scatter(x=filtered_df['date'], y=filtered_df['Upper'], mode='lines', name='Upper Band'),
        go.Scatter(x=filtered_df['date'], y=filtered_df['MA20'], mode='lines', name='Middle Band'),
        go.Scatter(x=filtered_df['date'], y=filtered_df['Lower'], mode='lines', name='Lower Band')
    ])
    
    return line_chart, candlestick_chart, volume_chart, heatmap, moving_avg_chart, rsi_chart, bollinger_bands_chart

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
