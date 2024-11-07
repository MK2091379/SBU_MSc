# Import required libraries
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# Load and preprocess dataset (dummy data creation for illustration)
df = pd.read_csv('bitcoin.csv')

df.columns = ['date', 'open', 'high', 'low', 'close', 'adj close', 'volume']

df['date'] = pd.to_datetime(df['date'])
df['day'] = df['date'].dt.day.astype('int')
df['month'] = df['date'].dt.month.astype('int')
df['year'] = df['date'].dt.year.astype('int')
df['weekday'] = df['date'].dt.day_name()

# Initialize Dash app with Bootstrap
app = Dash(__name__, external_stylesheets=['https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css'])

# App layout
app.layout = html.Div(className='container', style={'backgroundColor': '#ffffff', 'padding': '20px'}, children=[
    html.H1("Bitcoin Price Analysis Dashboard", className='text-center', style={'font-family': 'Arial, sans-serif', 'color': '#2c3e50'}),
    
    # Date range selector
    dcc.DatePickerRange(
        id='date-picker',
        start_date=df['date'].min(),
        end_date=df['date'].max(),
        display_format='YYYY-MM-DD',
        style={'margin-bottom': '20px', 'border-radius': '5px'}
    ),

    # Dropdown for selecting price type
    dcc.Dropdown(
        id='price-type',
        options=[{'label': col.capitalize(), 'value': col} for col in ['open', 'high', 'low', 'close', 'adj close']],
        value='close',
        style={'width': '40%', 'margin-bottom': '20px', 'border-radius': '5px'}
    ),

    # Line plot for selected price type
    dcc.Graph(id='price-line-chart', style={'border': '1px solid #3498db'}),
    
    # Candlestick chart
    dcc.Graph(id='candlestick-chart', style={'border': '1px solid #3498db'}),
    
    # Volume chart
    dcc.Graph(id='volume-chart', style={'border': '1px solid #3498db'}),

    # Correlation heatmap
    dcc.Graph(id='correlation-heatmap', style={'border': '1px solid #3498db'}),
    
    # Technical Indicators with checkboxes beside MACD and Bollinger Bands graphs
    html.Div(className='row', children=[
        html.Div(className='col-md-6', children=[
            dcc.Graph(id='MACD-signal-chart', style={'border': '1px solid #3498db'}),
            dcc.Checklist(
                id='macd-checkbox',
                options=[
                    {'label': 'Show MACD', 'value': 'MACD'},
                    {'label': 'Show Signal Line', 'value': 'Signal Line'}
                ],
                value=['MACD', 'Signal Line'],  # Default selected values
                inline=True,
                style={'margin-top': '10px'}
            )
        ]),
        html.Div(className='col-md-6', children=[
            dcc.Graph(id='bollinger-bands-chart', style={'border': '1px solid #3498db'}),
            dcc.Checklist(
                id='bollinger-checkbox',
                options=[
                    {'label': 'Show Upper Band', 'value': 'Upper'},
                    {'label': 'Show Middle Band', 'value': 'Middle'},
                    {'label': 'Show Lower Band', 'value': 'Lower'}
                ],
                value=['Upper', 'Middle', 'Lower'],  # Default selected values
                inline=True,
                style={'margin-top': '10px'}
            )
        ]),
    ]),
])

# Callbacks for interactivity
@app.callback(
    [Output('price-line-chart', 'figure'),
     Output('candlestick-chart', 'figure'),
     Output('volume-chart', 'figure'),
     Output('correlation-heatmap', 'figure'),
     Output('MACD-signal-chart', 'figure'),
     Output('bollinger-bands-chart', 'figure')],
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('price-type', 'value'),
     Input('macd-checkbox', 'value'),
     Input('bollinger-checkbox', 'value')]
)
def update_dashboard(start_date, end_date, price_type, selected_macd, selected_bollinger):
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    # Line chart for selected price type
    line_chart = go.Figure([go.Scatter(x=filtered_df['date'], y=filtered_df[price_type], mode='lines', name=price_type, line=dict(color='#3498db'))])
    line_chart.update_layout(title=f'Price of {price_type.capitalize()} Over Time', xaxis_title='Date', yaxis_title='Price', template='plotly_white', plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font=dict(color='#2c3e50'))
    
    # Candlestick chart
    candlestick_chart = go.Figure(data=[go.Candlestick(
        x=filtered_df['date'],
        open=filtered_df['open'],
        high=filtered_df['high'],
        low=filtered_df['low'],
        close=filtered_df['close'],
        increasing_line_color='#27ae60',
        decreasing_line_color='#e74c3c')])
    candlestick_chart.update_layout(title='Candlestick Chart', xaxis_title='Date', yaxis_title='Price', template='plotly_white', plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font=dict(color='#2c3e50'))
    
    # Volume chart
    volume_chart = go.Figure([go.Bar(x=filtered_df['date'], y=filtered_df['volume'], name='Volume', marker_color='#3498db')])
    volume_chart.update_layout(title='Volume Over Time', xaxis_title='Date', yaxis_title='Volume', template='plotly_white', plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font=dict(color='#2c3e50'))
    
    # Correlation heatmap
    corr = filtered_df[['open', 'high', 'low', 'close', 'adj close', 'volume']].corr()
    heatmap = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='Blues'))
    heatmap.update_layout(title='Correlation Heatmap', xaxis_title='Features', yaxis_title='Features', template='plotly_white', plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font=dict(color='#2c3e50'))
    
    # MACD calculation
    filtered_df['MACD'] = filtered_df['close'].ewm(span=12, adjust=False).mean() - filtered_df['close'].ewm(span=26, adjust=False).mean()
    filtered_df['Signal Line'] = filtered_df['MACD'].ewm(span=9, adjust=False).mean()
    
    MACD_Signal_chart = go.Figure()
    if 'MACD' in selected_macd:
        MACD_Signal_chart.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['MACD'], mode='lines', name='MACD', line=dict(color='#3498db')))
    if 'Signal Line' in selected_macd:
        MACD_Signal_chart.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['Signal Line'], mode='lines', name='Signal Line', line=dict(color='#c0392b')))
    
    MACD_Signal_chart.update_layout(title='MACD and Signal Line', xaxis_title='Date', yaxis_title='MACD Value', template='plotly_white', plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font=dict(color='#2c3e50'))
    
    # Bollinger Bands calculation
    filtered_df['MA20'] = filtered_df[price_type].rolling(window=20).mean()
    filtered_df['Upper'] = filtered_df['MA20'] + 2 * filtered_df[price_type].rolling(window=20).std()
    filtered_df['Lower'] = filtered_df['MA20'] - 2 * filtered_df[price_type].rolling(window=20).std()
    
    bollinger_bands_chart = go.Figure()
    if 'Upper' in selected_bollinger:
        bollinger_bands_chart.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['Upper'], mode='lines', name='Upper Band', line=dict(color='#27ae60')))
    if 'Middle' in selected_bollinger:
        bollinger_bands_chart.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['MA20'], mode='lines', name='Middle Band', line=dict(color='#3498db')))
    if 'Lower' in selected_bollinger:
        bollinger_bands_chart.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['Lower'], mode='lines', name='Lower Band', line=dict(color='#e74c3c')))
    
    bollinger_bands_chart.update_layout(title='Bollinger Bands', xaxis_title='Date', yaxis_title='Price', template='plotly_white', plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font=dict(color='#2c3e50'))
    
    return line_chart, candlestick_chart, volume_chart, heatmap, MACD_Signal_chart, bollinger_bands_chart

# Run the app
if __name__ == '__main__':
    app.run_server()
