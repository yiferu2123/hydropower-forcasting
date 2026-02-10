import os
import io
import base64
import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# Initialize Dash App
app = Dash(__name__)
server = app.server  # Expose server for gunicorn

# --- Data Configuration ---
DATA_DIR = "."  # Current directory
datasets = {
    "Gibe I": "Gibe1.csv",
    "Gibe III": "Gibe3.csv",
    "Koka": "Koka Plant.csv",
    "Tana Beles": "Tana_Beles.csv",
    "Tekeze": "Tekeze.csv",
    "Fincha": "fincha.csv",
}
DATE_COLS = ['Date_GC', 'Date_EC']
WEATHER_COLS = ['T2M', 'PRECTOTCORR', 'ALLSKY_SFC_SW_DWN', 'RH2M', 'WS2M']
POWER_PREFIXES = ['U', 'V', 'W', 'Max_ALoad', 'Min_ALoad', 'Auxiliary', 'Water_Level', 'Energy', 'Discharge']

# --- Helper Functions ---
def identify_columns(df):
    cols = df.columns.tolist()
    date_c = [c for c in cols if c in DATE_COLS]
    weather_c = [c for c in cols if c in WEATHER_COLS]
    power_c = [c for c in cols if any(c.startswith(p) for p in POWER_PREFIXES) and c not in weather_c]
    other_c = [c for c in cols if c not in date_c + weather_c + power_c and c not in ['Date_EC', 'Date']]
    return date_c, weather_c, power_c, other_c

def clean_numeric(x):
    if isinstance(x, str):
        x = x.replace(',', '').replace(' ', '')
    return pd.to_numeric(x, errors='coerce')

def load_raw_formatted(filepath):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        return None
    
    date_c, weather_c, power_c, other_c = identify_columns(df)
    
    if 'Date_GC' in df.columns:
        df['Date'] = pd.to_datetime(df['Date_GC'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        df.drop(columns=[c for c in DATE_COLS if c in df.columns], inplace=True)
    
    all_numeric = weather_c + power_c + other_c
    for col in all_numeric:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric)
            
    return df

def apply_cleaning(df_in):
    df = df_in.copy()
    date_c, weather_c, power_c, other_c = identify_columns(df)
    
    for col in weather_c:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df[col] = np.clip(df[col], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    for col in power_c + other_c:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            low = df[col].quantile(0.05)
            high = df[col].quantile(0.95)
            df[col] = np.clip(df[col], low, high)
            
    return df

# Load Data Once on Startup
data_store = {}
print("Loading data...")
for name, file in datasets.items():
    path = os.path.join(DATA_DIR, file)
    if os.path.exists(path):
        raw = load_raw_formatted(path)
        if raw is not None:
            cleaned = apply_cleaning(raw)
            data_store[name] = cleaned
            print(f"Loaded {name}")
    else:
        print(f"Warning: {file} NOT FOUND")

# --- Modeling Functions ---
def prepare_features_direct(df, target_col, feature_cols=None):
    df = df.copy()
    if feature_cols is None:
        feature_cols = [c for c in df.select_dtypes(include=np.number).columns if c != target_col]
        
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
         date_col = next((c for c in df.columns if 'date' in c.lower()), None)
         if date_col:
             df[date_col] = pd.to_datetime(df[date_col])
             df = df.set_index(date_col)
    
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['dayofyear'] = df.index.dayofyear
    df['year'] = df.index.year
    
    x_cols = feature_cols + ['month', 'day', 'dayofyear', 'year']
    df[x_cols] = df[x_cols].fillna(df[x_cols].median())
    df = df.dropna(subset=[target_col])
    
    X = df[x_cols].values
    y = df[target_col].values
    
    return X, y, x_cols, df

def create_future_dataframe(df_historical, feature_cols, forecast_years):
    last_date = df_historical.index[-1]
    start_date = last_date + datetime.timedelta(days=1)
    days_to_predict = int(forecast_years * 365)
    end_date = start_date + datetime.timedelta(days=days_to_predict)
    
    future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    future_df = pd.DataFrame(index=future_dates)
    
    future_df['month'] = future_df.index.month
    future_df['day'] = future_df.index.day
    future_df['dayofyear'] = future_df.index.dayofyear
    future_df['year'] = future_df.index.year
    
    df_historical['month'] = df_historical.index.month
    monthly_stats = df_historical.groupby('month')[feature_cols].median().to_dict('index')
    
    for col in feature_cols:
        if col in df_historical.columns:
            future_df[col] = future_df['month'].map(lambda x: monthly_stats.get(x, {}).get(col, 0))
        else:
            future_df[col] = 0
            
    x_cols = feature_cols + ['month', 'day', 'dayofyear', 'year']
    return future_df[x_cols].values, future_dates

def build_dl_regressor(model_type, input_dim):
    model = Sequential()
    if model_type == 'LSTM':
        model.add(LSTM(64, input_shape=(1, input_dim), return_sequences=False))
    elif model_type == 'GRU':
        model.add(GRU(64, input_shape=(1, input_dim), return_sequences=False))
        
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1)) 
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def train_predict_evaluate(df, target_col, selected_model, forecast_years, xgb_params=None):
    if xgb_params is None:
        xgb_params = dict(
            n_estimators=100, learning_rate=0.03, max_depth=6,
            subsample=0.9, colsample_bytree=0.9,
            objective='reg:squarederror', n_jobs=-1, random_state=42
        )

    if target_col not in df.columns:
        return None, None, None, None, f"Target {target_col} not found!"
        
    feature_cols = [c for c in df.select_dtypes(include=np.number).columns if c != target_col]
    X, y, _, df_processed = prepare_features_direct(df, target_col, feature_cols)
    
    scaler_x = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(X)
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Simple Train/Test Split
    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
    
    models = {}
    history = {}
    
    if selected_model in ['Ensemble', 'Weighted_Avg']:
        types_to_train = ['LSTM', 'GRU', 'XGBoost']
    else:
        types_to_train = [selected_model]
        
    input_dim = X_train.shape[1]
    
    for m_type in types_to_train:
        if m_type == 'XGBoost':
            model = xgb.XGBRegressor(**xgb_params)
            model.fit(X_train, y_train)
            models[m_type] = model
            history[m_type] = {} 
        elif m_type in ('LSTM', 'GRU'):
            X_train_dl = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_test_dl = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
            
            model = build_dl_regressor(m_type, input_dim)
            # Reduced epochs/patience for web responsiveness
            es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=0)
            
            hist = model.fit(
                X_train_dl, y_train,
                validation_data=(X_test_dl, y_test),
                epochs=15, 
                batch_size=32, 
                verbose=0,
                callbacks=[es]
            )
            models[m_type] = model
            history[m_type] = hist.history

    # Forecasting Logic
    X_future, _ = create_future_dataframe(df_processed, feature_cols, forecast_years)
    X_future_scaled = scaler_x.transform(X_future)
    
    future_preds = []
    model_names = list(models.keys())
    
    # Calculate Weights if needed
    weights = {name: 1.0/len(model_names) for name in model_names}
    if selected_model == 'Weighted_Avg':
        errors = {}
        for name, m in models.items():
            if name == 'XGBoost':
                p = m.predict(X_test)
            else:
                X_test_dl = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
                p = m.predict(X_test_dl, verbose=0).flatten()
            errors[name] = np.sqrt(mean_squared_error(y_test, p))
        inv = {k: 1.0 / (v + 1e-6) for k, v in errors.items()}
        total = sum(inv.values())
        weights = {k: v / total for k, v in inv.items()}
    
    # Predict Future
    for name in model_names:
        m = models[name]
        if name == 'XGBoost':
            pred = m.predict(X_future_scaled)
        else:
            X_future_dl = X_future_scaled.reshape((X_future_scaled.shape[0], 1, X_future_scaled.shape[1]))
            pred = m.predict(X_future_dl, verbose=0).flatten()
        future_preds.append(pred)
    
    # Combine
    if selected_model == 'Weighted_Avg':
        final_scaled_pred = np.zeros_like(future_preds[0])
        for i, name in enumerate(model_names):
            final_scaled_pred += future_preds[i] * weights[name]
    elif selected_model == 'Ensemble':
        final_scaled_pred = np.mean(future_preds, axis=0)
    else:
        final_scaled_pred = future_preds[0]
        
    final_forecast = scaler_y.inverse_transform(final_scaled_pred.reshape(-1, 1)).flatten()
    return final_forecast, history, "Training Complete"

# --- Layout ---
app.layout = html.Div([
    html.H1("EEP Hydropower Forecasting Dashboard", style={'textAlign': 'center'}),
    
    html.Div([
        html.Label("Select Plant:"),
        dcc.Dropdown(
            id='plant-dropdown',
            options=[{'label': k, 'value': k} for k in data_store.keys()],
            value=list(data_store.keys())[0] if data_store else None
        ),
        
        html.Label("Select Target:"),
        dcc.Dropdown(id='target-dropdown'),
        
        html.Label("Select Model:"),
        dcc.Dropdown(
            id='model-dropdown',
            options=[
                {'label': 'LSTM', 'value': 'LSTM'},
                {'label': 'GRU', 'value': 'GRU'},
                {'label': 'XGBoost', 'value': 'XGBoost'},
                {'label': 'Ensemble', 'value': 'Ensemble'},
                {'label': 'Weighted Avg', 'value': 'Weighted_Avg'}
            ],
            value='Weighted_Avg'
        ),
        
        html.Label("Forecast Years:"),
        dcc.Slider(
            id='years-slider',
            min=1, max=10, step=1, value=4,
            marks={i: str(i) for i in range(1, 11)}
        ),
        
        html.Button("Run Forecast", id='run-btn', n_clicks=0, style={'marginTop': '20px', 'fontSize': '16px'}),
        html.Div(id='status-msg', style={'marginTop': '10px', 'color': 'blue'})
    ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
    
    html.Div([
        dcc.Loading(
            id="loading-1",
            type="default",
            children=[dcc.Graph(id='forecast-graph')]
        ),
    ], style={'width': '60%', 'display': 'inline-block', 'padding': '20px'})
])

# --- Callbacks ---
@app.callback(
    Output('target-dropdown', 'options'),
    Output('target-dropdown', 'value'),
    Input('plant-dropdown', 'value')
)
def update_targets(plant_name):
    if not plant_name or plant_name not in data_store:
        return [], None
    df = data_store[plant_name]
    # Filter for Energy and Water Level as requested
    options = [{'label': c, 'value': c} for c in ['Energy', 'Water_Level'] if c in df.columns]
    val = options[0]['value'] if options else None
    return options, val

@app.callback(
    Output('forecast-graph', 'figure'),
    Output('status-msg', 'children'),
    Input('run-btn', 'n_clicks'),
    State('plant-dropdown', 'value'),
    State('target-dropdown', 'value'),
    State('model-dropdown', 'value'),
    State('years-slider', 'value'),
    prevent_initial_call=True
)
def run_forecast(n_clicks, plant, target, model, years):
    if not plant or not target:
        return go.Figure(), "Please select plant and target."
    
    try:
        forecast, history, msg = train_predict_evaluate(data_store[plant], target, model, years)
    except Exception as e:
        return go.Figure(), f"Error: {str(e)}"
        
    # Plotting
    df = data_store[plant]
    last_date = df.index[-1]
    if not isinstance(last_date, pd.Timestamp):
         last_date = pd.to_datetime(last_date)

    future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, len(forecast) + 1)]
    
    hist_days = min(len(df), 3*365)
    hist_series = df[target].iloc[-hist_days:]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_series.index, y=hist_series.values, name='Historical', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=future_dates, y=forecast, name='Forecast', line=dict(color='red')))
    
    fig.update_layout(title=f"{model} Forecast: {target} ({years} Years)", template="plotly_white")
    
    return fig, "Forecast updated successfully!"

if __name__ == '__main__':
    app.run_server(debug=True)