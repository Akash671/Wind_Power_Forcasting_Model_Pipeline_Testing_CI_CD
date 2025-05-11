# wind_forecasting_pipeline.py

"""
Full End-to-End Wind Power Forecasting Pipeline
Includes: Data Preprocessing, Feature Engineering, Outlier Removal,
Model Training (Random Forest, XGBoost, LSTM), Evaluation, and Visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -------------------
# 1. Data Loading & Cleaning
# -------------------
df = pd.read_csv("data/Turbine_Data.csv")
df['DateTime'] = pd.to_datetime(df['Unnamed: 0'])
df.drop(['Unnamed: 0', 'WTG'], axis=1, inplace=True)
df['year'] = df['DateTime'].dt.year
df['month'] = df['DateTime'].dt.month
df['day'] = df['DateTime'].dt.day
df['hour'] = df['DateTime'].dt.hour
df['minute'] = df['DateTime'].dt.minute
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df.drop('DateTime', axis=1, inplace=True)
df.fillna(df.median(), inplace=True)

# -------------------
# 2. Feature Selection
# -------------------
selected_features = ['BearingShaftTemperature','GearboxBearingTemperature','GearboxOilTemperature', 
                     'GeneratorRPM','GeneratorWinding1Temperature','GeneratorWinding2Temperature', 
                     'HubTemperature','ReactivePower','RotorRPM','WindSpeed']
X = df[selected_features]
y = df['ActivePower']

# -------------------
# 3. Train-Test Split
# -------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------
# 4. Preprocessing
# -------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------
# 5. Random Forest Model
# -------------------
rf_model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

# -------------------
# 6. XGBoost Model
# -------------------
xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.8,
                             colsample_bytree=0.8, random_state=42)
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)

# -------------------
# 7. LSTM Model
# -------------------
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = lstm_model.fit(X_train_lstm, y_train, epochs=100, batch_size=32,
                         validation_split=0.2, callbacks=[early_stop], verbose=0)
y_pred_lstm = lstm_model.predict(X_test_lstm).flatten()

# -------------------
# 8. Evaluation Function
# -------------------
def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

evaluate_model("Random Forest", y_test, y_pred_rf)
evaluate_model("XGBoost", y_test, y_pred_xgb)
evaluate_model("LSTM", y_test, y_pred_lstm)

# -------------------
# 9. Plotting Results
# -------------------
def plot_predictions(y_true, y_pred, model_name):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.values[:100], label='Actual')
    plt.plot(y_pred[:100], label='Predicted')
    plt.title(f'{model_name} - Actual vs Predicted Wind Power')
    plt.xlabel('Sample Index')
    plt.ylabel('Wind Power Output')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_predictions(y_test, y_pred_rf, "Random Forest")
plot_predictions(y_test, y_pred_xgb, "XGBoost")
plot_predictions(y_test, y_pred_lstm, "LSTM")
