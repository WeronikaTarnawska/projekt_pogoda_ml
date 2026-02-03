import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.style.use('seaborn-v0_8-whitegrid')

FILENAME = 'wroclaw_ml_hourly.csv'

# 1. WCZYTANIE DANYCH
try:
    df = pd.read_csv(FILENAME, parse_dates=['time'], index_col='time')
    df = df.sort_index()
    print(f"Dane wejściowe: {df.shape}")
except FileNotFoundError:
    print(f"Brak pliku {FILENAME}")
    exit()

# 2. PRZYGOTOWANIE CELÓW
def get_targets(df_raw):
    temp = df_raw[['temp']].copy()

    # Definicja okresów
    mask_day = (temp.index.hour >= 6) & (temp.index.hour <= 17)
    mask_night = (temp.index.hour >= 18) | (temp.index.hour <= 5)

    # 1. Target Dnia (Średnia 06:00 - 17:00 przypisana do daty)
    target_day = temp.loc[mask_day, 'temp'].resample('D').mean()
    target_day.name = 'Target_Day'

    # 2. Target Nocy (Średnia 18:00 - 05:00 przypisana do daty wieczora)
    # Przesuwamy -6h, żeby noc 1/2 stycznia liczyła się do 1 stycznia
    temp_night_shifted = temp.loc[mask_night].copy()
    temp_night_shifted.index = temp_night_shifted.index - pd.Timedelta(hours=6)
    target_night = temp_night_shifted['temp'].resample('D').mean()
    target_night.name = 'Target_Night'

    return target_day, target_night

series_day_target, series_night_target = get_targets(df)

# 3. FEATURE ENGINEERING
def create_single_station_dataset(df_raw, snapshot_hour, target_series):
    print(f"Budowanie datasetu dla Snapshota: {snapshot_hour}:00 ---")

    # 1. Główny Snapshot (Moment startu prognozy)
    df_snap = df_raw[df_raw.index.hour == snapshot_hour].copy()
    df_snap.index = df_snap.index.normalize() # Data bez godziny

    # 2. Dane "Historyczne" (dla obliczenia trendów)
    # Pobieramy dane sprzed 3 godzin i sprzed 1 godziny
    # Uwaga: Musimy obsłużyć przejście przez północ (dla godziny 0-2),
    # ale zakładając snapshoty 06:00 i 18:00, wystarczy proste odjęcie godzin.

    lag_1h_idx = (snapshot_hour - 1)
    lag_3h_idx = (snapshot_hour - 3)

    df_lag1 = df_raw[df_raw.index.hour == lag_1h_idx].copy()
    df_lag3 = df_raw[df_raw.index.hour == lag_3h_idx].copy()

    # Normalizacja indeksów, żeby pasowały do df_snap
    df_lag1.index = df_lag1.index.normalize()
    df_lag3.index = df_lag3.index.normalize()

    # 3. Budowa Cech (X)
    X = pd.DataFrame(index=df_snap.index)

    # Stan Aktualny
    cols_current = ['temp', 'pres', 'rhum', 'wspd', 'cldc', 'prcp']
    for col in cols_current:
        if col in df_snap.columns:
            X[f'{col}_now'] = df_snap[col]

    # Fizyka: Trendy
    # Jak szybko zmienia się ciśnienie? (Ważne dla frontów)
    X['pres_trend_3h'] = df_snap['pres'] - df_lag3['pres']

    # Jak szybko zmienia się temperatura? (Dynamika nagrzewania/chłodzenia)
    X['temp_trend_1h'] = df_snap['temp'] - df_lag1['temp']
    X['temp_trend_3h'] = df_snap['temp'] - df_lag3['temp']

    # Punkt rosy (Dew Point)
    # Uproszczone przybliżenie: T - ((100 - RH)/5)
    X['dew_point_approx'] = X['temp_now'] - ((100 - X['rhum_now']) / 5)

    # --- Czas ---
    X['day_of_year_sin'] = np.sin(2 * np.pi * X.index.dayofyear / 365)
    X['day_of_year_cos'] = np.cos(2 * np.pi * X.index.dayofyear / 365)
    X['month'] = X.index.month

    # 4. Łączenie z Celem
    data = pd.concat([X, target_series], axis=1).dropna()

    return data

# DATASET 1: MODEL DNIA (Start 06:00)
data_day = create_single_station_dataset(df, snapshot_hour=6, target_series=series_day_target)

# DATASET 2: MODEL NOCY (Start 18:00)
data_night = create_single_station_dataset(df, snapshot_hour=18, target_series=series_night_target)


# 4. TRENING
def train_and_eval(data, target_col, model_name):
    test_days = 365
    train = data.iloc[:-test_days]
    test = data.iloc[-test_days:]

    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]

    model = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.02,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8, # Tutaj bierzemy więcej cech (0.8) niż w multi, bo mamy ich mało
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50
    )

    # Walidacja
    split = int(len(X_train) * 0.9)
    model.fit(
        X_train.iloc[:split], y_train.iloc[:split],
        eval_set=[(X_train.iloc[split:], y_train.iloc[split:])],
        verbose=0
    )

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"[{model_name}] RMSE: {rmse:.2f}°C | MAE: {mae:.2f}°C | R2: {r2:.4f}")

    return y_test, preds, model

print()
print("Wyniki")
y_day, p_day, m_day = train_and_eval(data_day, 'Target_Day', 'MODEL DNIA (Start 06:00)')
y_night, p_night, m_night = train_and_eval(data_night, 'Target_Night', 'MODEL NOCY (Start 18:00)')

# 5. WIZUALIZACJA I ANALIZA
fig, ax = plt.subplots(2, 1, figsize=(12, 10))

# Wykres Dnia
subset = 60
ax[0].plot(y_day.index[-subset:], y_day.values[-subset:], 'o-', color='orange', label='Rzeczywista (Dzień)', alpha=0.6)
ax[0].plot(y_day.index[-subset:], p_day[-subset:], 'x--', color='black', label='Prognoza Single')
ax[0].set_title('Prognoza Dnia (Dane tylko Wrocław 06:00)')
ax[0].legend()

# Wykres Nocy
ax[1].plot(y_night.index[-subset:], y_night.values[-subset:], 'o-', color='navy', label='Rzeczywista (Noc)', alpha=0.6)
ax[1].plot(y_night.index[-subset:], p_night[-subset:], 'x--', color='red', label='Prognoza Single')
ax[1].set_title('Prognoza Nocy (Dane tylko Wrocław 18:00)')
ax[1].legend()

plt.tight_layout()
plt.show()

# Importancja cech
plt.figure(figsize=(10,5))
xgb.plot_importance(m_night, max_num_features=12, importance_type='gain', title='Co decyduje o nocy (Single Station)?')
plt.show()
