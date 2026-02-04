import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.style.use('seaborn-v0_8-whitegrid')

FILENAME = 'wroclaw_multi_station_full.csv'

# ==========================================
# 1. KONFIGURACJA I DANE
# ==========================================
try:
    df = pd.read_csv(FILENAME, parse_dates=['time'], index_col='time')
    df = df.sort_index()
    print(f"Wczytano dane: {df.shape}")
except FileNotFoundError:
    print(f"Brak pliku {FILENAME}")
    exit()

# ==========================================
# 2. PRZYGOTOWANIE TARGEÓW (CELÓW)
# ==========================================
def get_targets(df_raw):
    temp = df_raw[['temp']].copy()

    # Definicja okresów
    # Dzień: 06:00 - 17:00 (12h)
    # Noc: 18:00 - 05:00 (12h)

    mask_day = (temp.index.hour >= 6) & (temp.index.hour <= 17)
    mask_night = (temp.index.hour >= 18) | (temp.index.hour <= 5)

    # 1. Target Dnia (Przypisany do daty kalendarzowej)
    target_day = temp.loc[mask_day, 'temp'].resample('D').mean()
    target_day.name = 'Target_Day'

    # 2. Target Nocy (Przypisany do daty "wieczora")
    # Trik: Noc z 1 na 2 stycznia chcemy przypisać do 1 stycznia (bo wtedy robimy prognozę)
    # Przesuwamy czas o -6h. Wtedy 18:00 (1.01) -> 12:00 (1.01), a 05:00 (2.01) -> 23:00 (1.01)
    # Wszystko wpada w 1 stycznia.
    temp_night_shifted = temp.loc[mask_night].copy()
    temp_night_shifted.index = temp_night_shifted.index - pd.Timedelta(hours=6)
    target_night = temp_night_shifted['temp'].resample('D').mean()
    target_night.name = 'Target_Night'

    return target_day, target_night

series_day_target, series_night_target = get_targets(df)

# ==========================================
# 3. FUNKCJA BUDUJĄCA DATASET (SNAPSHOT)
# ==========================================
def create_specialized_dataset(df_raw, snapshot_hour, target_series, neighbor_cities):
    """
    Tworzy dataset, gdzie cechami jest stan pogody o konkretnej godzinie (snapshot),
    a celem jest średnia temperatura nadchodzącego okresu.
    """
    print(f"\n--- Budowanie datasetu dla Snapshota: {snapshot_hour}:00 ---")

    # 1. Pobieramy tylko wiersze z godziny snapshota
    df_snap = df_raw[df_raw.index.hour == snapshot_hour].copy()

    # Normalizujemy indeks do północy, żeby móc połączyć z targetami po dacie
    df_snap.index = df_snap.index.normalize()

    # 2. Budowa cech (Features)
    X = pd.DataFrame(index=df_snap.index)

    # Baza (Wrocław)
    X['temp_start'] = df_snap['temp']
    X['pres_start'] = df_snap['pres']
    X['rhum_start'] = df_snap['rhum']
    X['wspd_start'] = df_snap['wspd']

    # Trend ciśnienia (Snapshot - 6h wcześniej)
    # Np. dla 06:00 sprawdzamy ciśnienie o 00:00
    prev_hour = (snapshot_hour - 6) % 24
    df_prev = df_raw[df_raw.index.hour == prev_hour].copy()
    df_prev.index = df_prev.index.normalize()
    # Musimy uważać na daty przy przejściu przez północ, ale normalize() i join załatwi sprawę
    # (Dla uproszczenia w tym kodzie pomijam idealne dopasowanie -6h jeśli zmienia się data,
    #  zostajemy przy prostych cechach snapshotowych i gradientach)

    # Sąsiedzi (Gradienty przestrzenne) - KLUCZOWE
    for city in neighbor_cities:
        col_t = f'temp_{city}'
        col_p = f'pres_{city}'
        if col_t in df_snap.columns:
            X[f'diff_temp_{city}'] = df_snap[col_t] - df_snap['temp'] # Czy tam jest cieplej?
        if col_p in df_snap.columns:
            X[f'diff_pres_{city}'] = df_snap[col_p] - df_snap['pres'] # Skąd wieje wiatr (geostroficzny)?

    # Czas
    X['day_of_year_sin'] = np.sin(2 * np.pi * X.index.dayofyear / 365)
    X['day_of_year_cos'] = np.cos(2 * np.pi * X.index.dayofyear / 365)

    # 3. Łączenie z Celem
    data = pd.concat([X, target_series], axis=1).dropna()

    return data

cities = ['Legnica', 'Opole', 'Poznan']

# === DATASET 1: PORANNY (Prognoza Dnia) ===
# Snapshot o 06:00 rano -> Przewidujemy średnią z 06:00-18:00
data_day = create_specialized_dataset(df, snapshot_hour=6, target_series=series_day_target, neighbor_cities=cities)

# === DATASET 2: WIECZORNY (Prognoza Nocy) ===
# Snapshot o 18:00 wieczorem -> Przewidujemy średnią z 18:00-06:00
data_night = create_specialized_dataset(df, snapshot_hour=18, target_series=series_night_target, neighbor_cities=cities)


# ==========================================
# 4. TRENING I WYNIKI
# ==========================================
def train_and_eval(data, target_col, model_name):
    # Podział
    test_days = 365
    train = data.iloc[:-test_days]
    test = data.iloc[-test_days:]

    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]

    # Model (Używam parametrów uniwersalnych, możesz wkleić te z tuningu)
    model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.015,
        max_depth=5,
        subsample=0.7,
        colsample_bytree=0.5,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50
    )

    # Walidacja na kawałku train
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

print("\n--- WYNIKI ---")
y_day, p_day, m_day = train_and_eval(data_day, 'Target_Day', 'MODEL DNIA (Start 06:00)')
y_night, p_night, m_night = train_and_eval(data_night, 'Target_Night', 'MODEL NOCY (Start 18:00)')

# ==========================================
# 5. WIZUALIZACJA
# ==========================================
fig, ax = plt.subplots(2, 1, figsize=(12, 10))

# Wykres Dnia
subset = 60
ax[0].plot(y_day.index[-subset:], y_day.values[-subset:], 'o-', color='orange', label='Rzeczywista (Dzień)', alpha=0.6)
ax[0].plot(y_day.index[-subset:], p_day[-subset:], 'x--', color='black', label='Prognoza (z 06:00)')
ax[0].set_title('Prognoza Dnia (na podstawie danych porannych)')
ax[0].legend()

# Wykres Nocy
ax[1].plot(y_night.index[-subset:], y_night.values[-subset:], 'o-', color='navy', label='Rzeczywista (Noc)', alpha=0.6)
ax[1].plot(y_night.index[-subset:], p_night[-subset:], 'x--', color='red', label='Prognoza (z 18:00)')
ax[1].set_title('Prognoza Nocy (na podstawie danych wieczornych)')
ax[1].legend()

plt.tight_layout()
plt.show()

# Importancja cech dla Nocy (zobaczmy czy gradienty działają)
plt.figure(figsize=(10,5))
xgb.plot_importance(m_night, max_num_features=10, importance_type='gain', title='Co decyduje o temperaturze nocy?')
plt.show()
