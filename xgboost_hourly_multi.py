import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

plt.style.use('seaborn-v0_8-whitegrid')

# ==========================================
# 1. KONFIGURACJA I WCZYTANIE
# ==========================================
FILENAME = 'wroclaw_multi_station_full.csv'

try:
    df = pd.read_csv(FILENAME, parse_dates=['time'], index_col='time')
    df = df.sort_index()
    print(f"1. Wczytano dane surowe: {df.shape}")
except FileNotFoundError:
    print(f"BŁĄD: Nie ma pliku {FILENAME}. Uruchom najpierw pobierz_dane_multi.py!")
    exit()

# ==========================================
# 2. FEATURE ENGINEERING: Cechy Fizyczne i Przestrzenne
# ==========================================

# A. Czas
df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
df['day_year_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
df['day_year_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)

# B. Fizyka (Wrocław) - uzupełnianie zerami, jeśli brakuje
cols_physics = ['pres', 'rhum', 'wspd']
for c in cols_physics:
    if c in df.columns:
        df[f'{c}_diff_3h'] = df[c].diff(3).fillna(0)

# C. Wiatr (Kierunek na wektory)
if 'wdir' in df.columns:
    df['wdir_sin'] = np.sin(np.deg2rad(df['wdir'].fillna(0)))
    df['wdir_cos'] = np.cos(np.deg2rad(df['wdir'].fillna(0)))

# D. Sąsiedzi (Gradienty)
# Szukamy kolumn typu 'temp_Legnica', 'pres_Opole'
neighbor_cols = [c for c in df.columns if '_' in c and ('Legnica' in c or 'Opole' in c or 'Poznan' in c)]
stations = set([c.split('_')[1] for c in neighbor_cols])

print(f"   Wykryto sąsiadów: {stations}")

for st in stations:
    if f'temp_{st}' in df.columns:
        df[f'grad_temp_{st}'] = df[f'temp_{st}'] - df['temp']
    if f'pres_{st}' in df.columns:
        df[f'grad_pres_{st}'] = df[f'pres_{st}'] - df['pres']

# ==========================================
# 3. FEATURE ENGINEERING: Kontekst Dzienny
# ==========================================

# Tworzymy pomocniczy DataFrame dzienny
# Ważne: shift(1) oznacza "dane z wczoraj".
df_daily = pd.DataFrame(index=df.resample('D').first().index)

# 1. Średnia temperatura wczoraj
df_daily['temp_mean_lag1'] = df['temp'].resample('D').mean().shift(1)

# 2. Temperatura Wieczorna Wczoraj (20-23)
evening_mask = df.index.hour >= 20
df_daily['temp_evening_lag1'] = df.loc[evening_mask, 'temp'].resample('D').mean().shift(1)

# 3. Temperatura Poranna Wczoraj (06-09)
morning_mask = (df.index.hour >= 6) & (df.index.hour <= 9)
df_daily['temp_morning_lag1'] = df.loc[morning_mask, 'temp'].resample('D').mean().shift(1)

# 4. Dynamika Wczoraj
df_daily['day_dynamic_lag1'] = df_daily['temp_evening_lag1'] - df_daily['temp_morning_lag1']

# 5. Ostatnie ciśnienie wczoraj
if 'pres' in df.columns:
    df_daily['pres_last_lag1'] = df['pres'].resample('D').last().shift(1)

# DOŁĄCZANIE DO DANYCH GODZINOWYCH
df['date_key'] = df.index.date
df_daily['date_key'] = df_daily.index.date
df = df.reset_index().merge(df_daily, on='date_key', how='left').set_index('time')
df = df.drop(columns=['date_key'])

print(f"2. Po dodaniu cech inżynieryjnych: {df.shape}")

# ==========================================
# 4. PRZYGOTOWANIE TRENINGU
# ==========================================

# Target: Temperatura za 24 h
df['Target'] = df['temp'].shift(-24)

# Czyszczenie (Ostateczne)
# Usuwamy wiersze, gdzie brakuje Targetu (koniec pliku) lub kluczowych zmiennych
# Zamiast dropna() na wszystko, robimy to mądrze:
initial_len = len(df)
df = df.dropna(subset=['Target', 'temp', 'temp_mean_lag1'])
# Pozostałe braki (np. pojedyncza dziura w wietrze) wypełniamy 0
df = df.fillna(0)

print(f"3. Gotowe do treningu (po czyszczeniu): {df.shape} (Usunięto {initial_len - len(df)} wierszy)")

if len(df) < 1000:
    print("ojoj, coś ucięło dane.")
    exit()

# Wybór kolumn (X)
# Usuwamy Target i kolumny nienumeryczne/śmieciowe
cols_exclude = ['Target', 'coco', 'wpgt', 'wpgt_Legnica', 'wpgt_Opole'] # wpgt często robi problemy
X_cols = [c for c in df.columns if c not in cols_exclude and df[c].dtype in ['float64', 'int64']]

X = df[X_cols]
y = df['Target']

# Podział (Ostatni rok jako test)
test_hours = 24 * 365
X_train, X_test = X.iloc[:-test_hours], X.iloc[-test_hours:]
y_train, y_test = y.iloc[:-test_hours], y.iloc[-test_hours:]

# ==========================================
# 5. MODEL XGBOOST
# ==========================================
print(f"\n--- TRENOWANIE MODELU (Cechy: {len(X_cols)}) ---")

model = xgb.XGBRegressor(
    n_estimators=4000,
    learning_rate=0.02,
    max_depth=8,
    subsample=0.7,
    colsample_bytree=0.7,
    n_jobs=-1,
    random_state=42,
    early_stopping_rounds=100
)

# Walidacja (10% zbioru treningowego)
split_idx = int(len(X_train) * 0.9)
X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

model.fit(
    X_tr, y_tr,
    eval_set=[(X_tr, y_tr), (X_val, y_val)],
    verbose=200
)

# ==========================================
# 6. WYNIKI I WYKRESY
# ==========================================
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"Wyniki na zbiorze testowym")
print(f"RMSE: {rmse:.2f} °C")
print(f"MAE:  {mae:.2f} °C")
print(f"R2:   {r2:.4f}")

plt.figure(figsize=(12, 8))
importances = model.feature_importances_
indices = np.argsort(importances)[-20:]
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [X_cols[i] for i in indices])
plt.xlabel('Waga (Feature Importance)')
plt.title('20 Najważniejszych czynników wpływających na temperaturę')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 6))
steps = 24 * 7 * 5
plt.plot(y_test.index[-steps:], y_test.values[-steps:], label='Prawdziwa Temp', color='black', alpha=0.6, linewidth=2)
plt.plot(y_test.index[-steps:], preds[-steps:], label='XGBoost', color='#d62728', linestyle='--', linewidth=2)
plt.title(f"Prognoza pogody: Ostatnie {steps / 24} dni (RMSE: {rmse:.2f}°C)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
