import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.style.use('seaborn-v0_8-whitegrid')

# 1. Wczytanie danych
try:
    df_hourly = pd.read_csv('wroclaw_ml_hourly.csv', parse_dates=['time'], index_col='time')
except FileNotFoundError:
    print("Brak pliku 'wroclaw_ml_hourly.csv'")
    exit()

df_hourly = df_hourly.sort_index()

# 2. Tworzenie celu

# Obliczamy prawdziwą średnią temperaturę dla każdego dnia
df_daily = df_hourly.resample('D').agg({
    'temp': 'mean'
})
df_daily.rename(columns={'temp': 'Daily_Avg_Temp'}, inplace=True)

# Celem jest średnia temperatura JUTRO
df_daily['Target'] = df_daily['Daily_Avg_Temp'].shift(-1)


# 3. Feature engineering
features = df_hourly.resample('D').agg({
    'temp': ['mean', 'max', 'min', 'std'], # std mówi nam czy pogoda była stabilna
    'pres': ['mean', 'min', lambda x: x.iloc[-1] - x.iloc[0]], # Średnia, min i ZMIANA ciśnienia w ciągu dnia
    'rhum': ['mean'],
    'wspd': ['mean', 'max'],
    'prcp': ['sum']
})


# Spłaszczamy nazwy kolumn (z MultiIndexu robi się np. temp_mean, temp_std)
features.columns = ['_'.join(col) for col in features.columns]
features = features.rename(columns={'pres_<lambda_0>': 'pres_trend_day'})

# 1. Temperatura Wieczorna (20:00 - 23:00)
# Temperatura tuż przed północą mówi o jutrze dosyć dużo
temp_evening = df_hourly[df_hourly.index.hour >= 20].resample('D')['temp'].mean()
features['temp_evening'] = temp_evening

# 2. Temperatura Poranna (06:00 - 09:00)
temp_morning = df_hourly[(df_hourly.index.hour >= 6) & (df_hourly.index.hour <= 9)].resample('D')['temp'].mean()
features['temp_morning'] = temp_morning

# 3. Dynamika dnia (Różnica wieczór - poranek)
# Czy dzień się nagrzewał, czy ochładzał pod koniec?
features['day_dynamic'] = features['temp_evening'] - features['temp_morning']

# 4. Ostatnie znane ciśnienie (o 23:00)
pres_last = df_hourly.resample('D')['pres'].last()
features['pres_last'] = pres_last

# 4. Łączenie i Czyszczenie

# Łączymy cechy (X) z celem (y)
data = pd.concat([features, df_daily['Target']], axis=1)

# Dodajemy informacje kalendarzowe (bazując na indeksie dziennym)
data['day_of_year_sin'] = np.sin(2 * np.pi * data.index.dayofyear / 365)
data['day_of_year_cos'] = np.cos(2 * np.pi * data.index.dayofyear / 365)

# Dodajemy opóźnienia na poziomie dziennym
# Co było wczoraj (czyli 2 dni temu względem prognozy)
data['temp_mean_lag1'] = data['temp_mean'].shift(1)
data['temp_evening_lag1'] = data['temp_evening'].shift(1)

# Usuwamy braki spowodowane naszymi przesunięciami
data = data.dropna()

X = data.drop(columns=['Target'])
y = data['Target']

# 5. Trening Modelu (XGBoost)

test_days = 365
X_train, X_test = X.iloc[:-test_days], X.iloc[-test_days:]
y_train, y_test = y.iloc[:-test_days], y.iloc[-test_days:]

model = xgb.XGBRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=5,
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1
)

# Walidacja na 10% zbioru treningowego
split = int(len(X_train) * 0.9)
X_tr, X_val = X_train.iloc[:split], X_train.iloc[split:]
y_tr, y_val = y_train.iloc[:split], y_train.iloc[split:]

model.fit(
    X_tr, y_tr,
    eval_set=[(X_tr, y_tr), (X_val, y_val)],
    verbose=0
)

# 6. Wyniki

preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"Wyniki na zbiorze testowym")
print(f"RMSE: {rmse:.2f} °C")
print(f"MAE:  {mae:.2f} °C")
print(f"R2:   {r2:.4f}")

# Wykres
plt.figure(figsize=(12, 6))
n_days=90
plt.plot(y_test.index[-n_days:], y_test.values[-n_days:], label='Prawdziwa Średnia', color='black')
plt.plot(y_test.index[-n_days:], preds[-n_days:], label='Prognoza XGBoost', color='green', linestyle='--')
plt.title('Prognoza Średniej Temperatury na podstawie danych godzinowych')
plt.legend()
plt.tight_layout()
# plt.savefig('xgb_demo_result.png')
plt.show()

xgb.plot_importance(model, max_num_features=15, title='Najważniejsze cechy', importance_type='gain')
plt.tight_layout()
# plt.savefig('xgb_importance.png')
plt.show()
