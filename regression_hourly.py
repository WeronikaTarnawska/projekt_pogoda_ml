import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.style.use('seaborn-v0_8-whitegrid')

# 1. Wczytanie danych
try:
    df_hourly = pd.read_csv('wroclaw_ml_hourly.csv', parse_dates=['time'], index_col='time')
except FileNotFoundError:
    print("Brak pliku 'wroclaw_ml_hourly.csv'")
    exit()

df_hourly = df_hourly.sort_index()

# 2. FEATURE ENGINEERING (To samo co wcześniej - bo działa!)

# Agregacja do danych dziennych
features = df_hourly.resample('D').agg({
    'temp': ['mean', 'min', 'max', 'std'],
    'pres': ['mean', lambda x: x.iloc[-1]], # Średnia i ostatnie znane ciśnienie
    'rhum': ['mean'],
    'wspd': ['mean']
})

# Spłaszczanie nazw kolumn
features.columns = ['_'.join(col) for col in features.columns]
features = features.rename(columns={'pres_<lambda_0>': 'pres_last'})

# Temp wieczorna (bardzo ważna dla nocy i poranka)
features['temp_evening'] = df_hourly[df_hourly.index.hour >= 20].resample('D')['temp'].mean()

# Cel: Średnia temperatura JUTRO
features['Target'] = features['temp_mean'].shift(-1)

# Cechy cykliczne
features['day_sin'] = np.sin(2 * np.pi * features.index.dayofyear / 365)
features['day_cos'] = np.cos(2 * np.pi * features.index.dayofyear / 365)

# Usunięcie braków
data = features.dropna()

# Podział X i y
X = data.drop(columns=['Target'])
y = data['Target']

# Podział chronologiczny
test_days = 365
X_train = X.iloc[:-test_days]
y_train = y.iloc[:-test_days]
X_test = X.iloc[-test_days:]
y_test = y.iloc[-test_days:]

# 3. Preprocessing danych

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. MODEL: Linear Regression (Ridge)

# Używamy Ridge
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# 5. Wyniki

preds = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"Wyniki na zbiorze testowym")
print(f"RMSE: {np.sqrt(mse):.2f} °C")
print(f"MAE:  {mae:.2f} °C")
print(f"r2:  {r2:.4f}")
print("-" * 30)

# Wagi równania
coefs = pd.DataFrame({
    'Cecha': X.columns,
    'Waga (Wpływ)': model.coef_
}).sort_values(by='Waga (Wpływ)', key=abs, ascending=False)

print("Najważniejsze wagi modelu:")
print(coefs.head(10))

plt.figure(figsize=(12, 6))
n_days=90
plt.plot(y_test.index[-n_days:], y_test.values[-n_days:], label='Prawdziwa', color='black')
plt.plot(y_test.index[-n_days:], preds[-n_days:], label='Regresja Liniowa', color='blue', linestyle='--')
plt.title('Regresja Liniowa')
plt.legend()
plt.show()
