from datetime import datetime
import meteostat as ms
import pandas as pd
import numpy as np

# 1. Konfiguracja lokalizacji (Wrocław)
POINT = ms.Point(51.1093, 17.0386, 120)

# 2. Ustalenie zakresu dat (od 2015 do dzisiaj)
START = datetime(2015, 1, 1)
END = datetime.now()

print(f"Szukam stacji i pobieram dane dla zakresu: {START.date()} - {END.date()}")

# Znajdź najbliższą stację
stations = ms.stations.nearby(POINT, limit=4)
if stations.empty:
    print("Nie znaleziono stacji w pobliżu.")
    exit()

WROCLAW_STATION_ID=stations.iloc[0].name
print(f"Wybrana stacja: {WROCLAW_STATION_ID} ({stations.iloc[0]['name']})")

# 3. Pobranie danych DZIENNYCH
df_daily = ms.daily(WROCLAW_STATION_ID, START, END).fetch()

# 4. Pobranie danych GODZINOWYCH
ms.config.block_large_requests = False
df_hourly = ms.hourly(WROCLAW_STATION_ID, START, END).fetch()

# 5. Wyczyszczenie danych i dodanie bardziej użytecznych kolumn
def clean_and_feature_engineer(df, is_hourly=False):
    # A. Usuwanie pustych kolumn
    df = df.dropna(axis=1, how='all')

    # B. Uzupełnianie braków (Interpolacja dla ciągłych, 0 dla zjawisk)
    cols_interp = ['temp', 'tmin', 'tmax', 'pres', 'wspd', 'rhum', 'dwpt']
    cols_zero = ['prcp', 'snwd', 'wpgt', 'cldc', 'tsun']

    for col in cols_interp:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear')

    for col in cols_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # C. Feature Engineering (Rozbijanie daty)
    df = df.reset_index() # Wyciągamy 'time' z indeksu

    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month # Przydatne dla prostszych modeli
    df['day'] = df['time'].dt.day
    df['day_of_year'] = df['time'].dt.dayofyear

    # Dodatkowe cechy tylko dla danych godzinowych
    if is_hourly:
        df['hour'] = df['time'].dt.hour
        # Transformacja cykliczna godziny
        # Dzięki temu godzina 23:00 jest "blisko" 00:00
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df = df.set_index('time')

    return df

df_daily_clean = clean_and_feature_engineer(df_daily, is_hourly=False)
df_hourly_clean = clean_and_feature_engineer(df_hourly, is_hourly=True)

# 6. Zapis
df_daily_clean.to_csv('wroclaw_ml_daily.csv')
df_hourly_clean.to_csv('wroclaw_ml_hourly.csv')

print("Gotowe")
print(f"Dane dzienne: {df_daily_clean.shape} są w pliku 'wroclaw_ml_daily.csv'")
print(f"Dane godzinowe: {df_hourly_clean.shape} są w pliku 'wroclaw_ml_hourly.csv'")
print()
print("Przykładowe dane godzinowe:")
print(df_hourly_clean[['temp', 'year', 'day_of_year', 'hour']].head())
