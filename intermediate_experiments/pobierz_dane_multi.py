from datetime import datetime
import meteostat as ms
import pandas as pd
import numpy as np

# Konfiguracja
START = datetime(2015, 1, 1)
END = datetime.now()
ms.config.block_large_requests = False

STATIONS = {
    'Wroclaw': '12424',  # Cel (Target)
    'Legnica': '12415',  # Zachód
    'Opole':   '12530',  # Wschód
    'Poznan':  '12330'   # Północ
}

# Definicje kolumn
# Dla Wrocławia bierzemy WSZYSTKO co może się przydać
COLS_MAIN = ['temp', 'rhum', 'prcp', 'snwd', 'wdir', 'wspd', 'wpgt', 'pres', 'cldc', 'coco']
# Dla sąsiadów bierzemy fizykę (Ciśnienie, Temp, Wiatr, Wilgotność)
COLS_NEIGHBOR = ['temp', 'pres', 'prcp', 'wspd', 'rhum', 'wdir']

print(f"Pobieranie danych (Full dla Wrocławia + Sąsiedzi)...")

dfs = []

for name, station_id in STATIONS.items():
    print(f"--- Pobieranie: {name} (ID: {station_id}) ---")
    df = ms.hourly(station_id, START, END).fetch()

    if df.empty:
        print(f"Ostrzeżenie: Brak danych dla {name}!")
        continue

    # Wybór kolumn w zależności od tego, czy to stacja główna czy sąsiad
    target_cols = COLS_MAIN if name == 'Wroclaw' else COLS_NEIGHBOR

    # Wybieramy tylko te, które faktycznie istnieją w pobranych danych
    available = [c for c in target_cols if c in df.columns]
    df = df[available]

    # Zmiana nazw (temp -> temp_Legnica)
    if name == 'Wroclaw':
        # Wrocławia nie zmieniamy, żeby zachować oryginalne nazwy (temp, pres...)
        pass
    else:
        df.columns = [f"{col}_{name}" for col in df.columns]

    dfs.append(df)

# Łączenie
df_combined = pd.concat(dfs, axis=1)
df_combined = df_combined.sort_index()

# 1. Interpolacja małych dziur (do 3h) w środku danych
df_combined = df_combined.interpolate(method='linear', limit=3)

# 2. Uzupełnianie zerami tam, gdzie to ma sens (deszcz, śnieg)
cols_zero = ['prcp', 'snwd', 'wpgt', 'cldc'] # Jeśli brakuje, to pewnie 0
for col in cols_zero:
    if col in df_combined.columns:
        df_combined[col] = df_combined[col].fillna(0)

# Dla sąsiadów też (np. prcp_Legnica)
for col in df_combined.columns:
    if any(x in col for x in ['prcp', 'snwd', 'wpgt']):
        df_combined[col] = df_combined[col].fillna(0)

# 3. ODCIĘCIE OGONA
# Jeśli na końcu są jakiekolwiek braki w kluczowych kolumnach (temp, pres), usuwamy te wiersze
# ponieważ jeszcze dane pogodowe na te dni się nie pojawiły
# dropna() usunie wszystkie wiersze z końca, które nie mają danych ze wszystkich stacji.
before_shape = df_combined.shape
df_combined = df_combined.dropna()
after_shape = df_combined.shape

print(f"Usunięto {before_shape[0] - after_shape[0]} godzin.")

filename = 'wroclaw_multi_station_full.csv'
df_combined.to_csv(filename)
print(f"Gotowe! Zapisano do: {filename}")
print(df_combined.head())
