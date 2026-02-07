import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.model_selection import train_test_split

# 1. WCZYTANIE DANYCH
FILENAME = 'wroclaw_multi_station_full.csv'
df = pd.read_csv(FILENAME, parse_dates=['time'], index_col='time')
df = df.sort_index()

# 2. FEATURE ENGINEERING

# A. Czas
df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

# B. Dynamika zmian (dla frontów)
# Spadek ciśnienia i wzrost wilgotności w ostatnich 3h i 6h
df['pres_trend_3h'] = df['pres'].diff(3)
df['rhum_trend_3h'] = df['rhum'].diff(3)
df['temp_trend_3h'] = df['temp'].diff(3)

# C. Opady u sąsiadów
neighbor_rain_cols = [c for c in df.columns if 'prcp_' in c]
df['neighbor_rain_sum'] = df[neighbor_rain_cols].sum(axis=1)

# D. Lagi
df['prcp_lag1'] = df['prcp'].shift(1)
df['rhum_lag1'] = df['rhum'].shift(1)

# E. Interakcja: Wilgotność + Ciśnienie (często wysoka wilgotność + niskie ciśnienie = deszcz)
df['rhum_pres_ratio'] = df['rhum'] / (df['pres'] + 1e-5)

# 3. PRZYGOTOWANIE TARGETU (KLASYFIKACJA)
# Przewidujemy czy w NASTĘPNEJ godzinie wystąpi jakikolwiek opad (> 0 mm)
df['Target_Rain'] = (df['prcp'].shift(-1) > 0).astype(int)

# Czyszczenie
df = df.dropna(subset=['Target_Rain'])
df = df.fillna(0)

# Wybór cech (usuwamy surowe kolumny tekstowe i Target)
cols_exclude = ['Target_Rain', 'coco', 'wpgt', 'prcp'] # usuwamy prcp, bo to silny przeciek informacji
X_cols = [c for c in df.columns if c not in cols_exclude and df[c].dtype in ['float64', 'int64']]

X = df[X_cols]
y = df['Target_Rain']

# Podział
test_hours = 24 * 365
X_train, X_test = X.iloc[:-test_hours], X.iloc[-test_hours:]
y_train, y_test = y.iloc[:-test_hours], y.iloc[-test_hours:]

# Obliczanie wagi dla klas niezbalansowanych (Scale Position Weight)
# ratio = liczba_negatywnych / liczba_pozytywnych
ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Stosunek brak deszczu / deszcz: {ratio:.2f}")

# ==========================================
# 4. MODEL XGBOOST CLASSIFIER
# ==========================================
model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    scale_pos_weight=ratio,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    early_stopping_rounds=50
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100
)

# 5. OCENA I WYNIKI
preds_prob = model.predict_proba(X_test)[:, 1] # Prawdopodobieństwo deszczu
preds_bin = (preds_prob > 0.5).astype(int)      # Klasyfikacja tak/nie

print()
print("Raport klasyfikacji")
print(classification_report(y_test, preds_bin))
print(f"ROC-AUC Score: {roc_auc_score(y_test, preds_prob):.4f}")

# Wykres ważności cech
plt.figure(figsize=(10, 8))
xgb.plot_importance(model, max_num_features=15, importance_type='gain')
plt.title("Co najbardziej zwiastuje deszcz we Wrocławiu?")
plt.show()

# Przykład prognozy prawdopodobieństwa
plt.figure(figsize=(15, 5))
subset_len = 24 * 14 # 2 tygodnie
plt.plot(y_test.index[-subset_len:], y_test.values[-subset_len:], 'k|', label='Faktyczny deszcz', markersize=10)
plt.fill_between(y_test.index[-subset_len:], 0, preds_prob[-subset_len:], color='blue', alpha=0.3, label='Prawdopodobieństwo deszczu')
plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Próg 50%')
plt.title("Prognoza prawdopodobieństwa wystąpienia opadów")
plt.legend()
plt.show()
