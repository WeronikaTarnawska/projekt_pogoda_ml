import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ustawienie stylu
sns.set_theme(style="whitegrid")

# 1. Wczytanie danych
try:
    df = pd.read_csv('wroclaw_ml_daily.csv')
except FileNotFoundError:
    print("Brak pliku 'wroclaw_ml_daily.csv'. Uruchom najpierw 'pobierz_dane.py'.")
    exit()

plt.figure(figsize=(12, 7))

# 2. Rysowanie punktów (Scatter plot)
# Oś X: Numer dnia w roku (1 = 1 stycznia, 365 = 31 grudnia)
# Oś Y: Temperatura
# Kolor (hue): Rok
scatter = plt.scatter(
    df['day_of_year'],
    df['temp'],
    c=df['year'],
    cmap='viridis', # Kolorystyka od fioletu (stare lata) do żółci (nowe lata)
    s=15,
    alpha=0.6,
    label='Pomiary dzienne'
)

# 3. Rysowanie linii średniej (Trend sezonowy)
# Obliczamy średnią temperaturę dla każdego z 366 dni w roku
daily_means = df.groupby('day_of_year')['temp'].mean()

plt.plot(
    daily_means.index,
    daily_means.values,
    color='red',
    linewidth=3,
    label='Średnia dla danego dnia (Wzorzec)'
)

# 4. Kosmetyka wykresu
plt.title('Sezonowość: Rozkład temperatury w zależności od dnia roku', fontsize=16)
plt.xlabel('Dzień roku (1 - 366)', fontsize=12)
plt.ylabel('Temperatura (°C)', fontsize=12)

# Dodanie paska legendy kolorów (dla lat)
cbar = plt.colorbar(scatter)
cbar.set_label('Rok')

plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
# plt.savefig('wykres_sezonowosc_prosty.png')
print("Zapisano wykres jako 'wykres_sezonowosc_prosty.png'")
plt.show()
