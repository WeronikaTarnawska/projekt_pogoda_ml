import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ustawienie estetyki wykresów
sns.set_theme(style="whitegrid")

# 1. Wczytanie danych
try:
    df_daily = pd.read_csv('wroclaw_ml_daily.csv', parse_dates=['time'], index_col='time')
    df_hourly = pd.read_csv('wroclaw_ml_hourly.csv', parse_dates=['time'], index_col='time')
except FileNotFoundError:
    print("Brak plików CSV! Uruchom najpierw 'pobierz_dane.py'.")
    exit()

# OKNO 1: ANALIZA DŁUGOTERMINOWA (Daily)
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
fig1.suptitle('Analiza Długoterminowa: Trendy i Sezonowość (Wrocław)', fontsize=16)

# Wykres 1.1: Trend wieloletni (Rolling Average)
# Rysujemy surowe dane
ax1.plot(df_daily.index, df_daily['temp'], color='lightgray', alpha=0.3, label='Dane dzienne')
# Rysujemy średnią kroczącą roczną (365 dni) - pokazuje zmianę klimatu
df_daily['rolling_year'] = df_daily['temp'].rolling(window=365, center=True).mean()
ax1.plot(df_daily.index, df_daily['rolling_year'], color='#d62728', linewidth=2.5, label='Trend roczny (klimat)')

ax1.set_title('Zmiana temperatury na przestrzeni lat (2015-teraz)')
ax1.set_ylabel('Temperatura (°C)')
ax1.legend()
ax1.set_xlim(df_daily.index.min(), df_daily.index.max())

# Wykres 1.2: Rozkład temperatur w miesiącach (Boxplot)
# To pokazuje "zmienność" - np. czy marzec jest bardziej nieprzewidywalny niż lipiec?
sns.boxplot(x='month', y='temp', data=df_daily, ax=ax2, palette="coolwarm", hue='month', legend=False)
ax2.set_title('Rozkład temperatur w poszczególnych miesiącach')
ax2.set_xlabel('Miesiąc')
ax2.set_ylabel('Temperatura (°C)')

plt.tight_layout()
# plt.savefig('wykres_trendy_roczne.png')


# OKNO 2: ANALIZA GODZINOWA (Hourly)
fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 12))
fig2.suptitle('Analiza Cyklu Dobowego', fontsize=16)

# Wykres 2.1: Typowy przebieg doby (Zima vs Lato)
# Grupujemy dane godzinowe po godzinie (0-23) i miesiącu
hourly_avg = df_hourly.groupby(['month', 'hour'])['temp'].mean().reset_index()

# Wybieramy Styczeń (Zima) i Lipiec (Lato) dla porównania
winter_day = hourly_avg[hourly_avg['month'] == 1]
summer_day = hourly_avg[hourly_avg['month'] == 7]
spring_day = hourly_avg[hourly_avg['month'] == 4]

ax3.plot(winter_day['hour'], winter_day['temp'], 'o-', color='blue', label='Styczeń (Typowy dzień)')
ax3.plot(summer_day['hour'], summer_day['temp'], 'o-', color='red', label='Lipiec (Typowy dzień)')
ax3.plot(spring_day['hour'], spring_day['temp'], 'o-', color='green', label='Kwiecień (Typowy dzień)')

ax3.set_title('Średni przebieg temperatury w ciągu doby')
ax3.set_xlabel('Godzina (0-23)')
ax3.set_ylabel('Średnia Temperatura (°C)')
ax3.set_xticks(range(0, 24))
ax3.grid(True, which='both', linestyle='--')
ax3.legend()

# Wykres 2.2: Heatmapa (Mapa Ciepła) - Godzina vs Miesiąc
pivot_table = df_hourly.pivot_table(values='temp', index='hour', columns='month', aggfunc='mean')

sns.heatmap(pivot_table, ax=ax4, cmap='Spectral_r', annot=True, fmt=".1f", cbar_kws={'label': 'Temp (°C)'})
ax4.set_title('Mapa Ciepła: Średnia temperatura (Miesiąc vs Godzina)')
ax4.set_xlabel('Miesiąc')
ax4.set_ylabel('Godzina')
ax4.invert_yaxis() # Żeby godzina 0 była na dole, a 23 na górze (intuicyjnie)

plt.tight_layout()
# plt.savefig('wykres_cykle_godzinowe.png')

print("Gotowe!")
print("1. 'wykres_trendy_roczne.png' - pokazuje zmiany na przestrzeni lat.")
print("2. 'wykres_cykle_godzinowe.png' - pokazuje jak zmienia się temperatura w ciągu dnia.")
plt.show()
