# Prognoza pogody

Projekt na kurs *uczenie maszynowe*, Instytut Informatyki Uniwersytetu Wrocławskiego

Autorzy: Weronika Tarnawska i Michał Łukasik

## Cel projektu

Predykcja pogody na podstawie danych historycznych

- Normalnie pogodę prognozuje się rozwiązując równania różniczkowe z danych z atmosfery
[(numerical weather prediction)](https://en.wikipedia.org/wiki/Numerical_weather_prediction)
- My chcemy podjąć próbę modelowania pogody jako szereg czasowy
- I zobaczyć jak dużo (czy cokolwiek) ponad losowy szum da się w ten sposób przewidzieć

Ograniczyliśmy problem prognozy do dwóch zadań:

1. Prognoza średniej temperatury na kolejny dzień i na kolejną noc
2. Prognoza "czy następnego dnia będzie padać"

## Dane

- [Meteostat](https://dev.meteostat.net/python)
- Korzystamy z danych godzinowych
- Robimy predykcje dla stacji Wrocław
- W jednym z modeli korzystamy również z danych z sąsiednich stacji: Legnica, Opole, Poznan

## Zawartość projektu

### Dependencje

Żeby uruchomić skrypty potrzebny będzie python3 (>=3.12 jest ok, nie gwarantuję, że starszy będzie działał),
i biblioteki: `numpy`,`pandas`, ... TODO

Można je zainstalować na przykład tak:

```bash
python3 -m venv venv
source venv/bin/activate
pip install TODO
```

### Pobranie danych

Przed uruchomieniem notebooków trzeba pobrać dane:

```bash
python download_data.py
```

Po uruchomieniu, w aktualnym katalogu pojawią się pliki 
(TODO) `wroclaw_multi_station.csv` i `wroclaw_multi_station_full.csv`,
zawierające odpowiednio dane czyste i po wstępnym preprocessingu.

### Notebooki

- `eda.ipynb` - Eksploracyjna analiza danych (EDA) – statystyki opisowe, rozkłady zmiennych, korelacje, brakujące wartości, obserwacje odstające, wnioski z analizy
- `preprocess.ipybn` - feature engineering i agregacja day/night

#### Prognoza temperatury

<!-- 
Szczegółowy opis użytych modeli – jeśli nie są to klasyczne algorytmy omówione na wykładzie
Metodologia ewaluacji – podział danych (train/test/validation), metryki jakości, walidacja krzyżowa
 -->

- baselines
- xgboots bez dodatkowych stacji
- xgboost z dodatkowymi stacjiami
- sarima
- regresja liniowa
- prophet

#### Prognoza deszczu

- baselines (persistance, season)
- regresjia logistyczna
- xgboost
- adaboost/ randomforest

### Wyniki

<!-- 
Szczegółowy opis uzyskanych wyników – porównanie modeli, analiza błędów, wizualizacje
Wnioski końcowe – podsumowanie, ograniczenia rozwiązania, perspektywy rozwoju 
-->

`results.md`

- coś o tym że problem jest trudny i jak się tak naprawdę przewiduje pogodę
- porównanie modeli do deszczu
- porównanie modeli do temperatury
- wnioski: że prognoza chujowa, ale lepsza niż nasz baseline, a że było to trudne, to myślę że ogłaszamy sukces