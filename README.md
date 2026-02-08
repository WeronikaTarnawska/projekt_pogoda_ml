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
i biblioteki: `numpy`,`pandas`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `statsmodels`, `meteostat`.

Można je zainstalować na przykład tak:

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas matplotlib seaborn scikit-learn xgboost statsmodels meteostat
```

### Pobranie danych

Przed uruchomieniem notebooków trzeba pobrać dane:

```bash
python download_data.py
```

Po uruchomieniu, w katalogu `data` pojawią się pliki `klodzko.csv`, `legnica.csv`, `opole.csv`, `poznan.csv`, `wroclaw.csv`, `wroclaw_daily.csv`.

Dane z których korzystają modele tworzone w `preprocess.ipynb` - ten notebook też trzeba uruchomić przed uruchamianiem modeli.

### Notebooki

- `eda.ipynb` - Eksploracyjna analiza danych
- `preprocess.ipybn` - łączenie danych z różnych stacji i uzupełnianie braków

#### Prognoza temperatury na dzień i noc

- `temperature_baselines.ipynb` - modele beseline do predykcji temperatury
- `temperature_xgboost.ipynb` - predykcja temperatury modelem xgboost
- `temperature_sarima.ipynb` predykcja temperatury modelem SARIMAX

#### Prognoza deszczu na następny dzień

`rain_classification.ipynb`

- baseline ("dziś tak jak wczoraj", "losowo z prawdopodobieństwem na podstawie przeszłych danych")
- xgboost, adaboost, randomforest
