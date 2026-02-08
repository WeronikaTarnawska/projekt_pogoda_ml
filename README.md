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
i biblioteki: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `statsmodels`, `meteostat`.

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
- `preprocess.ipynb` - łączenie danych z różnych stacji i uzupełnianie braków

#### Prognoza temperatury na dzień i noc

- `temperature_baselines.ipynb` - modele beseline do predykcji temperatury
- `temperature_xgboost.ipynb` - predykcja temperatury modelem xgboost
- `temperature_sarima.ipynb` predykcja temperatury modelem SARIMAX

#### Prognoza deszczu na następny dzień

`rain_classification.ipynb`

- baseline ("dziś tak jak wczoraj", "losowo z prawdopodobieństwem na podstawie przeszłych danych")
- xgboost, adaboost, randomforest

## Streszczenie i komentarz do wyników

Naszym celem było sprawdzenie, czy klasyczne metody uczenia maszynowego (ML) są w stanie skutecznie prognozować pogodę, traktując ją jako szereg czasowy, bez użycia skomplikowanych symulacji fizycznych atmosfery (NWP).

1. *Pozyskanie danych:* Pobraliśmy historyczne dane godzinowe (2015-2025) z serwisu Meteostat dla Wrocławia oraz stacji sąsiednich (Legnica, Opole, Poznań, Kłodzko).
2. *Inżynieria cech (Feature Engineering):*
    - Stworzyliśmy cechy oparte na opóźnieniach czasowych, trendach zmian ciśnienia/temperatury oraz aproksymacji punktu rosy.
    - Wprowadziliśmy *kontekst przestrzenny*: uwzględniliśmy różnice temperatur i ciśnień między Wrocławiem a stacjami sąsiednimi, co pozwoliło naszym modelom "widzieć" nadchodzące fronty atmosferyczne.
    - Zakodowaliśmy cykliczność czasu (sinus/cosinus dnia roku i godziny).

3. *Modelowanie:*

    - Do *predykcji temperatury* (regresja) wykorzystaliśmy model liniowy SARIMAX oraz nieliniowy XGBoost.
    - Do *predykcji opadów* (klasyfikacja binarna) zastosowaliśmy metody zespołowe (Ensemble Methods): XGBoost, AdaBoost oraz Random Forest.

4. *Ewaluacja:* Uzyskane wyniki zestawiliśmy z prostymi baseline'ami (Persistence - "jutro będzie jak dziś", Climatology - "będzie tak jak zwykle o tej porze roku").

### Wyniki

#### 1. Prognoza temperatury (Regresja)

W zadaniu predykcji temperatury najlepsze wyniki osiągnęliśmy używając modelu XGBoost zasilonego danymi z sąsiednich stacji, choć różnice względem prostszych wariantów nie były drastyczne.

- Największym wyzwaniem okazało się pokonanie prostego baseline'u typu *persistence* ("jutro będzie tak samo jak dziś"). Pogoda często zmienia się stopniowo, przez co proste skopiowanie temperatury z dnia poprzedniego daje pozornie świetne wyniki. Eksperymenty pokazały, że samo dodawanie kolejnych zmiennych nie zawsze skutkowało poprawą wyników. Dopiero selekcja cech pozwoliła nam osiągnąć błąd predykcji niższy niż w przypadku trywialnej metody przepisania wartości z poprzedniego dnia.
- *XGBoost vs SARIMAX:* Ostatecznie XGBoost poradził sobie lepiej niż liniowy SARIMAX. SARIMAX jest modelem liniowym, który świetnie radzi sobie z prostą sezonowością, ale słabo reaguje na nagłe, nieliniowe zmiany pogody. XGBoost, oparty na drzewach decyzyjnych, potrafi modelować skomplikowane interakcje, np. "jeśli ciśnienie spada ORAZ wiatr wieje z zachodu -> temperatura spadnie", co jest kluczowe w dynamice atmosfery.
- *Dzień vs Noc:* Prognozy nocne są obarczone zwykle mniejszym błędem niż dzienne, ponieważ w nocy temperatura dąży do stabilizacji (wypromieniowanie ciepła), podczas gdy w dzień na temperaturę wpływa zmienne zachmurzenie i insolacja (nasłonecznienie), co wprowadza większy czynnik chaotyczny.
- *Wpływ danych przestrzennych:* Dodanie danych z Legnicy i Opola poprawiło wynik modelu dziennego (spadek RMSE z 2.36°C do 1.97°C). Pogoda w danej lokalozacji w dużej mierze jest wynikiem napływu mas powietrza. Model "nauczył się", że jeśli w Legnicy (na zachód od Wrocławia) robi się zimniej, to z opóźnieniem czasowym ochłodzenie dotrze do Wrocławia.

#### 2. Prognoza deszczu (Klasyfikacja)

Prognozowanie opadów okazało się znacznie większym wyzwaniem. Choć nasze modele ML przebiły losowe zgadywanie, to tutaj też głównym przeciwnikiem był prosty schemat "jeśli padało wczoraj, będzie padać jutro".

- Baseline Persistence (67%) znów ustawił poprzeczkę wysoko. Tak samo jak przy temperaturze, krótki horyzont czasowy sprzyja prostym metodom bazującym na bezwładności pogody.
- Tutaj w wersji podstawowej XGBoost nie wykazał istotnej przewagi. W miarę zadowalające rezultaty dały Random Forest i AdaBoost - udało się przewidzieć trochę lepiej niż baseline.
- Podobnie jak wcześniej, pomogło użycie danych z dodatkowych stacji, co dobrze odzwierciedla to, że opady w większości zależą od przemieszczania się mas powietrza z miejsca na miejsce.

### Wnioski końcowe

Projekt udowodnił, że traktowanie pogody jako problemu uczenia maszynowego (Time Series Regression/Classification) pozwala osiągnąć wyniki istotnie lepsze od losowego szumu i prostych heurystyk.

Jednocześnie eksperyment ten był lekcją pokory wobec złożoności zjawisk atmosferycznych. Dane historyczne i statystyka to potężne narzędzia, ale mają swoje granice. Bez uwzględnienia praw fizyki i dynamiki płynów (na których opierają się profesjonalne modele numeryczne typu GFS czy ECMWF), "czysty" Machine Learning nie jest w stanie w pełni oddać dynamiki pogody. Nasze modele wykazują pewną zdolność predykcyjną, ale do produkcyjnej jakości serwisów takich jak Windy czy Meteoblue jeszcze im brakuje.
