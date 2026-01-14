# _

## konwencja

- **Robimy plik per "duży temat" (notebook / py)**

- **Na prezentację na środę 21.01 trzeba zrobić notebooka z najważniejszymi rzeczami**

- **Na oddawanie projektu też trzeba zrobić notebook prezentacyjny (i slajdy?)**

## kto co robi

| temat | kto |
|-------|-----|
|ARIMA,SARIMA,exponential smoothing|wtarn|
|dekompozycja szeregu czasowego i analiza tego|wtarn|
|prophet|młuk|
|jenen mały model przewiduje temperaturę za godzinę, drugi za dwie, itd,|wtarn (ale nie wiem czy w końcu to zrobię, może jak będę miała dużo czasu)|
|XGBoost (ciąg dalszy)|młuk|

## Pytania, pomysły, notatki

Potencjalne problemy:

- Normalnie pogodę prognozuje się [inaczej](https://en.wikipedia.org/wiki/Numerical_weather_prediction), bo pogoda zależy od innych czynników niż tylko przeszłe stany
- Modele lubią zauważać że pogoda zmienia się raczej wolno i przewidywać "jutro będzie tak jak dzisiaj"

Baseline:

- dzisiaj jest tak jak wczoraj
- zawsze średnia
- regresja liniowa
- regresja wielomianowa? (może być już fajna i przydatna do właściwych predykcji)

Lepsze pomysły:

- Metody uczenia zespołowego: Random Forest, AdaBoost, GradientBoost, XGBoost
  - "Małe" modele trenowane na podzbiorach danych z różnych okresów
  - jenen mały model przewiduje temperaturę za godzinę, drugi za dwie, itd, i potem z tego predykcja średniej temperatury danego dnia
  - do tego trzeba zrobić feature engineering: każda krotka dostaje dane o tym co się działo przez ostatni jakiś okres czasu (np temperatura wczoraj, średnia temperatura w tym tygodniu, miesiąc, pora roku) - chcemu obserwować sezonowość + lokalny trend
- lightbgm (on w sumie też robi ensemble) - był na liście 10
- dekompozycja szeregu czasowego: **wtarn**
  - szereg ma trend sezonowy, trend dobowy, i losowy szum wokół tego
  - sprawdzić jak dużo umiemy przewidzieć ponad ten losowy szum
  - jakaś analiza błędów vs sezon -> np bardziej mylimy się wiosną niż latem
- Inne metody do operowania na szeregach czasowych (ich nie było na wykładzie)
  - Wygładzanie wykładnicze (Exponential Smoothing / Holt-Winters) **wtarn**
  - Statystyka: ARIMA / SARIMA **wtarn**
  - Modele addytywne: Facebook Prophet - jest dostępny za freeko, biblioteka `prophet` **młuk**
- Dodać więcej stacji wokół punktu dla którego robimy predykcje. Wtedy można przewidywać na podstawie kierunku wiatru: jak wieje z północy do jutro będzie taka pogoda jak dzisiaj jest na północ od wrocławia.

---

## Inne losowe rzeczy

### Źródła i pomoce

- [kaggle: time series forecasting tutorial](https://www.kaggle.com/code/iamleonie/intro-to-time-series-forecasting)
- [tensorflow tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [?](https://www.tigerdata.com/blog/what-is-time-series-forecasting)
