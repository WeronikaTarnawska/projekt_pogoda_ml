import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

plt.style.use("seaborn-v0_8-whitegrid")

# Wczytanie danych
try:
    df_hourly = pd.read_csv(
        "wroclaw_ml_hourly.csv", parse_dates=["time"], index_col="time"
    )
except FileNotFoundError:
    print("Brak pliku 'wroclaw_ml_hourly.csv'")
    exit()

df_hourly = df_hourly.sort_index()

################################################

# Można sformułować problem (co dokładnie przewidujemy) na kilka sposobów

# Target 1: avg_temp_tomorrow_night
# Target 2: avg_temp_tomorrow_day
# Target 3: temp_tomorrow_same_hour
# Target 4: temp_next_hour

# A. 1 i 2 to taka predykcja dzienna - wynikiem jest średnia temperatura najbliższej nocy i kolejnego dnia
# B. 3 i 4 to predykcja godzinowa - wynikiem jest prognoza godzinowa na kolejny dzień (lub kolejny inny okres czasu)

# Tutaj przeprowadzimy ewaluację dla A, wariant B jest w `baseline_hourly_temp_prediction.py`

################################################

# Chcemy zacząć od prostych modeli pokazujące oczywiste rzeczy na temat pogody

# 1. Pogoda jutro będzie taka sama jak dzisiaj
# 2. Pogoda zawsze jest jakąś średnią z aktualnego miesiąca
# 3. Pogoda składa się z trendu sezonowego (pora roku) i lokalnego (dzień vs noc) i z losowego szumu

# Dla każdego modelu baseline zapisujemy RMSE, MAE, R2 na zbiorze testowym
# Dodatkowo informacje takie jak
# - czy bardziej mylimy się w dzień czy w nocy
# - wykres "błąd vs pora dnia"
# - dla których pór roku się bardziej mylimy a dla których mniej
# - wykres "błąd vs miesiąc"

# Zbiór testowy to ostatni rok w danych

################################################

def prep_data():
    # Define Day (06:00 - 17:59) and Night (18:00 - 05:59)
    # We aggregate by calendar date. "Night" here means hours 0-5 of this day and hours 19-23 of PREVIOUS date.
    mask_day_hours = (df_hourly.index.hour >= 6) & (df_hourly.index.hour <= 17)
    mask_night_hours = (df_hourly.index.hour >= 18) | (df_hourly.index.hour <= 5)

    # 1. Calculate 'temp_day_current'
    # Simple aggregation by date for day hours
    df_day_part = df_hourly[mask_day_hours].copy()
    series_day_temp = df_day_part["temp"].resample("D").mean()

    # 2. Calculate 'temp_night_current'
    df_night_part = df_hourly[mask_night_hours].copy()
    # Night associated with Jan 2nd = Jan 1st (18-23) + Jan 2nd (00-05)
    # Solution: Shift time FORWARD by 6 hours.
    # 18:00 (D-1) + 6h = 00:00 (D) -> Date D
    # 05:00 (D)   + 6h = 11:00 (D) -> Date D
    df_night_part["night_group_date"] = (
        df_night_part.index + pd.Timedelta(hours=6)
    ).date
    series_night_temp = df_night_part.groupby("night_group_date")["temp"].mean()
    series_night_temp.index = pd.to_datetime(series_night_temp.index)
    # print(df_night_part.head(24))

    # 3. Merge into daily DataFrame
    df_daily = pd.DataFrame(
        {"temp_day_current": series_day_temp, "temp_night_current": series_night_temp}
    )

    # Drop days with incomplete data
    df_daily = df_daily.dropna()

    # Create Targets (Shift -1 for Tomorrow)
    df_daily["target_temp_day"] = df_daily["temp_day_current"].shift(-1)
    df_daily["target_temp_night"] = df_daily["temp_night_current"].shift(-1)
    df_daily = df_daily.dropna()

    df_daily["month"] = df_daily.index.month
    # print(df_daily.head(24))
    return df_daily


################################################
# Evaluation Helper Function
################################################
def evaluate_model(y_true_day, y_pred_day, y_true_night, y_pred_night, model_name, test_df):
    print(f"\n--- {model_name} ---")

    # Calculate metrics
    metrics = {}
    for period, y_true, y_pred in [
        ("Day", y_true_day, y_pred_day),
        ("Night", y_true_night, y_pred_night),
    ]:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        metrics[period] = {"RMSE": rmse, "MAE": mae, "R2": r2}
        print(f"[{period:<5}] RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

    # Calculate errors for plotting
    test_copy = test_df.copy()
    test_copy["error_day"] = y_true_day - y_pred_day
    test_copy["error_night"] = y_true_night - y_pred_night

    # Plot Error vs Month
    fig, ax = plt.subplots(figsize=(10, 5))
    monthly_mae_day = test_copy.groupby("month")["error_day"].apply(
        lambda x: np.mean(np.abs(x))
    )
    monthly_mae_night = test_copy.groupby("month")["error_night"].apply(
        lambda x: np.mean(np.abs(x))
    )

    ax.plot(monthly_mae_day.index, monthly_mae_day.values, label="Day MAE", marker="o")
    ax.plot(
        monthly_mae_night.index, monthly_mae_night.values, label="Night MAE", marker="s"
    )

    ax.set_title(f"{model_name}: Mean Absolute Error by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("MAE")
    ax.set_xticks(range(1, 13))
    ax.legend()
    plt.tight_layout()
    # plt.show()


################################################
# Decomposition for baseline 3
################################################
def predict_decomposition(train_df, test_df, target_col, add_noise=True):
    """
    Decomposes the time series into Trend + Seasonality + Residuals.
    Predicts: Trend(test) + Seasonality(test) + RandomNoise(train_std).
    """

    # 1. Global Trend (Dopasowujemy regresję liniową do danych treningowych)
    X_train = train_df.index.map(pd.Timestamp.toordinal).values.reshape(
        -1, 1
    )  # Convert dates to ordinal numbers for regression
    y_train = train_df[target_col].values

    model_trend = LinearRegression()
    model_trend.fit(X_train, y_train)
    trend_train = model_trend.predict(X_train)

    # 2. Seasonality (Po odjęciu trendu, wyliczamy średnią dla każdego dnia roku)
    detrended_train = y_train - trend_train
    temp_train = pd.DataFrame(  # Create a temporary DF to group by day_of_year
        {"detrended": detrended_train, "day_of_year": train_df.index.dayofyear}
    )
    seasonal_profile = temp_train.groupby("day_of_year")["detrended"].mean()
    seasonality_train = train_df.index.dayofyear.map(seasonal_profile).fillna(0)

    # 3. Noise Analysis
    residuals_train = detrended_train - seasonality_train
    noise_std = residuals_train.to_numpy().std()

    # ================= PREDICTION =================

    # 1. Predict Trend for Test
    X_test = test_df.index.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    pred_trend = model_trend.predict(X_test)

    # 2. Predict Seasonality for Test
    # Map the learned seasonal profile to test dates
    pred_seasonality = test_df.index.dayofyear.map(seasonal_profile)

    # Handle missing keys (should not be present in the full dataset)
    if pred_seasonality.isna().any():
        pred_seasonality = pred_seasonality.fillna(0)

    # 3. Add Random Noise
    prediction = pred_trend + pred_seasonality
    if add_noise:
        # Generate random noise with the same std as training residuals
        np.random.seed(42)
        random_noise = np.random.normal(0, noise_std, size=len(test_df))
        prediction += random_noise

    return prediction


################################################
if __name__ == "__main__":
    # Data Preparation
    df_daily = prep_data()

    # Split (Test = Last 365 Days)
    last_timestamp = df_daily.index.max()

    cutoff_date = last_timestamp - pd.Timedelta(days=365)

    train = df_daily[df_daily.index <= cutoff_date]
    test = df_daily[df_daily.index > cutoff_date].copy()

    print(f"Train set: {train.index.min().date()} to {train.index.max().date()}")
    print(f"Test set: {test.index.min().date()} to {test.index.max().date()}")


    ################################################
    # 1. Baseline: Tomorrow will be the same as today (Persistence)
    ################################################

    pred_day_persistence = test["temp_day_current"]
    pred_night_persistence = test["temp_night_current"]

    evaluate_model(
        test["target_temp_day"],
        pred_day_persistence,
        test["target_temp_night"],
        pred_night_persistence,
        "Baseline 1: Persistence",
        test,
    )

    ################################################
    # 2. Baseline: Monthly Average (Climatology)
    ################################################

    monthly_means_day = train.groupby("month")["temp_day_current"].mean()
    monthly_means_night = train.groupby("month")["temp_night_current"].mean()

    pred_day_monthly = test["month"].map(monthly_means_day)
    pred_night_monthly = test["month"].map(monthly_means_night)

    evaluate_model(
        test["target_temp_day"],
        pred_day_monthly,
        test["target_temp_night"],
        pred_night_monthly,
        "Baseline 2: Monthly Average",
        test,
    )

    ################################################
    # 3. Baseline: Decomposition (Trend + Season + Noise)
    ################################################

    pred_day_decomp = predict_decomposition(train, test, "temp_day_current", add_noise=True)
    pred_night_decomp = predict_decomposition(
        train, test, "temp_night_current", add_noise=True
    )

    evaluate_model(
        test["target_temp_day"],
        pred_day_decomp,
        test["target_temp_night"],
        pred_night_decomp,
        "Baseline 3: Decomposition (+Noise)",
        test,
    )

    ################################################
    # 3.1 Baseline: Decomposition (Trend + Season)
    ################################################

    pred_day_decomp = predict_decomposition(
        train, test, "temp_day_current", add_noise=False
    )
    pred_night_decomp = predict_decomposition(
        train, test, "temp_night_current", add_noise=False
    )

    evaluate_model(
        test["target_temp_day"],
        pred_day_decomp,
        test["target_temp_night"],
        pred_night_decomp,
        "Baseline 3: Decomposition",
        test,
    )

    plt.show()
