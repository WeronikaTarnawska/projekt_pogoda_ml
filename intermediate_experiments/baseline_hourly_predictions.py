import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import re

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

# Tutaj przeprowadzimy ewaluację dla B

################################################

# Chcemy zacząć od prostych modeli pokazujące oczywiste rzeczy na temat pogody

# 1. Pogoda jutro będzie taka sama jak dzisiaj (dla target 3)
# 1.1. Pogoda za godzinę będzie taka jak teraz + średnia różnica temperatury między tą a kolejną godziną z przeszłości
# 2. Pogoda zawsze jest jakąś średnią z aktualnego miesiąca o tej godzinie
# 3. Pogoda składa się z trendu globalnego, sezonowego (pora roku) i lokalnego (dzień vs noc) i losowego szumu

# Dla każdego modelu baseline zapisujemy RMSE, MAE, R2 na zbiorze testowym
# Dodatkowo informacje takie jak
# - czy bardziej mylimy się w dzień czy w nocy
# - wykres "błąd vs pora dnia"
# - dla których pór roku się bardziej mylimy a dla których mniej
# - wykres "błąd vs miesiąc"

# Zbiór testowy to ostatni rok w danych (ostatnie 365 dni)

################################################


################################################
# Data Preparation
################################################
def prep_data(df_hourly):
    # Create Targets
    df_hourly["target_next_hour"] = df_hourly["temp"].shift(-1)
    df_hourly["target_tomorrow_same_hour"] = df_hourly["temp"].shift(-24)

    # Drop NaNs created by lagging/shifting
    df_hourly = df_hourly.dropna()

    # Split into Train and Test (Test = Last 365 Days)
    last_timestamp = df_hourly.index.max()
    cutoff_date = last_timestamp - pd.Timedelta(days=365)

    train = df_hourly[df_hourly.index <= cutoff_date]
    test = df_hourly[df_hourly.index > cutoff_date].copy()

    print(f"Train range: {train.index.min()} -> {train.index.max()}")
    print(f"Test range:  {test.index.min()} -> {test.index.max()}")
    return train, test


################################################
# Evaluation
################################################
def evaluate_hourly_model(y_true, y_pred, model_name, target_name=""):
    # Align indices (drop NaNs if prediction generated them)
    common_index = y_true.index.intersection(y_pred.index)
    y_true = y_true.loc[common_index]
    y_pred = y_pred.loc[common_index]

    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Print Table
    print("\n" + "=" * 60)
    print(f"{model_name}, {target_name}")
    print("-" * 60)
    print(f"{'Metric':<20} | {'Value':<10}")
    print("-" * 60)
    print(f"{'RMSE':<20} | {rmse:<10.2f}")
    print(f"{'MAE':<20} | {mae:<10.2f}")
    print(f"{'R2 Score':<20} | {r2:<10.4f}")
    print("=" * 60)

    # Analysis: Error Analysis
    errors = np.abs(y_true - y_pred)
    df_err = pd.DataFrame(
        {"error": errors, "hour": y_true.index.hour, "month": y_true.index.month}
    )

    # Plotting
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        num=f"{re.sub(r'[^\w]', '', model_name.replace(' ', '_'))}",
    )
    fig.suptitle(f"{model_name}, {target_name}")
    
    # Error vs Hour of Day
    hourly_mae = df_err.groupby("hour")["error"].mean()
    axes[0].plot(hourly_mae.index, hourly_mae.values, marker="o", color="tab:blue")
    axes[0].set_title("Mean Absolute Error vs. Hour of Day")
    axes[0].set_xlabel("Hour (0-23)")
    axes[0].set_ylabel("MAE")
    axes[0].set_xticks(range(0, 24, 2))

    # Error vs Month
    monthly_mae = df_err.groupby("month")["error"].mean()
    axes[1].plot(monthly_mae.index, monthly_mae.values, marker="s", color="tab:green")
    axes[1].set_title("Mean Absolute Error vs. Month")
    axes[1].set_xlabel("Month (1-12)")
    axes[1].set_ylabel("MAE")
    axes[1].set_xticks(range(1, 13))

    plt.tight_layout()
    # plt.show()


################################################
# 3. Baseline: Decomposition (Trend + Seasonality + Noise)
################################################
def predict_decomposition(train_df, test_df, target_col, hour_shift=0, add_noise=True):
    """
    Decomposes the time series into Trend + Seasonality + Residuals.
    Predicts temperature for time t + hour_shift.
    """
    # 1. Fit Global Linear Trend on Train
    X_train = train_df.index.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    y_train = train_df[target_col].values

    model_trend = LinearRegression()
    model_trend.fit(X_train, y_train)
    trend_train = model_trend.predict(X_train)

    train_detrended = train_df.copy()
    train_detrended["detrended"] = y_train - trend_train

    # 2. Calculate Seasonality (Month, Hour) on Train
    # Step A: Monthly
    # seasonal_profile_month = train_detrended.groupby("month")["detrended"].mean()
    # train_monthly_component = (
    #     train_df.reset_index()
    #     .merge(seasonal_profile_month.rename("monthly_seas"), on="month", how="left")
    #     ["monthly_seas"].values
    # )
    # train_detrended["residuals_after_month"] = train_detrended["detrended"] - train_monthly_component

    # Step A': Day of year
    train_detrended["day_of_year"] = train_detrended.index.dayofyear
    seasonal_profile_doy = train_detrended.groupby("day_of_year")["detrended"].mean()
    train_doy_component = (
        train_detrended["day_of_year"].map(seasonal_profile_doy).fillna(0).values
    )
    train_detrended["residuals_after_doy"] = (
        train_detrended["detrended"] - train_doy_component
    )

    # 3. Step B: Hourly (Daily Profile)
    # seasonal_profile_hour = train_detrended.groupby("hour")["residuals_after_month"].mean()
    # train_hourly_component = (
    #     train_df.reset_index()
    #     .merge(seasonal_profile_hour.rename("hourly_seas"), on="hour", how="left")
    #     ["hourly_seas"].values
    # )
    # residuals = train_detrended["residuals_after_month"].values - train_hourly_component
    seasonal_profile_hour = train_detrended.groupby("hour")[
        "residuals_after_doy"
    ].mean()
    train_hourly_component = (
        train_detrended["hour"].map(seasonal_profile_hour).fillna(0).values
    )
    residuals = train_detrended["residuals_after_doy"].values - train_hourly_component

    # 4. Calculate Noise Std
    # Reszta to: Oryginał - Trend - Miesiąc - Godzina
    noise_std = np.std(residuals)

    # --- PREDICTION (Predict for t + shift) ---
    target_time_index = test_df.index + pd.Timedelta(hours=hour_shift)
    # A. Predict Trend
    X_test = target_time_index.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    pred_trend = model_trend.predict(X_test)

    # B. Predict Monthly Component
    pred_doy_component = (
        target_time_index.dayofyear.map(seasonal_profile_doy).fillna(0).values
    )

    # C. Predict Hourly Component
    pred_hourly_component = (
        target_time_index.hour.map(seasonal_profile_hour).fillna(0).values
    )
    # D. Combine All
    prediction = pred_trend + pred_doy_component + pred_hourly_component

    if add_noise:
        np.random.seed(42)
        prediction += np.random.normal(0, noise_std, size=len(test_df))

    # Return as Series with original index (so it aligns with y_true)
    return pd.Series(prediction, index=test_df.index)


train, test = prep_data(df_hourly)

################################################
# 1. Baseline: Tomorrow same hour (Persistence 24h)
################################################

pred_persistence_24h = test["temp"]  # Today's temp is the prediction for tomorrow

evaluate_hourly_model(
    test["target_tomorrow_same_hour"],
    pred_persistence_24h,
    "Baseline 1: Persistence (24h)",
    "target_tomorrow_same_hour",
)

################################################
# 1.1 Baseline: Next Hour Persistence + Delta
################################################
# "Temp in 1 hour = Current Temp + Average change for this hour"
# Change = Temp(t+1) - Temp(t)
train_diffs = train.copy()
train_diffs["diff"] = train_diffs["target_next_hour"] - train_diffs["temp"]
hourly_changes = train_diffs.groupby("hour")["diff"].mean()

# Predict: Take current temp in test + look up average change for that hour
pred_persistence_1h = test["temp"] + test["hour"].map(hourly_changes)

evaluate_hourly_model(
    test["target_next_hour"],
    pred_persistence_1h,
    "Baseline 1.1: Persistence (1h) + Delta",
    "target_next_hour",
)

################################################
# 2. Baseline: Monthly-Hourly Average (Climatology)
################################################
# "Temp is always the average for that specific Month and Hour"
lookup_table = train.groupby(["month", "hour"])["temp"].mean()
lookup_table.name = "avg_temp"

# Case A: Predicting Tomorrow Same Time (t + 24h)
test_with_pred_24h = (
    test.reset_index()
    .merge(lookup_table, on=["month", "hour"], how="left")
    .set_index("time")
)
pred_climatology_24h = test_with_pred_24h["avg_temp"]
evaluate_hourly_model(
    test["target_tomorrow_same_hour"],
    pred_climatology_24h,
    "Baseline 2: Climatology",
    "Target: Tomorrow Same Hour",
)
# Case B: Predicting Next Hour (t + 1h)
# If it is 12:00 now, we want to know the typical temp for 13:00.
test_copy = test.copy()
test_copy["lookup_hour"] = (test_copy["hour"] + 1) % 24
test_with_pred_1h = (
    test_copy.reset_index()
    .merge(
        lookup_table,
        left_on=["month", "lookup_hour"],  # Join test's "future hour"
        right_on=["month", "hour"],  # With lookup's "hour"
        how="left",
    )
    .set_index("time")
)
pred_climatology_1h = test_with_pred_1h["avg_temp"]
evaluate_hourly_model(
    test["target_next_hour"],
    pred_climatology_1h,
    "Baseline 2: Climatology",
    "Target: Next Hour",
)

################################################
# 3. Baseline: Decomposition (Trend + Seasonality + Noise)
################################################
pred_decomp_24h_n = predict_decomposition(
    train, test, "temp", hour_shift=24, add_noise=True
)
evaluate_hourly_model(
    test["target_tomorrow_same_hour"],
    pred_decomp_24h_n,
    "Baseline 3: Decomposition Noise=True",
    "target_tomorrow_same_hour",
)

################################################
# 3.1 Baseline: Decomposition (Trend + Seasonality)
################################################
# The two cases give almost identical results, but I checket just for completeness

# Case A: Predict Next Hour (Shift = +1)
pred_decomp_next_hour = predict_decomposition(
    train, test, "temp", hour_shift=1, add_noise=False
)

evaluate_hourly_model(
    test["target_next_hour"],
    pred_decomp_next_hour,
    "Baseline 3: Decomposition (Next Hour)",
    "target_next_hour",
)

# Case B: Predict Tomorrow Same Hour (Shift = +24)
# For cyclical features (hour), t and t+24 are identical,
# but the linear trend may be slightly different (1 day further).
pred_decomp_24h = predict_decomposition(
    train, test, "temp", hour_shift=24, add_noise=False
)
evaluate_hourly_model(
    test["target_tomorrow_same_hour"],
    pred_decomp_24h,
    "Baseline 3: Decomposition (Tomorrow Same Hour)",
    "target_tomorrow_same_hour",
)

plt.show()
