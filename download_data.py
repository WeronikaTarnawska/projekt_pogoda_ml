from datetime import datetime
import meteostat as ms
import pandas as pd
import numpy as np
from pathlib import Path

START = datetime(2015, 1, 1)
END = datetime(2025, 12, 31)
ms.config.block_large_requests = False
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

STATIONS = {
    "Wroclaw": "12424",  # Cel
    "Legnica": "12415",  # Zachód
    "Opole": "12530",  # Wschód
    "Poznan": "12330",  # Północ
    "Klodzko": "12520",  # Południe
}
MAIN_STATION = "Wroclaw"

print(f"Stations: {STATIONS}\nDownloading data...")

for name, station_id in STATIONS.items():
    print(f"{name} (ID: {station_id})...")
    df = ms.hourly(station_id, START, END).fetch()
    df.to_csv(DATA_DIR / f"{name.lower()}.csv")

print(f"{MAIN_STATION} daily data...")
df_daily = ms.daily(STATIONS[MAIN_STATION], START, END).fetch()
df_daily.to_csv(DATA_DIR / f"{MAIN_STATION.lower()}_daily.csv")

print(f"Saved in {DATA_DIR}")