

# %%

# %% [markdown]
# ## Stap 1: Sensor Data Inladen & Initialiseren
#

# %%
import pandas as pd
import numpy as np
import requests
import logging
import json  
from datetime import datetime, timedelta
import matplotlib
from sklearn.metrics import mean_absolute_error, mean_squared_error


# === CONFIG ===

SENSOR_API_URL = "http://pi.local:8000/data"
SAMPLE_DATA_FILE = "extended_data_apr22_to_may28.json"  # alternatief voor API
USE_SAMPLE = True

TIMEZONE = "Europe/Brussels"
CO2_OUTLIER_THRESHOLD = 5000  # ppm

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def load_and_clean_sensor_data(api_url: str, use_sample: bool = False) -> pd.DataFrame:
    """Haalt sensordata op via API of laadt lokale sample JSON, converteert timestamps correct, verwijdert CO₂-outliers, en normaliseert presence."""
    if use_sample:
        logger.info("Laden van sample data uit %s", SAMPLE_DATA_FILE)
        with open(SAMPLE_DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        logger.info("Ophalen sensordata vanaf %s", api_url)
        resp = requests.get(api_url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

    df = pd.DataFrame(data)

    # === Timestamp verwerken ===
    logger.info("Converteren timestamps naar datetime (UTC) en dan Europe/Brussels")
    df['ds'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df['ds'] = df['ds'].dt.tz_convert('Europe/Brussels')

    # Slechte timestamps loggen & verwijderen
    bad_timestamps = df['ds'].isna().sum()
    if bad_timestamps > 0:
        logger.warning("%d timestamps konden niet worden geconverteerd en worden verwijderd", bad_timestamps)
        df = df.dropna(subset=['ds'])

    # === Outlier detectie voor CO₂ ===
    outlier_mask = df['co2'] > CO2_OUTLIER_THRESHOLD
    num_outliers = outlier_mask.sum()
    if num_outliers > 0:
        logger.warning("%d CO₂-metingen boven %d ppm (outlier), worden verwijderd", num_outliers, CO2_OUTLIER_THRESHOLD)
        df = df[~outlier_mask]

    # === Presence normaliseren ===
    df['presence'] = df['presence'].astype(bool)

    # === Sorteren op tijd ===
    df = df.sort_values('ds').reset_index(drop=True)

    logger.info("Sensordata succesvol verwerkt: %d rijen over", len(df))
    return df


# Switch tussen API en JSON sample via USE_SAMPLE
sensor_df = load_and_clean_sensor_data(SENSOR_API_URL, use_sample=USE_SAMPLE)

# === Preview van de belangrijkste kolommen ===
print(sensor_df[['ds', 'co2', 'temperature', 'humidity', 'presence']].tail())


# %% [markdown]
# # %% [markdown]
# ## Stap 2: Dagelijkse aggregatie voorbereiden voor Prophet  
# We zetten de tijdstempels om naar datum, groeperen per dag en berekenen gemiddelden en presence‐ratio.
# 

# %%
# %%
import logging

logger.info("Stap 2: Dagelijkse aggregatie uitvoeren")

# Maak een copy en extract date-only (naive datetime)
sensor_daily = sensor_df.copy()
sensor_daily['date'] = sensor_daily['ds'].dt.tz_convert(None).dt.date

# Groeperen per dag: gemiddelden en aanwezigheid
df_daily = (
    sensor_daily
    .groupby('date')
    .agg(
        co2_mean=('co2', 'mean'),
        temp_mean=('temperature', 'mean'),
        hum_mean=('humidity', 'mean'),
        presence_count=('presence', 'sum'),
        total_count=('co2', 'count')
    )
    .reset_index()
    .rename(columns={'date': 'ds'})
)

# Zet 'ds' om naar datetime op middernacht
df_daily['ds'] = pd.to_datetime(df_daily['ds'])

SLEEP_HOURS = 8

# Zorg dat presence_count + slaapuren niet boven 24 uitkomt
df_daily['presence_count'] = (df_daily['presence_count'] + SLEEP_HOURS).clip(upper=24)

# Daarna de ratio uitrekenen
df_daily['presence_ratio'] = df_daily['presence_count'] / df_daily['total_count']


# Controle: voldoende dagen voor Prophet
MIN_DAYS = 7
if df_daily.shape[0] < MIN_DAYS:
    logger.error("Te weinig dagen data: %d (minimaal %d nodig)", df_daily.shape[0], MIN_DAYS)
    raise ValueError(f"Minimaal {MIN_DAYS} dagen data vereist, gevonden: {df_daily.shape[0]}")

logger.info("Dagelijkse aggregatie compleet: %d dagen", df_daily.shape[0])

# Preview
print(df_daily)


# %% [markdown]
# # %% [markdown]
# ## Stap 3: Weerdata ophalen van Open-Meteo API  
# We halen dagelijkse weerdata op voor dezelfde periode als onze sensor‐data en loggen eventuele fouten.
# 

# %%
# %%
import logging
import requests

logger.info("Stap 3: Weerdata ophalen van Open-Meteo API")

# === CONFIG WEERDATA ===
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
LAT, LON = 50.8503, 4.3517  # Brussel
START_DATE = df_daily['ds'].min().strftime("%Y-%m-%d")
END_DATE   = df_daily['ds'].max().strftime("%Y-%m-%d")

params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "daily": ",".join([
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "cloudcover_mean",
        "wind_speed_10m_max",
        "shortwave_radiation_sum"
    ]),
    "timezone": "Europe/Brussels"
}

try:
    resp = requests.get(WEATHER_API_URL, params=params, timeout=10)
    resp.raise_for_status()
    weather_json = resp.json()
    logger.info("Weerdata succesvol opgehaald voor %s to %s", START_DATE, END_DATE)
except requests.RequestException as e:
    logger.error("Fout bij ophalen weerdata: %s", e)
    raise

# Zet om naar DataFrame
weather_df = pd.DataFrame({
    'ds': pd.to_datetime(weather_json['daily']['time']),
    'outside_temp_max': weather_json['daily']['temperature_2m_max'],
    'outside_temp_min': weather_json['daily']['temperature_2m_min'],
    'precipitation': weather_json['daily']['precipitation_sum'],
    'cloudcover': weather_json['daily']['cloudcover_mean'],
    'wind': weather_json['daily']['wind_speed_10m_max'],
    'solar_radiation': weather_json['daily']['shortwave_radiation_sum']
})

# Preview
print(weather_df)


# %% [markdown]
# # %% [markdown]
# ## Stap 4: Data Mergen & Feature Engineering  
# We combineren de dagelijkse sensordata met de weerdata, creëren lag-features, context-variabelen en cyclische dagencode.
# 

# %%
# %% 
# === Stap 4: Data Mergen & Feature Engineering ===
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
logger.info("Stap 4: Data Mergen & Feature Engineering starten")

# (1) Belgische feestdagen laden
current_year = datetime.now().year
try:
    import holidays
    COUNTRY_HOLIDAYS = holidays.Belgium(years=current_year)
except ImportError:
    logger.warning("holidays library niet beschikbaar, is_holiday wordt altijd False")
    COUNTRY_HOLIDAYS = []

# (2) Merge dagelijkse sensordata met weerdata
combined_df = pd.merge(df_daily, weather_df, on='ds', how='inner')
logger.info("Data gemerged: %d regels", combined_df.shape[0])

# (3) Bereken temperatuurverschil binnen–buiten
combined_df['temp_diff'] = combined_df['temp_mean'] - combined_df['outside_temp_max']

# (4) Lag-features voor temperatuur (1 & 2 dagen terug)
combined_df['temp_lag1'] = combined_df['temp_mean'].shift(1)
combined_df['temp_lag2'] = combined_df['temp_mean'].shift(2)

# (5) Lag-features voor CO₂ (1 & 2 dagen terug)
combined_df['co2_lag1'] = combined_df['co2_mean'].shift(1)
combined_df['co2_lag2'] = combined_df['co2_mean'].shift(2)

# (6) Eerst de lag-features opvullen, zodat we geen onnodige rijen dropen
for col in ['temp_lag1','temp_lag2','co2_lag1','co2_lag2']:
    combined_df[col] = combined_df[col].bfill()


# (7) Uurlijkse CO₂-diff en rapid rise flag
combined_df['co2_diff1']  = combined_df['co2_mean'] - combined_df['co2_mean'].shift(1)
threshold = 100
combined_df['rapid_rise'] = combined_df['co2_diff1'] > threshold

# (8) Context-variabelen
combined_df['is_weekend'] = combined_df['ds'].dt.dayofweek >= 5
combined_df['is_holiday'] = combined_df['ds'].dt.date.isin(COUNTRY_HOLIDAYS)

# (9) Cyclische encoding dag van de week
combined_df['dayofweek'] = combined_df['ds'].dt.dayofweek
combined_df['day_sin']   = np.sin(2 * np.pi * combined_df['dayofweek'] / 7)
combined_df['day_cos']   = np.cos(2 * np.pi * combined_df['dayofweek'] / 7)

# (10) Drop NaN (bv. uit co2_diff1 of holiday-lag)
combined_df = combined_df.dropna().reset_index(drop=True)
logger.info("Na feature engineering: %d rijen over", combined_df.shape[0])

# Preview
print(combined_df.columns.tolist())
print(combined_df.tail())



# %% [markdown]
# ## Stap 5: Modellen trainen  
# We splitsen dit in twee delen:
# 1. **5.1 — Prophet** voor temperatuurvoorspelling op dagbasis  
# 2. **5.2 — SARIMAX** voor CO₂-voorspelling op dagbasis  
# 

# %%
# === Stap 5.1 — Prophet-model voor temperatuur (met alle features) ===
import logging
from prophet import Prophet

logger.info("Stap 5.1: Prophet-model trainen voor temperatuur")

# Voorbereiding van het dataframe
prophet_df = combined_df[[
    'ds', 'temp_mean',
    'outside_temp_max','outside_temp_min','precipitation',
    'cloudcover','wind','solar_radiation',
    'is_weekend','is_holiday',
    'temp_lag1','temp_lag2',
    'day_sin','day_cos'
]].rename(columns={'temp_mean': 'y'}).copy()

# Check op NaNs
regressors = [
    'outside_temp_max','outside_temp_min','precipitation',
    'cloudcover','wind','solar_radiation',
    'is_weekend','is_holiday',
    'temp_lag1','temp_lag2','day_sin','day_cos'
]

expected_cols = ['ds', 'y'] + regressors
missing = set(expected_cols) - set(prophet_df.columns)
if missing:
    raise ValueError(f"❌ Ontbrekende kolommen voor Prophet: {missing}")

if prophet_df[expected_cols].isna().any().any():
    logger.warning("⚠️ Waarschuwing: Prophet-trainingsdata bevat NaNs! NaN-aantallen per kolom:")
    print(prophet_df[expected_cols].isna().sum())
    prophet_df[expected_cols] = prophet_df[expected_cols].fillna(method='bfill')

# Anti-overfitting Prophet model
data_span_days = (prophet_df['ds'].max() - prophet_df['ds'].min()).days
print(f"📅 Data span: {data_span_days} dagen")

if data_span_days < 30:
    print("⚠️ WAARSCHUWING: Dataset te klein voor betrouwbare Prophet voorspellingen")
    print("Voor beste resultaten: verzamel minimaal 30 dagen data")

# AANGEPAST Prophet-model (anti-overfitting)
temp_model = Prophet(
    growth='linear',                    # Simpel lineair ipv fancy growth
    yearly_seasonality=False,           # Uit - te weinig data
    weekly_seasonality=False if data_span_days < 21 else True,  # Alleen bij >3 weken
    daily_seasonality=False,            # Uit - er zijn al dag_sin/cos regressors
    seasonality_mode='additive',        # Eenvoudiger dan multiplicative
    changepoint_prior_scale=0.001,      # VEEL LAGER (was default 0.05)
    seasonality_prior_scale=0.01,       # LAGER (was default 10.0)
    holidays_prior_scale=0.01,          # LAGER (was default 10.0)
    n_changepoints=0 if data_span_days < 21 else 5,  # Weinig/geen changepoints
    interval_width=0.80                 # Smallere intervals
)

# **MINDER REGRESSORS** voor kleine datasets
if data_span_days < 21:
    # Alleen de belangrijkste regressors bij weinig data
    key_regressors = ['outside_temp_max', 'outside_temp_min', 'temp_lag1']
    logger.info(f"⚠️ Beperkte regressors door kleine dataset: {key_regressors}")
else:
    key_regressors = regressors

# Voeg regressors toe
for reg in key_regressors:
    temp_model.add_regressor(reg, prior_scale=0.1)  #LAGE prior_scale

# Train model
temp_model.fit(prophet_df)
logger.info("✅ Prophet-model voor temperatuur succesvol getraind op %d dagen", prophet_df.shape[0])


# %%
## Stap 5.2 — SARIMAX-model voor CO₂ (met AR(2) voor expliciete lag-1 & lag-2)

import logging
import statsmodels.api as sm


# 1) Definieer target en exogene variabelen
co2_series = combined_df.set_index('ds')['co2_mean'].astype(float)
exog = combined_df.set_index('ds')[[
    'presence_ratio', 'hum_mean', 'is_weekend',
    'is_holiday',
    'co2_diff1', 'rapid_rise'
]].astype(float)

# 2) Bouw SARIMAX: order=(2,0,1) bevat co2_lag1 & co2_lag2 in AR
model = sm.tsa.statespace.SARIMAX(
    co2_series,
    exog=exog,
    order=(2,0,1),
    trend='t',
    seasonal_order=(0,0,0,0),
    enforce_stationarity=False,
    enforce_invertibility=False
)
co2_results = model.fit(disp=False)

# 3) Modeldiagnose: toon AIC en p-waarden
logger.info("AIC SARIMAX CO₂: %.1f", co2_results.aic)
print(co2_results.summary().tables[1])

# %% [markdown]
# ## Stap 6: Hold-Out Evaluatie (20% laatste data) -> sarimax

# %%
import logging
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
import statsmodels.api as sm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Dynamische split: laatste 20% als testset
n = combined_df.shape[0]
test_size = max(int(n * 0.2), 1)
train_df = combined_df.iloc[: n - test_size].copy()
test_df  = combined_df.iloc[n - test_size :].copy()

# Evalueert SARIMAX op de laatste 20% van de data (out-of-sample test)
logger.info(f"Hold-Out: {train_df.shape[0]} train / {test_df.shape[0]} test")

if train_df.shape[0] < 2:
    logger.warning("Niet genoeg data voor hold-out evaluatie, stap 6 wordt overgeslagen")
else:
   

    # — SARIMAX hold-out —
    co2_train  = train_df.set_index('ds')['co2_mean']
    exog_train = train_df.set_index('ds')[[
        'presence_ratio','hum_mean', 'is_weekend',
        'is_holiday','co2_diff1','rapid_rise'
    ]].astype(float)

    model_hold = sm.tsa.statespace.SARIMAX(
        co2_train, exog=exog_train,
        order=(2,0,1), trend='t',
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res_hold = model_hold.fit(disp=False)

    exog_test = test_df.set_index('ds')[[
        'presence_ratio','hum_mean','is_weekend',
        'is_holiday','co2_diff1','rapid_rise'
    ]].astype(float)

    # SARIMAX hold-out forecast = voorspellen op testdata
    fc_hold = res_hold.get_forecast(steps=len(exog_test), exog=exog_test)

    y_true_c = test_df['co2_mean'].values
    y_pred_c = fc_hold.predicted_mean.values
    #Deze MAE & RMSE zijn op testdata (dus echte generalisatie)
    mae_c = mean_absolute_error(y_true_c, y_pred_c)
    rmse_c = np.sqrt(mean_squared_error(y_true_c, y_pred_c))
    logger.info(f"📉 SARIMAX Hold-Out MAE (test): {mae_c:.3f}, RMSE: {rmse_c:.3f}")



# %% [markdown]

# ## Stap 7: Prophet Cross-Validatie (out-of-sample performance)

# ## Prophet Cross-Validatie
from prophet.diagnostics import cross_validation, performance_metrics

# Bereken totale looptijd van de dataset
data_span_days = (prophet_df['ds'].max() - prophet_df['ds'].min()).days
print(f"📅 Data span: {data_span_days} dagen")

# horizon en periodes
if data_span_days >= 14:  
    horizon_days = 1        
    initial_days = max(7, data_span_days - 3)  
    period_days = 1
    
    logger.info(f"Voer Prophet cross-validatie uit: initial={initial_days}d, period={period_days}d, horizon={horizon_days}d")
    
    try:
        # Prophet time-based cross-validation
        df_cv = cross_validation(
            temp_model,
            initial=f"{initial_days} days",
            period=f"{period_days} day", 
            horizon=f"{horizon_days} days",
            parallel="threads"
        )
        
        # Bereken foutmaten
        df_p = performance_metrics(df_cv)
        mean_mae = df_p['mae'].mean()
        mean_rmse = df_p['rmse'].mean()
        
        # Debugging info
        print(f"Aantal cutoffs: {df_cv['cutoff'].nunique()}")
        print(f"Totale rijen in prophet_df: {len(prophet_df)}")
        
        # Vergelijk met simpele baseline
        y_actual = df_cv['y']
        persistence_mae = abs(df_cv['y'] - df_cv['y'].shift(1)).mean()
        
        print("\n📊 Sample van voorspellingen en targets:")
        print(df_cv[['ds', 'y', 'yhat']].head())
        
        print("\n📉 Prophet Cross-Validatie Resultaten:")
        print(df_p[['horizon', 'mae', 'rmse']])
        print(f"\n✅ Prophet test MAE: {mean_mae:.3f}")
        print(f"✅ Prophet test RMSE: {mean_rmse:.3f}")
        print(f"📊 Persistence baseline MAE: {persistence_mae:.3f}")
        
        # Waarschuwing bij slechte prestaties
        if mean_mae > persistence_mae * 1.5:
            print("⚠️  WAARSCHUWING: Prophet presteert slechter dan simpele persistence!")
            print("   Overweeg meer data te verzamelen of eenvoudiger model te gebruiken")
            
    except Exception as e:
        logger.error(f"Cross-validatie gefaald: {e}")
        print("❌ Cross-validatie niet mogelijk - dataset te klein of instabiel")
        
else:
    logger.warning(
        f"Niet genoeg data voor betrouwbare cross-validatie "
        f"(data span {data_span_days} dagen), stap 7 wordt overgeslagen"
    )
    
    # Alternatieve evaluatie voor zeer kleine datasets
    print("\n📊 Alternatieve evaluatie op laatste 20% van data:")
    split_idx = int(len(prophet_df) * 0.8)
    train_data = prophet_df.iloc[:split_idx]
    test_data = prophet_df.iloc[split_idx:]
    
    if len(test_data) >= 2:
        # Simpele train/test split
        simple_model = Prophet(
            growth='linear',
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.001,
            n_changepoints=0
        )
        for reg in key_regressors:
            simple_model.add_regressor(reg, prior_scale=0.1)
        
        simple_model.fit(train_data)
        forecast = simple_model.predict(test_data)
        
        simple_mae = abs(test_data['y'] - forecast['yhat']).mean()
        print(f"✅ Eenvoudige test MAE: {simple_mae:.3f}")


# %% [markdown]
# ## Stap 8: SARIMAX Walk-Forward Cross-Validatie

# %%
import numpy as np
import statsmodels.api as sm

# Sorteer data en bepaal start voor walk-forward
dates     = combined_df['ds'].sort_values().tolist()
n_initial = int(len(dates) * 0.6)
errors    = []

# Exact dezelfde exogene kolommen als in Stap 5.2
# Evalueert SARIMAX via rolling window: telkens 1 stap vooruit voorspellen
# Gebruik dezelfde exogene features als training
exog_cols = [ 'presence_ratio', 'hum_mean', 'is_weekend', 'is_holiday', 'co2_diff1', 'rapid_rise' ]


for i in range(n_initial, len(dates) - 1):
    # Splits
    train_slice = combined_df[combined_df['ds'] <= dates[i]]
    test_date   = dates[i + 1]

    # (1) Endogeen naar float
    co2_tr = train_slice.set_index('ds')['co2_mean'].astype(float)

    # (2) Exogeen naar float
    ex_tr = (
        train_slice
        .set_index('ds')[exog_cols]
        .astype(float)
    )

    # Bouw en fit SARIMAX
    mod_sf = sm.tsa.statespace.SARIMAX(
        co2_tr,
        exog=ex_tr,
        order=(2,0,1),
        trend='t',
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res_sf = mod_sf.fit(disp=False)

    # Ook de toekomstige exogenen casten
    ex_next = (
        combined_df[combined_df['ds'] == test_date]
        .set_index('ds')[exog_cols]
        .astype(float)
    )

    # Forecast en error
    pred   = res_sf.get_forecast(steps=1, exog=ex_next).predicted_mean.iloc[0]
    actual = combined_df.loc[combined_df['ds'] == test_date, 'co2_mean'].iloc[0]
    errors.append(actual - pred)

# Bereken metrics
# Dit is een test-gebaseerde evaluatie: telkens 1 stap vooruit
# Resultaten zijn realistisch en tonen hoe goed het model generaliseert
mae_rf  = np.mean(np.abs(errors))
rmse_rf = np.sqrt(np.mean(np.array(errors) ** 2))
print(f"📉 SARIMAX Rolling MAE (test): {mae_rf:.2f}, RMSE: {rmse_rf:.2f}")

metrics = pd.DataFrame({
    'Model (Test)': ['Prophet Temperature', 'SARIMAX CO₂'],
    'MAE': [mean_mae, mae_rf],
    'RMSE': [mean_rmse, rmse_rf]
})

print("\n📊 Test evaluatie (testdata, fitted values):")
print(metrics.to_string(index=False))



# %% [markdown]
# # %% [markdown]
# ## Stap 9: Forecast voor de komende week  
# We halen weerdata op voor de volgende 7 dagen, maken regressors aan, en voorspellen met Prophet (temperatuur) en SARIMAX (CO₂).
# 

# %%
# %%


# === 6.1 — Datumperiode definiëren ===
last_date    = combined_df['ds'].max()
future_start = last_date + timedelta(days=1)
future_end   = future_start + timedelta(days=6)

# === 6.2 — Weerforecast ophalen ===
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": future_start.strftime("%Y-%m-%d"),
    "end_date":   future_end.strftime("%Y-%m-%d"),
    "daily": ",".join([
        "temperature_2m_max", "temperature_2m_min",
        "precipitation_sum", "cloudcover_mean",
        "wind_speed_10m_max", "shortwave_radiation_sum"
    ]),
    "timezone": "Europe/Brussels"
}
weather_fc = requests.get(WEATHER_API_URL, params=params).json()

future_weather = pd.DataFrame({
    'ds':               pd.to_datetime(weather_fc['daily']['time']),
    'outside_temp_max': weather_fc['daily']['temperature_2m_max'],
    'outside_temp_min': weather_fc['daily']['temperature_2m_min'],
    'precipitation':    weather_fc['daily']['precipitation_sum'],
    'cloudcover':       weather_fc['daily']['cloudcover_mean'],
    'wind':             weather_fc['daily']['wind_speed_10m_max'],
    'solar_radiation':  weather_fc['daily']['shortwave_radiation_sum']
})

# === 6.2.1 — Context, cyclische encoding & sleep ratio ===
future_weather['is_weekend']       = future_weather['ds'].dt.dayofweek >= 5
future_weather['is_holiday']       = future_weather['ds'].dt.date.isin(COUNTRY_HOLIDAYS if COUNTRY_HOLIDAYS else [])
dow = future_weather['ds'].dt.dayofweek
future_weather['day_sin']          = np.sin(2 * np.pi * dow / 7)
future_weather['day_cos']          = np.cos(2 * np.pi * dow / 7)
# future_weather['sleep_time_ratio'] = 8 / 24  # aangenomen 8u slaap per dag

# === 6.2.2 — Vul overige exogenen in met defaults ===
future_weather['presence_ratio'] = 0.3 
future_weather['hum_mean']       = combined_df['hum_mean'].iloc[-1]
future_weather['temp_diff']      = 0.0
future_weather['co2_diff1']      = 0.0
future_weather['rapid_rise']     = False

# === 6.3 — Iteratieve temperatuurvoorspelling met Prophet ===
avg_temp_diff = (combined_df['temp_mean'] - combined_df['outside_temp_max']).mean()
prev_preds    = list(combined_df['temp_mean'].iloc[-2:])

lag1_list = []
lag2_list = []
forecast_temps = []

for _, row in future_weather.iterrows():
    lag1_list.append(prev_preds[-1])
    lag2_list.append(prev_preds[-2])
    
    feat = row.copy()
    feat['temp_lag1'] = prev_preds[-1]
    feat['temp_lag2'] = prev_preds[-2]
    
    yhat = temp_model.predict(pd.DataFrame([feat]))['yhat'].iloc[0]
    low  = feat['outside_temp_max']
    high = low + avg_temp_diff
    yhat = np.clip(yhat, low, high)
    
    future_weather.loc[future_weather['ds'] == feat['ds'], 'temp_diff'] = yhat - low
    forecast_temps.append(yhat)
    prev_preds.append(yhat)

future_weather['temp_lag1'] = lag1_list
future_weather['temp_lag2'] = lag2_list
future_weather['pred_temp'] = forecast_temps

# — 6.3b — Maak forecast_temp van future_weather —  
forecast_temp = future_weather[['ds', 'pred_temp']].copy()

# — 6.3c — Forecast CO₂ met SARIMAX —
exog_cols = [
    'presence_ratio', 'hum_mean', 'is_weekend',
    'is_holiday',
    'co2_diff1', 'rapid_rise'
]
exog_future = future_weather.set_index('ds')[exog_cols].astype(float)
co2_fc = co2_results.get_forecast(steps=len(exog_future), exog=exog_future)

forecast_co2 = pd.DataFrame({
    'pred_co2': co2_fc.predicted_mean.values,
    'ds': future_weather['ds'].values
})

# — 6.4 — Samenvoegen van temperatuur, CO₂ en alle features —
forecast_df = (
    forecast_temp
    .merge(forecast_co2, on='ds')
    .merge(
        future_weather.drop(columns=['pred_temp'], errors='ignore'),
        on='ds'
    )[[ 
        'ds',
        'pred_temp',
        'outside_temp_max','outside_temp_min','precipitation',
        'cloudcover','wind','solar_radiation',
        'is_weekend','is_holiday',
        'pred_co2'
    ]]
)

print("Forecast met alle gebruikte features:")
print(forecast_df.to_string(index=False))


# %% [markdown]
# # %% [markdown]
# ## Stap 10: Visualisatie & evaluatie  
# We plotten de voorspelde binnentemperatuur en CO₂-niveaus voor de komende week, inclusief comfort- en adviesgrenzen.
# 

# %%
# %%
import pandas as pd
import matplotlib.pyplot as plt

# Zorg dat 'ds' datetime is
forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])

# plt.style.use("seaborn-whitegrid")

# — Plot voorspelde binnentemperatuur —
plt.figure(figsize=(10, 5))
plt.plot(forecast_df['ds'], forecast_df['pred_temp'], marker='o', label='Voorspelde Temp (°C)')
plt.axhline(18, color='red', linestyle='--', label='Ondergrens Comfort (18 °C)')
plt.axhline(25, color='red', linestyle='--', label='Bovengrens Comfort (25 °C)')
plt.title("🌡️ Voorspelde Binnentemperatuur Volgende Week")
plt.xlabel("Datum")
plt.ylabel("Temperatuur (°C)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("Voorspelde_Binnentemperatuur_Volgende_Week.png")

# — Plot voorspelde CO₂-niveaus —
plt.figure(figsize=(10, 5))
plt.plot(forecast_df['ds'], forecast_df['pred_co2'], marker='o', label='Voorspelde CO₂ (ppm)')
plt.axhline(1000, color='red', linestyle='--', label='Adviesgrens CO₂ (1000 ppm)')
plt.title("🫁 Voorspeld CO₂-niveau Volgende Week")
plt.xlabel("Datum")
plt.ylabel("CO₂ (ppm)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("Voorspelde_CO₂-niveau_Volgende_Week.png")


# %% [markdown]
# 
# Grafiek 1 toont voor elke dag de voorspelde binnentemperatuur, met horizontale rode stippellijnen bij 18 °C en 25 °C als comfortgrenzen.
# 
# Grafiek 2 toont de voorspelde CO₂-concentratie, met een rode stippellijn bij 1000 ppm als adviesgrens.
# 

# %% [markdown]
# # %% [markdown]
# ## Stap 11: Model Evaluatie  
# Hier berekenen we de MAE en RMSE voor zowel in-sample temperatuur (Prophet) als CO₂ (SARIMAX).
# 

# %%
# %%
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# — 8.1 In-sample evaluatie Prophet (training data) —
prophet_df = combined_df[[
    'ds', 'temp_mean',
    'outside_temp_max', 'outside_temp_min', 'precipitation',
    'cloudcover', 'wind', 'solar_radiation',
    'is_weekend', 'is_holiday',
    'temp_lag1', 'temp_lag2',
    'day_sin', 'day_cos'
]].rename(columns={'temp_mean': 'y'})

insample = prophet_df.copy()
insample['yhat'] = temp_model.predict(insample)['yhat']
insample['real'] = insample['y']
mae_temp = mean_absolute_error(insample['real'], insample['yhat'])
rmse_temp = np.sqrt(mean_squared_error(insample['real'], insample['yhat']))

# — 8.2 In-sample evaluatie SARIMAX (training data) —
co2_series = combined_df.set_index('ds')['co2_mean']
co2_pred   = co2_results.fittedvalues  # fittedvalues = in-sample predictions
mae_co2    = mean_absolute_error(co2_series, co2_pred)
rmse_co2   = np.sqrt(mean_squared_error(co2_series, co2_pred))

# — 8.3 Metrics samenvatten: let op => enkel in-sample —
metrics = pd.DataFrame({
    'Model (Train/In-sample)': ['Prophet Temperature', 'SARIMAX CO₂'],
    'MAE': [mae_temp, mae_co2],
    'RMSE': [rmse_temp, rmse_co2]
})

print("\n📊 In-sample evaluatie (trainingsdata, fitted values):")
print(metrics.to_string(index=False))



# %% [markdown]
# # %% [markdown]
# ## Stap 12: Dagelijkse adviezen genereren  
# We doorlopen elke dag uit `forecast_df` en voegen adviezen toe op basis van de voorspelde CO₂- en temperatuurniveaus.
# 



# %%
# %%
# Adviezen genereren
advies_lijst = []
for _, row in forecast_df.iterrows():
    adviezen = []
    if row['pred_co2'] > 1000:
        adviezen.append("💨 Ventileer: CO₂ > 1000 ppm")
    if row['pred_temp'] > 25:
        adviezen.append("🔥 Zet verwarming lager (te warm)")
    elif row['pred_temp'] < 18:
        adviezen.append("❄️ Bijverwarmen aanbevolen (te koud)")
    if not adviezen:
        adviezen.append("✅ Alles binnen comfort en luchtkwaliteit")
    advies_lijst.append(" | ".join(adviezen))

forecast_df['advies'] = advies_lijst

# Resultaat tonen
print(forecast_df.to_string(index=False))

# — 6.6 — Forecast + Adviezen bewaren naar JSON —

# Converteer eerst de datumkolom naar strings
forecast_df['ds'] = forecast_df['ds'].dt.strftime("%Y-%m-%dT%H:%M:%S")

# Dan opslaan
output_path = "forecast_result.json"

# Eerst omzetten naar een Python list
forecast_records = forecast_df.to_dict(orient="records")

# Dan opslaan met mooie indentatie
import json
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(forecast_records, f, ensure_ascii=False, indent=4)

print(f"\nForecast inclusief adviezen succesvol opgeslaan als mooie JSON in {output_path}!")

import os
# Forecast ook toevoegen aan forecast_history.json (als lijst van lijsten)
history_path = "forecast_history.json"

# Laad bestaande lijst of maak nieuwe aan
if os.path.exists(history_path):
    with open(history_path, "r", encoding="utf-8") as f:
        try:
            history_list = json.load(f)
            if not isinstance(history_list, list):
                raise ValueError("forecast_history.json is geen lijst")
        except json.JSONDecodeError:
            history_list = []
else:
    history_list = []

# Voeg toe als nieuwe weekvoorspelling
history_list.append(forecast_records)

# Schrijf opnieuw weg
with open(history_path, "w", encoding="utf-8") as f:
    json.dump(history_list, f, ensure_ascii=False, indent=4)

print(f"✅ Nieuwe weekvoorspelling toegevoegd aan forecast_history.json (totaal {len(history_list)} weken)")

# %% [markdown]
# # %% [markdown]
# ## Stap 13: CO₂ Trend‐analyse (Laatste 7 dagen)  
# In deze stap onderzoeken we de CO₂‐concentraties van de voorbije week, uitgesplitst naar uur, weekend/weekdag, normale dag/feestdag en seizoen.
# 

# %%
# %%
import pandas as pd
import matplotlib.pyplot as plt

# === 13.1 Data voorbereiden ===
# Selecteer de laatste week uurlijkse data
sensor_last_week = sensor_df[sensor_df['ds'] >= sensor_df['ds'].max() - pd.Timedelta(days=7)].copy()
# Feature engineering
sensor_last_week['hour']       = sensor_last_week['ds'].dt.hour
sensor_last_week['is_weekend'] = sensor_last_week['ds'].dt.weekday >= 5
sensor_last_week['is_holiday'] = sensor_last_week['ds'].dt.date.isin(COUNTRY_HOLIDAYS if COUNTRY_HOLIDAYS else [])
sensor_last_week['season']     = (sensor_last_week['ds'].dt.month % 12 // 3 + 1) \
                                     .map({1:'Winter',2:'Spring',3:'Summer',4:'Autumn'})

# --- 13.2 Trend per uur van de dag ---
hourly = sensor_last_week.groupby('hour')['co2'].mean().reset_index()
plt.figure(figsize=(8,4))
plt.plot(hourly['hour'], hourly['co2'], marker='o')
plt.title("10.2 Gemiddelde CO₂ per uur")
plt.xlabel("Uur van de dag")
plt.ylabel("CO₂ (ppm)")
plt.xticks(range(0,24,2))
plt.grid(True)
plt.tight_layout()
plt.savefig("10.2_Gemiddelde_CO₂_per_uur.png")

# --- 13.3 Weekend vs Weekdag trend ---
wd = sensor_last_week.groupby('is_weekend')['co2'].mean().reset_index()
labels = ['Weekdag','Weekend']
plt.figure(figsize=(6,4))
plt.bar(labels, wd['co2'])
plt.title("10.3 Gemiddelde CO₂: Weekdag vs Weekend")
plt.ylabel("CO₂ (ppm)")
plt.tight_layout()
plt.savefig("10.3_Gemiddelde_CO₂_Weekdag_vs_Weekend.png")

# --- 13.4 Feestdag vs Normale dag trend ---
hd = sensor_last_week.groupby('is_holiday')['co2'].mean().reset_index()
labels = ['Normale dag','Feestdag']
plt.figure(figsize=(6,4))
plt.bar(labels, hd['co2'])
plt.title("10.4 Gemiddelde CO₂: Normale dag vs Feestdag")
plt.ylabel("CO₂ (ppm)")
plt.tight_layout()
plt.savefig("10.4_Gemiddelde_CO₂_Normale_dag_vs_Feestdag.png")

# --- 13.5 Seizoensverschil trend ---
seas = sensor_last_week.groupby('season')['co2'] \
        .mean().reindex(['Winter','Spring','Summer','Autumn']).reset_index()
plt.figure(figsize=(6,4))
plt.bar(seas['season'], seas['co2'])
plt.title("10.5 Gemiddelde CO₂ per seizoen")
plt.xlabel("Seizoen")
plt.ylabel("CO₂ (ppm)")
plt.tight_layout()
plt.savefig("10.5_Gemiddelde_CO₂_per_seizoen.png")


# %% [markdown]
# # %% [markdown]
# ## Stap 13.6: Temperatuur Trend‐analyse (Laatste 7 dagen)  
# We onderzoeken de gemiddelde temperatuur van de afgelopen 7 dagen, uitgesplitst naar:
# - Uur van de dag  
# - Weekdag vs. weekend  
# - Normale dag vs. feestdag  
# - Seizoen  
# 

# %%
# %%
import pandas as pd
import matplotlib.pyplot as plt

# === 13.6.1 Data voorbereiden ===

# — 13.6.2 Gemiddelde temperatuur per uur —
hourly_temp = sensor_last_week.groupby('hour')['temperature'].mean().reset_index()
plt.figure(figsize=(8,4))
plt.plot(hourly_temp['hour'], hourly_temp['temperature'], marker='o', color='tab:green')
plt.title("10.6.2 Gemiddelde Temperatuur per uur")
plt.xlabel("Uur van de dag")
plt.ylabel("Temperatuur (°C)")
plt.xticks(range(0,24,2))
plt.grid(True)
plt.tight_layout()
plt.savefig("10.6.2_Gemiddelde_Temperatuur_per_uur.png")

# — 13.6.3 Weekdag vs Weekend —
wd_temp = sensor_last_week.groupby('is_weekend')['temperature'].mean().reset_index()
labels_wd = ['Weekdag','Weekend']
plt.figure(figsize=(6,4))
plt.bar(labels_wd, wd_temp['temperature'], color=['tab:blue','tab:green'])
plt.title("10.6.3 Gemiddelde Temperatuur: Weekdag vs Weekend")
plt.ylabel("Temperatuur (°C)")
plt.tight_layout()
plt.savefig("10.6.3_Gemiddelde_Temperatuur_Weekdag_vs_Weekend.png")

# — 13.6.4 Normale dag vs Feestdag —
hd_temp = sensor_last_week.groupby('is_holiday')['temperature'].mean().reset_index()
labels_hd = ['Normale dag','Feestdag']
plt.figure(figsize=(6,4))
plt.bar(labels_hd, hd_temp['temperature'], color=['tab:gray','tab:red'])
plt.title("10.6.4 Gemiddelde Temperatuur: Normale dag vs Feestdag")
plt.ylabel("Temperatuur (°C)")
plt.tight_layout()
plt.savefig("10.6.4_Gemiddelde_Temperatuur_Normale_dag_vs_Feestdag.png")

# — 13.6.5 Seizoensverschil —
seas_temp = sensor_last_week.groupby('season')['temperature'] \
            .mean().reindex(['Winter','Spring','Summer','Autumn']).reset_index()
plt.figure(figsize=(6,4))
plt.bar(seas_temp['season'], seas_temp['temperature'], color='tab:cyan')
plt.title("10.6.5 Gemiddelde Temperatuur per seizoen")
plt.xlabel("Seizoen")
plt.ylabel("Temperatuur (°C)")
plt.tight_layout()
plt.savefig("10.6.5_Gemiddelde_Temperatuur_per_seizoen.png")


# %% [markdown]
# ## Stap 13.7 (geüpdatet): Visualiseer SARIMAX-coëfficiënten
# 
# Hier gebruiken we exact de exogene variabelen uit Stap 5.2 om hun geschatte parameters te tonen.
# 

# %%
# %% 
import matplotlib.pyplot as plt

# Exogene kolommen exact zoals in Stap 5.2
exog_cols = [
    'presence_ratio',
    'hum_mean',
    'is_weekend',
    'is_holiday',
    'co2_diff1',
    'rapid_rise'
]

# Haal de corresponderende coëfficiënten uit de SARIMAX-resultaten
coeffs = co2_results.params[exog_cols]

# Plot de parameterwaarden
plt.figure(figsize=(10,5))
coeffs.plot(kind='bar')
plt.title("SARIMAX Coëfficiënten voor Exogene Variabelen")
plt.ylabel("Parameterwaarde")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("SARIMAX_Coëfficiënten_voor_Exogene_Variabelen.png")


# %% [markdown]
# # %% [markdown]
# ## Stap 14: CO₂‐patroon kantoordagen vs thuis (22 & 24 april)  
# We vergelijken het gemiddelde uurlijkse CO₂‐niveau op de stage-/kantoor-dagen (22 en 24 april) met dat op thuisdagen in dezelfde periode.
# 

# %%
# %%
import pandas as pd
import matplotlib.pyplot as plt

# — 14.1 Data selectie laatste week —
last_week_cutoff = sensor_df['ds'].max() - pd.Timedelta(days=7)
df_week = sensor_df[sensor_df['ds'] >= last_week_cutoff].copy()

# — 14.2 Definieer kantoor-(stage)dagen en thuisdagen —
office_dates = [pd.to_datetime("2025-04-29").date()]
df_week['day'] = df_week['ds'].dt.date
df_stage = df_week[df_week['day'].isin(office_dates)]
df_home  = df_week[~df_week['day'].isin(office_dates)]

# — 14.3 Gemiddeld CO₂ per uur berekenen —
stage_hourly = df_stage.groupby(df_stage['ds'].dt.hour)['co2'].mean().reset_index(name='co2_stage')
home_hourly  = df_home.groupby(df_home['ds'].dt.hour)['co2'].mean().reset_index(name='co2_home')

# — 14.4 Plot vergelijking —
plt.figure(figsize=(8,4))
plt.plot(stage_hourly['ds'], stage_hourly['co2_stage'], marker='o', linestyle='-', label='Kantoor (29 april)')
plt.plot(home_hourly ['ds'], home_hourly ['co2_home'],  marker='s', linestyle='--', label='Thuis (overige dagen)')
plt.title("Gemiddeld uurlijkse CO₂: kantoor vs thuis")
plt.xlabel("Uur van de dag")
plt.ylabel("CO₂ (ppm)")
plt.xticks(range(0,24,2))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Gemiddeld_uurlijkse_CO₂_kantoor_vs_thuis.png")
# lijst van unieke 'thuis'-dagen
home_days = sorted(df_home['day'].unique())
print("Thuisdagen in deze analyse:", home_days)


# %% [markdown]
# # %% [markdown]
# ## Stap 15: In-sample Actual vs Predicted (Laatste 7 dagen)  
# Hier combineren we de dag­gemiddelde werkelijke waarden met de in-sample voorspellingen van Prophet (temperatuur) en SARIMAX (CO₂) voor dezelfde dagen, zodat je de model­nauwkeurigheid over de afgelopen week kunt zien.
# 

# %%
# %%
import pandas as pd
import matplotlib.pyplot as plt

# — 15.1 Definieer start van de analyseperiode —
start = df_daily['ds'].max() - pd.Timedelta(days=7)

# — 15.2 Werkelijke daggemiddelden inladen —
actual_daily = df_daily[['ds', 'co2_mean', 'temp_mean']].rename(
    columns={'co2_mean':'actual_co2', 'temp_mean':'actual_temp'}
)
actual_week = actual_daily[actual_daily['ds'] >= start]

# — 15.3 In-sample temperatuurvoorspellingen (Prophet) —
# prophet_df bevat ds, y (=temp_mean) en alle regressors
prophet_eval = prophet_df.rename(columns={'y':'actual_temp'})[
    ['ds', 'actual_temp'] + [c for c in prophet_df.columns if c not in ['ds','y']]
]
prophet_pred = temp_model.predict(prophet_eval)[['ds','yhat']].rename(columns={'yhat':'pred_temp'})
prophet_week = prophet_pred[prophet_pred['ds'] >= start]

# — 15.4 In-sample CO₂-voorspellingen (SARIMAX) —
co2_eval = actual_week[['ds','actual_co2']]
co2_pred = co2_results.fittedvalues.to_frame('pred_co2').reset_index().rename(columns={'index':'ds'})
co2_week = co2_pred[co2_pred['ds'] >= start]

# — 15.5 Samenvoegen temperatuur en CO₂ evaluatie —
eval_df = (
    actual_week
    .merge(prophet_week, on='ds', how='left')
    .merge(co2_week, on='ds', how='left')
)

# — 15.6 Plot CO₂ actual vs predicted —
plt.figure(figsize=(10,4))
plt.plot(eval_df['ds'], eval_df['actual_co2'], marker='o', label='Actual CO₂')
plt.plot(eval_df['ds'], eval_df['pred_co2'], marker='x', linestyle='--', label='Predicted CO₂')
plt.title("In-sample CO₂: Actual vs Predicted (Laatste 7 dagen)")
plt.xlabel("Datum")
plt.ylabel("CO₂ (ppm)")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig("In-sample_CO₂_Actual_vs_Predicted__Laatste_7_dagen.png")

# — 15.7 Plot Temperatuur actual vs predicted —
plt.figure(figsize=(10,4))
plt.plot(eval_df['ds'], eval_df['actual_temp'], marker='o', color='tab:green', label='Actual Temp')
plt.plot(eval_df['ds'], eval_df['pred_temp'], marker='x', linestyle='--', color='tab:red', label='Predicted Temp')
plt.title("In-sample Temperatuur: Actual vs Predicted (Laatste 7 dagen)")
plt.xlabel("Datum")
plt.ylabel("Temperatuur (°C)")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig("In-sample_Temperatuur_Actual_vs_Predicted__Laatste_7_dagen.png")


# %% [markdown]
# # %% [markdown]
# ## Stap 16: Uurdata vs Dagvoorspelling (Laatste 7 dagen)  
# In deze stap combineren we de uurlijkse metingen met de dagelijkse voorspellingen in één grafiek, waarbij de x-as één label per dag (middernacht + uur) toont.
# 

# %%
# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

# — 16.1 Data selectie laatste week —
sensor_last_week = sensor_df[sensor_df['ds'] >= sensor_df['ds'].max() - pd.Timedelta(days=7)]

if forecast_df['ds'].dtype == 'O':  
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])

forecast_last_week = forecast_df[forecast_df['ds'] >= forecast_df['ds'].max() - pd.Timedelta(days=7)]

# — 16.2 plotfunctie —
def plot_hourly_vs_daily(ax, actual_x, actual_y, pred_x, pred_y, ylabel, title, advice=None):
    ax.plot(actual_x, actual_y, label=f'Actual ({ylabel}, hourly)', alpha=0.6)
    ax.scatter(pred_x, pred_y, color='tab:orange', s=80, label=f'Predicted ({ylabel}, daily)')
    
    if advice is not None:
        val, style, lbl = advice
        ax.axhline(val, linestyle=style, color='red', label=lbl)
    

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %Hh'))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Datum en uur")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, linestyle=':', linewidth=0.5)

# — 16.3 CO₂ plot —
fig, ax = plt.subplots(figsize=(14, 5))
plot_hourly_vs_daily(
    ax,
    sensor_last_week['ds'], sensor_last_week['co2'],
    forecast_last_week['ds'], forecast_last_week['pred_co2'],
    ylabel="CO₂ (ppm)",
    title="Actual CO₂ (hourly) vs Predicted CO₂ (daily) – Laatste 7 dagen",
    advice=(1000, '--', 'Adviesgrens 1000 ppm')
)
plt.tight_layout()
plt.savefig("Actual_CO₂__hourly__vs_Predicted_CO₂__daily__Laatste_7_dagen.png")

# — 16.4 Temperatuur plot —
fig, ax = plt.subplots(figsize=(14, 5))
plot_hourly_vs_daily(
    ax,
    sensor_last_week['ds'], sensor_last_week['temperature'],
    forecast_last_week['ds'], forecast_last_week['pred_temp'],
    ylabel="Temperatuur (°C)",
    title="Actual Temp (hourly) vs Predicted Temp (daily) – Laatste 7 dagen",
    advice=(18, '--', 'Ondergrens 18°C')
)
plt.tight_layout()
plt.savefig("Actual_Temp__hourly__vs_Predicted_Temp__daily__Laatste_7_dagen.png")


# %% [markdown]
# # %% [markdown]
# ## Stap 17: Dagelijkse gemiddelden vs voorspellingen (Laatste 7 dagen)  
# We vergelijken de daggemiddelde werkelijke CO₂- en temperatuurwaarden met de daggemiddelde voorspellingen over de afgelopen week.
# 

# %%
# %%
import pandas as pd
import matplotlib.pyplot as plt

# — 17.1 Voorbereiden van 'day' kolommen —
sensor_df['day']   = sensor_df['ds'].dt.date
forecast_df['day'] = forecast_df['ds'].dt.date

# — 17.2 Maak daggemiddelden van sensor_df (actuals) —
actual_daily = (
    sensor_df
    .groupby('day')
    .agg(actual_co2=('co2','mean'),
         actual_temp=('temperature','mean'))
    .reset_index()
)

# — 17.3 Maak dagdata van forecast_df (predictions) —
pred_daily = (
    forecast_df
    .groupby('day')
    .agg(pred_co2=('pred_co2','mean'),
         pred_temp=('pred_temp','mean'))
    .reset_index()
)

# — 17.4 Filter: Alleen laatste 7 dagen —
today = pd.Timestamp.today().normalize()
actual_daily_filtered = actual_daily[actual_daily['day'] >= (today - pd.Timedelta(days=7)).date()]
pred_daily_filtered   = pred_daily[pred_daily['day'] >= (today - pd.Timedelta(days=7)).date()]

# — 17.5 Plot CO₂ vs voorspelling —
plt.figure(figsize=(12,5))
plt.plot(pd.to_datetime(actual_daily_filtered['day']), actual_daily_filtered['actual_co2'], 
         marker='o', linestyle='-', color='tab:blue', label='Actual CO₂')
plt.plot(pd.to_datetime(pred_daily_filtered['day']), pred_daily_filtered['pred_co2'], 
         marker='x', linestyle='--', color='tab:orange', label='Predicted CO₂')
plt.title("12.5 Daggemiddelde CO₂: actual vs voorspelling")
plt.xlabel("Datum")
plt.ylabel("CO₂ (ppm)")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig("12.5_Daggemiddelde_CO₂_actual_vs_voorspelling.png")

# — 17.6 Plot Temperatuur vs voorspelling —
plt.figure(figsize=(12,5))
plt.plot(pd.to_datetime(actual_daily_filtered['day']), actual_daily_filtered['actual_temp'], 
         marker='o', linestyle='-', color='tab:green', label='Actual Temp')
plt.plot(pd.to_datetime(pred_daily_filtered['day']), pred_daily_filtered['pred_temp'], 
         marker='x', linestyle='--', color='tab:red', label='Predicted Temp')
plt.title("12.6 Daggemiddelde Temperatuur: actual vs voorspelling")
plt.xlabel("Datum")
plt.ylabel("Temperatuur (°C)")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig("12.6_Daggemiddelde_Temperatuur_actual_vs_voorspelling.png")


# %% [markdown]
## Stap 18: Model performance visualisaties (Laatste 7 dagen)  

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# === 18.1 Voorbereiden van de data ===
# Temperature in-sample
temp_ins = prophet_df.copy()
temp_ins['yhat']     = temp_model.predict(temp_ins)['yhat']
temp_ins['residual'] = temp_ins['y'] - temp_ins['yhat']

# CO₂ in-sample
co2_ins = combined_df.set_index('ds')[['co2_mean']].rename(columns={'co2_mean':'y'})
co2_ins['yhat']      = co2_results.fittedvalues
co2_ins['residual']  = co2_ins['y'] - co2_ins['yhat']

last_date  = max(prophet_df['ds'].max(), co2_ins.index.max())
start_date = last_date - pd.Timedelta(days=7)

# Filter op de laatste 7 dagen
temp_ins = temp_ins[temp_ins['ds'] >= start_date]
co2_ins  = co2_ins[co2_ins.index >= start_date]

if temp_ins.empty or co2_ins.empty:
    logger.warning(
        "Geen in-sample data in de laatste 7 dagen voor residuals; "
        "stap 18.4 (histogrammen) wordt overgeslagen"
    )

# --- 18.2 Scatter plots Actual vs Predicted ---
fig, axes = plt.subplots(1, 2, figsize=(12,5))
# Temperatuur
axes[0].scatter(temp_ins['y'], temp_ins['yhat'], alpha=0.7)
axes[0].plot(
    [temp_ins['y'].min(), temp_ins['y'].max()],
    [temp_ins['y'].min(), temp_ins['y'].max()],
    'k--'
)
axes[0].set_title("18.2 Temp: Actual vs Predicted")
axes[0].set_xlabel("Actual (°C)")
axes[0].set_ylabel("Predicted (°C)")

# CO₂
axes[1].scatter(co2_ins['y'], co2_ins['yhat'], alpha=0.7, color='tab:orange')
axes[1].plot(
    [co2_ins['y'].min(), co2_ins['y'].max()],
    [co2_ins['y'].min(), co2_ins['y'].max()],
    'k--'
)
axes[1].set_title("18.2 CO₂: Actual vs Predicted")
axes[1].set_xlabel("Actual (ppm)")
axes[1].set_ylabel("Predicted (ppm)")

plt.tight_layout()
plt.savefig("18.2_Actual_vs_Predicted.png")

# --- 18.3 Residuals over tijd ---
fig, axes = plt.subplots(2, 1, figsize=(12,6))
axes[0].plot(temp_ins['ds'], temp_ins['residual'], marker='o', linestyle='-')
axes[0].axhline(0, color='black', linestyle='--')
axes[0].set_title("18.3 Temp Residuals over Tijd")
axes[0].set_ylabel("Residual (°C)")

axes[1].plot(co2_ins.index, co2_ins['residual'], marker='o', linestyle='-', color='tab:orange')
axes[1].axhline(0, color='black', linestyle='--')
axes[1].set_title("18.3 CO₂ Residuals over Tijd")
axes[1].set_ylabel("Residual (ppm)")
axes[1].set_xlabel("Datum")

plt.tight_layout()
plt.savefig("18.3_Residuals_over_Tijd.png")

# --- 18.4 Histogrammen van residuals ---
if not temp_ins.empty and not co2_ins.empty:
    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    mae_t = mean_absolute_error(temp_ins['y'], temp_ins['yhat'])
    axes[0].hist(temp_ins['residual'], bins=15, alpha=0.7)
    axes[0].set_title(f"18.4 Temp Residuals (MAE={mae_t:.2f} °C)")
    axes[0].set_xlabel("Residual (°C)")

    mae_c = mean_absolute_error(co2_ins['y'], co2_ins['yhat'])
    axes[1].hist(co2_ins['residual'], bins=15, alpha=0.7, color='tab:orange')
    axes[1].set_title(f"18.4 CO₂ Residuals (MAE={mae_c:.1f} ppm)")
    axes[1].set_xlabel("Residual (ppm)")

    plt.tight_layout()
    plt.savefig("18.4_Residuals_Histogram.png")






