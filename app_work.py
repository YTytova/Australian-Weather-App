import streamlit as st
import pandas as pd
import joblib


# 1. Page settings

st.set_page_config(
    page_title="Australian Weather",
    page_icon="☀️",
    layout="centered"
)


# 2. Load model

@st.cache_resource
def load_model():
    data = joblib.load("aussie_rain.joblib")  
    return data

data = load_model()
model = data["model"]
imputer = data["imputer"]
scaler = data["scaler"]
encoder = data["encoder"]
numeric_cols = data["numeric_cols"]
categorical_cols = data["categorical_cols"]


# 3. Initialize session_state

if "weather_state" not in st.session_state:
    st.session_state["weather_state"] = "default"


# 4. Title

st.title("🇦🇺 Australian Weather Forecast")
st.markdown("Check what the weather will be tomorrow — sunny or rainy")
st.divider()


# 5. User input

st.markdown("### 📍 Main indicators")
col1, col2 = st.columns(2)
with col1:
    Location = st.selectbox("City", [
        "Adelaide", "Albany", "Albury", "AliceSprings",
        "Brisbane", "Canberra", "Darwin", "Hobart", "Melbourne", "Sydney"
    ])
with col2:
    Rainfall = st.slider("Rainfall today (mm)", 0.0, 100.0, 0.0)

col1, col2 = st.columns(2)
with col1:
    MinTemp = st.slider("Minimum temperature (°C)", -5.0, 35.0, 10.0)
with col2:
    MaxTemp = st.slider("Maximum temperature (°C)", 0.0, 45.0, 25.0)

col1, col2 = st.columns(2)
with col1:
    Sunshine = st.slider("Sunshine hours", 0.0, 14.0, 7.0)
with col2:
    Evaporation = st.slider("Evaporation (mm)", 0.0, 20.0, 5.0)

st.markdown("---")
st.markdown("### 💨 Wind and pressure")
col1, col2, col3 = st.columns(3)
with col1:
    WindGustDir = st.selectbox("Wind gust direction", ["N","NE","E","SE","S","SW","W","NW"])
with col2:
    WindGustSpeed = st.slider("Gust speed (km/h)", 0, 150, 35)
with col3:
    RainToday = st.selectbox("Was there rain today?", ["No", "Yes"])

col1, col2, col3 = st.columns(3)
with col1:
    Pressure9am = st.slider("Pressure at 9am (hPa)", 980.0, 1040.0, 1015.0)
with col2:
    Pressure3pm = st.slider("Pressure at 3pm (hPa)", 980.0, 1040.0, 1012.0)
with col3:
    WindSpeed3pm = st.slider("Average wind speed (km/h)", 0, 80, 15)

st.markdown("---")
st.markdown("### 🌦️ Humidity and cloudiness")
col1, col2, col3 = st.columns(3)
with col1:
    Humidity9am = st.slider("Humidity at 9am (%)", 0, 100, 60)
with col2:
    Humidity3pm = st.slider("Humidity at 3pm (%)", 0, 100, 55)
with col3:
    Cloud3pm = st.slider("Cloudiness (0–9)", 0, 9, 4)


# 6. Build input dataframe

input_data = pd.DataFrame({
    "Location": [Location],
    "MinTemp": [MinTemp],
    "MaxTemp": [MaxTemp],
    "Rainfall": [Rainfall],
    "Evaporation": [Evaporation],
    "Sunshine": [Sunshine],
    "WindGustDir": [WindGustDir],
    "WindGustSpeed": [WindGustSpeed],
    "WindDir9am": ["N"],
    "WindDir3pm": ["N"],
    "WindSpeed9am": [10.0],
    "WindSpeed3pm": [WindSpeed3pm],
    "Humidity9am": [Humidity9am],
    "Humidity3pm": [Humidity3pm],
    "Pressure9am": [Pressure9am],
    "Pressure3pm": [Pressure3pm],
    "Cloud9am": [4],
    "Cloud3pm": [Cloud3pm],
    "Temp9am": [15.0],
    "Temp3pm": [MaxTemp - 3],
    "RainToday": ["Yes" if RainToday == "Yes" else "No"]
})

input_data[numeric_cols] = imputer.transform(input_data[numeric_cols])
scaled = scaler.transform(input_data[numeric_cols])
scaled_df = pd.DataFrame(scaled, columns=numeric_cols)
encoded = encoder.transform(input_data[categorical_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
X_ready = pd.concat([scaled_df, encoded_df], axis=1)


# 7. Prediction

st.markdown("---")
if st.button("Show forecast for tomorrow"):
    prob = model.predict_proba(X_ready)[0][1]
    prediction = int(prob > 0.5)

    # Зберігаємо стан
    st.session_state["weather_state"] = "rainy" if prediction else "sunny"

    # Вивід результату
    st.markdown("## Tomorrow's forecast:")
    if prediction == 1:
        st.warning(f"Chance of rain tomorrow: **{prob:.1%}**\n\nRain is likely tomorrow — better take an umbrella.")
    else:
        st.success(f"Chance of rain tomorrow: **{prob:.1%}**\n\nDry, pleasant weather is expected. A great day to enjoy the outdoors!")

st.caption("Australian Weather App")
