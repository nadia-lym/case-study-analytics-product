import streamlit as st
import pandas as pd
import zipfile

st.title("Analytical Products Case Study")
st.write("Prototype loading a cleaned sample of the Ride Austin dataset from ZIP.")

# Path to the ZIP file stored in your repo
ZIP_PATH = "archive.zip"

@st.cache_data
def load_and_clean_data(zip_path: str, n_samples: int = 200_000) -> pd.DataFrame:
    # --- Load from ZIP ---
    with zipfile.ZipFile(zip_path, "r") as z:
        csv_name = z.namelist()[0]  # assumes first file is the main CSV
        with z.open(csv_name) as f:
            df_full = pd.read_csv(f)

    # --- Sample for performance ---
    df = df_full.sample(n=min(n_samples, len(df_full)), random_state=42).copy()

    # --- Basic cleaning & type conversions ---

    # Datetime parsing
    for col in ["started_on", "completed_on"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Distance sanity filter
    if "distance_travelled" in df.columns:
        df = df[df["distance_travelled"] > 0]

    # Ratings: fill missing with median (prototype choice)
    for col in ["driver_rating", "rider_rating"]:
        if col in df.columns:
            # avoid error if all values are NaN
            if df[col].notna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

    # Surge factor numeric
    if "surge_factor" in df.columns:
        df["surge_factor"] = pd.to_numeric(df["surge_factor"], errors="coerce")

    # Requested car category to string
    if "requested_car_category" in df.columns:
        df["requested_car_category"] = df["requested_car_category"].astype(str)

    # Year sanity (for vehicle year)
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        # Filter out clearly invalid years (e.g., > 2030 or < 1990)
        df = df[(df["year"].isna()) | ((df["year"] >= 1990) & (df["year"] <= 2030))]

    # --- Derived features for analysis ---

    # Trip duration in minutes
    if {"started_on", "completed_on"}.issubset(df.columns):
        df["trip_duration_min"] = (
            (df["completed_on"] - df["started_on"]).dt.total_seconds() / 60
        )

    # Hour of day & weekday from start time
    if "started_on" in df.columns:
        df["hour"] = df["started_on"].dt.hour
        df["weekday"] = df["started_on"].dt.weekday  # 0=Mon, 6=Sun

    return df

# --- Load data ---
df = load_and_clean_data(ZIP_PATH)

# --- UI: Data preview ---
st.subheader("Raw Data Preview (Cleaned Sample)")
st.dataframe(df.head(100))

# --- UI: Profiling ---

st.subheader("Missing Values by Column")
missing = df.isna().sum().to_frame(name="missing_count")
missing["missing_pct"] = (missing["missing_count"] / len(df)).round(3)
st.dataframe(missing)

st.subheader("Numeric Columns Summary")
num_cols = df.select_dtypes(include=["number"]).columns
if len(num_cols) > 0:
    st.write(df[num_cols].describe().T)
else:
    st.info("No numeric columns detected.")

st.subheader("Categorical Columns Summary")
cat_cols = df.select_dtypes(include=["object"]).columns
if len(cat_cols) > 0:
    # Show top categories & counts for a few key categorical columns
    for col in cat_cols[:10]:  # limit to first 10 to avoid huge output
        st.markdown(f"**{col}**")
        st.write(df[col].value_counts().head(10))
else:
    st.info("No categorical columns detected.")
