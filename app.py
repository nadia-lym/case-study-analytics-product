import streamlit as st
import pandas as pd
import zipfile

st.set_page_config(layout="wide")
st.title("Analytical Products Case Study – Ride Austin Prototype")
st.write("Interactive prototype exploring a cleaned sample of the Ride Austin dataset.")

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
        if col in df.columns and df[col].notna().any():
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
        df["date_only"] = df["started_on"].dt.date
    else:
        df["hour"] = pd.NA
        df["weekday"] = pd.NA
        df["date_only"] = pd.NaT

    return df

# --- Load data ---
df = load_and_clean_data(ZIP_PATH)

# ==========================
# SIDEBAR FILTERS (global)
# ==========================
st.sidebar.header("Filters")

# Date range filter
if df["date_only"].notna().any():
    min_date = df["date_only"].dropna().min()
    max_date = df["date_only"].dropna().max()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date)
    )
else:
    date_range = (None, None)

# Hour of day filter
hour_range = st.sidebar.slider("Select Hour of Day", 0, 23, (0, 23))

# Surge range filter
if "surge_factor" in df.columns and df["surge_factor"].notna().any():
    min_surge = float(df["surge_factor"].min(skipna=True))
    max_surge = float(df["surge_factor"].max(skipna=True))
    surge_range = st.sidebar.slider(
        "Surge Factor Range",
        min_surge,
        max_surge,
        (min_surge, max_surge),
    )
else:
    surge_range = (0.0, 10.0)

# Ride category filter
if "requested_car_category" in df.columns:
    ride_category = st.sidebar.multiselect(
        "Ride Category",
        options=sorted(df["requested_car_category"].unique()),
        default=sorted(df["requested_car_category"].unique()),
    )
else:
    ride_category = []

# ==========================
# FILTERED DATAFRAME
# ==========================
df_filtered = df.copy()

if date_range[0] is not None and date_range[1] is not None:
    df_filtered = df_filtered[
        (df_filtered["date_only"] >= date_range[0]) &
        (df_filtered["date_only"] <= date_range[1])
    ]

df_filtered = df_filtered[
    df_filtered["hour"].between(hour_range[0], hour_range[1])
]

if "surge_factor" in df_filtered.columns:
    df_filtered = df_filtered[
        df_filtered["surge_factor"].between(surge_range[0], surge_range[1])
    ]

if "requested_car_category" in df_filtered.columns and ride_category:
    df_filtered = df_filtered[df_filtered["requested_car_category"].isin(ride_category)]

st.write(
    f"Filtered down to **{len(df_filtered):,} trips** from **{len(df):,}** sampled trips."
)

# ==========================
# TABS
# ==========================
tab_overview, tab_geo, tab_pricing, tab_profile = st.tabs(
    ["📊 Marketplace Overview", "🗺️ Geospatial View", "💲 Pricing & Elasticity", "🧪 Data Profiling"]
)

# --------------------------
# TAB 1: Marketplace Overview
# --------------------------
with tab_overview:
    st.subheader("Key Marketplace Metrics")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Trips", f"{len(df_filtered):,}")

    if "surge_factor" in df_filtered.columns:
        col2.metric("Avg Surge", round(df_filtered["surge_factor"].mean(), 2))
        pct_surge = df_filtered["surge_factor"].gt(1).mean() * 100
        col5.metric("% Trips in Surge", f"{pct_surge:.1f}%")
    else:
        col2.metric("Avg Surge", "N/A")
        col5.metric("% Trips in Surge", "N/A")

    if "trip_duration_min" in df_filtered.columns:
        col3.metric("Avg Duration (min)", round(df_filtered["trip_duration_min"].mean(), 2))
    else:
        col3.metric("Avg Duration (min)", "N/A")

    if "driver_rating" in df_filtered.columns:
        col4.metric("Avg Driver Rating", round(df_filtered["driver_rating"].mean(), 2))
    else:
        col4.metric("Avg Driver Rating", "N/A")

    # Trips by Hour (already have)
    st.subheader("Trips by Hour of Day (Typical Daily Pattern)")
    if "hour" in df_filtered.columns:
        hourly_counts = df_filtered.groupby("hour").size().sort_index()
        st.line_chart(hourly_counts)
    else:
        st.info("Hour information is not available in the dataset.")

    # Trips over Time (by date)
    st.subheader("Trips Over Time (by Date)")
    if "date_only" in df_filtered.columns:
        trips_by_date = df_filtered.groupby("date_only").size()
        st.line_chart(trips_by_date)
    else:
        st.info("Date information is not available.")

    # Average surge over time
    st.subheader("Average Surge Over Time")
    if {"date_only", "surge_factor"}.issubset(df_filtered.columns):
        surge_by_date = df_filtered.groupby("date_only")["surge_factor"].mean()
        st.line_chart(surge_by_date)
    else:
        st.info("Surge information is not available.")

    # Trip duration distribution
    st.subheader("Trip Duration Distribution (minutes)")
    if "trip_duration_min" in df_filtered.columns:
        duration_series = df_filtered["trip_duration_min"].clip(lower=0, upper=120)  # cap at 2 hours
        st.bar_chart(duration_series.value_counts(bins=20).sort_index())
    else:
        st.info("Trip duration not available.")

    # Driver rating distribution
    st.subheader("Driver Rating Distribution")
    if "driver_rating" in df_filtered.columns:
        st.bar_chart(df_filtered["driver_rating"].value_counts().sort_index())
    else:
        st.info("Driver rating not available.")

    st.subheader("Raw Data Preview (Filtered Sample)")
    st.dataframe(df_filtered.head(100))

# --------------------------
# TAB 2: Geospatial View
# --------------------------
with tab_geo:
    st.subheader("Pickup Hotspots (Sampled Map)")
    if {"start_location_lat", "start_location_long"}.issubset(df_filtered.columns):
        geo = df_filtered[["start_location_lat", "start_location_long"]].dropna().rename(
            columns={"start_location_lat": "lat", "start_location_long": "lon"}
        )
        # sample to keep map performant
        st.map(geo.sample(n=min(5000, len(geo)), random_state=42))
    else:
        st.info("Start location coordinates not available.")

    st.subheader("Average Surge by Start Zip Code")
    if {"start_zip_code", "surge_factor"}.issubset(df_filtered.columns):
        surge_by_zip = (
            df_filtered.dropna(subset=["start_zip_code"])
            .groupby("start_zip_code")["surge_factor"]
            .mean()
            .sort_values(ascending=False)
            .head(20)
        )
        st.bar_chart(surge_by_zip)
    else:
        st.info("Zip code / surge data not available.")

# --------------------------
# TAB 3: Pricing & Elasticity
# --------------------------
with tab_pricing:
    st.subheader("Trips by Surge Factor (Binned)")
    if "surge_factor" in df_filtered.columns:
        # Create bins
        surge_series = df_filtered["surge_factor"].clip(lower=0, upper=5)
        surge_bins = pd.cut(surge_series, bins=[0, 1, 1.25, 1.5, 2, 5], include_lowest=True)

        # Convert bin intervals to strings for safe plotting
        trips_by_surge_bin = surge_bins.value_counts().sort_index()
        trips_df = trips_by_surge_bin.reset_index()
        trips_df.columns = ["surge_bin", "trip_count"]
        trips_df["surge_bin"] = trips_df["surge_bin"].astype(str)

        st.bar_chart(trips_df.set_index("surge_bin"))
        st.caption("This approximates how ride volume changes across different surge levels.")
    else:
        st.info("Surge data not available.")

    st.subheader("Average Driver Rating by Surge Bin")
    if {"surge_factor", "driver_rating"}.issubset(df_filtered.columns):
        surge_bins = pd.cut(
            df_filtered["surge_factor"].clip(lower=0, upper=5),
            bins=[0, 1, 1.25, 1.5, 2, 5],
            include_lowest=True,
        )
        rating_by_surge = df_filtered.groupby(surge_bins)["driver_rating"].mean()
        rating_df = rating_by_surge.reset_index()
        rating_df.columns = ["surge_bin", "avg_rating"]
        rating_df["surge_bin"] = rating_df["surge_bin"].astype(str)

        st.bar_chart(rating_df.set_index("surge_bin"))
        st.caption("Helps explore whether higher surge correlates with worse experience.")
    else:
        st.info("Need both surge and rating data to show this chart.")

# --------------------------
# TAB 4: Data Profiling (your appendix)
# --------------------------
with tab_profile:
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
        for col in cat_cols[:10]:  # limit output
            st.markdown(f"**{col}**")
            st.write(df[col].value_counts().head(10))
    else:
        st.info("No categorical columns detected.")
