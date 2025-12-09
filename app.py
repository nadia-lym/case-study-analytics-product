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
tab_overview, tab_geo, tab_pricing, tab_revenue, tab_profile = st.tabs(
    ["📊 Marketplace Overview", "🗺️ Geospatial View", "💲 Pricing & Elasticity", "💰 Revenue Insights", "🧪 Data Profiling"]
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

    # ---------------------------------------
    # Demand Heatmap: Trips by Hour x Weekday
    # ---------------------------------------
    st.markdown("### Demand Heatmap: Trips by Hour × Weekday")

    # Prepare data
    heatmap_df = (
        df_filtered.groupby(["weekday", "hour"])
        .size()
        .reset_index(name="trip_count")
    )

    # Convert weekday number → label
    weekday_map = {
        0: "Mon",
        1: "Tue",
        2: "Wed",
        3: "Thu",
        4: "Fri",
        5: "Sat",
        6: "Sun"
    }
    heatmap_df["weekday_name"] = heatmap_df["weekday"].map(weekday_map)
    
    # Pivot to create matrix
    heatmap_pivot = heatmap_df.pivot(index="weekday_name", columns="hour", values="trip_count")
    
    # Plot heatmap using Altair
    import altair as alt
    
    heatmap_chart = (
        alt.Chart(heatmap_df)
        .mark_rect()
        .encode(
            x=alt.X("hour:O", title="Hour of Day"),
            y=alt.Y("weekday_name:O", title="Weekday", sort=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]),
            color=alt.Color("trip_count:Q", title="Trips", scale=alt.Scale(scheme="blues")),
            tooltip=[
                alt.Tooltip("weekday_name:N", title="Weekday"),
                alt.Tooltip("hour:O", title="Hour"),
                alt.Tooltip("trip_count:Q", title="Trips")
            ]
        )
        .properties(height=300)
    )
    
    st.altair_chart(heatmap_chart, use_container_width=True)



# --------------------------
# TAB 2: Geospatial View
# --------------------------
# --------------------------
# TAB 2: Geospatial View
# --------------------------
# --------------------------
# TAB 2: Geospatial View
# --------------------------
# --------------------------
# TAB 2: Geospatial View
# --------------------------
with tab_geo:
    st.header("Pickup Density Map")

    st.caption(
        "**Note:** Location coordinates in the Ride Austin dataset are "
        "obfuscated (lat/long rounded to whole degrees), so this map does *not* "
        "represent true pickup density. It is included only as a visual placeholder."
    )

    # Interactive map (even though coordinates are obfuscated)
    if {"start_location_lat", "start_location_long"}.issubset(df_filtered.columns):
        geo_map = (
            df_filtered[["start_location_lat", "start_location_long"]]
            .dropna()
            .rename(columns={
                "start_location_lat": "lat",
                "start_location_long": "lon"
            })
        )

        # Sample only to keep map fast & tidy
        st.map(geo_map.sample(n=min(4000, len(geo_map)), random_state=42))

    else:
        st.info("Start location coordinates not available in this dataset.")

    st.markdown("---")

    # ----------------------------------------------------
    # 1) Trips per Zip Code
    # ----------------------------------------------------
    st.subheader("Trip Volume by Start Zip Code")

    if "start_zip_code" in df_filtered.columns:
        trips_by_zip = (
            df_filtered.dropna(subset=["start_zip_code"])
            .groupby("start_zip_code")
            .size()
            .sort_values(ascending=False)
            .head(20)
        )

        st.bar_chart(trips_by_zip)
        st.caption("Top 20 zip codes ranked by total trip volume.")
    else:
        st.info("Zip code information not available.")

    st.markdown("---")

    # ----------------------------------------------------
    # 2) Average Surge by Zip Code (true supply tightness)
    # ----------------------------------------------------
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
        st.caption("Zip codes with highest average surge — indicating tight supply or high demand.")
    else:
        st.info("Cannot compute surge by zip code due to missing fields.")



# --------------------------
# TAB 3: Pricing & Elasticity
# --------------------------
with tab_pricing:
    st.subheader("Trips by Surge Factor (Binned)")

    if "surge_factor" in df_filtered.columns:
        # 1) Build surge bins
        surge_series = df_filtered["surge_factor"].clip(lower=0, upper=5)

        bins = [0, 1, 1.25, 1.5, 2, 5]
        surge_bins = pd.cut(surge_series, bins=bins, include_lowest=True)

        # 2) Trip volume per bin (demand)
        trips_by_surge_bin = surge_bins.value_counts().sort_index()
        trips_df = trips_by_surge_bin.reset_index()
        trips_df.columns = ["surge_bin", "trip_count"]
        trips_df["surge_bin"] = trips_df["surge_bin"].astype(str)

        st.bar_chart(trips_df.set_index("surge_bin"))
        st.caption("How ride volume distributes across different surge levels.")

        # 3) Elasticity curve: use bin midpoints as x-axis (numeric)
        #    Compute midpoints of each interval for plotting
        interval_index = trips_by_surge_bin.index
        bin_midpoints = [interval.left + (interval.right - interval.left)/2 for interval in interval_index]

        elasticity_df = pd.DataFrame({
            "surge_mid": bin_midpoints,
            "trip_count": trips_by_surge_bin.values
        })

        st.subheader("Elasticity Curve: Trips vs Surge Level")
        st.line_chart(elasticity_df.set_index("surge_mid"))
        st.caption(
            "This curve approximates demand elasticity: how trip volume changes as surge (effective price) increases."
        )

        # 4) Relative change vs baseline (no/low surge) to talk about lost trips
        st.subheader("Relative Change vs Baseline (No / Low Surge)")
        baseline_trips = elasticity_df.loc[elasticity_df["surge_mid"] <= 1.0, "trip_count"].sum()

        elasticity_df["rel_to_baseline_%"] = (
            (elasticity_df["trip_count"] - baseline_trips) / baseline_trips * 100
        ).round(1)

        st.write(
            "Baseline is defined as the total trip volume in bins with surge ≤ 1.0. "
            "Negative percentages indicate potential lost demand at higher surge levels."
        )
        st.dataframe(elasticity_df)

    else:
        st.info("Surge data not available for this slice of the dataset.")

    st.subheader("Average Driver Rating by Surge Bin")
    if {"surge_factor", "driver_rating"}.issubset(df_filtered.columns):
        surge_series = df_filtered["surge_factor"].clip(lower=0, upper=5)
        bins = [0, 1, 1.25, 1.5, 2, 5]
        surge_bins = pd.cut(surge_series, bins=bins, include_lowest=True)

        rating_by_surge = df_filtered.groupby(surge_bins)["driver_rating"].mean()
        rating_df = rating_by_surge.reset_index()
        rating_df.columns = ["surge_bin", "avg_rating"]
        rating_df["surge_bin"] = rating_df["surge_bin"].astype(str)

        st.bar_chart(rating_df.set_index("surge_bin"))
        st.caption("Helps explore whether higher surge correlates with worse experience.")
    else:
        st.info("Need both surge and rating data to show this chart.")
        
    # --------------------------
    # TAB 4: Revenue Insights
    # --------------------------
    with tab_revenue:
        st.header("Revenue Insights")
        st.write(
            "Estimated revenue modeling based on trip distance, surge factor, "
            "and a simplified Uber-style fare formula."
        )
    
        # -------------------------
        # Revenue Model Definition
        # -------------------------
        st.subheader("Estimated Fare Model")
    
        # Convert distance to miles
        df_filtered["distance_miles"] = df_filtered["distance_travelled"] / 1609.34
    
        BASE_FARE = 2.0
        COST_PER_MILE = 1.5
    
        df_filtered["estimated_fare"] = (
            BASE_FARE + (df_filtered["distance_miles"] * COST_PER_MILE * df_filtered["surge_factor"])
        )
    
        st.write(
            f"""
            **Fare Formula Used:**  
            Estimated Fare = ${BASE_FARE} + (Distance in miles × ${COST_PER_MILE} × Surge Factor)
            """
        )
    
        # ----------------------------------------
        # Gross Bookings vs Surge Bin
        # ----------------------------------------
        st.subheader("Estimated Gross Bookings vs Surge Level")
    
        surge_bins = [0, 1, 1.25, 1.5, 2, 5]
        surge_labels = ["0-1", "1-1.25", "1.25-1.5", "1.5-2", "2-5"]
    
        df_filtered["surge_bin"] = pd.cut(
            df_filtered["surge_factor"],
            bins=surge_bins,
            labels=surge_labels
        )
    
        gross_bookings_df = (
            df_filtered.groupby("surge_bin")
            .agg(
                total_gross_bookings=("estimated_fare", "sum"),
                total_trips=("estimated_fare", "count"),
                avg_fare=("estimated_fare", "mean"),
            )
            .reset_index()
        )
    
        st.dataframe(gross_bookings_df)
    
        import altair as alt
    
        gb_chart = (
            alt.Chart(gross_bookings_df)
            .mark_bar()
            .encode(
                x=alt.X("surge_bin:N", title="Surge Level"),
                y=alt.Y("total_gross_bookings:Q", title="Estimated Gross Bookings ($)"),
                tooltip=["surge_bin", "total_gross_bookings", "total_trips", "avg_fare"],
            )
            .properties(height=300)
        )
    
        st.altair_chart(gb_chart, use_container_width=True)
    
        # ----------------------------------------
        # Estimated Lost Revenue Due to Elasticity
        # ----------------------------------------
        st.subheader("Estimated Lost Revenue Due to Surge-Induced Demand Drop")
    
        elasticity_df = (
            df_filtered.groupby("surge_bin")
            .size()
            .reset_index(name="trip_count")
        )
    
        surge_midpoints = {
            "0-1": 0.5,
            "1-1.25": 1.125,
            "1.25-1.5": 1.375,
            "1.5-2": 1.75,
            "2-5": 3.5,
        }
        elasticity_df["surge_mid"] = elasticity_df["surge_bin"].map(surge_midpoints)
    
        baseline_trip_volume = elasticity_df.loc[
            elasticity_df["surge_bin"] == "0-1", "trip_count"
        ].values[0]
    
        elasticity_df["rel_to_baseline_%"] = (
            (elasticity_df["trip_count"] - baseline_trip_volume) / baseline_trip_volume * 100
        )
    
        elasticity_df["lost_trips"] = (
            (baseline_trip_volume - elasticity_df["trip_count"]).clip(lower=0)
        )
    
        baseline_avg_fare = gross_bookings_df.loc[
            gross_bookings_df["surge_bin"] == "0-1", "avg_fare"
        ].values[0]
    
        elasticity_df["lost_revenue"] = elasticity_df["lost_trips"] * baseline_avg_fare
    
        st.dataframe(
            elasticity_df[["surge_mid", "trip_count", "lost_trips", "lost_revenue"]]
        )
    
        loss_chart = (
            alt.Chart(elasticity_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("surge_mid:Q", title="Surge Level"),
                y=alt.Y("lost_revenue:Q", title="Estimated Lost Revenue ($)"),
            )
            .properties(height=300)
        )
    
        st.altair_chart(loss_chart, use_container_width=True)


# --------------------------
# TAB 5: Data Profiling (appendix)
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
