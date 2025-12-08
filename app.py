import streamlit as st
import pandas as pd
import zipfile
from io import BytesIO

st.title("Analytical Products Case Study")
st.write("Prototype loading Ride Austin dataset from ZIP with a sample for performance.")

# Path to the ZIP file stored in your repo
zip_path = "archive.zip"

@st.cache_data
def load_data(zip_path, n_samples=200000):
    # Open the ZIP and read the CSV inside
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Get the first CSV file in the ZIP
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df_full = pd.read_csv(f)
    
    # Take a random sample for faster loading
    df_sample = df_full.sample(n=min(n_samples, len(df_full)), random_state=42)
    return df_sample

df = load_data(zip_path)

# Display raw data
st.subheader("Raw Data Preview (Sample)")
st.dataframe(df.head(100))  # show first 100 rows

# Display summary statistics
st.subheader("Summary Statistics")
st.write(df.describe(include='all'))
