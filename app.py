import streamlit as st
import requests
from io import BytesIO
import pandas as pd
import altair as alt

# Ensure FastAPI URL is correct
FASTAPI_URL = "http://127.0.0.1:8000/classify/"

st.set_page_config(page_title="Log Classification Dashboard", layout="wide")

st.title("📊 Log Classification Dashboard")
st.write("Upload a CSV file for batch classification.")

# File Uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}")

    # Read the file as bytes
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}

    # Send file to FastAPI server
    try:
        with st.spinner('Classifying logs...'):
            response = requests.post(FASTAPI_URL, files=files)

        if response.status_code == 200:
            # Use BytesIO to handle the file content in memory
            output = BytesIO(response.content)
            output.seek(0)
            df = pd.read_csv(output)

            st.write("### Classification Results")
            st.dataframe(df)

            st.download_button("Download Processed CSV", output, file_name="classified_logs.csv")

            st.write("### Classification Summary")
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X('target_label', title='Classified Label'),
                y='count()',
                color=alt.Color('target_label', title='Classified Label', scale=alt.Scale(scheme='category20b'))
            ).properties(
                width=600,
                height=400
            )
            st.altair_chart(chart)

        else:
            st.error(f"Server Error: {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to FastAPI server. Is it running?")