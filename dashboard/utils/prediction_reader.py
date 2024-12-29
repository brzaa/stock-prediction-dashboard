# dashboard/utils/prediction_reader.py
import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account
import json

@st.cache_resource
def get_gcs_client():
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    return storage.Client(credentials=credentials)

def get_latest_predictions(model_type: str, bucket_name: str = "mlops-brza") -> dict:
    """Read latest predictions from storage"""
    try:
        st.write(f"Attempting to fetch predictions for {model_type}...")  # Debug line
        client = get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f'live_predictions/{model_type}/latest.json')
        
        st.write(f"Checking if blob exists at: live_predictions/{model_type}/latest.json")  # Debug line
        if blob.exists():
            results = json.loads(blob.download_as_string())
            st.write("Successfully loaded predictions")  # Debug line
            return results
            
        st.write("No predictions found in storage")  # Debug line
        return None
    except Exception as e:
        st.error(f"Error loading predictions: {str(e)}")
        return None
