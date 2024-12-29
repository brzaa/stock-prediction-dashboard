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
    try:
        client = get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f'live_predictions/{model_type}/latest.json')
        
        if blob.exists():
            results = json.loads(blob.download_as_string())
            return results
        return None
    except Exception as e:
        st.error(f"Error loading predictions: {str(e)}")
        return None
