import streamlit as st
import pandas as pd
from google.cloud import storage
from google.oauth2 import service_account
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .stRadio [role=radiogroup] {
        gap: 1rem;
    }
    .stRadio label {
        background: #1E1E1E;
        padding: 0.8rem;
        border-radius: 0.5rem;
        min-width: 120px;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
MODEL_TYPES = {
    "XGBoost": "xgboost",
    "LSTM": "lstm",
    "Decision Tree": "decision_tree",
    "LightGBM": "lightgbm"
}

# Initialize session state if not exists
if 'page' not in st.session_state:
    st.session_state.page = 'app'

# Sidebar navigation
with st.sidebar:
    st.title("app")
    if st.button("Historical Analysis"):
        st.session_state.page = 'historical'
    if st.button("Live Predictions"):
        st.session_state.page = 'live'

@st.cache_resource
def get_gcs_client():
    """Initialize GCS client with credentials"""
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        return storage.Client(credentials=credentials)
    except Exception as e:
        st.error(f"Error initializing GCS client: {str(e)}")
        return None

# Your other functions remain the same (load_model_results, get_latest_predictions, etc.)
# ... (keep all the functions from previous code)

def main():
    st.title("📈 Stock Price Prediction Dashboard")
    
    # Add page selection at the top
    model_type = st.selectbox(
        "Select Model",
        list(MODEL_TYPES.keys())
    )

    # Center the analysis type options
    analysis_type = st.radio(
        "Analysis Type",
        ["Historical Analysis", "Live Predictions", "Model Comparison"]
    )

    # Only show Run ID input for Historical Analysis
    run_id = None
    if analysis_type == "Historical Analysis":
        run_id = st.text_input(
            "Enter Run ID",
            help="Enter the model run ID to view historical results"
        )

    # Display content based on selection
    if analysis_type == "Live Predictions":
        display_live_predictions(model_type)
    elif analysis_type == "Model Comparison":
        display_model_comparison()
    elif analysis_type == "Historical Analysis" and run_id:
        results = load_model_results("mlops-brza", run_id)
        if results:
            display_historical_results(results)

# Keep all your display functions (display_metrics, plot_predictions, etc.) the same

if __name__ == "__main__":
    main()
