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

# Constants
MODEL_TYPES = {
    "XGBoost": "xgboost",
    "LSTM": "lstm",
    "Decision Tree": "decision_tree",
    "LightGBM": "lightgbm"
}

# GCP Project ID
GCP_PROJECT = "mlops-thesis"  # Replace with your actual GCP project ID

@st.cache_resource
def get_gcs_client():
    """Initialize GCS client with credentials"""
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        return storage.Client(credentials=credentials, project=GCP_PROJECT)
    except Exception as e:
        st.error(f"Error initializing GCS client: {str(e)}")
        return None

def load_model_results(bucket_name: str, run_id: str) -> dict:
    """Load model results from GCS"""
    try:
        client = get_gcs_client()
        if not client:
            return None
            
        bucket = client.bucket(bucket_name)
        
        # Check different possible paths
        paths = [
            f'models/{run_id}/results.json',
            f'model_outputs/{run_id}/results.json'
        ]
        
        for path in paths:
            blob = bucket.blob(path)
            if blob.exists():
                return json.loads(blob.download_as_string())
        
        st.error(f"No results found for Run ID: {run_id}")
        return None
        
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
        return None

def get_latest_predictions(model_type: str) -> dict:
    """Fetch latest predictions"""
    try:
        client = get_gcs_client()
        if not client:
            return None
            
        bucket = client.bucket("mlops-brza")
        path = f'live_predictions/{model_type.lower()}/latest.json'
        
        blob = bucket.blob(path)
        if blob.exists():
            return json.loads(blob.download_as_string())
            
        st.warning(f"No predictions found for {model_type}")
        return None
        
    except Exception as e:
        st.error(f"Error fetching predictions: {str(e)}")
        return None

def load_all_models_results(bucket_name: str) -> list:
    """Load all model results"""
    try:
        client = get_gcs_client()
        if not client:
            return []
            
        bucket = client.bucket(bucket_name)
        all_results = []
        
        for blob in bucket.list_blobs(prefix='models/'):
            if blob.name.endswith('results.json'):
                try:
                    results = json.loads(blob.download_as_string())
                    all_results.append(results)
                except:
                    continue
                    
        return all_results
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
        return []

def plot_predictions(results):
    """Plot predictions vs actual values"""
    if not results or 'predictions' not in results or 'actual_values' not in results:
        st.error("Invalid results format")
        return

    dates = pd.date_range(
        end=pd.Timestamp.now(),
        periods=len(results['actual_values']),
        freq='M'
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=results['actual_values'],
        name="Actual",
        line=dict(color="#2E86C1", width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=results['predictions'],
        name="Predicted",
        line=dict(color="#E74C3C", width=2)
    ))

    fig.update_layout(
        title="Stock Price Predictions vs Actual Values",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        height=500,
        xaxis=dict(
            tickformat="%b %Y",
            tickangle=45
        )
    )

    st.plotly_chart(fig, use_container_width=True)

def display_metrics(metrics):
    """Display metrics in columns"""
    cols = st.columns(4)
    with cols[0]:
        st.metric("MSE", f"{metrics['mse']:.4f}")
    with cols[1]:
        st.metric("RMSE", f"{metrics['rmse']:.4f}")
    with cols[2]:
        st.metric("R²", f"{metrics['r2']:.4f}")
    with cols[3]:
        st.metric("MAE", f"{metrics['mae']:.4f}")

def display_model_comparison():
    """Display model comparison"""
    all_results = load_all_models_results("mlops-brza")
    
    if not all_results:
        st.warning("No model results available")
        return
        
    # Create comparison dataframe
    comparison_data = []
    for result in all_results:
        if 'metrics' in result:
            comparison_data.append({
                'Model': result.get('model_type', 'Unknown'),
                'MSE': result['metrics']['mse'],
                'RMSE': result['metrics']['rmse'],
                'R²': result['metrics']['r2'],
                'MAE': result['metrics']['mae']
            })
    
    df = pd.DataFrame(comparison_data)
    
    # Display metrics
    st.header("Model Performance Comparison")
    st.dataframe(df)
    
    # Plot comparisons
    metrics = ['MSE', 'RMSE', 'R²', 'MAE']
    for metric in metrics:
        fig = px.bar(
            df,
            x='Model',
            y=metric,
            title=f'{metric} Comparison',
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

def display_live_predictions(model_type):
    """Display live predictions"""
    st.header("Live Predictions")
    
    results = get_latest_predictions(model_type)
    if results:
        st.success(f"Last Updated: {results.get('timestamp', 'Unknown')}")
        if 'metrics' in results:
            display_metrics(results['metrics'])
        plot_predictions(results)
    else:
        st.warning("No predictions available")

def display_historical_results(results):
    """Display historical analysis"""
    if 'metrics' in results:
        st.header("Model Performance")
        display_metrics(results['metrics'])
    
    st.header("Predictions")
    plot_predictions(results)
    
    with st.expander("Model Details"):
        st.json(results)

def main():
    st.title("📈 Stock Price Prediction Dashboard")
    
    # Model selection
    model_type = st.selectbox(
        "Select Model",
        list(MODEL_TYPES.keys())
    )
    
    # Analysis type selection
    analysis_type = st.radio(
        "Analysis Type",
        ["Historical Analysis", "Live Predictions", "Model Comparison"]
    )
    
    # Optional Run ID input
    run_id = None
    if analysis_type == "Historical Analysis":
        run_id = st.text_input("Enter Run ID")
    
    # Display content based on selection
    if analysis_type == "Live Predictions":
        display_live_predictions(model_type)
    elif analysis_type == "Model Comparison":
        display_model_comparison()
    elif run_id:
        results = load_model_results("mlops-brza", run_id)
        if results:
            display_historical_results(results)

if __name__ == "__main__":
    main()
