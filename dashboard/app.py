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
st.set_page_config(page_title="Stock Price Prediction Dashboard", page_icon="📈", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main > div {padding-top: 1rem;}
    .block-container {padding-top: 1rem;}
    .element-container {margin-bottom: 1rem;}
    </style>
""", unsafe_allow_html=True)

# Constants
MODEL_TYPES = {
    "XGBoost": "xgboost",
    "LSTM": "lstm",
    "Decision Tree": "decision_tree",
    "LightGBM": "lightgbm"
}

# Cache GCS client
@st.cache_resource
def get_gcs_client():
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        return storage.Client(credentials=credentials)
    except Exception as e:
        st.error(f"Error initializing GCS client: {str(e)}")
        return None

def load_model_results(bucket_name: str, run_id: str) -> dict:
    try:
        client = get_gcs_client()
        if not client:
            return None
        
        bucket = client.bucket(bucket_name)
        paths = [
            f'model_outputs/{run_id}/results.json',
            f'models/lstm_optimized/{run_id}/results.json',
            f'models/lstm_advanced/{run_id}/results.json',
            f'models/lstm_default/{run_id}/results.json'
        ]
        
        # Add paths for each model type
        for model_type in MODEL_TYPES.values():
            paths.append(f'models/{model_type}/{run_id}/results.json')
        
        for path in paths:
            blob = bucket.blob(path)
            if blob.exists():
                return json.loads(blob.download_as_string())
        
        st.error(f"No results found for Run ID: {run_id}")
        return None
        
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
        return None

def load_all_models_results(bucket_name: str) -> list:
    try:
        client = get_gcs_client()
        if not client:
            return []
            
        bucket = client.bucket(bucket_name)
        all_results = []
        
        # Search in models directory
        for blob in bucket.list_blobs(prefix='models/'):
            if blob.name.endswith('results.json'):
                try:
                    results = json.loads(blob.download_as_string())
                    if 'metrics' in results:
                        if 'model_type' not in results:
                            for model_type in MODEL_TYPES.values():
                                if model_type in blob.name:
                                    results['model_type'] = model_type
                                    break
                        all_results.append(results)
                except:
                    continue
                    
        return all_results
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
        return []

def get_latest_predictions(model_type: str) -> dict:
    try:
        client = get_gcs_client()
        if not client:
            return None
            
        internal_model_type = MODEL_TYPES.get(model_type, model_type.lower())
        bucket = client.bucket("mlops-brza")
        
        paths = [
            f'live_predictions/{internal_model_type}/latest.json',
            f'live_predictions/{internal_model_type}_optimized/latest.json'
        ]
        
        for path in paths:
            blob = bucket.blob(path)
            if blob.exists():
                return json.loads(blob.download_as_string())
                
        st.warning(f"No predictions found for {model_type}")
        return None
        
    except Exception as e:
        st.error(f"Error fetching predictions: {str(e)}")
        return None

def plot_predictions(results):
    if not results or 'predictions' not in results or 'actual_values' not in results:
        st.error("Invalid results format for plotting")
        return

    dates = []
    if 'dates' in results:
        dates = pd.to_datetime(results['dates'])
    else:
        dates = pd.date_range(
            end=pd.Timestamp.now(),
            periods=len(results['actual_values']),
            freq='M'
        )

    df = pd.DataFrame({
        'Date': dates,
        'Actual': results['actual_values'],
        'Predicted': results['predictions']
    })

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Actual'],
        name="Actual",
        line=dict(color="#2E86C1", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Predicted'],
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
            tickangle=45,
            gridcolor="rgba(255,255,255,0.1)"
        ),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig, use_container_width=True)

def display_metrics(metrics):
    cols = st.columns(4)
    for i, (col, (metric, value)) in enumerate(zip(cols, metrics.items())):
        with col:
            st.metric(
                label=metric.upper(),
                value=f"{value:.4f}"
            )

def display_model_comparison():
    results = load_all_models_results("mlops-brza")
    if not results:
        st.warning("No model results available")
        return

    st.subheader("Model Performance Comparison")
    
    # Create comparison dataframe
    comparison_data = []
    for result in results:
        if 'metrics' in result and 'model_type' in result:
            comparison_data.append({
                'Model Type': result['model_type'].replace('_', ' ').title(),
                'Run ID': result.get('run_id', 'N/A'),
                'MSE': result['metrics']['mse'],
                'RMSE': result['metrics']['rmse'],
                'R²': result['metrics']['r2'],
                'MAE': result['metrics']['mae']
            })
    
    df = pd.DataFrame(comparison_data)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        best_mse = df.loc[df['MSE'].idxmin()]
        st.metric("Best MSE", best_mse['Model Type'], f"{best_mse['MSE']:.4f}")
    with col2:
        best_rmse = df.loc[df['RMSE'].idxmin()]
        st.metric("Best RMSE", best_rmse['Model Type'], f"{best_rmse['RMSE']:.4f}")
    with col3:
        best_r2 = df.loc[df['R²'].idxmax()]
        st.metric("Best R²", best_r2['Model Type'], f"{best_r2['R²']:.4f}")
    with col4:
        best_mae = df.loc[df['MAE'].idxmin()]
        st.metric("Best MAE", best_mae['Model Type'], f"{best_mae['MAE']:.4f}")

    # Display table
    st.dataframe(
        df.style.highlight_min(['MSE', 'RMSE', 'MAE'])
            .highlight_max(['R²'])
            .format({
                'MSE': '{:.6f}',
                'RMSE': '{:.6f}',
                'R²': '{:.6f}',
                'MAE': '{:.6f}'
            })
    )

def display_historical_results(results):
    if 'metrics' in results:
        st.subheader("Model Metrics")
        display_metrics(results['metrics'])
    
    st.subheader("Price Predictions")
    plot_predictions(results)

def main():
    st.title("📈 Stock Price Prediction Dashboard")

    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        # Select Model dropdown
        model_type = st.selectbox("Select Model", list(MODEL_TYPES.keys()))

        # Analysis Type radio buttons (in the same order as you requested)
        pages = ["Model Comparison", "Historical Analysis", "Live Predictions"]
        current_page = st.radio("Select Analysis Type", pages)
        
        # Run ID input (only show for Historical Analysis)
        if current_page == "Historical Analysis":
            run_id = st.text_input("Enter Run ID")
    
    # Main content area
    if current_page == "Model Comparison":
        display_model_comparison()
    elif current_page == "Historical Analysis":
        if 'run_id' in locals():
            results = load_model_results("mlops-brza", run_id)
            if results:
                display_historical_results(results)
    elif current_page == "Live Predictions":
        results = get_latest_predictions(model_type)
        if results:
            st.success(f"Last Updated: {results.get('timestamp', 'Unknown')}")
            display_historical_results(results)

if __name__ == "__main__":
    main()
