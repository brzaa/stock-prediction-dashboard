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
AVAILABLE_MODELS = ["XGBoost", "LSTM", "Decision Tree", "LightGBM"]
MODEL_TYPE_MAPPING = {
    "XGBoost": "xgboost",
    "LSTM": "lstm",
    "Decision Tree": "decision_tree",
    "LightGBM": "lightgbm"
}

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

def load_model_results(bucket_name: str, run_id: str) -> dict:
    """Load model results from GCS with multiple path checking"""
    try:
        client = get_gcs_client()
        if not client:
            return None
            
        bucket = client.bucket(bucket_name)
        
        # Define all possible paths for model results
        paths = [
            f'model_outputs/{run_id}/results.json',
            f'model_outputs/lightgbm/{run_id}/results.json',
            f'model_outputs/lstm/{run_id}/results.json',
            f'model_outputs/decision_tree/{run_id}/results.json',
            f'model_outputs/xgboost/{run_id}/results.json'
        ]
        
        for path in paths:
            blob = bucket.blob(path)
            if blob.exists():
                st.success(f"Found results at: {path}")
                return json.loads(blob.download_as_string())
        
        st.error(f"No results found for Run ID: {run_id}")
        return None
        
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
        return None

def load_all_models_results(bucket_name: str) -> list:
    """Load results from all models for comparison"""
    try:
        client = get_gcs_client()
        if not client:
            return []
            
        bucket = client.bucket(bucket_name)
        all_results = []
        model_types = ['xgboost', 'lstm', 'decision_tree', 'lightgbm']
        
        # Search in all possible paths
        for blob in bucket.list_blobs(prefix='model_outputs/'):
            if blob.name.endswith('results.json'):
                try:
                    results = json.loads(blob.download_as_string())
                    # Add model type if not present
                    if 'model_type' not in results:
                        for model_type in model_types:
                            if model_type in blob.name:
                                results['model_type'] = model_type
                                break
                    all_results.append(results)
                except Exception as e:
                    st.warning(f"Skipping malformed result file: {blob.name}")
                    continue
                    
        return all_results
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
        return []

def get_latest_predictions(model_type: str) -> dict:
    """Fetch latest predictions for a specific model"""
    try:
        client = get_gcs_client()
        if not client:
            return None
            
        bucket = client.bucket("mlops-brza")
        blob = bucket.blob(f'live_predictions/{model_type}/latest.json')
        
        if not blob.exists():
            st.warning(f"No predictions found for {model_type}")
            return None
            
        return json.loads(blob.download_as_string())
    except Exception as e:
        st.error(f"Error fetching predictions: {str(e)}")
        return None

def plot_predictions(results):
    """Plot predictions vs actual values"""
    if not results or 'predictions' not in results or 'actual_values' not in results:
        st.error("Invalid results format for plotting")
        return
        
    fig = go.Figure()
    
    # Use dates if available, otherwise use index
    x_values = results.get('dates', list(range(len(results['actual_values']))))
    
    fig.add_trace(go.Scatter(
        x=x_values,
        y=results['actual_values'],
        name="Actual",
        line=dict(color="#2E86C1", width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=x_values,
        y=results['predictions'],
        name="Predicted",
        line=dict(color="#E74C3C", width=2)
    ))
    
    fig.update_layout(
        title="Stock Price Predictions vs Actual Values",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_white",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_feature_importance(results):
    """Plot feature importance for tree-based models"""
    if not results.get('feature_importance') or not results.get('feature_names'):
        st.info("Feature importance not available for this model type")
        return
        
    importance_df = pd.DataFrame({
        'Feature': results['feature_names'],
        'Importance': results['feature_importance']
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance'
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def plot_model_comparison(all_results):
    """Create comparison plots for model metrics"""
    if not all_results:
        st.warning("No results available for comparison")
        return
        
    comparison_data = []
    for result in all_results:
        if 'metrics' in result and 'model_type' in result:
            comparison_data.append({
                'Model Type': result.get('model_type', 'Unknown').title(),
                'Run ID': result.get('run_id', 'N/A'),
                'Timestamp': result.get('timestamp', 'N/A'),
                'MSE': result['metrics']['mse'],
                'RMSE': result['metrics']['rmse'],
                'R²': result['metrics']['r2'],
                'MAE': result['metrics']['mae']
            })
    
    if not comparison_data:
        st.warning("No valid comparison data available")
        return
        
    df = pd.DataFrame(comparison_data)
    
    # Create comparison plots
    metrics = ['MSE', 'RMSE', 'R²', 'MAE']
    for metric in metrics:
        fig = px.bar(
            df,
            x='Model Type',
            y=metric,
            title=f'{metric} Comparison Across Models',
            color='Model Type',
            hover_data=['Run ID', 'Timestamp']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Display best models table
    st.header("Best Models by Metric")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Lowest Error Models")
        best_mse = df.loc[df['MSE'].idxmin()]
        best_rmse = df.loc[df['RMSE'].idxmin()]
        best_mae = df.loc[df['MAE'].idxmin()]
        
        st.write("Best MSE:", best_mse['Model Type'], f"({best_mse['MSE']:.4f})")
        st.write("Best RMSE:", best_rmse['Model Type'], f"({best_rmse['RMSE']:.4f})")
        st.write("Best MAE:", best_mae['Model Type'], f"({best_mae['MAE']:.4f})")
    
    with col2:
        st.subheader("Highest R² Model")
        best_r2 = df.loc[df['R²'].idxmax()]
        st.write("Best R²:", best_r2['Model Type'], f"({best_r2['R²']:.4f})")

    # Detailed comparison table
    st.header("Detailed Model Comparison")
    st.dataframe(
        df.style.highlight_min(['MSE', 'RMSE', 'MAE'])
            .highlight_max(['R²'])
    )

def display_metrics(metrics):
    """Display model performance metrics in columns"""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("MSE", f"{metrics['mse']:.2f}")
    with col2:
        st.metric("RMSE", f"{metrics['rmse']:.2f}")
    with col3:
        st.metric("R²", f"{metrics['r2']:.2f}")
    with col4:
        st.metric("MAE", f"{metrics['mae']:.2f}")

def main():
    st.title("📈 Stock Price Prediction Dashboard")
    
    st.sidebar.title("Dashboard Controls")
    
    view_type = st.sidebar.radio(
        "Select View",
        ["Individual Model Analysis", "Model Comparison", "Live Predictions"]
    )
    
    if view_type == "Live Predictions":
        st.header("🔴 Live Stock Price Predictions")
        
        model_type = st.sidebar.selectbox(
            "Select Model",
            AVAILABLE_MODELS,
            key="live_model_select"
        )
        
        # Convert display name to internal name
        internal_model_type = MODEL_TYPE_MAPPING[model_type]
        
        auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
        if auto_refresh:
            st.empty()
            time.sleep(30)
            st.experimental_rerun()
        
        if st.button("🔄 Refresh Predictions"):
            st.experimental_rerun()
        
        results = get_latest_predictions(internal_model_type)
        
        if results:
            st.success(f"Last Updated: {results['timestamp']}")
            
            st.subheader("Model Performance Metrics")
            display_metrics(results['metrics'])
            
            st.subheader("Predictions vs Actual Values")
            plot_predictions(results)
            
            with st.expander("Model Parameters"):
                st.json(results.get('parameters', {}))
            
            # Add download button for predictions
            st.sidebar.download_button(
                label="Download Predictions",
                data=json.dumps(results, indent=2),
                file_name=f"{internal_model_type}_predictions.json",
                mime="application/json"
            )
        else:
            st.warning("No predictions available.")
            st.info("Check if the live prediction service is running.")
    
    elif view_type == "Model Comparison":
        st.header("Model Performance Comparison")
        all_results = load_all_models_results("mlops-brza")
        if all_results:
            plot_model_comparison(all_results)
        else:
            st.warning("No results found for comparison.")
    
    else:  # Individual Model Analysis
        model_type = st.sidebar.selectbox(
            "Select Model Type",
            AVAILABLE_MODELS
        )
        
        run_id = st.sidebar.text_input("Enter Run ID")
        
        if run_id:
            results = load_model_results("mlops-brza", run_id)
            
            if results:
                st.subheader(f"Model Type: {results.get('model_type', 'Unknown').title()}")
                st.text(f"Training Time: {results.get('timestamp', 'Unknown')}")
                
                st.header("Model Performance Metrics")
                display_metrics(results['metrics'])
                
                st.header("Model Predictions")
                plot_predictions(results)
                
                # Show feature importance for tree-based models
                if results.get('model_type') in ['xgboost', 'lightgbm', 'decision_tree']:
                    st.header("Feature Importance")
                    plot_feature_importance(results)
                
                with st.expander("Model Parameters"):
                    st.json(results.get('parameters', {}))
                
                with st.expander("Training Details"):
                    st.text(f"Run ID: {run_id}")
                    st.text(f"Training Time: {results['timestamp']}")
            else:
                st.warning("Run ID not found or invalid.")

if __name__ == "__main__":
    main()
