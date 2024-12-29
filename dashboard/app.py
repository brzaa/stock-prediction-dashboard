# dashboard/app.py
import streamlit as st
import pandas as pd
from google.cloud import storage
from google.oauth2 import service_account
import json
import plotly.graph_objects as go
import plotly.express as px
from utils.prediction_reader import get_latest_predictions

# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

@st.cache_resource
def get_gcs_client():
    """Initialize GCS client with credentials"""
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    return storage.Client(credentials=credentials)

def load_model_results(bucket_name: str, run_id: str) -> dict:
    """Load model results from GCS with multiple path checking"""
    try:
        client = get_gcs_client()
        bucket = client.bucket(bucket_name)
        
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
        bucket = client.bucket(bucket_name)
        
        all_results = []
        model_types = ['xgboost', 'lstm', 'decision_tree', 'lightgbm']
        
        for blob in bucket.list_blobs(prefix='model_outputs/'):
            if blob.name.endswith('results.json'):
                try:
                    results = json.loads(blob.download_as_string())
                    if 'model_type' not in results:
                        for model_type in model_types:
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

def plot_predictions(results):
    """Plot predictions vs actual values"""
    fig = go.Figure()
    
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

def plot_model_comparison(all_results):
    """Create comparison plots for model metrics"""
    comparison_data = []
    for result in all_results:
        comparison_data.append({
            'Model Type': result.get('model_type', 'Unknown'),
            'Run ID': result.get('run_id', 'N/A'),
            'Timestamp': result.get('timestamp', 'N/A'),
            'MSE': result['metrics']['mse'],
            'RMSE': result['metrics']['rmse'],
            'R²': result['metrics']['r2'],
            'MAE': result['metrics']['mae']
        })
    
    df = pd.DataFrame(comparison_data)
    
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
            ["XGBoost", "Decision Tree", "LightGBM", "LSTM"],
            key="live_model_select"
        )
        
        auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
        if auto_refresh:
            st.empty()
            time.sleep(30)
            st.rerun()
        
        if st.button("🔄 Refresh Predictions"):
            with st.spinner("Fetching new predictions..."):
                try:
                    st.empty()
                    results = get_latest_predictions(model_type.lower())
                    if results:
                        st.success("Successfully fetched new predictions!")
                    else:
                        st.error("No predictions available")
                        st.info("Please make sure the prediction service is running")
                except Exception as e:
                    st.error(f"Error refreshing predictions: {str(e)}")
            
        with st.spinner("Fetching latest predictions..."):
            results = get_latest_predictions(model_type.lower())
        
        if results:
            st.success(f"Last Updated: {results['timestamp']}")
            st.subheader("Model Performance Metrics")
            metrics = results['metrics']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("MSE", f"{metrics['mse']:.2f}")
            with col2:
                st.metric("RMSE", f"{metrics['rmse']:.2f}")
            with col3:
                st.metric("R²", f"{metrics['r2']:.2f}")
            with col4:
                st.metric("MAE", f"{metrics['mae']:.2f}")
            
            st.subheader("Predictions vs Actual Values")
            plot_predictions(results)
            
            with st.expander("Model Parameters"):
                st.json(results.get('parameters', {}))
                
            st.sidebar.download_button(
                label="Download Latest Predictions",
                data=json.dumps(results, indent=2),
                file_name=f"live_{model_type.lower()}_predictions.json",
                mime="application/json"
            )
        else:
            st.warning("No live predictions available. The prediction service might be offline.")
            st.info("Try clicking the Refresh button or check if the prediction service is running.")
    
    elif view_type == "Model Comparison":
        st.header("Model Performance Comparison")
        results = load_all_models_results("mlops-brza")
        if results:
            plot_model_comparison(results)
        else:
            st.warning("No model results found")
    
    else:
        model_type = st.sidebar.selectbox(
            "Select Model Type",
            ["XGBoost", "Decision Tree", "LSTM", "LightGBM"]
        )
        
        run_id = st.sidebar.text_input("Enter Run ID")
        
        if run_id:
            results = load_model_results("mlops-brza", run_id)
            
            if results:
                st.subheader(f"Model Type: {results.get('model_type', model_type)}")
                st.text(f"Training Time: {results['timestamp']}")
                
                st.header("Model Performance Metrics")
                metrics = results['metrics']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MSE", f"{metrics['mse']:.2f}")
                with col2:
                    st.metric("RMSE", f"{metrics['rmse']:.2f}")
                with col3:
                    st.metric("R²", f"{metrics['r2']:.2f}")
                with col4:
                    st.metric("MAE", f"{metrics['mae']:.2f}")
                
                st.header("Model Predictions")
                plot_predictions(results)
                
                with st.expander("Model Parameters"):
                    st.json(results.get('parameters', {}))
                
                st.sidebar.download_button(
                    label="Download Results",
                    data=json.dumps(results, indent=2),
                    file_name=f"{model_type.lower()}_{run_id}.json",
                    mime="application/json"
                )
        else:
            st.info("👈 Please enter a Run ID in the sidebar to view results")

if __name__ == "__main__":
    main()
