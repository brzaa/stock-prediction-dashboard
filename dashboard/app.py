import streamlit as st
import pandas as pd
from google.cloud import storage
from google.oauth2 import service_account
import json
import plotly.graph_objects as go
import plotly.express as px

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
        
        # List of possible paths to check
        paths = [
            f'model_outputs/{run_id}/results.json',  # Default path
            f'model_outputs/lstm/{run_id}/results.json',  # LSTM specific path
            f'model_outputs/decision_tree/{run_id}/results.json',  # Decision Tree path
            f'model_outputs/xgboost/{run_id}/results.json'  # XGBoost path
        ]
        
        # Try each path
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

def plot_predictions(results):
    """Plot predictions vs actual values"""
    fig = go.Figure()
    
    # Use dates if available
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

def main():
    st.title("📈 Stock Price Prediction Dashboard")
    
    st.sidebar.title("Dashboard Controls")
    
    # Add model type selection
    model_type = st.sidebar.selectbox(
        "Select Model Type",
        ["XGBoost", "Decision Tree", "LSTM"]
    )
    
    run_id = st.sidebar.text_input("Enter Run ID")
    
    if run_id:
        results = load_model_results("mlops-brza", run_id)
        
        if results:
            # Display model type and timestamp
            st.subheader(f"Model Type: {results.get('model_type', model_type)}")
            st.text(f"Training Time: {results['timestamp']}")
            
            # Display metrics
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
            
            # Predictions plot
            st.header("Model Predictions")
            plot_predictions(results)
            
            # Feature importance (only for tree-based models)
            if model_type in ["XGBoost", "Decision Tree"]:
                st.header("Feature Importance")
                plot_feature_importance(results)
            
            # Model parameters
            with st.expander("Model Parameters"):
                st.json(results['parameters'])
            
            # Training details
            with st.expander("Training Details"):
                st.text(f"Run ID: {results['run_id']}")
                st.text(f"Model Type: {results.get('model_type', model_type)}")
                st.text(f"Training Time: {results['timestamp']}")
            
            # Download results
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
