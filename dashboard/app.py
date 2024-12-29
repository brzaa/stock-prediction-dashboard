
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
            'Timestamp': result['timestamp'],
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
            hover_data=['Timestamp']
        )
        st.plotly_chart(fig, use_container_width=True)

    st.header("Detailed Model Comparison")
    st.dataframe(df.style.highlight_min(['MSE', 'RMSE', 'MAE']).highlight_max(['R²']))

def main():
    st.title("📈 Stock Price Prediction Dashboard")
    
    st.sidebar.title("Dashboard Controls")
    
    view_type = st.sidebar.radio(
        "Select View",
        ["Live Predictions", "Model Comparison"]
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
            
            if 'drift_detected' in results:
                st.subheader("Data Drift Analysis")
                if results['drift_detected']:
                    st.warning("🚨 Data drift detected!")
                    if 'drifted_features' in results:
                        st.write("Drifted Features:")
                        for feature in results['drifted_features']:
                            feature_idx = feature.get('feature_idx', 'Unknown')
                            p_value = feature.get('p_value', 0)
                            st.write(f"- Feature {feature_idx}: p-value = {p_value:.4f}")
                else:
                    st.success("✅ No data drift detected")
            
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
        # Load results for comparison
        client = get_gcs_client()
        bucket = client.bucket("mlops-brza")
        all_results = []
        for blob in bucket.list_blobs(prefix='live_predictions/'):
            if blob.name.endswith('latest.json'):
                results = json.loads(blob.download_as_string())
                all_results.append(results)
        if all_results:
            plot_model_comparison(all_results)
        else:
            st.warning("No model results found for comparison.")

if __name__ == "__main__":
    main()
