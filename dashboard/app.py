
import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from utils.prediction_reader import get_latest_predictions
from utils.live_prediction_service import fetch_and_preprocess_data, train_and_log_model, ModelMonitor
from google.cloud import storage
import logging

# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

def main():
    st.title("📈 Stock Price Prediction Dashboard")
    st.sidebar.title("Dashboard Controls")
    view_type = st.sidebar.radio(
        "Select View",
        ["Individual Model Analysis", "Model Comparison", "Live Predictions"]
    )
    
    bucket_name = "mlops-brza"
    project_id = "mlops-thesis"
    
    if view_type == "Live Predictions":
        st.header("🔴 Live Stock Price Predictions")
        model_type = st.sidebar.selectbox(
            "Select Model for Live Predictions",
            ["XGBoost", "Decision Tree", "LightGBM"],
            key="live_model_select"
        )
        
        if st.button("🔄 Refresh Predictions"):
            st.write("Fetching and processing new data...")
            try:
                # Fetch and preprocess new data
                X_train, y_train, X_test, y_test = fetch_and_preprocess_data(project_id, bucket_name)
                
                st.write("Training model...")
                run_id, model, metrics = train_and_log_model(X_train, y_train, X_test, y_test)
                
                st.success("Model trained and predictions updated!")
                st.write(f"Run ID: {run_id}")
                
                # Display metrics
                st.header("Model Performance Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MSE", f"{metrics['mse']:.2f}")
                with col2:
                    st.metric("RMSE", f"{metrics['rmse']:.2f}")
                with col3:
                    st.metric("R²", f"{metrics['r2']:.2f}")
                with col4:
                    st.metric("MAE", f"{metrics['mae']:.2f}")
                
                # Plot predictions
                predictions = pd.DataFrame({
                    "Actual": y_test.values,
                    "Predicted": model.predict(X_test)
                }, index=X_test.index)
                plot_predictions({
                    "actual_values": predictions["Actual"].tolist(),
                    "predictions": predictions["Predicted"].tolist(),
                    "dates": [d.strftime("%Y-%m-%d") for d in X_test.index]
                })
                
                st.success("Live predictions refreshed successfully!")
            except Exception as e:
                st.error(f"Error refreshing predictions: {str(e)}")
    
    elif view_type == "Individual Model Analysis":
        st.header("📊 Individual Model Analysis")
        # Original functionality remains here

    elif view_type == "Model Comparison":
        st.header("📈 Model Comparison")
        # Original functionality remains here

if __name__ == "__main__":
    main()
