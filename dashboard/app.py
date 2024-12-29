import streamlit as st
import pandas as pd
from google.cloud import storage
from google.oauth2 import service_account
import json
import plotly.graph_objects as go
import plotly.express as px
from utils.prediction_reader import get_latest_predictions
from utils.live_prediction_utils import LivePredictionPipeline

# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

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
    
    # Add view selection
    view_type = st.sidebar.radio(
        "Select View",
        ["Individual Model Analysis", "Live Predictions"]
    )
    
    if view_type == "Live Predictions":
        st.header("🔴 Live Stock Price Predictions")
        
        # Model selection for live predictions
        model_type = st.sidebar.selectbox(
            "Select Model for Live Predictions",
            ["XGBoost", "Decision Tree", "LightGBM"],
            key="live_model_select"
        )
        
        # Add refresh button
        if st.button("🔄 Refresh Predictions"):
            with st.spinner(f"Getting live predictions for {model_type}..."):
                results = get_latest_predictions(model_type.lower(), bucket_name="mlops-brza")
                if results:
                    # Show last update time
                    st.info(f"Last Updated: {results['timestamp']}")
                    
                    # Display metrics
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
                    
                    # Plot predictions
                    plot_predictions(results)
                    
                    # Feature importance for tree-based models
                    if model_type.lower() in ["xgboost", "decision_tree", "lightgbm"]:
                        st.subheader("Feature Importance")
                        plot_feature_importance(results)
                    
                    # Show drift detection results
                    st.subheader("Data Drift Analysis")
                    if results['drift_detected']:
                        st.warning("🚨 Data drift detected!")
                        st.write("Drifted Features:")
                        for feature in results['drifted_features']:
                            st.write(f"- {feature['feature']}: p-value = {feature['p_value']:.4f}")
                    else:
                        st.success("✅ No data drift detected")
                    
                    # Show model parameters
                    with st.expander("Model Parameters"):
                        st.json(results['parameters'])
        else:
            st.info("Click 'Refresh Predictions' to get the latest predictions")
    
    elif view_type == "Individual Model Analysis":
        st.info("This feature is under development for integration.")

if __name__ == "__main__":
    main()
