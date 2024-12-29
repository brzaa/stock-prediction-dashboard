# File: dashboard/app.py

import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from utils.live_prediction_service import StockPredictor
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

def plot_model_comparison(all_results):
    """Create comparison plots for model metrics"""
    comparison_data = []
    for result in all_results:
        comparison_data.append({
            'Model Type': result.get('model_type', 'Unknown'),
            'Run ID': result['run_id'],
            'Timestamp': result['timestamp'],
            'MSE': result['metrics']['mse'],
            'RMSE': result['metrics']['rmse'],
            'R²': result['metrics']['r2'],
            'MAE': result['metrics']['mae']
        })
    
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

def main():
    st.title("📈 Stock Price Prediction Dashboard")
    st.sidebar.title("Dashboard Controls")
    view_type = st.sidebar.radio(
        "Select View",
        ["Individual Model Analysis", "Model Comparison", "Live Predictions"]
    )
    
    bucket_name = "mlops-brza"
    
    if view_type == "Live Predictions":
        st.header("🔴 Live Stock Price Predictions")
        model_type = st.sidebar.selectbox(
            "Select Model for Live Predictions",
            ["XGBoost", "Decision Tree", "LightGBM"],
            key="live_model_select"
        )
        
        if st.button("🔄 Refresh Predictions"):
            st.write("Using MASB_latest.csv for predictions...")
            try:
                predictor = StockPredictor(bucket_name=bucket_name)
                predictor.run_predictions()  # Executes prediction logic with MASB_latest.csv
                
                # Fetch the latest predictions
                blob_path = f"live_predictions/{model_type.lower()}/latest.json"
                client = predictor.client
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                
                if blob.exists():
                    results = json.loads(blob.download_as_string())
                    st.success("Predictions refreshed successfully!")
                    
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
                    
                    # Plot predictions
                    plot_predictions(results)
                    
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
                    st.error(f"No predictions found for model: {model_type}")
            except Exception as e:
                st.error(f"Error running predictions: {str(e)}")
    
    elif view_type == "Individual Model Analysis":
        st.header("📊 Individual Model Analysis")
        model_type = st.sidebar.selectbox(
            "Select Model Type",
            ["XGBoost", "Decision Tree", "LightGBM", "LSTM"]
        )
        
        run_id = st.sidebar.text_input("Enter Run ID")
        
        if run_id:
            try:
                predictor = StockPredictor(bucket_name=bucket_name)
                blob_path = f"model_outputs/{model_type.lower()}/{run_id}/results.json"
                client = predictor.client
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                
                if blob.exists():
                    results = json.loads(blob.download_as_string())
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
                    
                    # Feature importance (for tree-based models)
                    if model_type.lower() in ["xgboost", "decision_tree", "lightgbm"]:
                        st.header("Feature Importance")
                        feature_importance = pd.DataFrame({
                            "Feature": results['feature_names'],
                            "Importance": results['feature_importance']
                        }).sort_values(by="Importance", ascending=False)
                        st.bar_chart(feature_importance.set_index("Feature"))
                    
                    # Model parameters
                    with st.expander("Model Parameters"):
                        st.json(results['parameters'])
                else:
                    st.error(f"Results not found for Run ID: {run_id}")
            except Exception as e:
                st.error(f"Error fetching model analysis: {str(e)}")
        else:
            st.info("👈 Please enter a Run ID in the sidebar to view results")
    
    elif view_type == "Model Comparison":
        st.header("📈 Model Performance Comparison")
        try:
            predictor = StockPredictor(bucket_name=bucket_name)
            client = predictor.client
            bucket = client.bucket(bucket_name)
            
            # Fetch results for all models
            all_results = []
            for blob in bucket.list_blobs(prefix="model_outputs/"):
                if blob.name.endswith("results.json"):
                    results = json.loads(blob.download_as_string())
                    all_results.append(results)
            
            if all_results:
                plot_model_comparison(all_results)
            else:
                st.warning("No model results found")
        except Exception as e:
            st.error(f"Error fetching model comparison: {str(e)}")

if __name__ == "__main__":
    main()
