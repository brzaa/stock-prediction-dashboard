import streamlit as st
import pandas as pd
from pathlib import Path
from google.cloud import storage
import json
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configure the page
st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better styling
st.markdown("""
    <style>
        .main {
            padding: 0rem 1rem;
        }
        .stMetric {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

class Dashboard:
    def __init__(self, bucket_name="mlops-brza"):
        """Initialize dashboard with GCS bucket connection"""
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def load_model_results(self, run_id):
        """Load model results from GCS"""
        try:
            blob = self.bucket.blob(f'model_outputs/{run_id}/results.json')
            results = json.loads(blob.download_as_string())
            return results
        except Exception as e:
            st.error(f"Error loading results: {str(e)}")
            return None

    def show_metrics_section(self, metrics):
        """Display model performance metrics"""
        st.header("📊 Model Performance Metrics")
        cols = st.columns(4)
        with cols[0]:
            st.metric("MSE", f"{metrics['mse']:.2f}")
        with cols[1]:
            st.metric("RMSE", f"{metrics['rmse']:.2f}")
        with cols[2]:
            st.metric("R²", f"{metrics['r2']:.2f}")
        with cols[3]:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")

    def plot_predictions(self, results):
        """Create and display prediction vs actual plot"""
        st.header("🎯 Predictions vs Actual Values")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=results['actual_values'],
            name="Actual",
            line=dict(color="#2E86C1", width=2)
        ))
        fig.add_trace(go.Scatter(
            y=results['predictions'],
            name="Predicted",
            line=dict(color="#E74C3C", width=2)
        ))
        
        fig.update_layout(
            title="Stock Price Predictions",
            xaxis_title="Time",
            yaxis_title="Price",
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def show_model_details(self, results):
        """Display model training details and parameters"""
        st.header("🔍 Model Details")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Training Information")
            st.write(f"Run ID: {results['run_id']}")
            st.write(f"Training Date: {results['timestamp']}")
            
        with col2:
            st.subheader("Model Parameters")
            if 'parameters' in results:
                for param, value in results['parameters'].items():
                    st.write(f"{param}: {value}")

def main():
    st.title("📈 Stock Price Prediction Dashboard")
    
    # Initialize dashboard
    dashboard = Dashboard()
    
    # Sidebar for controls
    st.sidebar.title("Dashboard Controls")
    
    # Add run ID input
    run_id = st.sidebar.text_input("Enter Run ID")
    
    if run_id:
        # Load and display results
        results = dashboard.load_model_results(run_id)
        if results:
            dashboard.show_metrics_section(results['metrics'])
            dashboard.plot_predictions(results)
            dashboard.show_model_details(results)
            
            # Add download button for results
            st.sidebar.download_button(
                label="Download Results",
                data=json.dumps(results, indent=2),
                file_name=f"results_{run_id}.json",
                mime="application/json"
            )
    else:
        st.info("👈 Please enter a Run ID in the sidebar to view results")

if __name__ == "__main__":
    main()
