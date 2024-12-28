import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from google.cloud import storage
import json
from datetime import datetime

st.set_page_config(page_title="Historical Analysis", layout="wide")

class HistoricalAnalysis:
    def __init__(self, bucket_name="mlops-brza"):
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def load_historical_data(self):
        """Load historical prediction results"""
        runs = []
        for blob in self.bucket.list_blobs(prefix='model_outputs/'):
            if blob.name.endswith('results.json'):
                try:
                    data = json.loads(blob.download_as_string())
                    runs.append(data)
                except Exception as e:
                    st.warning(f"Error loading run: {str(e)}")
        return runs

    def plot_performance_trend(self, runs):
        """Plot model performance trend over time"""
        st.header("📈 Model Performance Trend")
        
        # Prepare data for plotting
        performance_data = {
            'timestamp': [datetime.strptime(run['timestamp'], '%Y-%m-%d %H:%M:%S') 
                         for run in runs],
            'r2': [run['metrics']['r2'] for run in runs],
            'rmse': [run['metrics']['rmse'] for run in runs]
        }
        
        # Create R² trend plot
        fig_r2 = px.line(
            performance_data,
            x='timestamp',
            y='r2',
            title="R² Score Over Time"
        )
        st.plotly_chart(fig_r2)
        
        # Create RMSE trend plot
        fig_rmse = px.line(
            performance_data,
            x='timestamp',
            y='rmse',
            title="RMSE Over Time"
        )
        st.plotly_chart(fig_rmse)

    def show_run_comparison(self, runs):
        """Compare different model runs"""
        st.header("🔄 Run Comparison")
        
        # Create comparison table
        comparison_data = []
        for run in runs:
            comparison_data.append({
                'Run ID': run['run_id'][:8],
                'Timestamp': run['timestamp'],
                'R²': f"{run['metrics']['r2']:.4f}",
                'RMSE': f"{run['metrics']['rmse']:.4f}",
                'MSE': f"{run['metrics']['mse']:.4f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison)

def main():
    st.title("📊 Historical Analysis")
    
    analysis = HistoricalAnalysis()
    
    # Load historical data
    with st.spinner("Loading historical data..."):
        runs = analysis.load_historical_data()
