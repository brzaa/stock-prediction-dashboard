import streamlit as st
import pandas as pd
from google.cloud import storage
from google.oauth2 import service_account
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
        .stAlert {
            padding: 1rem;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

class Dashboard:
    def __init__(self):
        """Initialize dashboard with GCP authentication"""
        try:
            # Create API client with service account info from secrets
            credentials = service_account.Credentials.from_service_account_info(
                st.secrets["gcp_service_account"]
            )
            self.client = storage.Client(credentials=credentials)
            self.bucket = self.client.bucket("mlops-brza")  # Your bucket name
        except Exception as e:
            st.error(f"Error initializing GCP client: {str(e)}")
            raise

    def load_model_results(self, run_id):
        """Load model results from GCS"""
        try:
            blob = self.bucket.blob(f'model_outputs/{run_id}/results.json')
            return json.loads(blob.download_as_string())
        except Exception as e:
            st.error(f"Error loading results: {str(e)}")
            return None

    def show_metrics_section(self, metrics):
        """Display model performance metrics"""
        st.header("📊 Model Performance Metrics")
        
        cols = st.columns(4)
        with cols[0]:
            st.metric(
                "MSE",
                f"{metrics['mse']:.2f}",
                delta=None,
                help="Mean Squared Error"
            )
        with cols[1]:
            st.metric(
                "RMSE",
                f"{metrics['rmse']:.2f}",
                delta=None,
                help="Root Mean Squared Error"
            )
        with cols[2]:
            st.metric(
                "R²",
                f"{metrics['r2']:.4f}",
                delta=None,
                help="R-squared (Coefficient of Determination)"
            )
        with cols[3]:
            st.metric(
                "MAE",
                f"{metrics['mae']:.2f}",
                delta=None,
                help="Mean Absolute Error"
            )

    def plot_predictions(self, results):
        """Create and display prediction vs actual plot"""
        st.header("🎯 Predictions vs Actual Values")
        
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(go.Scatter(
            y=results['actual_values'],
            name="Actual",
            line=dict(color="#2E86C1", width=2)
        ))
        
        # Add predicted values
        fig.add_trace(go.Scatter(
            y=results['predictions'],
            name="Predicted",
            line=dict(color="#E74C3C", width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': "Stock Price Predictions",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Time Period",
            yaxis_title="Price",
            template="plotly_white",
            height=500,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def plot_error_distribution(self, results):
        """Plot error distribution"""
        st.header("📉 Error Analysis")
        
        # Calculate errors
        errors = [pred - actual for pred, actual 
                 in zip(results['predictions'], results['actual_values'])]
        
        # Create histogram
        fig = px.histogram(
            errors,
            title="Prediction Error Distribution",
            labels={'value': 'Error', 'count': 'Frequency'},
            color_discrete_sequence=['#3498DB']
        )
        
        fig.update_layout(
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def show_feature_importance(self, results):
        """Display feature importance if available"""
        if 'feature_importance' in results:
            st.header("🔍 Feature Importance")
            
            importance_df = pd.DataFrame({
                'Feature': results['feature_importance']['names'],
                'Importance': results['feature_importance']['scores']
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Feature Importance Analysis"
            )
            
            fig.update_layout(
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("📈 Stock Price Prediction Dashboard")
    
    # Initialize dashboard
    try:
        dashboard = Dashboard()
    except Exception as e:
        st.error("Failed to initialize dashboard. Please check your credentials.")
        return

    # Sidebar
    st.sidebar.title("Dashboard Controls")
    
    # Run ID input
    run_id = st.sidebar.text_input(
        "Enter Run ID",
        help="Enter the run ID from your model training"
    )
    
    if run_id:
        # Load and display results
        results = dashboard.load_model_results(run_id)
        
        if results:
            # Show timestamp of the run
            st.sidebar.info(f"Run timestamp: {results['timestamp']}")
            
            # Display metrics and plots
            dashboard.show_metrics_section(results['metrics'])
            dashboard.plot_predictions(results)
            dashboard.plot_error_distribution(results)
            dashboard.show_feature_importance(results)
            
            # Add download button for results
            st.sidebar.download_button(
                label="📥 Download Results",
                data=json.dumps(results, indent=2),
                file_name=f"results_{run_id}.json",
                mime="application/json"
            )
            
            # Add additional information
            with st.sidebar.expander("ℹ️ Model Information"):
                st.write(f"Run ID: {run_id}")
                st.write(f"Total predictions: {len(results['predictions'])}")
        else:
            st.warning("No results found for the provided Run ID")
    else:
        st.info("👈 Please enter a Run ID in the sidebar to view results")

if __name__ == "__main__":
    main()
