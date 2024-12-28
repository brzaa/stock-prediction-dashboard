import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from google.cloud import storage
import mlflow
import json
from datetime import datetime, timedelta

st.set_page_config(page_title="Live Predictions", layout="wide")

class LivePredictions:
    def __init__(self, bucket_name="mlops-brza"):
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def load_model(self, run_id):
        """Load the model for predictions"""
        model_path = f"runs:/{run_id}/model"
        try:
            return mlflow.xgboost.load_model(model_path)
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None

    def make_predictions(self, model, data):
        """Make predictions on new data"""
        try:
            predictions = model.predict(data)
            return predictions
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")
            return None

    def plot_predictions(self, actual, predicted):
        """Plot actual vs predicted values"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=actual,
            name="Actual",
            line=dict(color="#2E86C1")
        ))
        
        fig.add_trace(go.Scatter(
            y=predicted,
            name="Predicted",
            line=dict(color="#E74C3C")
        ))
        
        fig.update_layout(
            title="Live Predictions",
            xaxis_title="Time",
            yaxis_title="Price",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("🔮 Live Predictions")
    
    predictions = LivePredictions()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload new data (CSV)",
        type=['csv'],
        help="Upload a CSV file with the same format as training data"
    )
    
    # Model selection
    run_id = st.text_input("Enter Model Run ID")
    
    if uploaded_file and run_id:
        try:
            # Load and preprocess data
            data = pd.read_csv(uploaded_file)
            
            # Load model and make predictions
            model = predictions.load_model(run_id)
            if model:
                predicted_values = predictions.make_predictions(model, data)
                if predicted_values is not None:
                    predictions.plot_predictions(data['close'], predicted_values)
                    
                    # Download predictions
                    results_df = pd.DataFrame({
                        'actual': data['close'],
                        'predicted': predicted_values
                    })
                    
                    st.download_button(
                        label="Download Predictions",
                        data=results_df.to_csv(index=False),
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
    else:
        st.info("Please upload data and enter a Run ID to make predictions")

if __name__ == "__main__":
    main()
