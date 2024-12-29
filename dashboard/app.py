import streamlit as st
import pandas as pd
from google.cloud import storage
from google.oauth2 import service_account
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom CSS to improve layout
st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stRadio [role=radiogroup] {
        gap: 1rem;
        flex-direction: row !important;
    }
    .stRadio label {
        background: #1E1E1E;
        padding: 1rem;
        border-radius: 0.5rem;
        min-width: 150px;
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .sidebar .element-container {
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #1E1E1E;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .st-emotion-cache-16idsys p {
        font-size: 14px;
        margin-bottom: 0px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding-right: 4px;
        padding-left: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
MODEL_TYPES = {
    "XGBoost": "xgboost",
    "LSTM": "lstm",
    "Decision Tree": "decision_tree",
    "LightGBM": "lightgbm"
}

@st.cache_resource
def get_gcs_client():
    """Initialize GCS client with credentials"""
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        return storage.Client(credentials=credentials)
    except Exception as e:
        st.error(f"Error initializing GCS client: {str(e)}")
        return None

def load_model_results(bucket_name: str, run_id: str) -> dict:
    """Load model results from GCS with improved path handling"""
    try:
        client = get_gcs_client()
        if not client:
            return None
            
        bucket = client.bucket(bucket_name)
        
        # Check all possible model paths
        paths = [
            f'models/{MODEL_TYPES[model_type].lower()}/{run_id}/results.json'
            for model_type in MODEL_TYPES
        ]
        paths.extend([
            f'models/lstm_optimized/{run_id}/results.json',
            f'models/lstm_advanced/{run_id}/results.json',
            f'models/lstm_default/{run_id}/results.json',
            f'model_outputs/{run_id}/results.json'  # Legacy path
        ])
        
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

def get_latest_predictions(model_type: str) -> dict:
    """Fetch latest predictions for a specific model"""
    try:
        client = get_gcs_client()
        if not client:
            return None
            
        internal_model_type = MODEL_TYPES.get(model_type, model_type.lower())
        bucket = client.bucket("mlops-brza")
        
        # Check multiple possible paths
        paths = [
            f'live_predictions/{internal_model_type}/latest.json',
            f'live_predictions/{internal_model_type}_optimized/latest.json',
            f'live_predictions/{internal_model_type}_advanced/latest.json'
        ]
        
        for path in paths:
            blob = bucket.blob(path)
            if blob.exists():
                return json.loads(blob.download_as_string())
                
        st.warning(f"No predictions found for {model_type}")
        return None
        
    except Exception as e:
        st.error(f"Error fetching predictions: {str(e)}")
        return None

def load_all_models_results(bucket_name: str) -> list:
    """Load all model results for comparison"""
    try:
        client = get_gcs_client()
        if not client:
            return []
            
        bucket = client.bucket(bucket_name)
        all_results = []
        
        # Search in both models and model_outputs directories
        for prefix in ['models/', 'model_outputs/']:
            for blob in bucket.list_blobs(prefix=prefix):
                if blob.name.endswith('results.json'):
                    try:
                        results = json.loads(blob.download_as_string())
                        if 'metrics' in results:  # Only include valid results
                            # Add model type if not present
                            if 'model_type' not in results:
                                for model_type in MODEL_TYPES.values():
                                    if model_type in blob.name:
                                        results['model_type'] = model_type
                                        break
                            all_results.append(results)
                    except Exception as e:
                        st.warning(f"Skipping malformed result file: {blob.name}")
                        continue
                    
        return all_results
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
        return []

def plot_predictions(results):
    """Plot predictions vs actual values with improved visualization"""
    if not results or 'predictions' not in results or 'actual_values' not in results:
        st.error("Invalid results format for plotting")
        return

    # Convert dates to proper format
    dates = []
    if 'dates' in results:
        dates = pd.to_datetime(results['dates'])
    else:
        # Generate monthly dates if not available
        dates = pd.date_range(
            end=pd.Timestamp.now(),
            periods=len(results['actual_values']),
            freq='M'
        )

    # Create DataFrame for better handling
    df = pd.DataFrame({
        'Date': dates,
        'Actual': results['actual_values'],
        'Predicted': results['predictions']
    })

    fig = go.Figure()

    # Add traces with improved styling
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Actual'],
        name="Actual",
        line=dict(color="#2E86C1", width=2, dash='solid'),
        mode='lines'
    ))

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Predicted'],
        name="Predicted",
        line=dict(color="#E74C3C", width=2, dash='solid'),
        mode='lines'
    ))

    # Improve layout
    fig.update_layout(
        title="Stock Price Predictions vs Actual Values",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.5)"
        ),
        xaxis=dict(
            tickformat="%b %Y",
            tickangle=45,
            gridcolor="rgba(255, 255, 255, 0.1)",
            showgrid=True
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.1)",
            showgrid=True
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
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
        title='Feature Importance',
        template="plotly_dark"
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_metrics(metrics):
    """Display model performance metrics in columns"""
    cols = st.columns(4)
    metrics_list = [
        ("MSE", metrics['mse'], "Mean Squared Error"),
        ("RMSE", metrics['rmse'], "Root Mean Squared Error"),
        ("R²", metrics['r2'], "R-squared Score"),
        ("MAE", metrics['mae'], "Mean Absolute Error")
    ]
    
    for col, (metric, value, description) in zip(cols, metrics_list):
        with col:
            st.markdown(f"<div class='metric-card'>", unsafe_allow_html=True)
            st.metric(
                metric,
                f"{value:.4f}",
                help=description
            )
            st.markdown("</div>", unsafe_allow_html=True)

def display_model_comparison():
    """Display model comparison dashboard"""
    st.header("Model Performance Comparison")
    
    all_results = load_all_models_results("mlops-brza")
    if not all_results:
        st.warning("No model results available for comparison")
        return
        
    # Create comparison dataframe
    comparison_data = []
    for result in all_results:
        if 'metrics' in result and 'model_type' in result:
            comparison_data.append({
                'Model Type': result['model_type'].replace('_', ' ').title(),
                'Run ID': result.get('run_id', 'N/A'),
                'Timestamp': result.get('timestamp', 'N/A'),
                'MSE': result['metrics']['mse'],
                'RMSE': result['metrics']['rmse'],
                'R²': result['metrics']['r2'],
                'MAE': result['metrics']['mae']
            })
    
    df = pd.DataFrame(comparison_data)
    
    # Display key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_mse = df.loc[df['MSE'].idxmin()]
        st.metric("Best MSE", 
                 best_mse['Model Type'],
                 f"{best_mse['MSE']:.4f}")
    
    with col2:
        best_rmse = df.loc[df['RMSE'].idxmin()]
        st.metric("Best RMSE",
                 best_rmse['Model Type'],
                 f"{best_rmse['RMSE']:.4f}")
    
    with col3:
        best_r2 = df.loc[df['R²'].idxmax()]
        st.metric("Best R²",
                 best_r2['Model Type'],
                 f"{best_r2['R²']:.4f}")
    
    with col4:
        best_mae = df.loc[df['MAE'].idxmin()]
        st.metric("Best MAE",
                 best_mae['Model Type'],
                 f"{best_mae['MAE']:.4f}")
    
    # Plot comparisons
    st.subheader("Metric Comparison Across Models")
    
    tab1, tab2 = st.tabs(["📊 Charts", "📑 Details"])
    
    with tab1:
        metrics = ['MSE', 'RMSE', 'R²', 'MAE']
        for metric in metrics:
            fig = px.bar(
                df,
                x='Model Type',
                y=metric,
                title=f'{metric} by Model',
                color='Model Type',
                template="plotly_dark"
            )
            fig.update_layout(
                showlegend=False,
                height=400,
                xaxis_tickangle=45,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Detailed table
        st.dataframe(
            df.style.highlight_min(['MSE', 'RMSE', 'MAE'])
                .highlight_max(['R²'])
                .format({
                    'MSE': '{:.6f}',
                    'RMSE': '{:.6f}',
                    'R²': '{:.6f}',
                    'MAE': '{:.6f}'
                }),
            use_container_width=True
        )

def display_live_predictions(model_type):
    """Display live predictions for a specific model"""
    st.header("🔴 Live Stock Price Predictions")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh Predictions"):
            st.experimental_rerun()
        
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
        if auto_refresh:
            time.sleep(30)
            st.experimental_rerun()
    
    results = get_latest_predictions(model_type)
    
    if results:
        st.success(f"Last Updated: {results.get('timestamp', 'Unknown')}")
        display_prediction_results(results)
    else:
        st.warning(f"No predictions available for {model_type}")
        st.info("Please check if the live prediction service is running")

def display_prediction_results(results):
    """Display prediction results including metrics and plots"""
    if 'metrics' in results:
        st.subheader("Model Performance Metrics")
        display_metrics(results['metrics'])
    
    st.subheader("Predictions vs Actual Values")
    plot_predictions(results)
    
    # Show model parameters in expander
    with st.expander("Model Parameters"):
        st.json(results.get('parameters', {}))
    
    # Add download button for predictions
    st.download_button(
        label="📥 Download Predictions",
        data=json.dumps(results, indent=2),
        file_name=f"{results.get('model_type', 'predictions')}.json",
        mime="application/json"
    )

def display_historical_results(results):
    """Display historical analysis results"""
    st.subheader(f"Model Type: {results.get('model_type', 'Unknown').title()}")
    timestamp = results.get('timestamp', 'Unknown')
    if timestamp != 'Unknown':
        timestamp = pd.to_datetime(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    st.text(f"Training Time: {timestamp}")
    
    if 'metrics' in results:
        st.header("Model Performance Metrics")
        display_metrics(results['metrics'])
    
    st.header("Model Predictions")
    plot_predictions(results)
    
    # Show feature importance for tree-based models
    if results.get('model_type') in ['xgboost', 'lightgbm', 'decision_tree']:
        st.header("Feature Importance")
        plot_feature_importance(results)
    
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Model Parameters", expanded=True):
            st.json(results.get('parameters', {}))
    
    with col2:
        with st.expander("Training Details", expanded=True):
            st.text(f"Run ID: {results.get('run_id', 'Unknown')}")
            st.text(f"Training Time: {timestamp}")
            if 'training_time' in results:
                st.text(f"Training Duration: {results['training_time']:.2f}s")

def main():
    st.title("📈 Stock Price Prediction Dashboard")

    # Create three columns for controls
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        model_type = st.selectbox(
            "Select Model",
            list(MODEL_TYPES.keys()),
            key="model_select"
        )
    
    with col2:
        analysis_type = st.radio(
            "Analysis Type",
            ["Historical Analysis", "Live Predictions", "Model Comparison"],
            horizontal=True
        )
    
    if analysis_type == "Historical Analysis":
        with col3:
            run_id = st.text_input(
                "Enter Run ID",
                help="Enter the model run ID to view historical results"
            )
    
    # Main content area
    st.markdown("---")  # Add a divider
    
    if analysis_type == "Live Predictions":
        display_live_predictions(model_type)
    elif analysis_type == "Model Comparison":
        display_model_comparison()
    else:
        if run_id:
            results = load_model_results("mlops-brza", run_id)
            if results:
                display_historical_results(results)

if __name__ == "__main__":
    main()
