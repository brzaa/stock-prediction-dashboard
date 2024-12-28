import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
from google.cloud import storage
from datetime import datetime

def preprocess_data(data, split_date=None, is_training=True):
    """
    Preprocess data for model training or prediction
    """
    # Feature engineering functions
    def add_technical_indicators(df):
        df['ma7'] = df['close'].rolling(window=7).mean()
        df['ma21'] = df['close'].rolling(window=21).mean()
        df['rsi'] = calculate_rsi(df['close'])
        return df
    
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    # Process data
    processed_data = data.copy()
    processed_data = add_technical_indicators(processed_data)
    
    if is_training and split_date:
        train_data = processed_data[processed_data.index < split_date]
        test_data = processed_data[processed_data.index >= split_date]
        
        return prepare_features(train_data), prepare_features(test_data)
    
    return prepare_features(processed_data)
