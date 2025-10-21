import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import preprocess_data

def test_preprocess_data():
    """Test básico de preprocesamiento"""
    df = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'quality': np.random.randint(3, 9, 100)
    })
    
    X_train, X_test, y_train, y_test, scaler, features = preprocess_data(df)
    
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]
    print("test_preprocess_data passed")

def test_data_split():
    """Verifica que la división sea correcta"""
    df = pd.DataFrame({
        'feature1': np.random.rand(100),
        'quality': np.random.randint(3, 9, 100)
    })
    
    X_train, X_test, _, _, _, _ = preprocess_data(df, test_size=0.2)
    
    total = X_train.shape[0] + X_test.shape[0]
    assert abs(total - 100) <= 1  # Tolerancia por redondeo
    print("test_data_split passed")