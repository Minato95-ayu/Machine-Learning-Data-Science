import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

def train_xgboost_model(X, y):
    """
    Trains an XGBoost regression model for House Price Prediction.
    Includes hyperparameter definition and model evaluation.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize XGBoost Regressor
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    # Fit the model
    print("Training XGBoost Model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50,
        verbose=100
    )

    # Predictions
    preds = model.predict(X_test)
    
    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print(f"\\nModel Performance:")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R2 Score: {r2:.4f}")
    
    return model

if __name__ == '__main__':
    # Simulated data for demonstration
    # In a real scenario, this would load preprocessed data
    print("XGBoost Training Pipeline initialized.")
