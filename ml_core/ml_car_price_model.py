import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def load_data():
    try:
        data_path = Path(__file__).resolve().parents[1] / "data" / "cleaned_car_data.csv"

        if(not data_path.exists()):
            print(f"Error: File not found at {data_path}")
            raise FileNotFoundError(f"File not found at {data_path}")
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} records from {data_path.name}")
        return df

    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        raise
    
def split_data(df, target_col="sellingprice", test_size=0.2, random_state=42):
    
    # select numeric features only
    df = df.select_dtypes(include=['int64','float64'])
    # Remove non-finite values that can cause overflow/invalid matmul during prediction.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0, subset=[target_col])
    df = df.fillna(df.median(numeric_only=True))

    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"Train set: {len(X_train)} records")
    print(f"Test set: {len(X_test)} records")
    return X_train, X_test, y_train, y_test


def train_model_with_randomforestregression(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=100, # 100 decision trees
        max_depth=25, # decision trees 25 limit depth
        min_samples_leaf=2,
        n_jobs=-1, # use all CPU cores for training
        random_state=42,
    )
    model.fit(X_train, y_train)
    
    # Feature Importance - Identify which features are most influential in predicting car prices.
    importances = model.feature_importances_
    feature_names = X_train.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    print("\n--- Feature Importance ---")
    print(feature_importance_df)
    
    return model

def train_model_with_lineregression(X_train, y_train):
    # Scale features + L2 regularization to avoid numeric overflow on large datasets.
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(X_train, y_train)
    print("Linear model training completed.")
    return model

def evaluate_model(model, X_test, y_test):
    # Suppress known spurious matmul RuntimeWarnings observed in some numpy/sklearn builds.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*encountered in matmul",
            category=RuntimeWarning,
        )
        predictions = model.predict(X_test)
        train_pred = model.predict(X_train)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print("Train R2:", r2_score(y_train, train_pred))
    print("Test R2:", r2_score(y_test, predictions))
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")


def evaluate_model_with_cross_validation(model, X, y, cv=5):
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kfold, scoring="r2", n_jobs=-1)
    print("Cross validation scores:", scores)
    print("Average CV score:", np.mean(scores))

def compare_models(X_train, y_train, X_test, y_test):
    print("\n--- Evaluating Random Forest Regressor ---")
    rf_model = train_model_with_randomforestregression(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test)
    evaluate_model_with_cross_validation(rf_model, X_train, y_train)

    print("\n--- Evaluating Linear Regression ---")
    lr_model = train_model_with_lineregression(X_train, y_train)
    evaluate_model(lr_model, X_test, y_test)
    evaluate_model_with_cross_validation(lr_model, X_train, y_train)


if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    compare_models(X_train, y_train, X_test, y_test)
