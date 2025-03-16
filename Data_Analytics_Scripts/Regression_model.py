import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    # 1) Load Processed Data
    data_path = "../Data/Processed/merged_features.csv"
    data = pd.read_csv(data_path)
    data['Date'] = pd.to_datetime(data['Date'])

    # 2) Feature Selection & Target
    feature_cols = ['Close', 'Volume', 'ema_20', 'day_of_week']
    target_col = 'next_day_price'

    # 3) Train-Test Split (Time-Based)
    cutoff_date = '2023-03-01'
    train_data = data[data['Date'] < cutoff_date]
    test_data  = data[data['Date'] >= cutoff_date]

    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_test  = test_data[feature_cols]
    y_test  = test_data[target_col]

    # 4) Handle Missing Data
    y_train.fillna(method='ffill', inplace=True)
    X_train.fillna(method='ffill', inplace=True)
    X_test.fillna(method='ffill', inplace=True)
    y_test.fillna(method='ffill', inplace=True)

    # 5) Random Forest Model Training with Hyperparameter Tuning

    # Initialize the base model
    rf_reg = RandomForestRegressor(random_state=42)

    # Define hyperparameter grid for tuning
    param_grid = {
        'n_estimators': [100, 200, 300],      # Number of trees in the forest
        'max_depth': [10, 20, 30, None],      # Maximum depth of each tree
        'min_samples_split': [2, 5, 10],      # Minimum samples required to split a node
        'min_samples_leaf': [1, 2, 4],        # Minimum samples required at each leaf node
        'max_features': ['sqrt', 'log2']      # Number of features to consider for best split
    }

    # Perform Grid Search with Cross Validation
    print("Starting Grid Search for Random Forest...")
    grid_search = GridSearchCV(
        estimator=rf_reg,
        param_grid=param_grid,
        cv=5,                    # 5-fold cross validation
        n_jobs=-1,              # Use all available cores
        scoring='neg_mean_squared_error',
        verbose=1
    )

    # Fit the grid search
    grid_search.fit(X_train, y_train)

    # Get best model and parameters
    best_rf = grid_search.best_estimator_
    print("\nBest Parameters:", grid_search.best_params_)

    # Make predictions with best model
    rf_preds = best_rf.predict(X_test)

    # Calculate metrics
    rf_mse = mean_squared_error(y_test, rf_preds)
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_r2  = r2_score(y_test, rf_preds)

    print("\nRandom Forest Regressor Metrics:")
    print(f"MSE: {rf_mse:.2f}")
    print(f"MAE: {rf_mae:.2f}")
    print(f"R^2: {rf_r2:.2f}")

    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_rf.feature_importances_
    })
    print("\nFeature Importance:")
    print(feature_importance.sort_values('importance', ascending=False))

    # Save the best model
    with open("../Streamlit_app/best_reg_model.pkl", "wb") as f:
        pickle.dump(best_rf, f)

    print("\nBest Random Forest model saved as 'best_reg_model.pkl'")

if __name__ == '__main__':
    main()
