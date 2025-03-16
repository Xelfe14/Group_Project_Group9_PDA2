import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    # 1) Load Processed Data
    data_path = "../Data/Processed/merged_features.csv"
    data = pd.read_csv(data_path)
    data['Date'] = pd.to_datetime(data['Date'])

    # 2) Feature Selection & Target
    # Adjust this list if your notebooks introduced new features
    feature_cols = ['Close', 'Volume', 'ema_20', 'day_of_week']
    target_col = 'target_class'

    # 3) Train-Test Split (Time-Based or Random)
    # Example: Time-based split if your notebook does it that way
    cutoff_date = '2023-03-01'
    train_data = data[data['Date'] < cutoff_date]
    test_data  = data[data['Date'] >= cutoff_date]

    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_test  = test_data[feature_cols]
    y_test  = test_data[target_col]

    # Alternatively, if your updated notebook uses a random split:
    # X = data[feature_cols]
    # y = data[target_col]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    # 4) Handle Missing Data (if any)
    # Forward-fill or drop, based on your notebook approach
    y_train.fillna(method='ffill', inplace=True)
    X_train.fillna(method='ffill', inplace=True)
    X_test.fillna(method='ffill', inplace=True)
    y_test.fillna(method='ffill', inplace=True)

    # 5) Random Forest Model Training with Hyperparameter Tuning

    # Initialize base model
    rf_clf = RandomForestClassifier(random_state=42)

    # Define hyperparameter grid for tuning
    param_grid = {
        'n_estimators': [100, 200, 300],      # Number of trees
        'max_depth': [10, 20, 30, None],      # Maximum depth of trees
        'min_samples_split': [2, 5, 10],      # Minimum samples for splitting
        'min_samples_leaf': [1, 2, 4],        # Minimum samples at leaf
        'class_weight': ['balanced', None],    # Class weight consideration
        'max_features': ['sqrt', 'log2']       # Feature selection for splits
    }

    # Perform Grid Search with Cross Validation
    print("Starting Grid Search for Random Forest Classifier...")
    grid_search = GridSearchCV(
        estimator=rf_clf,
        param_grid=param_grid,
        cv=5,                    # 5-fold cross validation
        n_jobs=-1,              # Use all available cores
        scoring='accuracy',      # Optimize for accuracy
        verbose=1
    )

    # Fit the grid search
    grid_search.fit(X_train, y_train)

    # Get best model and parameters
    best_rf = grid_search.best_estimator_
    print("\nBest Parameters:", grid_search.best_params_)

    # Make predictions with best model
    rf_preds = best_rf.predict(X_test)

    # Calculate and display metrics
    accuracy = accuracy_score(y_test, rf_preds)
    print("\nRandom Forest Classification Metrics:")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, rf_preds))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, rf_preds))

    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_rf.feature_importances_
    })
    print("\nFeature Importance:")
    print(feature_importance.sort_values('importance', ascending=False))

    # Save the best model
    with open("../Streamlit_app/best_clf_model.pkl", "wb") as f:
        pickle.dump(best_rf, f)

    print("\nBest Random Forest Classifier saved as 'best_clf_model.pkl'")

if __name__ == '__main__':
    main()
