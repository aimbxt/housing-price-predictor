from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
MODEL_PATH = Path(__file__).resolve().parent / "housing_price_model.pkl"


def calculate_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """Return the regression metrics used throughout the analysis."""
    mse = mean_squared_error(actual, predicted)
    return {
        "mae": mean_absolute_error(actual, predicted),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "r2": r2_score(actual, predicted),
    }


def plot_exploratory_data(df: pd.DataFrame) -> None:
    """Create the exploratory plots from the notebook."""
    plt.scatter(df["MedInc"], df["Price"])
    plt.xlabel("Median Income")
    plt.ylabel("House Price")
    plt.title("Median Income vs House Price")
    plt.show()

    plt.scatter(df["AveRooms"], df["Price"])
    plt.xlabel("Average Rooms")
    plt.ylabel("House Price")
    plt.title("Average Rooms vs House Price")
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True)
    plt.title("Feature Correlation")
    plt.show()


def main() -> None:
    # Load the California housing dataset and combine features with the target.
    housing = fetch_california_housing(as_frame=True)
    df = housing.data.copy()
    df["Price"] = housing.target

    plot_exploratory_data(df)

    X = df.drop(columns=["Price"])
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    # Train and evaluate the baseline linear regression model.
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    linear_predictions = linear_model.predict(X_test)
    linear_metrics = calculate_metrics(y_test, linear_predictions)

    comparison = pd.DataFrame(
        {
            "Actual": y_test.to_numpy(),
            "Predicted": linear_predictions,
        }
    )
    print("Linear regression sample predictions:")
    print(comparison.head(10))
    print("\nLinear regression metrics:")
    print(f"MAE:  {linear_metrics['mae']}")
    print(f"MSE:  {linear_metrics['mse']}")
    print(f"RMSE: {linear_metrics['rmse']}")
    print(f"R²:   {linear_metrics['r2']}")

    coefficients = pd.DataFrame(
        {
            "Feature": X.columns,
            "Coefficient": linear_model.coef_,
        }
    )
    print("\nLinear regression coefficients:")
    print(coefficients)

    # Scale features before training a second linear regression model.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scaled_model = LinearRegression()
    scaled_model.fit(X_train_scaled, y_train)
    scaled_predictions = scaled_model.predict(X_test_scaled)
    scaled_metrics = calculate_metrics(y_test, scaled_predictions)

    print("\nScaled linear regression metrics:")
    print(f"MAE:  {scaled_metrics['mae']}")
    print(f"RMSE: {scaled_metrics['rmse']}")
    print(f"R²:   {scaled_metrics['r2']}")

    # Train a random forest with the notebook's preset hyperparameters.
    forest_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE,
    )
    forest_model.fit(X_train, y_train)
    forest_predictions = forest_model.predict(X_test)
    forest_metrics = calculate_metrics(y_test, forest_predictions)

    print("\nRandom forest metrics:")
    print(f"MAE:  {forest_metrics['mae']}")
    print(f"RMSE: {forest_metrics['rmse']}")
    print(f"R²:   {forest_metrics['r2']}")

    importance = pd.DataFrame(
        {
            "Feature": X.columns,
            "Importance": forest_model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)
    print("\nRandom forest feature importance:")
    print(importance)

    cv_scores = cross_val_score(
        forest_model,
        X_train,
        y_train,
        cv=5,
        scoring="r2",
    )
    print(f"\nMean cross-validation R²: {cv_scores.mean()}")

    # Tune the forest and evaluate the best model on the held-out test set.
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 20],
    }
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE),
        param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    final_predictions = best_model.predict(X_test)
    final_metrics = calculate_metrics(y_test, final_predictions)

    print("\nFinal model metrics:")
    print(f"MAE:  {final_metrics['mae']}")
    print(f"RMSE: {final_metrics['rmse']}")
    print(f"R²:   {final_metrics['r2']}")

    plt.scatter(y_test, final_predictions)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
    )
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted House Prices")
    plt.show()

    print("\nFinal model performance")
    print("-----------------------")
    print(f"MAE:  {final_metrics['mae']:.3f}")
    print(f"RMSE: {final_metrics['rmse']:.3f}")
    print(f"R²:   {final_metrics['r2']:.3f}")
    print(f"Average error: ${final_metrics['mae'] * 100000:,.0f}")

    joblib.dump(best_model, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")

    # Predict a price for one example with all eight expected features.
    new_house = pd.DataFrame(
        [
            {
                "MedInc": 5.0,
                "HouseAge": 20.0,
                "AveRooms": 5.5,
                "AveBedrms": 1.0,
                "Population": 1000.0,
                "AveOccup": 3.0,
                "Latitude": 34.0,
                "Longitude": -118.0,
            }
        ],
        columns=X.columns,
    )
    prediction = best_model.predict(new_house)
    print(f"Predicted house price: ${prediction[0] * 100000:,.0f}")


if __name__ == "__main__":
    main()
