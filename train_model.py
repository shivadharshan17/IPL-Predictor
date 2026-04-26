import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("ipl_ml_dataset.csv")

X = df.drop("final_score", axis=1)
y = df["final_score"]

categorical_cols = ["batting_team", "bowling_team"]

numerical_cols = [
    "batting_strength",
    "boundary_strength",
    "bowling_economy",
    "bowling_wicket_strength",
    "wickets_lost",
    "run_rate"
]

# -----------------------------
# Preprocessing
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numerical_cols)
    ]
)

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Model Selection
# -----------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Neural Network": MLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=42
    )
}

best_model_name = None
best_cv_mae = float("inf")
best_pipeline = None

print("\nMODEL COMPARISON")
print("-" * 60)

for name, regressor in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", regressor)
    ])

    cv_mae = -cross_val_score(
        pipe,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_absolute_error"
    ).mean()

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(name)
    print(f"CV MAE   : {cv_mae:.2f}")
    print(f"Test MAE : {mae:.2f}")
    print(f"RMSE     : {rmse:.2f}")
    print(f"R2 Score : {r2:.2f}")
    print("-" * 60)

    # Automatically select model with lowest CV MAE
    if cv_mae < best_cv_mae:
        best_cv_mae = cv_mae
        best_model_name = name
        best_pipeline = pipe

print(f"\nAutomatically Selected Best Model: {best_model_name}")
print(f"Best CV MAE: {best_cv_mae:.2f}")

# -----------------------------
# Hyperparameter tuning
# -----------------------------
param_dist = None

if best_model_name == "Random Forest":
    param_dist = {
        "model__n_estimators": [100, 150, 200, 300],
        "model__max_depth": [6, 8, 10, 12, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4]
    }

elif best_model_name == "Gradient Boosting":
    param_dist = {
        "model__n_estimators": [100, 150, 200, 300],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__max_depth": [2, 3, 4, 5],
        "model__min_samples_split": [2, 5, 10]
    }

elif best_model_name == "Neural Network":
    param_dist = {
        "model__hidden_layer_sizes": [(64, 32), (128, 64), (100,)],
        "model__alpha": [0.0001, 0.001, 0.01],
        "model__learning_rate_init": [0.001, 0.005, 0.01]
    }

elif best_model_name == "Linear Regression":
    param_dist = None

if param_dist is not None:
    print(f"\nTuning {best_model_name} using RandomizedSearchCV...")

    search = RandomizedSearchCV(
        best_pipeline,
        param_distributions=param_dist,
        n_iter=15,
        cv=3,
        scoring="neg_mean_absolute_error",
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    final_model = search.best_estimator_

    print("\nBest Parameters:")
    print(search.best_params_)

else:
    print("\nNo hyperparameter tuning required for Linear Regression.")
    final_model = best_pipeline

# -----------------------------
# Final Evaluation
# -----------------------------
final_pred = final_model.predict(X_test)

final_mae = mean_absolute_error(y_test, final_pred)
final_rmse = mean_squared_error(y_test, final_pred) ** 0.5
final_r2 = r2_score(y_test, final_pred)

print("\nFINAL SELECTED MODEL")
print("-" * 60)
print("Model:", best_model_name)
print(f"Final MAE : {final_mae:.2f}")
print(f"Final RMSE: {final_rmse:.2f}")
print(f"Final R2  : {final_r2:.2f}")
print("-" * 60)

# -----------------------------
# Save final model
# -----------------------------
joblib.dump(final_model, "model_pipeline.pkl")

print("\nSaved final best model as model_pipeline.pkl")