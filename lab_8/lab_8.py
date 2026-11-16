
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from pymatgen.core import Composition
from matminer.datasets import load_dataset
from scipy.stats import randint, loguniform
from matminer.featurizers.composition import ElementProperty

from sklearn.base import clone
from sklearn.linear_model import RidgeCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (
    train_test_split,
    KFold,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    StackingRegressor
)
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


def evaluate_model(name, X_test, y_test, method, model, cv_rmse=np.nan):
    y_pred = model.predict(X_test)

    return {
        "Model": name,
        "Method": method,
        "CV RMSE": cv_rmse,
        "Test R2": r2_score(y_test, y_pred),
        "Test MAE": mean_absolute_error(y_test, y_pred),
        "Test RMSE": np.sqrt(
            mean_squared_error(y_test, y_pred)
        )
    }

def main():
    df = load_dataset("matbench_steels")
    df["composition"] = df["composition"].apply(
        lambda value: value
        if isinstance(value, Composition)
        else Composition(value)
    )

    featurizer = ElementProperty.from_preset("magpie")

    df_features = featurizer.featurize_dataframe(
        df,
        col_id="composition"
    )

    X = df_features[featurizer.feature_labels()]
    y = df_features["yield strength"]

    models = {
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=200,
                max_depth=5,
                random_state=42
            )
        ),

        "AdaBoost": make_pipeline(
            SimpleImputer(strategy="median"),
            AdaBoostRegressor(
                n_estimators=200,
                learning_rate=0.05,
                random_state=42
            )
        ),

        "XGBoost": make_pipeline(
            SimpleImputer(strategy="median"),
            XGBRegressor(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.05,
                random_state=42
            )
        )
    }

    grid_params = {

        "Random Forest": {
            "randomforestregressor__n_estimators": [100, 200],
            "randomforestregressor__max_depth": [5, 10, None],
            "randomforestregressor__min_samples_leaf": [1, 3]
        },

        "AdaBoost": {
            "adaboostregressor__n_estimators": [100, 200],
            "adaboostregressor__learning_rate": [0.01, 0.05, 0.1]
        },

        "XGBoost": {
            "xgbregressor__n_estimators": [150, 300],
            "xgbregressor__max_depth": [2, 3, 5],
            "xgbregressor__learning_rate": [0.03, 0.1]
        },

        "Stacking": {
            "passthrough": [False, True],
            "final_estimator__alphas": [
                [0.1, 1, 10],
                [1, 10, 100]
            ]
        }
    }

    random_params = {

        "Random Forest": {
            "randomforestregressor__n_estimators": randint(100, 401),
            "randomforestregressor__max_depth": [5, 10, 15, None],
            "randomforestregressor__min_samples_leaf": randint(1, 6)
        },

        "AdaBoost": {
            "adaboostregressor__n_estimators": randint(100, 401),
            "adaboostregressor__learning_rate": loguniform(0.01, 0.5)
        },

        "XGBoost": {
            "xgbregressor__n_estimators": randint(100, 501),
            "xgbregressor__max_depth": randint(2, 7),
            "xgbregressor__learning_rate": loguniform(0.01, 0.3),
            "xgbregressor__subsample": [0.7, 0.8, 0.9, 1.0]
        },

        "Stacking": {
            "passthrough": [False, True],
            "final_estimator__alphas": [
                [0.01, 0.1, 1],
                [0.1, 1, 10],
                [1, 10, 100]
            ]
        }
    }

    models["Stacking"] = StackingRegressor(
        estimators=[
            ("rf", models["Random Forest"]),
            ("ada", models["AdaBoost"]),
            ("xgb", models["XGBoost"])
        ],
        final_estimator=RidgeCV(),
        cv=5
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    cv = KFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    statistics = []
    best_models = {}



    for name, model in models.items():

        print(f"\n{name}")

        baseline = clone(model)
        baseline.fit(X_train, y_train)

        statistics.append(evaluate_model(
            name,
            X_test,
            y_test,
            "Baseline",
            baseline
        ))

        grid = GridSearchCV(
            estimator=clone(model),
            param_grid=grid_params[name],
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=2
        )

        grid.fit(X_train, y_train)

        statistics.append(evaluate_model(
            name,
            X_test,
            y_test,
            "GridSearchCV",
            grid.best_estimator_,
            -grid.best_score_
        ))

        print("\nGridSearchCV:")
        print(grid.best_params_)

        random_search = RandomizedSearchCV(
            estimator=clone(model),
            param_distributions=random_params[name],
            n_iter=10 if name != "Stacking" else 6,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=2,
            random_state=42
        )

        random_search.fit(X_train, y_train)

        statistics.append(evaluate_model(
            name,
            X_test,
            y_test,
            "RandomizedSearchCV",
            random_search.best_estimator_,
            -random_search.best_score_
        ))

        print("\nRandomizedSearchCV:")
        print(random_search.best_params_)

        if grid.best_score_ >= random_search.best_score_:
            best_models[name] = grid.best_estimator_
        else:
            best_models[name] = random_search.best_estimator_

    comparison = pd.DataFrame(statistics)

    print("\nRESULTS")

    print(
        comparison
        .sort_values(["Model", "Test RMSE"])
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
