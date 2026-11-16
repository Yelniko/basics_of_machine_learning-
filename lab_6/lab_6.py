import kagglehub

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from scipy.stats import loguniform, randint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)

def evaluate_classifier(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    print("\nModel Evaluation")

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nClassification Report")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    print("\nConfusion Matrix")

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        cmap="Blues",
        values_format="d"
    )

    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def iris():
    dataset_path = kagglehub.dataset_download("uciml/iris")
    csv_files = list(Path(dataset_path).rglob("*.csv"))

    df = pd.read_csv(csv_files[0])
    print(df.head())

    X = df.drop(columns=["Id", "Species"])
    y = df["Species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "Linear SVM": make_pipeline(
            StandardScaler(),
            SVC(kernel="linear", C=1.0)
        ),

        "RBF SVM": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma="scale")
        ),

        "Polynomial SVM": make_pipeline(
            StandardScaler(),
            SVC(kernel="poly", degree=3, C=1.0)
        )
    }

    search_spaces = {
        "Linear SVM": {
            "grid": {
                "svc__C": [0.01, 0.1, 1, 10, 100]
            },
            "random": {
                "svc__C": loguniform(0.01, 100)
            }
        },
        "RBF SVM": {
            "grid": {
                "svc__C": [0.1, 1, 10, 100],
                "svc__gamma": [0.001, 0.01, 0.1, 1]
            },
            "random": {
                "svc__C": loguniform(0.01, 100),
                "svc__gamma": loguniform(0.0001, 10)
            }
        },
        "Polynomial SVM": {
            "grid": {
                "svc__C": [0.1, 1, 10],
                "svc__gamma": [0.01, 0.1, 1],
                "svc__degree": [2, 3, 4]
            },
            "random": {
                "svc__C": loguniform(0.01, 100),
                "svc__gamma": loguniform(0.0001, 10),
                "svc__degree": randint(2, 5)
            }
        }
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = []
    best_models = {}

    for name, model in models.items():

        print(f"\n{name}")

        spaces = search_spaces[name]

        searches = {

            "GridSearchCV": GridSearchCV(
                estimator=model,
                param_grid=spaces["grid"],
                scoring="f1_macro",
                cv=cv,
                n_jobs=-1
            ),

    "RandomizedSearchCV": RandomizedSearchCV(
                estimator=model,
                param_distributions=spaces["random"],
                n_iter=20,
                scoring="f1_macro",
                cv=cv,
                n_jobs=-1,
                random_state=42
            )
        }

        for search_name, search in searches.items():
            print(f"\nRunning {search_name}...")

            search.fit(X_train, y_train)

            best_models[(name, search_name)] = (
                search.best_estimator_
            )

            results.append({
                "Model": name,
                "Search": search_name,
                "CV F1": search.best_score_,
                "Parameters": search.best_params_
            })

            print(f"Best F1: {search.best_score_:.4f}")
            print(f"Parameters: {search.best_params_}")

    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(
        by="CV F1",
        ascending=False
    )

    print("\nRESULTS")

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

def main():
   iris()

if __name__ == "__main__":
    main()