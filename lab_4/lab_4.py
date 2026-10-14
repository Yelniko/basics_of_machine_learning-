import kagglehub

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN


def basAnalitks(df):
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nType:")
    print(df.dtypes)

    print("\nIs Null:")
    print(df.isnull().sum())

    numeric_cols = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class']

    for col in numeric_cols:
        plt.figure(figsize=(8, 4))

        plt.hist(
            df[col].dropna(),
            bins=30
        )

        mode_value = df[col].mode().iloc[0]

        plt.axvline(
            mode_value,
            linestyle="--",
            label=f"Mode = {mode_value:.2f}"
        )

        plt.title(f"Мода та розподіл: {col}")
        plt.xlabel(col)
        plt.ylabel("Кількість")
        plt.legend()

        plt.show()

    for col in numeric_cols:
        plt.figure(figsize=(8, 4))

        plt.hist(
            df[col].dropna(),
            bins=30
        )

        median_value = df[col].median()

        plt.axvline(
            median_value,
            linestyle="--",
            label=f"Median = {median_value:.2f}"
        )

        plt.title(f"Медіана: {col}")
        plt.xlabel(col)
        plt.ylabel("Кількість")
        plt.legend()

        plt.show()

    for col in numeric_cols:
        plt.figure(figsize=(8, 4))

        plt.hist(
            df[col].dropna(),
            bins=30
        )

        mean_value = df[col].mean()
        median_value = df[col].median()

        plt.axvline(
            mean_value,
            linestyle="--",
            label=f"Mean = {mean_value:.2f}"
        )

        plt.axvline(
            median_value,
            linestyle=":",
            label=f"Median = {median_value:.2f}"
        )

        plt.title(f"Середнє значення та медіана: {col}")
        plt.xlabel(col)
        plt.ylabel("Кількість")
        plt.legend()

        plt.show()

    for col in numeric_cols:

        skew = df[col].skew()

        if abs(skew) < 0.5:
            result = "майже симетричний"

        elif skew > 0:
            result = "права асиметрія"

        else:
            result = "ліва асиметрія"

        print(
            f"{col}: {skew:.3f} -> {result}"
        )

    for col in numeric_cols:
        plt.figure(figsize=(8, 4))

        plt.hist(
            df[col].dropna(),
            bins=40
        )

        plt.title(
            f"{col}, skewness = {df[col].skew():.3f}"
        )

        plt.xlabel(col)
        plt.ylabel("Кількість")

        plt.show()

    for col in numeric_cols:
        mean_value = df[col].mean()
        std_value = df[col].std()

        plt.figure(figsize=(8, 4))

        plt.hist(
            df[col].dropna(),
            bins=30
        )

        plt.axvline(
            mean_value,
            linestyle="--",
            label="Mean"
        )

        plt.axvline(
            mean_value - std_value,
            linestyle=":",
            label="Mean - Std"
        )

        plt.axvline(
            mean_value + std_value,
            linestyle=":",
            label="Mean + Std"
        )

        plt.title(
            f"Стандартне відхилення: {col}"
        )

        plt.xlabel(col)
        plt.ylabel("Кількість")

        plt.legend()

        plt.show()

    statistics = pd.DataFrame({
        "Mean": df[numeric_cols].mean(),
        "Median": df[numeric_cols].median(),
        "Mode": [
            df[col].mode().iloc[0]
            for col in numeric_cols
        ],
        "Variance": df[numeric_cols].var(),
        "Std": df[numeric_cols].std(),
        "Skewness": df[numeric_cols].skew(),
        "Min": df[numeric_cols].min(),
        "Max": df[numeric_cols].max()
    })
    print(statistics)

    df_corr = df.copy()

    corr = df_corr[numeric_cols].corr(
        method="spearman"
    )

    mask = np.zeros_like(
        corr,
        dtype=np.bool_
    )

    mask[np.triu_indices_from(mask)] = True

    sns.set_style("whitegrid")

    plt.figure(figsize=(13, 10))

    sns.heatmap(
        corr,
        annot=True,
        mask=mask,
        cmap="RdBu_r",
        linewidths=0.9,
        linecolor="white",
        fmt=".2f",
        center=0,
        square=True,
        vmin=-1,
        vmax=1
    )

    plt.title(
        "Кореляція Спірмена між ознаками",
        fontsize=18,
        pad=20
    )

    plt.tight_layout()
    plt.show()

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nType:")
    print(df.dtypes)

    print("\nIs Null:")
    print(df.isnull().sum())

    numeric_cols = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14',
                    'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
                    'Amount', 'Class']

    for col in numeric_cols:
        plt.figure(figsize=(8, 4))

        plt.hist(
            df[col].dropna(),
            bins=30
        )

        mode_value = df[col].mode().iloc[0]

        plt.axvline(
            mode_value,
            linestyle="--",
            label=f"Mode = {mode_value:.2f}"
        )

        plt.title(f"Мода та розподіл: {col}")
        plt.xlabel(col)
        plt.ylabel("Кількість")
        plt.legend()

        plt.show()

    for col in numeric_cols:
        plt.figure(figsize=(8, 4))

        plt.hist(
            df[col].dropna(),
            bins=30
        )

        median_value = df[col].median()

        plt.axvline(
            median_value,
            linestyle="--",
            label=f"Median = {median_value:.2f}"
        )

        plt.title(f"Медіана: {col}")
        plt.xlabel(col)
        plt.ylabel("Кількість")
        plt.legend()

        plt.show()

    for col in numeric_cols:
        plt.figure(figsize=(8, 4))

        plt.hist(
            df[col].dropna(),
            bins=30
        )

        mean_value = df[col].mean()
        median_value = df[col].median()

        plt.axvline(
            mean_value,
            linestyle="--",
            label=f"Mean = {mean_value:.2f}"
        )

        plt.axvline(
            median_value,
            linestyle=":",
            label=f"Median = {median_value:.2f}"
        )

        plt.title(f"Середнє значення та медіана: {col}")
        plt.xlabel(col)
        plt.ylabel("Кількість")
        plt.legend()

        plt.show()

    for col in numeric_cols:

        skew = df[col].skew()

        if abs(skew) < 0.5:
            result = "майже симетричний"

        elif skew > 0:
            result = "права асиметрія"

        else:
            result = "ліва асиметрія"

        print(
            f"{col}: {skew:.3f} -> {result}"
        )

    for col in numeric_cols:
        plt.figure(figsize=(8, 4))

        plt.hist(
            df[col].dropna(),
            bins=40
        )

        plt.title(
            f"{col}, skewness = {df[col].skew():.3f}"
        )

        plt.xlabel(col)
        plt.ylabel("Кількість")

        plt.show()

    for col in numeric_cols:
        mean_value = df[col].mean()
        std_value = df[col].std()

        plt.figure(figsize=(8, 4))

        plt.hist(
            df[col].dropna(),
            bins=30
        )

        plt.axvline(
            mean_value,
            linestyle="--",
            label="Mean"
        )

        plt.axvline(
            mean_value - std_value,
            linestyle=":",
            label="Mean - Std"
        )

        plt.axvline(
            mean_value + std_value,
            linestyle=":",
            label="Mean + Std"
        )

        plt.title(
            f"Стандартне відхилення: {col}"
        )

        plt.xlabel(col)
        plt.ylabel("Кількість")

        plt.legend()

        plt.show()

    statistics = pd.DataFrame({
        "Mean": df[numeric_cols].mean(),
        "Median": df[numeric_cols].median(),
        "Mode": [
            df[col].mode().iloc[0]
            for col in numeric_cols
        ],
        "Variance": df[numeric_cols].var(),
        "Std": df[numeric_cols].std(),
        "Skewness": df[numeric_cols].skew(),
        "Min": df[numeric_cols].min(),
        "Max": df[numeric_cols].max()
    })
    print(statistics)

    df_corr = df.copy()

    corr = df_corr[numeric_cols].corr(
        method="spearman"
    )

    mask = np.zeros_like(
        corr,
        dtype=np.bool_
    )

    mask[np.triu_indices_from(mask)] = True

    sns.set_style("whitegrid")

    plt.figure(figsize=(13, 10))

    sns.heatmap(
        corr,
        annot=True,
        mask=mask,
        cmap="RdBu_r",
        linewidths=0.9,
        linecolor="white",
        fmt=".2f",
        center=0,
        square=True,
        vmin=-1,
        vmax=1
    )

    plt.title(
        "Кореляція Спірмена між ознаками",
        fontsize=18,
        pad=20
    )

    plt.tight_layout()
    plt.show()

    X = df.drop(columns=["Class"])
    y = df["Class"]

    mi = mutual_info_classif(
        X,
        y,
        random_state=42
    )

    mi_scores = pd.Series(
        mi,
        index=X.columns
    ).sort_values(ascending=False)

    print(mi_scores.head(15))

    mi_scores.head(15).sort_values().plot(
        kind="barh",
        figsize=(10, 6),
        title="Mutual Information з Class"
    )

    plt.xlabel("Mutual Information")
    plt.tight_layout()
    plt.show()

def logistic_regression(X_train, y_train):
    return LogisticRegression(
        max_iter=1000,
        random_state=42
    ).fit(X_train, y_train)

def decision_tree(X_train, y_train):
    return DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=5,
        random_state=42
    ).fit(X_train, y_train)

def random_forest(X_train, y_train):
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        n_jobs=-1,
        random_state=42
    ).fit(X_train, y_train)

def knn(X_train, y_train):
    return KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        n_jobs=-1
    ).fit(X_train, y_train)

def train_models(X_train, y_train):
    return {
        "Logistic Regression": logistic_regression(X_train, y_train),
        "Decision Tree": decision_tree(X_train, y_train),
        "Random Forest": random_forest(X_train, y_train),
        "KNN": knn(X_train, y_train)
    }

def train(X_train, y_train):
    return train_models(X_train, y_train)

def train_smote(X_train, y_train):
    X_smote, y_smote = SMOTE(
        random_state=42
    ).fit_resample(X_train, y_train)

    return train_models(X_smote, y_smote)

def train_adasyn(X_train, y_train):
    X_adasyn, y_adasyn = ADASYN(
        random_state=42
    ).fit_resample(X_train, y_train)

    return train_models(X_adasyn, y_adasyn)

def main():
    dataset_path = kagglehub.dataset_download("whenamancodes/fraud-detection")
    csv_files = list(Path(dataset_path).rglob("*.csv"))

    df = pd.read_csv(csv_files[0])
    print(df.head())

    basAnalitks(df)
    '''
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "Original": train(X_train_scaled, y_train),
        "SMOTE": train_smote(X_train_scaled, y_train),
        "ADASYN": train_adasyn(X_train_scaled, y_train)
    }

    results = []

    for dataset_name, classifiers in models.items():
        for model_name, model in classifiers.items():
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]

            results.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Precision": precision_score(
                    y_test, y_pred, zero_division=0
                ),
                "Recall": recall_score(
                    y_test, y_pred, zero_division=0
                ),
                "F1": f1_score(
                    y_test, y_pred, zero_division=0
                ),
                "Average Precision": average_precision_score(
                    y_test, y_proba
                )
            })

    results_df = pd.DataFrame(results)

    print(results_df.round(4))
'''



if __name__ == "__main__":
    main()