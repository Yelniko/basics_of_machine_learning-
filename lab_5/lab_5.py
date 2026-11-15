import kagglehub

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.feature_selection import (
    mutual_info_regression,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

def descriptive_analysis(df, target=None, exclude=None, max_plots=10,
    bins=30, task="auto", mi_sample=10000 ):
    if target is not None and target not in df.columns:
        raise ValueError(f"Target '{target}' not found")

    if task not in ("auto", "classification", "regression"):
        raise ValueError("Invalid task")

    if not df.columns.is_unique:
        raise ValueError("Column names must be unique")

    exclude = list(exclude or [])

    print(f"\nDataset shape: {df.shape}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    overview = pd.DataFrame({
        "Type": df.dtypes.astype(str),
        "Missing": df.isna().sum(),
        "Missing %": df.isna().mean() * 100,
        "Unique": df.nunique()
    })

    print("\nDataset overview:")
    print(overview.round(2))

    ignored = exclude.copy()

    if target is not None:
        ignored.append(target)

    features = df.drop(columns=ignored, errors="ignore")

    numeric = features.select_dtypes(
        include="number"
    ).replace([np.inf, -np.inf], np.nan)

    categorical = features.select_dtypes(
        include=["object", "category", "string", "bool"]
    )

    print("\nNumeric columns:")
    print(numeric.columns.tolist())

    print("\nCategorical columns:")
    print(categorical.columns.tolist())

    statistics = pd.DataFrame()

    if len(numeric.columns) > 0:

        statistics = pd.DataFrame({
            "Mean": numeric.mean(),
            "Median": numeric.median(),
            "Mode": numeric.apply(
                lambda s: (
                    s.mode().iloc[0]
                    if not s.mode().empty
                    else np.nan
                )
            ),
            "Variance": numeric.var(),
            "Std": numeric.std(),
            "Skewness": numeric.skew(),
            "Min": numeric.min(),
            "Max": numeric.max()
        })

        print("\nDescriptive statistics:")
        print(statistics.round(3))

    columns = (
        numeric.columns
        if max_plots is None
        else numeric.columns[:max_plots]
    )

    for col in columns:

        values = numeric[col].dropna()

        if values.empty:
            continue

        mean = values.mean()
        median = values.median()
        std = values.std()
        skew = values.skew()

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.hist(
            values,
            bins=bins,
            color="#4C78A8",
            alpha=0.75,
            edgecolor="white"
        )

        ax.axvline(
            mean,
            color="#E45756",
            linestyle="--",
            label=f"Mean: {mean:.2f}"
        )

        ax.axvline(
            median,
            color="#54A24B",
            linestyle=":",
            label=f"Median: {median:.2f}"
        )

        if pd.notna(std):
            ax.axvspan(
                mean - std,
                mean + std,
                color="#F2CF5B",
                alpha=0.2,
                label=f"Std: {std:.2f}"
            )

        ax.set_title(
            f"{col} | Skewness: {skew:.3f}"
        )

        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

        ax.legend()
        ax.grid(alpha=0.2)

        plt.tight_layout()
        plt.show()


    for col in categorical.columns:

        print(f"\nCategory distribution: {col}")

        print(
            categorical[col]
            .value_counts(dropna=False)
            .head(10)
        )

    valid_numeric = numeric.loc[
        :,
        numeric.nunique() > 1
    ]

    correlation = valid_numeric.corr(
        method="spearman"
    )

    if correlation.shape[1] >= 2:

        size = min(
            18,
            max(8, len(correlation.columns) * 0.6)
        )

        plt.figure(figsize=(size, size))

        mask = np.triu(
            np.ones_like(correlation, dtype=bool)
        )

        sns.heatmap(
            correlation,
            mask=mask,
            annot=len(correlation.columns) <= 18,
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=0.5
        )

        plt.title("Spearman Correlation")

        plt.tight_layout()
        plt.show()

    mi_scores = pd.Series(dtype=float)

    if target is not None:

        data = df.loc[df[target].notna()]

        if mi_sample is not None and len(data) > mi_sample:
            data = data.sample(
                n=mi_sample,
                random_state=42
            )

        X = data.drop(
            columns=ignored,
            errors="ignore"
        ).select_dtypes(
            include=["number", "object", "category", "string", "bool"]
        ).copy()

        y = data[target]

        discrete = []

        for col in X.columns:

            is_categorical = (
                not pd.api.types.is_numeric_dtype(X[col])
                or pd.api.types.is_bool_dtype(X[col])
            )

            if is_categorical:

                X[col] = pd.factorize(
                    X[col].astype("string").fillna("__MISSING__")
                )[0]

            else:

                values = pd.to_numeric(
                    X[col],
                    errors="coerce"
                ).replace([np.inf, -np.inf], np.nan)

                X[col] = values.fillna(
                    values.median()
                    if values.notna().any()
                    else 0
                )

            discrete.append(is_categorical)

        valid = X.nunique() > 1

        discrete = np.array(discrete)[valid.to_numpy()]
        X = X.loc[:, valid]

        if len(X.columns) > 0 and len(y) >= 5 and y.nunique() > 1:

            selected_task = task

            if task == "auto":
                selected_task = (
                    "regression"
                    if (
                        pd.api.types.is_numeric_dtype(y)
                        and y.nunique() > 20
                    )
                    else "classification"
                )

            if selected_task == "classification":

                mi = mutual_info_classif(
                    X,
                    y.astype(str),
                    discrete_features=discrete,
                    random_state=42
                )

            else:

                mi = mutual_info_regression(
                    X,
                    pd.to_numeric(y),
                    discrete_features=discrete,
                    random_state=42
                )

            mi_scores = pd.Series(
                mi,
                index=X.columns
            ).sort_values(ascending=False)

            print("\nMutual Information:")
            print(mi_scores.head(15).round(4))

            mi_scores.head(15).sort_values().plot(
                kind="barh",
                figsize=(10, 6),
                color="#4C78A8"
            )

            plt.title(f"Mutual Information with {target}")
            plt.xlabel("Mutual Information")
            plt.tight_layout()
            plt.show()

    return {
        "overview": overview,
        "statistics": statistics,
        "correlation": correlation,
        "mutual_information": mi_scores
    }

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

    descriptive_analysis(df, target="Species", exclude=["Id"])

    X = df.drop(columns=["Id", "Species"])
    y = df["Species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)

    model.fit(X_train, y_train)
    evaluate_classifier(model, X_test, y_test)

def boston_house_prices():
    columns = [
        "CRIM",
        "ZN",
        "INDUS",
        "CHAS",
        "NOX",
        "RM",
        "AGE",
        "DIS",
        "RAD",
        "TAX",
        "PTRATIO",
        "B",
        "LSTAT",
        "MEDV"
    ]

    dataset_path = kagglehub.dataset_download("vikrishnan/boston-house-prices")
    csv_files = list(Path(dataset_path).rglob("*.csv"))

    df = pd.read_csv(csv_files[0], sep=r"\s+", header=None, names=columns)
    print(df.head())

    descriptive_analysis(df, target="MEDV", exclude=["Id"])

    X = df.drop(columns=["MEDV"])
    y = df["MEDV"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    result = {
        "tree_regration": DecisionTreeRegressor( max_depth=3, random_state=42).fit(X_train, y_train),
        "polynomialeatures": make_pipeline(PolynomialFeatures( degree=2,include_bias=False),
                                           LinearRegression()).fit(X_train, y_train)
    }

    statistics = []

    for name, model in result.items():
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        statistics.append({
            "Model": name,

            "Train R2": r2_score(
                y_train,
                y_train_pred
            ),

            "Test R2": r2_score(
                y_test,
                y_test_pred
            ),

            "MAE": mean_absolute_error(
                y_test,
                y_test_pred
            ),

            "RMSE": np.sqrt(
                mean_squared_error(
                    y_test,
                    y_test_pred
                )
            )
        })

    comparison = pd.DataFrame(statistics)
    print(comparison.round(4))

    fig, axes = plt.subplots(
        1,
        len(result),
        figsize=(15, 6)
    )

    for ax, (name, model) in zip(axes, result.items()):
        y_pred = model.predict(X_test)

        ax.scatter(
            y_test,
            y_pred,
            alpha=0.65,
            color="#4C78A8",
            edgecolors="white"
        )

        minimum = min(y_test.min(), y_pred.min())
        maximum = max(y_test.max(), y_pred.max())

        ax.plot(
            [minimum, maximum],
            [minimum, maximum],
            color="#E45756",
            linestyle="--",
            label="Ideal prediction"
        )

        ax.set_title(
            f"{name}\nR² = {r2_score(y_test, y_pred):.3f}"
        )

        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")

        ax.legend()
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()

def main():
   # iris()
   boston_house_prices()

if __name__ == "__main__":
    main()