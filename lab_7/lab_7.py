import kagglehub

import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif

from sklearn.feature_selection import (
    mutual_info_regression,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)
from sklearn.metrics import (
    matthews_corrcoef,
    balanced_accuracy_score,
    accuracy_score,
    recall_score,
    f1_score,
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

def main():
    dataset_path = kagglehub.dataset_download("fridrichmrtn/user-churn-dataset")
    csv_files = list(Path(dataset_path).rglob("*.csv"))

    df = pd.read_csv(csv_files[0])
    print(df.head())

    # descriptive_analysis(df, target="target_class", exclude=["Id"])

    data = df.copy()

    X = data.drop(columns=["target_class", "visitorid", "time_to_tran"], errors="ignore")
    X = X.replace(-1, np.nan)

    y = data["target_class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {

        "KNN": make_pipeline(
            SimpleImputer(
                strategy="median",
                add_indicator=True
            ),
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=5
            )
        ),

        "Random Forest": make_pipeline(
            SimpleImputer(
                strategy="median",
                add_indicator=True
            ),
            RandomForestClassifier(
                n_estimators=150,
                random_state=42,
                n_jobs=-1
            )
        ),

        "SVM": make_pipeline(
            SimpleImputer(
                strategy="median",
                add_indicator=True
            ),
            StandardScaler(),
            LinearSVC(
                C=1.0,
                max_iter=5000,
                random_state=42
            )
        ),

        "Gradient Boosting": make_pipeline(
            SimpleImputer(
                strategy="median",
                add_indicator=True
            ),
            HistGradientBoostingClassifier(
                max_iter=150,
                random_state=42
            )
        )
    }

    statistics = []

    for name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        statistics.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(
                y_test, y_pred
            ),
            "F1 Macro": f1_score(
                y_test, y_pred,
                average="macro"
            ),
            "MCC": matthews_corrcoef(
                y_test, y_pred
            ),
            "Recall Class 0": recall_score(
                y_test, y_pred,
                pos_label=0,
                zero_division=0
            )
        })

    comparison = pd.DataFrame(statistics)

    print(
        comparison.sort_values(
            "MCC",
            ascending=False
        ).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


if __name__ == "__main__":
    main()