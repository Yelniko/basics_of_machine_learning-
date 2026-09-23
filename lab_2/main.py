import kagglehub
from kagglehub import KaggleDatasetAdapter
from scipy.stats import linregress

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN


def main():
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "arashnic/imbalanced-data-practice",
        "aug_train.csv"
    )

    print(df.head())

    print("Розмир DataFrame:", df.shape)

    print("\nСтолбці:")
    print(df.columns.tolist())

    print("\nТипи:")
    print(df.dtypes)

    print("\nПропущенні значення:")
    print(df.isnull().sum())

    numeric_cols = ["Age", "Annual_Premium", "Vintage"]

    print(df[numeric_cols].describe())

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

    cov_matrix = df[numeric_cols].cov()

    print("Коваріація:", cov_matrix)

    pairs = [
        ("Age", "Annual_Premium"),
        ("Age", "Vintage"),
        ("Vintage", "Annual_Premium")
    ]

    x = df["Age"]
    y = df["Annual_Premium"]

    result = linregress(x, y)

    for x_col, y_col in pairs:
        corr = df[x_col].corr(df[y_col])

        plt.figure(figsize=(8, 5))

        plt.scatter(
            df[x_col],
            df[y_col],
            alpha=0.15
        )

        plt.xlabel(x_col)
        plt.ylabel(y_col)

        plt.title(
            f"{x_col} vs {y_col}, "
            f"r = {corr:.3f}"
        )

        plt.show()

    df_corr = df.copy()

    df_corr["Gender_Code"] = df_corr["Gender"].map({
        "Female": 0,
        "Male": 1
    })

    df_corr["Vehicle_Damage_Code"] = df_corr["Vehicle_Damage"].map({
        "No": 0,
        "Yes": 1
    })

    df_corr["Vehicle_Age_Code"] = df_corr["Vehicle_Age"].map({
        "< 1 Year": 0,
        "1-2 Year": 1,
        "> 2 Years": 2
    })

    corr_columns = [
        "Age",
        "Annual_Premium",
        "Vintage",
        "Driving_License",
        "Previously_Insured",
        "Gender_Code",
        "Vehicle_Damage_Code",
        "Vehicle_Age_Code",
        "Response"
    ]

    corr = df_corr[corr_columns].corr(
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

    plt.figure(figsize=(8, 5))

    plt.scatter(
        x,
        y,
        alpha=0.15
    )

    x_line = np.linspace(
        x.min(),
        x.max(),
        100
    )

    y_line = (
            result.intercept
            + result.slope * x_line
    )

    plt.plot(
        x_line,
        y_line
    )

    plt.xlabel("Age")
    plt.ylabel("Annual Premium")

    plt.title(
        f"Лінійна регресія, R² = "
        f"{result.rvalue ** 2:.4f}"
    )

    plt.show()

    #баланс

    features = [
        "Age",
        "Annual_Premium",
        "Vintage"
    ]

    X = df[features]
    y = df["Response"]

    plt.figure(figsize=(7, 4))

    plt.hist(
        y,
        bins=[-0.5, 0.5, 1.5],
        rwidth=0.8
    )

    plt.xticks([0, 1])

    plt.xlabel("Response")
    plt.ylabel("Кількість")
    plt.title("Баланс класів до SMOTE / ADASYN")

    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    sample_size = min(
        10000,
        len(X_train)
    )

    sample_index = X_train.sample(
        sample_size,
        random_state=42
    ).index

    plt.figure(figsize=(9, 6))

    for response in [0, 1]:
        mask = y_train.loc[sample_index] == response

        plt.scatter(
            X_train.loc[sample_index][mask]["Age"],
            X_train.loc[sample_index][mask]["Annual_Premium"],
            alpha=0.25,
            label=f"Response = {response}"
        )

    plt.xlabel("Age")
    plt.ylabel("Annual Premium")
    plt.title("Дані до SMOTE")

    plt.legend()
    plt.show()

    smote = SMOTE(
        random_state=42
    )

    X_smote, y_smote = smote.fit_resample(
        X_train,
        y_train
    )

    smote_df = X_smote.copy()
    smote_df["Response"] = y_smote.values

    sample_smote = smote_df.sample(
        min(15000, len(smote_df)),
        random_state=42
    )

    plt.figure(figsize=(9, 6))

    for response in [0, 1]:
        data = sample_smote[
            sample_smote["Response"] == response
            ]

        plt.scatter(
            data["Age"],
            data["Annual_Premium"],
            alpha=0.25,
            label=f"Response = {response}"
        )

    plt.xlabel("Age")
    plt.ylabel("Annual Premium")
    plt.title("Дані після SMOTE")

    plt.legend()
    plt.show()

    adasyn = ADASYN(
        random_state=42
    )

    X_adasyn, y_adasyn = adasyn.fit_resample(
        X_train,
        y_train
    )

    adasyn_df = X_adasyn.copy()

    adasyn_df["Response"] = (
        y_adasyn.values
    )

    sample_adasyn = adasyn_df.sample(
        min(15000, len(adasyn_df)),
        random_state=42
    )

    plt.figure(figsize=(9, 6))

    for response in [0, 1]:
        data = sample_adasyn[
            sample_adasyn["Response"] == response
            ]

        plt.scatter(
            data["Age"],
            data["Annual_Premium"],
            alpha=0.25,
            label=f"Response = {response}"
        )

    plt.xlabel("Age")
    plt.ylabel("Annual Premium")

    plt.title(
        "Дані після ADASYN"
    )

    plt.legend()

    plt.show()

if __name__ == '__main__':
    main()
