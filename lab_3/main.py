import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

def ridge(df):
    X = df.drop(columns="MedHouseVal")
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=1.0)

    model.fit(
        X_train_scaled,
        y_train
    )

    y_pred = model.predict(X_test_scaled)

    print(
        "R2:",
        r2_score(y_test, y_pred)
    )

    print(
        "MSE:",
        mean_squared_error(y_test, y_pred)
    )

    coefficients = pd.DataFrame({
        "feature": X.columns,
        "coefficient": model.coef_
    })

    print(coefficients.sort_values(
        "coefficient",
        key=abs,
        ascending=False
    ))

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.4
    )

    minimum = min(y_test.min(), y_pred.min())
    maximum = max(y_test.max(), y_pred.max())

    plt.plot(
        [minimum, maximum],
        [minimum, maximum]
    )

    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.title("Ridge Regression: Actual vs Predicted")

    plt.show()

def lasso(df):
    X = df.drop(columns="MedHouseVal")
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Lasso(
        alpha=0.01,
        max_iter=10000
    )

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    print("R2:", r2_score(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))

    coefficients = pd.DataFrame({
        "feature": X.columns,
        "coefficient": model.coef_
    })

    print(
        coefficients.sort_values(
            "coefficient",
            key=abs,
            ascending=False
        )
    )

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.4
    )

    minimum = min(y_test.min(), y_pred.min())
    maximum = max(y_test.max(), y_pred.max())

    plt.plot(
        [minimum, maximum],
        [minimum, maximum]
    )

    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.title("Lasso Regression: Actual vs Predicted")

    plt.show()

def elastic_net(df):
    X = df.drop(columns="MedHouseVal")
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = ElasticNet(
        alpha=0.01,
        l1_ratio=0.5,
        max_iter=10000
    )

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    print("R2:", r2_score(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))

    coefficients = pd.DataFrame({
        "feature": X.columns,
        "coefficient": model.coef_
    })

    print(
        coefficients.sort_values(
            "coefficient",
            key=abs,
            ascending=False
        )
    )

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.4
    )

    minimum = min(y_test.min(), y_pred.min())
    maximum = max(y_test.max(), y_pred.max())

    plt.plot(
        [minimum, maximum],
        [minimum, maximum]
    )

    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.title("Elastic Net: Actual vs Predicted")

    plt.show()

def main():
    housing = fetch_california_housing(as_frame=True)

    print(housing.feature_names)
    print(housing.target_names)

    df = housing.frame

    ridge(df)
    lasso(df)
    elastic_net(df)

if __name__ == "__main__":
    main()