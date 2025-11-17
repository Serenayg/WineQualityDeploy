import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

if __name__ == "__main__":
    df = pd.read_csv("data/clean_wine.csv")
    X = df.drop("quality", axis=1)
    y = df["quality"]

    model = LinearRegression()
    model.fit(X, y)

    os.makedirs("/opt/ml/model", exist_ok=True)
    joblib.dump(model, os.path.join("/opt/ml/model", "model.joblib"))