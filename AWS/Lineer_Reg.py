import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib


def model_fn(model_dir):
    """Load trained model"""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])

    args = parser.parse_args()

    # Load CSV file
    file = os.path.join(args.train, "clean_wine.csv")
    dataset = pd.read_csv(file)

    # Split features and target
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Save model
    joblib.dump(regressor, os.path.join(args.model_dir, "model.joblib"))