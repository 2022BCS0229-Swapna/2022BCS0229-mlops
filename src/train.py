import pandas as pd
import mlflow
import joblib
import json
import argparse

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

ROLL_NO = "2022BCS0229"
NAME = "Swapna"

def load_data(feature_set):
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    if feature_set == "reduced":
        X = X.iloc[:, :2]

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train(args):
    mlflow.set_experiment(f"{ROLL_NO}_experiment")

    X_train, X_test, y_train, y_test = load_data(args.feature_set)

    if args.model == "rf":
        model = RandomForestClassifier(n_estimators=args.n_estimators)
    else:
        model = SVC()

    with mlflow.start_run(run_name=args.run_name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")

        mlflow.log_param("model", args.model)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("feature_set", args.feature_set)
        mlflow.log_param("dataset_version", args.dataset_version)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        joblib.dump(model, "models/model.pkl")

        metrics = {
            "accuracy": acc,
            "f1_score": f1,
            "name": NAME,
            "roll_no": ROLL_NO
        }

        with open("metrics/metrics.json", "w") as f:
            json.dump(metrics, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, default="run1")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--feature_set", type=str, default="all")
    parser.add_argument("--model", type=str, default="rf")
    parser.add_argument("--dataset_version", type=str, default="v1")

    args = parser.parse_args()
    train(args)