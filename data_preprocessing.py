import pandas as pd
import sklearn.preprocessing
import sklearn.model_selection
from sklearn.impute import SimpleImputer


def load_and_preprocess_data(csv_path="creditcard.csv"):
    # Load dataset
    df = pd.read_csv(csv_path)

    # Separate features and target BEFORE fitting any imputer or scaler
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Stratified 80/20 train-test split
    X_train_raw, X_test_raw, y_train, y_test = sklearn.model_selection.train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()

    # 1. Fit median imputer ONLY on training data
    num_cols = X_train.select_dtypes(include=["number"]).columns
    imputer = SimpleImputer(strategy="median")
    X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
    X_test[num_cols] = imputer.transform(X_test[num_cols])

    # 2. Fit scalers ONLY on training data (RobustScaler handles Amount outliers better)
    amount_scaler = sklearn.preprocessing.RobustScaler()
    time_scaler = sklearn.preprocessing.StandardScaler()

    X_train["scaled_amount"] = amount_scaler.fit_transform(X_train[["Amount"]])
    X_test["scaled_amount"] = amount_scaler.transform(X_test[["Amount"]])

    X_train["scaled_time"] = time_scaler.fit_transform(X_train[["Time"]])
    X_test["scaled_time"] = time_scaler.transform(X_test[["Time"]])

    # Drop raw unscaled columns
    X_train = X_train.drop(["Amount", "Time"], axis=1)
    X_test = X_test.drop(["Amount", "Time"], axis=1)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "amount_scaler": amount_scaler,
        "time_scaler": time_scaler,
        "imputer": imputer,
    }