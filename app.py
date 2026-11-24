import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn.metrics
import streamlit as st

st.set_page_config(page_title="Credit Card Fraud Dashboard", layout="wide")

st.title("Credit Card Fraud Detection Dashboard")
st.write("Course project for 24CSE0316 (AI and Machine Learning)")


@st.cache_resource
def load_saved_stuff():
    return joblib.load("saved_models.pkl")


@st.cache_data
def load_raw_csv():
    return pd.read_csv("creditcard.csv")


try:
    artifacts = load_saved_stuff()
    df_raw = load_raw_csv()
except FileNotFoundError:
    st.error("Missing saved files! Run `python train.py` first to generate saved_models.pkl.")
    st.stop()

models = artifacts["models"]
results = artifacts["results"]
cv_scores = artifacts["cv_scores"]
X_test = artifacts["X_test"]
y_test = artifacts["y_test"]
X_train = artifacts["X_train"]

menu = st.sidebar.radio(
    "Navigation",
    [
        "Exploratory Data Analysis",
        "PCA Projection",
        "Model Results & Metrics",
        "Feature Importance",
        "Test Custom Transaction",
    ],
)

# ----------------------------
# Page 1: EDA
# ----------------------------
if menu == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis (EDA)")

    st.subheader("Class Distribution")
    counts = df_raw["Class"].value_counts()
    st.write(counts)

    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        df_raw["Class"].value_counts().plot(kind="bar", ax=ax1)
        ax1.set_title("Class counts (0 = normal, 1 = fraud)")
        ax1.set_yscale("log")
        st.pyplot(fig1)

    with col2:
        st.write("Percentage breakdown:")
        st.write(df_raw["Class"].value_counts(normalize=True) * 100)
        st.info("Note: Fraud accounts for only 0.17% of total records. Accuracy alone is misleading here.")

    st.subheader("Amount & Time Distributions")
    fig2, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].hist(df_raw[df_raw.Class == 0]["Amount"], bins=50, alpha=0.6, label="normal")
    axs[0].hist(df_raw[df_raw.Class == 1]["Amount"], bins=50, alpha=0.6, label="fraud")
    axs[0].set_title("Amount distribution")
    axs[0].legend()

    axs[1].hist(df_raw["Time"], bins=50)
    axs[1].set_title("Time distribution")
    st.pyplot(fig2)

    st.subheader("Correlations with Target Class")
    corr = df_raw.corr()["Class"].sort_values()
    fig3, ax3 = plt.subplots(figsize=(6, 8))
    corr.drop("Class").plot(kind="barh", ax=ax3)
    ax3.set_title("Correlation of each feature with Class")
    st.pyplot(fig3)

# ----------------------------
# Page 2: PCA
# ----------------------------
elif menu == "PCA Projection":
    st.header("PCA 2D Visualization")

    X_pca_train = artifacts["X_pca_train"]
    y_train = artifacts["y_train"]
    pca = artifacts["pca"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X_pca_train[y_train == 0, 0], X_pca_train[y_train == 0, 1], s=5, alpha=0.3, label="normal")
    ax.scatter(X_pca_train[y_train == 1, 0], X_pca_train[y_train == 1, 1], s=8, alpha=0.7, label="fraud", color="red")
    ax.legend()
    ax.set_title("2D PCA projection of training data")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    st.pyplot(fig)

    st.write("Explained variance ratio:", pca.explained_variance_ratio_)

# ----------------------------
# Page 3: Model Metrics
# ----------------------------
elif menu == "Model Results & Metrics":
    st.header("Model Evaluation")

    table_data = {}
    for name, res in results.items():
        table_data[name] = {k: v for k, v in res.items() if k not in ["preds", "probas"]}

    res_df = pd.DataFrame(table_data).T
    st.subheader("Test Set Performance (Sorted by F1-Score)")
    st.dataframe(res_df.sort_values("f1", ascending=False))

    st.subheader("5-Fold Cross-Validation Scores (F1)")
    cv_df = pd.DataFrame(list(cv_scores.items()), columns=["Model", "CV F1 Mean +/- Std"])
    st.table(cv_df)

    st.subheader("Confusion Matrix Display")
    selected_model_name = st.selectbox("Select Model to Display CM", list(models.keys()))

    cm = sklearn.metrics.confusion_matrix(y_test, results[selected_model_name]["preds"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=["normal", "fraud"])
    disp.plot(cmap="Blues", ax=ax)
    ax.set_title(selected_model_name)
    st.pyplot(fig)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("ROC Curves")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        for name, res in results.items():
            fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, res["probas"])
            ax2.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})")
        ax2.plot([0, 1], [0, 1], "k--")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.legend(fontsize=8)
        ax2.set_title("ROC Curves")
        st.pyplot(fig2)

    with c2:
        st.subheader("Precision-Recall Curves")
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        for name, res in results.items():
            prec, rec, _ = sklearn.metrics.precision_recall_curve(y_test, res["probas"])
            ax3.plot(rec, prec, label=f"{name} (PR-AUC={res['pr_auc']:.3f})")
        ax3.set_xlabel("Recall")
        ax3.set_ylabel("Precision")
        ax3.legend(fontsize=8)
        ax3.set_title("Precision-Recall Curves")
        st.pyplot(fig3)

# ----------------------------
# Page 4: Feature Importance
# ----------------------------
elif menu == "Feature Importance":
    st.header("Feature Importance")

    rf = models["Random Forest"]
    importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

    st.subheader("Random Forest - Top 15 Features")
    fig, ax = plt.subplots(figsize=(7, 5))
    importances.head(15).plot(kind="barh", ax=ax)
    ax.invert_yaxis()
    ax.set_title("Random Forest feature importance (top 15)")
    st.pyplot(fig)

# ----------------------------
# Page 5: Prediction Interface + SHAP
# ----------------------------
elif menu == "Test Custom Transaction":
    st.header("Predict Single Transaction Fraud")
    st.caption(
        "Use the buttons below to load a real unseen sample from the test set, or enter custom values manually."
    )

    # Initialize session state defaults
    if "sample_amount" not in st.session_state:
        st.session_state["sample_amount"] = 150.0
        st.session_state["sample_time"] = 25000.0
        for i in range(1, 29):
            st.session_state[f"V{i}"] = 0.0

    # One-click test sample loaders from raw test indices
    b1, b2 = st.columns(2)
    if b1.button("Load Random Fraud Sample (from Test Set)"):
        fraud_indices = y_test[y_test == 1].index
        chosen_idx = np.random.choice(fraud_indices)
        raw_row = df_raw.loc[chosen_idx]
        st.session_state["sample_amount"] = float(raw_row["Amount"])
        st.session_state["sample_time"] = float(raw_row["Time"])
        for i in range(1, 29):
            st.session_state[f"V{i}"] = float(raw_row[f"V{i}"])

    if b2.button("Load Random Normal Sample (from Test Set)"):
        normal_indices = y_test[y_test == 0].index
        chosen_idx = np.random.choice(normal_indices)
        raw_row = df_raw.loc[chosen_idx]
        st.session_state["sample_amount"] = float(raw_row["Amount"])
        st.session_state["sample_time"] = float(raw_row["Time"])
        for i in range(1, 29):
            st.session_state[f"V{i}"] = float(raw_row[f"V{i}"])

    chosen_model_name = st.selectbox("Choose Classification Model", list(models.keys()))
    selected_model = models[chosen_model_name]

    c1, c2 = st.columns(2)
    with c1:
        user_amount = st.number_input("Transaction Amount ($)", key="sample_amount")
    with c2:
        user_time = st.number_input("Time Elapsed (seconds)", key="sample_time")

    st.write("##### PCA Features (V1 - V28):")
    v_dict = {}
    cols = st.columns(4)
    for i in range(1, 29):
        v_dict[f"V{i}"] = cols[(i - 1) % 4].number_input(f"V{i}", step=0.1, key=f"V{i}")

    if st.button("Predict Fraud Status"):
        scaled_amount_val = artifacts["amount_scaler"].transform([[user_amount]])[0][0]
        scaled_time_val = artifacts["time_scaler"].transform([[user_time]])[0][0]

        input_data = v_dict.copy()
        input_data["scaled_amount"] = scaled_amount_val
        input_data["scaled_time"] = scaled_time_val

        input_df = pd.DataFrame([input_data])[X_train.columns]

        pred = selected_model.predict(input_df)[0]
        prob = selected_model.predict_proba(input_df)[0][1]

        st.write("---")
        if pred == 1:
            st.error(f"Prediction: **FRAUD DETECTED** | Fraud Probability: {prob:.2%}")
        else:
            st.success(f"Prediction: **NORMAL TRANSACTION** | Fraud Probability: {prob:.2%}")

        # Local SHAP explanation for tree-based models
        if chosen_model_name in ["Random Forest", "Decision Tree"]:
            st.subheader("Top 3 Features Driving This Prediction (SHAP)")
            explainer = shap.TreeExplainer(selected_model)
            raw_shap = explainer.shap_values(input_df)

            if isinstance(raw_shap, list):
                row_shap = raw_shap[1][0]
            elif raw_shap.ndim == 3:
                row_shap = raw_shap[0, :, 1]
            else:
                row_shap = raw_shap[0]

            shap_df = pd.DataFrame(
                {
                    "Feature": X_train.columns,
                    "Input Value": input_df.iloc[0].values,
                    "SHAP Impact (Fraud Risk)": row_shap,
                }
            )
            shap_df["Abs Impact"] = shap_df["SHAP Impact (Fraud Risk)"].abs()
            top_3 = shap_df.sort_values("Abs Impact", ascending=False).head(3).drop(columns=["Abs Impact"])

            st.table(top_3)
            st.caption("Positive SHAP impact increases fraud probability; negative SHAP impact lowers it.")