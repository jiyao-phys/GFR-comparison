import os
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings("ignore")

TRAIN_FILE = "data/allelement0320_magpie_features_rfe_alloy_selected_features.xlsx"
OUT_DIR = "out/atom/metal0320/shap_outputs"
OUT_XLSX = os.path.join(OUT_DIR, "shap_exports.xlsx")
TEST_SIZE = 0.2
RANDOM_STATE = 42
TOP_N_SUMMARY = 10
TOP_N_DEPENDENCY = 6

BEST_XGB_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 3,
    "subsample": 0.75,
    "colsample_bytree": 0.6,
    "gamma": 0.1,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "random_state": RANDOM_STATE,
    "use_label_encoder": False,
    "verbosity": 0,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_jobs": 1,
}

os.makedirs(OUT_DIR, exist_ok=True)


def read_data(fname):
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Cannot find {fname}")
    df = pd.read_excel(fname, sheet_name=0)
    if df.shape[1] < 2:
        raise ValueError("Input must have at least 2 columns (features + label)")
    X_df = df.iloc[:, :-1].copy()
    y = pd.Series(df.iloc[:, -1].values).astype(int).values
    return X_df, y


def train_model(X_train, y_train, params):
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train, y_train)
    return clf


def get_shap_values(clf, X_df):
    explainer = shap.TreeExplainer(clf, feature_perturbation="interventional")
    try:
        exp = explainer(X_df)
        vals = np.asarray(exp.values)
    except Exception:
        vals = np.asarray(explainer.shap_values(X_df))
    if vals.ndim == 3:
        shap_vals = vals[:, :, -1]
    else:
        shap_vals = vals
    return shap_vals


def find_knee_point(x_data, y_data, window_length=5, polyorder=2):
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    if len(x_data) < window_length:
        return float(np.median(x_data))
    if window_length % 2 == 0:
        window_length += 1
    if polyorder >= window_length:
        polyorder = window_length - 1
        if polyorder < 1:
            polyorder = 1
    order = np.argsort(x_data)
    sx = x_data[order]
    sy = y_data[order]
    y2 = savgol_filter(sy, window_length, polyorder, deriv=2)
    return float(sx[np.argmax(np.abs(y2))])


def build_summary(X_test, shap_values, top_n):
    mean_abs = np.abs(shap_values).mean(axis=0)
    full = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    summary = full.head(top_n).sort_values("mean_abs_shap", ascending=True).reset_index(drop=True)
    return full, summary


def build_dependency_table(X_test, y_test, shap_values, top_n):
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_features = (
        pd.DataFrame({"feature": X_test.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)["feature"]
        .tolist()
    )
    rows = []
    for feature in top_features:
        idx = X_test.columns.get_loc(feature)
        x_data = X_test[feature]
        y_data = shap_values[:, idx]
        try:
            threshold_val = find_knee_point(x_data, y_data)
        except Exception:
            threshold_val = float(x_data.median())
        rows.append({
            "feature": feature,
            "median": float(x_data.median()),
            "mean": float(x_data.mean()),
            "std": float(x_data.std()),
            "min": float(x_data.min()),
            "max": float(x_data.max()),
            "shap_mean": float(y_data.mean()),
            "shap_abs_mean": float(np.abs(y_data).mean()),
            "knee_threshold": threshold_val,
        })
    return pd.DataFrame(rows)


def main():
    X_df, y = read_data(TRAIN_FILE)
    print(f"Loaded {X_df.shape[0]} samples, {X_df.shape[1]} features.")

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train {X_train.shape[0]} | Test {X_test.shape[0]}")

    clf = train_model(X_train.values, y_train, BEST_XGB_PARAMS)

    y_pred = clf.predict(X_test.values)
    y_pred_proba = clf.predict_proba(X_test.values)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Accuracy {acc:.4f} | AUC {auc:.4f}")

    shap_values = get_shap_values(clf, X_test)

    full_importance, summary_df = build_summary(X_test, shap_values, TOP_N_SUMMARY)
    dependency_df = build_dependency_table(X_test, y_test, shap_values, TOP_N_DEPENDENCY)

    metrics_df = pd.DataFrame({
        "metric": ["accuracy", "auc", "n_test", "n_features"],
        "value": [acc, auc, X_test.shape[0], X_test.shape[1]],
    })

    shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
    feature_df = X_test.reset_index(drop=True)
    feature_df.insert(0, "y_true", y_test)
    feature_df.insert(1, "y_pred", y_pred)
    feature_df.insert(2, "y_proba", y_pred_proba)

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="model_metrics", index=False)
        full_importance.to_excel(writer, sheet_name="feature_importance", index=False)
        summary_df.to_excel(writer, sheet_name="summary_top_features", index=False)
        dependency_df.to_excel(writer, sheet_name="dependency_knee_points", index=False)
        shap_df.to_excel(writer, sheet_name="shap_values_test", index=False)
        feature_df.to_excel(writer, sheet_name="test_set_predictions", index=False)

    print(f"Saved {OUT_XLSX}")


if __name__ == "__main__":
    main()
