import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split

warnings.filterwarnings("ignore")

IN_FILE = "data/allelement0320_magpie_features_rfe_oxide_selected_features.xlsx"
OOF_OUT = "out/atom/oxide0320/oof_results.xlsx"
ROC_OUT = "out/atom/oxide0320/roc_data.xlsx"
CONF_OUT = "out/atom/oxide0320/confusion_metrics.xlsx"
METRICS_OUT = "out/atom/oxide0320/metrics_data.xlsx"
METRICS_FIG = "out/atom/oxide0320/metrics_figure.png"

RNG = 42
N_SPLITS_FINAL = 10
FPR_GRID_POINTS = 101
LC_POINTS = 20
LC_TEST_SIZE = 0.2
RATIOS = np.linspace(0.05, 0.5, 10)
RATIO_REPEATS = 10
THR_GRID = np.linspace(0.0, 1.0, 101)
SELECTED_THRESHOLD = 0.35

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
    "random_state": RNG,
    "use_label_encoder": False,
    "verbosity": 0,
    "objective": "binary:logistic",
    "eval_metric": "auc",
}


def prepare_output_dirs(*paths):
    for path in paths:
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)


def safe_div(a, b):
    return a / b if b else np.nan


def make_model():
    return xgb.XGBClassifier(**BEST_XGB_PARAMS)


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {path} in current folder.")

    df = pd.read_excel(path, sheet_name=0)
    if df.shape[1] < 2:
        raise ValueError("Input data must have at least 2 columns.")

    x = df.iloc[:, :-1].values
    y = pd.Series(df.iloc[:, -1].values).astype(int).values
    print(f"Loaded {x.shape[0]} samples, {x.shape[1]} features.")
    return x, y


def metric_counts(tn, fp, fn, tp):
    total = tp + tn + fp + fn
    out = {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "TPR": safe_div(tp, tp + fn),
        "FNR": safe_div(fn, tp + fn),
        "TNR": safe_div(tn, tn + fp),
        "FPR": safe_div(fp, tn + fp),
        "PPV": safe_div(tp, tp + fp),
        "FDR": safe_div(fp, tp + fp),
        "NPV": safe_div(tn, tn + fn),
        "FOR": safe_div(fn, tn + fn),
        "ACC": safe_div(tp + tn, total),
    }
    return out


def metric_row(values, label=None, fold=None):
    row = {}
    if fold is not None:
        row["fold"] = fold
    if label is not None:
        row["threshold_type"] = label

    row.update({f"{k}_count": values[k] for k in ("TP", "FP", "TN", "FN")})
    row.update({k: values[k] for k in ("TPR", "FNR", "TNR", "FPR", "PPV", "FDR", "NPV", "FOR", "ACC")})
    row.update({
        f"{k}_pct": values[k] * 100 if isinstance(values[k], float) else np.nan
        for k in ("TPR", "FNR", "TNR", "FPR", "PPV", "FDR", "NPV", "FOR", "ACC")
    })
    return row


def summarize_confusion(confusions, label):
    rows = []
    totals = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

    for fold, (tn, fp, fn, tp) in enumerate(confusions, start=1):
        values = metric_counts(tn, fp, fn, tp)
        rows.append(metric_row(values, label=label, fold=fold))
        for key in totals:
            totals[key] += values[key]

    agg = metric_counts(totals["TN"], totals["FP"], totals["FN"], totals["TP"])
    return pd.DataFrame(rows), pd.DataFrame([metric_row(agg, label=label)])


def run_final_cv(x, y):
    split = StratifiedKFold(n_splits=N_SPLITS_FINAL, shuffle=True, random_state=RNG)
    n_samples = x.shape[0]

    fpr_grid = np.linspace(0.0, 1.0, FPR_GRID_POINTS)
    oof_proba = np.zeros(n_samples, dtype=float)
    oof_pred = np.zeros(n_samples, dtype=int)
    oof_fold = np.full(n_samples, -1, dtype=int)

    train_probs = []
    train_y = []
    test_probs = []
    test_y = []
    tpr_on_grid = []
    aucs = []
    youden_thresholds = []
    conf_05 = []
    conf_youden = []

    for fold, (train_idx, test_idx) in enumerate(split.split(x, y), start=1):
        print(f"Final CV: training fold {fold}/{N_SPLITS_FINAL} ...")
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = make_model()
        model.fit(x_train, y_train)

        p_train = model.predict_proba(x_train)[:, 1]
        p_test = model.predict_proba(x_test)[:, 1]

        train_probs.append(p_train)
        train_y.append(y_train)
        test_probs.append(p_test)
        test_y.append(y_test)

        oof_proba[test_idx] = p_test
        oof_fold[test_idx] = fold - 1

        fpr, tpr, thresholds = roc_curve(y_test, p_test)
        aucs.append(roc_auc_score(y_test, p_test))
        tpr_on_grid.append(np.interp(fpr_grid, fpr, tpr))

        best_idx = np.argmax(tpr - fpr)
        best_thr = thresholds[best_idx]
        youden_thresholds.append(best_thr)

        pred_05 = (p_test >= 0.5).astype(int)
        conf_05.append(confusion_matrix(y_test, pred_05, labels=[0, 1]).ravel())

        pred_youden = (p_test >= best_thr).astype(int)
        conf_youden.append(confusion_matrix(y_test, pred_youden, labels=[0, 1]).ravel())

        oof_pred[test_idx] = pred_05

    return {
        "fpr_grid": fpr_grid,
        "oof_proba": oof_proba,
        "oof_pred": oof_pred,
        "oof_fold": oof_fold,
        "train_probs": train_probs,
        "train_y": train_y,
        "test_probs": test_probs,
        "test_y": test_y,
        "tpr_matrix": np.vstack(tpr_on_grid),
        "aucs": np.array(aucs),
        "youden_thresholds": np.array(youden_thresholds),
        "conf_05": conf_05,
        "conf_youden": conf_youden,
    }


def save_oof(y, cv):
    out = pd.DataFrame({
        "index": np.arange(len(y)),
        "y_true": y,
        "y_proba": cv["oof_proba"],
        "y_pred_0.5": cv["oof_pred"],
        "fold": cv["oof_fold"],
    })
    out.to_excel(OOF_OUT, index=False)
    print(f"Saved OOF predictions to {OOF_OUT}")


def roc_tables(cv):
    matrix = cv["tpr_matrix"]
    fpr_grid = cv["fpr_grid"]
    mean_tpr = np.mean(matrix, axis=0)
    std_tpr = np.std(matrix, axis=0)

    roc_mean = pd.DataFrame({"fpr": fpr_grid, "mean_tpr": mean_tpr, "std_tpr": std_tpr})
    roc_folds = pd.DataFrame(matrix.T, columns=[f"tpr_f{i + 1}" for i in range(matrix.shape[0])])
    roc_folds.insert(0, "fpr", fpr_grid)
    auc_df = pd.DataFrame({"fold": np.arange(1, len(cv["aucs"]) + 1), "auc": cv["aucs"]})
    auc_summary = pd.DataFrame({"mean_auc": [np.mean(cv["aucs"])], "std_auc": [np.std(cv["aucs"])]})
    return roc_mean, roc_folds, auc_df, auc_summary


def save_roc(cv):
    roc_mean, roc_folds, auc_df, auc_summary = roc_tables(cv)
    with pd.ExcelWriter(ROC_OUT, engine="openpyxl") as writer:
        roc_mean.to_excel(writer, sheet_name="roc_mean_std", index=False)
        roc_folds.to_excel(writer, sheet_name="roc_per_fold_interp", index=False)
        auc_df.to_excel(writer, sheet_name="auc_per_fold", index=False)
        auc_summary.to_excel(writer, sheet_name="auc_summary", index=False)
    print(f"Saved ROC data to {ROC_OUT}")
    return roc_mean, roc_folds


def oof_confusion_table(y, proba, threshold, label):
    tn, fp, fn, tp = confusion_matrix(y, (proba >= threshold).astype(int), labels=[0, 1]).ravel()
    values = metric_counts(tn, fp, fn, tp)
    return pd.DataFrame([metric_row(values, label=label)])


def oof_youden_table(y, proba):
    fpr, tpr, thresholds = roc_curve(y, proba)
    threshold = thresholds[np.argmax(tpr - fpr)]
    out = oof_confusion_table(y, proba, threshold, "oof_youden")
    out.insert(1, "youden_threshold", threshold)
    return out


def save_confusion(y, cv):
    per_05, agg_05 = summarize_confusion(cv["conf_05"], "0.5")
    per_youden, agg_youden = summarize_confusion(cv["conf_youden"], "youden")
    agg_oof_05 = oof_confusion_table(y, cv["oof_proba"], 0.5, "oof_0.5")
    agg_oof_youden = oof_youden_table(y, cv["oof_proba"])

    with pd.ExcelWriter(CONF_OUT, engine="openpyxl") as writer:
        per_05.to_excel(writer, sheet_name="per_fold_conf_0.5", index=False)
        agg_05.to_excel(writer, sheet_name="agg_conf_0.5", index=False)
        per_youden.to_excel(writer, sheet_name="per_fold_conf_youden", index=False)
        agg_youden.to_excel(writer, sheet_name="agg_conf_youden", index=False)
        agg_oof_05.to_excel(writer, sheet_name="agg_oof_0.5", index=False)
        agg_oof_youden.to_excel(writer, sheet_name="agg_oof_youden", index=False)
        pd.DataFrame({
            "fold": np.arange(1, len(cv["youden_thresholds"]) + 1),
            "youden_threshold": cv["youden_thresholds"],
            "auc": cv["aucs"],
        }).to_excel(writer, sheet_name="fold_thresholds_aucs", index=False)

    print(f"Saved confusion metrics and summaries to {CONF_OUT}")


def learning_curve_data(x, y):
    x_pool, x_test, y_pool, y_test = train_test_split(
        x,
        y,
        test_size=LC_TEST_SIZE,
        stratify=y,
        random_state=RNG,
    )
    n_pool = x_pool.shape[0]
    if n_pool < 2:
        raise ValueError("Not enough samples in training pool.")

    low = max(2, int(max(1, n_pool * 0.05)))
    sizes = np.unique(np.linspace(low, n_pool, LC_POINTS, dtype=int))
    used, train_mse, test_mse = [], [], []

    for size in sizes:
        if size >= n_pool:
            x_sub, y_sub = x_pool, y_pool
        else:
            try:
                sampler = StratifiedShuffleSplit(n_splits=1, train_size=size, random_state=RNG)
                idx, _ = next(sampler.split(x_pool, y_pool))
                x_sub, y_sub = x_pool[idx], y_pool[idx]
            except ValueError:
                continue

        try:
            model = make_model()
            model.fit(x_sub, y_sub)
            p_train = model.predict_proba(x_sub)[:, 1]
            p_test = model.predict_proba(x_test)[:, 1]
        except Exception as exc:
            print(f"  warning: skipping size {size} due to error: {exc}")
            continue

        used.append(len(y_sub))
        train_mse.append(np.mean((y_sub - p_train) ** 2))
        test_mse.append(np.mean((y_test - p_test) ** 2))

    if not used:
        raise RuntimeError("No valid training sizes could be used.")

    return np.array(used), np.array(train_mse), np.array(test_mse)


def accuracy_by_ratio(x, y):
    means, stds = [], []
    for ratio in RATIOS:
        test_size = ratio / (1.0 + ratio)
        scores = []
        for rep in range(RATIO_REPEATS):
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=test_size,
                stratify=y,
                random_state=RNG + rep,
            )
            model = make_model()
            model.fit(x_train, y_train)
            scores.append(accuracy_score(y_test, model.predict(x_test)))
        means.append(np.mean(scores))
        stds.append(np.std(scores))
    return np.array(means), np.array(stds)


def mean_precision(curves, grid):
    rows = []
    for precision, recall in curves:
        try:
            rows.append(np.interp(grid, recall[::-1], precision[::-1]))
        except Exception:
            rows.append(np.full_like(grid, np.nan))
    rows = np.vstack(rows)
    return grid, np.nanmean(rows, axis=0), np.nanstd(rows, axis=0)


def mean_score_by_threshold(probs_list, y_list, scorer):
    values = []
    for probs, y_true in zip(probs_list, y_list):
        values.append([scorer(y_true, (probs >= thr).astype(int)) for thr in THR_GRID])
    values = np.array(values)
    return THR_GRID, np.nanmean(values, axis=0), np.nanstd(values, axis=0)


def selected_threshold_confusion(probs_list, y_list):
    rows = []
    totals = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

    for probs, y_true in zip(probs_list, y_list):
        pred = (probs >= SELECTED_THRESHOLD).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        values = metric_counts(tn, fp, fn, tp)
        row = {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
        row.update(values)
        rows.append(row)
        for key in totals:
            totals[key] += int(values[key])

    agg = metric_counts(totals["TN"], totals["FP"], totals["FN"], totals["TP"])
    agg_row = {"TN": totals["TN"], "FP": totals["FP"], "FN": totals["FN"], "TP": totals["TP"]}
    agg_row.update(agg)
    return pd.DataFrame(rows), pd.DataFrame([agg_row])


def oof_counts_at_selected_threshold(y, proba):
    tn, fp, fn, tp = confusion_matrix(y, (proba >= SELECTED_THRESHOLD).astype(int), labels=[0, 1]).ravel()
    values = metric_counts(tn, fp, fn, tp)
    row = {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
    row.update(values)
    return pd.DataFrame([row])


def make_plot_data(x, y, cv, roc_mean, roc_folds):
    lc_size, lc_train_mse, lc_test_mse = learning_curve_data(x, y)
    acc_mean, acc_std = accuracy_by_ratio(x, y)

    rec_grid = np.linspace(0.0, 1.0, 101)
    train_pr = []
    test_pr = []
    for p_train, y_train, p_test, y_test in zip(
        cv["train_probs"],
        cv["train_y"],
        cv["test_probs"],
        cv["test_y"],
    ):
        train_pr.append(precision_recall_curve(y_train, p_train)[:2])
        test_pr.append(precision_recall_curve(y_test, p_test)[:2])

    rec_grid, mean_prec_train, std_prec_train = mean_precision(train_pr, rec_grid)
    _, mean_prec_test, std_prec_test = mean_precision(test_pr, rec_grid)

    thr, mean_f1_train, std_f1_train = mean_score_by_threshold(
        cv["train_probs"],
        cv["train_y"],
        lambda a, b: f1_score(a, b, zero_division=0),
    )
    _, mean_f1_test, std_f1_test = mean_score_by_threshold(
        cv["test_probs"],
        cv["test_y"],
        lambda a, b: f1_score(a, b, zero_division=0),
    )
    thr_acc, mean_acc_train, std_acc_train = mean_score_by_threshold(
        cv["train_probs"],
        cv["train_y"],
        accuracy_score,
    )
    _, mean_acc_test, std_acc_test = mean_score_by_threshold(
        cv["test_probs"],
        cv["test_y"],
        accuracy_score,
    )

    conf_fold, conf_agg = selected_threshold_confusion(cv["test_probs"], cv["test_y"])
    conf_oof = oof_counts_at_selected_threshold(y, cv["oof_proba"])

    with pd.ExcelWriter(METRICS_OUT, engine="openpyxl") as writer:
        pd.DataFrame({
            "train_size": lc_size,
            "train_mse": lc_train_mse,
            "test_mse": lc_test_mse,
        }).to_excel(writer, sheet_name="learning_curve", index=False)
        pd.DataFrame({
            "ratio": RATIOS,
            "acc_mean": acc_mean,
            "acc_std": acc_std,
        }).to_excel(writer, sheet_name="acc_vs_ratio", index=False)
        pd.DataFrame({
            "recall": rec_grid,
            "mean_precision_train": mean_prec_train,
            "std_precision_train": std_prec_train,
        }).to_excel(writer, sheet_name="pr_train", index=False)
        pd.DataFrame({
            "recall": rec_grid,
            "mean_precision_test": mean_prec_test,
            "std_precision_test": std_prec_test,
        }).to_excel(writer, sheet_name="pr_test", index=False)
        pd.DataFrame({
            "threshold": thr,
            "mean_f1_train": mean_f1_train,
            "std_f1_train": std_f1_train,
            "mean_f1_test": mean_f1_test,
            "std_f1_test": std_f1_test,
        }).to_excel(writer, sheet_name="f1_vs_threshold", index=False)
        pd.DataFrame({
            "threshold": thr_acc,
            "mean_acc_train": mean_acc_train,
            "std_acc_train": std_acc_train,
            "mean_acc_test": mean_acc_test,
            "std_acc_test": std_acc_test,
        }).to_excel(writer, sheet_name="acc_vs_threshold", index=False)
        roc_mean.to_excel(writer, sheet_name="roc_mean_std", index=False)
        roc_folds.to_excel(writer, sheet_name="roc_per_fold_interp", index=False)
        conf_fold.to_excel(writer, sheet_name=f"conf_per_fold_thr_{SELECTED_THRESHOLD}", index=True)
        conf_agg.to_excel(writer, sheet_name=f"conf_agg_thr_{SELECTED_THRESHOLD}", index=False)
        conf_oof.to_excel(writer, sheet_name=f"oof_conf_thr_{SELECTED_THRESHOLD}", index=False)

    print(f"Saved b-f data to {METRICS_OUT}")
    return {
        "lc_size": lc_size,
        "lc_train_mse": lc_train_mse,
        "lc_test_mse": lc_test_mse,
        "acc_mean": acc_mean,
        "acc_std": acc_std,
        "rec_grid": rec_grid,
        "mean_prec_train": mean_prec_train,
        "std_prec_train": std_prec_train,
        "mean_prec_test": mean_prec_test,
        "std_prec_test": std_prec_test,
        "thr": thr,
        "mean_f1_train": mean_f1_train,
        "std_f1_train": std_f1_train,
        "mean_f1_test": mean_f1_test,
        "std_f1_test": std_f1_test,
        "thr_acc": thr_acc,
        "mean_acc_train": mean_acc_train,
        "std_acc_train": std_acc_train,
        "mean_acc_test": mean_acc_test,
        "std_acc_test": std_acc_test,
    }


def draw_metrics(y, cv, data):
    plt.rcParams.update({"figure.figsize": (16, 10)})
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    ax = axs[0, 0]
    ax.plot(data["lc_size"], data["lc_train_mse"], label="training")
    ax.plot(data["lc_size"], data["lc_test_mse"], label="test")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Mean squared error")
    ax.set_title("(b) Learning curves")
    ax.legend()
    ax.grid(True)

    ax = axs[0, 1]
    ax.errorbar(RATIOS, data["acc_mean"], yerr=data["acc_std"], marker="o", linestyle="-")
    ax.set_xlabel("Ratio of test to training set size")
    ax.set_ylabel("Accuracy")
    ax.set_title("(c) ACC vs test/train ratio")
    ax.grid(True)
    ax.axhline(max(np.mean(y == 0), np.mean(y == 1)), color="gray", linestyle="--", label="majority baseline")
    ax.legend()

    pred_oof = (cv["oof_proba"] >= SELECTED_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred_oof, labels=[0, 1]).ravel()
    recall_thr = safe_div(tp, tp + fn)
    precision_thr = safe_div(tp, tp + fp)

    ax = axs[0, 2]
    ax.plot(data["rec_grid"], data["mean_prec_test"], label="test (mean)")
    ax.fill_between(
        data["rec_grid"],
        data["mean_prec_test"] - data["std_prec_test"],
        data["mean_prec_test"] + data["std_prec_test"],
        alpha=0.2,
    )
    ax.plot(data["rec_grid"], data["mean_prec_train"], label="train (mean)")
    ax.fill_between(
        data["rec_grid"],
        data["mean_prec_train"] - data["std_prec_train"],
        data["mean_prec_train"] + data["std_prec_train"],
        alpha=0.2,
    )
    ax.hlines(np.mean(y == 1), 0, 1, linestyles="--", label="random guess")
    ax.scatter(
        [recall_thr],
        [precision_thr],
        color="red",
        zorder=10,
        label=f"thr={SELECTED_THRESHOLD} (rec={recall_thr:.3f}, prec={precision_thr:.3f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("(d) Precision-Recall curves")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)

    ax = axs[1, 0]
    ax.plot(THR_GRID, data["mean_f1_train"], label="train")
    ax.fill_between(
        THR_GRID,
        data["mean_f1_train"] - data["std_f1_train"],
        data["mean_f1_train"] + data["std_f1_train"],
        alpha=0.2,
    )
    ax.plot(THR_GRID, data["mean_f1_test"], label="test")
    ax.fill_between(
        THR_GRID,
        data["mean_f1_test"] - data["std_f1_test"],
        data["mean_f1_test"] + data["std_f1_test"],
        alpha=0.2,
    )
    ax.axvline(SELECTED_THRESHOLD, color="gray", linestyle="--", linewidth=0.8)
    ax.scatter(
        [SELECTED_THRESHOLD],
        [np.interp(SELECTED_THRESHOLD, THR_GRID, data["mean_f1_test"])],
        color="red",
        zorder=10,
        label=f"thr={SELECTED_THRESHOLD}",
    )
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 score")
    ax.set_title("(e) F1 score vs threshold")
    ax.legend()
    ax.grid(True)

    fpr_thr = safe_div(fp, fp + tn)
    tpr_thr = safe_div(tp, tp + fn)
    matrix = cv["tpr_matrix"]

    ax = axs[1, 1]
    for i in range(matrix.shape[0]):
        ax.plot(cv["fpr_grid"], matrix[i, :], color="lightgray", linewidth=0.8)
    ax.plot(cv["fpr_grid"], np.mean(matrix, axis=0), label=f"mean ROC (AUC={np.mean(cv['aucs']):.3f})")
    ax.fill_between(
        cv["fpr_grid"],
        np.mean(matrix, axis=0) - np.std(matrix, axis=0),
        np.mean(matrix, axis=0) + np.std(matrix, axis=0),
        alpha=0.2,
    )
    ax.plot([0, 1], [0, 1], "--", color="k", linewidth=0.8, label="random guess")
    ax.scatter(
        [fpr_thr],
        [tpr_thr],
        color="red",
        zorder=10,
        label=f"thr={SELECTED_THRESHOLD} (FPR={fpr_thr:.3f}, TPR={tpr_thr:.3f})",
    )
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("(f) ROC curves")
    ax.legend()
    ax.grid(True)

    ax = axs[1, 2]
    ax.plot(data["thr_acc"], data["mean_acc_train"], label="train")
    ax.fill_between(
        data["thr_acc"],
        data["mean_acc_train"] - data["std_acc_train"],
        data["mean_acc_train"] + data["std_acc_train"],
        alpha=0.2,
    )
    ax.plot(data["thr_acc"], data["mean_acc_test"], label="test")
    ax.fill_between(
        data["thr_acc"],
        data["mean_acc_test"] - data["std_acc_test"],
        data["mean_acc_test"] + data["std_acc_test"],
        alpha=0.2,
    )
    ax.axvline(SELECTED_THRESHOLD, color="gray", linestyle="--", linewidth=0.8)
    ax.scatter(
        [SELECTED_THRESHOLD],
        [np.interp(SELECTED_THRESHOLD, data["thr_acc"], data["mean_acc_test"])],
        color="red",
        zorder=10,
        label=f"thr={SELECTED_THRESHOLD}",
    )
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Accuracy")
    ax.set_title("(g) Accuracy vs threshold")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(METRICS_FIG, dpi=200)
    print(f"Saved metrics figure to {METRICS_FIG}")
    plt.show()


def main():
    prepare_output_dirs(OOF_OUT, ROC_OUT, CONF_OUT, METRICS_OUT, METRICS_FIG)
    x, y = load_data(IN_FILE)
    cv = run_final_cv(x, y)
    save_oof(y, cv)
    roc_mean, roc_folds = save_roc(cv)
    save_confusion(y, cv)
    plot_data = make_plot_data(x, y, cv, roc_mean, roc_folds)
    draw_metrics(y, cv, plot_data)


if __name__ == "__main__":
    main()
