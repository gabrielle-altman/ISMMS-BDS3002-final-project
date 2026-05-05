"""
Random Forest classifier for splice site prediction.

Trains a RandomForestClassifier on donor and acceptor splice site data, with
two strategies for handling class imbalance:
    1. No resampling
    2. Undersampling the majority class to match the minority class

For each experiment, prints accuracy / confusion matrix / classification
report / AUC, and saves ROC, precision-recall, and feature-importance plots.

Inputs are the donor/acceptor train/test TSVs produced by preprocess.py.

Usage:
    python random_forest.py --data-dir data/ --plot-dir plots/
"""

import argparse
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample


SEQ_LENGTH = 140
N_ESTIMATORS = 100
RANDOM_STATE = 123

# Each splice-site type: (label_prefix, true_class, false_class)
SITE_TYPES = [
    ("donor", "EI_true", "EI_false"),
    ("acceptor", "IE_true", "IE_false"),
]


def load_split_data(data_dir):
    """Load the four train/test files written by preprocess.py."""
    dtype = {"index": int, "exon_num": str, "intron_num": str}
    return {
        "donor_train": pd.read_table(os.path.join(data_dir, "donors_train.txt"), sep="\t", dtype=dtype),
        "donor_test": pd.read_table(os.path.join(data_dir, "donors_test.txt"), sep="\t", dtype=dtype),
        "acceptor_train": pd.read_table(os.path.join(data_dir, "acceptors_train.txt"), sep="\t", dtype=dtype),
        "acceptor_test": pd.read_table(os.path.join(data_dir, "acceptors_test.txt"), sep="\t", dtype=dtype),
    }


def expand_sequences(df):
    """
    Expand the 'seq' column into one column per nucleotide position, dropping
    metadata columns. Returns a dataframe whose first column is 'classification'
    and whose remaining columns are nucleotide positions 1..140.
    """
    expanded = pd.concat([df, df["seq"].str.split("", expand=True)], axis=1)
    # str.split("") produces empty strings at positions 0 and 141, so drop those
    expanded = expanded.drop(columns=["id", "index", "exon_num", "intron_num", "seq", 0, 141])
    expanded.columns = expanded.columns.astype(str)
    return expanded


def plot_class_distribution(all_data, plot_path):
    """Save a single bar plot showing class counts across all four classes."""
    fig, ax = plt.subplots(1, 1)
    sns.countplot(x=all_data["classification"], ax=ax)
    ax.set_xlabel("Classification")
    ax.set_ylabel("Counts")
    ax.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_xticklabels(["False Donor", "True Donor", "False Acceptor", "True Acceptor"])
    plt.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved class distribution plot to: {plot_path}")


def encode_features(X_train, X_test, y_train, y_test):
    """Label-binarize the target and one-hot encode the per-position bases."""
    lb = preprocessing.LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)

    cats = [["A", "C", "G", "T"]] * SEQ_LENGTH
    enc = OneHotEncoder(categories=cats, handle_unknown="ignore")
    X_train = enc.fit_transform(X_train)
    X_test = enc.transform(X_test)
    return X_train, X_test, y_train, y_test, enc


def downsample_majority(train_df, true_class, false_class):
    """Random undersample the majority (false) class to match the minority (true) class.

    Standard sklearn idiom; see e.g. https://elitedatascience.com/imbalanced-classes
    """
    df_majority = train_df[train_df["classification"] == false_class]
    df_minority = train_df[train_df["classification"] == true_class]
    df_majority_downsampled = resample(
        df_majority,
        replace=False,
        n_samples=df_minority.shape[0],
        random_state=RANDOM_STATE,
    )
    return pd.concat([df_majority_downsampled, df_minority])


def evaluate_and_print(clf, X_test, y_test):
    """Print accuracy, confusion matrix, classification report, and ROC AUC."""
    y_pred = clf.predict(X_test)
    print("ACCURACY:", metrics.accuracy_score(y_test, y_pred))
    print("CONFUSION MATRIX:")
    print(metrics.confusion_matrix(y_test, y_pred))
    print("CLASSIFICATION REPORT:")
    print(metrics.classification_report(y_test, y_pred))
    print("AUC ROC SCORE:", metrics.roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
    print("")


def save_curves(clf, X_test, y_test, plot_dir, tag):
    """Save ROC and precision-recall curves for this experiment."""
    metrics.RocCurveDisplay.from_estimator(clf, X_test, y_test)
    plt.savefig(os.path.join(plot_dir, f"RFC_{tag}_AUC.png"))
    plt.close()

    PrecisionRecallDisplay.from_estimator(clf, X_test, y_test)
    plt.savefig(os.path.join(plot_dir, f"RFC_{tag}_precisionRecall.png"))
    plt.close()


def save_feature_importance(clf, enc, train_features, plot_dir, results_dir, tag):
    """Save per-position feature-importance bar plots and a TSV of values."""
    encoded_features = enc.get_feature_names_out(train_features)
    importances = clf.feature_importances_
    importances_per_position = np.reshape(importances, (SEQ_LENGTH, 4))

    importances_df = pd.DataFrame(importances_per_position, columns=["A", "C", "G", "T"])
    importances_df["base_position"] = importances_df.index + 1
    importances_df.to_csv(
        os.path.join(results_dir, f"RFC_{tag}_featureimportance.txt"),
        index=False,
        sep="\t",
    )

    # Flat bar plot: one bar per (position, base) feature
    plt.figure()
    plt.bar(encoded_features, importances)
    plt.xticks([])
    plt.xlabel("Position")
    plt.ylabel("Feature importance")
    plt.savefig(os.path.join(plot_dir, f"RFC_{tag}_featureimportance.png"))
    plt.close()

    # Stacked bar plot: importance contribution of each base, stacked by position
    plt.figure()
    plt.bar(importances_df["base_position"], importances_df["A"], color="r")
    plt.bar(importances_df["base_position"], importances_df["C"],
            bottom=importances_df["A"], color="b")
    plt.bar(importances_df["base_position"], importances_df["G"],
            bottom=importances_df["A"] + importances_df["C"], color="orange")
    plt.bar(importances_df["base_position"], importances_df["T"],
            bottom=importances_df["A"] + importances_df["C"] + importances_df["G"], color="g")
    plt.xlabel("Position")
    plt.ylabel("Feature importance")
    plt.legend(["A", "C", "G", "T"])
    plt.savefig(os.path.join(plot_dir, f"RFC_{tag}_featureimportance_stacked.png"))
    plt.close()


def run_experiment(train_df, test_df, tag, plot_dir, results_dir):
    """Train an RFC on the given train_df, evaluate against test_df, and save plots."""
    print(f"\n=== {tag} ===")

    X_train_raw = train_df.iloc[:, 1:SEQ_LENGTH + 1]
    y_train = train_df.iloc[:, 0]
    X_test_raw = test_df.iloc[:, 1:SEQ_LENGTH + 1]
    y_test = test_df.iloc[:, 0]

    X_train, X_test, y_train, y_test, enc = encode_features(
        X_train_raw, X_test_raw, y_train, y_test
    )

    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train.ravel())

    evaluate_and_print(clf, X_test, y_test)
    save_curves(clf, X_test, y_test, plot_dir, tag)
    save_feature_importance(
        clf, enc, list(X_train_raw.columns), plot_dir, results_dir, tag
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data",
                        help="Directory containing the train/test TSVs (default: ./data).")
    parser.add_argument("--plot-dir", default="plots",
                        help="Directory to write plots (default: ./plots).")
    parser.add_argument("--results-dir", default="results",
                        help="Directory to write feature-importance TSVs (default: ./results).")
    args = parser.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    splits = load_split_data(args.data_dir)
    expanded = {key: expand_sequences(df) for key, df in splits.items()}

    # Class distribution plot across all four classes
    all_data = pd.concat([
        splits["donor_train"], splits["donor_test"],
        splits["acceptor_train"], splits["acceptor_test"],
    ])
    plot_class_distribution(all_data, os.path.join(args.plot_dir, "distOfClasses.png"))

    # Run all four experiments: {donor, acceptor} x {no resampling, downsampled}
    for site, true_class, false_class in SITE_TYPES:
        train_df = expanded[f"{site}_train"]
        test_df = expanded[f"{site}_test"]

        # No resampling
        run_experiment(train_df, test_df, tag=site, plot_dir=args.plot_dir,
                       results_dir=args.results_dir)

        # Undersampled
        train_downsampled = downsample_majority(train_df, true_class, false_class)
        run_experiment(train_downsampled, test_df, tag=f"{site}_downsampled",
                       plot_dir=args.plot_dir, results_dir=args.results_dir)


if __name__ == "__main__":
    main()