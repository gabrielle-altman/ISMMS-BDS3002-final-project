"""
Standalone 2-layer CNN for splice site prediction with motif visualization.

Architecture (Table 1 in the report):
    Conv1d(4 -> num_filters, kernel_size) -> ReLU -> MaxPool1d(2) -> Dropout
    Conv1d(num_filters -> 32, kernel_size) -> ReLU -> MaxPool1d(2) -> Dropout
    Flatten -> Linear -> 2 classes

The first convolutional layer's filter weights are exported as CSVs after
training. Each CSV holds a (kernel_size x 4) weight matrix that can be
rendered as a position weight matrix / motif logo for downstream motif
discovery (Figures 7 and 8 in the report).

Reuses the dataset, sampler, training loop, and orchestration from
lstm_unidirectional.py — see the imports near the top.

Usage:
    python cnn.py <prefix> <lr_init> <lr_patience> \\
                                       <num_filters> <kernel_size> <es_patience> \\
                                       [--data-dir DIR] [--plot-dir DIR] \\
                                       [--results-dir DIR]
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.model_selection
import torch
import torch.nn as nn
from sklearn import metrics
from torch.optim import lr_scheduler
from torch.utils import data

from lstm_unidirectional import (
    ACCEPTOR_CLASS_TO_IDX,
    BATCH_SIZE,
    DONOR_CLASS_TO_IDX,
    NUM_EPOCHS,
    EarlyStopper,  # noqa: F401  (re-exported for any user inspecting this module)
    make_weighted_sampler,
    model_eval,
    model_train,
    visualize_training_results,
)


# One-hot encoding for the 4 nucleotides plus H (unknown -> all-zero vector)
ONE_HOT = {
    "A": [1.0, 0.0, 0.0, 0.0],
    "C": [0.0, 1.0, 0.0, 0.0],
    "G": [0.0, 0.0, 1.0, 0.0],
    "T": [0.0, 0.0, 0.0, 1.0],
    "H": [0.0, 0.0, 0.0, 0.0],
}


# Adapted from https://towardsdatascience.com/modeling-dna-sequences-with-pytorch-de28b0a05036
def one_hot_encode(seq):
    """Convert a DNA sequence into a (seq_len, 4) one-hot encoded numpy array."""
    allowed = set("ACTGH")
    if not set(seq).issubset(allowed):
        invalid = set(seq) - allowed
        raise ValueError(
            f"Sequence contains chars not in allowed alphabet (ACGTH): {invalid}"
        )
    return np.array([ONE_HOT[x] for x in seq])


class OneHotSpliceDataset(data.Dataset):
    """One-hot encoded sequence dataset for either donor or acceptor sites."""

    def __init__(self, df, class_to_idx, seq_col="seq", target_col="classification"):
        self.seqs = list(df[seq_col].values)
        self.seq_len = len(self.seqs[0])
        self.ohe_seqs = torch.stack(
            [torch.tensor(one_hot_encode(x)) for x in self.seqs]
        )
        labels = [class_to_idx[x] for x in df[target_col].values]
        self.labels = torch.tensor(labels).unsqueeze(1)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.ohe_seqs[idx], self.labels[idx]


class CNN(nn.Module):
    """Two-layer 1D CNN for binary classification of one-hot encoded sequences."""

    def __init__(self, seq_len, num_filters=32, kernel_size=15, num_classes=2):
        super().__init__()
        self.seq_len = seq_len

        # Compute the size of the flattened conv output. Each conv reduces
        # length by (kernel_size - 1); each MaxPool1d(2) halves it (floor).
        len_after_conv1 = seq_len - kernel_size + 1
        len_after_pool1 = len_after_conv1 // 2
        len_after_conv2 = len_after_pool1 - kernel_size + 1
        len_after_pool2 = len_after_conv2 // 2
        flatten_size = num_filters * len_after_pool2

        self.conv_net = nn.Sequential(
            nn.Conv1d(4, num_filters, kernel_size=kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.2),
            nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.2),
            nn.Flatten(),
            nn.Linear(flatten_size, num_classes),
        )

    def forward(self, xb):
        # Permute to (batch_size, 4 channels, seq_len) for Conv1d
        xb = xb.permute(0, 2, 1)
        return self.conv_net(xb)


def export_first_conv_weights(model, stage, results_dir, prefix):
    """
    Walk the model's children, find the first Conv1d layer, and write each
    filter's (kernel_size x 4) weight matrix to a separate CSV. These CSVs
    feed the motif visualizations (Figures 7 and 8 in the report).
    """
    print(f"\nExtracting first-layer Conv1d weights for {stage}...")
    first_conv = None
    for child in model.modules():
        if isinstance(child, nn.Conv1d):
            first_conv = child
            break
    if first_conv is None:
        print("  No Conv1d layer found; nothing to export.")
        return

    # Shape: (num_filters, in_channels=4, kernel_size)
    weights = first_conv.weight.detach().cpu().numpy()
    print(f"  First Conv1d: {first_conv}")
    print(f"  Weight tensor shape: {weights.shape}")

    for i, filter_weights in enumerate(weights):
        # filter_weights: (4, kernel_size) -> transpose to (kernel_size, 4)
        # so each row is a position and each column is a base (A/C/G/T)
        df = pd.DataFrame(filter_weights.T, columns=["A", "C", "G", "T"])
        out_path = os.path.join(
            results_dir, f"{prefix}_CNN_BEST_MODEL_{stage}_weights_ConvFilter{i}.csv"
        )
        df.to_csv(out_path, index=False)
    print(f"  Wrote {len(weights)} filter CSVs to {results_dir}")


def run_site(stage, train_df, test_df, class_to_idx, args, device):
    """Train and evaluate a CNN for one site type, then export filter weights."""
    print(f"\n{'=' * 40}\n  {stage.upper()}\n{'=' * 40}")

    # Train/val split (stratified)
    val_size = len(train_df) - int(0.8 * len(train_df))
    train_split, val_split = sklearn.model_selection.train_test_split(
        train_df,
        stratify=train_df["classification"],
        test_size=val_size,
        shuffle=True,
    )

    train_ds = OneHotSpliceDataset(train_split, class_to_idx)
    val_ds = OneHotSpliceDataset(val_split, class_to_idx)
    test_ds = OneHotSpliceDataset(test_df, class_to_idx)

    sampler = make_weighted_sampler(train_split)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False)

    print(f"Training size: {len(train_ds)}")
    print(f"Validation size: {len(val_ds)}")
    print(f"Test size: {len(test_ds)}")

    # Build model
    seq_len = train_ds.seq_len
    model = CNN(
        seq_len=seq_len,
        num_filters=args.num_filters,
        kernel_size=args.kernel_size,
    ).double()
    print(model)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr_init, momentum=0.9, weight_decay=0.0001
    )
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=args.lr_patience, factor=0.5
    )
    criterion = nn.CrossEntropyLoss()

    # Train (using the shared training loop from lstm_unidirectional)
    best_model_path = os.path.join(
        args.results_dir, f"{args.prefix}_{stage}_best_model.pt"
    )
    train_losses, train_accs, val_losses, val_accs = model_train(
        model, device, train_loader, val_loader, val_ds,
        criterion, optimizer, scheduler, best_model_path,
        NUM_EPOCHS, stage, args.es_patience, args.results_dir, args.prefix,
    )

    # Evaluate on test set (using best checkpoint)
    model.load_state_dict(torch.load(best_model_path))
    _, labels_test = test_ds[:]
    test_loss, test_acc, pred, probs = model_eval(model, test_loader, criterion, device)
    print(f"Test - Loss:{test_loss:.6f}\tAcc:{test_acc:.3f}")

    y_true = labels_test.cpu().detach().numpy()
    y_score = probs.cpu().detach().numpy()
    y_pred = pred.cpu().detach().numpy()

    print("CONFUSION MATRIX:")
    print(metrics.confusion_matrix(y_true, y_pred))
    print("CLASSIFICATION REPORT:")
    print(metrics.classification_report(y_true, y_pred))
    print(f"AUC ROC SCORE: {metrics.roc_auc_score(y_true, y_score)}")

    # Save plots
    metrics.RocCurveDisplay.from_predictions(y_true, y_score)
    plt.savefig(os.path.join(args.plot_dir, f"{args.prefix}_{stage}_AUC.pdf"))
    plt.close()

    metrics.PrecisionRecallDisplay.from_predictions(y_true, y_score)
    plt.savefig(os.path.join(args.plot_dir, f"{args.prefix}_{stage}_precisionRecall.pdf"))
    plt.close()

    visualize_training_results(
        train_losses, val_losses, train_accs, val_accs,
        num_epochs=len(train_losses),
        model_name=os.path.basename(best_model_path),
        batch_size=BATCH_SIZE,
        savepath=os.path.join(args.plot_dir, f"{args.prefix}_{stage}_visualizations_accuracy.pdf"),
    )

    # Export first-layer Conv1d weights for motif visualization (Figs 7/8)
    export_first_conv_weights(model, stage, args.results_dir, args.prefix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", help="Prefix used when naming output files.")
    parser.add_argument("lr_init", type=float, help="Initial learning rate.")
    parser.add_argument("lr_patience", type=int,
                        help="Patience (epochs) for ReduceLROnPlateau.")
    parser.add_argument("num_filters", type=int,
                        help="Number of convolutional filters per layer.")
    parser.add_argument("kernel_size", type=int,
                        help="Convolutional kernel size (use an odd integer).")
    parser.add_argument("es_patience", type=int,
                        help="Patience (epochs) for early stopping.")
    parser.add_argument("--data-dir", default="data",
                        help="Directory with the train/test TSVs (default: ./data).")
    parser.add_argument("--plot-dir", default="plots",
                        help="Directory to write plots (default: ./plots).")
    parser.add_argument("--results-dir", default="results",
                        help="Directory for checkpoints, per-epoch CSVs, "
                             "and filter-weight CSVs (default: ./results).")
    args = parser.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Prefix: {args.prefix}")
    print(f"PyTorch: {torch.__version__}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    donor_train = pd.read_csv(os.path.join(args.data_dir, "donors_train.txt"), sep="\t")
    donor_test = pd.read_csv(os.path.join(args.data_dir, "donors_test.txt"), sep="\t")
    acceptor_train = pd.read_csv(os.path.join(args.data_dir, "acceptors_train.txt"), sep="\t")
    acceptor_test = pd.read_csv(os.path.join(args.data_dir, "acceptors_test.txt"), sep="\t")

    run_site("donor", donor_train, donor_test, DONOR_CLASS_TO_IDX, args, device)
    run_site("acceptor", acceptor_train, acceptor_test, ACCEPTOR_CLASS_TO_IDX, args, device)


if __name__ == "__main__":
    main()