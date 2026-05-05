"""
CNN / Bidirectional LSTM hybrid for splice site prediction (Table 4).

Architecture:
    Conv1d(4 -> num_filters, kernel_size, padding) -> Dropout
    Conv1d(num_filters -> num_filters, kernel_size, padding) -> Dropout
    BiLSTM(input_size=num_filters, hidden_size) -> concat both directions
    Linear(hidden_size * 2 -> fc_size) -> ReLU -> Linear(fc_size -> 2)

The CNN extracts local sequence features, the BiLSTM captures positional
context across the sequence, and the FC head classifies. After training,
the first convolutional layer's filter weights are exported as CSVs for
motif visualization (Figures 7 and 8 in the report).

Reuses dataset, sampler, training loop, and helpers from lstm_unidirectional.py and
the one-hot encoded dataset / weight export from cnn.py.

Hyperparameters can be set per site (donor vs acceptor) since the best
sweep settings differed: in the report, donor used num_filters=128 and
acceptor used num_filters=64.

Usage:
    python cnn_bilstm.py <prefix> <lr_init> \\
        <lr_patience> <hidden_size> <es_patience> \\
        --num-filters-donor N --kernel-size-donor K \\
        --num-filters-acceptor N --kernel-size-acceptor K \\
        [--data-dir DIR] [--plot-dir DIR] [--results-dir DIR]
"""

import argparse
import math
import os

import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection
import torch
import torch.nn as nn
from sklearn import metrics
from torch.optim import lr_scheduler

from cnn import OneHotSpliceDataset, export_first_conv_weights
from lstm_unidirectional import (
    ACCEPTOR_CLASS_TO_IDX,
    BATCH_SIZE,
    DONOR_CLASS_TO_IDX,
    NUM_EPOCHS,
    make_weighted_sampler,
    model_eval,
    model_train,
    visualize_training_results,
)


class CNN_LSTM(nn.Module):
    """Two convolutional layers followed by a bidirectional LSTM head."""

    def __init__(self, num_classes, num_filters, kernel_size,
                 hidden_size, num_layers=1, fc_size=128, dropout=0.25):
        super().__init__()
        # `same`-style padding so the convolutions don't shrink the sequence,
        # which keeps the LSTM input length tied to the original seq_len.
        padding = math.floor(kernel_size / 2)
        self.conv1 = nn.Conv1d(4, num_filters, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size, padding=padding)
        self.lstm = nn.LSTM(
            input_size=num_filters,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc_1 = nn.Linear(hidden_size * 2, fc_size)
        self.fc = nn.Linear(fc_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, xb):
        # Permute to (batch_size, 4 channels, seq_len) for Conv1d
        xb = xb.permute(0, 2, 1)
        out = self.conv1(xb)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.dropout(out)
        # Back to (batch_size, seq_len, channels=num_filters) for the LSTM
        out = out.permute(0, 2, 1)
        _, (hn, _) = self.lstm(out)
        # Concatenate the final hidden states from both directions
        hid_st = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        out = self.fc_1(hid_st)
        out = self.relu(out)
        out = self.fc(out)
        return out


def run_site(stage, train_df, test_df, class_to_idx, args, device,
             num_filters, kernel_size, model_cls=None):
    """Train and evaluate a CNN/BiLSTM for one site type, then export filter weights."""
    if model_cls is None:
        model_cls = CNN_LSTM
    print(f"\n{'=' * 40}\n  {stage.upper()}"
          f"  (num_filters={num_filters}, kernel_size={kernel_size})"
          f"\n{'=' * 40}")

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
    model = model_cls(
        num_classes=2,
        num_filters=num_filters,
        kernel_size=kernel_size,
        hidden_size=args.hidden_size,
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
    parser.add_argument("hidden_size", type=int, help="LSTM hidden state size.")
    parser.add_argument("es_patience", type=int,
                        help="Patience (epochs) for early stopping.")

    # Per-site filter configuration: in the report, donor used 128 filters
    # and acceptor used 64 filters from the hyperparameter sweep.
    parser.add_argument("--num-filters-donor", type=int, default=128,
                        help="Number of conv filters for the donor model (default: 128).")
    parser.add_argument("--kernel-size-donor", type=int, default=9,
                        help="Conv kernel size for the donor model (default: 9, odd).")
    parser.add_argument("--num-filters-acceptor", type=int, default=64,
                        help="Number of conv filters for the acceptor model (default: 64).")
    parser.add_argument("--kernel-size-acceptor", type=int, default=9,
                        help="Conv kernel size for the acceptor model (default: 9, odd).")

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

    run_site(
        "donor", donor_train, donor_test, DONOR_CLASS_TO_IDX, args, device,
        num_filters=args.num_filters_donor,
        kernel_size=args.kernel_size_donor,
    )
    run_site(
        "acceptor", acceptor_train, acceptor_test, ACCEPTOR_CLASS_TO_IDX, args, device,
        num_filters=args.num_filters_acceptor,
        kernel_size=args.kernel_size_acceptor,
    )


if __name__ == "__main__":
    main()