"""
CNN / Bidirectional LSTM hybrid with max pooling for splice site prediction
(Table 5).

Same as cnn_bilstm.py but with a MaxPool1d(2) layer
inserted after each Conv1d:

    Conv1d -> MaxPool1d(2) -> Dropout
    Conv1d -> MaxPool1d(2) -> Dropout
    BiLSTM -> Linear -> ReLU -> Linear -> 2 classes

Pooling halves the sequence length after each convolution, retaining only the
most strongly activated position within each window. After training, the
first convolutional layer's filter weights are exported as CSVs for motif
visualization.

Reuses the training pipeline and helpers from cnn_bilstm
and overrides only the model class.

Default per-site hyperparameters from the report's sweep:
    donor: num_filters=128, kernel_size=9
    acceptor: num_filters=128, kernel_size=7

Usage:
    python cnn_bilstm_maxpool.py <prefix> <lr_init> \\
        <lr_patience> <hidden_size> <es_patience> \\
        [--num-filters-donor N --kernel-size-donor K] \\
        [--num-filters-acceptor N --kernel-size-acceptor K] \\
        [--data-dir DIR] [--plot-dir DIR] [--results-dir DIR]
"""

import argparse
import os

import pandas as pd
import torch
import torch.nn as nn

from cnn_bilstm import CNN_LSTM, run_site
from lstm_unidirectional import (
    ACCEPTOR_CLASS_TO_IDX,
    DONOR_CLASS_TO_IDX,
)


class CNN_LSTM_MaxPool(CNN_LSTM):
    """CNN/BiLSTM variant that adds MaxPool1d(2) after each Conv1d.

    Inherits the BiLSTM and FC head from CNN_LSTM. Only the convolutional
    feature extractor changes: pooling halves the sequence length after each
    conv, so the LSTM sees a shorter, downsampled feature sequence.
    """

    def __init__(self, num_classes, num_filters, kernel_size,
                 hidden_size, num_layers=1, fc_size=128, dropout=0.25,
                 pool_kernel_size=2):
        super().__init__(
            num_classes=num_classes,
            num_filters=num_filters,
            kernel_size=kernel_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            fc_size=fc_size,
            dropout=dropout,
        )
        self.max_pool1 = nn.MaxPool1d(pool_kernel_size, stride=pool_kernel_size)
        self.max_pool2 = nn.MaxPool1d(pool_kernel_size, stride=pool_kernel_size)

    def forward(self, xb):
        # Permute to (batch_size, 4 channels, seq_len) for Conv1d
        xb = xb.permute(0, 2, 1)
        out = self.conv1(xb)
        out = self.max_pool1(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.max_pool2(out)
        out = self.dropout(out)
        # Back to (batch_size, seq_len, channels) for the LSTM
        out = out.permute(0, 2, 1)
        _, (hn, _) = self.lstm(out)
        hid_st = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        out = self.fc_1(hid_st)
        out = self.relu(out)
        out = self.fc(out)
        return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", help="Prefix used when naming output files.")
    parser.add_argument("lr_init", type=float, help="Initial learning rate.")
    parser.add_argument("lr_patience", type=int,
                        help="Patience (epochs) for ReduceLROnPlateau.")
    parser.add_argument("hidden_size", type=int, help="LSTM hidden state size.")
    parser.add_argument("es_patience", type=int,
                        help="Patience (epochs) for early stopping.")

    # Per-site filter configuration: in the report, both donor and acceptor
    # used 128 filters; donor used kernel size 9 and acceptor used 7.
    parser.add_argument("--num-filters-donor", type=int, default=128,
                        help="Number of conv filters for the donor model (default: 128).")
    parser.add_argument("--kernel-size-donor", type=int, default=9,
                        help="Conv kernel size for the donor model (default: 9, odd).")
    parser.add_argument("--num-filters-acceptor", type=int, default=128,
                        help="Number of conv filters for the acceptor model (default: 128).")
    parser.add_argument("--kernel-size-acceptor", type=int, default=7,
                        help="Conv kernel size for the acceptor model (default: 7, odd).")

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
        model_cls=CNN_LSTM_MaxPool,
    )
    run_site(
        "acceptor", acceptor_train, acceptor_test, ACCEPTOR_CLASS_TO_IDX, args, device,
        num_filters=args.num_filters_acceptor,
        kernel_size=args.kernel_size_acceptor,
        model_cls=CNN_LSTM_MaxPool,
    )


if __name__ == "__main__":
    main()