"""
Bidirectional LSTM for splice site prediction.

Same architecture as lstm_unidirectional.py but with `bidirectional=True` on the LSTM
layer. The forward pass concatenates the final hidden states from both
directions before the FC head, so the first FC layer's input size is
`hidden_size * 2`.

Reuses the dataset, training loop, evaluation, and orchestration from
lstm_unidirectional.py and overrides only the model class. Optionally appends a
hyperparameter-search row (lr_init, embed_size, hidden_size, test_acc,
F1 pos, F1 macro, AUC) to a TSV per site for later sweep analysis.

Usage:
    python lstm_bidirectional.py <prefix> <lr_init> <lr_patience> \\
                                 <embed_size> <hidden_size> <es_patience> \\
                                 [--data-dir DIR] [--plot-dir DIR] \\
                                 [--results-dir DIR] [--paramsearch-log DIR]
"""

import argparse
import os

import pandas as pd
import torch
import torch.nn as nn

from lstm_unidirectional import (
    ACCEPTOR_CLASS_TO_IDX,
    DONOR_CLASS_TO_IDX,
    LSTM,
    VOCAB_SIZE,
    run_site,
)


class BidirectionalLSTM(LSTM):
    """Bidirectional LSTM variant.

    Inherits from the unidirectional LSTM and overrides only the LSTM layer,
    the first FC layer's input size, and the forward pass (which concatenates
    the final hidden states from both directions).
    
    The forward-pass concat pattern `torch.cat((hn[-2], hn[-1]), dim=1)` to
    extract both directions' final hidden states is a common PyTorch idiom;
    see e.g. Gal Hever's BiLSTM tutorial.
    """

    def __init__(self, num_classes, input_size, hidden_size, num_layers,
                 vocab_size=VOCAB_SIZE, fc_size=128):
        super().__init__(
            num_classes=num_classes,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            vocab_size=vocab_size,
            fc_size=fc_size,
        )
        # Override the LSTM with a bidirectional one and resize the first FC layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc_1 = nn.Linear(hidden_size * 2, fc_size)

    def forward(self, x):
        embeds = self.embeddings(x)
        _, (hn, _) = self.lstm(embeds)
        # Concatenate the final hidden states from both directions
        hid_st = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        out = self.fc_1(hid_st)
        out = self.relu(out)
        out = self.fc(out)
        return out


def append_paramsearch_row(log_dir, stage, args, results):
    """Append a single TSV row of hyperparameters and test metrics for sweep analysis."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"LSTM_bidirectional_paramsearch.{stage}s.txt")
    row = "\t".join([
        str(args.lr_init),
        str(args.embed_size),
        str(args.hidden_size),
        f"{results['test_acc']:.3f}",
        f"{results['test_F1_pos']:.2f}",
        f"{results['test_F1_macro']:.2f}",
        f"{results['test_AUC']:.3f}",
    ])
    with open(log_path, "a") as f:
        f.write("\n" + row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", help="Prefix used when naming output files.")
    parser.add_argument("lr_init", type=float, help="Initial learning rate.")
    parser.add_argument("lr_patience", type=int,
                        help="Patience (epochs) for ReduceLROnPlateau.")
    parser.add_argument("embed_size", type=int, help="Embedding dimension.")
    parser.add_argument("hidden_size", type=int, help="LSTM hidden state size.")
    parser.add_argument("es_patience", type=int,
                        help="Patience (epochs) for early stopping.")
    parser.add_argument("--data-dir", default="data",
                        help="Directory with the train/test TSVs (default: ./data).")
    parser.add_argument("--plot-dir", default="plots",
                        help="Directory to write plots (default: ./plots).")
    parser.add_argument("--results-dir", default="results",
                        help="Directory for checkpoints and per-epoch CSVs (default: ./results).")
    parser.add_argument("--paramsearch-log", default=None,
                        help="If set, append test metrics to a TSV in this directory "
                             "for hyperparameter sweep tracking.")
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

    donor_results = run_site(
        "donor", donor_train, donor_test, DONOR_CLASS_TO_IDX,
        args, device, model_cls=BidirectionalLSTM,
    )
    acceptor_results = run_site(
        "acceptor", acceptor_train, acceptor_test, ACCEPTOR_CLASS_TO_IDX,
        args, device, model_cls=BidirectionalLSTM,
    )

    if args.paramsearch_log:
        append_paramsearch_row(args.paramsearch_log, "donor", args, donor_results)
        append_paramsearch_row(args.paramsearch_log, "acceptor", args, acceptor_results)


if __name__ == "__main__":
    main()