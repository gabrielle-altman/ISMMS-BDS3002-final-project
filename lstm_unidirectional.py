"""
Unidirectional LSTM for splice site prediction.

Trains separate LSTM models on donor and acceptor splice site data, using a
learnable embedding layer over the 5-base alphabet (A, C, G, T, H). Handles
class imbalance via WeightedRandomSampler; selects the best epoch by
F1-macro on the validation set; uses early stopping and ReduceLROnPlateau.

For each site, prints test confusion matrix / classification report / AUC
and saves ROC, precision-recall, and training-history plots.

Inputs are the donor/acceptor train/test TSVs produced by preprocess.py.

Usage:
    python lstm_unidirectional.py <prefix> <lr_init> <lr_patience> <embed_size> \\
                         <hidden_size> <es_patience> \\
                         [--data-dir DIR] [--plot-dir DIR] [--results-dir DIR]
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
from torch.utils.data import WeightedRandomSampler


# Class-to-index dictionaries for donor and acceptor sites
DONOR_CLASS_TO_IDX = {"EI_false": 0, "EI_true": 1}
ACCEPTOR_CLASS_TO_IDX = {"IE_false": 0, "IE_true": 1}

# Base-to-index encoding (H is the unknown base)
BASE_TO_IDX = {"A": 1, "C": 2, "G": 3, "T": 4, "H": 0}
VOCAB_SIZE = 5  # ACGT + H

# Training defaults
BATCH_SIZE = 64
NUM_EPOCHS = 50
VAL_FRACTION = 0.2  # of the training set


def encode(seq):
    """Convert a DNA sequence into a numpy array of base indices (1..4 for ACGT, 0 for H)."""
    allowed = set("ACTGH")
    if not set(seq).issubset(allowed):
        invalid = set(seq) - allowed
        raise ValueError(
            f"Sequence contains chars not in allowed alphabet (ACGTH): {invalid}"
        )
    return np.array([BASE_TO_IDX[x] for x in seq])


class SpliceDataset(data.Dataset):
    """Encoded sequence dataset for either donor or acceptor sites."""

    def __init__(self, df, class_to_idx, seq_col="seq", target_col="classification"):
        self.seqs = list(df[seq_col].values)
        self.seq_len = len(self.seqs[0])

        # Encode all sequences once at construction time
        self.encoded_seqs = torch.stack(
            [torch.tensor(encode(x)) for x in self.seqs]
        )

        labels = [class_to_idx[x] for x in df[target_col].values]
        self.labels = torch.tensor(labels).unsqueeze(1)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.encoded_seqs[idx], self.labels[idx]


class LSTM(nn.Module):
    """Unidirectional LSTM with learned embeddings and two fully connected layers."""

    def __init__(self, num_classes, input_size, hidden_size, num_layers,
                 vocab_size=VOCAB_SIZE, fc_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.embeddings = nn.Embedding(vocab_size, input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc_1 = nn.Linear(hidden_size, fc_size)
        self.fc = nn.Linear(fc_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        embeds = self.embeddings(x)
        _, (hn, _) = self.lstm(embeds)
        # Use the last hidden state as input to the FC head
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out


class EarlyStopper:
    """Stop training once validation loss has failed to improve for `patience` epochs.

    Adapted from a common StackOverflow pattern for early stopping in PyTorch.
    """

    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def make_weighted_sampler(train_df):
    """Build a WeightedRandomSampler that oversamples the minority class."""
    class_counts = train_df.classification.value_counts()
    sample_weights = [1 / class_counts[c] for c in train_df.classification.values]
    weights_tensor = torch.FloatTensor(sample_weights)
    return WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=len(train_df),
        replacement=True,
    )


def model_eval(model, loader, criterion, device):
    """Evaluate model on a DataLoader; return loss, accuracy, predictions, P(class=1)."""
    model.eval()
    eval_loss = 0
    correct = 0
    predictions = torch.tensor([], device=device)
    probabilities = torch.tensor([], device=device)
    with torch.no_grad():
        for batch_data, target in loader:
            batch_data = batch_data.to(device)
            target = target.to(device)
            output = model(batch_data)
            eval_loss += criterion(output, target.squeeze()).item()
            pred = output.argmax(dim=1, keepdim=True)
            predictions = torch.cat((predictions, pred), 0)
            probability = nn.functional.softmax(output, dim=1)[:, 1]
            probabilities = torch.cat((probabilities, probability), 0)
            correct += pred.eq(target.view_as(pred)).sum().item()
    eval_loss /= len(loader)
    eval_acc = correct / len(loader.dataset)
    return eval_loss, eval_acc, predictions, probabilities


def model_train(model, device, train_loader, val_loader, val_ds, criterion,
                optimizer, scheduler, best_model_path, epochs, stage,
                es_patience, results_dir, prefix):
    """
    Train the model, saving the best-by-F1-macro checkpoint and per-epoch history.
    Returns lists of train/val losses and accuracies for plotting.
    """
    model = model.to(device)
    best_acc = 0
    best_F1_true = 0
    best_F1_macro = 0
    best_epoch = 0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    val_F1_true_scores, val_F1_macro_scores = [], []
    early_stopper = EarlyStopper(patience=es_patience, min_delta=0.005)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        for batch_data, target in train_loader:
            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            target = target.to(device)
            output = model(batch_data)
            loss = criterion(output, target.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        train_loss /= len(train_loader)
        train_acc = correct / len(train_loader.dataset)

        _, labels_val = val_ds[:]
        val_loss, val_acc, val_pred, _ = model_eval(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        y_true = labels_val.cpu().detach().numpy()
        y_pred = val_pred.cpu().detach().numpy()
        val_F1_true = metrics.f1_score(y_true, y_pred, pos_label=1, average="binary")
        val_F1_macro = metrics.f1_score(y_true, y_pred, average="macro")
        val_F1_true_scores.append(val_F1_true)
        val_F1_macro_scores.append(val_F1_macro)

        curr_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Train Epoch: {epoch}\tLoss: {train_loss:.6f}\tAcc: {train_acc:.3f}"
            f"\tVal - Loss:{val_loss:.6f}\tAcc:{val_acc:.3f}"
            f"\tF1 pos class:{val_F1_true:.3f}\tF1 macro:{val_F1_macro:.3f}"
            f"\tLR:{curr_lr:.8f}"
        )

        # Save per-epoch history every epoch so partial runs leave a record
        results = {
            "train_losses": train_losses, "train_accuracys": train_accs,
            "val_losses": val_losses, "val_accuracys": val_accs,
            "val_F1_posclass": val_F1_true_scores,
            "val_F1_macro": val_F1_macro_scores,
        }
        results_path = os.path.join(results_dir, f"{prefix}_{stage}_epochs.csv")
        pd.DataFrame(results).to_csv(results_path, index=True)

        scheduler.step(val_loss)
        if val_F1_macro > best_F1_macro:
            torch.save(model.state_dict(), best_model_path)
            best_acc = val_acc
            best_epoch = epoch
            best_F1_true = val_F1_true
            best_F1_macro = val_F1_macro

        if early_stopper.early_stop(val_loss):
            print("stopping epochs early")
            break

    print(
        f"Best Epoch: {best_epoch}\tBest Accuracy: {best_acc:.3f}"
        f"\tBest F1 pos class: {best_F1_true:.3f}\tBest F1 Macro: {best_F1_macro:.3f}"
    )
    return train_losses, train_accs, val_losses, val_accs


def visualize_training_results(train_loss, val_loss, train_acc, val_acc,
                               num_epochs, model_name, batch_size, savepath):
    """Plot train/val loss and accuracy curves side by side."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"{model_name} training | Batch size: {batch_size}", fontsize=16)
    epochs_range = list(range(1, num_epochs + 1))
    axs[0].plot(epochs_range, train_loss, label="train_loss")
    axs[0].plot(epochs_range, val_loss, label="val_loss")
    axs[0].legend(loc="best")
    axs[0].set(xlabel="epochs", ylabel="loss")
    axs[1].plot(epochs_range, train_acc, label="train_acc")
    axs[1].plot(epochs_range, val_acc, label="val_acc")
    axs[1].legend(loc="best")
    axs[1].set(xlabel="epochs", ylabel="accuracy")
    plt.savefig(savepath)
    plt.close(fig)


def run_site(stage, train_df, test_df, class_to_idx, args, device, model_cls=None):
    """Train and evaluate a model for one site type (donor or acceptor).

    Returns a dict of test-set metrics (acc, F1 pos class, F1 macro, AUC) so
    callers can log results across hyperparameter sweeps.
    """
    if model_cls is None:
        model_cls = LSTM
    print(f"\n{'=' * 40}\n  {stage.upper()}\n{'=' * 40}")

    # Train/val split (stratified)
    val_size = len(train_df) - int(0.8 * len(train_df))
    train_split, val_split = sklearn.model_selection.train_test_split(
        train_df,
        stratify=train_df["classification"],
        test_size=val_size,
        shuffle=True,
    )

    train_ds = SpliceDataset(train_split, class_to_idx)
    val_ds = SpliceDataset(val_split, class_to_idx)
    test_ds = SpliceDataset(test_df, class_to_idx)

    sampler = make_weighted_sampler(train_split)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False)

    print(f"Training size: {len(train_ds)}")
    print(f"Validation size: {len(val_ds)}")
    print(f"Test size: {len(test_ds)}")

    # Build model, optimizer, scheduler
    num_classes = len(train_df.classification.unique())
    model = model_cls(
        num_classes=num_classes,
        input_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=1,
    ).double()
    print(model)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr_init, momentum=0.9, weight_decay=0.0001
    )
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=args.lr_patience, factor=0.5
    )
    criterion = nn.CrossEntropyLoss()

    # Train
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
    test_auc = metrics.roc_auc_score(y_true, y_score)
    print(f"AUC ROC SCORE: {test_auc}")

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

    return {
        "test_acc": test_acc,
        "test_F1_pos": metrics.f1_score(y_true, y_pred, pos_label=1, average="binary"),
        "test_F1_macro": metrics.f1_score(y_true, y_pred, average="macro"),
        "test_AUC": test_auc,
    }


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
    args = parser.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Prefix: {args.prefix}")
    print(f"PyTorch: {torch.__version__}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    donor_train = pd.read_csv(os.path.join(args.data_dir, "donors_train.txt"), sep="\t")
    donor_test = pd.read_csv(os.path.join(args.data_dir, "donors_test.txt"), sep="\t")
    acceptor_train = pd.read_csv(os.path.join(args.data_dir, "acceptors_train.txt"), sep="\t")
    acceptor_test = pd.read_csv(os.path.join(args.data_dir, "acceptors_test.txt"), sep="\t")

    run_site("donor", donor_train, donor_test, DONOR_CLASS_TO_IDX, args, device)
    run_site("acceptor", acceptor_train, acceptor_test, ACCEPTOR_CLASS_TO_IDX, args, device)


if __name__ == "__main__":
    main()