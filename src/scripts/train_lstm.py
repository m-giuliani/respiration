import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score
import os
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import sys
sys.path.append(os.path.abspath(os.curdir).split('respirazione')[0] + 'respirazione')
from config.definitions import ROOT_DIR
from datetime import datetime

from src.threshold_estimator import ThresholdEstimator
from src.timeseriesdataset import TimeSeriesDataset, collate_fn

torch.manual_seed(42)

NUM_CLASSES = 3
PAD_LABEL = -1


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def macro_scores(all_labels, all_preds):
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return precision, recall, f1


def log_epoch(writer, split, epoch, loss, precision, recall, f1):
    # I file che riusano questi loop (tuning, feature selection) passano writer=None:
    # senza questo, tutti i trial scriverebbero sugli stessi tag della stessa run.
    if writer is None:
        return
    writer.add_scalar(f'Loss/{split}', loss, epoch)
    writer.add_scalar(f'Precision/{split}', precision, epoch)
    writer.add_scalar(f'Recall/{split}', recall, epoch)
    writer.add_scalar(f'F1/{split}', f1, epoch)


def training_loop(model, dataloader, loss_function, optimizer, epoch, device="cpu", writer=None):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    for data, lengths, targets in tqdm(dataloader):
        data = data.to(device)
        targets = targets.to(device)
        # pack_padded_sequence vuole le lunghezze sulla CPU
        lengths = lengths.cpu()

        optimizer.zero_grad()
        preds = model(data, lengths).view(-1, NUM_CLASSES)
        targets = targets.view(-1)
        loss = loss_function(preds, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        mask = (targets != PAD_LABEL)
        all_labels.extend(targets[mask].cpu().numpy())
        all_preds.extend(preds.argmax(dim=1)[mask].cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    precision, recall, f1 = macro_scores(all_labels, all_preds)
    log_epoch(writer, 'train', epoch, epoch_loss, precision, recall, f1)
    return epoch_loss, f1, precision, recall


def validation_loop(model, dataloader, loss_function, epoch, device="cpu", writer=None):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for data, lengths, targets in tqdm(dataloader):
            data = data.to(device)
            targets = targets.to(device)
            lengths = lengths.cpu()

            preds = model(data, lengths).view(-1, NUM_CLASSES)
            targets = targets.view(-1)
            loss = loss_function(preds, targets)

            running_loss += loss.item()

            mask = (targets != PAD_LABEL)
            all_labels.extend(targets[mask].cpu().numpy())
            all_preds.extend(preds.argmax(dim=1)[mask].cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    precision, recall, f1 = macro_scores(all_labels, all_preds)
    log_epoch(writer, 'valid', epoch, epoch_loss, precision, recall, f1)
    return epoch_loss, f1, precision, recall


def build_optimizer(cfg, model):
    if cfg.optimizer.name == 'adam':
        return optim.Adam(model.parameters(),
                          lr=cfg.optimizer.lr,
                          betas=tuple(cfg.optimizer.betas),
                          weight_decay=cfg.optimizer.weight_decay)
    if cfg.optimizer.name == 'sgd':
        return optim.SGD(model.parameters(),
                         lr=cfg.optimizer.lr,
                         momentum=cfg.optimizer.momentum,
                         weight_decay=cfg.optimizer.weight_decay,
                         nesterov=cfg.optimizer.nesterov)
    raise ValueError(f"Optimizer non riconosciuto: {cfg.optimizer.name}")


def hparams_for_tensorboard(cfg):
    # add_hparams accetta solo scalari: i sottogruppi vanno appiattiti e prefissati,
    # altrimenti model.name e optimizer.name si sovrascrivono a vicenda.
    def scalars(d):
        return {k: v for (k, v) in d.items()
                if isinstance(v, (int, float, str, bool, torch.Tensor))}

    return {**scalars(cfg),
            **{f'model/{k}': v for (k, v) in scalars(cfg.model).items()},
            **{f'optimizer/{k}': v for (k, v) in scalars(cfg.optimizer).items()}}


@hydra.main(config_path="hyperparams", config_name="config", version_base='1.3')
def main(cfg: DictConfig):

    print("\n\n\n", cfg, "\n\n\n")

    device = get_device()
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    writer = SummaryWriter(f'runs/lstm_experiments_h/{current_time}')

    columns = cfg.features.columns
    num_epochs = int(cfg.epochs_final)
    hidden_size = int(cfg.model.hidden_size)
    num_layers = int(cfg.model.num_layers)
    dataset_path = Path(cfg.data_dir) / cfg.dataset.path
    total_dataset = TimeSeriesDataset(dataset_path, columns=columns)

    model = ThresholdEstimator(len(total_dataset.features), hidden_size, num_layers,
                               dropout=cfg.model.dropout).to(device)
    loss_function = nn.CrossEntropyLoss(ignore_index=PAD_LABEL)
    optimizer = build_optimizer(cfg, model)

    index_tr, index_te = train_test_split(range(len(total_dataset)), random_state=42)
    train_dataset = TimeSeriesDataset(dataset_path, index_tr, columns=columns)
    test_dataset = TimeSeriesDataset(dataset_path, index_te, columns=columns)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    # Configurazione dell'early stopping
    patience = 5
    min_delta = 0.001
    best_val_loss = float("inf")
    best_model_state = None
    early_stop_counter = 0

    for e in tqdm(range(num_epochs)):
        tr_loss, tr_f1, tr_precision, tr_recall = training_loop(
            model, train_dataloader, loss_function, optimizer, e, device=device, writer=writer)
        val_loss, val_f1, val_precision, val_recall = validation_loop(
            model, test_dataloader, loss_function, e, device=device, writer=writer)

        if best_val_loss - val_loss > min_delta:
            best_val_loss = val_loss
            # deepcopy: state_dict() restituisce riferimenti ai tensori del modello,
            # che le epoche successive sovrascriverebbero in place.
            best_model_state = deepcopy(model.state_dict())
            early_stop_counter = 0
            print(f"Epoch {e}: Validation loss improved to {val_loss:.4f}. Counter reset.")
        else:
            early_stop_counter += 1
            print(f"Epoch {e}: No improvement in validation loss. Counter: {early_stop_counter}/{patience}")

        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {e}. Best validation loss: {best_val_loss:.4f}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    writer.add_hparams(hparams_for_tensorboard(cfg),
                       {'tr_loss': tr_loss, 'val_loss': val_loss,
                        'tr_f1': tr_f1, 'val_f1': val_f1,
                        'tr_precision': tr_precision, 'val_precision': val_precision,
                        'tr_recall': tr_recall, 'val_recall': val_recall})
    writer.close()


if __name__ == '__main__':
    main()
