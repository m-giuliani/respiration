"""Loop di addestramento e valutazione, condivisi da training, tuning e feature selection."""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from respiration.data.dataset import TimeSeriesDataset, collate_fn

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
    # Tuning e feature selection passano writer=None: senza questo tutti i trial
    # scriverebbero sugli stessi tag della stessa run.
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

    for data, lengths, targets in tqdm(dataloader, leave=False):
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


def collect_predictions(model, dataloader, loss_function, device="cpu"):
    """Valuta il modello restituendo etichette e predizioni, oltre alla loss.

    Serve sia ai loop di validazione sia alla misura finale sul test set, che
    ha bisogno delle predizioni grezze per le metriche per classe.
    """
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for data, lengths, targets in tqdm(dataloader, leave=False):
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

    return running_loss / len(dataloader), all_labels, all_preds


def validation_loop(model, dataloader, loss_function, epoch, device="cpu", writer=None, split='valid'):
    epoch_loss, all_labels, all_preds = collect_predictions(model, dataloader, loss_function, device)
    precision, recall, f1 = macro_scores(all_labels, all_preds)
    log_epoch(writer, split, epoch, epoch_loss, precision, recall, f1)
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


class LengthBucketBatchSampler(Sampler):
    """Mette nello stesso batch le sequenze di lunghezza simile.

    I test durano da 80 a 1090 passi. Con i batch formati a caso ognuno viene
    paddato alla sequenza piu' lunga che ci capita dentro, quindi buona parte
    del calcolo finisce sul padding, che la loss poi ignora. Ordinando per
    lunghezza il padding dentro ogni batch e' minimo; la casualita' resta
    nell'ordine in cui i batch vengono presentati.
    """

    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        order = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        batches = [order[i:i + self.batch_size]
                   for i in range(0, len(order), self.batch_size)]
        if self.shuffle:
            batches = [batches[i] for i in torch.randperm(len(batches)).tolist()]
        return iter(batches)

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def make_dataloader(dataset_path, index, columns, batch_size, shuffle, bucket=True):
    dataset = TimeSeriesDataset(dataset_path, index, columns=columns)
    if not bucket:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    sampler = LengthBucketBatchSampler(dataset.sequence_lengths(), batch_size, shuffle=shuffle)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)


def cross_validate(dataset_path, columns, folds, build_model, make_optimizer,
                   epochs, batch_size, device, on_fold_end=None):
    """Addestra e valuta una configurazione su ogni fold del dev set.

    Restituisce la lista degli F1 migliori, uno per fold. on_fold_end viene
    chiamata dopo ogni fold con (indice_fold, punteggi_finora): serve a Optuna
    per potare un trial senza pagare tutti e cinque i fold.
    """
    loss_function = nn.CrossEntropyLoss(ignore_index=PAD_LABEL)
    fold_scores = []

    for fold, (train_index, val_index) in enumerate(folds):
        # stesso seme per ogni configurazione sullo stesso fold: i confronti
        # non devono dipendere dall'inizializzazione casuale dei pesi
        torch.manual_seed(42 + fold)

        train_dataloader = make_dataloader(dataset_path, train_index, columns, batch_size, True)
        val_dataloader = make_dataloader(dataset_path, val_index, columns, batch_size, False)

        model = build_model().to(device)
        optimizer = make_optimizer(model)

        best_val_f1 = 0.0
        for epoch in range(epochs):
            training_loop(model, train_dataloader, loss_function, optimizer, epoch, device=device)
            _, val_f1, _, _ = validation_loop(model, val_dataloader, loss_function, epoch, device=device)
            best_val_f1 = max(best_val_f1, val_f1)

        fold_scores.append(best_val_f1)
        if on_fold_end is not None:
            on_fold_end(fold, fold_scores)

    return fold_scores
