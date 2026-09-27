import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import json
import os
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.curdir).split('respirazione')[0] + 'respirazione')
from config.definitions import ROOT_DIR
from datetime import datetime

from src.splits import make_splits
from src.threshold_estimator import ThresholdEstimator
from src.timeseriesdataset import TimeSeriesDataset
from src.training import (PAD_LABEL, build_optimizer, get_device, make_dataloader,
                          training_loop, validation_loop)

torch.manual_seed(42)


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

    columns = list(cfg.features.columns)
    num_epochs = int(cfg.epochs_final)
    hidden_size = int(cfg.model.hidden_size)
    num_layers = int(cfg.model.num_layers)
    dataset_path = Path(cfg.data_dir) / cfg.dataset.path
    total_dataset = TimeSeriesDataset(dataset_path, columns=columns)

    folds, dev_index, test_index = make_splits(len(total_dataset),
                                               n_folds=cfg.split.n_folds,
                                               test_size=cfg.split.test_size,
                                               seed=cfg.split.seed)
    # L'addestramento finale usa il primo fold: si allena sul suo train e ferma
    # l'early stopping sulla sua validation. Il test set entra in gioco una volta
    # sola, alla fine, e non influenza nessuna scelta.
    train_index, val_index = folds[0]
    print(f"train: {len(train_index)} soggetti | validation: {len(val_index)} | test: {len(test_index)}")

    train_dataloader = make_dataloader(dataset_path, train_index, columns, cfg.batch_size, True)
    val_dataloader = make_dataloader(dataset_path, val_index, columns, cfg.batch_size, False)
    test_dataloader = make_dataloader(dataset_path, test_index, columns, cfg.batch_size, False)

    model = ThresholdEstimator(len(total_dataset.features), hidden_size, num_layers,
                               dropout=cfg.model.dropout).to(device)
    loss_function = nn.CrossEntropyLoss(ignore_index=PAD_LABEL)
    optimizer = build_optimizer(cfg, model)

    # Configurazione dell'early stopping
    patience = cfg.patience
    min_delta = 0.001
    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = -1
    early_stop_counter = 0

    for e in tqdm(range(num_epochs)):
        tr_loss, tr_f1, tr_precision, tr_recall = training_loop(
            model, train_dataloader, loss_function, optimizer, e, device=device, writer=writer)
        val_loss, val_f1, val_precision, val_recall = validation_loop(
            model, val_dataloader, loss_function, e, device=device, writer=writer)

        if best_val_loss - val_loss > min_delta:
            best_val_loss = val_loss
            # deepcopy: state_dict() restituisce riferimenti ai tensori del modello,
            # che le epoche successive sovrascriverebbero in place.
            best_model_state = deepcopy(model.state_dict())
            best_epoch = e
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

    # Misura finale sul test set, mai usato per scegliere nulla.
    test_loss, test_f1, test_precision, test_recall = validation_loop(
        model, test_dataloader, loss_function, best_epoch, device=device, writer=writer, split='test')
    print(f"\nTest set ({len(test_index)} soggetti, epoca {best_epoch}): "
          f"F1 = {test_f1:.4f} | precision = {test_precision:.4f} | "
          f"recall = {test_recall:.4f} | loss = {test_loss:.4f}")

    writer.add_hparams(hparams_for_tensorboard(cfg),
                       {'tr_loss': tr_loss, 'val_loss': val_loss, 'test_loss': test_loss,
                        'tr_f1': tr_f1, 'val_f1': val_f1, 'test_f1': test_f1,
                        'tr_precision': tr_precision, 'val_precision': val_precision,
                        'test_precision': test_precision,
                        'tr_recall': tr_recall, 'val_recall': val_recall,
                        'test_recall': test_recall})
    writer.close()

    results = {
        'features': columns,
        'model': OmegaConf.to_container(cfg.model),
        'optimizer': OmegaConf.to_container(cfg.optimizer),
        'batch_size': cfg.batch_size,
        'best_epoch': best_epoch,
        'validation': {'loss': val_loss, 'f1': val_f1,
                       'precision': val_precision, 'recall': val_recall},
        'test': {'loss': test_loss, 'f1': test_f1,
                 'precision': test_precision, 'recall': test_recall,
                 'n_subjects': len(test_index)},
    }
    with open(Path(ROOT_DIR) / 'src/scripts/test_results.json', 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    main()
