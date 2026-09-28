import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import json
import logging
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from respiration.paths import CHECKPOINT_DIR, RESULTS_DIR
from respiration.splits import make_splits
from respiration.models.threshold_estimator import ThresholdEstimator
from respiration.data.dataset import TimeSeriesDataset
from respiration.training import (PAD_LABEL, build_optimizer, collect_predictions,
                                  feature_stats, get_device, macro_scores, make_dataloader,
                                  training_loop, validation_loop)

torch.manual_seed(42)

log = logging.getLogger(__name__)


NUM_CLASSES = 3
CLASS_NAMES = {0: 'sotto AT', 1: 'tra AT e RC', 2: 'sopra RC'}


def hparams_for_tensorboard(cfg):
    # add_hparams accetta solo scalari: i sottogruppi vanno appiattiti e prefissati,
    # altrimenti model.name e optimizer.name si sovrascrivono a vicenda.
    def scalars(d):
        return {k: v for (k, v) in d.items()
                if isinstance(v, (int, float, str, bool, torch.Tensor))}

    return {**scalars(cfg),
            **{f'model/{k}': v for (k, v) in scalars(cfg.model).items()},
            **{f'optimizer/{k}': v for (k, v) in scalars(cfg.optimizer).items()}}


@hydra.main(config_path="../conf", config_name="config", version_base='1.3')
def main(cfg: DictConfig):

    log.info("configurazione:\n%s", OmegaConf.to_yaml(cfg))

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
    log.info(f"train: {len(train_index)} soggetti | validation: {len(val_index)} | test: {len(test_index)}")

    # Le statistiche di standardizzazione vengono dai soli soggetti di training e
    # si applicano invariate a validation e test: il test set non contribuisce
    # nemmeno alla scala degli ingressi.
    stats = feature_stats(dataset_path, train_index, columns) if cfg.standardize else None
    if stats is not None:
        log.info("standardizzazione attiva: media e deviazione dai %d soggetti di training",
                 len(train_index))
    bs = cfg.optimizer.batch_size
    train_dataloader = make_dataloader(dataset_path, train_index, columns, bs, True, stats=stats)
    val_dataloader = make_dataloader(dataset_path, val_index, columns, bs, False, stats=stats)
    test_dataloader = make_dataloader(dataset_path, test_index, columns, bs, False, stats=stats)

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
            log.info(f"Epoch {e}: Validation loss improved to {val_loss:.4f}. Counter reset.")
        else:
            early_stop_counter += 1
            log.info(f"Epoch {e}: No improvement in validation loss. Counter: {early_stop_counter}/{patience}")

        if early_stop_counter >= patience:
            log.info(f"Early stopping at epoch {e}. Best validation loss: {best_val_loss:.4f}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Misura finale sul test set, mai usato per scegliere nulla.
    test_loss, test_labels, test_preds = collect_predictions(
        model, test_dataloader, loss_function, device)
    test_precision, test_recall, test_f1 = macro_scores(test_labels, test_preds)
    writer.add_scalar('Loss/test', test_loss, best_epoch)
    writer.add_scalar('F1/test', test_f1, best_epoch)
    log.info(f"\nTest set ({len(test_index)} soggetti, epoca {best_epoch}): "
          f"F1 = {test_f1:.4f} | precision = {test_precision:.4f} | "
          f"recall = {test_recall:.4f} | loss = {test_loss:.4f}")

    # La classe centrale e' la minoritaria: la macro F1 la pesa come le altre,
    # quindi il dettaglio per classe dice dove il modello sbaglia davvero.
    labels_order = list(range(NUM_CLASSES))
    per_class = {
        'names': CLASS_NAMES,
        'f1': f1_score(test_labels, test_preds, average=None, labels=labels_order, zero_division=0).tolist(),
        'precision': precision_score(test_labels, test_preds, average=None, labels=labels_order, zero_division=0).tolist(),
        'recall': recall_score(test_labels, test_preds, average=None, labels=labels_order, zero_division=0).tolist(),
        'support': [int((torch.tensor(test_labels) == c).sum()) for c in labels_order],
    }
    log.info("\nDettaglio per classe sul test set:")
    for c in labels_order:
        log.info(f"  {c} ({CLASS_NAMES[c]:11}): F1 {per_class['f1'][c]:.3f} | "
              f"precision {per_class['precision'][c]:.3f} | recall {per_class['recall'][c]:.3f} | "
              f"{per_class['support'][c]} campioni")
    cm = confusion_matrix(test_labels, test_preds, labels=labels_order).tolist()
    log.info("matrice di confusione (righe = vero, colonne = predetto):")
    for c, row in zip(labels_order, cm):
        log.info(f"  {CLASS_NAMES[c]:11} {row}")

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
        'standardize': bool(cfg.standardize),
        'model': OmegaConf.to_container(cfg.model),
        'optimizer': OmegaConf.to_container(cfg.optimizer),
        'best_epoch': best_epoch,
        'validation': {'loss': val_loss, 'f1': val_f1,
                       'precision': val_precision, 'recall': val_recall},
        'test': {'loss': test_loss, 'f1': test_f1,
                 'precision': test_precision, 'recall': test_recall,
                 'n_subjects': len(test_index),
                 'per_class': per_class,
                 'confusion_matrix': cm},
    }
    with open(RESULTS_DIR / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Senza questo l'unico modo di riavere il modello addestrato e' rifare il
    # training. E' deterministico, ma non e' una scusa per non salvarlo.
    checkpoint_dir = CHECKPOINT_DIR
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / 'threshold_estimator.pt'
    torch.save({'model_state_dict': model.state_dict(),
                'features': columns,
                # senza le statistiche il checkpoint non e' riutilizzabile: gli
                # ingressi andrebbero standardizzati con numeri diversi
                'stats': None if stats is None else [stats[0].tolist(), stats[1].tolist()],
                'model': OmegaConf.to_container(cfg.model),
                'best_epoch': best_epoch,
                'test_f1': test_f1}, checkpoint_path)
    log.info(f"\nModello salvato in {checkpoint_path}")


if __name__ == '__main__':
    main()
