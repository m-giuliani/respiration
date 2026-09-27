"""Ricerca degli iperparametri con Optuna, in cross validation sul dev set.

Va lanciato dopo resp-select-features: usa il feature set che quello ha selezionato
(config group `selected`), così i parametri vengono cercati per il modello che
si userà davvero. Il test set non viene mai visto durante la ricerca.
"""

import optuna
import hydra
import torch
import torch.optim as optim
import yaml
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import json
import logging

from respirazione.paths import CONF_DIR, RESULTS_DIR
from respirazione.splits import make_splits
from respirazione.models.threshold_estimator import ThresholdEstimator
from respirazione.data.dataset import TimeSeriesDataset
from respirazione.training import cross_validate, get_device


log = logging.getLogger(__name__)


def objective(trial, cfg, dataset_path, columns, folds, device):
    # Funzione obiettivo per la ricerca degli iperparametri tramite Optuna.
    # Restituisce l'F1 medio sui fold di validazione: una configurazione buona
    # su un fold solo non deve vincere per fortuna.

    # Definiamo lo spazio di ricerca per gli iperparametri
    hidden_size = trial.suggest_int("hidden_size", cfg.search.hidden_min, cfg.search.hidden_max)
    num_layers = trial.suggest_int("num_layers", cfg.search.layers_min, cfg.search.layers_max)
    lr = trial.suggest_float("lr", cfg.search.lr_min, cfg.search.lr_max, log=True)
    batch_size = trial.suggest_categorical("batch_size", list(cfg.search.batch_sizes))

    # Tipo di ottimizzatore da utilizzare
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])

    dropout = trial.suggest_float("dropout", 0, 0.5) if num_layers > 1 else 0.0
    weight_decay = trial.suggest_float("weight_decay", cfg.search.wd_min, cfg.search.wd_max, log=True)

    def build_model():
        return ThresholdEstimator(input_size=len(columns), hidden_size=hidden_size,
                                  num_layers=num_layers, dropout=dropout)

    def make_optimizer(model):
        if optimizer_name == "adam":
            return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    def on_fold_end(fold, scores):
        # Report dell'F1 medio dopo ogni fold: i trial scarsi vengono potati
        # senza pagare tutti e cinque i fold.
        trial.report(sum(scores) / len(scores), fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    fold_scores = cross_validate(dataset_path, columns, folds, build_model, make_optimizer,
                                 epochs=cfg.epochs_tuning, batch_size=batch_size,
                                 device=device, on_fold_end=on_fold_end)
    trial.set_user_attr("fold_f1", fold_scores)
    return sum(fold_scores) / len(fold_scores)


def write_tuned_config(best_params):
    """Scrive gli iperparametri vincenti come normali config group di Hydra.

    Group e non override globali: così `model=tenet_lstm` o `optimizer=sgd`
    da riga di comando vincono, come in qualunque progetto Hydra. Ricopiarli a
    mano in config.yaml e' il passaggio che in passato ha prodotto override
    inline incoerenti con i file di gruppo.
    """
    optimizer = {"name": best_params["optimizer"],
                 "lr": best_params["lr"],
                 "weight_decay": best_params["weight_decay"],
                 # batch_size sta nel gruppo optimizer perche' e' un
                 # iperparametro di ottimizzazione come gli altri, e perche'
                 # altrimenti il valore trovato dal tuning non avrebbe un posto
                 # dove essere scritto.
                 "batch_size": best_params["batch_size"]}
    if best_params["optimizer"] == "adam":
        optimizer["betas"] = [0.9, 0.999]
    else:
        optimizer["momentum"] = 0.9
        optimizer["nesterov"] = False

    model = {"name": "tuned_lstm",
             "hidden_size": best_params["hidden_size"],
             "num_layers": best_params["num_layers"],
             # dropout viene suggerito solo con piu' di un layer: con un layer
             # solo nn.LSTM lo ignorerebbe e avvisa, quindi la chiave puo' non
             # esserci tra i best_params.
             "dropout": best_params.get("dropout", 0.0)}

    base = CONF_DIR
    written = []
    for group, payload in (("model", model), ("optimizer", optimizer)):
        path = base / group / "tuned.yaml"
        with open(path, "w") as f:
            f.write("# Generato da resp-tune: non modificare a mano, viene sovrascritto.\n")
            yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)
        written.append(path)
    return written


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    device = get_device()
    dataset_path = Path(cfg.data_dir) / cfg.dataset.path
    columns = list(cfg.features.columns)

    total_dataset = TimeSeriesDataset(dataset_path)
    folds, dev_index, test_index = make_splits(len(total_dataset),
                                               n_folds=cfg.split.n_folds,
                                               test_size=cfg.split.test_size,
                                               seed=cfg.split.seed)
    log.info(f"feature: {columns}")
    log.info(f"dev: {len(dev_index)} soggetti in {cfg.split.n_folds} fold | test: {len(test_index)} (non toccato)")

    def logging_callback(study, trial):
        log.info(f"Trial {trial.number} ({trial.state.name}): "
              f"value = {trial.value if trial.value is not None else float('nan'):.4f} | "
              f"best so far = {study.best_value:.4f}")

    #crea lo studio per massimizzare f1, interrompe usando mediana
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.split.seed),
        # n_startup_trials basso: con 5 i primi cinque trial pagavano tutti i
        # fold senza poter essere potati, ed era la parte piu' costosa della run.
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1),
    )
    study.optimize(lambda trial: objective(trial, cfg, dataset_path, columns, folds, device),
                   n_trials=cfg.n_trials, callbacks=[logging_callback])

    log.info("Best hyperparameters: %s", study.best_params)
    log.info("Best F1 Score: %s", study.best_value)

    best_params = dict(study.best_params) #parametri associati al miglior valore di f1
    output = {
        "protocol": f"{cfg.split.n_folds}-fold CV sul dev set, test set escluso",
        "features": columns,
        "n_trials": cfg.n_trials,
        "epochs_per_fold": cfg.epochs_tuning,
        "best_params": best_params,
        "best_mean_f1": study.best_value,
        "best_fold_f1": study.best_trial.user_attrs.get("fold_f1"),
        "n_pruned": sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED),
    }

    # Path assoluto: Hydra puo' essere lanciato da qualunque cartella.
    with open(RESULTS_DIR / 'best_params.json', 'w') as f:
        json.dump(output, f, indent=4)

    for path in write_tuned_config(best_params):
        log.info(f"Config generata in {path}")


if __name__ == "__main__":
    main()
