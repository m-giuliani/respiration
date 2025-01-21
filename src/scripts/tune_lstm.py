import optuna
import hydra
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from datetime import datetime
import sys
from pathlib import Path
from omegaconf import DictConfig
import os
sys.path.append(os.path.abspath(os.curdir).split('respirazione')[0] + 'respirazione')
from config.definitions import ROOT_DIR
import json
from src.threshold_estimator import ThresholdEstimator
from src.timeseriesdataset import TimeSeriesDataset, collate_fn
from train_lstm import training_loop, validation_loop

device = "cpu"
def objective(trial, cfg):
    

    
    # Funzione obiettivo per la ricerca degli iperparametri tramite Optuna.
    # Parametri:
    # - trial: oggetto di Optuna che tiene traccia dei tentativi.
    # - cfg: configurazione che contiene le impostazioni del progetto.
    

    
     # Definiamo lo spazio di ricerca per gli iperparametri
    hidden_size = trial.suggest_int("hidden_size", 32, 512)  
    num_layers = trial.suggest_int("num_layers", 1, 4)
    lr = trial.suggest_float("lr", 1e-5, 1e-2) 
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    
    # Tipo di ottimizzatore da utilizzare
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])

    dropout = trial.suggest_float("dropout", 0, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 0, 0.01)
    
    
    dataset_path = Path(cfg.data_dir) / cfg.dataset.path
    columns = cfg.features.columns
    total_dataset = TimeSeriesDataset(dataset_path, columns=columns)
    

    index_tr, index_te = train_test_split(range(len(total_dataset)), random_state=42)
    train_dataset = TimeSeriesDataset(dataset_path, index_tr, columns=columns)
    test_dataset = TimeSeriesDataset(dataset_path, index_te, columns=columns)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    model = ThresholdEstimator(
        input_size=len(total_dataset.features),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    #cerco i valori migliori
    best_val_f1 = 0
    for epoch in range(cfg.epochs_tuning): 
        tr_loss, tr_f1, _, _ = training_loop(model, train_dataloader, loss_function, optimizer, epoch)
        val_loss, val_f1, _, _ = validation_loop(model, test_dataloader, loss_function, epoch)
        trial.report(val_f1, epoch) # Report dell'F1 score per l'epoca corrente
        if trial.should_prune(): #verifica se il trial deve essere "potato"
            raise optuna.exceptions.TrialPruned()
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

    return best_val_f1

@hydra.main(config_path="hyperparams", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    def logging_callback(study, trial):
        print(f"Trial {trial.number}: Best Value so far = {study.best_value}")

    #crea lo studio per massimizzare f1, interrompe usando mediana 
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
    # Esegue l'ottimizzazione su 30 trial
    study.optimize(lambda trial: objective(trial, cfg), n_trials=30, callbacks=[logging_callback])  

    print("Best hyperparameters:", study.best_params)
    print("Best F1 Score:", study.best_value)
 
    best_params = study.best_params #parametri associale al miglior valore di f1
    best_params['best_f1'] = study.best_value #valore restituito da objective

    output_path = Path('src/scripts/best_params.json')

    with open(output_path, 'w') as f:
        json.dump(best_params, f, indent=4)

if __name__ == "__main__":
    main()
