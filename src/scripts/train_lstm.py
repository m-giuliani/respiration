import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score
import os
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

current_time = datetime.now().strftime("%b%d_%H-%M-%S")
log_dir = os.path.join("runs", current_time)
writer = SummaryWriter(f'runs/lstm_experiments_h/{current_time}')

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

def training_loop(model, dataloader, loss_function, optimizer, epoch):
    running_loss = 0.0
    all_labels = []
    all_preds = []

    for data, lengths, targets in tqdm(dataloader):
        data.to(device)
        lengths.to(device)
        targets.to(device)
        preds = model(data, lengths)

        preds = preds.view(-1, 3).to(device)
        targets = targets.view(-1).to(device)
        loss = loss_function(preds, targets)
        loss.backward()
        optimizer.step()
        model.zero_grad()

        running_loss += loss.item()

        mask = (targets != -1)
        all_labels.extend(targets[mask].cpu().numpy())
        _, predicted = torch.max(preds.data, 1)
        all_preds.extend(predicted[mask].cpu().numpy())

    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    writer.add_scalar('Loss/train', running_loss / len(dataloader), epoch)
    writer.add_scalar('Precision/train', precision, epoch)
    writer.add_scalar('Recall/train', recall, epoch)
    writer.add_scalar('F1/train', f1, epoch)
    return running_loss / len(dataloader), f1, precision, recall


def validation_loop(model, dataloader, loss_function, epoch):
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for data, lengths, targets in tqdm(dataloader):
            preds = model(data, lengths)
            preds = preds.view(-1, 3).cpu()
            targets = targets.view(-1).cpu()
            loss = loss_function(preds, targets)

            running_loss += loss.item()

            mask = (targets != -1)
            all_labels.extend(targets[mask].cpu().numpy())
            _, predicted = torch.max(preds.data, 1)
            all_preds.extend(predicted[mask].cpu().numpy())

        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')

        writer.add_scalar('Loss/valid', running_loss / len(dataloader), epoch)
        writer.add_scalar('Precision/valid', precision, epoch)
        writer.add_scalar('Recall/valid', recall, epoch)
        writer.add_scalar('F1/valid', f1, epoch)
    return running_loss / len(dataloader), f1, precision, recall


@hydra.main(config_path="hyperparams", config_name="config", version_base='1.3')
def main(cfg: DictConfig):

    print("\n\n\n",cfg,"\n\n\n")

    all_columns = ['t', 'Rf', 'VO2', 'VCO2', 'VE/VO2', 'VE/VCO2', 'HR', 'VO2/HR', 'Load',
                   'label_at', 'label_rc']
    # columns = ['Rf', 'VO2', 'VCO2', 'VE/VO2', 'VE/VCO2', 'HR', 'VO2/HR', 'Load']
    columns = cfg.features.columns

    #columns = [col for col in cfg.features.columns if col != 'Load']

    all_columns = columns + cfg.dataset.labels
    working_dir = os.getcwd()
    orig_cwd = hydra.utils.get_original_cwd()
    lr = cfg.optimizer.lr
    num_epochs = int(cfg.epochs_final)
    # window_size = ['']
    hidden_size = int(cfg.model.hidden_size)
    num_layers = int(cfg.model.num_layers)
    dataset_path = Path(cfg.data_dir) / cfg.dataset.path
    total_dataset = TimeSeriesDataset(dataset_path, columns=columns)

    model = ThresholdEstimator(len(total_dataset.features), hidden_size, num_layers, dropout=cfg.model.dropout).to(device)
    loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    if cfg.optimizer.name == 'adam':
        betas = cfg.optimizer.betas
        wd = cfg.optimizer.weight_decay
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif cfg.optimizer.name == 'sgd':
        m = 0.9
        n = True
        wd = cfg.optimizer.weight_decay
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=m, weight_decay=wd, nesterov=True)

    index_tr, index_te = train_test_split(range(len(total_dataset)), random_state=42)
    train_dataset = TimeSeriesDataset(dataset_path, index_tr, columns=columns)
    test_dataset = TimeSeriesDataset(dataset_path, index_te, columns=columns)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)


     # Configurazion dell'early stopping
    patience = 5  
    min_delta = 0.001  
    best_val_loss = float("inf")
    best_model_state = None
    early_stop_counter = 0

    for e in tqdm(range(num_epochs)):
        tr_loss, tr_f1, tr_precision, tr_recall = training_loop(model, train_dataloader, loss_function, optimizer, e)
        val_loss, val_f1, val_precision, val_recall = validation_loop(model, test_dataloader, loss_function, e)
    
        if best_val_loss - val_loss > min_delta:
            best_val_loss = val_loss
            best_model_state = model.state_dict() 
            early_stop_counter = 0 
            print(f"Epoch {e}: Validation loss improved to {val_loss:.4f}. Counter reset.")
        else:
            early_stop_counter += 1  
            print(f"Epoch {e}: No improvement in validation loss. Counter: {early_stop_counter}/{patience}")
        
        
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {e}. Best validation loss: {best_val_loss:.4f}")
            model.load_state_dict(best_model_state)  
            break

    hyp_cfg = {**{k:v for (k,v) in cfg.items() if isinstance(v, (int, float, str, bool, torch.Tensor))}, **cfg.model, **cfg.optimizer}
    writer.add_hparams(hyp_cfg, {'tr_loss': tr_loss, 'val_loss': val_loss,
                             'tr_f1': tr_f1, 'val_f1': val_f1,
                             'tr_precision': tr_precision, 'val_precision': val_precision,
                             'tr_recall': tr_recall, 'val_recall': val_recall})
    writer.close()


if __name__ == '__main__':
    main()