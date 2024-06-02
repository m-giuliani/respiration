import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score
import os
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from config.definitions import ROOT_DIR
from datetime import datetime

from src.threshold_estimator import ThresholdEstimator
from src.timeseriesdataset import TimeSeriesDataset, collate_fn

torch.manual_seed(42)

current_time = datetime.now().strftime("%b%d_%H-%M-%S")
log_dir = os.path.join("runs", current_time)
writer = SummaryWriter(f'runs/lstm_experiments/{current_time}')

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

        # Flatten the output and labels to match the expected input of CrossEntropyLoss
        preds = preds.view(-1, 3).cuda()
        targets = targets.view(-1).cuda()
        loss = loss_function(preds, targets)
        loss.backward()
        optimizer.step()
        model.zero_grad()

        running_loss += loss.item()

        mask = (targets != -1)
        all_labels.extend(targets[mask].cpu().numpy())
        _, predicted = torch.max(preds.data, 1)
        all_preds.extend(predicted[mask].cpu().numpy())

    # Calculate precision and recall
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    # Log metrics to TensorBoard
    writer.add_scalar('Loss/train', running_loss / len(dataloader), epoch)
    writer.add_scalar('Precision/train', precision, epoch)
    writer.add_scalar('Recall/train', recall, epoch)
    return running_loss / len(dataloader), precision, recall


def validation_loop(model, dataloader, epoch):
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

        # Calculate precision and recall
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/valid', running_loss / len(dataloader), epoch)
        writer.add_scalar('Precision/valid', precision, epoch)
        writer.add_scalar('Recall/valid', recall, epoch)
    return running_loss / len(dataloader), precision, recall


@hydra.main(config_path="runs/hyperparams", config_name="config")
def main(cfg: DictConfig):
    # will-be hyperparameters
    all_columns = ['t', 'Rf', 'VO2', 'VCO2', 'VE/VO2', 'VE/VCO2', 'HR', 'VO2/HR', 'Load',
                   'label_at', 'label_rc']
    config_file_path = "runs/hyperparams/config.yaml"
    
    with open(config_file_path) as f:
        config = yaml.safe_load(f)
    lr = config['lr']
    num_epochs = int(config['epochs'])
    # window_size = ['']
    columns = ['Rf', 'VO2', 'VCO2', 'VE/VO2', 'VE/VCO2', 'HR', 'VO2/HR', 'Load']
    hidden_size = int(config['hidden_size'])
    num_layers = int(config['num_layers'])
    dataset_path = Path(ROOT_DIR) / 'data/sequence_dataset.pkl'
    total_dataset = TimeSeriesDataset(dataset_path, columns=columns)

    model = ThresholdEstimator(len(total_dataset.features), hidden_size, num_layers).to(device)
    loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    index_tr, index_te = train_test_split(range(len(total_dataset)), random_state=42)
    train_dataset = TimeSeriesDataset(dataset_path, index_tr, columns=columns)
    test_dataset = TimeSeriesDataset(dataset_path, index_te, columns=columns)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    for e in tqdm(range(num_epochs)):
        tr_loss, tr_precision, tr_recall = training_loop(model, train_dataloader, loss_function, optimizer, e)
        val_loss, val_precision, val_recall = validation_loop(model, test_dataloader, e)
    writer.add_hparams(config, {'tr_loss': tr_loss, 'val_loss': val_loss, 'tr_precision': tr_precision,
                                'val_precision': val_precision, 'tr_recall': tr_recall, 'val_recall': val_recall})
    writer.close()


if __name__ == '__main__':
    main()
