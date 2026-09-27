import torch
import hydra
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path
from omegaconf import DictConfig
from itertools import combinations
import os
import json
sys.path.append(os.path.abspath(os.curdir).split('respirazione')[0] + 'respirazione')
from config.definitions import ROOT_DIR
from src.threshold_estimator import ThresholdEstimator
from src.timeseriesdataset import TimeSeriesDataset, collate_fn
from train_lstm import training_loop, validation_loop, get_device, PAD_LABEL


@hydra.main(config_path="hyperparams", config_name="config", version_base='1.3')
def main(cfg: DictConfig):

    device = get_device()
    dataset_path = Path(cfg.data_dir) / cfg.dataset.path

    groups = cfg.feature_groups
    print("\n\n\n", groups, "\n\n\n")

    def evaluate_feature_group(group_name, group_features):
        print(f"Testing group: {group_name}")

        total_dataset = TimeSeriesDataset(dataset_path, columns=group_features)
        index_tr, index_te = train_test_split(range(len(total_dataset)), random_state=42)
        train_dataset = TimeSeriesDataset(dataset_path, index_tr, columns=group_features)
        test_dataset = TimeSeriesDataset(dataset_path, index_te, columns=group_features)

        train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

        model = ThresholdEstimator(input_size=len(group_features),
                                   hidden_size=cfg.model.hidden_size,
                                   num_layers=cfg.model.num_layers,
                                   dropout=cfg.model.dropout).to(device)
        loss_function = torch.nn.CrossEntropyLoss(ignore_index=PAD_LABEL)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)

        best_val_f1 = 0
        for epoch in range(cfg.epochs_tuning):
            training_loop(model, train_dataloader, loss_function, optimizer, epoch, device=device)
            _, val_f1, _, _ = validation_loop(model, test_dataloader, loss_function, epoch, device=device)
            best_val_f1 = max(best_val_f1, val_f1)

        return best_val_f1

    # Genera combinazioni di feature
    def generate_combinations(groups):
        combined_groups = {}
        keys = list(groups.keys())
        for i in range(1, len(keys) + 1):
            for combo in combinations(keys, i):
                group_name = "_".join(combo)
                group_features = []
                for key in combo:
                    group_features.extend(groups[key])
                combined_groups[group_name] = group_features
        return combined_groups

    all_groups = generate_combinations(groups)
    print("\n\n\n", all_groups, "\n\n\n")

    results = {}
    for group_name, group_features in all_groups.items():
        f1_score = evaluate_feature_group(group_name, group_features)
        results[group_name] = f1_score

    # ordina dal migliore
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    if not sorted_results:
        print("Nessun risultato disponibile. Controlla i dati o la configurazione.")
        return
    print("\nRisultati ordinati:")
    for group_name, f1_score in sorted_results:
        print(f"{group_name}: F1 = {f1_score:.4f}")

    # Path assoluto: Hydra puo' essere lanciato da qualunque cartella.
    output_path = Path(ROOT_DIR) / "src/scripts/feature_results.json"
    results_dict = {
        "f1_scores": results,
        "best_group": {
            "name": sorted_results[0][0],
            "f1": sorted_results[0][1]
        }
    }

    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=4)

    best_group = sorted_results[0]
    print(f"Miglior gruppo: {best_group[0]} con F1 = {best_group[1]:.4f}")


if __name__ == "__main__":
    main()
