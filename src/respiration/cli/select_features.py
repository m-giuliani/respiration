"""Seleziona il gruppo di feature migliore in cross validation sul dev set.

Va lanciato prima del tuning: gli iperparametri vanno cercati sul feature set
che si userà davvero, non su un altro. Per non favorire nessun gruppo, tutti
vengono confrontati con lo stesso modello baseline definito in config.
"""

import hydra
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from itertools import combinations
from pathlib import Path
import json
import logging

from respiration.paths import CONF_DIR, RESULTS_DIR
from respiration.splits import make_splits
from respiration.models.threshold_estimator import ThresholdEstimator
from respiration.data.dataset import TimeSeriesDataset
from respiration.training import cross_validate, get_device


log = logging.getLogger(__name__)


def generate_combinations(groups):
    """Tutte le combinazioni non vuote dei gruppi di feature."""
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


@hydra.main(config_path="../conf", config_name="config", version_base='1.3')
def main(cfg: DictConfig):

    device = get_device()
    dataset_path = Path(cfg.data_dir) / cfg.dataset.path
    baseline = cfg.feature_selection

    groups = OmegaConf.to_container(cfg.feature_groups)
    all_groups = generate_combinations(groups)
    log.info(f"{len(all_groups)} combinazioni da valutare in {cfg.split.n_folds}-fold CV")

    total_dataset = TimeSeriesDataset(dataset_path)
    folds, dev_index, test_index = make_splits(len(total_dataset),
                                               n_folds=cfg.split.n_folds,
                                               test_size=cfg.split.test_size,
                                               seed=cfg.split.seed)
    log.info(f"dev: {len(dev_index)} soggetti | test: {len(test_index)} (non toccato)")

    def evaluate_feature_group(group_name, group_features):
        def build_model():
            return ThresholdEstimator(input_size=len(group_features),
                                      hidden_size=baseline.hidden_size,
                                      num_layers=baseline.num_layers,
                                      dropout=baseline.dropout)

        def make_optimizer(model):
            return torch.optim.Adam(model.parameters(), lr=baseline.lr)

        fold_scores = cross_validate(dataset_path, group_features, folds,
                                     build_model, make_optimizer,
                                     epochs=baseline.epochs,
                                     batch_size=baseline.batch_size,
                                     device=device, standardize=cfg.standardize)
        mean_f1 = sum(fold_scores) / len(fold_scores)
        spread = max(fold_scores) - min(fold_scores)
        log.info(f"  {group_name}: F1 = {mean_f1:.4f} (fold: {[round(s, 3) for s in fold_scores]}, spread {spread:.3f})")
        return mean_f1, fold_scores

    results = {}
    for group_name, group_features in all_groups.items():
        log.info(f"Testing group: {group_name} ({len(group_features)} feature)")
        mean_f1, fold_scores = evaluate_feature_group(group_name, group_features)
        results[group_name] = {'mean_f1': mean_f1, 'fold_f1': fold_scores,
                               'columns': group_features}

    # ordina dal migliore
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_f1'], reverse=True)
    if not sorted_results:
        log.info("Nessun risultato disponibile. Controlla i dati o la configurazione.")
        return

    log.info("\nRisultati ordinati (F1 medio in cross validation):")
    for group_name, res in sorted_results:
        log.info(f"{group_name}: F1 = {res['mean_f1']:.4f}")

    best_name, best = sorted_results[0]
    output_path = RESULTS_DIR / "feature_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "protocol": f"{cfg.split.n_folds}-fold CV sul dev set, test set escluso",
            "baseline_model": OmegaConf.to_container(baseline),
            "results": {k: v for k, v in sorted_results},
            "best_group": {"name": best_name, "mean_f1": best['mean_f1'],
                           "columns": best['columns']},
        }, f, indent=4)

    # Scrive il gruppo vincente come normale config group di Hydra, così il
    # tuning e l'addestramento lo usano senza ricopiarlo a mano e una scelta
    # esplicita da riga di comando (features=columns_all) continua a vincere.
    out = CONF_DIR / "features" / "selected.yaml"
    with open(out, "w") as f:
        f.write("# Generato da resp-select-features: gruppo vincente in cross validation.\n")
        yaml.safe_dump({"name": f"selected_{best_name}",
                        "columns": list(best['columns'])}, f,
                       default_flow_style=False, sort_keys=False)

    log.info(f"\nMiglior gruppo: {best_name} con F1 = {best['mean_f1']:.4f}")
    log.info(f"Scritto in {out}")


if __name__ == "__main__":
    main()
