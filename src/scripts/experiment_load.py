"""Perche' Load peggiora il modello: la scala o il contenuto?

La feature selection ha stabilito che aggiungere Load al gruppo vincente fa
scendere l'F1. Restano due spiegazioni che quel confronto non separa:

1. Load descrive il protocollo della rampa, non la risposta del soggetto,
   quindi e' informazione che non generalizza.
2. Load e' l'unica colonna che load_data.py non normalizza mai, perche' il suo
   valore a riposo e' 0. Entra in watt grezzi (media 66, std 76) mentre ogni
   altra feature e' un rapporto con media tra 0.96 e 4.55.

Questo script confronta, a parita' di fold, epoche e modello, il gruppo
vincente da solo contro lo stesso gruppo piu' Load nelle tre forme: grezza,
rapportata al picco del soggetto, e riscalata di una costante. Se una forma
normalizzata recupera il calo, il problema era la scala. Se nessuna lo
recupera, resta l'ipotesi del contenuto.
"""

import hydra
import json
import os
import sys
import torch
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
sys.path.append(os.path.abspath(os.curdir).split('respirazione')[0] + 'respirazione')
from config.definitions import ROOT_DIR

from src.splits import make_splits
from src.threshold_estimator import ThresholdEstimator
from src.timeseriesdataset import TimeSeriesDataset
from src.training import cross_validate, get_device


@hydra.main(config_path="hyperparams", config_name="config", version_base='1.3')
def main(cfg: DictConfig):
    device = get_device()
    dataset_path = Path(cfg.data_dir) / cfg.dataset.path
    baseline = cfg.feature_selection
    base_columns = list(cfg.features.columns)

    varianti = {
        'senza Load': base_columns,
        'Load grezza (watt)': base_columns + ['Load'],
        'Load / picco del soggetto': base_columns + ['Load_peak'],
        'Load / 100 (costante)': base_columns + ['Load_scaled'],
    }

    folds, dev_index, test_index = make_splits(len(TimeSeriesDataset(dataset_path)),
                                               n_folds=cfg.split.n_folds,
                                               test_size=cfg.split.test_size,
                                               seed=cfg.split.seed)
    print(f"base: {len(base_columns)} feature | dev {len(dev_index)} soggetti in "
          f"{cfg.split.n_folds} fold | test {len(test_index)} (non toccato)")
    print(f"modello baseline: hidden {baseline.hidden_size}, {baseline.num_layers} layer, "
          f"dropout {baseline.dropout}, lr {baseline.lr}, batch {baseline.batch_size}, "
          f"{baseline.epochs} epoche\n")

    risultati = {}
    for nome, columns in varianti.items():
        scores = cross_validate(
            dataset_path, columns, folds,
            lambda columns=columns: ThresholdEstimator(len(columns),
                                                       baseline.hidden_size,
                                                       baseline.num_layers,
                                                       dropout=baseline.dropout),
            lambda m: torch.optim.Adam(m.parameters(), lr=baseline.lr),
            epochs=baseline.epochs, batch_size=baseline.batch_size, device=device)
        media = sum(scores) / len(scores)
        risultati[nome] = {'columns': columns, 'fold_f1': scores, 'mean_f1': media}
        print(f"  {nome:28} F1 = {media:.4f}   fold: {[round(s, 3) for s in scores]}")

    riferimento = risultati['senza Load']['mean_f1']
    print(f"\nscarto rispetto al gruppo senza Load ({riferimento:.4f}):")
    for nome, r in risultati.items():
        if nome == 'senza Load':
            continue
        print(f"  {nome:28} {r['mean_f1'] - riferimento:+.4f}")

    output = Path(ROOT_DIR) / 'src/scripts/load_experiment.json'
    with open(output, 'w') as f:
        json.dump({'protocol': f"{cfg.split.n_folds}-fold CV sul dev set, "
                               f"{baseline.epochs} epoche per fold, stessi fold e stesso modello",
                   'baseline_model': OmegaConf.to_container(baseline),
                   'base_columns': base_columns,
                   'variants': risultati}, f, indent=4)
    print(f"\nScritto in {output}")


if __name__ == '__main__':
    main()
