"""Perche' Load peggiora il modello: la scala o il contenuto?

La feature selection ha stabilito che aggiungere Load al gruppo vincente fa
scendere l'F1. Restano due spiegazioni che quel confronto non separa:

1. Load descrive il protocollo della rampa, non la risposta del soggetto,
   quindi e' informazione che non generalizza.
2. Load e' l'unica colonna che il caricamento dati non normalizza mai, perche' il suo
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
import logging
import torch
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from respiration.paths import RESULTS_DIR
from respiration.splits import make_splits
from respiration.models.threshold_estimator import ThresholdEstimator
from respiration.data.dataset import TimeSeriesDataset
from respiration.training import cross_validate, get_device


log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base='1.3')
def main(cfg: DictConfig):
    device = get_device()
    dataset_path = Path(cfg.data_dir) / cfg.dataset.path
    baseline = cfg.feature_selection
    # La base deve essere priva di qualunque colonna di carico: cfg.features
    # punta al gruppo vincente, che dopo questo stesso esperimento include
    # Load_peak, e altrimenti la variante grezza ne aggiungerebbe una seconda.
    base_columns = [c for c in cfg.features.columns if not c.startswith('Load')]

    # In combinazione: la scala di Load conta o no, a parita' di informazione?
    in_combinazione = {
        'senza Load': base_columns,
        'Load grezza (watt)': base_columns + ['Load'],
        'Load / picco del soggetto': base_columns + ['Load_peak'],
        'Load / 100 (costante)': base_columns + ['Load_scaled'],
    }
    # Da sola: quanto segnale porta Load di per se'. Load e Load_scaled hanno la
    # stessa forma e conservano il carico assoluto in watt; Load_peak lo
    # cancella, perche' porta la rampa di ogni soggetto da 0 a 1.
    da_sola = {
        'solo Load grezza': ['Load'],
        'solo Load / picco': ['Load_peak'],
        'solo Load / 100': ['Load_scaled'],
    }
    varianti = {**in_combinazione, **da_sola}

    folds, dev_index, test_index = make_splits(len(TimeSeriesDataset(dataset_path)),
                                               n_folds=cfg.split.n_folds,
                                               test_size=cfg.split.test_size,
                                               seed=cfg.split.seed)
    log.info(f"base senza carico: {base_columns}")
    log.info(f"base: {len(base_columns)} feature | dev {len(dev_index)} soggetti in "
          f"{cfg.split.n_folds} fold | test {len(test_index)} (non toccato)")
    log.info(f"modello baseline: hidden {baseline.hidden_size}, {baseline.num_layers} layer, "
          f"dropout {baseline.dropout}, lr {baseline.lr}, batch {baseline.batch_size}, "
          f"{baseline.epochs} epoche\n")

    output = RESULTS_DIR / 'load_experiment.json'

    def salva(risultati):
        # Scrive dopo ogni variante: una run interrotta a metà lascia comunque
        # i risultati già calcolati invece di buttarli.
        with open(output, 'w') as f:
            json.dump({'protocol': f"{cfg.split.n_folds}-fold CV sul dev set, "
                                   f"{baseline.epochs} epoche per fold, stessi fold e stesso modello",
                       'baseline_model': OmegaConf.to_container(baseline),
                       'base_columns': base_columns,
                       'variants': risultati,
                       'in_combinazione': [k for k in in_combinazione if k in risultati],
                       'da_sola': [k for k in da_sola if k in risultati]}, f, indent=4)

    risultati = {}
    for nome, columns in varianti.items():
        scores = cross_validate(
            dataset_path, columns, folds,
            lambda columns=columns: ThresholdEstimator(len(columns),
                                                       baseline.hidden_size,
                                                       baseline.num_layers,
                                                       dropout=baseline.dropout),
            lambda m: torch.optim.Adam(m.parameters(), lr=baseline.lr),
            epochs=baseline.epochs, batch_size=baseline.batch_size, device=device,
            standardize=cfg.standardize)
        media = sum(scores) / len(scores)
        risultati[nome] = {'columns': columns, 'fold_f1': scores, 'mean_f1': media}
        log.info(f"  {nome:28} F1 = {media:.4f}   fold: {[round(s, 3) for s in scores]}")
        salva(risultati)

    riferimento = risultati['senza Load']['mean_f1']
    log.info(f"\nin combinazione, scarto rispetto al gruppo senza Load ({riferimento:.4f}):")
    for nome in in_combinazione:
        if nome == 'senza Load':
            continue
        log.info(f"  {nome:28} {risultati[nome]['mean_f1'] - riferimento:+.4f}")
    log.info("\nda sola:")
    for nome in da_sola:
        log.info(f"  {nome:28} {risultati[nome]['mean_f1']:.4f}")

    log.info(f"\nScritto in {output}")


if __name__ == '__main__':
    main()
