"""Errore sulla soglia in secondi e watt, invece della F1 per respiro.

La F1 per respiro dice quanti istanti sono classificati bene. Chi legge un CPET
vuole sapere un'altra cosa: di quanto e' sbagliato il punto in cui AT e RC
vengono collocati. Questo comando prende il modello addestrato, proietta le sue
predizioni sulla sequenza 0->1->2 piu' vicina e confronta le due transizioni con
quelle segnate dall'operatore.
"""

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from respiration.data.dataset import TimeSeriesDataset
from respiration.models.threshold_estimator import ThresholdEstimator
from respiration.paths import CHECKPOINT_DIR, RESULTS_DIR
from respiration.splits import make_splits
from respiration.thresholds import threshold_errors
from respiration.training import get_device

log = logging.getLogger(__name__)

# Tolleranze entro cui riportare la quota di soggetti. Non sono soglie cliniche
# validate: servono a leggere la distribuzione dell'errore, non a promuovere il
# modello.
TOLLERANZE_SECONDI = (15, 30, 60)
TOLLERANZE_WATT = (10, 20, 30)


def riassunto(valori, unita, tolleranze):
    v = np.asarray(valori, dtype=float)
    out = {
        'unita': unita,
        'errore_assoluto_medio': float(np.abs(v).mean()),
        'errore_assoluto_mediano': float(np.median(np.abs(v))),
        # La media firmata e' fragile: su RC un solo soggetto a +270 W la porta
        # da negativa a positiva. La mediana firmata dice la direzione vera.
        'bias_medio': float(v.mean()),
        'bias_mediano': float(np.median(v)),
        'deviazione': float(v.std()),
        'peggiore': float(np.abs(v).max()),
        'entro': {str(t): float((np.abs(v) <= t).mean()) for t in tolleranze},
    }
    return out


@hydra.main(config_path="../conf", config_name="config", version_base='1.3')
def main(cfg: DictConfig):
    device = get_device()
    dataset_path = Path(cfg.data_dir) / cfg.dataset.path

    checkpoint_path = CHECKPOINT_DIR / 'threshold_estimator.pt'
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Nessun modello in {checkpoint_path}: lanciare prima resp-train")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    columns = list(ckpt['features'])
    stats = None if ckpt.get('stats') is None else (
        np.array(ckpt['stats'][0], dtype=np.float32),
        np.array(ckpt['stats'][1], dtype=np.float32))

    # Un modello addestrato su ingressi standardizzati, valutato senza
    # standardizzarli, produce predizioni sbagliate senza segnalare nulla.
    if stats is None and cfg.standardize:
        raise ValueError(
            f"{checkpoint_path.name} non contiene le statistiche di "
            "standardizzazione ma la config ha standardize=true. Rilanciare "
            "resp-train per rigenerare il checkpoint.")

    model = ThresholdEstimator(len(columns), ckpt['model']['hidden_size'],
                               ckpt['model']['num_layers'],
                               dropout=ckpt['model']['dropout']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    total = TimeSeriesDataset(dataset_path)
    folds, dev_index, test_index = make_splits(len(total),
                                               n_folds=cfg.split.n_folds,
                                               test_size=cfg.split.test_size,
                                               seed=cfg.split.seed)
    log.info("modello da %s | feature %s | standardizzato %s",
             checkpoint_path.name, columns, stats is not None)
    log.info("valutazione su %d soggetti di test", len(test_index))

    dataset = TimeSeriesDataset(dataset_path, test_index, columns=columns, stats=stats)
    per_soggetto = []
    with torch.no_grad():
        for i in range(len(dataset)):
            features, labels = dataset[i]
            lengths = torch.tensor([len(features)])
            preds = model(features.unsqueeze(0).to(device), lengths)
            preds = preds.view(-1, 3).argmax(dim=1).cpu().numpy()

            df = dataset.df_map[dataset.file_name(i)]
            secondi = df['t'].dt.total_seconds().to_numpy()
            watt = df['Load'].to_numpy(dtype=float)
            e = threshold_errors(preds, labels.flatten().numpy(), secondi, watt)
            e['subject'] = dataset.file_name(i)
            per_soggetto.append(e)

    risultati = {'n_subjects': len(per_soggetto), 'features': columns,
                 'per_subject': per_soggetto, 'summary': {}}
    for soglia in ('at', 'rc'):
        risultati['summary'][soglia] = {
            'secondi': riassunto([s[soglia]['seconds'] for s in per_soggetto],
                                 'secondi', TOLLERANZE_SECONDI),
            'watt': riassunto([s[soglia]['watts'] for s in per_soggetto],
                              'watt', TOLLERANZE_WATT),
            'respiri': riassunto([s[soglia]['breaths'] for s in per_soggetto],
                                 'respiri', (5, 10, 20)),
        }

    for soglia in ('at', 'rc'):
        log.info("")
        log.info("%s, su %d soggetti:", soglia.upper(), len(per_soggetto))
        for unita in ('secondi', 'watt', 'respiri'):
            r = risultati['summary'][soglia][unita]
            entro = "  ".join(f"entro {k}: {v:.0%}" for k, v in r['entro'].items())
            log.info("  %-8s errore assoluto mediano %6.1f (medio %6.1f) | bias mediano %+6.1f | peggiore %6.1f | %s",
                     unita, r['errore_assoluto_mediano'], r['errore_assoluto_medio'],
                     r['bias_mediano'], r['peggiore'], entro)

    out = RESULTS_DIR / 'threshold_error.json'
    with open(out, 'w') as f:
        json.dump(risultati, f, indent=4)
    log.info("")
    log.info("Scritto in %s", out)


if __name__ == '__main__':
    main()
