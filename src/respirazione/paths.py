"""Percorsi del progetto, ricavati dalla posizione del package.

Sostituisce il vecchio `config/definitions.py`, che ricavava la radice con
`os.path.abspath(os.curdir).split('respirazione')[0] + 'respirazione'`: si
rompeva se la cartella veniva rinominata e dipendeva dalla directory corrente.
Vale per un'installazione editabile (`pip install -e .`), che e' il modo in cui
questo progetto va usato.
"""

from pathlib import Path

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Le config stanno dentro il package, quindi si trovano rispetto a questo file e
# non rispetto alla radice: funzionano anche se il package viene spostato.
CONF_DIR = Path(__file__).resolve().parent / "conf"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


# Permette alle config Hydra di riferirsi alla radice del progetto con
# ${project_root:}, invece di dipendere dalla directory da cui si lancia.
OmegaConf.register_new_resolver("project_root", lambda: str(PROJECT_ROOT), replace=True)
