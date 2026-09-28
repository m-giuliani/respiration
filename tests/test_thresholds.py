import numpy as np
import pytest

from respiration.thresholds import monotone_transitions, threshold_errors, true_transitions


def forza_bruta(preds, num_classes=3):
    """Cerca la coppia (a, b) migliore esaminando tutte le coppie: O(T^2)."""
    T = len(preds)
    migliore, coppia = -1, None
    for a in range(T + 1):
        for b in range(a, T + 1):
            atteso = np.concatenate([np.zeros(a), np.ones(b - a), np.full(T - b, 2)])
            accordo = int((np.asarray(preds) == atteso).sum())
            if accordo > migliore:
                migliore, coppia = accordo, (a, b)
    return coppia, migliore


def accordo(preds, a, b):
    T = len(preds)
    atteso = np.concatenate([np.zeros(a), np.ones(b - a), np.full(T - b, 2)])
    return int((np.asarray(preds) == atteso).sum())


def test_sequenza_pulita_ritrova_le_transizioni_esatte():
    preds = [0, 0, 0, 1, 1, 2, 2, 2]
    assert monotone_transitions(preds) == (3, 5)


def test_tutti_zeri_mette_le_transizioni_alla_fine():
    a, b = monotone_transitions([0, 0, 0, 0])
    assert a == b == 4


def test_tutti_due_mette_le_transizioni_all_inizio():
    a, b = monotone_transitions([2, 2, 2, 2])
    assert a == b == 0


@pytest.mark.parametrize("seme", range(25))
def test_coincide_con_la_forza_bruta(seme):
    rng = np.random.default_rng(seme)
    preds = rng.integers(0, 3, size=rng.integers(4, 30))
    a, b = monotone_transitions(preds)
    _, migliore = forza_bruta(preds)
    # la coppia puo' non essere unica, ma l'accordo deve essere il massimo
    assert accordo(preds, a, b) == migliore
    assert a <= b


def test_resiste_a_una_predizione_isolata_sbagliata():
    # un singolo 2 anticipato non deve spostare RC all'inizio
    preds = [0, 0, 0, 2, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2]
    a, b = monotone_transitions(preds)
    assert b >= 8, f"RC collocato a {b}, troppo presto"


def test_transizioni_vere_dalle_etichette():
    assert true_transitions([0, 0, 1, 1, 2, 2]) == (2, 4)
    assert true_transitions([0, 0, 0]) == (3, 3)


def test_errore_firmato_e_nelle_unita_giuste():
    preds = [0, 0, 0, 1, 1, 2, 2]
    labels = [0, 0, 1, 1, 2, 2, 2]
    secondi = np.arange(7) * 10.0
    watt = np.arange(7) * 25.0
    e = threshold_errors(preds, labels, secondi, watt)
    assert e['at']['breaths'] == 1           # predetto a 3, vero a 2
    assert e['at']['seconds'] == 10.0
    assert e['at']['watts'] == 25.0
    assert e['rc']['breaths'] == 1           # predetto a 5, vero a 4
