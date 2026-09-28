"""Dalle predizioni per respiro all'errore sulla soglia.

La F1 per respiro e' un surrogato: dice quanti istanti sono classificati bene,
non di quanto sbaglia il punto in cui la soglia viene collocata. Un modello puo'
sbagliare centinaia di respiri sparsi e azzeccare la transizione entro pochi
watt, o il contrario, e le due cose sono clinicamente diversissime.

Le predizioni grezze non danno due sole transizioni: oscillano. Servono quindi
due indici, uno per AT e uno per RC, che riassumano la sequenza predetta. Qui si
prende la coppia che concorda al massimo con le predizioni fra tutte le sequenze
fisiologicamente valide, cioe' zeri, poi uni, poi due. E' la proiezione della
predizione rumorosa sullo spazio delle etichette ammissibili.
"""

import numpy as np


def monotone_transitions(preds, num_classes=3):
    """Indici (at, rc) della sequenza 0->1->2 che concorda piu' con `preds`.

    Massimizza il numero di istanti in accordo. Si calcola in tempo lineare:
    detti C0, C1, C2 i conteggi cumulati delle tre classi, l'accordo della
    coppia (a, b) e' C0[a] + (C1[b]-C1[a]) + (C2[T]-C2[b]), che si separa in
    f(a) = C0[a]-C1[a] piu' g(b) = C1[b]-C2[b] piu' una costante. Basta quindi
    scorrere b tenendo il massimo di f fino a b, senza esaminare le T^2 coppie.
    """
    preds = np.asarray(preds)
    T = len(preds)
    cum = np.zeros((num_classes, T + 1), dtype=np.int64)
    for c in range(num_classes):
        cum[c, 1:] = np.cumsum(preds == c)

    f = cum[0] - cum[1]          # conviene mettere il confine AT in a
    g = cum[1] - cum[2]          # conviene mettere il confine RC in b

    best_f = np.maximum.accumulate(f)          # massimo di f per a <= b
    totale = best_f + g
    b = int(np.argmax(totale))
    a = int(np.argmax(f[:b + 1]))
    return a, b


def true_transitions(labels):
    """Indici veri di AT e RC dalla colonna di etichette."""
    labels = np.asarray(labels).astype(int)
    at = np.argmax(labels >= 1) if (labels >= 1).any() else len(labels)
    rc = np.argmax(labels >= 2) if (labels >= 2).any() else len(labels)
    return int(at), int(rc)


def threshold_errors(preds, labels, seconds, watts):
    """Errore firmato su AT e RC, in respiri, secondi e watt.

    Positivo = la soglia predetta arriva dopo quella vera.
    """
    at_pred, rc_pred = monotone_transitions(preds)
    at_true, rc_true = true_transitions(labels)
    out = {}
    for nome, p, t in (('at', at_pred, at_true), ('rc', rc_pred, rc_true)):
        p = min(p, len(seconds) - 1)
        t = min(t, len(seconds) - 1)
        out[nome] = {
            'breaths': p - t,
            'seconds': float(seconds[p] - seconds[t]),
            'watts': float(watts[p] - watts[t]),
            'index_pred': p,
            'index_true': t,
        }
    return out
