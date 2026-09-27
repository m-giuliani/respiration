# respirazione

Stima delle soglie ventilatorie da test CPET con una LSTM.

Un test da sforzo cardiopolmonare (CPET) incrementale attraversa due soglie:
la **soglia anaerobica (AT)** e il **punto di compenso respiratorio (RC)**.
Individuarle è il lavoro che un medico dello sport fa a mano guardando i
grafici. Qui il problema è posto come classificazione per singolo respiro:
data la serie temporale delle variabili respiratorie, etichettare ogni
istante con la fascia in cui si trova.

| classe | significato | quota dei campioni |
|---|---|---|
| 0 | sotto AT | 30.1% |
| 1 | tra AT e RC | 15.5% |
| 2 | sopra RC | 54.4% |

## I dati

82 file Excel in `data/File_CPET/`, uno per soggetto, esportati dal
metabolimetro. Ogni file ha quattro fogli: `Test` (la serie respiro per
respiro), `AT` e `RC` (gli istanti delle due soglie, segnati da un operatore),
`Media Rest` (i valori a riposo, usati per normalizzare).

- 43199 respiri in totale
- sequenze da 80 a 1090 passi, mediana 508
- ogni file è un soggetto diverso, quindi una sequenza equivale a un soggetto

`src/scripts/load_data.py` li compatta in `data/sequence_dataset.pkl`,
normalizzando ogni colonna sul rispettivo valore di riposo e ricavando le
etichette dagli istanti di AT e RC.

## La pipeline

Le tre fasi vanno eseguite in quest'ordine, perché ognuna produce la
configurazione della successiva.

```
load_data.py      xlsx -> sequence_dataset.pkl
      |
feature_lstm.py   confronta i 15 gruppi di feature in 5-fold CV
      |           -> hyperparams/features/selected.yaml
tune_lstm.py      Optuna sul gruppo vincente, 5-fold CV
      |           -> hyperparams/model/tuned.yaml
      |           -> hyperparams/optimizer/tuned.yaml
train_lstm.py     addestramento finale e misura sul test set
                  -> test_results.json, checkpoints/threshold_estimator.pt
```

I file generati sono normali config group di Hydra, versionati nel repo e
selezionati nei `defaults` di `config.yaml`. Sono scritti dagli script e non
vanno modificati a mano: ricopiare a mano i risultati del tuning in
`config.yaml` e' il passaggio che in passato ha prodotto override incoerenti
con i file di gruppo. Essendo group e non override globali, una scelta
esplicita da riga di comando continua a vincere su di loro.

`batch_size` sta nel gruppo `optimizer` insieme a `lr` e `weight_decay`:
e' un iperparametro di ottimizzazione, e il valore trovato dal tuning ha
bisogno di un posto dove essere scritto.

### Suddivisione dei dati

`src/splits.py` tiene da parte il **20% dei soggetti (17) come test set** e
divide i 65 rimanenti in 5 fold. Feature selection, ricerca degli
iperparametri ed early stopping lavorano solo sui fold. Il test set viene
letto una volta sola, alla fine.

Questo è necessario, non pedanteria: con 13 soggetti per fold l'F1 oscilla
fino a 0.16 tra un fold e l'altro. Un singolo split poteva restituire 0.86
o 0.70 sullo stesso modello a seconda di come cadeva.

## Come si lancia

Dipendenze (verificate con torch 2.14, hydra 1.3.7, optuna 5.0, pandas 2.3,
scikit-learn 1.7):

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch pandas numpy scikit-learn hydra-core omegaconf optuna tqdm openpyxl tensorboard pytest
```

Dalla radice del repo:

```bash
python src/scripts/load_data.py         # solo se il pickle va rigenerato
python src/scripts/feature_lstm.py
python src/scripts/tune_lstm.py
python src/scripts/train_lstm.py
pytest src/test
tensorboard --logdir runs
```

Tutto è configurato in `src/scripts/hyperparams/config.yaml` e sovrascrivibile
da riga di comando, come da convenzione Hydra:

```bash
python src/scripts/train_lstm.py optimizer=sgd model=tenet_lstm epochs_final=10
python src/scripts/train_lstm.py features=columns_all optimizer.batch_size=8
python src/scripts/tune_lstm.py n_trials=100 search.hidden_max=128
```

Per partire prima che il tuning abbia mai girato, o per ignorare i risultati
salvati: `features=columns_all model=tenet_lstm optimizer=adam`.

## Risultati

### Quali feature servono

15 combinazioni dei quattro gruppi, 5-fold CV, stesso modello baseline per
tutte così il confronto misura le feature e non gli iperparametri.

| gruppo | F1 medio | n. feature |
|---|---|---|
| **respiratory + cardiac + metabolic + load** | **0.7924** | 8 |
| respiratory + metabolic + load | 0.7898 | 6 |
| respiratory + metabolic | 0.7861 | 5 |
| respiratory + cardiac + metabolic | 0.7844 | 7 |
| cardiac + metabolic + load | 0.7573 | 5 |
| cardiac + metabolic | 0.7474 | 4 |
| metabolic | 0.7471 | 2 |
| metabolic + load | 0.7396 | 3 |
| respiratory + cardiac + load | 0.7113 | 6 |
| respiratory + cardiac | 0.7065 | 5 |
| cardiac + load | 0.6864 | 3 |
| respiratory + load | 0.6014 | 4 |
| respiratory | 0.5964 | 3 |
| cardiac | 0.5500 | 2 |
| load | 0.5238 | 1 |

Il numero più alto è del gruppo completo, ma **i primi quattro sono a pari
merito**: da 0.7924 a 0.7844 sono 0.008 di scarto, dentro un rumore tra fold
che su questi dati vale 0.08-0.09. Dire che il gruppo completo batte gli altri
tre non è supportato dai dati.

Sulla parsimonia: `respiratory + metabolic` fa 0.7861 con cinque feature
(`Rf, VE/VO2, VE/VCO2, VO2, VCO2`), indistinguibile dal gruppo completo a
otto. E `metabolic` da solo, cioè **`VO2` e `VCO2` e nient'altro**, fa 0.7471:
perde 0.045 rispetto a otto feature usandone due. Quasi tutto il segnale sta
nello scambio dei gas; il resto aggiunge poco.

Il gruppo `load` **da solo** è il peggiore di tutti (0.5238): la potenza
erogata rapportata al picco del soggetto, da sola, non basta a collocare le
soglie.

### `Load`, e un problema più grande: la rete segue le scale

`load_data.py` normalizza ogni colonna sul suo valore a riposo, ma solo se
quel valore è diverso da zero. A riposo la potenza erogata è 0 su **tutti gli
82 file**, quindi la divisione viene sempre saltata: `Load` entrava nella rete
in watt grezzi, media 66 e deviazione 76, mentre ogni altra feature è un
rapporto con media tra 0.96 e 4.55.

`experiment_load.py` misura tre forme della stessa colonna a parità di fold,
epoche e modello. `Load / 100` è l'esperimento chiave: ha la **forma identica**
ai watt grezzi, quindi contiene esattamente la stessa informazione, e cambia
solo la magnitudine.

Aggiunta alle sette feature fisiologiche:

| variante | F1 medio | scarto |
|---|---|---|
| senza `Load` | 0.7844 | — |
| `Load` grezza (watt) | 0.7677 | −0.0167 |
| `Load` / picco del soggetto | 0.7924 | +0.0080 |
| `Load` / 100 (costante) | 0.7847 | +0.0003 |

Da sola, come unica feature:

| variante | F1 medio | per fold |
|---|---|---|
| `Load` grezza (watt) | 0.6844 | 0.716 0.701 0.699 0.690 0.617 |
| `Load` / picco del soggetto | 0.5238 | 0.353 0.589 0.580 0.512 0.585 |
| `Load` / 100 (costante) | 0.4783 | 0.390 0.451 0.627 0.389 0.534 |

Due conclusioni, la seconda più importante della prima.

**Sul carico.** In combinazione, i watt grezzi peggiorano il modello e la
stessa colonna riscalata non lo fa: il danno era la scala. Quindi l'ipotesi
che `Load` peggiorasse il modello perché descrive il protocollo invece della
risposta del soggetto **non è supportata**. Anzi, da sola la potenza in watt
assoluti fa 0.6844, più di qualunque altra forma: le soglie cadono a carichi
assoluti abbastanza riproducibili tra soggetti, e rapportare al picco di
ciascuno (`Load_peak`) cancella quell'informazione.

**Sulla rete.** `Load` grezza e `Load / 100` contengono la stessa
informazione, e da sole danno 0.6844 contro 0.4783: **0.206 di divario per una
semplice divisione per 100**, con i fold che non si sovrappongono. Il modello
non sta rispondendo al contenuto delle feature ma alla loro magnitudine. Il
pipeline non standardizza niente: ogni colonna finisce nella rete con la scala
che le capita dal rapporto sul riposo, e quelle scale sono molto diverse fra
loro (`VO2` ha deviazione 2.44, `VE/VCO2` 0.17).

Questo è un confondente su **tutta** la tabella dei gruppi qui sopra: parte di
quei confronti misura la fortuna di scala delle colonne, non il loro contenuto
informativo. Il rimedio è standardizzare tutte le feature sulle statistiche
del dev set e rifare il confronto. Finché non è fatto, il ranking dei gruppi
va letto come indicativo.

### Quali iperparametri servono

40 trial Optuna (10 potati) sulle sette feature selezionate, 5-fold CV.
Migliore configurazione trovata: `hidden 151, 1 layer, adam lr 2.09e-3,
weight decay 4.31e-6, batch 32`.

Confrontata a parità di fold ed epoche con il baseline scelto a mano:

| configurazione | F1 medio | per fold |
|---|---|---|
| baseline (hidden 128, 2 layer, dropout 0.2, lr 1e-3, batch 16) | **0.7844** | 0.832 0.767 0.824 0.753 0.746 |
| migliore di Optuna | 0.7792 | 0.807 0.742 0.825 0.759 0.763 |

**La ricerca non ha trovato niente di meglio.** Lo scarto è 0.005 e le due
configurazioni si scambiano i fold a vicenda: è rumore. Il problema è
sensibile alle feature — 0.557 a 0.784 tra il gruppo peggiore e il migliore —
e insensibile agli iperparametri.

### Il modello finale

Configurazione baseline (vincitrice in validazione), otto feature del gruppo
selezionato, early stopping all'epoca 15 con i pesi migliori all'epoca 10.
Sui **17 soggetti mai visti da nessuna scelta**:

```
macro F1 0.7260    precision 0.7305    recall 0.7224
```

| classe | F1 | precision | recall | campioni |
|---|---|---|---|---|
| sotto AT | 0.825 | 0.828 | 0.822 | 2764 |
| **tra AT e RC** | **0.440** | 0.464 | 0.420 | 1597 |
| sopra RC | 0.912 | 0.900 | 0.926 | 5964 |

Matrice di confusione (righe = vero, colonne = predetto):

```
                predetto
              0     1     2
vero  0    2272   489     3
      1     314   670   613
      2     157   286  5521
```

**Il risultato che conta è il 0.440.** Il modello riconosce male la fascia tra
AT e RC, cioè esattamente la zona per cui esiste il progetto. Dei 1597 respiri
veri in quella fascia ne azzecca 670: 613 finiscono sopra RC e 314 sotto AT.
La banda viene compressa da entrambi i lati, quindi in pratica il modello
**stima AT in ritardo e RC in anticipo**. La macro F1 di 0.73 è tenuta in
piedi dalle due classi facili, che sono tratti lunghi e omogenei dove basta
seguire l'andamento generale.

Per confronto, lo stesso modello addestrato sulle sette feature senza carico
(il gruppo che vinceva prima che `Load` venisse normalizzata) dà macro F1
0.7295 ma **0.398 sulla fascia centrale**: leggermente meglio in media,
peggio sulla classe che interessa. Le due macro F1 differiscono di 0.0035 su
una misura singola a 17 soggetti, quindi quella differenza non significa
nulla; il divario di 0.042 sulla classe centrale è più sostanzioso ma resta
una misura sola.

Va dichiarato che il test set è stato letto **due volte**, una per ciascuno
dei due feature set. Nessuna delle due letture ha influenzato una scelta —
il gruppo di feature è cambiato perché è cambiata la normalizzazione di
`Load`, decisa in cross validation — ma due letture sono due letture, e un
terzo giro andrebbe fatto su dati nuovi.

Le run sono deterministiche: esecuzioni separate danno gli stessi numeri fino
alla quarta cifra. Il modello addestrato finisce in
`checkpoints/threshold_estimator.pt` (non versionato: è rigenerabile).

## Limiti, cioè cosa questi numeri non dimostrano

- **Manca il confronto col metodo clinico.** La domanda vera non è se la F1
  per respiro è 0.73, è se questo modello si avvicina alle soglie meglio o
  peggio del V-slope e degli equivalenti ventilatori che si usano oggi.
  Senza quel confronto non si sa se 0.73 è un buon risultato.
- **La metrica è un surrogato.** A nessuno interessa classificare i singoli
  respiri: interessa di quanti secondi o di quanti watt sbaglia la soglia
  stimata. Quell'errore non è ancora misurato, ed è la metrica da aggiungere.
- **Il test set è un campione di 17 soggetti.** L'intervallo di confidenza
  attorno a 0.7295 è largo. La media in CV (0.784) è più stabile ma è
  calcolata su dati che hanno partecipato alle scelte.
- **Le etichette sono di un solo operatore** e sono trattate come verità. Non
  c'è una stima della variabilità tra operatori, che nella lettura manuale
  delle soglie non è trascurabile.
- **Nessuna standardizzazione delle feature**, e il modello è dimostrabilmente
  sensibile alla scala (0.206 di F1 per una divisione per 100, vedi sopra).
  Questo confonde in parte tutta la tabella dei gruppi di feature. È la cosa
  più importante da sistemare.
- **La normalizzazione salta silenziosamente** le colonne il cui valore di
  riposo è 0 o mancante (`load_data.py`). Su questi dati capita per `Load` su
  tutti gli 82 file, senza alcun avviso.
- **`data/*.xlsx`** contiene 8 file fuori da `File_CPET/`, tre dei quali
  (`Id_10`, `Id_58`, `Id_70`) non sono nel dataset. Non è documentato perché
  siano esclusi.

## Struttura

```
config/definitions.py        ROOT_DIR
src/
  splits.py                 test set + fold di cross validation
  training.py               loop, metriche, cross_validate, batching per lunghezza
  threshold_estimator.py    l'LSTM
  timeseriesdataset.py      Dataset e collate
  lstm.py                   LSTM scritta a mano, incompleta, non usata
  scripts/
    load_data.py            xlsx -> pickle
    feature_lstm.py         feature selection
    tune_lstm.py            Optuna
    train_lstm.py           addestramento finale + test
    lstm_tutorial.py        tutorial PyTorch, non usato
    hyperparams/            config Hydra (dataset, features, model, optimizer)
  test/                     test del Dataset
data/                       xlsx e pickle
```

Le sequenze di lunghezza simile finiscono nello stesso batch
(`LengthBucketBatchSampler`): con i batch casuali il 40% del calcolo finiva
su padding, ora il 18%.

### Da sistemare

- **standardizzare le feature** sulle statistiche del dev set, e rifare il
  confronto tra gruppi: è la modifica che cambierebbe di più i risultati
- `pyproject.toml` e `pip install -e .`, per togliere i tre
  `sys.path.append(... .split('respirazione')[0] ...)` che si rompono se la
  cartella viene rinominata
- `conf/` fuori da `src/`
- gli script usano `print()`, quindi i file `.log` che Hydra crea in
  `outputs/` sono vuoti: tutte le run archiviate non hanno una riga di output
- codice morto: `src/lstm.py`, `src/scripts/lstm_tutorial.py`, le 50 righe
  commentate in testa a `load_data.py`
- `.gitignore` ha `.pkl` invece di `*.pkl`
