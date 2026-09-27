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

| gruppo | F1 medio |
|---|---|
| **respiratory + cardiac + metabolic** | **0.7771** |
| respiratory + metabolic | 0.7626 |
| respiratory + cardiac + metabolic + load | 0.7435 |
| metabolic | 0.7431 |
| cardiac + metabolic + load | 0.7413 |
| metabolic + load | 0.7360 |
| respiratory + metabolic + load | 0.7338 |
| respiratory + load | 0.7247 |
| cardiac + metabolic | 0.7147 |
| respiratory + cardiac + load | 0.7037 |
| load | 0.6877 |
| cardiac + load | 0.6841 |
| respiratory + cardiac | 0.6814 |
| respiratory | 0.5731 |
| cardiac | 0.5574 |

Vince `Rf, VE/VO2, VE/VCO2, HR, VO2/HR, VO2, VCO2`: tutto tranne `Load`.

**`Load` peggiora il modello**, e in modo sistematico: aggiunto al gruppo
vincente porta 0.7771 a 0.7435, e il calo si ripete in ogni coppia con e
senza (`metabolic` 0.7431 → 0.7360, `respiratory_metabolic` 0.7626 → 0.7338).

Il perché però non è ancora stabilito, e ci sono due spiegazioni distinte che
questi esperimenti non separano:

1. **`Load` descrive il protocollo, non il soggetto.** La potenza erogata è
   la rampa impostata dall'operatore. Il modello può impararla a memoria e
   contare i watt invece di guardare la ventilazione, sbagliando su chi ha una
   rampa diversa.
2. **`Load` è l'unica feature non normalizzata.** `load_data.py` divide ogni
   colonna per il suo valore a riposo, ma solo se quel valore è diverso da
   zero. A riposo la potenza erogata è 0 su **tutti gli 82 file**, quindi la
   divisione viene sempre saltata: `Load` entra nella rete in watt grezzi
   (ordine delle centinaia) mentre tutte le altre feature sono rapporti
   intorno a 1. Un input con quella magnitudine può dominare l'ingresso a
   prescindere da cosa contenga.

Per distinguerle basta normalizzare `Load` in un altro modo — sul picco, o
standardizzandola — e rifare il confronto. Finché non è fatto, la conclusione
sicura è solo che `Load` **così come è trattata adesso** va tolta.

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

Baseline (vincitore in validazione), early stopping all'epoca 28 con i pesi
migliori all'epoca 23. Sui **17 soggetti mai visti da nessuna scelta**:

```
macro F1 0.7295    precision 0.7617    recall 0.7202
```

| classe | F1 | precision | recall | campioni |
|---|---|---|---|---|
| sotto AT | 0.858 | 0.850 | 0.866 | 2764 |
| **tra AT e RC** | **0.398** | 0.547 | 0.312 | 1597 |
| sopra RC | 0.933 | 0.888 | 0.982 | 5964 |

Matrice di confusione (righe = vero, colonne = predetto):

```
                predetto
              0     1     2
vero  0    2393   348    23
      1     383   499   715
      2      40    65  5859
```

**Il risultato che conta è il 0.398.** Il modello non riconosce la fascia tra
AT e RC, cioè esattamente la zona per cui esiste il progetto. Dei 1597
respiri veri in quella fascia ne azzecca 499: 715 finiscono sopra RC e 383
sotto AT. La banda viene compressa da entrambi i lati, quindi in pratica il
modello **stima AT in ritardo e RC in anticipo**. La macro F1 di 0.73 è
tenuta in piedi dalle due classi facili, che sono tratti lunghi e omogenei
dove basta seguire l'andamento generale.

Le run sono deterministiche: esecuzioni separate danno gli stessi numeri fino
alla quarta cifra. Il modello addestrato finisce in
`checkpoints/threshold_estimator.pt` (non versionato: e' rigenerabile).

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
- **La normalizzazione salta silenziosamente** le colonne il cui valore di
  riposo è 0 o mancante (`load_data.py`). Su questi dati capita per `Load` su
  tutti gli 82 file, ed è la ragione per cui la conclusione su `Load` resta
  ambigua (vedi sopra). Nessun'altra colonna è interessata, ma il salto
  avviene senza alcun avviso.
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

- `pyproject.toml` e `pip install -e .`, per togliere i tre
  `sys.path.append(... .split('respirazione')[0] ...)` che si rompono se la
  cartella viene rinominata
- `conf/` fuori da `src/`
- gli script usano `print()`, quindi i file `.log` che Hydra crea in
  `outputs/` sono vuoti: tutte le run archiviate non hanno una riga di output
- codice morto: `src/lstm.py`, `src/scripts/lstm_tutorial.py`, le 50 righe
  commentate in testa a `load_data.py`
- `.gitignore` ha `.pkl` invece di `*.pkl`
