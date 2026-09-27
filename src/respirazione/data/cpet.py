"""Lettura dei file CPET e costruzione del dataset.

Ogni file Excel ha quattro fogli: `Test` con la serie respiro per respiro, `AT`
e `RC` con gli istanti delle due soglie segnati da un operatore, `Media Rest`
con i valori a riposo usati per normalizzare.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Scala fissa per Load: divide i watt per una costante, quindi cambia solo la
# magnitudine dell'ingresso senza usare informazione del soggetto.
LOAD_SCALE = 100.0


def load_and_convert(cpet_data_folder):
    """Legge tutti gli .xlsx della cartella e restituisce {nome file: DataFrame}.

    Ogni DataFrame ha le colonne normalizzate sul valore a riposo del soggetto e
    una colonna `label` finale con 0 sotto AT, 1 tra AT e RC, 2 sopra RC.
    """
    cpet_data_folder = Path(cpet_data_folder)
    out_data = dict()

    for file_name in sorted(os.listdir(cpet_data_folder)):
        if Path(file_name).suffix != '.xlsx':
            continue
        log.info("%s", file_name)

        xls = pd.ExcelFile(cpet_data_folder / file_name)
        test_df = pd.read_excel(xls, 'Test')
        at_df = pd.read_excel(xls, 'AT')
        rc_df = pd.read_excel(xls, 'RC')
        media_rest_df = pd.read_excel(xls, 'Media Rest')
        for df in (test_df, at_df, rc_df, media_rest_df):
            df.rename(columns={'Power': 'Load'}, inplace=True)

        # Le prime righe contengono le unita' di misura, non dati
        test_df = test_df.drop([0, 1]).reset_index(drop=True)
        media_rest_df = media_rest_df.drop([0]).reset_index(drop=True)

        numeric_cols = test_df.columns.drop('t')
        test_df['t'] = [datetime.combine(datetime.min, t) - datetime.min for t in test_df['t']]
        test_df[numeric_cols] = test_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        media_rest_df[numeric_cols] = media_rest_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        # Normalizzazione sui valori a riposo. Il controllo salta le colonne il
        # cui valore di riposo e' 0 o mancante: su questi dati capita per 'Load'
        # su tutti i file, perche' a riposo non si pedala.
        for col in numeric_cols:
            rest_value = media_rest_df.iloc[0][col]
            if rest_value:
                test_df[col] /= float(rest_value)
            else:
                log.debug("%s: valore a riposo nullo per %r, colonna non normalizzata",
                          file_name, col)

        # Due varianti di Load su scale confrontabili con le altre feature. La
        # rete e' molto sensibile alla magnitudine degli ingressi (0.206 di F1
        # tra Load e Load/100 da sole), quindi la scala va tenuta sotto occhio.
        #   Load_peak   rapporto sul picco del soggetto, come le altre feature
        #               sono rapporti sul suo riposo
        #   Load_scaled stessa forma, solo riscalata di una costante
        peak = test_df['Load'].max()
        test_df['Load_peak'] = test_df['Load'] / peak if peak else test_df['Load']
        test_df['Load_scaled'] = test_df['Load'] / LOAD_SCALE

        if test_df.isna().any(axis=1).sum() != 0:
            log.warning("%s contiene righe con valori mancanti", file_name)

        at_time = datetime.combine(datetime.min, at_df.iloc[1]['t']) - datetime.min
        rc_time = datetime.combine(datetime.min, rc_df.iloc[1]['t']) - datetime.min
        at_index = np.argmin(np.abs(test_df['t'] - at_time))
        rc_index = np.argmin(np.abs(test_df['t'] - rc_time))

        # Il confronto e' stretto, quindi il campione esattamente sulla soglia
        # resta nella fascia precedente.
        steps = np.arange(len(test_df))
        is_at = steps > at_index
        is_rc = steps > rc_index
        test_df['label'] = np.where(is_at, np.where(is_rc, 2, 1), 0)

        out_data[file_name] = test_df

    return out_data
