"""Compatta i file CPET in un unico pickle di DataFrame."""

import logging
from pickle import dump

from respirazione.data.cpet import load_and_convert
from respirazione.paths import DATA_DIR

log = logging.getLogger(__name__)

CPET_FOLDER = DATA_DIR / 'File_CPET'
OUTPUT = DATA_DIR / 'sequence_dataset.pkl'


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log.info("Compact CPET files in a single dataset: %s", CPET_FOLDER)
    out_data = load_and_convert(CPET_FOLDER)
    with open(OUTPUT, 'wb') as f:
        dump(out_data, f)
    log.info("Scritti %d soggetti in %s", len(out_data), OUTPUT)


if __name__ == "__main__":
    main()
