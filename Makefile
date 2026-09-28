# Ordine della pipeline: ogni fase produce la configurazione della successiva.
.PHONY: install data features tune train threshold-error reproduce experiment test all clean

install:
	pip install -e ".[dev]"

data:
	resp-prepare-data

features:
	resp-select-features

tune:
	resp-tune

train:
	resp-train

threshold-error:
	resp-threshold-error

# Il modello finale del README: config baseline, non quelle di resp-tune.
reproduce:
	resp-train model=baseline optimizer=baseline
	resp-threshold-error

experiment:
	resp-experiment-load

test:
	pytest

all: features tune train

clean:
	rm -rf outputs runs checkpoints
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
