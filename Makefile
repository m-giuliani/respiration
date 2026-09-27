# Ordine della pipeline: ogni fase produce la configurazione della successiva.
.PHONY: install data features tune train experiment test all clean

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

experiment:
	resp-experiment-load

test:
	pytest

all: features tune train

clean:
	rm -rf outputs runs checkpoints
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
