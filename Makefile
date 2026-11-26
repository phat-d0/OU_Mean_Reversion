PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: venv install test run ui all clean-results

venv:
	python3.11 -m venv .venv

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

test:
	$(PY) -m pytest

run:
	$(PY) run_all.py

ui:
	$(PY) -m streamlit run ou_model/ui/app.py

all: test run

clean-results:
	rm -rf results/equities results/commodities results/fx
