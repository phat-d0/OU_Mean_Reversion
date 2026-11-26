"""``python -m ou_model.assets.commodities`` runs every commodity pair."""

import logging

from .runner import run

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for path in run():
        print(path)
