"""End-to-end pipeline: run every asset-class module and validate everything written to results/."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from ou_model.assets import commodities, equities, fx
from ou_model.core.results import validate_pair_results

ASSET_MODULES = {"equities": equities, "commodities": commodities, "fx": fx}


def run_asset_class(name: str, config_dir: Path, results_root: Path) -> list[Path]:
    """Run one asset-class module and return the result directories it wrote."""
    start = time.perf_counter()
    paths = ASSET_MODULES[name].run(config_dir=config_dir, results_root=results_root)
    logging.info("%s: %d pairs in %.1fs", name, len(paths), time.perf_counter() - start)
    return list(paths)


def validate_all(paths: list[Path]) -> dict[str, list[str]]:
    """Validate every result directory against CONTRACTS.md; returns only the failures."""
    failures = {str(p): validate_pair_results(p) for p in paths}
    return {p: errs for p, errs in failures.items() if errs}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", nargs="+", choices=sorted(ASSET_MODULES), default=list(ASSET_MODULES))
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the selected asset classes, validate the outputs, return a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    paths: list[Path] = []
    for name in args.assets:
        paths += run_asset_class(name, args.config_dir, args.results_root)
    failures = validate_all(paths)
    for path, errs in failures.items():
        logging.error("invalid results in %s: %s", path, errs)
    logging.info("%d pairs written, %d invalid", len(paths), len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
