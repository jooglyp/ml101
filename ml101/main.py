"""The main entrypoints live here."""
import json
import logging

from . import log, sampler, utils

LOGGER = logging.getLogger(__name__)


def main():
    log.init()

    dataset = sampler.DataPreparer()

    LOGGER.info("Loading data from disk...")

    with open("/tmp/data.csv", "r") as fileobj:
        dataset.load(fileobj)
    utils.print_delimiter()
    LOGGER.info(dataset.raw_data)

    dataset.apply_pca()
