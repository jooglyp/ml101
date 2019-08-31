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

    dataset.fit()
    utils.print_delimiter()
