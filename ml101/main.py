"""The main entrypoints live here."""
import json
import logging

from . import log, sampler

LOGGER = logging.getLogger(__name__)


def main():
    log.init()

    dataset = sampler.DataPreparer()

    LOGGER.info("Loading data from disk...")

    with open("/tmp/data.json", "r") as fileobj:
        dataset.load(fileobj)
