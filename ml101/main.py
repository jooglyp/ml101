"""The main entrypoints live here."""
import json
import logging

from . import log, sampler, utils, model

LOGGER = logging.getLogger(__name__)


def main():
    log.init()

    dataset = sampler.DataPreparer()

    LOGGER.info("Loading data from disk...")

    with open("/tmp/data.csv", "r") as fileobj:
        dataset.load(fileobj)

    dataset.fit()

    crossvalidator = model.CrossValidation(dataset.x_rnn_resampled, dataset.y_rnn_resampled,
                                           dataset.X.columns, 'is_bad')
    crossvalidator.kfold_cv()
    utils.print_delimiter()
