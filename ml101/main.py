"""The main entrypoints live here."""
import logging

import numpy
import pandas

from . import log, model

LOGGER = logging.getLogger(__name__)


def main():
    log.init()

    # Assignment Simulation:
    LOGGER.info("Loading data from disk...")

    with open("/tmp/data.csv", "r") as fileobj:
        raw_data = pandas.read_csv(fileobj)
        X = raw_data[raw_data.columns.difference(["is_bad"])]
        y = numpy.array(raw_data[["is_bad"]])

    xgboost_model = model.XGBoostModel()

    xgboost_model.evaluate(X, y)
