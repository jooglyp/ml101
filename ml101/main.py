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

    LOGGER.info(xgboost_model.evaluate(X, y))
    # Assuming X is a pandas dataframe...
    # TODO: Make a fake X that might match the client side input
    # TODO: Test the entire api. Call predict_proba and predict with a fake X
    # TODO: Make note that dask is not talking to local threads very well. Not an actual error.
    LOGGER.info(
        xgboost_model.predict_proba(pandas.DataFrame(xgboost_model.model.X.compute()))
    )
