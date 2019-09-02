"""The main entrypoints live here."""
import logging

import pandas
import numpy

from . import log, sampler, utils, model

LOGGER = logging.getLogger(__name__)


def main():
    log.init()

    # Assignment Simulation:
    dataset = sampler.DataPreparer()
    LOGGER.info("Loading data from disk...")
    #---#
    """
    with open("/tmp/data.csv", "r") as fileobj:
        dataset.assignment_pca(fileobj)
    dataset.sample(assignment=True)
    """

    with open("/tmp/data.csv", "r") as fileobj:
        raw_data = pandas.read_csv(fileobj)
        X = raw_data[raw_data.columns.difference(['is_bad'])]
        y = numpy.array(raw_data[['is_bad']])
        dataset.clientside_pca(X)
    dataset.sample(y=y, neighbors=2, sample_proportion=0.3)

    #---#
    LOGGER.info(dataset.x_rnn_resampled)
    LOGGER.info(dataset.y_rnn_resampled)
    xgboost_model = model.ML101Model(dataset.x_rnn_resampled, dataset.y_rnn_resampled,
                                     dataset.X.columns, 'is_bad', dataset.important_covariates,
                                     dataset.model_covariates, X, y)
    xgboost_model.assignment_fit()

    #---------------------------------------#
    # Client-Side Simulation:
    # client_xgboost_model = model.ExecuteML101Model()
    # client_xgboost_model.fit(X_Pandas, y_ndarray, grid_search=None)

    utils.print_delimiter()
