"""The main entrypoints live here."""
import json
import logging

from . import log, sampler, utils, model

LOGGER = logging.getLogger(__name__)


def main():
    log.init()

    # Assignment Simulation:
    dataset = sampler.DataPreparer()
    LOGGER.info("Loading data from disk...")
    with open("/tmp/data.csv", "r") as fileobj:
        dataset.assignment_pca(fileobj)
    dataset.sample(assignment=True)
    xgboost_model = model.ML101Model(dataset.x_rnn_resampled, dataset.y_rnn_resampled,
                                     dataset.X.columns, 'is_bad')
    xgboost_model.assignment_fit()

    # Client-Side Simulation:
    # client_xgboost_model = model.ExecuteML101Model()
    # client_xgboost_model.fit(X_Pandas, y_ndarray, grid_search=None)

    utils.print_delimiter()
