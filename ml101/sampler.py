from __future__ import annotations

import logging
import datetime
import itertools
import random
import typing
from decimal import Decimal

import copy
import numpy
import pandas
from imblearn.under_sampling import RepeatedEditedNearestNeighbours, RandomUnderSampler

from . import utils
from . import pca

LOGGER = logging.getLogger(__name__)


class DataPreparer:
    def __init__(self):
        """Load data and apply PCA."""
        self.raw_data = None
        self.cleaned_data = None  # Important: will contain both X and y data in one pandas dataframe!
        self.important_covariates = None
        self.model_covariates = None  # a list of the X's to use in sampling.
        self.X = None  # Model X covariates containing no NaN values
        self.y = None  # Model y dependent variable containing no NaN values
        self.x_random_resampled = None
        self.y_random_resampled = None
        self.x_rnn_resampled = None
        self.y_rnn_resampled = None

    def clientside_pca(self, X: pandas.DataFrame):
        # Concatenate data
        pca_application = pca.ApplyPCA(X)
        pca_application.coerce_data()
        self.cleaned_data = pca_application.yield_clean_data(autorestrictions=True)
        LOGGER.info(self.cleaned_data)
        LOGGER.info(self.cleaned_data.columns)

        # cleaned_data contains only X here.
        self.important_covariates, self.model_covariates = pca_application.apply_pca(
            self.cleaned_data, pca_application.clientside_covariate_exclusion, assignment=False)

    def assignment_pca(self, csv) -> None:
        """Loads csv into memory as pandas dataframe and applies some transformations."""
        self.raw_data = pandas.read_csv(csv)

        pca_application = pca.ApplyPCA(self.raw_data)
        pca_application.coerce_data()
        self.cleaned_data = pca_application.yield_clean_data(categorical_restriction=['addr_state', 'zip_code'])
        LOGGER.info(self.cleaned_data)
        LOGGER.info(self.cleaned_data.columns)
        exclude_variables = ['mths_since_last_delinq', 'mths_since_last_record',
                             *pca_application.yield_categorical_variables]
        self.important_covariates, self.model_covariates = pca_application.apply_pca(
            self.cleaned_data, exclude_variables, assignment=True)  # cleaned_data contains X and y here.

    def resampling(self):
        self.x_random_resampled, self.y_random_resampled = self.random_undersampling(self.X, self.y)
        utils.print_delimiter()
        LOGGER.info(self.x_random_resampled)
        utils.print_delimiter()
        LOGGER.info(self.y_random_resampled)

        self.x_rnn_resampled, self.y_rnn_resampled = self.rnn_undersampling(self.X, self.y)
        utils.print_delimiter()
        LOGGER.info(self.x_rnn_resampled)
        utils.print_delimiter()
        LOGGER.info(self.y_rnn_resampled)

    def sample(self, y: numpy.ndarray = None, assignment=False):
        """
        self.model_covariates are always a list of the X's.
        Returns: Resampled dataset ready for model fitting.

        """
        # TODO: unit test to ensure X and y data are the same length
        if assignment:
            X, y = self.split_data_for_sampling(covariates=self.model_covariates)
            utils.print_delimiter()
            LOGGER.info(len(X))
            utils.print_delimiter()
            LOGGER.info(len(y))
            self.resampling()
        else:
            X, y = self.prepare_data_for_sampling(self.model_covariates, y)
            utils.print_delimiter()
            LOGGER.info(len(X))
            utils.print_delimiter()
            LOGGER.info(len(y))
            self.resampling()

    def split_data_for_sampling(self, covariates: list) -> typing.Tuple[pandas.DataFrame, numpy.ndarray]:
        """

        Args:
            covariates: X training covariates to use for sampling.
        Returns: Tuple of the matrix of covariates and the matrix of dependent variable, y.

        """
        utils.print_delimiter()
        LOGGER.info("Splitting data for resampling with the following covariates: {}".format(covariates))
        data = copy.deepcopy(self.cleaned_data[covariates])
        LOGGER.info("Original Dataset Size: {}".format(len(data)))
        LOGGER.info("Dropping row data across model covariates containing NaN values.")
        data.dropna(inplace=True)  # drop rows that contain nan across any covariates
        LOGGER.info(data)
        self.X = data[data.columns.difference(['is_bad'])]
        self.y = numpy.array(data[['is_bad']])
        return self.X, self.y

    def prepare_data_for_sampling(self, covariates: list,
                                  y: numpy.ndarray) -> typing.Tuple[pandas.DataFrame, numpy.ndarray]:
        """

        Args:
            covariates: X training covariates to use for sampling.
        Returns: Tuple of the matrix of covariates and the matrix of dependent variable, y.

        """
        utils.print_delimiter()
        LOGGER.info("X will include the following covariates: {}".format(covariates))
        data = copy.deepcopy(self.cleaned_data[covariates])
        LOGGER.info("Original Dataset Size: {}".format(len(data)))
        LOGGER.info("Dropping row data across model covariates containing NaN values.")
        data['actual_y'] = y
        data.dropna(inplace=True)  # drop rows that contain nan across any covariates
        LOGGER.info(data)
        self.X = data[data.columns.difference(['actual_y'])]
        self.y = numpy.array(data[['actual_y']])
        return self.X, self.y

    @staticmethod
    def rnn_undersampling(x: pandas.DataFrame, y: numpy.ndarray) -> typing.Tuple[pandas.DataFrame,
                                                                                 pandas.DataFrame]:
        """
        Repeated Edited Nearest Neighbors.
        Args:
            x: X training covariates for the ML model.
            y: y training binary outcomes of the ML model.

        Returns: resampled (undersampled) observations that reduce bias in the receiving operating characteristic (ROC).

        """
        #TODO: unit test to ensure X and y lengths are the same
        rnn_undersampler = RepeatedEditedNearestNeighbours(random_state=82, n_neighbors=4, return_indices=True,
                                                           kind_sel='mode', max_iter=400, ratio='majority')

        X_resampled, y_resampled, resampled_idx = rnn_undersampler.fit_sample(copy.deepcopy(x), copy.deepcopy(y))
        LOGGER.info(X_resampled)
        LOGGER.info("RNN undersampling yielded {} number of X_resampled observations".format(len(X_resampled)))
        LOGGER.info(y_resampled)
        LOGGER.info("RNN undersampling yielded {} number of y_resampled observations".format(len(y_resampled)))
        return X_resampled, y_resampled

    @staticmethod
    def random_undersampling(x: pandas.DataFrame, y: numpy.ndarray) -> typing.Tuple[pandas.DataFrame,
                                                                                    pandas.DataFrame]:
        """
        Random Undersampling.
        Args:
            x: X training covariates for the ML model.
            y: y training binary outcomes of the ML model.

        Returns: resampled (undersampled) observations that reduce bias in the receiving operating characteristic (ROC).

        """
        #TODO: unit test to ensure X and y lengths are the same
        random_undersampler = RandomUnderSampler(ratio={1: 1000, 0: 8000})

        X_resampled, y_resampled = random_undersampler.fit_sample(copy.deepcopy(x), copy.deepcopy(y).ravel())
        LOGGER.info(X_resampled)
        LOGGER.info("Random undersampling yielded {} number of X_resampled observations".format(len(X_resampled)))
        LOGGER.info(y_resampled)
        LOGGER.info("Random undersampling yielded {} number of y_resampled observations".format(len(X_resampled)))
        return X_resampled, y_resampled
