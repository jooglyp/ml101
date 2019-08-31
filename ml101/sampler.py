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
        self.cleaned_data = None
        self.important_covariates = None

    def load(self, csv) -> None:
        """Loads csv into memory as pandas dataframe and applies some transformations."""
        self.raw_data = pandas.read_csv(csv)

        pca_application = pca.ApplyPCA(self.raw_data)
        pca_application.coerce_data()
        self.cleaned_data = pca_application.yield_clean_data(categorical_restriction=['addr_state', 'zip_code'])
        LOGGER.info(self.cleaned_data)
        LOGGER.info(self.cleaned_data.columns)
        exclude_variables = ['mths_since_last_delinq', 'mths_since_last_record',
                             *pca_application.yield_categorical_variables]
        self.important_covariates = pca_application.apply_pca(self.cleaned_data, exclude_variables)

    def fit(self):
        """
        Fit XGBoost model to undersampled data using K-folds cross-validation and F1-score, LogLoss optimization.
        Returns: Optimal fitted model to be used on out-of-sample data.

        """
        X, y = self.split_data_for_sampling()
        utils.print_delimiter()
        LOGGER.info(X)
        utils.print_delimiter()
        LOGGER.info(y)

    def split_data_for_sampling(self) -> typing.Tuple[pandas.DataFrame, numpy.ndarray]:
        """

        Returns: Tuple of the matrix of covariates and the matrix of dependent variable, y.

        """
        LOGGER.info(self.cleaned_data)
        X = self.cleaned_data[self.cleaned_data.columns.difference(['is_bad'])]
        y = numpy.array(self.cleaned_data[['is_bad']])
        return X, y

    @staticmethod
    def rnn_undersampling(x: pandas.DataFrame, y: pandas.DataFrame) -> typing.Tuple[pandas.DataFrame,
                                                                                    pandas.DataFrame]:
        """
        Repeated Edited Nearest Neighbors.
        Args:
            x: X training covariates for the ML model.
            y: y training binary outcomes of the ML model.

        Returns: resampled (undersampled) observations that reduce bias in the receiving operating characteristic (ROC).

        """
        rnn_undersampler = RepeatedEditedNearestNeighbours(random_state=82, n_neighbors=2, return_indices=True,
                                                           kind_sel='mode', max_iter=400, ratio='majority')

        X_resampled, y_resampled, resampled_idx = rnn_undersampler.fit_sample(copy.deepcopy(x), copy.deepcopy(y))
        LOGGER.info(X_resampled)
        LOGGER.info(y_resampled)
        return X_resampled, y_resampled

    @staticmethod
    def random_undersampling(x: pandas.DataFrame, y: pandas.DataFrame) -> typing.Tuple[pandas.DataFrame,
                                                                                       pandas.DataFrame]:
        """
        Random Undersampling.
        Args:
            x: X training covariates for the ML model.
            y: y training binary outcomes of the ML model.

        Returns: resampled (undersampled) observations that reduce bias in the receiving operating characteristic (ROC).

        """
        random_undersampler = RandomUnderSampler(ratio={1: 1000, 0: 100})

        X_resampled, y_resampled, resampled_idx = random_undersampler.fit_sample(copy.deepcopy(x), copy.deepcopy(y))
        LOGGER.info(X_resampled)
        LOGGER.info(y_resampled)
        return X_resampled, y_resampled
