"""PCA for identifying variables with most covariance."""

import logging
import datetime
import itertools
import random
import typing
from decimal import Decimal

import numpy
import pandas
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from . import utils

LOGGER = logging.getLogger(__name__)


class CleanData:
    def __init__(self, raw_data):
        LOGGER.info("Instantiated Data Cleaning Application.")
        self.dataset = raw_data

    def _identify_covariate_types(self) -> dict:
        """
        Identify covariate_types
        Returns: dictionary of covariate types

        """
        # TODO: unit test that there are in fact column.name and column.dtype for each column
        covariate_types = {}
        for column in list(self.dataset):
            covariate_dtype = self.dataset[column].dtype
            if covariate_dtype == numpy.float64 or covariate_dtype == numpy.int64:
                covariate_types[column] = self.dataset[column].dtype.name
            else:
                covariate_types[column] = 'str'
        utils.print_delimiter()
        LOGGER.info(covariate_types)
        return covariate_types

    @staticmethod
    def _identify_categorical_covariates(covariate_types: dict) -> list:
        """

        Args:
            covariate_types: dictionary of covariate names and their data-types.

        Returns: list of covariates that are categorical.

        """
        return [element for element in covariate_types if covariate_types[element] is 'str']

    @staticmethod
    def _identify_numerical_covariates(categoricals: list, covariate_types: dict) -> list:
        """

        Args:
            categoricals: list of categorical variable names.
            covariate_types: dictionary of covariate names and their data-types.

        Returns: list of covariates that are numerical.

        """
        return [element for element in covariate_types if element not in categoricals]


class ApplyPCA(CleanData):
    def __init__(self, raw_data):
        super().__init__(raw_data)
        LOGGER.info("Instantiated PCA Application.")
        self.dataset = raw_data

    @staticmethod
    def _categorical_encoding(vector: pandas.DataFrame) -> pandas.DataFrame:
        """

        Args:
            vector: pandas dataframe of categorical variable to be used as prefix for hot encoding.

        Returns: matrix of integers that correspond to categorical variable.

        """
        prefix = vector.name
        dummy = OneHotEncoder()
        dummy_category = LabelEncoder()
        categories = numpy.zeros((vector.shape[0], 1))
        utils.print_delimiter()
        LOGGER.info(categories)

        categorical_matrix = dummy_category.fit_transform(vector.reshape(-1, 1))
        categorical_matrix = dummy.fit_transform(categorical_matrix.reshape(-1, 1)).toarray()
        categorical_matrix = pandas.DataFrame(categorical_matrix[:, 1:])

        encoded_matrix = pandas.DataFrame(numpy.hstack((categories, categorical_matrix)))
        encoded_matrix.columns = [str(prefix) + str("_") + str(n) for n in list(encoded_matrix.columns)]

        encoded_matrix_df = pandas.DataFrame(encoded_matrix)
        utils.print_delimiter()
        LOGGER.info(encoded_matrix_df)
        return encoded_matrix_df

    def apply_pca(self) -> pandas.DataFrame:
        """
        Identify non-numerical covariates and numerical covariates and apply pca.
        Returns: pandas dataframe

        """
        covariate_types = self._identify_covariate_types()
        categorical_covariates = self._identify_categorical_covariates(covariate_types)
        utils.print_delimiter()
        LOGGER.info(categorical_covariates)
        numerical_covariates = self._identify_numerical_covariates(categorical_covariates, covariate_types)
        utils.print_delimiter()
        LOGGER.info(numerical_covariates)
        return pandas.DataFrame([])
