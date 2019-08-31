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
        self.categorical_covariates = None
        self.numerical_covariates = None

    def _coerce_alphanumeric(self, value):
        """

        Args:
            value: a vector's row value

        Returns: if value is a number, na, or nan, coerce to float64 or np.nan

        """
        try:
            return numpy.float64(value)
        except ValueError:
            return numpy.nan

    @staticmethod
    def _vector_type(all_strings: bool):
        """

        Args:
            all_strings: vector data-type as string evaluated as True or False

        Returns:'str' or 'numerical'

        """
        if bool(all_strings) is False:
            return 'numerical'
        return 'str'

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
                all_strings = self.dataset[column].str.isnumeric().eq(False).all()
                covariate_types[column] = self._vector_type(all_strings)
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

    def _generate_variable_metadata(self):
        """Identify dataframe data-types."""
        covariate_types = self._identify_covariate_types()
        self.categorical_covariates = self._identify_categorical_covariates(covariate_types)
        utils.print_delimiter()
        LOGGER.info("Categorical Covariates:")
        LOGGER.info(self.categorical_covariates)
        self.numerical_covariates = self._identify_numerical_covariates(self.categorical_covariates, covariate_types)
        utils.print_delimiter()
        LOGGER.info("Numerical Covariates:")
        LOGGER.info(self.numerical_covariates)

    def coerce_data(self):
        """Identify dataframe data-types and coerce dataset to well known datatypes."""
        self._generate_variable_metadata()
        for column in list(self.dataset):
            if column in self.numerical_covariates:
                LOGGER.info("Numerical Coercion")
                self.dataset[column] = self.dataset[column].apply(self._coerce_alphanumeric)
            elif column in self.categorical_covariates:
                LOGGER.info("String Coercion")
                self.dataset[column] = self.dataset[column].astype(str)
        utils.print_delimiter()
        LOGGER.info(self.dataset)
        LOGGER.info(self.dataset.dtypes)


class ApplyPCA(CleanData):
    def __init__(self, raw_data):
        super().__init__(raw_data)
        LOGGER.info("Instantiated PCA Application.")
        self.dataset = raw_data

    @staticmethod
    def _categorical_encoding(vector: numpy.ndarray, _name: str) -> pandas.DataFrame:
        """

        Args:
            vector: pandas dataframe of categorical variable to be used as prefix for hot encoding.

        Returns: matrix of integers that correspond to categorical variable.

        """
        LOGGER.info(vector)
        dummy = OneHotEncoder(categories='auto')
        dummy_category = LabelEncoder()
        categories = numpy.zeros((vector.shape[0], 1))
        utils.print_delimiter()
        LOGGER.info(categories)

        categorical_matrix = dummy_category.fit_transform(vector.reshape(-1, 1).ravel())
        categorical_matrix = dummy.fit_transform(categorical_matrix.reshape(-1, 1)).toarray()
        categorical_matrix = pandas.DataFrame(categorical_matrix[:, 1:])

        encoded_matrix = pandas.DataFrame(numpy.hstack((categories, categorical_matrix)))
        encoded_matrix.columns = [str(_name) + str("_") + str(n) for n in list(encoded_matrix.columns)]

        encoded_matrix_df = pandas.DataFrame(encoded_matrix)
        utils.print_delimiter()
        LOGGER.info(encoded_matrix_df)
        return encoded_matrix_df

    def _encode_categoricals(self) -> dict:
        """
        Creates categorical one-hot-encoded matrices for all categorical covariates
        Returns: dictionary of the form {'<categotical covariate>: pandas.DataFrame}

        """
        encoded_categoricals = {}
        for categorical_covariate in self.categorical_covariates:
            encoded_matrix_df = self._categorical_encoding(numpy.array(self.dataset[categorical_covariate]),
                                                           categorical_covariate)
            encoded_categoricals['categorical_covariate'] = encoded_matrix_df
            utils.print_delimiter()
        return encoded_categoricals

    def apply_pca(self) -> pandas.DataFrame:
        """
        Identify non-numerical covariates and numerical covariates and apply pca.
        Returns: pandas dataframe

        """
        return pandas.DataFrame([])
