"""PCA for identifying variables with most covariance."""

import logging
import datetime
import itertools
import random
import typing
from decimal import Decimal
import functools

import matplotlib.pyplot as plt
import numpy
import pandas
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

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
        self.categorical_map = {}  # dictionary of tuples

    @staticmethod
    def _categorical_encoding(vector: numpy.ndarray, _name: str) -> typing.Tuple[pandas.DataFrame, numpy.ndarray]:
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
        encoded_matrix.columns = [str(_name) + '_' + str(n) for n in list(encoded_matrix.columns)]

        encoded_matrix_df = pandas.DataFrame(encoded_matrix)
        utils.print_delimiter()
        LOGGER.info(encoded_matrix_df)
        return encoded_matrix_df, categories

    def _encode_categoricals(self, categorical_restriction: list = None) -> dict:
        """
        Creates categorical one-hot-encoded matrices for all categorical covariates
        Returns: dictionary of the form {'<categotical covariate>: pandas.DataFrame}

        """
        encoded_categoricals = {}
        if categorical_restriction:
            self.categorical_covariates = [categorical for categorical in self.categorical_covariates
                                           if categorical not in categorical_restriction]
        for categorical_covariate in self.categorical_covariates:
            encoded_matrix_df, categories = self._categorical_encoding(numpy.array(self.dataset[categorical_covariate]),
                                                                       categorical_covariate)
            encoded_categoricals[categorical_covariate] = encoded_matrix_df
            self.categorical_map[categorical_covariate] = (categories, encoded_matrix_df.columns)
            utils.print_delimiter()
        return encoded_categoricals

    def _concatenate_dataframes(self, dataframes: list) -> pandas.DataFrame:
        """
        All dataframes must be of the same size.
        Args:
            dataframes: a list of dataframes to perform dataframe concatenation on.

        Returns:

        """
        # TODO: unit test that asserts that dataframes are of the same size
        LOGGER.info("Concatenating Dataframes")
        datas = functools.reduce(lambda left, right: pandas.merge(left, right,
                                                                  left_index=True, right_index=True, how='outer'), dataframes)
        LOGGER.info("Final Dataset:")
        return datas

    @property
    def yield_categorical_variables(self) -> list:
        """Return a list of all categorical variable names."""
        covariates = []
        for mapping in self.categorical_map.items():
            covariates.append([*list(mapping[1][1])])
        flattened_list = list(itertools.chain(*covariates))
        return flattened_list

    def yield_clean_data(self, categorical_restriction: list = None) -> pandas.DataFrame:
        """

        Args:
            categorical_restriction: list of variable names that will not be used in generating final dataset.

        Returns: pandas dataframe.

        """
        LOGGER.info("Creating Categorical Covariate Matrices.")
        numerical_dataframe = self.dataset[[column for column in self.dataset.columns
                                            if column in self.numerical_covariates]]
        LOGGER.info("Numerical Covariates Dataframe:")
        LOGGER.info(numerical_dataframe)
        encoded_categoricals = self._encode_categoricals(categorical_restriction)  # dictionary of dataframes
        categorical_dataframes = list(encoded_categoricals.values())
        LOGGER.info("List of Categorical Covariates Dataframes")
        LOGGER.info(categorical_dataframes)
        dataframes = [numerical_dataframe, *categorical_dataframes]
        return self._concatenate_dataframes(dataframes)

    def apply_pca(self, df: pandas.DataFrame, excluded_variables: list) -> pandas.DataFrame:
        """
        Identify non-numerical covariates and numerical covariates and apply pca.

        Args:
            df: pandas dataframe that is ready for pca.
            excluded_variables: variables that should be excluded from pca due to observations missing at random (MAR).

        Returns: pandas dataframe

        """
        df = df.drop(excluded_variables, axis=1)  # drop variables that will not covary much due to MAR
        df.dropna(inplace=True)
        z_scaler = StandardScaler()
        z_data = z_scaler.fit_transform(df)
        LOGGER.info(z_data)
        pca = PCA(n_components=4)
        pca_trafo = pca.fit(z_data)
        LOGGER.info(dir(pca_trafo))
        LOGGER.info(pca_trafo.explained_variance_ratio_)
        plt.semilogy(pca_trafo.explained_variance_ratio_, '--o')
        return pandas.DataFrame([])
