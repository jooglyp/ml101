from __future__ import annotations

import copy
import logging
import random
import typing

import numpy
import pandas
from imblearn.under_sampling import RandomUnderSampler, RepeatedEditedNearestNeighbours
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import pca

LOGGER = logging.getLogger(__name__)


class DataPreparer:
    def __init__(self):
        """Load data and apply PCA."""
        self.raw_data = None
        self.cleaned_data = (
            None
        )  # Important: will contain both X and y data in one pandas dataframe!
        self.important_covariates = None
        self.model_covariates = None  # a list of the X's to use in sampling.
        self.X = None  # Model X covariates containing no NaN values
        self.y = None  # Model y dependent variable containing no NaN values
        self.x_random_resampled = None
        self.y_random_resampled = None
        self.x_rnn_resampled = None
        self.y_rnn_resampled = None

    def clientside_pca(self, X: pandas.DataFrame, category_limit: int):
        # Concatenate data
        pca_application = pca.ApplyPCA(X)
        pca_application.coerce_data()
        self.cleaned_data = pca_application.yield_clean_data(
            category_limit, autorestrictions=True
        )
        LOGGER.info(self.cleaned_data)
        LOGGER.info(self.cleaned_data.columns)

        # cleaned_data contains only X here.
        self.important_covariates, self.model_covariates = pca_application.apply_pca(
            self.cleaned_data,
            pca_application.clientside_covariate_exclusion,
        )

    def assignment_pca(self, csv, category_limit: int) -> None:
        """Loads csv into memory as pandas dataframe and applies some transformations."""
        self.raw_data = pandas.read_csv(csv)

        pca_application = pca.ApplyPCA(self.raw_data)
        pca_application.coerce_data()
        self.cleaned_data = pca_application.yield_clean_data(
            category_limit, categorical_restriction=["addr_state", "zip_code"]
        )
        LOGGER.info(self.cleaned_data)
        LOGGER.info(self.cleaned_data.columns)
        exclude_variables = [
            "mths_since_last_delinq",
            "mths_since_last_record",
            *pca_application.yield_categorical_variables,
        ]
        self.important_covariates, self.model_covariates = pca_application.apply_pca(
            self.cleaned_data, exclude_variables
        )  # cleaned_data contains X and y here.

    def secondary_pca(
        self, X: pandas.DataFrame, pca_components: int = 4
    ) -> pandas.DataFrame:
        """
        Second round of PCA for strictly creating components to use in model fitting.
        Args:
            X: explicit declaration of covariates matrix (containing no Id field)

        Returns: PCA transformation of original cleaned X dataframe.

        """
        # PCA Transformation:
        z_scaler = StandardScaler()
        z_data = z_scaler.fit_transform(X)
        z_data_df = pandas.DataFrame(z_data, columns=X.columns)
        pca = PCA(n_components=pca_components)
        component_lables = ["component_" + str(i) for i in range(pca_components)]
        pca_model = pca.fit(z_data)
        pca_ndarray = pca_model.transform(z_data)
        XPCA_df = pandas.DataFrame(pca_ndarray, columns=component_lables)
        LOGGER.info(XPCA_df)
        return XPCA_df

    def resampling(self, neighbors: int, sample_proportion: float, pca_components: int):
        """Take PCA a second time and balance the sample according to proportion of y labels w.r.t. component X's."""
        LOGGER.info(self.X)
        if len(self.X.columns) < pca_components:
            pca_components = round(
                0.5 * len(self.X.columns)
            )  # ensure that data reduction is possible.
        XPCA_df = self.secondary_pca(self.X, pca_components=pca_components)
        # Note that self.X and self.y are untouched but are passed to model fitting for recursive fitting.

        self.x_random_resampled, self.y_random_resampled = self.random_undersampling(
            XPCA_df, self.y, sample_proportion
        )
        LOGGER.info(self.x_random_resampled)
        LOGGER.info(self.y_random_resampled)

        self.x_rnn_resampled, self.y_rnn_resampled = self.rnn_undersampling(
            XPCA_df, self.y, neighbors
        )
        LOGGER.info(self.x_rnn_resampled)
        LOGGER.info(self.y_rnn_resampled)

    def construct_new_covariates_list(
        self,
        remaining_covariates_size: int,
        remaining_covariates: list,
        pca_proportion: float,
    ) -> list:
        new_covariates = []
        secure_random = random.SystemRandom()
        LOGGER.info(pca_proportion)
        LOGGER.info(remaining_covariates_size)
        LOGGER.info(round(pca_proportion * remaining_covariates_size))
        LOGGER.info(remaining_covariates)
        if int(remaining_covariates_size) > 0:
            new_size = round(pca_proportion * remaining_covariates_size)
            for i in range(new_size):
                try:
                    pick = secure_random.choice(list(remaining_covariates))
                    new_covariates.append(pick)
                    list(remaining_covariates).remove(pick)
                except TypeError:
                    LOGGER.info("exited covariate randomization")
                    break
        LOGGER.info(new_covariates)
        return new_covariates

    def randomize_top_covariates(
        self,
        pca_importance: pandas.DataFrame,
        model_covariates: list,
        pca_proportion: float,
    ):
        if pca_importance is not None:
            LOGGER.info("Building a Random Top Covariates Filter")
            LOGGER.info(pca_importance)
            base_covariates = pca_importance[
                0
            ].values  # values of series, which are variable names
            LOGGER.info(base_covariates)
            LOGGER.info(model_covariates)
            remaining_covariates = [
                cov for cov in model_covariates if cov not in base_covariates
            ]
            LOGGER.info(remaining_covariates)
            remaining_covariates_size = len(remaining_covariates)
            new_covariates = self.construct_new_covariates_list(
                remaining_covariates_size, remaining_covariates, pca_proportion
            )
            LOGGER.info(new_covariates)
            new_base_covariates = self.construct_new_covariates_list(
                len(base_covariates), base_covariates, pca_proportion
            )
            LOGGER.info(new_base_covariates)
            self.model_covariates = list(new_base_covariates) + list(new_covariates)
            LOGGER.info(self.model_covariates)
            return self.model_covariates
        else:
            return self.model_covariates

    def sample(
        self,
        y: numpy.ndarray = None,
        X: pandas.DataFrame = None,
        pca_importance: pandas.DataFrame = None,
        model_covariates: list = None,
        neighbors=2,
        sample_proportion=0.9,
        pca_proportion=0.95,
        pca_components=4,
    ):
        """
        self.model_covariates are always a list of the X's.
        Args:
            y: numpy array of dependent variables.
            X: pandas dataframe of covariates.
            pca_importance: pandas dataframe of variables ranked by importance according to pca.
            model_covariates: covariates to use in sampling.
            neighbors: neighbors to use in rnn undersampling.
            sample_proportion: sample proportion to use in random undersampling.
            pca_proportion: subset of top 2/3 pca variables to use in a second subsetting of pca variables.

        Returns: Returns: Resampled dataset ready for model fitting.

        """
        # TODO: unit test to ensure X and y data are the same length
        self.model_covariates = self.randomize_top_covariates(
            pca_importance, model_covariates, pca_proportion
        )
        X, y = self.prepare_data_for_sampling(self.model_covariates, y)
        LOGGER.info(len(X))
        LOGGER.info(len(y))
        self.resampling(neighbors, sample_proportion, pca_components)

    def split_data_for_sampling(
        self, covariates: list
    ) -> typing.Tuple[pandas.DataFrame, numpy.ndarray]:
        """

        Args:
            covariates: X training covariates to use for sampling.
        Returns: Tuple of the matrix of covariates and the matrix of dependent variable, y.

        """
        data = copy.deepcopy(self.cleaned_data[covariates])
        LOGGER.info("Original Dataset Size: {}".format(len(data)))
        LOGGER.info("Dropping row data across model covariates containing NaN values.")
        data.dropna(inplace=True)  # drop rows that contain nan across any covariates
        LOGGER.info(data)
        self.X = data[data.columns.difference(["is_bad"])]
        self.X = self.check_id(self.X)
        LOGGER.info(
            "Splitting data for resampling with the following covariates: {}".format(
                self.X
            )
        )
        self.y = numpy.array(data[["is_bad"]])
        return self.X, self.y

    def prepare_data_for_sampling(
        self, covariates: list, y: numpy.ndarray
    ) -> typing.Tuple[pandas.DataFrame, numpy.ndarray]:
        """

        Args:
            covariates: X training covariates to use for sampling.
        Returns: Tuple of the matrix of covariates and the matrix of dependent variable, y.

        """
        data = copy.deepcopy(self.cleaned_data[covariates])
        LOGGER.info("Original Dataset Size: {}".format(len(data)))
        LOGGER.info("Dropping row data across model covariates containing NaN values.")
        data["actual_y"] = y
        data.dropna(inplace=True)  # drop rows that contain nan across any covariates
        LOGGER.info(data)
        self.X = data[data.columns.difference(["actual_y"])]
        self.X = self.check_id(self.X)
        LOGGER.info(
            "X will include the following covariates: {}".format(self.X.columns)
        )
        self.y = numpy.array(data[["actual_y"]])
        return self.X, self.y

    def check_id(self, df: pandas.DataFrame) -> pandas.DataFrame:
        if "Id" in df.columns:
            adjusted_df = df.drop(["Id"], axis=1)
            LOGGER.info(adjusted_df)
            return adjusted_df
        elif "id" in df.columns:
            adjusted_df = df.drop(["id"], axis=1)
            LOGGER.info(adjusted_df)
            return adjusted_df
        elif "index" in df.columns:
            adjusted_df = df.drop(["index"], axis=1)
            LOGGER.info(adjusted_df)
            return adjusted_df
        else:
            return df

    def rnn_undersampling(
        self, x: pandas.DataFrame, y: numpy.ndarray, neighbors: int
    ) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Repeated Edited Nearest Neighbors.
        Args:
            x: X training covariates for the ML model.
            y: y training binary outcomes of the ML model.

        Returns: resampled (undersampled) observations that reduce bias in the receiving operating characteristic (ROC).

        """
        # TODO: unit test to ensure X and y lengths are the same
        # TODO: unit test to ensure Id is not in x or y
        x = self.check_id(x)
        rnn_undersampler = RepeatedEditedNearestNeighbours(
            random_state=82,
            n_neighbors=neighbors,
            return_indices=True,
            kind_sel="mode",
            max_iter=400,
            ratio="majority",
        )

        X_resampled, y_resampled, resampled_idx = rnn_undersampler.fit_sample(
            copy.deepcopy(x), copy.deepcopy(y)
        )
        LOGGER.info(X_resampled)
        LOGGER.info(
            "RNN undersampling yielded {} number of X_resampled observations".format(
                len(X_resampled)
            )
        )
        LOGGER.info(y_resampled)
        LOGGER.info(
            "RNN undersampling yielded {} number of y_resampled observations".format(
                len(y_resampled)
            )
        )
        return X_resampled, y_resampled

    def random_undersampling(
        self, x: pandas.DataFrame, y: numpy.ndarray, sample_proportion: float = 0.8
    ) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Random Undersampling.
        Args:
            x: X training covariates for the ML model.
            y: y training binary outcomes of the ML model.

        Returns: resampled (undersampled) observations that reduce bias in the receiving operating characteristic (ROC).

        """
        # TODO: unit test to ensure X and y lengths are the same
        # TODO: unit test to ensure Id is not in x or y
        x = self.check_id(x)
        LOGGER.info(len(x))
        target_size = round(sample_proportion * len(x))
        LOGGER.info(target_size)
        random_undersampler = RandomUnderSampler(sampling_strategy=sample_proportion)

        X_resampled, y_resampled = random_undersampler.fit_sample(
            copy.deepcopy(x), copy.deepcopy(y).ravel()
        )
        LOGGER.info(X_resampled)
        LOGGER.info(
            "Random undersampling yielded {} number of X_resampled observations".format(
                len(X_resampled)
            )
        )
        LOGGER.info(y_resampled)
        LOGGER.info(
            "Random undersampling yielded {} number of y_resampled observations".format(
                len(X_resampled)
            )
        )
        return X_resampled, y_resampled
