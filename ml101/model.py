"""Model: Extreme Gradient Boosting Regressor wrapped in a K-Fold crossvalidator."""

import itertools
import logging
from statistics import mean

import dask
import dask.array as da
import numpy
import pandas
from dask.distributed import Client
from dask_ml.model_selection import KFold
from dask_ml.xgboost import XGBClassifier
from sklearn.externals import joblib
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_squared_error,
)
from sklearn.preprocessing import MinMaxScaler

from . import sampler

LOGGER = logging.getLogger(__name__)
CLIENT = dask.distributed.Client()


class ML101Model:
    def __init__(
        self,
        X: numpy.ndarray,
        y: numpy.ndarray,
        ylabel: str,
        important_covariates: pandas.DataFrame,
        model_covariates: list,
        original_Xdf: pandas.DataFrame,
        original_yarray: numpy.ndarray,
    ):
        self.X = da.from_array(X, chunks=X.shape)
        self.y = da.from_array(y, chunks=y.shape)
        self.ylabel = ylabel
        self.important_covariates = important_covariates
        self.model_covariates = model_covariates
        self.original_Xdf = original_Xdf
        self.original_yarray = original_yarray
        self.ytest_iterations = []  # list to store kfold crossvalidation results
        self.ypred_iterations = []  # list to store kfold crossvalidation results
        self.predicted_probability_iterations = (
            []
        )  # list to store predicted probabilities
        self.predictions = []  # list to store label predictions
        LOGGER.info(self.X)
        LOGGER.info(self.y)

    def scale_data(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_scaled = scaler.fit_transform(self.X)
        return x_scaled

    def kfold_cv(self):
        """
        K-fold crossvalidator.
        Returns: fitted values and test values to be used for model optimization.

        """
        with joblib.parallel_backend("dask"):
            self.xgb_est = XGBClassifier(
                max_depth=5,
                subsample=0.7,
                scale_pos_weight=2,
                num_class=1,
                learning_rate=0.05,
            )
            cv = KFold(n_splits=8, random_state=24, shuffle=True)
            for train_index, test_index in cv.split(self.X):
                X_train, X_test, y_train, y_test = (
                    self.X[train_index],
                    self.X[test_index],
                    self.y[train_index],
                    self.y[test_index],
                )
                self.xgb_est.fit(X_train, y_train)
                y_pred = self.xgb_est.predict(X_test)
                self.predictions.append(y_pred)
                self.ypred_iterations.append(y_pred)
                self.ytest_iterations.append(y_test)
                self.predicted_probability_iterations.append(
                    self.xgb_est.predict_proba(X_test)
                )

    def predict(self, X: pandas.DataFrame) -> numpy.ndarray:
        return self.xgb_est.predict(X)

    def predict_probability(self, X: pandas.DataFrame) -> numpy.ndarray:
        numpy_array = numpy.array(X)
        dask_array = da.from_array(numpy_array, chunks=numpy_array.shape)
        return self.xgb_est.predict_proba(dask_array).compute()


class Evaluators:
    """Evaluates model accuracy using a confusion matrix, precision/recall (f-1 score), and log-loss."""

    def __init__(self, mlmodel: ML101Model):
        self.mlmodel = mlmodel
        self.ytest_iterations = self.mlmodel.ytest_iterations
        self.ypred_iterations = self.mlmodel.ypred_iterations
        self.confusion_matrices = []
        self.loglosses = []
        self.rmses = []
        self.f1scores = []

    def compute_rmse(self) -> float:
        """RMSE Evaluation. Lower is better."""
        for ytest, ypred in zip(self.ytest_iterations, self.ypred_iterations):
            ytest = ytest.compute()
            ypred = ypred.compute()

            mse = mean_squared_error(ytest, ypred)
            self.rmses.append(numpy.sqrt(mse))
        LOGGER.info(self.rmses)
        return mean(self.rmses)

    def compute_f1score(self) -> float:
        """RMSE Evaluation. Lower is better."""
        for ytest, ypred in zip(self.ytest_iterations, self.ypred_iterations):
            ytest = ytest.compute()
            ypred = ypred.compute()

            f1score = f1_score(ytest, ypred)
            self.f1scores.append(f1score)
        LOGGER.info(self.f1scores)
        return mean(self.f1scores)

    def compute_confusion_matrices(self):
        """
        Compute true-negative, false-positive, false-negative, true-positive from accuracies; yield confusion-matrix.
        Accuracy essentially fit + score and is interpreted as:
        >>> y_pred = [0, 2, 1, 3,0]
        >>> y_true = [0, 1, 2, 3,0]
        >>> print(accuracy_score(y_true, y_pred))
        >>> 0.6
        Returns: a list of confusion matrix pandas DataFrames as a class attribute.

        """
        for ytest, ypred in zip(self.ytest_iterations, self.ypred_iterations):
            ytest = ytest.compute()
            ypred = ypred.compute()

            insample_accuracy = accuracy_score(ypred, ytest)
            tn, fp, fn, tp = confusion_matrix(ypred, ytest).ravel()
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            LOGGER.info(
                "Accuracy: {}, Precision: {}, Recall: {}".format(
                    round(insample_accuracy, 2), precision, recall
                )
            )
            self.confusion_matrices.append(
                pandas.DataFrame(confusion_matrix(ypred, ytest))
            )

    def compute_conditional_log_loss(self) -> float:
        """
        LogLoss Evaluation. Lower is better.
        binary-logistic' uses -(y*log(y_pred) + (y-1)*(log(1-y_pred))
        """
        for ytest, ypred in zip(self.ytest_iterations, self.ypred_iterations):
            ytest = ytest.compute()
            ypred = ypred.compute()

            logloss = log_loss(ytest, ypred, normalize=True)
            self.loglosses.append(logloss)
        LOGGER.info(self.loglosses)
        return mean(self.loglosses)


class ParameterOptimizer(Evaluators):
    """Evaluates model kfold crossvalidations to obtain optimized model parameters for best fit."""

    # TODO: use normalized mutual information score to evaluate how good the sampling was. Adjust sampling accordingly.

    def tune(
        self,
        grid_neighbors: int,
        grid_sample_proportion: float,
        category_limit: int,
        pca_proportion: float,
        pca_components: int,
    ):
        self.compute_confusion_matrices()
        avg_rmse = self.compute_rmse()
        avg_logloss = self.compute_conditional_log_loss()
        avg_f1score = self.compute_f1score()

        important_covariates = self.mlmodel.important_covariates
        last_model_covariates = self.mlmodel.model_covariates
        dataset = sampler.DataPreparer()
        dataset.clientside_pca(self.mlmodel.original_Xdf, category_limit=category_limit)
        dataset.sample(
            self.mlmodel.original_yarray,
            neighbors=grid_neighbors,
            sample_proportion=grid_sample_proportion,
            pca_importance=important_covariates,
            model_covariates=last_model_covariates,
            pca_proportion=pca_proportion,
            pca_components=pca_components,
        )
        mlmodel = self.mlmodel.__class__(
            dataset.x_rnn_resampled,
            dataset.y_rnn_resampled,
            dataset.X.columns,
            dataset.important_covariates,
            dataset.model_covariates,
            self.mlmodel.original_Xdf,
            self.mlmodel.original_yarray,
        )
        mlmodel.kfold_cv()
        optimizer = self.__class__(mlmodel)
        optimizer.compute_confusion_matrices()
        average_rmse = optimizer.compute_rmse()
        average_log_loss = optimizer.compute_conditional_log_loss()
        average_f1 = optimizer.compute_f1score()

        LOGGER.info("Change in Average RMSE: {}".format(average_rmse - avg_rmse))
        LOGGER.info(
            "Change in Average Log Loss: {}".format(average_log_loss - avg_logloss)
        )
        LOGGER.info("Change in Average F1 Score: {}".format(average_f1 - avg_f1score))

        return self.calculate_composite_loss(average_log_loss, average_rmse, average_f1)

    def calculate_composite_loss(
        self, average_log_loss: float, average_rmse: float, average_f1: float
    ) -> float:
        index_ll_rmse = average_log_loss * (1 + average_rmse)
        index_f1 = 1 - average_f1
        composite = index_ll_rmse * (1 + index_f1)
        return composite

    def evaluate(self) -> dict:
        """Return {'f1_score': 0.3, 'logloss': 0.7}"""
        return {
            "f1_score": self.compute_f1score(),
            "logloss": self.compute_conditional_log_loss(),
        }


class XGBoostModel:
    def __init__(self):
        self.original_model: ML101Model = None
        self.model: ML101Model = None
        self.optimiser: ParameterOptimizer = None

    @staticmethod
    def coerce_client_data(X: pandas.DataFrame) -> pandas.DataFrame:
        dataset = sampler.DataPreparer()
        dataset.clientside_pca(X, category_limit=50)
        y = numpy.random.choice([0, 1], size=(len(X),), p=[1.0 / 3, 2.0 / 3])
        dataset.sample(y)
        #TODO: This was a short-hand way to obtain X; dataset.sample() should be refactored not to require y.
        return X

    def fit(self, X: pandas.DataFrame, y: numpy.ndarray) -> None:
        """

        Args:
            X: pandas dataframe of model covariates
            y: numpy array of dependent y variable.
            grid_search: parameters to fine tune model via sampling and pca.

        Returns:

        """
        kwargs = self.tune_parameters(X, y)

        dataset = sampler.DataPreparer()
        dataset.clientside_pca(X, category_limit=kwargs.pop("category_limit"))
        dataset.sample(
            y,
            neighbors=kwargs.pop("grid_neighbors"),
            sample_proportion=kwargs.pop("grid_sample_proportion"),
            pca_importance=self.original_model.important_covariates,
            model_covariates=self.original_model.model_covariates,
            pca_proportion=kwargs.pop("pca_proportion"),
            pca_components=kwargs.pop("pca_components"),
        )

        self.model = ML101Model(
            dataset.x_rnn_resampled,
            dataset.y_rnn_resampled,
            dataset.X.columns,
            dataset.important_covariates,
            dataset.model_covariates,
            self.original_model.original_Xdf,
            self.original_model.original_yarray,
        )
        self.model.kfold_cv()

        self.optimiser = ParameterOptimizer(self.model)

    def predict(self, X: pandas.DataFrame) -> numpy.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pandas.DataFrame) -> numpy.ndarray:
        X = self.coerce_client_data(X)
        return self.model.predict_probability(X)

    def evaluate(self, X: pandas.DataFrame, y: numpy.ndarray) -> dict:
        self.fit(X, y)
        return self.optimiser.evaluate()

    def tune_parameters(self, X: pandas.DataFrame, y: numpy.ndarray) -> dict:
        """
        Examples:
        >>> param_grid = [
        >>>     ("grid_neighbors", [2]),
        >>>     ("grid_sample_proportion", [0.9]),
        >>>     ("category_limit", [10]),
        >>>     ("pca_proportion", [0.95]),
        >>>     ("pca_components", [4]),
        >>> ]
        >>> param_grid = [
        >>>     ("grid_neighbors", [2, 3, 4]),
        >>>     ("grid_sample_proportion", [0.9, 0.7, 0.5]),
        >>>     ("category_limit", [10, 100, 300]),
        >>>     ("pca_proportion", [0.95, 0.9, 0.8]),
        >>>     ("pca_components", [4, 5, 6]),
        >>> ]
        """
        param_grid = [
            ("grid_neighbors", [2, 3, 4]),
            ("grid_sample_proportion", [0.9, 0.7, 0.5]),
            ("category_limit", [10, 100, 300]),
            ("pca_proportion", [0.95, 0.9, 0.8]),
            ("pca_components", [4, 5, 6]),
        ]

        dataset = sampler.DataPreparer()
        dataset.clientside_pca(X, category_limit=50)
        dataset.sample(y)

        self.original_model = ML101Model(
            dataset.x_rnn_resampled,
            dataset.y_rnn_resampled,
            dataset.X.columns,
            dataset.important_covariates,
            dataset.model_covariates,
            X,
            y,
        )
        self.original_model.kfold_cv()

        optimizer = ParameterOptimizer(self.original_model)

        best_loss = None
        best_parameters = None

        for args in itertools.product(*[item[1] for item in param_grid]):
            kwargs = {}
            for index, arg in enumerate(args):
                kwargs[param_grid[index][0]] = arg
            loss = optimizer.tune(**kwargs)

            if best_loss is None or loss < best_loss:
                best_parameters = kwargs
                best_loss = loss

        LOGGER.info("Best parameters: %s", best_parameters)
        return best_parameters
