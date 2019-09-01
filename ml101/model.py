"""Model: Extreme Gradient Boosting Regressor wrapped in a K-Fold crossvalidator."""

import logging

import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error, f1_score, precision_score
from sklearn.metrics import recall_score, average_precision_score, normalized_mutual_info_score, log_loss
from dask.distributed import Client
import dask
import dask.dataframe as dd
import dask.array as da
from dask_ml.xgboost import XGBClassifier
from dask_ml.model_selection import KFold
import numpy

from .pca import ApplyPCA
from .sampler import DataPreparer
from . import utils, sampler

LOGGER = logging.getLogger(__name__)
CLIENT = dask.distributed.Client()


class ML101Model:
    def __init__(self, X: numpy.ndarray, y: numpy.ndarray, xlabels: list, ylabel: str):
        self.X = da.from_array(X, chunks=X.shape)
        self.y = da.from_array(y, chunks=y.shape)
        self.xlabels = xlabels
        self.ylabel = ylabel
        self.ytest_iterations = []  # list to store kfold crossvalidation results
        self.ypred_iterations = []  # list to store kfold crossvalidation results
        self.predicted_probability_iterations = []  # list to store predicted probabilities
        self.predictions = []  # list to store label predictions
        LOGGER.info(self.X)
        LOGGER.info(self.y)

    def scale_data(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_scaled = scaler.fit_transform(self.X)
        return x_scaled

    @staticmethod
    def dask_transform(X: pandas.DataFrame) -> dask.dataframe.DataFrame:
        return dd.from_pandas(X, npartitions=1)

    def kfold_cv(self):
        """
        K-fold crossvalidator.
        Returns: fitted values and test values to be used for model optimization.

        """
        with joblib.parallel_backend('dask'):
            xgb_est = XGBClassifier()
            cv = KFold(n_splits=10, random_state=84, shuffle=True)
            for train_index, test_index in cv.split(self.X):
                X_train, X_test, y_train, y_test = self.X[train_index], self.X[test_index], \
                                                   self.y[train_index], self.y[test_index]
                xgb_est.fit(X_train, y_train)
                y_pred = xgb_est.predict(X_test)
                self.predictions.append(y_pred)
                self.ypred_iterations.append(y_pred)
                self.ytest_iterations.append(y_test)
                self.predicted_probability_iterations.append(xgb_est.predict_proba(X_test))
                LOGGER.info(self.predictions)

    def assignment_fit(self):
        self.kfold_cv()


class ExecuteML101Model:
    def fit(self, X: pandas.DataFrame, y: numpy.ndarray, grid_search=None):
        """

        Args:
            X: pandas dataframe of model covariates
            y: numpy array of dependent y variable.
            grid_search: parameters to fine tune model via sampling and pca.

        Returns:

        """
        #TODO: calls optimizer to get best xgb_est.
        dataset = sampler.DataPreparer()
        dataset.clientside_pca(X)
        dataset.sample(y)
        mlmodel = ML101Model(dataset.x_rnn_resampled, dataset.y_rnn_resampled,
                                     dataset.X.columns, 'is_bad')
        mlmodel.kfold_cv()
        optimizer = ParameterOptimizer(mlmodel.ytest_iterations, mlmodel.ypred_iterations)

    def predict(self, X: pandas.DataFrame) -> pandas.DataFrame:
        #TODO: uses best xgb_est and X TESTSET to obtain out-of-sample predictions.
        return None

    def predict_proba(self, X: pandas.DataFrame) -> pandas.DataFrame:
        #TODO: predicts label probabilities of predicted y using X TESTSET.
        return None

    def evaluate(self, X: pandas.DataFrame, y: numpy.ndarray) -> dict:
        return None

    def tune_parameters(self, X: pandas.DataFrame, y: numpy.ndarray) -> dict:
        return None


class Evaluators:
    """Evaluates model accuracy using a confusion matrix, precision/recall (f-1 score), and log-loss."""
    def __init__(self, ytest_iterations: list, ypred_iterations: list):
        self.ytest_iterations = ytest_iterations
        self.ypred_iterations = ypred_iterations
        self.confusion_matrices = []
        self.loglosses = []
        self.rmses = []

    def compute_rmse(self):
        """RMSE Evaluation. Lower is better."""
        for ytest, ypred in zip(self.ytest_iterations, self.ypred_iterations):
            mse = mean_squared_error(ytest, ypred)
            self.rmses.append(numpy.sqrt(mse))
        utils.print_delimiter()
        LOGGER.info(self.rmses)

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
            insample_accuracy = accuracy_score(ypred, ytest)
            tn, fp, fn, tp = confusion_matrix(ypred, ytest).ravel()
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            LOGGER.info('Accuracy: {}, Precision: {}, Recall: {}'.format(round(insample_accuracy, 2),
                                                                         precision, recall))
            self.confusion_matrices.append(pandas.DataFrame(confusion_matrix(ypred, ytest)))
            utils.print_delimiter()
            LOGGER.info(self.confusion_matrices)

    def compute_conditional_log_loss(self):
        """
        LogLoss Evaluation. Lower is better.
        binary-logistic' uses -(y*log(y_pred) + (y-1)*(log(1-y_pred))
        """
        for ytest, ypred in zip(self.ytest_iterations, self.ypred_iterations):
            logloss = log_loss(ytest, ypred, normalize=True)
            self.loglosses.append(logloss)
        utils.print_delimiter()
        LOGGER.info(self.loglosses)


class ParameterOptimizer(Evaluators, ML101Model, ApplyPCA, DataPreparer):
    """Evaluates model kfold crossvalidations to obtain optimized model parameters for best fit."""
    def __init__(self, ytest_iterations: list, ypred_iterations: list):
        super().__init__(ytest_iterations, ypred_iterations)
        self.ytest_iterations = ytest_iterations
        self.ypred_iterations = ypred_iterations

    #TODO: use grid search using scores as input, then adjusting pca and sampling params, and getting new scores output.
    #TODO: use normalized mutual information score to evaluate how good the sampling was. Adjust sampling accordingly.