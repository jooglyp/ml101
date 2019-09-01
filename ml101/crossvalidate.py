"""K-Fold crossvalidator"""

import logging

import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from dask.distributed import Client
import dask
import dask.dataframe as dd
import dask.array as da
from dask_ml.xgboost import XGBRegressor
from dask_ml.model_selection import KFold
import numpy

LOGGER = logging.getLogger(__name__)
CLIENT = dask.distributed.Client()


class CrossValidation:
    def __init__(self, X: numpy.ndarray, y: numpy.ndarray, xlabels: list, ylabel: str):
        self.X = da.from_array(X, chunks=X.shape)
        self.y = da.from_array(y, chunks=y.shape)
        self.xlabels = xlabels
        self.ylabel = ylabel
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
        with joblib.parallel_backend('dask'):
            scores = []
            xgb_est = XGBRegressor()
            cv = KFold(n_splits=10, random_state=84, shuffle=True)
            for train_index, test_index in cv.split(self.X):
                X_train, X_test, y_train, y_test = self.X[train_index], self.X[test_index], \
                                                   self.y[train_index], self.y[test_index]
                xgb_est.fit(X_train, y_train)
                scores.append(xgb_est.score(X_test, y_test))
            LOGGER.info(scores)
