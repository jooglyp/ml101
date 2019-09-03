import os
import unittest

import numpy
import pandas

import ml101.sampler

__folder__ = os.path.dirname(os.path.realpath(__file__))


class Test(unittest.TestCase):
    def test_no_nan(self):
        with open(os.path.join(__folder__, "data.csv"), "r") as fileobj:
            data = pandas.read_csv(fileobj)

        X = data[data.columns.difference(["is_bad"])]
        y = numpy.array(data[["is_bad"]])

        dataset = ml101.sampler.DataPreparer()
        dataset.clientside_pca(X, category_limit=2)
        dataset.sample(y)

        for column in dataset.X.columns:
            series = dataset.X[column]
            self.assertEquals(series.isnull().values.any(), False)

    def test_categoricals_remain(self):
        with open(os.path.join(__folder__, "data.csv"), "r") as fileobj:
            data = pandas.read_csv(fileobj)

        X = data[data.columns.difference(["is_bad"])]
        y = numpy.array(data[["is_bad"]])

        dataset = ml101.sampler.DataPreparer()
        dataset.clientside_pca(X, category_limit=2)
        dataset.sample(y)

        original_columns = X.columns
        all_columns = dataset.cleaned_data
        categorical_columns = set(all_columns) - set(original_columns)
        self.assertIsNotNone(categorical_columns)

    def test_X_y_same_size(self):
        with open(os.path.join(__folder__, "data.csv"), "r") as fileobj:
            data = pandas.read_csv(fileobj)

        X = data[data.columns.difference(["is_bad"])]
        y = numpy.array(data[["is_bad"]])

        dataset = ml101.sampler.DataPreparer()
        dataset.clientside_pca(X, category_limit=2)
        dataset.sample(y)

        self.assertEqual(len(dataset.X), len(dataset.y))

    def test_XSample_ySample_same_size(self):
        with open(os.path.join(__folder__, "data.csv"), "r") as fileobj:
            data = pandas.read_csv(fileobj)

        X = data[data.columns.difference(["is_bad"])]
        y = numpy.array(data[["is_bad"]])

        dataset = ml101.sampler.DataPreparer()
        dataset.clientside_pca(X, category_limit=2)
        dataset.sample(y)

        self.assertEqual(
            len(dataset.x_random_resampled), len(dataset.y_random_resampled)
        )
