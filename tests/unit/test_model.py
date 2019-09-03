import os
import unittest

import numpy
import pandas

import ml101.model
import ml101.sampler

__folder__ = os.path.dirname(os.path.realpath(__file__))


class Test(unittest.TestCase):
    def test_reproducable_base_model_f1(self):
        with open(os.path.join(__folder__, "data.csv"), "r") as fileobj:
            data = pandas.read_csv(fileobj)

        X = data[data.columns.difference(["is_bad"])]
        y = numpy.array(data[["is_bad"]])

        avg_f1scores = []
        for i in range(2):
            dataset = ml101.sampler.DataPreparer()
            dataset.clientside_pca(X, category_limit=2)
            dataset.sample(y)

            model = ml101.model.ML101Model(
                dataset.x_rnn_resampled,
                dataset.y_rnn_resampled,
                dataset.X.columns,
                dataset.important_covariates,
                dataset.model_covariates,
                X,
                y,
            )
            model.kfold_cv()
            optimizer = ml101.model.ParameterOptimizer(model)
            avg_f1scores.append(optimizer.compute_f1score())
        self.assertEqual(avg_f1scores[0], avg_f1scores[1])
