import os
import unittest

import numpy
import pandas

import ml101.sampler

__folder__ = os.path.dirname(os.path.realpath(__file__))


class TestDataset(unittest.TestCase):
    def setUp(self) -> None:
        with open(os.path.join(__folder__, "data.csv"), "r") as fileobj:
            data = pandas.read_csv(fileobj)

        X = data[data.columns.difference(["is_bad"])]
        y = numpy.array(data[["is_bad"]])


        dataset = ml101.sampler.DataPreparer()
        dataset.clientside_pca(X, category_limit=2)
        dataset.sample(y)

    def test_data_is_clean(self):
        return None
