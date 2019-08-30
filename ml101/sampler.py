from __future__ import annotations

import logging
import datetime
import itertools
import random
import typing
from decimal import Decimal

import numpy
import pandas

from . import utils
from . import pca

LOGGER = logging.getLogger(__name__)


class DataPreparer:
    def __init__(self):
        """"""
        self.raw_data = None

    def load(self, csv) -> None:
        """Loads csv into memory as pandas dataframe and applies some transformations."""
        self.raw_data = pandas.read_csv(csv)

        pca_application = pca.ApplyPCA(self.raw_data)
        pca_application.apply_pca()
