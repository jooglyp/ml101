from __future__ import annotations

import datetime
import itertools
import random
import typing
from decimal import Decimal

import numpy
import pandas


class DataPreparer:
    def __init__(self):
        """"""
        self.raw_data = None

    def _load(self, csv) -> None:
        """Loads csv into memory as pandas dataframe and applies some transformations."""
        self.raw_data = pandas.read_csv(csv)
