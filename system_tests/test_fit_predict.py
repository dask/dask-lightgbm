import gzip
import unittest

import dask.dataframe as dd
import numpy as np
from dask import delayed
from dask.distributed import Client
from sklearn.metrics import confusion_matrix

import dask_lightgbm.core as dlgbm


@delayed
def load_data(path):
    with gzip.open(path) as fp:
        X = np.loadtxt(fp, delimiter=",")
        return X


class FitPredictTest(unittest.TestCase):

    def setUp(self):
        self.client = Client("scheduler:8786")

    def test_classify_newsread(self):
        data = dd.read_csv("./system_tests/data/*.gz", compression="gzip", blocksize=None)
        dX = data.iloc[:, :-1]
        dy = data.iloc[:, -1]

        d_classif = dlgbm.LGBMClassifier(n_estimators=50)
        d_classif.fit(dX, dy)

        dy_pred = d_classif.predict(dX)

        print(confusion_matrix(dy.compute(), dy_pred.compute()))

        self.assertGreaterEqual((dy == dy_pred).sum()/len(dy), 0.9)
