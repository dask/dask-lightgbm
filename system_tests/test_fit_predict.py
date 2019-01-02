import os
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
        # Test with either distributed scheduler or threaded scheduler (easier to setup)
        if os.getenv('SCHEDULER'):
            self.client = Client(os.getenv('SCHEDULER'))
        else:
            self.client = Client()

    def test_classify_newsread(self):
        data = dd.read_csv("./system_tests/data/*.gz", compression="gzip", blocksize=None)
        dX = data.iloc[:, :-1]
        dy = data.iloc[:, -1]

        d_classif = dlgbm.LGBMClassifier(n_estimators=50, local_listen_port=12400)
        d_classif.fit(dX, dy)

        dy_pred = d_classif.predict(dX, client=self.client)

        print(confusion_matrix(dy.compute(), dy_pred.compute()))

        acc_score = (dy == dy_pred).sum()/len(dy)
        acc_score = acc_score.compute()
        print(acc_score)

        self.assertGreaterEqual(acc_score, 0.8)

    def test_regress_newsread(self):
        data = dd.read_csv("./system_tests/data/*.gz", compression="gzip", blocksize=None)
        dX = data.iloc[:, 1:]
        dy = data.iloc[:, 0]

        d_regress = dlgbm.LGBMRegressor(n_estimators=50, local_listen_port=13400)
        d_regress.fit(dX, dy)

        dy_pred = d_regress.predict(dX, client=self.client)

        # The dask_ml.metrics.r2_score method fails with dataframes so we compute the R2 score ourselves
        numerator = ((dy - dy_pred) ** 2).sum()
        denominator = ((dy - dy.mean()) ** 2).sum()
        r2_score = 1 - numerator / denominator
        r2_score = r2_score.compute()
        print(r2_score)

        self.assertGreaterEqual(r2_score, 0.8)
