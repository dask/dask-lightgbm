import os

import dask.dataframe as dd
import pytest
from dask.distributed import Client

import dask_lightgbm.core as dlgbm


@pytest.fixture(scope='module')
def client():
    with Client(os.getenv('SCHEDULER')) as client:
        yield client


@pytest.fixture()
def listen_port():
    listen_port.port += 10
    return listen_port.port


listen_port.port = 12400


def test_classify_newsread(client, listen_port):
    data = dd.read_csv('./system_tests/data/*.gz', compression='gzip', blocksize=None)
    dX = data.iloc[:, :-1]
    dy = data.iloc[:, -1]

    d_classif = dlgbm.LGBMClassifier(n_estimators=50, local_listen_port=listen_port)
    d_classif.fit(dX, dy)

    dy_pred = d_classif.predict(dX, client=client)

    acc_score = (dy == dy_pred).sum() / len(dy)
    acc_score = acc_score.compute()
    print(acc_score)

    assert acc_score > 0.8


def test_regress_newsread(client, listen_port):
    data = dd.read_csv('./system_tests/data/*.gz', compression='gzip', blocksize=None)
    dX = data.iloc[:, 1:]
    dy = data.iloc[:, 0]

    d_regress = dlgbm.LGBMRegressor(n_estimators=50, local_listen_port=listen_port)
    d_regress.fit(dX, dy)

    dy_pred = d_regress.predict(dX, client=client)

    # The dask_ml.metrics.r2_score method fails with dataframes so we compute the R2 score ourselves
    numerator = ((dy - dy_pred) ** 2).sum()
    denominator = ((dy - dy.mean()) ** 2).sum()
    r2_score = 1 - numerator / denominator
    r2_score = r2_score.compute()
    print(r2_score)

    assert r2_score > 0.8
