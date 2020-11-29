import os

import dask.dataframe as dd
import pytest
from dask.distributed import Client
from numpy import sqrt

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


def test_ranking_newsread(client, listen_port):
    data = dd.read_csv('./system_tests/data/*.gz', compression='gzip', blocksize=None)

    # group every ~11ish rows for ranking.
    dg = data.index.to_series().mod(50000)
    data = data.set_index(dg, sorted=False)
    data = client.persist(data)

    dX = data.iloc[:, :-1]
    dy = data.iloc[:, -1].mod(4)
    dg = data.index.to_series()
    dg = dg.map_partitions(lambda p: p.groupby(p, sort=False).apply(lambda z: z.shape[0]))

    d_rnk = dlgbm.LGBMRanker(n_estimators=50, local_listen_port=listen_port)
    d_rnk.fit(dX, dy, dg)

    dy_pred = d_rnk.predict(dX, client=client)

    # predicted scores are returned in a dask.array, difficult to col-concat
    # with dy, so determine (label, score) correlation by hand.
    mu_y, mu_pred = dy.mean(), dy_pred.mean()
    numerator = ((dy - mu_y) * (dy_pred - mu_pred)).sum()
    d1 = ((dy - mu_y) ** 2).sum()
    d2 = ((dy_pred - mu_pred) ** 2).sum().compute()
    r = (numerator / sqrt(d1 * d2)).compute()
    print(r)

    assert r > 0.7
