import numpy as np
import pandas as pd
import sparse
import lightgbm
import scipy.sparse
import pytest

import dask.array as da
from dask.array.utils import assert_eq
import dask.dataframe as dd
from dask.distributed import Client
from sklearn.datasets import make_blobs
from distributed.utils_test import gen_cluster, loop, cluster  # noqa
from sklearn.metrics import confusion_matrix

import dask_lightgbm.core as dlgbm

# Workaround for conflict with distributed 1.23.0
# https://github.com/dask/dask-xgboost/pull/27#issuecomment-417474734
from concurrent.futures import ThreadPoolExecutor
import distributed.comm.utils

distributed.comm.utils._offload_executor = ThreadPoolExecutor(max_workers=2)


def _create_data(n_samples=100, centers=2, output="array", chunk_size=50):
    X, y = make_blobs(n_samples=n_samples, centers=centers)
    w = np.random.rand(shape=X.shape[0])*0.01

    if output == "array":
        dX = da.from_array(X, (chunk_size, X.shape[1]))
        dy = da.from_array(y, chunk_size)
        dw = da.from_array(w, chunk_size)
    elif output == "dataframe":
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        y_df = pd.Series(y, name="target")
        dX = dd.from_pandas(X_df, chunksize=chunk_size)
        dy = dd.from_pandas(y_df, chunksize=chunk_size)
        dw = dd.from_array(w, chunksize=chunk_size)
    elif output == "scipy_csr_matrix":
        dX = da.from_array(X, chunks=(chunk_size, X.shape[1])).map_blocks(scipy.sparse.csr_matrix)
        dy = da.from_array(y, chunks=chunk_size)
        dw = da.from_array(w, chunk_size)
    elif output == "sparse":
        dX = da.from_array(X, chunks=(chunk_size, X.shape[1])).map_blocks(sparse.COO)
        dy = da.from_array(y, chunks=chunk_size)
        dw = da.from_array(w, chunk_size)

    return X, y, w, dX, dy, dw


@pytest.mark.parametrize("output, listen_port, centers", [ #noqa
    ('array', 11400, [[-4, -4], [4, 4]]),
    ('array', 12400, [[-4, -4], [4, 4], [-4, 4]]),
    ('scipy_csr_matrix', 13400, [[-4, -4], [4, 4]]),
    ('scipy_csr_matrix', 14400, [[-4, -4], [4, 4], [-4, 4]]),
    ('sparse', 15400, [[-4, -4], [4, 4]]),
    ('sparse', 16400, [[-4, -4], [4, 4], [-4, 4]]),
    ('dataframe', 17400, [[-4, -4], [4, 4]]),
    ('dataframe', 18400, [[-4, -4], [4, 4], [-4, 4]])
    ])  # noqa
def test_classifier(loop, output, listen_port, centers):
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop) as client:
            X, y, w, dX, dy, dw = _create_data(output=output, centers=centers)

            a = dlgbm.LGBMClassifier(local_listen_port=listen_port)
            a = a.fit(dX, dy, sample_weight=dw)
            p1 = a.predict(dX, client=client)
            p1 = p1.compute()

            b = lightgbm.LGBMClassifier()
            b.fit(X, y, sample_weight=w)
            p2 = b.predict(X)
            print(confusion_matrix(y, p1))
            print(confusion_matrix(y, p2))

            assert_eq(p1, p2)
            assert_eq(y, p1)
            assert_eq(y, p2)


@pytest.mark.parametrize("output, listen_port, centers", [ #noqa
    ('array', 21400, [[-4, -4], [4, 4]]),
    ('array', 22400, [[-4, -4], [4, 4], [-4, 4]]),
    ('scipy_csr_matrix', 23400, [[-4, -4], [4, 4]]),
    ('scipy_csr_matrix', 24400, [[-4, -4], [4, 4], [-4, 4]]),
    ('sparse', 25400, [[-4, -4], [4, 4]]),
    ('sparse', 26400, [[-4, -4], [4, 4], [-4, 4]]),
    ('dataframe', 27400, [[-4, -4], [4, 4]]),
    ('dataframe', 28400, [[-4, -4], [4, 4], [-4, 4]])
    ])  # noqa
def test_classifier_proba(loop, output, listen_port, centers):
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop) as client:
            X, y, w, dX, dy, dw = _create_data(output=output, centers=centers)

            a = dlgbm.LGBMClassifier(local_listen_port=listen_port)
            a = a.fit(dX, dy, sample_weight=dw)
            p1 = a.predict_proba(dX, client=client)
            p1 = p1.compute()

            b = lightgbm.LGBMClassifier()
            b.fit(X, y, sample_weight=w)
            p2 = b.predict_proba(X)

            assert_eq(p1, p2, atol=0.3)


def test_classifier_local_predict(loop): #noqa
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop):
            X, y, w, dX, dy, dw = _create_data(output="array")

            a = dlgbm.LGBMClassifier(local_listen_port=11400)
            a = a.fit(dX, dy, sample_weight=dw)
            p1 = a.to_local().predict(dX)

            b = lightgbm.LGBMClassifier()
            b.fit(X, y, sample_weight=w)
            p2 = b.predict(X)

            assert_eq(p1, p2)
            assert_eq(y, p1)
            assert_eq(y, p2)


def test_build_network_params():
    workers_ips = [
        "tcp://192.168.0.1:34545",
        "tcp://192.168.0.2:34346",
        "tcp://192.168.0.3:34347"
    ]

    params = dlgbm.build_network_params(workers_ips, "tcp://192.168.0.2:34346", 12400, 120)
    exp_params = {
        "machines": "192.168.0.1:12400,192.168.0.2:12401,192.168.0.3:12402",
        "local_listen_port": 12401,
        "num_machines": len(workers_ips),
        "listen_time_out": 120
    }
    assert exp_params == params


@gen_cluster(client=True, timeout=None, check_new_threads=False)
def test_errors(c, s, a, b):
    def f(part):
        raise Exception('foo')
    df = dd.demo.make_timeseries()
    df = df.map_partitions(f, meta=df._meta)
    with pytest.raises(Exception) as info:
        yield dlgbm.train(c, df, df.x, params={}, model_factory=lightgbm.LGBMClassifier)
        assert 'foo' in str(info.value)
