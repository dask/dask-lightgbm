import dask.array as da
import dask.dataframe as dd
import lightgbm
import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr, spearmanr
import scipy.sparse
import sparse
from dask.array.utils import assert_eq
from dask_ml.metrics import accuracy_score, r2_score

from distributed.utils_test import client, cluster_fixture, loop, gen_cluster  # noqa
from sklearn.datasets import make_blobs, make_regression
from sklearn.metrics import confusion_matrix
from sklearn.utils import check_random_state

import dask_lightgbm.core as dlgbm


data_output = ['array', 'scipy_csr_matrix', 'sparse', 'dataframe']
data_centers = [[[-4, -4], [4, 4]], [[-4, -4], [4, 4], [-4, 4]]]


@pytest.fixture()
def listen_port():
    listen_port.port += 10
    return listen_port.port


listen_port.port = 13000


def _make_ranking(n_samples=100, n_features=20, avg_gs=10, gmax=4, random_state=0):
    """Generate a mock learning-to-rank dataset - feature vectors grouped together with
    integer-valued graded relevance scores."""
    rnd_generator = check_random_state(random_state)

    y_vec, group_vec = np.empty((0,), dtype=int), np.empty((0,), dtype=int)
    group = 0

    # build target, group ID vectors.
    while len(y_vec) < n_samples:
        gsize = rnd_generator.poisson(avg_gs)
        if not gsize:
            continue

        rel = rnd_generator.choice(range(gmax + 1), size=gsize, replace=True)
        y_vec = np.append(y_vec, rel)
        group_vec = np.append(group_vec, [group] * gsize)
        group += 1

    y_vec, group_vec = y_vec[0:n_samples], group_vec[0:n_samples]

    # build feature data, X. Transform first few into informative features.
    n_informative = int(np.ceil(n_features * 0.25))
    x_grid = np.linspace(0, stop=1, num=gmax + 2)
    X = np.random.uniform(size=(n_samples, n_features))

    # make first n_informative features values bucketed according to relevance scores.
    def bucket_fn(z):
        return np.random.uniform(x_grid[z], high=x_grid[z + 1])

    for j in range(n_informative):
        X[:, j] = np.apply_along_axis(bucket_fn, axis=0, arr=y_vec)

    return X, y_vec, group_vec


def _create_ranking_data(n_samples=100, output='dataframe', chunk_size=10):
    X, y, g = _make_ranking(n_samples=n_samples, random_state=42)
    rnd = np.random.RandomState(42)
    w = rnd.rand(X.shape[0]) * 0.01

    if output != 'dataframe':
        raise ValueError('ranking objective only supported for output = "dataframe" (dask.dataframe.core.DataFrames)')

    # add target, weight, and group to DataFrame so that partitions abide by group boundaries.
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    X_df['y'] = y
    X_df['g'] = g
    X_df['w'] = w

    # set_index ensures partitions are based on group id. See https://bit.ly/3pAWyNw.
    X_df.set_index('g', inplace=True)
    dX = dd.from_pandas(X_df, chunksize=chunk_size)

    # separate target, weight from features.
    dy = dX['y']
    dw = dX['w']
    dX = dX.drop(columns=['y', 'w'])
    dg = dX.index.to_series()

    # encode group identifiers into run-length encoding, the format LightGBMRanker is expecting
    # so that within each partition, sum(g) = n_samples.
    dg = dg.map_partitions(lambda p: p.groupby('g', sort=False).apply(lambda z: z.shape[0]))
    gid_vec, gid_rle = np.unique(g, return_counts=True)
    g = gid_rle[np.argsort(gid_vec)]

    return X, y, w, g, dX, dy, dw, dg


def _create_data(objective, n_samples=100, centers=2, output='array', chunk_size=50):
    if objective == 'classification':
        X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=42)
    elif objective == 'regression':
        X, y = make_regression(n_samples=n_samples, random_state=42)
    else:
        raise ValueError(objective)
    rnd = np.random.RandomState(42)
    w = rnd.rand(X.shape[0])*0.01

    if output == 'array':
        dX = da.from_array(X, (chunk_size, X.shape[1]))
        dy = da.from_array(y, chunk_size)
        dw = da.from_array(w, chunk_size)
    elif output == 'dataframe':
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        y_df = pd.Series(y, name='target')
        dX = dd.from_pandas(X_df, chunksize=chunk_size)
        dy = dd.from_pandas(y_df, chunksize=chunk_size)
        dw = dd.from_array(w, chunksize=chunk_size)
    elif output == 'scipy_csr_matrix':
        dX = da.from_array(X, chunks=(chunk_size, X.shape[1])).map_blocks(scipy.sparse.csr_matrix)
        dy = da.from_array(y, chunks=chunk_size)
        dw = da.from_array(w, chunk_size)
    elif output == 'sparse':
        dX = da.from_array(X, chunks=(chunk_size, X.shape[1])).map_blocks(sparse.COO)
        dy = da.from_array(y, chunks=chunk_size)
        dw = da.from_array(w, chunk_size)

    return X, y, w, dX, dy, dw


@pytest.mark.parametrize('output', data_output)
@pytest.mark.parametrize('centers', data_centers)
def test_classifier(output, centers, client, listen_port):  # noqa
    X, y, w, dX, dy, dw = _create_data('classification', output=output, centers=centers)

    a = dlgbm.LGBMClassifier(time_out=5, local_listen_port=listen_port)
    a = a.fit(dX, dy, sample_weight=dw, client=client)
    p1 = a.predict(dX, client=client)
    s1 = accuracy_score(dy, p1)
    p1 = p1.compute()

    b = lightgbm.LGBMClassifier()
    b.fit(X, y, sample_weight=w)
    p2 = b.predict(X)
    s2 = b.score(X, y)
    print(confusion_matrix(y, p1))
    print(confusion_matrix(y, p2))

    assert_eq(s1, s2)
    print(s1)

    assert_eq(p1, p2)
    assert_eq(y, p1)
    assert_eq(y, p2)


@pytest.mark.parametrize('output', data_output)
@pytest.mark.parametrize('centers', data_centers)
def test_classifier_proba(output, centers, client, listen_port):  # noqa
    X, y, w, dX, dy, dw = _create_data('classification', output=output, centers=centers)

    a = dlgbm.LGBMClassifier(time_out=5, local_listen_port=listen_port)
    a = a.fit(dX, dy, sample_weight=dw, client=client)
    p1 = a.predict_proba(dX, client=client)
    p1 = p1.compute()

    b = lightgbm.LGBMClassifier()
    b.fit(X, y, sample_weight=w)
    p2 = b.predict_proba(X)

    assert_eq(p1, p2, atol=0.3)


def test_classifier_local_predict(client, listen_port):  # noqa
    X, y, w, dX, dy, dw = _create_data('classification', output='array')

    a = dlgbm.LGBMClassifier(time_out=5, local_listen_port=listen_port)
    a = a.fit(dX, dy, sample_weight=dw, client=client)
    p1 = a.to_local().predict(dX)

    b = lightgbm.LGBMClassifier()
    b.fit(X, y, sample_weight=w)
    p2 = b.predict(X)

    assert_eq(p1, p2)
    assert_eq(y, p1)
    assert_eq(y, p2)


@pytest.mark.parametrize('output', data_output)
def test_regressor(output, client, listen_port):  # noqa
    X, y, w, dX, dy, dw = _create_data('regression', output=output)

    a = dlgbm.LGBMRegressor(time_out=5, local_listen_port=listen_port, seed=42)
    a = a.fit(dX, dy, client=client, sample_weight=dw)
    p1 = a.predict(dX, client=client)
    if output != 'dataframe':
        s1 = r2_score(dy, p1)
    p1 = p1.compute()

    b = lightgbm.LGBMRegressor(seed=42)
    b.fit(X, y, sample_weight=w)
    s2 = b.score(X, y)
    p2 = b.predict(X)

    # Scores should be the same
    if output != 'dataframe':
        assert_eq(s1, s2, atol=.01)

    # Predictions should be roughly the same
    assert_eq(y, p1, rtol=1., atol=50.)
    assert_eq(y, p2, rtol=1., atol=50.)


@pytest.mark.parametrize('output', data_output)
@pytest.mark.parametrize('alpha', [.1, .5, .9])
def test_regressor_quantile(output, client, listen_port, alpha):  # noqa
    X, y, w, dX, dy, dw = _create_data('regression', output=output)

    a = dlgbm.LGBMRegressor(local_listen_port=listen_port, seed=42, objective='quantile', alpha=alpha)
    a = a.fit(dX, dy, client=client, sample_weight=dw)
    p1 = a.predict(dX, client=client).compute()
    q1 = np.count_nonzero(y < p1) / y.shape[0]

    b = lightgbm.LGBMRegressor(seed=42, objective='quantile', alpha=alpha)
    b.fit(X, y, sample_weight=w)
    p2 = b.predict(X)
    q2 = np.count_nonzero(y < p2) / y.shape[0]

    # Quantiles should be right
    np.isclose(q1, alpha, atol=.1)
    np.isclose(q2, alpha, atol=.1)


def test_regressor_local_predict(client, listen_port):  # noqa
    X, y, w, dX, dy, dw = _create_data('regression', output='array')

    a = dlgbm.LGBMRegressor(local_listen_port=listen_port, seed=42)
    a = a.fit(dX, dy, sample_weight=dw, client=client)
    p1 = a.predict(dX)
    p2 = a.to_local().predict(X)
    s1 = r2_score(dy, p1)
    p1 = p1.compute()
    s2 = a.to_local().score(X, y)
    print(s1)

    # Predictions and scores should be the same
    assert_eq(p1, p2)
    np.isclose(s1, s2)


def test_ranker(client, listen_port):  # noqa
    X, y, w, g, dX, dy, dw, dg = _create_ranking_data(output='dataframe')

    a = dlgbm.LGBMRanker(time_out=5, local_listen_port=listen_port, seed=42)
    a = a.fit(dX, dy, sample_weight=dw, group=dg, client=client)
    rnk1 = a.predict(dX, client=client)
    rnk1 = rnk1.compute()

    b = lightgbm.LGBMRanker(seed=42)
    b.fit(X, y, sample_weight=w, group=g)
    rnk2 = b.predict(X)

    # distributed ranker should do a pretty good job of ranking
    assert spearmanr(rnk1, y).correlation > 0.95

    # distributed scores should give virtually same ranking as local model.
    assert pearsonr(rnk1, rnk2)[0] > 0.98


def test_ranker_local_predict(client, listen_port):  # noqa
    X, y, w, g, dX, dy, dw, dg = _create_ranking_data(output='dataframe')

    a = dlgbm.LGBMRanker(local_listen_port=listen_port, seed=42)
    a = a.fit(dX, dy, group=dg, client=client)
    rnk1 = a.predict(dX)
    rnk1 = rnk1.compute()
    rnk2 = a.to_local().predict(X)

    # distributed and local scores should be the same.
    assert_eq(rnk1, rnk2)


def test_build_network_params():
    workers_ips = [
        'tcp://192.168.0.1:34545',
        'tcp://192.168.0.2:34346',
        'tcp://192.168.0.3:34347'
    ]

    params = dlgbm.build_network_params(workers_ips, 'tcp://192.168.0.2:34346', 12400, 120)
    exp_params = {
        'machines': '192.168.0.1:12400,192.168.0.2:12401,192.168.0.3:12402',
        'local_listen_port': 12401,
        'num_machines': len(workers_ips),
        'time_out': 120
    }
    assert exp_params == params


@gen_cluster(client=True, timeout=None)
def test_errors(c, s, a, b):
    def f(part):
        raise Exception('foo')
    df = dd.demo.make_timeseries()
    df = df.map_partitions(f, meta=df._meta)
    with pytest.raises(Exception) as info:
        yield dlgbm.train(c, df, df.x, params={}, model_factory=lightgbm.LGBMClassifier)
        assert 'foo' in str(info.value)
