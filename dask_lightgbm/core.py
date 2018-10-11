import logging
from collections import defaultdict

import lightgbm
import numpy as np
import pandas as pd
from lightgbm.basic import _safe_call, _LIB
from toolz import first, assoc

try:
    import sparse
    import scipy.sparse as ss
except ImportError:
    sparse = False
    ss = False

from dask import delayed
from dask.distributed import wait, default_client, get_worker

logger = logging.getLogger(__name__)


def parse_host_port(address):
    if '://' in address:
        address = address.rsplit('://', 1)[1]
    host, port = address.split(':')
    port = int(port)
    return host, port


def build_network_params(worker_addresses, local_worker_ip, local_listen_port, listen_time_out):
    addr_port_map = {addr: (local_listen_port + i) for i, addr in enumerate(worker_addresses)}
    params = {
        "machines": ",".join([parse_host_port(addr)[0] + ":" + str(port) for addr, port in addr_port_map.items()]),
        "local_listen_port": addr_port_map[local_worker_ip],
        "listen_time_out": listen_time_out,
        "num_machines": len(addr_port_map)
    }
    return params


def concat(L):
    if isinstance(L[0], np.ndarray):
        return np.concatenate(L, axis=0)
    elif isinstance(L[0], (pd.DataFrame, pd.Series)):
        return pd.concat(L, axis=0)
    elif ss and isinstance(L[0], ss.spmatrix):
        return ss.vstack(L, format='csr')
    elif sparse and isinstance(L[0], sparse.SparseArray):
        return sparse.concatenate(L, axis=0)
    else:
        raise TypeError("Data must be either numpy arrays or pandas dataframes"
                        ". Got %s" % type(L[0]))


def _fit_local(params, model_factory, list_of_parts, worker_addresses, local_listen_port=12400, listen_time_out=120,
               **kwargs):
    network_params = build_network_params(worker_addresses, get_worker().address, local_listen_port,
                                          listen_time_out)
    params = {**params, **network_params}

    data, labels = zip(*list_of_parts)  # Prepare data
    data = concat(data)  # Concatenate many parts into one
    labels = concat(labels)

    try:
        classifier = model_factory(**params)
        classifier.fit(data, labels)
    finally:
        _safe_call(_LIB.LGBM_NetworkFree())
    return classifier


def train(client, X, y, params, model_factory, **kwargs):
    data_parts = X.to_delayed()
    label_parts = y.to_delayed()
    if isinstance(data_parts, np.ndarray):
        assert data_parts.shape[1] == 1
        data_parts = data_parts.flatten().tolist()
    if isinstance(label_parts, np.ndarray):
        assert label_parts.ndim == 1 or label_parts.shape[1] == 1
        label_parts = label_parts.flatten().tolist()
    # Arrange parts into pairs.  This enforces co-locality
    parts = list(map(delayed, zip(data_parts, label_parts)))
    parts = client.compute(parts)  # Start computation in the background
    wait(parts)
    # for part in parts:
    #     if part.status == 'error':
    #         part  # trigger error locally
    key_to_part_dict = dict([(part.key, part) for part in parts])
    who_has = client.who_has(parts)
    worker_map = defaultdict(list)
    for key, workers in who_has.items():
        worker_map[first(workers)].append(key_to_part_dict[key])
    ncores = client.ncores()  # Number of cores per worker
    params['tree_learner'] = "data"
    # Tell each worker to init the booster on the chunks/parts that it has locally
    futures_classifiers = [client.submit(_fit_local,
                                         model_factory=model_factory,
                                         params=assoc(params, 'num_threads', ncores[worker]),
                                         list_of_parts=list_of_parts,
                                         worker_addresses=list(worker_map.keys()),
                                         local_listen_port=params.get("local_listen_port", 12400),
                                         listen_time_out=params.get("listen_time_out", 120),
                                         **kwargs)
                           for worker, list_of_parts in worker_map.items()]

    results = client.gather(futures_classifiers)
    results = [v for v in results if v]
    return results[0]


class LGBMRegressor(lightgbm.LGBMRegressor):

    def fit(self, X, y=None, **kwargs):
        client = default_client()
        model_factory = lightgbm.LGBMRegressor
        params = self.get_params(True)

        model = train(client, X, y, params, model_factory, **kwargs)
        self.set_params(**model.get_params())
        self._Booster = model._Booster
        self._le = model._le
        self._classes = model._classes
        self._n_classes = model._n_classes
        self._n_features = model._n_features
        self._evals_result = model._evals_result
        self._best_iteration = model._best_iteration
        self._best_score = model._best_score

        return self


class LGBMClassifier(lightgbm.LGBMClassifier):

    def fit(self, X, y=None, **kwargs):
        """Fit a gradient boosting classifier

        Parameters
        ----------
        X : array-like [n_samples, n_features]
            Feature Matrix. May be a dask.array or dask.dataframe
        y : array-like
            Labels
        classes : sequence, optional
            The unique values in `y`. If no specified, this will be
            eagerly computed from `y` before training.

        Returns
        -------
        self : LGBMClassifier
        """
        client = default_client()
        model_factory = lightgbm.LGBMClassifier
        params = self.get_params(True)

        model = train(client, X, y, params, model_factory, **kwargs)
        self.set_params(**model.get_params())
        self._Booster = model._Booster
        self._le = model._le
        self._classes = model._classes
        self._n_classes = model._n_classes
        self._n_features = model._n_features
        self._evals_result = model._evals_result
        self._best_iteration = model._best_iteration
        self._best_score = model._best_score

        return self
