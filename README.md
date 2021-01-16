Dask-LightGBM - DEPRECATED
==========================

THIS REPOSITORY IS DEPRECATED
-----------------------------

This repository is deprecated and it is no longer maintained. The code was migrated into LightGBM package - https://github.com/microsoft/LightGBM.

[![Build Status](https://github.com/dask/dask-lightgbm/workflows/CI/badge.svg)](https://github.com/dask/dask-lightgbm/actions?query=workflow%3ACI)

Distributed training with LightGBM and Dask.distributed

This repository enables you to perform distributed training with LightGBM on
Dask.Array and Dask.DataFrame collections. It is based on dask-xgboost package.

Usage
-----
Load your data into distributed data-structure, which can be either Dask.Array or Dask.DataFrame.
Connect to a Dask cluster using Dask.distributed.Client.
Let dask-lightgbm train a model or make predictions for you.
See system tests for a sample code:
<https://github.com/dask/dask-lightgbm/blob/master/system_tests/test_fit_predict.py>

How this works
--------------
Dask is used mainly for accessing the cluster and managing data.
The library assures that both features and a label for each sample are located on the same worker.
It also lets each worker to know addresses and available ports of all other workers.
The distributed training is performed by LightGBM library itself using sockets.
See more details on distributed training in LightGBM here:
<https://github.com/microsoft/LightGBM/blob/master/docs/Parallel-Learning-Guide.rst>
