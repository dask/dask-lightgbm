Dask-LightGBM
=============

[![Build Status](https://travis-ci.org/dask/dask-lightgbm.svg?branch=master)](https://travis-ci.org/dask/dask-lightgbm)

Distributed training with LightGBM and Dask.distributed

This repository enables you to perform distributed training with LightGBM on
Dask.Array and Dask.DataFrame collections. It is based on dask-xgboost package.

Usage
-----
Load your data into distributed data-structure Dask.Array or Dask.DataFrame.
Connect to a Dask cluster using and let dask-lightgbm train the model for you.
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
