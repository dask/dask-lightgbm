DOCKER_TAG?=latest
PYTHON_VERSION?=3.7

build:
	@docker build -t dask-lightgbm:${DOCKER_TAG} --build-arg VERSION=${PYTHON_VERSION} -f Dockerfile.test .

unit-test:
	@docker run dask-lightgbm:${DOCKER_TAG} pytest --pylama --cov=dask_lightgbm dask_lightgbm

system-test:
	@DOCKER_TAG=${DOCKER_TAG} docker-compose up --abort-on-container-exit
