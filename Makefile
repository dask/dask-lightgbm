DOCKER_TAG ?= latest

init:
	@pipenv install --dev

build:
	@docker build -t dask-lightgmb:${DOCKER_TAG} --build-arg version=3.6 -f Dockerfile.test .

unit-test:
	@docker run dask-lightgmb:${DOCKER_TAG} pytest --pylama --cov=dask_lightgbm dask_lightgbm

system-test:
	@docker-compose up --abort-on-container-exit
