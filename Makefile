DOCKER_TAG?=latest
PYTHON_VERSION?=3.8

clean:
	@rm -rf build dist *.egg-info

build:
	@docker build -t dask-lightgbm:${DOCKER_TAG} --build-arg VERSION=${PYTHON_VERSION} -f Dockerfile.test .

unit-test:
	@docker run dask-lightgbm:${DOCKER_TAG} pytest --pylama --cov=dask_lightgbm dask_lightgbm

system-test:
	@DOCKER_TAG=${DOCKER_TAG} docker-compose up --abort-on-container-exit

publish: clean
	@poetry publish --build
