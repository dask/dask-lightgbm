FROM python:3.6-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake \
        make \
        gcc \
        g++ \
        libssl-dev \
        automake \
        libtool \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ADD . /app
ENV PYTHONPATH=/app:/usr/lib/python3.6/site-packages/

WORKDIR /app

RUN python -m pip install ".[dev]"
RUN python -m pip install ".[sparse]"
