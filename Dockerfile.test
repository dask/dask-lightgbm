ARG VERSION
FROM python:${VERSION}-buster

RUN pip install poetry
COPY . /app
WORKDIR /app

ENV POETRY_VIRTUALENVS_CREATE=false
RUN poetry install --no-interaction --extras "sparse"
