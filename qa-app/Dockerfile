FROM python:3.10.5-slim
LABEL description="DocSearch QA API image"
ENV PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.6.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONPATH="/app:${PYTHONPATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    python3-dev
RUN pip install poetry==${POETRY_VERSION}

COPY pyproject.toml poetry.lock /app/
WORKDIR /app
RUN python -m poetry install --no-root --only main && \
    apt-get remove -y gcc python3-dev && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* &&\
    mkdir /app/logs

COPY data /app/data
COPY src /app/src
COPY static /app/static
COPY templates /app/templates
COPY parameters.yaml /app/parameters.yaml

RUN python src/get_model_artifacts.py

EXPOSE 8000
CMD ["poetry", "run", "uvicorn", "src.run_api:app", "--host", "0.0.0.0", "--port", "8000"]
