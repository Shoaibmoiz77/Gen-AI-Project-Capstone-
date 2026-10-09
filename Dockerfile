FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY data ./data
# Build the index at image build time so the container starts instantly.
RUN groundwork ingest --src data/corpus --index /app/.index

ENV GROUNDWORK_INDEX_DIR=/app/.index
EXPOSE 8000
RUN useradd --create-home app && chown -R app /app
USER app

CMD ["uvicorn", "groundwork.api:app", "--host", "0.0.0.0", "--port", "8000"]
