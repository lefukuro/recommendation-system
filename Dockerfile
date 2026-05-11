FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./

RUN pip install uv
RUN uv sync

COPY . .

CMD [".venv/bin/python", "src/ch.py"]