FROM python:3-slim
LABEL authors="amadousy"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["mlflow", "server"]