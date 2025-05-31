FROM python:3-slim
LABEL authors="amadousy"

COPY requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

ENTRYPOINT ["mlflow", "server"]