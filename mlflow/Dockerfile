FROM python:3-bullseye
LABEL authors="amadousy"

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

ENTRYPOINT ["mlflow", "server"]