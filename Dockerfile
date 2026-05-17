FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    mlflow==3.12.0 \
    cloudpickle==3.1.2 \
    numpy==2.4.4 \
    tensorflow==2.21.0

COPY wwtp_rnn_mlflow/wwtp_rnn_bundle/mlruns ./mlruns

EXPOSE 5000

CMD ["mlflow", "models", "serve", \
    "-m", "/app/mlruns/248838729491991513/models/m-306e58e5dc5449ea9d9c7250ea4e2d3f/artifacts", \
    "--host", "0.0.0.0", \
    "--port", "5000", \
    "--env-manager", "local"]
