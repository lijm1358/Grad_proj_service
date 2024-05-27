FROM apache/airflow:2.9.1

# COPY data/test_data.pt /opt/airflow/data/test_data.pt
COPY data/item2idx.pt /opt/airflow/data/item2idx.pt
COPY gcp_airflow_cr.json /opt/airflow/gcp_airflow_cr.json
COPY gcp_retrain_key /opt/airflow/gcp_retrain_key

RUN pip install apache-airflow-providers-google apache-airflow-providers-mysql apache-airflow-providers-ssh torch
