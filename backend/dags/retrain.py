import os
from datetime import datetime, timedelta

import torch
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.operators.compute import ComputeEngineStartInstanceOperator
from airflow.providers.google.cloud.transfers.gcs_to_local import GCSToLocalFilesystemOperator
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from airflow.providers.ssh.operators.ssh import SSHOperator
from airflow.utils.dates import days_ago


def fetch_db_and_update_data() -> None:
    before_24h = datetime.now() - timedelta(days=1)
    before_24h = before_24h.strftime("%Y-%m-%d")
    # before_24h = "2024-01-01"
    request = f"""SELECT user_id, item_id
    FROM loguseriteminteraction
    WHERE timestamp >= '{before_24h}'
    ORDER BY timestamp ASC"""
    mysql_hook = MySqlHook(mysql_conn_id="mysql_db")
    connection = mysql_hook.get_conn()
    cursor = connection.cursor()
    cursor.execute(request)
    sources = cursor.fetchall()

    original_data = torch.load("test_data.pt")
    item2idx = torch.load("data/item2idx.pt")

    print(original_data[123])
    for user_id, item_id in sources:
        user_index = user_id - 1
        item_index = item2idx[int(item_id[1:])]
        original_data[user_index].append(item_index)

    train_data = {k: v[:-2] for k, v in original_data.items()}
    valid_data = {k: v[:-1] for k, v in original_data.items()}
    test_data = {k: v for k, v in original_data.items()}  # copy

    torch.save(train_data, "./train_data.pt")
    torch.save(valid_data, "./valid_data.pt")
    torch.save(test_data, "./test_data.pt")


with DAG(
    dag_id="retrain_recommender",
    description="Retrain recommendation model with data until yesterday.",
    start_date=days_ago(1),
    schedule_interval="0 1 * * *",  # 매일 01:00에 실행
    tags=["grad_proj"],
) as dag:
    gce_instance_start = ComputeEngineStartInstanceOperator(
        task_id="gcp_compute_start_task",
        project_id=os.environ["GCE_PROJECT_ID"],
        zone=os.environ["GCE_ZONE"],
        resource_id=os.environ["GCE_RESOURCE_ID"],
    )

    upload_file = LocalFilesystemToGCSOperator(
        task_id="upload_file",
        src=["./train_data.pt", "./valid_data.pt", "./test_data.pt"],
        dst=(datetime.now() - timedelta(days=1)).strftime("%Y%m%d")[2:] + "/",
        bucket=os.environ["GCP_MODEL_BUCKET_NAME"],
    )

    download_file = GCSToLocalFilesystemOperator(
        task_id="download_file",
        object_name=(datetime.now() - timedelta(days=2)).strftime("%Y%m%d")[2:] + "/test_data.pt",
        # object_name="240519/test_data.pt", # initial file
        bucket=os.environ["GCP_MODEL_BUCKET_NAME"],
        filename="./test_data.pt",
    )

    update_data = PythonOperator(task_id="fetch_logs", python_callable=fetch_db_and_update_data)

    run_train = SSHOperator(
        task_id="ssh_run_train",
        ssh_conn_id="retrain_ssh",
        command="export PATH=/home/grad8553/miniconda3/bin:/home/grad8553/miniconda3/condabin:$PATH; \
            export GOOGLE_APPLICATION_CREDENTIALS=/home/grad8553/Grad_proj/sequential/gcp_airflow_cr.json; \
            cd ~/Grad_proj/sequential; nohup python -u main.py 1>logs.out 2>&1 &",
    )

    download_file >> update_data >> upload_file >> run_train
    gce_instance_start >> run_train
