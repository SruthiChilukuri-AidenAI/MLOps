import mlflow
import typer
from datetime import datetime
import os
import subprocess

from mlflow_utils import get_mlflow_cfg

# from src.constants import PROJECT_ROOT_PATH, PROJECT_EXPERIMENT_NAME
# PROJECT_EXPERIMENT_NAME="mlops_test_clf01"


def start_pipeline():
    cfg=get_mlflow_cfg()
    # print(cfg)
    # mlflow.set_tracking_uri("http://34.229.134.14:5000/")
    # mlflow.set_experiment('mlops_test_clf03')
    mlflow.set_tracking_uri(cfg['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment(cfg['PROJECT_EXPERIMENT_NAME'])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = 'run{}'.format(timestamp)
    # print(run_name)
    # with mlflow.start_run(run_name=run_name):
    with mlflow.start_run(run_name=run_name) as run:
        run_id=mlflow.active_run().info.run_id
        # os.system(f"set MLFLOW_EXP_RUN_ID={run_id}")
        # os.environ['MLFLOW_EXP_RUN_ID']=run_id
        print(run_id)
        #subprocess.run(["dvc", "params", "modify", "params.yaml", 'mlflow.run_id', run_id])
        # mlflow.log_artifact(PROJECT_ROOT_PATH / "dvc.yaml")


if __name__ == "__main__":
    # typer.run(start_pipeline)
    start_pipeline()
