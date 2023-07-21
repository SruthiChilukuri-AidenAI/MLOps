import mlflow
import typer
from datetime import datetime
import os

from mlflow_utils import get_mlflow_cfg

# from src.constants import PROJECT_ROOT_PATH, PROJECT_EXPERIMENT_NAME
# PROJECT_EXPERIMENT_NAME="mlops_test_clf01"


def start_pipeline():
    cfg=get_mlflow_cfg()
    mlflow.set_tracking_uri(cfg['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment(cfg['PROJECT_EXPERIMENT_NAME'])
    run_id=os.getenv('MLFLOW_EXP_RUN_ID')
    print('run_id in dummy stage=',run_id)
    # with mlflow.start_run(run_name=run_name):
    with mlflow.start_run(run_id=run_id) as run:
        act_run_id=mlflow.active_run().info.run_id
        print('active run_id=',act_run_id)
        # mlflow.log_artifact(PROJECT_ROOT_PATH / "dvc.yaml")


if __name__ == "__main__":
    # typer.run(start_pipeline)
    start_pipeline()