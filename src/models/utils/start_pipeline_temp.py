import mlflow
from datetime import datetime
mlflow.set_tracking_uri("http://34.229.134.14:5000/")
mlflow.set_experiment('mlops_test_clf03')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_name = 'run{}'.format(timestamp)
#print(run_name)
with mlflow.start_run(run_name=run_name):
    print(mlflow.active_run().info.run_id)
    mlflow.log_param('batch size',32)
