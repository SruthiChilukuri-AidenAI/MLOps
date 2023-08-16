import os
import pickle
import sys

import mlflow
import yaml
from sklearn.naive_bayes import MultinomialNB

from utils.mlflow_utils import get_mlflow_cfg

# read the command line params
if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython3 train.py features-dir-path model-filename\n")
    sys.exit(1)

features_path = sys.argv[1]
model_filename = sys.argv[2]

# read pipeline params
params = yaml.safe_load(open("params.yaml"))["train"]

alpha = params["alpha"]

# load the train features
features_train_pkl = os.path.join(features_path, "train.pkl")
with open(features_train_pkl, "rb") as f:
    train_data = pickle.load(f)

X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]

cfg = get_mlflow_cfg()
mlflow.set_tracking_uri(cfg["MLFLOW_TRACKING_URI"])
mlflow.set_experiment(cfg["PROJECT_EXPERIMENT_NAME"])
print("run_id=", os.getenv("MLFLOW_RUNID"))
with mlflow.start_run(run_id=os.getenv("MLFLOW_RUNID")) as run:
    mlflow.log_param("alpha", alpha)
# train the model
clf = MultinomialNB(alpha=alpha)
clf.fit(X, y)

# save the model
with open(model_filename, "wb") as f:
    pickle.dump(clf, f)
