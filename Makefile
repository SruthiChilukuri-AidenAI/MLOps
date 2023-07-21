run_pipeline:
	export MLFLOW_RUNID=`python src/utils/start_pipeline.py`; \
	dvc repro
