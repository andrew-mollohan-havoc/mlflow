import mlflow
from mlflow.entities import ViewType
import os
from urllib.parse import urlencode



# Append ?token=... to every MLflow REST request (stopgap)
try:
    from mlflow.utils import rest_utils
    _orig_http_request = rest_utils.http_request

    def _http_request_with_token(host_creds, endpoint, method, *args, **kwargs):
        TOKEN = os.getenv("MLFLOW_QUERY_TOKEN")
        if TOKEN:
            sep = '&' if '?' in endpoint else '?'
            endpoint = f"{endpoint}{sep}{urlencode({'token': TOKEN})}"
        return _orig_http_request(host_creds, endpoint, method, *args, **kwargs)

    rest_utils.http_request = _http_request_with_token
except Exception as e:
    # Optional: log or ignore if patching fails; MLflow will proceed without query token
    pass

mlflow.set_tracking_uri("https://mlflow.havocai.net")
mlflow.set_experiment("andrewm-test")
# Get all experiments, including active and archived
all_experiments = mlflow.search_experiments(view_type=ViewType.ALL)
for exp in all_experiments:
    print(f"Experiment ID: {exp.experiment_id}, Name: {exp.name}, Lifecycle Stage: {exp.lifecycle_stage}")
