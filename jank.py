import mlflow
import os
from urllib.parse import urlencode


# Append ?token=... to every MLflow REST request (stopgap)
try:
    from mlflow.utils import rest_utils
    _orig_http_request = rest_utils.http_request

    TOKEN = os.getenv("MLFLOW_QUERY_TOKEN")

    def _http_request_with_token(host_creds, endpoint, method, *args, **kwargs):
        if TOKEN:
            sep = '&' if '?' in endpoint else '?'
            endpoint = f"{endpoint}{sep}{urlencode({'token': TOKEN})}"
        return _orig_http_request(host_creds, endpoint, method, *args, **kwargs)

    rest_utils.http_request = _http_request_with_token
except Exception as e:
    # Optional: log or ignore if patching fails; MLflow will proceed without query token
    pass

mlflow.set_tracking_uri("https://mlflow.havocai.net")
print(mlflow.get_experiment_by_name("Default"))