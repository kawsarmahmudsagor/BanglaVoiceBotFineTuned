"""Best-effort forwarding of run params, metrics and status to MLflow. Never raises into training."""

import logging

from mlops_shared.contract import Metric

from .executor import Forwarder, NullForwarder

log = logging.getLogger(__name__)

_MAX_PARAM_LEN = 500
_PARAM_BATCH = 100
_METRIC_BATCH = 1000
_STATUS = {"done": "FINISHED", "failed": "FAILED", "cancelled": "KILLED"}


def flatten(d: dict, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = str(v)
    return out


class MlflowForwarder(Forwarder):
    def __init__(self, uri: str, client=None):
        self.uri = uri
        self.client = client
        self.mlflow_run_id: str | None = None

    def start(self, run_id: str, params: dict, tags: dict) -> str | None:
        try:
            from mlflow.entities import Param

            if self.client is None:
                from mlflow.tracking import MlflowClient

                self.client = MlflowClient(tracking_uri=self.uri)
            name = f"bangla-voice-{tags.get('task', 'unknown')}"
            exp = self.client.get_experiment_by_name(name)
            exp_id = exp.experiment_id if exp is not None else self.client.create_experiment(name)
            run = self.client.create_run(exp_id, run_name=run_id, tags={k: str(v) for k, v in tags.items()})
            self.mlflow_run_id = run.info.run_id
            flat = [Param(k, v[:_MAX_PARAM_LEN]) for k, v in flatten(params).items()]
            for i in range(0, len(flat), _PARAM_BATCH):
                self.client.log_batch(self.mlflow_run_id, params=flat[i : i + _PARAM_BATCH])
        except Exception:
            log.exception("mlflow start failed for run %s; tracking disabled", run_id)
            if self.mlflow_run_id is None:
                return None
            # run exists but params failed: keep tracking the run
        return self.mlflow_run_id

    def metrics(self, items: list[Metric]) -> None:
        if self.mlflow_run_id is None or not items:
            return
        try:
            if len(items) == 1:
                m = items[0]
                self.client.log_metric(
                    self.mlflow_run_id, f"{m.split}/{m.name}", m.value, timestamp=int(m.ts * 1000), step=m.step
                )
                return
            from mlflow.entities import Metric as MlflowMetric

            batch = [MlflowMetric(f"{m.split}/{m.name}", m.value, int(m.ts * 1000), m.step) for m in items]
            for i in range(0, len(batch), _METRIC_BATCH):
                self.client.log_batch(self.mlflow_run_id, metrics=batch[i : i + _METRIC_BATCH])
        except Exception:
            log.exception("mlflow metrics logging failed")

    def finish(self, state: str) -> None:
        if self.mlflow_run_id is None:
            return
        try:
            self.client.set_terminated(self.mlflow_run_id, _STATUS.get(state, "FAILED"))
        except Exception:
            log.exception("mlflow finish failed")


def make_forwarder(settings) -> Forwarder:
    uri = settings.mlflow_tracking_uri
    return MlflowForwarder(uri) if uri else NullForwarder()
