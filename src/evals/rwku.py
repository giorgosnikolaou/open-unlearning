from evals.base import Evaluator


class RWKUEvaluator(Evaluator):
    def __init__(self, eval_cfg, **kwargs):
        super().__init__("RWKU", eval_cfg, **kwargs)

    def summarize(self, logs):
        summary = super().summarize(logs)
        for metric_name, metric_results in logs.items():
            if metric_name not in self.metrics:
                continue
            for key in ("fm", "rm"):
                if key in metric_results:
                    summary[f"{metric_name}_{key}"] = metric_results[key]
        return summary
