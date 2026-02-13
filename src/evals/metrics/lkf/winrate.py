"""LKF Win Rate Metric comparing unlearned model against pretrained baseline."""
import logging
from typing import Dict, Any

from evals.metrics.base import unlearning_metric
from evals.metrics.lkf.judge_quality import EvalWR


logger = logging.getLogger(__name__)


@unlearning_metric(name="winrate_lkf")
def winrate_lkf(model, **kwargs) -> Dict[str, Any]:
    """Evaluate win rate of unlearned model vs pretrained baseline using LLM judge (local or Gemini).

    This metric compares the quality of responses from an unlearned model against
    a pretrained baseline by:
    1. Loading repetitiveness evaluation results for both models
    2. Using an LLM judge to assess which model's response is better
    3. Computing win rate (where ties count as 0.5 wins)

    Note: This metric requires that repetitiveness_lkf has been run for both the
    pretrained baseline and the unlearned model.

    Args:
        model: The model to evaluate (not directly used, serves as context).
        **kwargs: Additional arguments including:
            - gemini_api_key: Optional Gemini API key (only for gemini judge)
            - eval_task: Task identifier
            - judge: Judge configuration dict
            - output_base: Output paths configuration dict
            - model_name: Name of the model being evaluated
            - task_name: Name of the evaluation task

    Returns:
        Dict containing:
            - agg_value: Win rate (0-1, higher is better)
            - wins: Number of wins for unlearned model
            - losses: Number of losses
            - ties: Number of ties
            - total_comparisons: Total comparisons made
    """
    gemini_api_key = kwargs.get("gemini_api_key")

    # Extract configuration from kwargs
    eval_task = kwargs.get("eval_task")
    judge_cfg = kwargs.get("judge", {})
    output_base_cfg = kwargs.get("output_base", {})
    model_name = kwargs.get("model_name", "model")
    task_name = kwargs.get("task_name", "model")

    # Extract judge configuration
    judge_type = judge_cfg.get("type", "local")
    judge_model = judge_cfg.get("model")
    vllm_base_url = judge_cfg.get("vllm_base_url")

    logger.info(f"Starting win-rate evaluation with {judge_type} judge")

    # Initialize evaluator (file existence checks happen in EvalWR.__init__)
    evaluator = EvalWR(
        model_name=model_name,
        task_name=task_name,
        output_base=output_base_cfg,
        api_key=gemini_api_key,
        judge_type=judge_type,
        judge_model=judge_model,
        vllm_base_url=vllm_base_url,
    )

    # Run win-rate evaluation (loads cached results if available)
    evaluator.win_rate_evaluation()

    # The win_rate_evaluation() method doesn't return values, so we need to
    # load the results from the output file
    import json
    with open(evaluator.out_file_path, 'r') as f:
        results = json.load(f)

    win_rate = results["winrate"]
    counts = results["counts"]

    logger.info(f"Win-rate evaluation complete: {win_rate:.4f}")

    return {
        "agg_value": win_rate,
        "win_rate": win_rate,
        "wins": counts["wins"],
        "losses": counts["losses"],
        "ties": counts["ties"],
        "total_comparisons": len(results["results"]),
    }