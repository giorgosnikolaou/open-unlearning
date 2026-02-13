"""LKF Retain Quality Metric using LLM judge (local or Gemini)."""
import logging
import os
from typing import Any, Dict

from tqdm import tqdm

from evals.metrics.base import unlearning_metric
from evals.metrics.lkf.judge_eval import EvalJUDGE
from evals.metrics.lkf.utils import Evaluator as LKFEvaluator
from evals.metrics.lkf.utils import get_api_key
from evals.metrics.lkf.worst_eval import AvgEval

logger = logging.getLogger(__name__)


@unlearning_metric(name="retain_quality_lkf")
def retain_quality_lkf(model, **kwargs) -> Dict[str, Any]:
    """Evaluate retain quality using LKF methodology with LLM judge (local or Gemini).

    This metric evaluates how well a model retains knowledge on the retain set by:
    1. Generating responses to retain-set questions
    2. Using an LLM judge to assess if responses are correct
    3. Computing average-case metric (J_avg)

    Args:
        model: The model to evaluate.
        **kwargs: Additional arguments including:
            - tokenizer: Model tokenizer
            - data: Retain dataset
            - output_dir: Directory for saving results
            - gemini_api_key: Optional Gemini API key (only for gemini judge)
            - dataset_name: HF dataset name
            - dataset_split: HF dataset split
            - eval_task: Task identifier
            - judge: Judge configuration dict
            - generation: Generation configuration dict
            - output_base: Output paths configuration dict
            - max_samples: Maximum samples to evaluate (default: 400)

    Returns:
        Dict containing:
            - agg_value: Average-case metric J_avg (higher is better)
            - J_avg: Average accuracy
    """
    tokenizer = kwargs["tokenizer"]
    data = kwargs["data"]  # Raw data from HFDataset
    output_dir = kwargs.get("output_dir", "./lkf_eval")
    gemini_api_key = kwargs.get("gemini_api_key")
    max_samples = kwargs.get("max_samples", 400)

    # Extract configuration from kwargs
    dataset_name = kwargs.get("dataset_name")
    dataset_split = kwargs.get("dataset_split", "train")
    eval_task = kwargs.get("eval_task")
    judge_cfg = kwargs.get("judge", {})
    generation_cfg = kwargs.get("generation", {})
    output_base_cfg = kwargs.get("output_base", {})

    # Extract judge configuration
    judge_type = judge_cfg.get("type", "local")
    judge_model = judge_cfg.get("model")
    judge_quantize = judge_cfg.get("quantize", False)
    chunk_size = judge_cfg.get("chunk_size")
    vllm_base_url = judge_cfg.get("vllm_base_url")
    fix_qwen_keys = judge_cfg.get("fix_qwen_keys", True)
    substring_heuristic = judge_cfg.get("substring_heuristic", True)

    # Shared judge instance (created once by LKFEvaluator, avoids reloading)
    judge_instance = kwargs.get("judge_instance")

    # Validate API key only if using Gemini
    api_key = None
    if judge_type == "gemini":
        api_key = get_api_key(gemini_api_key)

    model_name = kwargs.get("model_name", "model")
    task_name = kwargs.get("task_name", "retain_eval")

    # Create evaluator to get paths and generate responses
    evaluator = LKFEvaluator(
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        eval_task=eval_task,
        output_base=output_base_cfg,
        generation=generation_cfg,
        icr=False,
        task='retain'
    )
    gen_file = evaluator.out_path
    judge_file = evaluator.jg_file_path

    # Check if already evaluated
    if os.path.exists(judge_file):
        logger.info(f"Using cached judge results: {judge_file}")
    else:
        # Run generation if needed
        if not os.path.exists(gen_file):
            logger.info("Generating responses for retain set")
            # Generate responses for each question
            for idx, sample in tqdm(enumerate(data)):
                if idx >= max_samples:
                    break
                questions = {k: sample[k] for k in sample.keys() if k != 'answer'}
                batch = {qid: evaluator.get_template(q) for qid, q in questions.items()}
                evaluator.evaluate(model, batch, tokenizer)

            evaluator.save_logs()

        # Run judge evaluation
        if os.path.exists(gen_file):
            logger.info(f"Running {judge_type} judge for retain set")
            judge = EvalJUDGE(
                dataset_name=dataset_name,
                dataset_split=dataset_split,
                eval_task=eval_task,
                output_base=output_base_cfg,
                api_key=api_key,
                task='retain',
                gen_file=gen_file,
                icr=False,
                chunk_size=chunk_size,
                judge_type=judge_type,
                judge_model=judge_model,
                judge_quantize=judge_quantize,
                vllm_base_url=vllm_base_url,
                fix_qwen_keys=fix_qwen_keys,
                substring_heuristic=substring_heuristic,
                judge_instance=judge_instance,
            )
            judge.generate()
            judge_file = judge.jg_file_path
        else:
            raise FileNotFoundError(f"Generation file not found: {gen_file}")

    # Compute average-case metrics
    logger.info("Computing average-case metrics")
    avg_eval = AvgEval(
        run_name=task_name,
        files=[judge_file],
        task='retain',
        max_samples=max_samples
    )
    results = avg_eval.evaluate()

    return {
        "agg_value": results["J_avg"],
        "J_avg": results["J_avg"],
    }
