import json
import logging
import os
from typing import Any, Dict, List, Optional

from evals.metrics.lkf.judge_utils import (
    average_case_acc, 
    check_yes_no,
    compute_any_one_average
)

logger = logging.getLogger(__name__)


class WorstEval:
    """Evaluates language model performance under "worst-case" scenarios, especially for forgetting tasks.

    This class reads evaluation results from two JSON files (typically representing
    evaluations with and without in-context retention, and paraphrased variations).
    It processes these results to determine "yes/no" answers for questions and then
    computes various accuracy metrics, including a "worst-case" accuracy,
    which likely combines performance across different evaluation settings.
    Results are logged to a JSONL file.

    Attributes:
        name: The name or identifier of the model being evaluated.
        out_dir: The base directory where the evaluation results will be saved.
        responsefiles: A list of two file paths to JSON files containing model responses.
        run_name: A unique identifier for the specific evaluation run.
        task: The type of task being evaluated (e.g., 'forget').
        logs: A dictionary to store the final computed metrics for this evaluation.
        out_path: The full file path where the logs will be saved.
    """

    responsefiles: List[str]
    run_name: str
    task: str
    logs: Dict[str, Any]
    out_path: str

    def __init__(
        self,
        run_name: str,
        files: Optional[List[str]] = None,
        task: str = 'forget'
    ) -> None:
        """Initialize the WorstEval.

        Args:
            name: The name or identifier of the model.
            run_name: A unique identifier for the evaluation run.
            worst_dir: Directory where worst-case evaluation results will be saved.
            files: List of two file paths to JSON files with model responses.
            task: The evaluation task type.
        """
        self.responsefiles = files or []
        self.run_name = run_name
        self.task = task
        self.logs = {}

    def evaluate(self) -> Dict[str, Any]:
        """Performs the worst-case evaluation by processing model responses from files.

        Reads two sets of evaluation responses from `self.responsefiles`. For each set,
        it iterates through questions, converts responses to a 'yes/no' format
        using `check_yes_no`, and computes an overall average using `compute_any_one_average`.
        Finally, it computes a combined "worst-case" accuracy across both sets
        and stores all computed metrics in `self.logs` before saving them.

        Returns:
            Dictionary with agg_value and component metrics.
        """
        with open(self.responsefiles[0]) as f:
            evals_para = json.load(f)
        with open(self.responsefiles[1]) as f:
            evals_icr = json.load(f)

        # Align samples by GT to handle mismatched lengths (some chunks may have
        # failed during judging, causing different sample counts per file)
        if len(evals_para) != len(evals_icr):
            logger.warning(
                f"Mismatched sample counts: para={len(evals_para)}, icr={len(evals_icr)}. "
                "Aligning by GT field (only samples present in both files will be used)."
            )
            icr_by_gt = {d["GT"]: d for d in evals_icr}
            aligned_para, aligned_icr = [], []
            for d in evals_para:
                if d["GT"] in icr_by_gt:
                    aligned_para.append(d)
                    aligned_icr.append(icr_by_gt[d["GT"]])
            logger.info(f"Aligned to {len(aligned_para)} common samples")
            evals_para = aligned_para
            evals_icr = aligned_icr

        q_names = list({k: v for k, v in evals_para[0].items() if k != "GT"}.keys())

        try:
            ans_newpara1: Dict[str, List[int]] = {}
            for qi in q_names:
                ans_list = []
                for ix, d in enumerate(evals_para):
                    ans = check_yes_no(d[f'{qi}'], 0)
                    ans_list.append(ans)
                ans_newpara1[f'{qi}'] = ans_list
            accs1, _ = compute_any_one_average(ans_newpara1, prefix="ans_")
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to compute paraphrase accuracy: {e}")
            accs1 = [None]

        logger.info(f"J_P: {accs1[-1]}")

        try:
            ans_newpara2: Dict[str, List[int]] = {}
            for qi in q_names:
                ans_list = []
                for ix, d in enumerate(evals_icr):
                    ans = check_yes_no(d[f'{qi}'], 0)
                    ans_list.append(ans)
                ans_newpara2[f'{qi}'] = ans_list
            accs2, _ = compute_any_one_average(ans_newpara2, prefix="ans_")

        except (KeyError, ValueError) as e:
            logger.error(f"Failed to compute ICR accuracy: {e}")
            accs2 = [None]

        logger.info(f"J_ICR: {accs2[-1]}")

        acc_tens: Dict[str, List[int]] = {}
        for qi in q_names:
            acc_tens[f'outs_1_{qi}'] = ans_newpara1[qi]
            acc_tens[f'outs_2_{qi}'] = ans_newpara2[qi]
        accov2, rowacc = compute_any_one_average(acc_tens)
        logger.info(f"J_W: {accov2[-1]}")

        self.logs = {
            "Task_name": self.run_name,
            "Set": self.task,
            "J_P": accs1[-1],
            "J_ICR": accs2[-1],
            "J_W": accov2[-1],
            "agg_value": accov2[-1],  # Framework-compatible key
        }

        return self.logs


class AvgEval:
    """Evaluates language model performance for "average-case" scenarios, typically for retain set.

    This class reads evaluation results from a single JSON file, processes responses
    to 'yes/no' answers, and computes an "average-case" accuracy.
    Results are logged to a JSONL file, similar to `WorstEval` but focused on average performance.

    Attributes:
        name: The name or identifier of the model being evaluated.
        out_dir: The base directory where the evaluation results will be saved.
        responsefiles: A list containing a single file path to a JSON file with responses.
        run_name: A unique identifier for the specific evaluation run.
        task: The type of task being evaluated (e.g., 'retain').
        logs: A dictionary to store the final computed metrics for this evaluation.
        out_path: The full file path where the logs will be saved.
        max_samples: Maximum number of samples to evaluate (configurable).
    """

    out_dir: str
    responsefiles: List[str]
    run_name: str
    task: str
    logs: Dict[str, Any]
    out_path: str
    max_samples: int

    def __init__(
        self,
        run_name: str,
        files: Optional[List[str]] = None,
        task: str = 'retain',
        max_samples: int = 400
    ) -> None:
        """Initialize the AvgEval.

        Args:
            name: The name or identifier of the model.
            run_name: A unique identifier for the evaluation run.
            worst_dir: Directory where average-case evaluation results will be saved.
            files: List containing a single file path to JSON file with responses.
            task: The evaluation task type.
            max_samples: Maximum number of samples to evaluate (default: 400).
        """
        self.responsefiles = files or []
        self.run_name = run_name
        self.task = task
        self.max_samples = max_samples
        self.logs = {}

    def evaluate(self) -> Dict[str, Any]:
        """Performs the average-case evaluation by processing model responses from a file.

        Reads evaluation responses from the first file in `self.responsefiles`.
        It iterates through questions, converts responses to a 'yes/no' format
        using `check_yes_no`, and computes an overall average accuracy using
        `average_case_acc`. Results are then stored in `self.logs` and saved.

        Returns:
            Dictionary with agg_value and J_avg metric.
        """
        with open(self.responsefiles[0]) as f:
            evals_para = json.load(f)[:self.max_samples]
        q_names = list({k: v for k, v in evals_para[0].items() if k != "GT"}.keys())

        try:
            ans_newpara1: Dict[str, List[int]] = {}
            for qi in q_names:
                ans_list = []
                for ix, d in enumerate(evals_para):
                    ans = check_yes_no(d[f'{qi}'], 0)
                    ans_list.append(ans)
                ans_newpara1[f'{qi}'] = ans_list
            accs = average_case_acc(ans_newpara1, prefix="ans_")
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to compute average accuracy: {e}")
            accs = None

        logger.info(f"J_avg: {accs}")

        self.logs = {
            "Task_name": self.run_name,
            "Set": self.task,
            "J_avg": accs,
            "agg_value": accs,  # Framework-compatible key
        }

        return self.logs
