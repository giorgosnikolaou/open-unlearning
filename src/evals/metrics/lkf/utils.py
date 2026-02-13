import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer


def get_api_key(api_key: Optional[str] = None) -> str:
    """Get Gemini API key from parameter or environment.

    Args:
        api_key: Optional API key. If None, will check GOOGLE_API_KEY environment variable.

    Returns:
        The API key string.

    Raises:
        ValueError: If no API key is provided and GOOGLE_API_KEY is not set.
    """
    if api_key is not None:
        return api_key

    key = os.environ.get("GOOGLE_API_KEY")
    if key is None:
        raise ValueError(
            "Gemini API key required. Set GOOGLE_API_KEY environment "
            "variable or pass via api_key parameter."
        )
    return key


def setup_logger(
    log_file_name: str = "application.log",
    log_dir: str = "./",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """Sets up a logger that prints to the console and saves to a file.

    Args:
        log_file_name: The name of the log file.
        console_level: The minimum logging level for console output.
        file_level: The minimum logging level for file output.

    Returns:
        Configured logger instance.
    """
    # Get the logger instance
    logger = logging.getLogger("LKFEVAL")
    logger.setLevel(logging.DEBUG)  # Set overall level to lowest
    logger.propagate = True  # Allow messages to propagate to parent logger for console output

    # Ensure the logs directory exists
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, log_file_name)

    # Ensure the logger doesn't add handlers multiple times
    if not logger.handlers:
        # Only add file handler - console output handled by parent logger
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger.info(f"Logger '{logger.name}' set up")
    return logger


TEMP_P = (
    "You are a good model. I want you to answer this question with a short response. "
    "The output should just be the answer without any preceding/succeeding text. "
    "Respond with a short, relevant answer only. "
    "A few examples: "
    "1. Question: Where did Olympics 2012 happen? Answer: London "
    "2. Question: What is the capital city of Australia? Answer: The capital is Canberra "
    "3. Question: Which year did World-war 2 end? Answer: 1945 "
    "Now it's your turn. "
)


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Evaluator:
    """Handles the evaluation process for a language model across forget/retain tasks.

    This class manages loading evaluation data, constructing prompts (with or without
    in-context examples), running model inference, logging results, and managing
    output directories for evaluation outcomes. It supports evaluation for "forget"
    and other tasks, and can incorporate in-context retention (ICR) data.

    Attributes:
        name: The name of the model being evaluated.
        icr_data: Flag indicating whether in-context retention (ICR) data is used.
        dataset_name: HuggingFace dataset name.
        dataset_split: Dataset split to use.
        eval_task: Task identifier (e.g., 'forget', 'retain').
        task: The specific evaluation task (e.g., 'forget', 'retain').
        output_base: Output paths configuration dict.
        generation: Generation configuration dict.
        icr_dataset: Dataset containing ICR examples, loaded on init if icr_data=True.
        logs: A list to store dictionaries of generated responses for logging.
        out_path: Full path to the main output JSONL file for generations.
        jg_file_path: Full path to the JudgeEvals evaluation JSONL file.
        log_path: Full path to the evaluation log file.
        logger: Logger instance for logging evaluation progress.
        TEMP_P: Template prompt string.
    """

    icr_data: bool
    dataset_name: str
    dataset_split: str
    eval_task: str
    task: str
    output_base: Dict[str, str]
    generation: Dict[str, Any]
    icr_dataset: Optional[Dataset]
    logs: List[Dict[str, Any]]
    out_path: str
    jg_file_path: str
    log_path: str
    logger: logging.Logger
    TEMP_P: str

    def __init__(
        self,
        dataset_name: str,
        dataset_split: str,
        eval_task: str,
        output_base: Dict[str, str],
        generation: Dict[str, Any],
        icr: bool,
        task: str
    ) -> None:
        """Initialize the Evaluator.

        Args:
            dataset_name: HuggingFace dataset name.
            dataset_split: HuggingFace dataset split.
            eval_task: Task identifier.
            model_name: Name of the model being evaluated.
            output_base: Output paths configuration dict.
            generation: Generation configuration dict.
            icr: Whether to use in-context retention examples.
            task: The evaluation task name (e.g., 'forget', 'retain').
        """
        self.icr_data = icr
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.eval_task = eval_task
        self.task = task
        self.output_base = output_base
        self.generation = generation
        self.icr_dataset = self.load_icr() if self.icr_data else None
        self.logs = []
        self.set_out_dirs()
        self.set_template()
        self.logger = setup_logger(self.log_path, self.output_base.get("logs_dir", "./logs"))
        self.logger.info(f"Output filepath {self.log_path}")
        self.logger.info(f"Evaluating {self.task} with ICR? {self.icr_data}")

        seed_everything()

    def set_template(self) -> None:
        """Set the prompt template."""
        self.TEMP_P = TEMP_P

    def load_icr(self) -> Dataset:
        """Loads the in-context retention (ICR) dataset.

        The dataset is loaded from "nmndeep/LKF-retain_standard" split 'train'
        using the Hugging Face datasets library. Used only when self.icr_data=True
        and self.task == 'forget'.

        Returns:
            The loaded Hugging Face Dataset for ICR.
        """
        dataset = load_dataset("nmndeep/LKF-retain_standard", split='train')
        return dataset # type: ignore

    def get_ret_icr(self, num_ic_examples: int = 3) -> List[List[str]]:
        """Retrieves a specified number of in-context retention examples.

        Randomly samples `num_ic_examples` examples from the loaded `icr_dataset` and
        formats them as a list of `[question, answer]` pairs.

        Args:
            num_ic_examples: The number of in-context examples to retrieve.

        Returns:
            A list of lists, where each inner list contains a question and answer pair.
        """
        sampled_dataset = self.icr_dataset.shuffle().select(range(num_ic_examples)) # type: ignore
        icr_examples = [
            [queries['question'], queries['answer']] # type: ignore
            for queries in sampled_dataset
        ]

        return icr_examples

    def get_template(self, ex: str) -> str:
        """Constructs a prompt template for the LLM.

        The template includes a base prefix (`TEMP_P`). If `self.icr_data` is True,
        it prepends a set of in-context examples retrieved by `get_ret_icr`.

        Args:
            ex: The specific question or example to be included in the prompt.

        Returns:
            The fully constructed prompt string ready for model inference.
        """
        if not self.icr_data:
            return f"{self.TEMP_P}QUESTION:{ex}, \n ANSWER:"

        else:
            num_ic_examples = 3
            in_context_examples = self.get_ret_icr(num_ic_examples)
            ic_string_parts = [
                f"{idx + 4} Question: {q} Answer: {a}"
                for idx, (q, a) in enumerate(in_context_examples)
            ]
            ic_examples_str = "\n".join(ic_string_parts)

            return f"{self.TEMP_P}{ic_examples_str}\n\nNow it's your turn." + f"QUESTION:{ex}, ANSWER: "

    def set_out_dirs(self) -> None:
        """Sets up output directories and file paths for evaluation results and logs.

        Creates necessary directories and constructs file paths for generated outputs
        (JSONL), JudgeGains evaluation files (JSONL), and log files, based on
        the model name, task name, evaluation task, and ICR usage.

        Args:
            prefix: A prefix for the output filenames.
        """
        generations_dir = self.output_base.get("generations_dir", "./lkf_generations")
        judgments_dir = self.output_base.get("judgments_dir", "./lkf_judgments")

        os.makedirs(generations_dir, exist_ok=True)
        os.makedirs(judgments_dir, exist_ok=True)

        icr_suffix = f'icr_{self.icr_data}'
        filename = f"{self.eval_task}_{icr_suffix}.jsonl"

        self.out_path = os.path.join(generations_dir, filename)
        self.jg_file_path = os.path.join(judgments_dir, filename)
        self.log_path = f"{icr_suffix}.log"

    def everything_evaluated(self) -> bool:
        """Checks if all necessary evaluation results already exist on disk.

        For the 'forget' task, it checks for both the current `jg_file_path`
        and its counterpart with `_icr_False`. For other tasks, it only checks
        the current `jg_file_path`. This is used to skip re-evaluation if results are cached.

        Returns:
            True if the evaluation results are complete and exist, False otherwise.
        """
        primary_file_exists = os.path.exists(self.jg_file_path)

        if self.task == 'forget':
            other_file_path = self.jg_file_path.replace("True", "False")
            other_file_exists = os.path.exists(other_file_path)
            return primary_file_exists and other_file_exists
        else:
            return primary_file_exists

    def load_logs_from_file(self) -> Tuple[bool, bool]:
        """Returns the cache status of existing results.

        Returns:
            Tuple of (generation_exists, judge_eval_exists).
        """
        gen_exists = os.path.exists(self.out_path)
        jg_eval_exists = os.path.exists(self.jg_file_path)

        if gen_exists:
            self.logger.info(f"Logs for this setup seem to exist!")
            self.logger.info(f"Existing evaluations are at {self.log_path}")
            self.logger.info(f"Checking JudgeEvals")
            if jg_eval_exists:
                self.logger.info(f"JudgeEvals also exist")
            else:
                jg_eval_exists = False
            return (gen_exists, jg_eval_exists)
        else:
            return (gen_exists, jg_eval_exists)

    def save_logs(self) -> None:
        """Save the logs to a JSON file."""
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        with open(self.out_path, "w") as f:
            json.dump(self.logs, f, indent=4)
        self.logger.info(f"✅ Saved {len(self.logs)} generations to: {self.out_path}")

    def evaluate(
        self,
        model: PreTrainedModel,
        batch: Dict[str, str],
        tokenizer: PreTrainedTokenizer
    ) -> None:
        """Generates responses from the model for a batch of prompts.

        Takes a batch of prompts, tokenizes them, and uses the provided model
        to generate text based on generation parameters.
        The generated responses are then extracted and appended to `self.logs`.

        Args:
            model: The language model to use for generation.
            batch: A dictionary where keys are unique identifiers (e.g., question IDs)
                  and values are the prompt strings.
            tokenizer: The tokenizer corresponding to the model.
        """
        prompts = list(batch.values())
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to('cuda')

        with torch.no_grad():
            outputs = model.generate(
                **inputs, # type: ignore
                max_new_tokens=self.generation.get("max_new_tokens", 128),
                do_sample=self.generation.get("do_sample", False),
                top_p=0,
                temperature=self.generation.get("temperature", 0.0),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        extracted_response = []
        for x in decoded:
            extracted_response.append(x.split("ANSWER:")[-1].strip())
        result_dict = {f'ans_{qid}': response for qid, response in zip(batch.keys(), extracted_response)}


        self.logs.append(result_dict)
