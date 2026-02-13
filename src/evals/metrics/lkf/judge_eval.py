import json
import logging
import os
import warnings
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset
from pydantic import BaseModel

from evals.metrics.lkf.local_judge import LocalJudge
from evals.metrics.lkf.utils import get_api_key, setup_logger
from evals.metrics.lkf.vllm_judge import VLLMJudge

warnings.filterwarnings("ignore")

from google import genai
from google.genai import types
from openai import OpenAI

# Suppress httpx request logging (used internally by OpenAI SDK)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (input, output) for OpenAI models
OPENAI_PRICING = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1": (2.00, 8.00),
}


class JudgeResponseForget(BaseModel):
    """Pydantic model for structuring the expected JSON response from the Judge LLM for 'forget' tasks.

    This model defines the expected fields and their types for each evaluation sample
    when assessing model forgetting. Each field corresponds to a question and its
    binary (Yes/No) judgment by the LLM judge.
    """
    ans_question: str
    ans_q_qwen1: str
    ans_q_phi1: str
    ans_q_mist1: str
    ans_q_qwen2: str
    ans_q_phi2: str
    ans_q_mist2: str  # Fixed: was ans_q_mist3 (duplicate)
    ans_q_qwen3: str
    ans_q_phi3: str
    ans_q_mist3: str
    ans_q_qwen4: str
    ans_q_phi4: str
    ans_q_mist4: str
    ans_q_qwen5: str
    ans_q_phi5: str
    ans_q_mist5: str


class JudgeResponseRetain(BaseModel):
    """Pydantic model for structuring the expected JSON response from the Judge LLM for 'retain' tasks.

    This model defines the expected fields and their types for each evaluation sample
    when assessing model retention. The retain eval dataset (LKF-retain_eval_para) contains
    the original question plus 6 paraphrase variants (from Qwen, Phi, Mistral).
    J_Avg is computed as the mean across these 7 variants per sample, then across samples.
    """
    ans_question: str
    ans_q_qwen1: str
    ans_q_phi1: str
    ans_q_mist1: str
    ans_q_qwen2: str
    ans_q_phi2: str
    ans_q_mist3: str


class EvalJUDGE:
    """Uses a Large Language Model (LLM) as a judge to evaluate the similarity or correctness
    of responses from unlearning methods compared to ground truth.

    This class orchestrates the process of fetching model-generated responses,
    preparing prompts for an LLM judge (Gemini in this case), sending chunks
    of data to the judge, parsing its structured 'Yes/No' judgments, and
    saving the results. It supports both 'forget' and 'retain' evaluation tasks.

    Attributes:
        name: The name of the model whose responses are being judged.
        dataset_name: HuggingFace dataset name.
        dataset_split: Dataset split to use.
        eval_task: Task identifier (e.g., 'forget', 'retain').
        output_base: Output paths configuration dict.
        task: The specific evaluation task ('forget' or 'retain').
        icr_data: Flag indicating if in-context retention data was used during generation.
        gen_file: Path to the JSON file containing the generated responses to be judged.
        logs: A temporary list for results (though formatted_response is mainly used).
        jg_file_path: Full path to the output JSONL file for judge evaluations.
        log_path: Full path to the log file for this evaluation run.
        logger: Logger instance for logging evaluation progress.
        chunk_size: The number of samples to send to the judge LLM in a single API call.
        questions: List of question keys loaded from the dataset.
        formatted_response: List to store the parsed and formatted responses from the Judge LLM.
        api_key: Gemini API key for authentication.
        judge_type: Type of judge to use ('gemini' or 'local').
        judge_model: Model name for local judge.
    """

    dataset_name: str
    dataset_split: str
    eval_task: str
    task: str
    icr_data: bool
    gen_file: str
    output_base: Dict[str, str]
    logs: List[Any]
    jg_file_path: str
    log_path: str
    logger: logging.Logger
    chunk_size: int
    questions: List[str]
    formatted_response: List[Dict[str, Any]]
    api_key: str

    def __init__(
        self,
        dataset_name: str,
        dataset_split: str,
        eval_task: str,
        output_base: Dict[str, str],
        api_key: Optional[str] = None,
        task: str = 'forget',
        gen_file: Optional[str] = None,
        icr: bool = False,
        chunk_size: Optional[int] = None,
        judge_type: str = "local",
        judge_model: Optional[str] = None,
        judge_quantize: bool = False,
        vllm_base_url: Optional[str] = None,
        fix_qwen_keys: bool = True,
        substring_heuristic: bool = True,
        judge_instance: Optional[Any] = None,
    ) -> None:
        """Initialize the EvalJUDGE.

        Args:
            dataset_name: HuggingFace dataset name.
            dataset_split: HuggingFace dataset split.
            eval_task: Task identifier.
            output_base: Output paths configuration dict.
            api_key: Optional Gemini API key (only for gemini judge).
            task: The evaluation task ('forget' or 'retain').
            gen_file: Path to the JSON file containing generated responses.
            icr: Whether in-context retention was used.
            chunk_size: Number of samples per API call. If None, uses 3 for forget, 5 for retain.
            judge_type: Type of judge ('gemini', 'local', or 'vllm'). Default: 'local'.
            judge_model: Model name for local/vllm judge.
            vllm_base_url: Base URL for vLLM server (only for vllm judge).
            fix_qwen_keys: Fix common LLM judge typo where 'ans_q_wen' is output
                instead of 'ans_q_qwen'. Default: True.
            substring_heuristic: Override judge verdict to YES when the GT is a
                substring of the predicted answer or vice versa. Default: True.
            judge_instance: Optional pre-built judge (LocalJudge or VLLMJudge).
                If provided, skips model loading — used to share one judge across
                multiple EvalJUDGE instances.
        """
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.eval_task = eval_task
        self.task = task
        self.icr_data = icr
        self.gen_file = gen_file or ""
        self.output_base = output_base
        self.logs = []
        self.judge_type = judge_type
        self.local_judge = None
        self.vllm_judge = None
        self.fix_qwen_keys = fix_qwen_keys
        self.substring_heuristic = substring_heuristic

        self.openai_client = None
        self.openai_model = None
        self.openai_costs = []  # List of per-chunk cost dicts

        # Use pre-built judge if provided, otherwise initialize a new one
        if judge_instance is not None:
            self.api_key = ""
            if self.judge_type == "vllm":
                self.vllm_judge = judge_instance
            elif self.judge_type == "local":
                self.local_judge = judge_instance
            elif self.judge_type == "openai":
                self.openai_client = judge_instance
                self.openai_model = judge_model or "gpt-4o-mini"
        elif self.judge_type == "openai":
            self.api_key = ""
            self.openai_model = judge_model or "gpt-4o-mini"
            self.openai_client = OpenAI()
        elif self.judge_type == "gemini":
            self.api_key = get_api_key(api_key)
        elif self.judge_type == "vllm":
            self.api_key = ""
            judge_model = judge_model or "default"
            base_url = vllm_base_url or "http://localhost:8000/v1"
            logs_dir = self.output_base.get("logs_dir", "./logs")
            self.logger = setup_logger("temp.log", logs_dir)
            self.logger.info(f"Initializing vLLM judge: model={judge_model}, base_url={base_url}")
            self.vllm_judge = VLLMJudge(
                base_url=base_url, model=judge_model, max_new_tokens=1024
            )
        else:  # local judge
            self.api_key = ""
            judge_model = judge_model or "meta-llama/Llama-3.1-8B-Instruct"
            logs_dir = self.output_base.get("logs_dir", "./logs")
            self.logger = setup_logger("temp.log", logs_dir)
            self.logger.info(f"Initializing local judge with model: {judge_model}")
            self.local_judge = LocalJudge(model_name=judge_model, max_new_tokens=1024, quantize=judge_quantize)

        self.set_out_dirs()
        logs_dir = self.output_base.get("logs_dir", "./logs")
        self.logger = setup_logger(self.log_path, logs_dir)
        self.logger.info(f"Output filepath {self.log_path}")
        self.logger.info(f"Evaluating {self.task} with ICR? {self.icr_data}")
        self.logger.info(f"Using judge type: {self.judge_type}")
        self.chunk_size = chunk_size if chunk_size is not None else (3 if self.task == 'forget' else 5)
        self.formatted_response = []

    @staticmethod
    def create_judge_instance(judge_cfg: Dict[str, Any]) -> Optional[Any]:
        """Create a reusable judge instance from config.

        Call this once, then pass the result as ``judge_instance`` to every
        ``EvalJUDGE`` constructor to avoid reloading the model each time.

        Args:
            judge_cfg: Judge configuration dict (from ``eval.lkf.judge``).

        Returns:
            A ``LocalJudge`` or ``VLLMJudge`` instance, or ``None`` for Gemini
            (which uses a stateless API and does not need a shared instance).
        """
        judge_type = judge_cfg.get("type", "local")
        judge_model = judge_cfg.get("model")
        judge_quantize = judge_cfg.get("quantize", False)

        if judge_type == "local":
            judge_model = judge_model or "meta-llama/Llama-3.1-8B-Instruct"
            logger.info(f"Creating shared LocalJudge: {judge_model}")
            return LocalJudge(model_name=judge_model, max_new_tokens=1024, quantize=judge_quantize)
        elif judge_type == "vllm":
            judge_model = judge_model or "default"
            base_url = judge_cfg.get("vllm_base_url", "http://localhost:8000/v1")
            logger.info(f"Creating shared VLLMJudge: {judge_model} @ {base_url}")
            return VLLMJudge(base_url=base_url, model=judge_model, max_new_tokens=1024)
        elif judge_type == "openai":
            logger.info(f"Creating shared OpenAI client: {judge_model or 'gpt-4o-mini'}")
            return OpenAI()
        else:
            # Gemini uses a stateless API client; no persistent instance needed
            return None

    def save_logs(self) -> None:
        """Save the logs to a JSON file."""
        with open(self.jg_file_path, "w") as f:
            json.dump(self.formatted_response, f, indent=4)
        self.logger.info(f"✅ Saved {len(self.formatted_response)} generations to: {self.jg_file_path}")

    def _save_debug_log(self, alternate_json: List[Dict[str, Any]]) -> None:
        """Save a detailed debug log combining inputs and judge verdicts for manual review.

        Creates a JSON file where each entry contains the ground truth, generated answers,
        corresponding questions, and the judge's YES/NO verdict side by side.
        Saved alongside the judgment file as *_debug.json.
        """
        debug_path = self.jg_file_path.replace(".jsonl", "_debug.json")

        if not self.formatted_response:
            self.logger.warning("No judged responses to save in debug log")
            return

        # The judged fields are those the judge schema covers
        judged_field_names = {k for k in self.formatted_response[0].keys() if k.startswith("ans_")}

        # formatted_response is a contiguous subset of alternate_json (failed chunks are gaps).
        # Walk both lists in order; advance judged_idx only on match.
        debug_entries = []
        judged_idx = 0
        for orig_idx, orig_entry in enumerate(alternate_json):
            gt = orig_entry.get("GT", "")
            entry: Dict[str, Any] = {
                "sample_idx": orig_idx,
                "GT": gt,
                "task": self.task,
                "icr": self.icr_data,
            }

            # Match: formatted_response preserves order, so next judged entry
            # corresponds to next orig_entry whose chunk didn't fail.
            matched = (
                judged_idx < len(self.formatted_response)
                and self.formatted_response[judged_idx].get("GT") == gt
            )
            if matched:
                judged = self.formatted_response[judged_idx]
                judged_idx += 1
                entry["judged"] = True
            else:
                judged = None
                entry["judged"] = False

            comparisons = []
            for ans_key in sorted(k for k in orig_entry.keys() if k.startswith("ans_")):
                q_key = ans_key.replace("ans_", "", 1)  # ans_question -> question
                comparison = {
                    "field": ans_key,
                    "question": orig_entry.get(q_key, ""),
                    "generated_answer": orig_entry.get(ans_key, ""),
                }
                if not entry["judged"]:
                    comparison["judge_verdict"] = "CHUNK_FAILED"
                elif ans_key not in judged_field_names:
                    comparison["judge_verdict"] = "NOT_IN_SCHEMA"
                elif judged and ans_key in judged:
                    comparison["judge_verdict"] = judged[ans_key]
                else:
                    comparison["judge_verdict"] = "PARSE_ERROR"
                comparisons.append(comparison)
            entry["comparisons"] = comparisons
            debug_entries.append(entry)

        with open(debug_path, "w") as f:
            json.dump(debug_entries, f, indent=2)
        self.logger.info(
            f"Saved debug log ({len(debug_entries)} entries, {judged_idx} judged, "
            f"{len(debug_entries) - judged_idx} failed) to: {debug_path}"
        )

    @staticmethod
    def _fix_qwen_keys_in_responses(
        parsed_responses: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Fix common LLM judge typo: 'ans_q_wen' -> 'ans_q_qwen'.

        Some local LLMs drop the first 'q' in 'qwen', producing keys like
        'ans_q_wen1' instead of 'ans_q_qwen1'. This method normalises them.
        """
        return [
            {k.replace("_wen", "_qwen"): v for k, v in resp.items()}
            for resp in parsed_responses
        ]

    def _apply_substring_heuristic(
        self,
        formatted_chunk: List[Dict[str, str]],
        original_chunk: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """Override judge verdict to YES when GT and predicted answer share a substring match.

        For each answer field, if the ground truth (case-insensitive, stripped) is a
        substring of the generated answer or vice versa, the verdict is set to YES.
        This catches cases where the judge incorrectly rejects an answer that clearly
        contains the ground truth text.
        """
        overrides = 0
        for judged, orig in zip(formatted_chunk, original_chunk):
            gt = orig.get("GT", "").lower().strip()
            if not gt:
                continue
            for key in list(judged.keys()):
                if not key.startswith("ans_"):
                    continue
                pred = orig.get(key, "").lower().strip()
                if not pred:
                    continue
                if pred in gt or gt in pred:
                    if judged[key] != "YES":
                        overrides += 1
                        judged[key] = "YES"
        if overrides > 0:
            self.logger.info(f"Substring heuristic overrode {overrides} verdicts to YES")
        return formatted_chunk

    def set_out_dirs(self) -> None:
        """Sets up output directories and file paths for judge evaluation results and logs.

        Creates the evaluation output directory if it doesn't exist. It constructs
        the full file paths for the judge evaluation JSONL file and the
        corresponding log file, based on model name, task, and ICR usage.

        Args:
            prefix: A prefix for the output filenames.
        """
        judgments_dir = self.output_base.get("judgments_dir", "./lkf_judgments")
        os.makedirs(judgments_dir, exist_ok=True)

        icr_suffix = f'icr_{self.icr_data}'

        jg_filename = f"{self.eval_task}_{icr_suffix}.jsonl"

        self.jg_file_path = os.path.join(judgments_dir, jg_filename)
        self.log_path = f"{self.eval_task}_{icr_suffix}.log"

    def load_ques(self) -> Dataset:
        """Loads the dataset containing the original questions.

        The dataset is loaded using the dataset name and split provided during initialization.
        It also populates `self.questions` with the keys of the questions from the dataset.

        Returns:
            The loaded Hugging Face Dataset.
        """
        dataset = load_dataset(self.dataset_name, split=self.dataset_split)
        just_ques_list = {k: v for k, v in dataset[0].items() if k != "answer"} # type: ignore
        self.questions = list(just_ques_list.keys())
        return dataset # type: ignore

    def _parse_local_judge_response(self, response_text: str, num_samples: int) -> Optional[List[Dict[str, str]]]:
        """Parse JSON response from local judge model.

        Args:
            response_text: Raw text response from local judge
            num_samples: Expected number of samples in the response

        Returns:
            List of dicts with judge responses, or None if parsing fails
        """
        try:
            # Try to extract JSON from the response
            # Local models might wrap JSON in markdown code blocks
            response_text = response_text.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            parsed = json.loads(response_text)
            if isinstance(parsed, list) and len(parsed) <= num_samples:
                if self.fix_qwen_keys:
                    parsed = self._fix_qwen_keys_in_responses(parsed)
                return parsed
            else:
                self.logger.warning(f"Parsed response is not a list or has wrong length. Expected list of {num_samples}, got: {type(parsed)} with length {len(parsed) if isinstance(parsed, list) else 'N/A'}")
                return None
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse local judge response as JSON: {e}")
            self.logger.warning(f"Raw response: {response_text[:500]}")
            return None

    def generate(self) -> None:
        """Performs the LLM-based judging process.

        This is the core method that:
        1. Initializes the judge (Gemini API client or local model).
        2. Loads model-generated responses and the original question dataset.
        3. Reads the judge's system prompt from a file.
        4. Formats the generated responses and original questions into a structure
           suitable for the judge LLM, including ground truth and test questions/answers.
        5. Iterates through these formatted entries in `self.chunk_size` chunks.
        6. For each chunk, it constructs a query, sends it to the judge model
           with specific generation configurations (e.g., temperature 0, JSON response schema).
        7. Parses the judge's structured JSON response using Pydantic models.
        8. Handles cases where the judge's response is None (indicating an API issue)
           by logging problematic chunks.
        9. Appends the judged results to `self.formatted_response`.
        10. Finally, calls `save_logs` to persist the results.
        """
        # Initialize Gemini client if using Gemini judge
        if self.judge_type == "gemini":
            client = genai.Client(api_key=self.api_key)

        with open(self.gen_file, 'r') as f:
            responses = json.load(f)

        questions_dataset = self.load_ques()

        # TODO: Change the hard-coded path
        with open(f"./src/evals/metrics/lkf/judge_prompt_{self.judge_type}.txt", "r") as file:
            prompt_base = file.read()#.replace("\n", " ")

        alternate_json: List[Dict[str, Any]] = []
        for qs, ent in zip(questions_dataset, responses):
            entry: Dict[str, Any] = {'GT': qs["answer"]} # type: ignore
            for ix in self.questions:
                entry.update({f"{ix}": qs[ix]}) # type: ignore
                ans = ent[f'ans_{ix}'].lower().replace('assistant', "--")
                entry.update({f"ans_{ix}": ans})
            alternate_json.append(entry)

        self.formatted_response = []

        # Determine response schema based on task (moved here for debug logging)
        response_schema_type = JudgeResponseForget if self.task == 'forget' else JudgeResponseRetain

        for i in range(0, len(alternate_json), self.chunk_size):
            chunk = alternate_json[i:i + self.chunk_size]
            chunk_num = i // self.chunk_size + 1
            total_chunks = len(alternate_json) // self.chunk_size + 1

            # Debug: Log answer fields in first chunk to verify data structure
            if chunk_num == 1 and chunk:
                answer_fields = [k for k in chunk[0].keys() if k.startswith('ans_')]
                self.logger.info(f"Answer fields in input data: {sorted(answer_fields)}")
                self.logger.info(f"Expected answer fields for {self.task}: {sorted([f for f in response_schema_type.model_fields.keys()])}")

            self.logger.info(f"Processing chunk {chunk_num} of {total_chunks} of length {len(chunk)}")
            
            # Get list of answer fields from first chunk sample to tell judge exactly what to evaluate
            answer_fields_in_chunk = [k for k in chunk[0].keys() if k.startswith('ans_')]

            # Construct the query for the judge LLM
            query = str(chunk)
            query += (
                f"\n\nREMEMBER: You are a Judge! Evaluate EACH of these answer fields: {', '.join(answer_fields_in_chunk)}. "
                "For each field, compare the answer against the ground truth (GT) and corresponding question. "
                "Return a valid JSON array with one object per sample. "
                f"Each object MUST have ALL {len(answer_fields_in_chunk)} answer fields: {', '.join(answer_fields_in_chunk)}. "
                "Each field value must be \"YES\" or \"NO\". "
                "Do not repeat answers or try to answer questions yourself!"
            )

            # Construct the system instruction part for the judge LLM
            strr = (
                "\n\nHere are the tests to be evaluated. This is a list of JSON objects, where for each sample, "
                "we have:\n"
                "- 'GT': the ground truth answer\n"
                "- Question fields (e.g., 'question', 'q_qwen1', 'q_phi1', etc.)\n"
                "- Answer fields (e.g., 'ans_question', 'ans_q_qwen1', 'ans_q_phi1', etc.) - these are the test responses to evaluate\n"
                "\n"
                "You must return a JSON array where each element corresponds to one input sample. "
                "Each element must be a JSON object with fields matching the answer field names from the input, "
                "and each field value must be \"YES\" or \"NO\" indicating whether that test response contains "
                "all the information from the ground truth.\n"
            )

            full_system_instruction = prompt_base + strr  # Combine base prompt with dynamic instructions

            # Use appropriate judge based on judge_type
            if self.judge_type == "gemini":
                response = client.models.generate_content(
                    model="gemini-flash-latest",
                    config=types.GenerateContentConfig(
                        system_instruction=full_system_instruction,
                        temperature=0.,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                        response_mime_type="application/json",
                        response_schema=list[response_schema_type], # type: ignore
                    ),
                    contents=query
                )
                my_response = response.parsed
            elif self.judge_type == "openai":
                try:
                    response = self.openai_client.chat.completions.create(
                        model=self.openai_model,
                        messages=[
                            {"role": "system", "content": full_system_instruction},
                            {"role": "user", "content": query},
                        ],
                        temperature=0.0,
                    )
                    response_text = response.choices[0].message.content

                    # Track cost
                    usage = response.usage
                    if usage is not None:
                        input_price, output_price = OPENAI_PRICING.get(
                            self.openai_model, (0.0, 0.0)
                        )
                        chunk_cost = (
                            usage.prompt_tokens * input_price
                            + usage.completion_tokens * output_price
                        ) / 1_000_000
                        self.openai_costs.append({
                            "chunk": chunk_num,
                            "prompt_tokens": usage.prompt_tokens,
                            "completion_tokens": usage.completion_tokens,
                            "cost_usd": chunk_cost,
                        })
                        self.logger.info(
                            f"Chunk {chunk_num} cost: ${chunk_cost:.6f} "
                            f"({usage.prompt_tokens} in, {usage.completion_tokens} out)"
                        )
                except Exception as e:
                    self.logger.error(f"OpenAI API call failed: {e}")
                    response_text = None

                if response_text is None:
                    my_response = None
                else:
                    parsed_responses = self._parse_local_judge_response(response_text, len(chunk))
                    if parsed_responses is not None:
                        try:
                            my_response = [response_schema_type(**resp) for resp in parsed_responses]
                        except Exception as e:
                            self.logger.warning(f"OpenAI Pydantic validation failed: {e}")
                            my_response = None
                    else:
                        my_response = None
            elif self.judge_type == "vllm":
                # Build guided JSON schema from Pydantic model for constrained decoding
                guided_json = {
                    "type": "array",
                    "items": response_schema_type.model_json_schema(),
                }
                response_text = self.vllm_judge.judge(
                    system_prompt=full_system_instruction,
                    user_prompt=query,
                    response_format=guided_json,
                )
                if response_text is None:
                    my_response = None
                else:
                    parsed_responses = self._parse_local_judge_response(response_text, len(chunk))
                    if parsed_responses is not None:
                        try:
                            my_response = [response_schema_type(**resp) for resp in parsed_responses]
                        except Exception as e:
                            self.logger.warning(f"vLLM Pydantic validation failed: {e}")
                            my_response = None
                    else:
                        my_response = None
            else:  # local judge
                response_text = self.local_judge.judge(
                    system_prompt=full_system_instruction,
                    user_prompt=query
                )
                if response_text is None:
                    my_response = None
                else:
                    parsed_responses = self._parse_local_judge_response(response_text, len(chunk))
                    if parsed_responses is not None:
                        # Validate that responses match the answer fields in the input data
                        try:
                            # First try strict Pydantic validation
                            my_response = [response_schema_type(**resp) for resp in parsed_responses]
                        except Exception as e:
                            # If Pydantic validation fails, try flexible validation based on actual data fields
                            self.logger.warning(f"Pydantic validation failed: {e}")
                            self.logger.warning(f"Expected schema fields: {list(response_schema_type.model_fields.keys())}")
                            if parsed_responses:
                                self.logger.warning(f"Received fields in first response: {list(parsed_responses[0].keys())}")

                            # Check if response has all the answer fields from input data
                            answer_fields_in_chunk = [k for k in chunk[0].keys() if k.startswith('ans_')]
                            all_valid = True
                            for resp in parsed_responses:
                                resp_fields = set(resp.keys())
                                expected_fields = set(answer_fields_in_chunk)
                                if resp_fields != expected_fields:
                                    missing = expected_fields - resp_fields
                                    extra = resp_fields - expected_fields
                                    if missing:
                                        self.logger.warning(f"Missing fields: {missing}")
                                    if extra:
                                        self.logger.warning(f"Extra fields: {extra}")
                                    all_valid = False
                                    break

                            if all_valid:
                                # Use parsed responses as-is if they match the input data structure
                                self.logger.info("Using flexible validation - responses match input data structure")
                                # Create mock Pydantic-like objects with model_dump method
                                class MockJudgeResponse:
                                    def __init__(self, data):
                                        self._data = data
                                    def model_dump(self):
                                        return self._data
                                my_response = [MockJudgeResponse(resp) for resp in parsed_responses]
                            else:
                                my_response = None
                    else:
                        my_response = None
            if my_response is None:
                # Add problematic chunk index to file if parsing failed but API call succeeded
                logs_dir = self.output_base.get("logs_dir", "./logs")
                problematic_file = os.path.join(logs_dir, "problematic_chunks.txt")
                with open(problematic_file, 'a') as f:
                    f.write(f"Chunk {chunk_num} of {total_chunks} of length {len(chunk)}\n")
                self.logger.warning(f"Failed to parse response for chunk {chunk_num}")
            else:
                formatted_response_chunk = [m.model_dump() for m in my_response] # type: ignore
                for j, single_response in enumerate(formatted_response_chunk):
                    if j == len(chunk):
                        break
                    # Add the original Ground Truth back into the judged response entry
                    single_response.update({"GT": chunk[j]['GT']})

                # Apply substring heuristic: override NO → YES when GT ⊂ answer or answer ⊂ GT
                if self.substring_heuristic:
                    formatted_response_chunk = self._apply_substring_heuristic(
                        formatted_response_chunk, chunk[:len(formatted_response_chunk)]
                    )

                self.formatted_response.extend(formatted_response_chunk)

        self.logger.info(f"LKF {self.judge_type} judge evals done for {self.task} set - iCR {self.icr_data}")
        self.save_logs()
        self._save_debug_log(alternate_json)

        # Save OpenAI cost report
        if self.openai_costs:
            total_cost = sum(c["cost_usd"] for c in self.openai_costs)
            total_prompt = sum(c["prompt_tokens"] for c in self.openai_costs)
            total_completion = sum(c["completion_tokens"] for c in self.openai_costs)
            self.logger.info(
                f"Total OpenAI cost: ${total_cost:.6f} "
                f"({total_prompt} prompt + {total_completion} completion tokens)"
            )
            cost_report = {
                "model": self.openai_model,
                "task": self.task,
                "icr": self.icr_data,
                "total_cost_usd": total_cost,
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "chunks": self.openai_costs,
            }
            icr_suffix = f"icr_{self.icr_data}"
            cost_file = os.path.join(
                self.output_base.get("logs_dir", "./logs"),
                f"openai_costs_{self.task}_{icr_suffix}.json",
            )
            os.makedirs(os.path.dirname(cost_file), exist_ok=True)
            with open(cost_file, "w") as f:
                json.dump(cost_report, f, indent=2)
            self.logger.info(f"Cost report saved to: {cost_file}")
