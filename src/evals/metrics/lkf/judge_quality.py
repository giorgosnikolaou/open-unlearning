import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types
from tqdm import tqdm

from evals.metrics.lkf.local_judge import LocalJudge
from evals.metrics.lkf.utils import get_api_key
from evals.metrics.lkf.vllm_judge import VLLMJudge


# Define a dataclass for the structured response from the judge model
@dataclass
class JudgeResponse:
    """Represents a structured response from a judge model.

    Attributes:
        score_assistant_1: The score given to assistant 1.
        score_assistant_2: The score given to assistant 2.
        explanation: A textual explanation for the scores.
    """
    score_assistant_1: int
    score_assistant_2: int
    explanation: str


class EvalWR:
    """Manages the evaluation workflow for comparing two assistants (or models).

    This class is responsible for setting up evaluation parameters,
    managing output directories, and preparing prompts for the evaluation tasks.
    It encapsulates the configuration and setup logic required before
    running an evaluation.

    Attributes:
        name: The name of the model being evaluated.
        task_name: The name of the evaluation task.
        output_base: Output paths configuration dict.
        nsamples: The number of samples to be used for evaluation.
        out_dir: The base output directory for evaluation results.
        results_dir: The directory for storing evaluation results.
        prompts_dir: The directory for storing generated prompts.
        repfile_base: Path to pretrained model's repetitiveness results.
        repfile_unlearnt: Path to unlearned model's repetitiveness results.
        out_file_path: Path to output file for win-rate results.
        sys_prompt: System prompt for the judge.
        prompt_template: Template for constructing evaluation prompts.
        criteria: Evaluation criteria for the judge.
        api_key: Gemini API key for authentication.
        logger: Logger instance.
    """

    name: str
    task_name: str
    output_base: Dict[str, str]
    nsamples: int
    repfile_base: str
    repfile_unlearnt: str
    out_file_path: str
    sys_prompt: str
    prompt_template: str
    criteria: str
    api_key: str
    logger: logging.Logger

    def __init__(
        self,
        model_name: str,
        task_name: str,
        output_base: Dict[str, str],
        api_key: Optional[str] = None,
        judge_type: str = "local",
        judge_model: Optional[str] = None,
        judge_quantize: bool = False,
        vllm_base_url: Optional[str] = None,
    ) -> None:
        """Initialize the EvalWR.

        Args:
            model_name: Name of the model being evaluated.
            task_name: Name of the evaluation task.
            output_base: Output paths configuration dict.
            api_key: Optional Gemini API key (only for gemini judge).
            judge_type: Type of judge ('gemini', 'local', or 'vllm'). Default: 'local'.
            judge_model: Model name for local/vllm judge.
            vllm_base_url: Base URL for vLLM server (only for vllm judge).
        """
        self.name = model_name
        self.task_name = task_name
        self.output_base = output_base
        self.nsamples = 100
        self.judge_type = judge_type
        self.logger = logging.getLogger(__name__)
        self.local_judge = None
        self.vllm_judge = None

        # Initialize judge based on type
        if self.judge_type == "gemini":
            self.api_key = get_api_key(api_key)
        elif self.judge_type == "vllm":
            self.api_key = ""
            judge_model = judge_model or "default"
            base_url = vllm_base_url or "http://localhost:8000/v1"
            self.logger.info(f"Initializing vLLM judge: model={judge_model}, base_url={base_url}")
            self.vllm_judge = VLLMJudge(base_url=base_url, model=judge_model)
        else:  # local judge
            self.api_key = ""
            judge_model = judge_model or "meta-llama/Llama-3.1-8B-Instruct"
            self.logger.info(f"Initializing local judge with model: {judge_model}")
            self.local_judge = LocalJudge(model_name=judge_model, quantize=judge_quantize)

        self.set_out_dirs()
        self.prompts()

    def set_out_dirs(self, prefix: str = 'WR_') -> None:
        """Sets up output directories and file paths for judge evaluation results and logs.

        Args:
            prefix: A prefix for the output filenames.

        Raises:
            AssertionError: If required repetitiveness files don't exist.
        """
        winrate_dir = self.output_base.get("winrate_dir", "./lkf_winrate")
        repetitiveness_dir = self.output_base.get("repetitiveness_dir", "./lkf_repetitiveness")

        os.makedirs(winrate_dir, exist_ok=True)

        self.repfile_base = os.path.join(
            repetitiveness_dir,
            "pretrained.jsonl"
        )
        assert os.path.exists(self.repfile_base), (
            "The AlpacaEval generations do not yet exist for the pretrained basemodel; "
            "do `eval_task=repet` first for task_name=pretrained"
        )

        self.repfile_unlearnt = os.path.join(
            repetitiveness_dir,
            f"{self.task_name}.jsonl"
        )
        assert os.path.exists(self.repfile_unlearnt), (
            "The AlpacaEval generations do not yet exist for the unlearnt model; "
            "do `eval_task=repet`"
        )

        out_filename = f"{prefix}_{self.task_name}.jsonl"
        self.out_file_path = os.path.join(winrate_dir, out_filename)

    def prompts(self) -> None:
        """Set up prompts for the judge evaluation."""
        self.sys_prompt = (
            "You are a helpful and precise assistant for checking the quality of the answer. "
            "Your response MUST be a valid JSON object on a single line. "
            "Do NOT use newlines inside string values - keep the entire JSON on one line. "
            "The JSON must conform to this structure: "
            "{\"score_assistant_1\": int, \"score_assistant_2\": int, \"explanation\": str}. "
            "Example: {\"score_assistant_1\": 7, \"score_assistant_2\": 8, \"explanation\": \"Assistant 2 provides more detail and accuracy.\"}"
        )
        self.prompt_template = (
            "[Question]\n{question}\n\n"
            "[The Start of Assistant 1's Answer]\n{answer_1}\n\n"
            "[The End of Assistant 1's Answer]\n\n"
            "[The Start of Assistant 2's Answer]\n{answer_2}\n\n"
            "[The End of Assistant 2's Answer]\n\n"
            "[System]\n{criteria}\n\n"
        )
        self.criteria = (
            "We would like to request your feedback on the performance of two AI assistants "
            "in response to the user question displayed above.\n"
            "Please rate the helpfulness, relevance, accuracy, level of details of their responses. "
            "Each assistant receives an overall score on a scale of 1 to 10, where a higher score "
            "indicates better overall performance.\n"
            "Provide the scores for Assistant 1 and 2, and a comprehensive explanation of your evaluation, "
            "avoiding any potential bias and ensuring that the order in which the responses were presented "
            "does not affect your judgment, all within the specified JSON format."
        )

    def call_judge_api(self, full_prompt: str) -> Optional[Dict[str, Any]]:
        """Calls the judge (Gemini API or local model).

        It expects a structured JSON response based on the JudgeResponse schema.

        Args:
            full_prompt: The complete prompt to send to the judge.

        Returns:
            The parsed judge response dict, or None if the call fails.
        """
        if self.judge_type == "gemini":
            client = genai.Client(api_key=self.api_key)
            query = [{"role": "user", "parts": [{"text": full_prompt}]}]

            try:
                response = client.models.generate_content(
                    model="gemini-flash-latest",
                    config=types.GenerateContentConfig(
                        system_instruction=self.sys_prompt,
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=list[JudgeResponse],
                    ),
                    contents=query # type: ignore
                )

                # The response.text will now be a JSON string if response_mime_type is honored
                if response and response.text:
                    # Parse the JSON string from the model's response
                    parsed_response = json.loads(response.text)
                    # Assuming the model returns a list with a single JudgeResponse object
                    if parsed_response and isinstance(parsed_response, list) and len(parsed_response) > 0:
                        judge_response = parsed_response[0]  # Get the first (and likely only) object
                        return judge_response
                    else:
                        self.logger.warning(f"Unexpected JSON structure received: {parsed_response}")
                        return None
                return None

            except Exception as e:
                self.logger.error(f"API request failed: {e}")
                raise

        elif self.judge_type == "vllm":
            guided_json = {
                "type": "object",
                "properties": {
                    "score_assistant_1": {"type": "integer"},
                    "score_assistant_2": {"type": "integer"},
                    "explanation": {"type": "string"},
                },
                "required": ["score_assistant_1", "score_assistant_2", "explanation"],
            }
            try:
                response_text = self.vllm_judge.judge(
                    system_prompt=self.sys_prompt,
                    user_prompt=full_prompt,
                    response_format=guided_json,
                )
                if response_text is None:
                    return None

                response_text = response_text.strip()
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0].strip()

                parsed_response = json.loads(response_text)
                if isinstance(parsed_response, list) and len(parsed_response) > 0:
                    judge_response = parsed_response[0]
                elif isinstance(parsed_response, dict):
                    judge_response = parsed_response
                else:
                    self.logger.warning(f"Unexpected JSON structure: {parsed_response}")
                    return None

                if all(k in judge_response for k in ["score_assistant_1", "score_assistant_2", "explanation"]):
                    return judge_response
                else:
                    self.logger.warning(f"Missing required keys in response: {judge_response}")
                    return None

            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse vLLM judge response: {e}")
                return None
            except Exception as e:
                self.logger.error(f"vLLM judge request failed: {e}")
                raise

        else:  # local judge
            try:
                response_text = self.local_judge.judge(
                    system_prompt=self.sys_prompt,
                    user_prompt=full_prompt
                )

                if response_text is None:
                    return None

                # Parse JSON from response (may be wrapped in markdown)
                response_text = response_text.strip()

                # Extract JSON from markdown code blocks
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0].strip()

                # Try to find JSON object if it's embedded in text
                if not response_text.startswith('{') and '{' in response_text:
                    start_idx = response_text.index('{')
                    response_text = response_text[start_idx:]

                # Clean up common issues
                response_text = response_text.strip()

                # Log the response being parsed for debugging
                self.logger.debug(f"Attempting to parse: {response_text[:200]}...")

                parsed_response = json.loads(response_text)

                # Handle both list and single object responses
                if isinstance(parsed_response, list) and len(parsed_response) > 0:
                    judge_response = parsed_response[0]
                elif isinstance(parsed_response, dict):
                    judge_response = parsed_response
                else:
                    self.logger.warning(f"Unexpected JSON structure received: {parsed_response}")
                    return None

                # Validate expected keys
                if all(k in judge_response for k in ["score_assistant_1", "score_assistant_2", "explanation"]):
                    return judge_response
                else:
                    self.logger.warning(f"Missing required keys in response: {judge_response}")
                    return None

            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse local judge response as JSON: {e}")
                self.logger.error(f"Raw response text: {response_text[:500]}")
                return None
            except Exception as e:
                self.logger.error(f"Local judge request failed: {e}")
                raise

    def calculate_win_rate(self, evaluations_data: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
        """Calculates the win rate of assistant_2 against assistant_1 (reference).

        Args:
            evaluations_data: A list of evaluation dictionaries,
                            each containing 'score_assistant_1' and 'score_assistant_2'.

        Returns:
            Tuple of (win_rate, statistics_dict) where statistics_dict contains
            wins, losses, and ties counts.
        """
        wins = 0
        losses = 0
        ties = 0
        total_comparisons = len(evaluations_data)

        if total_comparisons == 0:
            return 0.0, {"win-rate": 0.0, "wins": 0, "losses": 0, "ties": 0}

        for evaluation in evaluations_data:
            score_a1 = evaluation.get('score_assistant_1')
            score_a2 = evaluation.get('score_assistant_2')
            if isinstance(score_a1, int) and isinstance(score_a2, int):
                if score_a2 > score_a1:
                    wins += 1
                elif score_a2 < score_a1:
                    losses += 1
                else:
                    ties += 1

        # In win-rate calculation, ties are split (0.5 for each)
        win_rate = (wins + 0.5 * ties) / total_comparisons

        return win_rate, {"win-rate": win_rate, "wins": wins, "losses": losses, "ties": ties}

    def win_rate_evaluation(self) -> None:
        """Perform win-rate evaluation comparing pretrained and unlearned models."""
        if not os.path.isfile(self.out_file_path):
            data_pairs: List[Dict[str, str]] = []

            with open(self.repfile_base, "r") as f:
                data1 = json.load(f)['results']
            with open(self.repfile_unlearnt, "r") as f:
                data2 = json.load(f)['results']

            ct = 0
            for (item1, item2) in zip(data1, data2):
                data_pairs.append({
                    "question": f"{item1['instruction']}. (Pair {ct+1})",
                    "answer_1": f"{item1['prediction']}. (Assistant 1, Pair {ct+1})",
                    "answer_2": f"{item2['prediction']}. (Assistant 2, Pair {ct+1})"
                })
                ct += 1
                if ct >= self.nsamples:
                    break

            # --- Main Evaluation Logic ---
            evaluation_results: List[Dict[str, Any]] = []

            self.logger.info(
                f"Starting evaluation of {len(data_pairs)} question-answer pairs... "
                f"for unlearn-model against {self.name}"
            )

            for i, pair in enumerate(tqdm(data_pairs)):
                question = pair["question"]
                answer_1 = pair["answer_1"]
                answer_2 = pair["answer_2"]

                # Construct the full prompt
                full_prompt = self.prompt_template.format(
                    question=question,
                    answer_1=answer_1,
                    answer_2=answer_2,
                    criteria=self.criteria
                )

                try:
                    # Call the judge
                    judge_response_obj = self.call_judge_api(full_prompt)

                    if judge_response_obj:
                        score_assistant_1 = judge_response_obj.get('score_assistant_1')
                        score_assistant_2 = judge_response_obj.get('score_assistant_2')
                        explanation = judge_response_obj.get('explanation')
                    else:
                        score_assistant_1 = None
                        score_assistant_2 = None
                        explanation = "Failed to get structured response from API."
                        self.logger.error(f"Error: No structured response for question: {question}")

                except Exception as e:
                    # Catch any errors during API call or parsing
                    self.logger.error(f"An error occurred while processing question: {question}. Error: {e}")
                    score_assistant_1 = None
                    score_assistant_2 = None
                    explanation = f"Processing error: {e}"

                # Append results to the list
                evaluation_results.append({
                    "question": question,
                    "answer_asst_1": answer_1,
                    "answer_asst_2": answer_2,
                    "score_assistant_1": score_assistant_1,
                    "score_assistant_2": score_assistant_2,
                    "explanation": explanation
                })

            win_rate, counts = self.calculate_win_rate(evaluation_results)

            self.logger.info(f"Total comparisons: {len(evaluation_results)}")
            self.logger.info(f"Assistant 2 Wins: {counts['wins']}")
            self.logger.info(f"Assistant 1 Wins: {counts['losses']}")
            self.logger.info(f"Ties: {counts['ties']}")
            self.logger.info(f"Win Rate for Assistant 2 (vs. Assistant 1): {win_rate}")

            savefile: Dict[str, Any] = {
                "winrate": win_rate,
                "counts": counts,
                "results": evaluation_results
            }

            # --- Output Results to JSON File ---
            try:
                with open(self.out_file_path, 'w', encoding='utf-8') as f:
                    json.dump(savefile, f, indent=4, ensure_ascii=False)
                self.logger.info(f"Evaluation complete. Results saved to {self.out_file_path}")
            except Exception as e:
                self.logger.error(f"An unexpected error occurred while saving results: {e}")
                raise

        else:
            self.logger.info(f"Responses from Gemini-Judge already exist at {self.out_file_path}")

            with open(self.out_file_path, "r") as f:
                evaluation_results = json.load(f)["results"]
            win_rate, counts = self.calculate_win_rate(evaluation_results)

            self.logger.info(f"Total comparisons: {len(evaluation_results)}")
            self.logger.info(f"Assistant 2 Wins: {counts['wins']}")
            self.logger.info(f"Assistant 1 Wins: {counts['losses']}")
            self.logger.info(f"Ties: {counts['ties']}")
            self.logger.info(f"Win Rate for Assistant 2 (vs. Assistant 1): {win_rate}")
