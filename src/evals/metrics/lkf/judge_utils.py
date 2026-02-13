from typing import Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt


def compute_any_one_average(
    data: Dict[str, List[int]],
    prefix: str = "outs_"
) -> Tuple[List[float], List[float]]:
    """Compute coverage and row-wise means for binary data.

    Args:
        data: Dictionary mapping keys to lists of binary values (0 or 1).
        prefix: String prefix to filter relevant keys.

    Returns:
        Tuple of (accuracies, row_means) where:
            - accuracies: List of cumulative coverage after each row
            - row_means: List of mean values for each row
    """
    relevant_keys = [k for k in data if k.startswith(prefix)]
    filtered_data = [data[key] for key in relevant_keys if key in data]
    N = len(filtered_data[0])
    coverage = [0] * N
    accuracies: List[float] = []
    row_means = [sum(row) / len(row) for row in filtered_data]

    for row in filtered_data:
        for i in range(N):
            if row[i] == 1:
                coverage[i] = 1
        accuracy = sum(coverage) / N
        accuracies.append(accuracy)

    return accuracies, row_means


def check_yes_no(input_str: str, inside: int = 0) -> int:
    """Convert yes/no string to binary value.

    Args:
        input_str: String to check for yes/no variants.
        inside: If non-zero, always returns 1.

    Returns:
        1 for 'yes' variants, 0 for 'no' variants.

    Raises:
        ValueError: If input is empty or not recognized as yes/no.
    """
    yes_variants = {"yes", "yeah"}
    no_variants = {"no", "nope", "not sure"}

    normalized_input = input_str.strip().lower()

    if not normalized_input:
        raise ValueError("Input cannot be empty or just whitespace.")

    if inside:
        return 1

    if normalized_input in yes_variants:
        return 1
    if normalized_input in no_variants:
        return 0
    else:
        raise ValueError("Input not recognized as 'yes' or 'no'.")


def average_case_acc(
    data: Dict[str, Union[List[int], List[List[int]]]],
    prefix: str = "outs_"
) -> float:
    """Compute average accuracy across filtered data.

    Args:
        data: Dictionary mapping keys to lists or lists of lists of values.
        prefix: String prefix to filter relevant keys.

    Returns:
        The average accuracy as a float.
    """
    relevant_keys = [k for k in data if k.startswith(prefix)]
    filtered_data = [data[key] for key in relevant_keys if key in data]

    if isinstance(filtered_data[0], list):
        arr: npt.NDArray[np.float64] = np.array(filtered_data)
        # Step 1: Mean across rows (axis=0)
        mean_across = np.mean(arr, axis=0)
        # Step 2: Mean across all values
        final_mean = np.mean(mean_across)
    else:
        arr = np.array(filtered_data)
        final_mean = np.mean(arr, axis=0)

    return float(final_mean)


