"""
Straight-forward LOSS attack, as described in https://ieeexplore.ieee.org/abstract/document/8429311
"""

from evals.metrics.mia.all_attacks import Attack
from evals.metrics.utils import evaluate_probability


class LOSSAttack(Attack):
    def compute_batch_values(self, batch):
        """Compute probabilities and losses for the batch."""
        return evaluate_probability(self.model, batch)

    def compute_score(self, sample_stats):
        """Return the average loss for the sample."""
        return sample_stats["avg_loss"]


class TotalLOSSAttack(Attack):
    """LOSS attack using total (summed) NLL as in the RWKU paper (Jin et al., 2024)."""

    def compute_batch_values(self, batch):
        return evaluate_probability(self.model, batch)

    def compute_score(self, sample_stats):
        """Return the total (summed) loss for the sample."""
        return sample_stats["total_loss"]
