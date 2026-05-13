import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from scorers import TokenImportanceScorer
from data.Cache import Cache


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_kl_divergence(model, target_model, inputs):
    with torch.no_grad():
        ref_outputs = target_model(**inputs)

    ref_probs = F.log_softmax(ref_outputs.logits, dim=-1)
    ref_probs = ref_probs.view(-1, ref_outputs.logits.shape[-1])

    outputs = model(**inputs)
    current_probs = F.log_softmax(outputs.logits, dim=-1)
    current_probs = current_probs.view(-1, outputs.logits.shape[-1])

    # minimum KL divergence
    return nn.functional.kl_div(
        current_probs, ref_probs, reduction="batchmean", log_target=True
    ), outputs


def compute_batch_nll(model, inputs):
    # get the sum loss for each sequence in a batch
    # NOTE: not same as model(**inputs).loss but has sum loss for each seq in a batch
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss, outputs


def compute_dpo_loss(model, ref_model, win_inputs=None, lose_inputs=None, beta=1.0):
    if win_inputs is None and lose_inputs is None:
        raise ValueError("Both win_inputs and lose_inputs can't be None")

    win_log_ratio, lose_log_ratio = 0.0, 0.0
    win_outputs, lose_outputs = None, None

    if win_inputs is not None:
        win_loss, win_outputs = compute_batch_nll(model, win_inputs)
        with torch.no_grad():
            win_ref_loss, _ = compute_batch_nll(ref_model, win_inputs)
        win_log_ratio = -(win_loss - win_ref_loss)

    if lose_inputs is not None:
        lose_loss, lose_outputs = compute_batch_nll(model, lose_inputs)
        with torch.no_grad():
            lose_ref_loss, _ = compute_batch_nll(ref_model, lose_inputs)
        lose_log_ratio = -(lose_loss - lose_ref_loss)

    loss = -2 / beta * F.logsigmoid(beta * (win_log_ratio - lose_log_ratio)).mean()
    return loss, (win_outputs, lose_outputs)


def compute_undial_loss(model, ref_model, inputs, beta):
    # Forward pass on the student (trainable) model
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]

    shift_labels = labels[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()

    # Forward pass on the teacher model (no grad)
    with torch.no_grad():
        teacher_logits = ref_model(**inputs).logits
    shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()

    # Build the mask that identifies the tokens need to be unlearned
    mask = torch.zeros_like(shift_teacher_logits)
    batch_idx = torch.arange(mask.shape[0]).view(-1, 1, 1)
    seq_idx = torch.arange(mask.shape[1]).view(1, -1, 1)
    mask[batch_idx, seq_idx, shift_labels.unsqueeze(-1)] = 1.0

    # Adjust teacher logits: subtract di_strength on the correct token
    pre_softmax = shift_teacher_logits - mask * beta
    soft_label = F.softmax(pre_softmax, dim=-1)

    loss_fct = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        soft_label.view(-1, soft_label.size(-1)),
    )
    return loss.mean(), outputs


def compute_fundial_loss(model, ref_model, inputs, beta, mask):
    """UNDIAL self-distillation loss gated by a per-token binary ``mask``.

    ``mask`` has shape ``(B, T-1)`` and is aligned to ``shift_labels``. The
    final loss averages over selected tokens only (``sum(mask).clamp(min=1)``),
    so empty-mask samples contribute zero without exploding the gradient.
    """
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]

    shift_labels = labels[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()

    with torch.no_grad():
        teacher_logits = ref_model(**inputs).logits
    shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()

    one_hot = torch.zeros_like(shift_teacher_logits)
    batch_idx = torch.arange(one_hot.shape[0]).view(-1, 1, 1)
    seq_idx = torch.arange(one_hot.shape[1]).view(1, -1, 1)
    one_hot[batch_idx, seq_idx, shift_labels.unsqueeze(-1)] = 1.0

    pre_softmax = shift_teacher_logits - one_hot * beta
    soft_label = F.softmax(pre_softmax, dim=-1)

    loss_fct = nn.CrossEntropyLoss(reduction="none")
    token_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        soft_label.view(-1, soft_label.size(-1)),
    )

    mask_flat = mask.reshape(-1).to(token_loss.device).to(token_loss.dtype)
    n_selected = mask_flat.sum().clamp(min=1.0)
    loss = (token_loss * mask_flat).sum() / n_selected
    return loss, outputs


def compute_scored_fundial_loss(
    cache: Cache,
    ref_model: PreTrainedModel,
    scorer: TokenImportanceScorer,
    forget_inputs: dict[str, Any],
    beta: float,
) -> torch.Tensor:
    """FUNDIAL forget loss with the learned scorer replacing the spaCy hard mask.

    Reuses UNDIAL's beta-suppressed teacher target; per-token CE is weighted by
    the scorer's continuous scores g_t in [0, 1] instead of FUNDIAL's binary
    noun/entity mask. Averaged over scorer-valid positions:
    ``sum(g * CE)[mask] / sum(g[mask]).clamp(min=1)``.
    """
    with torch.no_grad():
        scores, mask = scorer.score(cache)

    student_logits = cache.outputs.logits
    labels = forget_inputs["labels"]
    shift_labels = labels[..., 1:].contiguous()
    shift_student_logits = student_logits[..., :-1, :].contiguous()

    with torch.no_grad():
        teacher_logits = ref_model(**forget_inputs).logits
    shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()

    one_hot = torch.zeros_like(shift_teacher_logits)
    batch_idx = torch.arange(one_hot.shape[0]).view(-1, 1, 1)
    seq_idx = torch.arange(one_hot.shape[1]).view(1, -1, 1)
    one_hot[batch_idx, seq_idx, shift_labels.unsqueeze(-1)] = 1.0
    pre_softmax = shift_teacher_logits - one_hot * beta
    soft_label = F.softmax(pre_softmax, dim=-1)

    loss_fct = nn.CrossEntropyLoss(reduction="none")
    token_loss = loss_fct(
        shift_student_logits.view(-1, shift_student_logits.size(-1)),
        soft_label.view(-1, soft_label.size(-1)),
    ).view(scores.shape)

    weighted = token_loss * scores
    n_selected = scores[mask].sum().clamp(min=1.0)
    return weighted[mask].sum() / n_selected


def compute_wga_loss(model, inputs, beta):
    outputs = model(**inputs)
    labels = inputs["labels"]
    labels = labels.to(outputs.logits.device)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    lm_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    weight_ce = ((-lm_loss).exp().detach()) ** beta
    forget_loss = -(weight_ce * lm_loss)[shift_labels.view(-1) != -100].mean()
    return forget_loss, outputs


def compute_satimp_loss(model, inputs, beta1, beta2):
    outputs = model(**inputs)
    labels = inputs["labels"]
    labels = labels.to(outputs.logits.device)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    lm_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    weight_sat = ((-lm_loss).exp().detach()) ** beta1
    weight_imp = (1 - (-lm_loss).exp().detach()) ** beta2
    forget_loss = -((weight_sat * weight_imp) * lm_loss)[
        shift_labels.view(-1) != -100
    ].mean()
    return forget_loss, outputs


def compute_satimp_loss_custom(model, inputs, beta1, beta2):
    outputs = model(**inputs)
    labels = inputs["labels"]
    labels = labels.to(outputs.logits.device)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    lm_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    weight_imp = (1 - (-lm_loss).exp().detach()) ** beta2
    lm_loss = weight_imp * lm_loss
    weight_sat = ((-lm_loss).exp().detach()) ** beta1
    forget_loss = -(weight_sat * lm_loss)[shift_labels.view(-1) != -100].mean()
    return forget_loss, outputs


################
# JensUn Utils #
################


def compute_js_avg(model, inputs):
    # get the sum loss for each sequence in a batch
    # NOTE: not same as model(**inputs).loss but has sum loss for each seq in a batch

    # Compute probabilities from logits
    outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)

    # Define the uniform distribution over the vocabulary
    uniform_dist = torch.full_like(probs, 1.0 / probs.size(-1))

    # Compute the midpoint distribution
    m = 0.5 * (probs + uniform_dist)

    # Compute KL divergence (adding a small value to avoid log(0))
    kl_p_m = F.kl_div(m.log(), probs, reduction="batchmean")
    kl_q_m = F.kl_div(m.log(), uniform_dist, reduction="batchmean")

    # Compute final JS Divergence
    js_div = 0.5 * (kl_p_m + kl_q_m)

    return js_div, outputs


def jensun_retain_loss(
    model: PreTrainedModel, target_model: PreTrainedModel, inputs: dict[str, Any]
):
    with torch.no_grad():
        ref_outputs: CausalLMOutputWithPast = target_model(**inputs)

    ref_logits = ref_outputs.logits
    epsilon = 1e-9
    ref_probs = F.softmax(ref_logits, dim=-1)

    outputs: CausalLMOutputWithPast = model(**inputs)
    logits = outputs.logits
    current_probs = F.softmax(logits, dim=-1)

    # Compute the midpoint distribution
    m = 0.5 * (ref_probs + current_probs)
    m = torch.clamp(m, min=epsilon)  # Added this line

    # Clamp input distributions as well to avoid log(0)
    ref_probs = torch.clamp(ref_probs, min=epsilon)
    current_probs = torch.clamp(current_probs, min=epsilon)

    # Compute KL divergences manually (P‖M and Q‖M)
    kl_p_m = torch.sum(ref_probs * (torch.log(ref_probs) - torch.log(m)), dim=-1).mean()
    kl_q_m = torch.sum(
        current_probs * (torch.log(current_probs) - torch.log(m)), dim=-1
    ).mean()

    # Compute final JS Divergence
    js_div = 0.5 * (kl_p_m + kl_q_m)

    return js_div, outputs


def compute_uniform_ce_avg(
    model: PreTrainedModel, inputs: dict[str, Any]
) -> tuple[torch.Tensor, CausalLMOutputWithPast]:
    # Forward pass to get logits
    outputs: CausalLMOutputWithPast = model(**inputs)
    logits = outputs.logits

    # Compute log probabilities from logits
    log_probs = F.log_softmax(logits, dim=-1)

    # Define the uniform target distribution over the vocabulary
    vocab_size = logits.size(-1)
    uniform_dist = torch.full_like(log_probs, 1.0 / vocab_size)

    # Compute cross-entropy loss: CE(target, log_probs)
    ce_loss = -(uniform_dist * log_probs).sum(dim=-1)  # sum over vocab
    ce_loss = ce_loss.mean()  # average over batch and sequence if needed

    return ce_loss, outputs


def jensun_multitok_loss(
    model: PreTrainedModel,
    forget_inputs: dict[str, Any],
    target_tokens: list[int] = [2822, 4623],
) -> tuple[torch.Tensor, CausalLMOutputWithPast]:
    outputs: CausalLMOutputWithPast = model(**forget_inputs)
    logits = outputs.logits
    epsilon = 1e-9

    probs = F.softmax(logits, dim=-1)
    peaked_dist = torch.zeros_like(probs)

    seq_len = logits.shape[1]
    expected_tokens = torch.tensor(target_tokens).repeat(
        seq_len // len(target_tokens) + 1
    )[:seq_len]

    # TODO: This should be `peaked_dist[..., expected_tokens] = 1.0 / len(target_tokens)`
    # Right now, the vector is has a probability mass equal with len(target_tokens).
    peaked_dist[..., expected_tokens] = 1.0  # Set probability 1.0 at resp tokens

    # Midpoint distribution
    m = 0.5 * (probs + peaked_dist)
    m = torch.clamp(m, min=epsilon)  # Ensure no zeros

    # Clamp input distributions as well to avoid log(0)
    probs = torch.clamp(probs, min=epsilon)
    peaked_dist = torch.clamp(peaked_dist, min=epsilon)

    # Compute KL divergences manually (P || M and Q || M)
    kl_p_m = torch.sum(probs * (torch.log(probs) - torch.log(m)), dim=-1)
    kl_q_m = torch.sum(peaked_dist * (torch.log(peaked_dist) - torch.log(m)), dim=-1)

    js_div = 0.5 * (kl_p_m.mean() + kl_q_m.mean())

    return js_div, outputs


##################
# Self Balancing #
##################


def reweighted_NLL(
    cache: Cache,
    scorer: TokenImportanceScorer,
    scorer_requires_grad: bool = False,
    invert_probabilities: bool = False,
    normalize_token_loss: bool = False,
    beta: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    scorer_ctx = torch.enable_grad() if scorer_requires_grad else torch.no_grad()
    with scorer_ctx:
        scores, mask = scorer.score(cache)

        if invert_probabilities:
            scores = torch.where(mask, 1 - scores, 0)

    token_loss = cache.token_loss.detach() if scorer_requires_grad else cache.token_loss

    if normalize_token_loss:
        token_loss = F.normalize(token_loss, p=1, dim=-1)

    token_loss = token_loss * scores
    saturation = (
        torch.ones_like(scores)
        if beta is None
        else (-token_loss).exp().detach() ** beta
    )
    token_loss = token_loss * saturation
    weighted_loss = token_loss[mask].mean()

    # saturation = torch.ones_like(scores) if beta is None else (-token_loss).exp().detach() ** beta
    # weighted_loss = (
    #     token_loss *
    #     saturation *
    #     scores
    # )[mask].mean()

    # Revert inverted scores before returning
    if invert_probabilities:
        scores = torch.where(mask, 1 - scores, 0)

    return weighted_loss, scores, mask


def reweighted_NLL_correct(
    cache: Cache,
    scorer: TokenImportanceScorer,
    scorer_requires_grad: bool = False,
    invert_probabilities: bool = False,
    normalize_token_loss: bool = False,
    beta: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    scorer_ctx = torch.enable_grad() if scorer_requires_grad else torch.no_grad()
    with scorer_ctx:
        scores, mask = scorer.score(cache)

        if invert_probabilities:
            scores = torch.where(mask, 1 - scores, 0)

    token_loss = cache.token_loss.detach() if scorer_requires_grad else cache.token_loss

    if normalize_token_loss:
        token_loss = F.normalize(token_loss, p=1, dim=-1)

    saturation = (
        torch.ones_like(scores)
        if beta is None
        else (-token_loss).exp().detach() ** beta
    )
    weighted_loss = (token_loss * saturation * scores)[mask].mean()

    # Revert inverted scores before returning
    if invert_probabilities:
        scores = torch.where(mask, 1 - scores, 0)

    return weighted_loss, scores, mask


def reweighted_softmax_NLL(
    cache: Cache,
    scorer: TokenImportanceScorer,
    scorer_requires_grad: bool = False,
    invert_probabilities: bool = False,
    normalize_token_loss: bool = False,
    beta: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    scorer_ctx = torch.enable_grad() if scorer_requires_grad else torch.no_grad()
    with scorer_ctx:
        scores, mask = scorer.score(cache)

        scores_to_use = (
            torch.where(mask, 1 - scores, 0) if invert_probabilities else scores.clone()
        )

        scores_to_use = F.softmax(
            scores_to_use.masked_fill(~mask, -float("inf")), dim=-1
        ).masked_fill(~mask, 0)

    token_loss = cache.token_loss.detach() if scorer_requires_grad else cache.token_loss

    if normalize_token_loss:
        token_loss = F.normalize(token_loss, p=1, dim=-1)

    token_loss = token_loss * scores_to_use
    saturation = (
        torch.ones_like(scores)
        if beta is None
        else (-token_loss).exp().detach() ** beta
    )
    token_loss = token_loss * saturation
    weighted_loss = token_loss[mask].mean()

    # saturation = torch.ones_like(scores) if beta is None else (-token_loss).exp().detach() ** beta
    # weighted_loss = (
    #     token_loss *
    #     saturation *
    #     scores
    # )[mask].mean()

    return weighted_loss, scores, scores_to_use, mask


############################
# Scorer-Adjusted Methods  #
############################


def compute_scored_batch_nll(
    cache: Cache,
    scorer: TokenImportanceScorer,
    score_scale: float = 2.0,
    rescale: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sequence NLL with scorer reweighting: sum(w_t * token_loss_t).

    With zero-init sigmoid scorer, initial g_t = 0.5, so score_scale=2 gives initial weights=1.

    When rescale=True, weights are normalized per sequence so they sum to T (the
    sequence length). This keeps the scored NLL on the same scale as unscored NLL,
    which is important when computing log-ratios (e.g. DPO, NPO).

    Returns:
        scored_nll: (B,) per-sequence scored NLL
        scores: (B, T-1)
        mask: (B, T-1)
    """
    with torch.no_grad():
        scores, mask = scorer.score(cache)
    weights = score_scale * scores
    if rescale:
        seq_lengths = mask.sum(dim=-1, keepdim=True).float().clamp(min=1)
        weight_sums = (weights * mask.float()).sum(dim=-1, keepdim=True).clamp(min=1e-8)
        weights = weights * (seq_lengths / weight_sums)
    weighted = cache.token_loss * weights
    weighted = weighted * mask.float()
    return weighted.sum(dim=-1), scores, mask


def compute_scored_wga_loss(
    cache: Cache,
    scorer: TokenImportanceScorer,
    beta: float,
    score_scale: float = 2.0,
) -> torch.Tensor:
    """WGA loss with score-modulated exponent: exp(-loss)^(beta * score_scale * g_t)."""
    with torch.no_grad():
        scores, mask = scorer.score(cache)
    token_loss = cache.token_loss
    weight_ce = ((-token_loss).exp().detach()) ** (beta * score_scale * scores.detach())
    return -(weight_ce * token_loss)[mask].mean()


def compute_inverted_wga_loss(
    cache: Cache,
    scorer: TokenImportanceScorer,
    beta: float,
) -> torch.Tensor:
    """WGA loss with inverted-score exponent: exp(-loss)^(beta * (1 - g_t))."""
    with torch.no_grad():
        scores, mask = scorer.score(cache)
    token_loss = cache.token_loss
    weight_ce = ((-token_loss).exp().detach()) ** (beta * (1 - scores.detach()))
    return -(weight_ce * token_loss)[mask].mean()


def compute_inverted_scored_wga_loss(
    cache: Cache,
    scorer: TokenImportanceScorer,
    beta: float,
) -> torch.Tensor:
    """g_t * exp(-loss)^(beta * (1 - g_t)) weighted NLL."""
    with torch.no_grad():
        scores, mask = scorer.score(cache)
    token_loss = cache.token_loss
    weight_ce = ((-token_loss).exp().detach()) ** (beta * (1 - scores.detach()))
    return -(scores * weight_ce * token_loss)[mask].mean()


def scored_jensun_multitok_loss(
    cache: Cache,
    scorer: TokenImportanceScorer,
    target_tokens: list[int],
    score_scale: float = 2.0,
    rescale: bool = False,
) -> torch.Tensor:
    """JensUn JS-div reweighted by scorer scores.

    Computes per-position JS-div from cache logits, trims to (B, T-1) by dropping
    the last position (aligns with scorer scores at logit positions 0..T-2),
    multiplies by weights, averages over valid positions.

    When rescale=True, weights are normalized per sequence so they sum to T,
    keeping the weighted JS-div on the same scale as the unweighted version.
    """
    with torch.no_grad():
        scores, mask = scorer.score(cache)

    logits = cache.outputs.logits  # (B, T, V)
    epsilon = 1e-9

    probs = F.softmax(logits, dim=-1)
    peaked_dist = torch.zeros_like(probs)

    seq_len = logits.shape[1]
    expected_tokens = torch.tensor(target_tokens, device=logits.device).repeat(
        seq_len // len(target_tokens) + 1
    )[:seq_len]
    peaked_dist[..., expected_tokens] = 1.0

    m = 0.5 * (probs + peaked_dist)
    m = torch.clamp(m, min=epsilon)
    probs = torch.clamp(probs, min=epsilon)
    peaked_dist = torch.clamp(peaked_dist, min=epsilon)

    kl_p_m = torch.sum(probs * (torch.log(probs) - torch.log(m)), dim=-1)  # (B, T)
    kl_q_m = torch.sum(
        peaked_dist * (torch.log(peaked_dist) - torch.log(m)), dim=-1
    )  # (B, T)
    per_pos_js = 0.5 * (kl_p_m + kl_q_m)  # (B, T)

    # Drop last position to align with scores (B, T-1)
    per_pos_js = per_pos_js[:, :-1]

    weights = score_scale * scores
    if rescale:
        seq_lengths = mask.sum(dim=-1, keepdim=True).float().clamp(min=1)
        weight_sums = (weights * mask.float()).sum(dim=-1, keepdim=True).clamp(min=1e-8)
        weights = weights * (seq_lengths / weight_sums)

    weighted_js = per_pos_js * weights
    return weighted_js[mask].mean()
