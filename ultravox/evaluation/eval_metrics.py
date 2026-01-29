from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Dict, List, Tuple

from ultravox.data import types
from ultravox.evaluation import eval_types
from ultravox.evaluation import gpt_eval_audiobench
from ultravox.evaluation import gpt_eval_bigbench
from ultravox.evaluation import gpt_eval_boolq
from ultravox.evaluation import gpt_eval_conv
from ultravox.evaluation import gpt_eval_instruct
from ultravox.evaluation import gpt_eval_voicebench
from ultravox.evaluation import ifeval_voicebench
from ultravox.evaluation import string_metrics

METRIC_REGISTRY: Dict[str, Callable[[eval_types.Sample], eval_types.Result]] = {
    "boolq": gpt_eval_boolq.evaluate_answer_boolq,
    "instruct": gpt_eval_instruct.evaluate_answer_instruct,
    "conversation": gpt_eval_conv.evaluate_conversation_response,
    "bigbench": gpt_eval_bigbench.evaluate_answer_bigbench,
    "audiobench_binary": gpt_eval_audiobench.evaluate_answer_audiobench_binary,
    "audiobench_scalar": gpt_eval_audiobench.evaluate_answer_audiobench,
    "exact_match_last_word": string_metrics.match_last_word,
    "partial_match": string_metrics.partial_match,
    "voicebench_yes_no": gpt_eval_voicebench.evaluate_yes_no_voicebench,
    "voicebench_scalar": gpt_eval_voicebench.evaluate_answer_voicebench,
    "voicebench_mcq": gpt_eval_voicebench.evaluate_mcq_voicebench,
    "voicebench_bbh": gpt_eval_voicebench.evaluate_bbh_voicebench,
    "voicebench_harm": gpt_eval_voicebench.evaluate_harm_voicebench,
    "voicebench_ifeval": ifeval_voicebench.IFEvaluator.instruction_following_evaluate,
}

CORPUS_METRIC_REGISTRY: Dict[
    str, Callable[[List[eval_types.Sample], Dict[str, Any]], eval_types.Result]
] = {
    "bleu": string_metrics.bleu,
    "wer": string_metrics.wer,
    # QA metrics (with article/punctuation removal)
    "f1_qa": string_metrics.f1_qa,
    "exact_match_qa": string_metrics.exact_match_qa,
    # Classification metrics (case-insensitive only)
    "exact_match": string_metrics.exact_match,
    # SLURP SLU metrics
    "slot_f1": string_metrics.slot_f1,
    "slu_accuracy": string_metrics.slu_accuracy,
    # Backward compatibility aliases
    "squad_f1": string_metrics.squad_f1,
    "squad_exact_match": string_metrics.squad_exact_match,
    "exact_match_normalized": string_metrics.exact_match_normalized,
}


def evaluate_answer(sample: eval_types.Sample, metric: str) -> eval_types.Result:
    if metric in METRIC_REGISTRY:
        return METRIC_REGISTRY[metric](sample)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def _compute_per_sample_scores(
    samples: List[eval_types.Sample], metric: str, args: Dict[str, Any]
) -> None:
    """Compute and store per-sample scores for corpus-level metrics.

    This populates the `score` field of each Sample object based on the metric type.
    For WER/BLEU, lower/higher is better respectively. All scores are stored as-is
    without normalization to preserve interpretability.
    """
    if metric == "wer":
        for sample in samples:
            sample.score = string_metrics.wer_single(
                sample.expected_answer, sample.generated_answer, args
            )
    elif metric == "bleu":
        for sample in samples:
            sample.score = string_metrics.bleu_single(
                sample.generated_answer, sample.expected_answer, args
            )
    elif metric in ("f1_qa", "squad_f1"):
        for sample in samples:
            sample.score = string_metrics.f1_qa_single(
                sample.generated_answer, sample.expected_answer
            )
    elif metric in ("exact_match_qa", "squad_exact_match"):
        for sample in samples:
            sample.score = string_metrics.exact_match_qa_single(
                sample.generated_answer, sample.expected_answer
            )
    elif metric in ("exact_match", "exact_match_normalized"):
        for sample in samples:
            sample.score = string_metrics.exact_match_single(
                sample.generated_answer, sample.expected_answer
            )
    elif metric == "slot_f1":
        for sample in samples:
            sample.score = string_metrics.slot_f1_single(
                sample.generated_answer, sample.expected_answer
            )
    elif metric == "slu_accuracy":
        for sample in samples:
            sample.score = string_metrics.slu_accuracy_single(
                sample.generated_answer, sample.expected_answer
            )


def evaluate_answers(
    samples: List[eval_types.Sample], metric_config: types.EvalConfig
) -> eval_types.Result:
    """Evaluate all samples and populate per-sample scores.

    This function computes evaluation metrics for all samples and stores
    the per-sample score in each Sample's `score` field. For per-sample
    metrics (METRIC_REGISTRY), it also stores the reason if available.
    """
    if metric_config.metric in CORPUS_METRIC_REGISTRY:
        # Compute per-sample scores first
        _compute_per_sample_scores(samples, metric_config.metric, metric_config.args)

        # Then compute the corpus-level metric
        metric_func = CORPUS_METRIC_REGISTRY[metric_config.metric]
        return metric_func(samples, metric_config.args)

    elif metric_config.metric in METRIC_REGISTRY:
        metric_fn = METRIC_REGISTRY[metric_config.metric]
        partial_metric_fn = partial(metric_fn, **metric_config.args)
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(partial_metric_fn, samples))

        # Store per-sample scores and reasons
        for sample, result in zip(samples, results):
            sample.score = result.score
            if hasattr(result, "reason"):
                sample.score_reason = result.reason

        total_score = sum(result.score for result in results)
        return eval_types.MeanResult(score=total_score / len(samples))
    else:
        raise ValueError(f"Unknown metric: {metric_config.metric}")


def aggregate_scores_by_field(
    samples: List[eval_types.Sample], field_name: str
) -> Dict[str, Tuple[float, int]]:
    """Group samples by field in extra_kwargs and compute per-group scores.

    Args:
        samples: List of evaluated samples with scores populated
        field_name: The field in extra_kwargs to group by (e.g., "task_type")

    Returns:
        Dict mapping group_key -> (mean_score, sample_count)
        Example: {"speech": (0.56, 520), "sound": (0.42, 240), "music": (0.45, 240)}
    """
    groups: Dict[str, List[float]] = {}

    for sample in samples:
        if sample.extra_kwargs and field_name in sample.extra_kwargs:
            group_key = sample.extra_kwargs[field_name]
            if group_key not in groups:
                groups[group_key] = []
            if sample.score is not None:
                groups[group_key].append(sample.score)

    results = {}
    for group_key, scores in groups.items():
        if scores:
            mean_score = sum(scores) / len(scores)
            results[group_key] = (mean_score, len(scores))

    return results
