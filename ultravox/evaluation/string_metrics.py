import argparse
import json
import re
import string
from collections import Counter
from typing import Any, Dict, List

import evaluate
import sacrebleu
import whisper_normalizer.basic as whisper_basic
import whisper_normalizer.english as whisper_english

from ultravox.evaluation import eval_types

# Arabic diacritic marks
arabic_diacritics = re.compile(r"[\u064B-\u065F\u0670]")

# Cache for expensive evaluate.load() calls
_WER_METRIC = None


def _get_wer_metric():
    """Get cached WER metric to avoid repeated loading."""
    global _WER_METRIC
    if _WER_METRIC is None:
        _WER_METRIC = evaluate.load("wer")
    return _WER_METRIC


def remove_diacritics(text):
    return arabic_diacritics.sub("", text)


def _get_wer_normalizer(lang_id: str):
    """Get the appropriate text normalizer for a language."""
    if lang_id == "en":
        return whisper_english.EnglishTextNormalizer()
    else:
        return whisper_basic.BasicTextNormalizer()


def _normalize_for_wer(
    reference: str, hypothesis: str, lang_id: str, cap_hypothesis_len: float = None
) -> tuple:
    """Normalize reference and hypothesis for WER computation."""
    normalizer = _get_wer_normalizer(lang_id)

    if lang_id == "ar":
        reference = remove_diacritics(reference)
        hypothesis = remove_diacritics(hypothesis)

    reference = normalizer(reference)
    hypothesis = normalizer(hypothesis)

    # Languages where we compute CER (space-separated characters)
    if lang_id in ["zh", "ja", "th", "lo", "my"]:
        reference = " ".join(list(reference))
        hypothesis = " ".join(list(hypothesis))

    # Cap the length of the hypothesis
    if cap_hypothesis_len is not None:
        hypothesis = hypothesis[: int(len(reference) * cap_hypothesis_len)]

    # Handle empty strings
    reference = reference if reference.strip() else "<silence>"
    hypothesis = hypothesis if hypothesis.strip() else "<silence>"

    return reference, hypothesis


def wer_single(
    reference: str, hypothesis: str, args: Dict[str, Any]
) -> float:
    """Compute WER for a single sample."""
    lang_id = args.get("lang_id", "<undefined>").lower()
    cap_hypothesis_len = args.get("cap_hypothesis_len", None)

    ref_norm, hyp_norm = _normalize_for_wer(
        reference, hypothesis, lang_id, cap_hypothesis_len
    )

    wer_metric = _get_wer_metric()
    wer_score = wer_metric.compute(predictions=[hyp_norm], references=[ref_norm])
    return wer_score * 100


def wer(samples: List[eval_types.Sample], args: Dict[str, Any]) -> eval_types.WerResult:
    """Compute WER or CER using Whisper's text normalization."""
    lang_id = args.get("lang_id", "<undefined>").lower()  # Ensure case-insensitive

    # Initialize the appropriate text normalizer
    if lang_id == "en":
        normalizer = whisper_english.EnglishTextNormalizer()
    else:
        normalizer = whisper_basic.BasicTextNormalizer()

    references = [sample.expected_answer for sample in samples]
    hypotheses = [sample.generated_answer for sample in samples]

    if lang_id == "ar":
        references = [remove_diacritics(ref) for ref in references]
        hypotheses = [remove_diacritics(hyp) for hyp in hypotheses]

    # Normalize both reference and hypothesis
    references = [normalizer(ref) for ref in references]
    hypotheses = [normalizer(hyp) for hyp in hypotheses]

    # Languages where we compute CER (space-separated characters)
    if lang_id in ["zh", "ja", "th", "lo", "my"]:
        # Convert to space-separated characters for CER
        references = [" ".join(list(ref)) for ref in references]
        hypotheses = [" ".join(list(hyp)) for hyp in hypotheses]

    # Cap the length of the hypothesis to some multiple of the reference length
    cap_hypothesis_len = args.get("cap_hypothesis_len", None)
    if cap_hypothesis_len is not None:
        hypotheses = [
            hyp[: int(len(ref) * cap_hypothesis_len)]
            for hyp, ref in zip(hypotheses, references)
        ]

    # Handle empty strings
    references = [e if e.strip() else "<silence>" for e in references]
    hypotheses = [s if s.strip() else "<silence>" for s in hypotheses]

    # Compute WER using space-separated words
    wer_metric = _get_wer_metric()
    wer_score = wer_metric.compute(predictions=hypotheses, references=references)
    return eval_types.WerResult(score=wer_score * 100)


def match_last_word(sample: eval_types.Sample) -> eval_types.ExactMatchResult:
    # Handle the case where generated_answer might be a boolean
    generated_answer_str = (
        str(sample.generated_answer).lower()
        if isinstance(sample.generated_answer, bool)
        else sample.generated_answer.lower()
    )
    last_words = re.findall(r"\b\w+\b(?=\W*$)", generated_answer_str)

    # Handle the case where expected_answer might be a boolean
    expected_answer_str = (
        str(sample.expected_answer).lower()
        if isinstance(sample.expected_answer, bool)
        else sample.expected_answer.lower()
    )
    expected_tf = re.findall(r"\b\w+\b(?=\W*$)", expected_answer_str)[-1]

    if not last_words:
        return eval_types.ExactMatchResult(score=0, reason="No last word found")

    last_word: str = last_words[-1]
    if last_word in ["yes", "true"]:
        last_word = "true"
    elif last_word in ["no", "false"]:
        last_word = "false"
    else:
        return eval_types.ExactMatchResult(score=0, reason="Last word not true/false")

    return eval_types.ExactMatchResult(
        score=last_word == expected_tf, reason="exact_match check"
    )


def partial_match(sample: eval_types.Sample) -> eval_types.ExactMatchResult:
    """Compute partial match score where expected answer should be part of generated answer.

    Returns a score of 1 if the expected answer is contained within the generated answer
    (case-insensitive), and 0 otherwise.
    """
    generated = sample.generated_answer.lower().strip()
    expected = sample.expected_answer.lower().strip()

    return eval_types.ExactMatchResult(
        score=int(expected in generated), reason="partial_match check"
    )


def bleu(
    samples: List[eval_types.Sample], args: Dict[str, Any]
) -> eval_types.BleuResult:
    """
    Compute corpus BLEU score for a list of samples.
    """
    references = [[sample.expected_answer for sample in samples]]
    hypotheses = [sample.generated_answer for sample in samples]
    score = sacrebleu.corpus_bleu(
        hypotheses=hypotheses, references=references, **args
    ).score
    return eval_types.BleuResult(score=score)


def bleu_single(hypothesis: str, reference: str, args: Dict[str, Any]) -> float:
    """Compute sentence-level BLEU score for a single sample."""
    score = sacrebleu.sentence_bleu(hypothesis, [reference], **args).score
    return score


def _normalize_qa_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.

    This normalization is used for extractive QA evaluation (SQuAD-style).
    """

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match_qa_single(prediction: str, ground_truth: str) -> float:
    """Compute exact match for extractive QA tasks.

    Normalizes by lowercasing, removing punctuation, articles, and extra whitespace.
    Returns 100.0 if exact match, 0.0 otherwise.
    """
    return (
        100.0
        if _normalize_qa_answer(prediction) == _normalize_qa_answer(ground_truth)
        else 0.0
    )


def f1_qa_single(prediction: str, ground_truth: str) -> float:
    """Compute F1 score for extractive QA tasks.

    Measures token-level overlap after QA normalization.
    Returns score in range [0, 100].
    """
    pred_tokens = _normalize_qa_answer(prediction).split()
    gt_tokens = _normalize_qa_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 100.0 if pred_tokens == gt_tokens else 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall) * 100


def exact_match_qa(
    samples: List[eval_types.Sample], args: Dict[str, Any]
) -> eval_types.MeanResult:
    """
    Compute exact match score for extractive question answering.

    Exact match measures whether the prediction exactly matches the ground truth
    after normalization (lowercasing, removing punctuation/articles/whitespace).
    Use this for QA tasks like HeySQuAD, SQuAD, etc.
    """
    scores = [
        exact_match_qa_single(sample.generated_answer, sample.expected_answer)
        for sample in samples
    ]
    return eval_types.MeanResult(score=sum(scores) / len(scores))


def exact_match_single(prediction: str, ground_truth: str) -> float:
    """Compute case-insensitive exact match for a single sample.

    Returns 100.0 if exact match after normalization, 0.0 otherwise.
    Used for classification tasks like emotion recognition, sentiment analysis, etc.
    Strips trailing punctuation to handle responses like "No." matching "no".
    """
    pred = prediction.strip().lower().rstrip(".,!?;:")
    gt = ground_truth.strip().lower().rstrip(".,!?;:")
    return 100.0 if pred == gt else 0.0


def exact_match(
    samples: List[eval_types.Sample], args: Dict[str, Any]
) -> eval_types.MeanResult:
    """
    Compute case-insensitive exact match score for classification tasks.

    This metric is used for tasks where the model output should exactly match
    one of several predefined labels (e.g., emotion recognition, sentiment analysis,
    accent classification). Performs case normalization, whitespace trimming, and
    strips trailing punctuation (e.g., "No." matches "no").

    Returns score as percentage (0-100).
    """
    scores = [
        exact_match_single(sample.generated_answer, sample.expected_answer)
        for sample in samples
    ]
    return eval_types.MeanResult(score=sum(scores) / len(scores) if scores else 0.0)


def f1_qa(
    samples: List[eval_types.Sample], args: Dict[str, Any]
) -> eval_types.MeanResult:
    """
    Compute F1 score for extractive question answering.

    F1 score measures token-level overlap between prediction and ground truth,
    accounting for both precision and recall. Use this for QA tasks like
    HeySQuAD, SQuAD, etc.
    """
    scores = [
        f1_qa_single(sample.generated_answer, sample.expected_answer)
        for sample in samples
    ]
    return eval_types.MeanResult(score=sum(scores) / len(scores))


# Backward compatibility aliases
squad_exact_match_single = exact_match_qa_single
squad_f1_single = f1_qa_single
squad_exact_match = exact_match_qa
squad_f1 = f1_qa
exact_match_normalized_single = exact_match_single
exact_match_normalized = exact_match


# ============================================================================
# SLURP Slot Filling Metrics
# ============================================================================


def _extract_slots(annotation: str) -> set:
    """Extract slots from SLURP annotation format.

    SLURP uses format: "text [slot_type : value] more text [slot_type2 : value2]"
    Example: "event reminder [event_name : mona] [date : tuesday]"

    Returns a set of (slot_type, value) tuples.
    """
    slot_pattern = re.compile(r"\[([^\]:]+)\s*:\s*([^\]]+)\]")
    slots = set()
    for match in slot_pattern.finditer(annotation):
        slot_type = match.group(1).strip().lower()
        slot_value = match.group(2).strip().lower()
        slots.add((slot_type, slot_value))
    return slots


def slot_f1_single(prediction: str, ground_truth: str) -> float:
    """Compute slot F1 score for a single SLURP sample.

    Extracts slots from both prediction and ground truth annotations,
    then computes F1 based on exact slot matches.

    Returns score in range [0, 100].
    """
    pred_slots = _extract_slots(prediction)
    gt_slots = _extract_slots(ground_truth)

    # Handle edge cases
    if len(gt_slots) == 0 and len(pred_slots) == 0:
        return 100.0  # Both empty = perfect match
    if len(gt_slots) == 0 or len(pred_slots) == 0:
        return 0.0  # One empty, one not = no match

    # Compute precision, recall, F1
    common = pred_slots & gt_slots
    precision = len(common) / len(pred_slots)
    recall = len(common) / len(gt_slots)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1 * 100


def slot_f1(
    samples: List[eval_types.Sample], args: Dict[str, Any]
) -> eval_types.MeanResult:
    """Compute slot F1 score for SLURP slot filling task.

    Measures how well the model extracts entities in [slot_type : value] format.
    Returns score as percentage (0-100).
    """
    scores = [
        slot_f1_single(sample.generated_answer, sample.expected_answer)
        for sample in samples
    ]
    return eval_types.MeanResult(score=sum(scores) / len(scores) if scores else 0.0)


def _extract_intent_from_slu(text: str) -> str:
    """Extract intent from combined SLU format.

    Expected format: "Intent: <intent_label>\nAnnotation: <annotation>"
    """
    match = re.search(r"Intent:\s*(\S+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    return ""


def _extract_annotation_from_slu(text: str) -> str:
    """Extract annotation from combined SLU format.

    Expected format: "Intent: <intent_label>\nAnnotation: <annotation>"
    """
    match = re.search(r"Annotation:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text  # Fallback to full text


def slu_accuracy_single(prediction: str, ground_truth: str) -> float:
    """Compute SLU accuracy for a single sample.

    Both intent and slots must be correct for a score of 100.
    Partial credit: 50 for correct intent, 50 * slot_f1 for slots.

    Returns score in range [0, 100].
    """
    pred_intent = _extract_intent_from_slu(prediction)
    gt_intent = _extract_intent_from_slu(ground_truth)

    pred_annotation = _extract_annotation_from_slu(prediction)
    gt_annotation = _extract_annotation_from_slu(ground_truth)

    # Intent accuracy (50 points)
    intent_score = 50.0 if pred_intent == gt_intent else 0.0

    # Slot F1 (50 points)
    slot_score = slot_f1_single(pred_annotation, gt_annotation) * 0.5

    return intent_score + slot_score


def slu_accuracy(
    samples: List[eval_types.Sample], args: Dict[str, Any]
) -> eval_types.MeanResult:
    """Compute combined SLU accuracy for SLURP.

    Evaluates both intent classification and slot filling together.
    Score is weighted: 50% intent accuracy + 50% slot F1.

    Returns score as percentage (0-100).
    """
    scores = [
        slu_accuracy_single(sample.generated_answer, sample.expected_answer)
        for sample in samples
    ]
    return eval_types.MeanResult(score=sum(scores) / len(scores) if scores else 0.0)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate JSON files using WER and BLEU."
    )
    parser.add_argument("input_file", type=str, help="Path to the input JSON file.")
    parser.add_argument(
        "--metric",
        type=str,
        choices=["wer", "bleu"],
        required=True,
        help="Metric to compute.",
    )
    parser.add_argument(
        "--lang_id", type=str, default="en", help="Language ID (e.g., en, zh, ja)."
    )
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = [eval_types.Sample(**sample) for sample in data]

    if args.metric == "wer":
        result = wer(samples, {"lang_id": args.lang_id})
    else:
        result = bleu(samples, {"tokenize": args.lang_id})

    print(f"{args.metric.upper()} Score: {result.score}")


if __name__ == "__main__":
    main()
