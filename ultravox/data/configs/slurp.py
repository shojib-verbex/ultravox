from ultravox.data import types

# SLURP (Spoken Language Understanding Resource Package) dataset
# HuggingFace: qmeeus/slurp
# Paper: https://arxiv.org/abs/2011.13205
# License: CC-BY-4.0 (text), CC-BY-NC-4.0 (audio)
#
# SLURP is a benchmark for end-to-end Spoken Language Understanding:
# - 101 intent classes across 18 domains
# - Slot filling with annotations like [slot_type : value]
# - ~72k training samples, ~13k test samples
#
# Domains: alarm, audio, calendar, cooking, datetime, email, general,
#          iot, lists, music, news, qa, recommendation, social,
#          takeaway, transport, weather

# ============================================================================
# Intent Classification Task
# ============================================================================

# User template for intent classification
SLURP_INTENT_USER_TEMPLATE = (
    f"{types.AUDIO_PLACEHOLDER}\n\n"
    "What is the intent of this spoken utterance?\n"
    "Respond with only the intent label (e.g., alarm_set, weather_query, play_music)."
)

# Base config for SLURP intent classification
SLURP_INTENT_CONFIG = types.DatasetConfig(
    name="slurp-intent",
    path="qmeeus/slurp",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=13078, split=types.DatasetSplit.TEST
        ),
    ],
    user_template=SLURP_INTENT_USER_TEMPLATE,
    audio_field="audio",
    transcript_template="{{sentence}}",
    # intent_label is added by SLURPDataset from the integer intent field
    assistant_template="{{intent_label}}",
    eval_config=types.EvalConfig(
        metric="exact_match",
        extra_kwargs_map={"domain": "domain"},
        breakdown_fields=["domain"],
    ),
    dataset_class="SLURPDataset",
)

# Development set for quick iteration
SLURP_INTENT_DEV_CONFIG = types.DatasetConfig(
    name="slurp-intent-dev",
    base="slurp-intent",
    splits=[
        types.DatasetSplitConfig(
            name="devel", num_samples=8690, split=types.DatasetSplit.VALIDATION
        ),
    ],
)

# ============================================================================
# Slot Filling Task
# ============================================================================

# User template for slot filling
SLURP_SLOT_USER_TEMPLATE = (
    f"{types.AUDIO_PLACEHOLDER}\n\n"
    "Extract entities from this spoken utterance.\n"
    "Use the format: [slot_type : value] for each entity found.\n"
    "Include the non-entity words as well to form the complete annotation."
)

# Config for SLURP slot filling
SLURP_SLOT_CONFIG = types.DatasetConfig(
    name="slurp-slot",
    path="qmeeus/slurp",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=13078, split=types.DatasetSplit.TEST
        ),
    ],
    user_template=SLURP_SLOT_USER_TEMPLATE,
    audio_field="audio",
    transcript_template="{{sentence}}",
    # annotation contains slot-annotated utterance like:
    # "event reminder [event_name : mona] [date : tuesday]"
    assistant_template="{{annotation}}",
    eval_config=types.EvalConfig(
        metric="slot_f1",
        extra_kwargs_map={"domain": "domain"},
        breakdown_fields=["domain"],
    ),
    dataset_class="SLURPDataset",
)

# Development set for slot filling
SLURP_SLOT_DEV_CONFIG = types.DatasetConfig(
    name="slurp-slot-dev",
    base="slurp-slot",
    splits=[
        types.DatasetSplitConfig(
            name="devel", num_samples=8690, split=types.DatasetSplit.VALIDATION
        ),
    ],
)

# ============================================================================
# Combined SLU Task (Intent + Slots)
# ============================================================================

# User template for combined SLU
SLURP_SLU_USER_TEMPLATE = (
    f"{types.AUDIO_PLACEHOLDER}\n\n"
    "Analyze this spoken utterance:\n"
    "1. Identify the intent\n"
    "2. Extract entities using [slot_type : value] format\n\n"
    "Respond in format:\n"
    "Intent: <intent_label>\n"
    "Annotation: <annotated utterance with slots>"
)

# Config for combined SLU evaluation
SLURP_SLU_CONFIG = types.DatasetConfig(
    name="slurp-slu",
    path="qmeeus/slurp",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=13078, split=types.DatasetSplit.TEST
        ),
    ],
    user_template=SLURP_SLU_USER_TEMPLATE,
    audio_field="audio",
    transcript_template="{{sentence}}",
    # Combined format for evaluation
    assistant_template="Intent: {{intent_label}}\nAnnotation: {{annotation}}",
    eval_config=types.EvalConfig(
        metric="slu_accuracy",
        extra_kwargs_map={"domain": "domain"},
        breakdown_fields=["domain"],
    ),
    dataset_class="SLURPDataset",
)

# ============================================================================
# Transcription Task (ASR baseline)
# ============================================================================

SLURP_ASR_USER_TEMPLATE = f"{types.AUDIO_PLACEHOLDER}\n\nTranscribe this utterance."

SLURP_ASR_CONFIG = types.DatasetConfig(
    name="slurp-asr",
    path="qmeeus/slurp",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=13078, split=types.DatasetSplit.TEST
        ),
    ],
    user_template=SLURP_ASR_USER_TEMPLATE,
    audio_field="audio",
    transcript_template="{{sentence}}",
    assistant_template="{{sentence}}",
    eval_config=types.EvalConfig(
        metric="wer",
        args={"lang_id": "en"},
    ),
    dataset_class="SLURPDataset",
)

configs = [
    SLURP_INTENT_CONFIG,
    SLURP_INTENT_DEV_CONFIG,
    SLURP_SLOT_CONFIG,
    SLURP_SLOT_DEV_CONFIG,
    SLURP_SLU_CONFIG,
    SLURP_ASR_CONFIG,
]
