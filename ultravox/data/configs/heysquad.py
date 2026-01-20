from ultravox.data import types

# User template: Context paragraph + spoken question
HEYSQUAD_USER_TEMPLATE = f"Answer the following question based on the context below. Provide only the answer text.\n\nContext: {{{{context}}}}\n\nQuestion: {types.AUDIO_PLACEHOLDER}"

# Human version assistant template (handles unanswerable questions)
HEYSQUAD_HUMAN_ASSISTANT_TEMPLATE = "{{answers[0]['text'] if answers else (plausible_answers[0]['text'] if plausible_answers else 'unanswerable')}}"

# Human version splits
HEYSQUAD_HUMAN_SPLITS = [
    types.DatasetSplitConfig(
        name="train", num_samples=72000, split=types.DatasetSplit.TRAIN
    ),
    types.DatasetSplitConfig(
        name="validation", num_samples=4160, split=types.DatasetSplit.TEST
    ),
]

# Machine version splits
HEYSQUAD_MACHINE_SPLITS = [
    types.DatasetSplitConfig(
        name="train", num_samples=87600, split=types.DatasetSplit.TRAIN
    ),
    types.DatasetSplitConfig(
        name="validation", num_samples=10600, split=types.DatasetSplit.TEST
    ),
]

# Base config with F1 metric (default)
HEYSQUAD_BASE_CONFIG = types.DatasetConfig(
    name="heysquad",
    user_template=HEYSQUAD_USER_TEMPLATE,
    transcript_template="{{transcription}}",
    eval_config=types.EvalConfig(metric="squad_f1"),
)

# Base config with Exact Match metric
HEYSQUAD_BASE_EM_CONFIG = types.DatasetConfig(
    name="heysquad-em",
    user_template=HEYSQUAD_USER_TEMPLATE,
    transcript_template="{{transcription}}",
    eval_config=types.EvalConfig(metric="squad_exact_match"),
)

# Human-spoken version with F1 metric
HEYSQUAD_HUMAN_CONFIG = types.DatasetConfig(
    name="heysquad-human",
    base="heysquad",
    path="yijingwu/HeySQuAD_human",
    splits=HEYSQUAD_HUMAN_SPLITS,
    assistant_template=HEYSQUAD_HUMAN_ASSISTANT_TEMPLATE,
)

# Human-spoken version with Exact Match metric
HEYSQUAD_HUMAN_EM_CONFIG = types.DatasetConfig(
    name="heysquad-human-em",
    base="heysquad-em",
    path="yijingwu/HeySQuAD_human",
    splits=HEYSQUAD_HUMAN_SPLITS,
    assistant_template=HEYSQUAD_HUMAN_ASSISTANT_TEMPLATE,
)

# Machine-generated version with F1 metric
HEYSQUAD_MACHINE_CONFIG = types.DatasetConfig(
    name="heysquad-machine",
    base="heysquad",
    path="yijingwu/HeySQuAD_machine",
    splits=HEYSQUAD_MACHINE_SPLITS,
    assistant_template="{{answer}}",
)

# Machine-generated version with Exact Match metric
HEYSQUAD_MACHINE_EM_CONFIG = types.DatasetConfig(
    name="heysquad-machine-em",
    base="heysquad-em",
    path="yijingwu/HeySQuAD_machine",
    splits=HEYSQUAD_MACHINE_SPLITS,
    assistant_template="{{answer}}",
)

configs = [
    HEYSQUAD_BASE_CONFIG,
    HEYSQUAD_BASE_EM_CONFIG,
    HEYSQUAD_HUMAN_CONFIG,
    HEYSQUAD_HUMAN_EM_CONFIG,
    HEYSQUAD_MACHINE_CONFIG,
    HEYSQUAD_MACHINE_EM_CONFIG,
]
