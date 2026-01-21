from ultravox.data import types

# Dynamic-SUPERB dataset configs
# HuggingFace: DynamicSuperb/* (325 datasets, 180 tasks)
# Paper: https://arxiv.org/abs/2309.09510
# License: CC-BY 4.0
#
# Selected tasks for Ultravox (speech-focused):
# 1. Emotion Recognition (RAVDESS) - 8 emotion classes
# 2. Speech Sentiment (MELD) - 3 sentiment classes
# 3. Stuttering Detection (SEP28k) - binary classification
# 4. Accent Classification (AccentDB) - 9 accent classes
# 5. ASR (LibriSpeech-TestOther) - transcription

# Common template: Audio + instruction (instruction already includes answer options)
DS_USER_TEMPLATE = f"{types.AUDIO_PLACEHOLDER}\n\n{{{{instruction}}}}"

# Base config for Dynamic-SUPERB tasks
DYNAMIC_SUPERB_BASE = types.DatasetConfig(
    name="dynamic-superb",
    user_template=DS_USER_TEMPLATE,
    audio_field="audio",
    transcript_template="",
    assistant_template="{{label}}",
    eval_config=types.EvalConfig(metric="exact_match"),
)

# Emotion Recognition - RAVDESS (240 samples)
# Classes: neutral, calm, happy, sad, angry, fearful, disgust, surprised
DS_EMOTION_CONFIG = types.DatasetConfig(
    name="dynamic-superb-emotion",
    base="dynamic-superb",
    path="DynamicSuperb/SuperbER_Ravdess",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=240, split=types.DatasetSplit.TEST
        )
    ],
)

# Speech Sentiment Analysis - MELD (200 samples)
# Classes: Positive, Negative, Neutral
DS_SENTIMENT_CONFIG = types.DatasetConfig(
    name="dynamic-superb-sentiment",
    base="dynamic-superb",
    path="DynamicSuperb/SpeechSentimentAnalysis_Meld",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=200, split=types.DatasetSplit.TEST
        )
    ],
)

# Stuttering Detection - SEP28k (1000 samples)
# Classes: yes, no (binary)
DS_STUTTERING_CONFIG = types.DatasetConfig(
    name="dynamic-superb-stuttering",
    base="dynamic-superb",
    path="DynamicSuperb/StutteringDetection_SEP28k",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=1000, split=types.DatasetSplit.TEST
        )
    ],
)

# Accent Classification - AccentDB Extended (200 samples)
# Classes: american, australian, bangla, british, indian, malayalam, odiya, telugu, welsh
DS_ACCENT_CONFIG = types.DatasetConfig(
    name="dynamic-superb-accent",
    base="dynamic-superb",
    path="DynamicSuperb/AccentClassification_AccentdbExtended",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=200, split=types.DatasetSplit.TEST
        )
    ],
)

# ASR - LibriSpeech Test-Other (200 samples)
# Uses WER metric instead of exact_match
DS_ASR_CONFIG = types.DatasetConfig(
    name="dynamic-superb-asr",
    base="dynamic-superb",
    path="DynamicSuperb/SuperbASR_LibriSpeech-TestOther",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=200, split=types.DatasetSplit.TEST
        )
    ],
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "en"}),
)

configs = [
    DYNAMIC_SUPERB_BASE,
    DS_EMOTION_CONFIG,
    DS_SENTIMENT_CONFIG,
    DS_STUTTERING_CONFIG,
    DS_ACCENT_CONFIG,
    DS_ASR_CONFIG,
]
