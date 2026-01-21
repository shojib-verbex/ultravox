from ultravox.data import types

# MMAU (Massive Multi-Task Audio Understanding) dataset
# HuggingFace: AudioLLMs/MMAU-mini
# Paper: https://arxiv.org/abs/2410.19168
# Tasks: 27 distinct audio understanding skills across speech, sound, and music
#
# Dataset contains three task types in `other_attributes.task`:
# - speech: ~520 samples (speaker ID, emotion, ASR-related tasks)
# - sound: ~240 samples (environmental sound classification)
# - music: ~240 samples (genre, instrument, tempo tasks)

# User template: Audio + instruction + choices
# Format choices list as newline-separated options
MMAU_USER_TEMPLATE = f"{types.AUDIO_PLACEHOLDER}\n\n{{{{instruction}}}}\n\n{{% for choice in choices %}}{{{{choice}}}}\n{{% endfor %}}"

# Base config for full MMAU (all tasks)
MMAU_BASE_CONFIG = types.DatasetConfig(
    name="mmau",
    path="AudioLLMs/MMAU-mini",
    subset="MMAU-mini",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=1000, split=types.DatasetSplit.TEST
        ),
    ],
    user_template=MMAU_USER_TEMPLATE,
    audio_field="context",
    # The instruction field contains the question
    transcript_template="{{instruction}}",
    # Answer format is "(A) Description" - extract just "A" for voicebench_mcq comparison
    # voicebench_mcq expects ground truth as single letter: A, B, C, or D
    assistant_template="{{answer[1] if answer.startswith('(') else answer}}",
    eval_config=types.EvalConfig(
        metric="voicebench_mcq",
        # Stricter scoring: no random guessing when answer can't be extracted
        # This prevents inflated scores from "I can't hear the audio" responses
        args={"random_on_parse_fail": False},
        # Map task type from dataset to extra_kwargs for breakdown reporting
        extra_kwargs_map={"task_type": "other_attributes.task"},
        # Report separate scores for each task type (speech, sound, music)
        breakdown_fields=["task_type"],
    ),
)

# Speech-only variant (~520 samples) - For voice/telephony assistant evaluation
# Filters to only speech tasks using other_attributes.task field
# Tasks include: speaker identification, emotion recognition, ASR-related, spoken language understanding
MMAU_SPEECH_CONFIG = types.DatasetConfig(
    name="mmau-speech",
    base="mmau",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=520, split=types.DatasetSplit.TEST
        ),
    ],
    row_filter={"other_attributes.task": "speech"},
)

# Sound-only variant (~240 samples) - Environmental sound understanding
# Tasks include: sound event detection, acoustic scene classification, source inference
MMAU_SOUND_CONFIG = types.DatasetConfig(
    name="mmau-sound",
    base="mmau",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=240, split=types.DatasetSplit.TEST
        ),
    ],
    row_filter={"other_attributes.task": "sound"},
)

# Music-only variant (~240 samples) - Music understanding
# Tasks include: genre classification, instrument identification, tempo detection
MMAU_MUSIC_CONFIG = types.DatasetConfig(
    name="mmau-music",
    base="mmau",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=240, split=types.DatasetSplit.TEST
        ),
    ],
    row_filter={"other_attributes.task": "music"},
)

configs = [
    MMAU_BASE_CONFIG,
    MMAU_SPEECH_CONFIG,
    MMAU_SOUND_CONFIG,
    MMAU_MUSIC_CONFIG,
]
