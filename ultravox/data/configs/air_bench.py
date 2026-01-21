from ultravox.data import types

# AIR-Bench (Audio InstRuction Benchmark) dataset
# HuggingFace: qyang1021/AIR-Bench-Dataset
# Paper: https://arxiv.org/abs/2402.07729
# License: CC-BY-NC-4.0
#
# AIR-Bench is the first benchmark for evaluating Large Audio-Language Models (LALMs).
# It assesses models on understanding various audio types (speech, sound, music)
# and following instructions.
#
# Foundation Benchmark: 19 tasks with ~24.7k multiple-choice questions
# - Speech: Gender, Age, Emotion, Intent, Entity Recognition, Grounding, Language ID, etc.
# - Sound: Acoustic Scene Classification, Sound QA, Audio Grounding, Vocal Sound
# - Music: Genre, Instruments, Mood, MIDI Analysis, Music QA
#
# The dataset requires custom loading as audio and metadata are stored separately.
# Audio is in Foundation/{task_name}_{dataset_name}/ folders
# Metadata is in Foundation/Foundation_meta.json

# User template: Audio + Question + Multiple choice options (A/B/C/D)
AIR_BENCH_USER_TEMPLATE = (
    f"{types.AUDIO_PLACEHOLDER}\n\n"
    "{{question}}\n\n"
    "A. {{choice_a}}\n"
    "B. {{choice_b}}\n"
    "C. {{choice_c}}\n"
    "{% if choice_d is defined and choice_d %}D. {{choice_d}}\n{% endif %}"
)

# Base config for AIR-Bench Foundation benchmark
# Note: This uses a custom dataset class (AIRBenchDataset) due to the
# non-standard data layout (separate JSON metadata + audio folders)
AIR_BENCH_BASE_CONFIG = types.DatasetConfig(
    name="air-bench",
    path="qyang1021/AIR-Bench-Dataset",
    subset="foundation",  # foundation or chat
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=24683, split=types.DatasetSplit.TEST
        ),
    ],
    user_template=AIR_BENCH_USER_TEMPLATE,
    audio_field="audio",
    transcript_template="",
    # Answer is the ground truth choice content (e.g., "street", "neutral", "A")
    # voicebench_mcq will extract the letter from model response
    assistant_template="{{answer_letter}}",
    eval_config=types.EvalConfig(
        metric="voicebench_mcq",
        args={"random_on_parse_fail": False},
        extra_kwargs_map={"task_type": "task_name"},
        breakdown_fields=["task_type"],
    ),
    # Custom dataset class required for AIR-Bench (separate JSON metadata + audio folders)
    dataset_class="AIRBenchDataset",
)

# ============================================================================
# Speech Tasks (~9k samples)
# ============================================================================

AIR_BENCH_SPEECH_GROUNDING_CONFIG = types.DatasetConfig(
    name="air-bench-speech-grounding",
    base="air-bench",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=981, split=types.DatasetSplit.TEST
        ),
    ],
    row_filter={"task_name": "Speech_Grounding"},
)

AIR_BENCH_SPEAKER_GENDER_CONFIG = types.DatasetConfig(
    name="air-bench-speaker-gender",
    base="air-bench",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=1992, split=types.DatasetSplit.TEST
        ),
    ],
    row_filter={"task_name": "Speaker_Gender_Recognition"},
)

AIR_BENCH_SPEAKER_AGE_CONFIG = types.DatasetConfig(
    name="air-bench-speaker-age",
    base="air-bench",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=1000, split=types.DatasetSplit.TEST
        ),
    ],
    row_filter={"task_name": "Speaker_Age_Prediction"},
)

AIR_BENCH_SPEAKER_EMOTION_CONFIG = types.DatasetConfig(
    name="air-bench-speaker-emotion",
    base="air-bench",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=2000, split=types.DatasetSplit.TEST
        ),
    ],
    row_filter={"task_name": "Speaker_Emotion_Recontion"},
)

AIR_BENCH_SPEAKER_INTENT_CONFIG = types.DatasetConfig(
    name="air-bench-speaker-intent",
    base="air-bench",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=1000, split=types.DatasetSplit.TEST
        ),
    ],
    row_filter={"task_name": "Speaker_Intent_Classification"},
)

AIR_BENCH_SPEAKER_NUMBER_CONFIG = types.DatasetConfig(
    name="air-bench-speaker-number",
    base="air-bench",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=1000, split=types.DatasetSplit.TEST
        ),
    ],
    row_filter={"task_name": "Speaker_Number_Verification"},
)

AIR_BENCH_SPEECH_ENTITY_CONFIG = types.DatasetConfig(
    name="air-bench-speech-entity",
    base="air-bench",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=1000, split=types.DatasetSplit.TEST
        ),
    ],
    row_filter={"task_name": "Speech_Entity_Reconition"},
)

AIR_BENCH_LANGUAGE_ID_CONFIG = types.DatasetConfig(
    name="air-bench-language-id",
    base="air-bench",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=1000, split=types.DatasetSplit.TEST
        ),
    ],
    row_filter={"task_name": "Spoken_Language_Identification"},
)

AIR_BENCH_SYNTHESIZED_VOICE_CONFIG = types.DatasetConfig(
    name="air-bench-synthesized-voice",
    base="air-bench",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=1000, split=types.DatasetSplit.TEST
        ),
    ],
    row_filter={"task_name": "Synthesized_Voice_Detection"},
)

# ============================================================================
# Sound Tasks (~5k samples)
# ============================================================================

AIR_BENCH_ACOUSTIC_SCENE_CONFIG = types.DatasetConfig(
    name="air-bench-acoustic-scene",
    base="air-bench",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=2000, split=types.DatasetSplit.TEST
        ),
    ],
    row_filter={"task_name": "Acoustic_Scene_Classification"},
)

AIR_BENCH_SOUND_AQA_CONFIG = types.DatasetConfig(
    name="air-bench-sound-aqa",
    base="air-bench",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=2000, split=types.DatasetSplit.TEST
        ),
    ],
    row_filter={"task_name": "Sound_AQA"},
)

AIR_BENCH_AUDIO_GROUNDING_CONFIG = types.DatasetConfig(
    name="air-bench-audio-grounding",
    base="air-bench",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=896, split=types.DatasetSplit.TEST
        ),
    ],
    row_filter={"task_name": "Audio_Grounding"},
)

AIR_BENCH_VOCAL_SOUND_CONFIG = types.DatasetConfig(
    name="air-bench-vocal-sound",
    base="air-bench",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=1000, split=types.DatasetSplit.TEST
        ),
    ],
    row_filter={"task_name": "vocal_sound_classification"},
)

# ============================================================================
# Music Tasks (~8.8k samples)
# ============================================================================

AIR_BENCH_MUSIC_GENRE_CONFIG = types.DatasetConfig(
    name="air-bench-music-genre",
    base="air-bench",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=2000, split=types.DatasetSplit.TEST
        ),
    ],
    row_filter={"task_name": "Music_Genre_Recognition"},
)

AIR_BENCH_MUSIC_INSTRUMENTS_CONFIG = types.DatasetConfig(
    name="air-bench-music-instruments",
    base="air-bench",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=2000, split=types.DatasetSplit.TEST
        ),
    ],
    row_filter={"task_name": "Music_Instruments_Classfication"},
)

AIR_BENCH_MUSIC_MOOD_CONFIG = types.DatasetConfig(
    name="air-bench-music-mood",
    base="air-bench",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=1000, split=types.DatasetSplit.TEST
        ),
    ],
    row_filter={"task_name": "Music_Mood_Recognition"},
)

AIR_BENCH_MUSIC_PITCH_CONFIG = types.DatasetConfig(
    name="air-bench-music-pitch",
    base="air-bench",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=1000, split=types.DatasetSplit.TEST
        ),
    ],
    row_filter={"task_name": "Music_Midi_Pitch_Analysis"},
)

AIR_BENCH_MUSIC_VELOCITY_CONFIG = types.DatasetConfig(
    name="air-bench-music-velocity",
    base="air-bench",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=1000, split=types.DatasetSplit.TEST
        ),
    ],
    row_filter={"task_name": "Music_Midi_Velocity_Analysis"},
)

AIR_BENCH_MUSIC_AQA_CONFIG = types.DatasetConfig(
    name="air-bench-music-aqa",
    base="air-bench",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=814, split=types.DatasetSplit.TEST
        ),
    ],
    row_filter={"task_name": "Music_AQA"},
)

# ============================================================================
# Grouped configs by domain
# ============================================================================

# All speech tasks combined (~9k samples)
AIR_BENCH_SPEECH_CONFIG = types.DatasetConfig(
    name="air-bench-speech",
    base="air-bench",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=9973, split=types.DatasetSplit.TEST
        ),
    ],
    row_filter={"_domain": "speech"},  # Custom filter handled in dataset class
)

# All sound tasks combined (~5.9k samples)
AIR_BENCH_SOUND_CONFIG = types.DatasetConfig(
    name="air-bench-sound",
    base="air-bench",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=5896, split=types.DatasetSplit.TEST
        ),
    ],
    row_filter={"_domain": "sound"},  # Custom filter handled in dataset class
)

# All music tasks combined (~8.8k samples)
AIR_BENCH_MUSIC_CONFIG = types.DatasetConfig(
    name="air-bench-music",
    base="air-bench",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=8814, split=types.DatasetSplit.TEST
        ),
    ],
    row_filter={"_domain": "music"},  # Custom filter handled in dataset class
)

configs = [
    AIR_BENCH_BASE_CONFIG,
    # Speech tasks
    AIR_BENCH_SPEECH_GROUNDING_CONFIG,
    AIR_BENCH_SPEAKER_GENDER_CONFIG,
    AIR_BENCH_SPEAKER_AGE_CONFIG,
    AIR_BENCH_SPEAKER_EMOTION_CONFIG,
    AIR_BENCH_SPEAKER_INTENT_CONFIG,
    AIR_BENCH_SPEAKER_NUMBER_CONFIG,
    AIR_BENCH_SPEECH_ENTITY_CONFIG,
    AIR_BENCH_LANGUAGE_ID_CONFIG,
    AIR_BENCH_SYNTHESIZED_VOICE_CONFIG,
    # Sound tasks
    AIR_BENCH_ACOUSTIC_SCENE_CONFIG,
    AIR_BENCH_SOUND_AQA_CONFIG,
    AIR_BENCH_AUDIO_GROUNDING_CONFIG,
    AIR_BENCH_VOCAL_SOUND_CONFIG,
    # Music tasks
    AIR_BENCH_MUSIC_GENRE_CONFIG,
    AIR_BENCH_MUSIC_INSTRUMENTS_CONFIG,
    AIR_BENCH_MUSIC_MOOD_CONFIG,
    AIR_BENCH_MUSIC_PITCH_CONFIG,
    AIR_BENCH_MUSIC_VELOCITY_CONFIG,
    AIR_BENCH_MUSIC_AQA_CONFIG,
    # Domain groups
    AIR_BENCH_SPEECH_CONFIG,
    AIR_BENCH_SOUND_CONFIG,
    AIR_BENCH_MUSIC_CONFIG,
]
