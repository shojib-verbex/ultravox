from ultravox.data import types

# MMAU (Massive Multi-Task Audio Understanding) dataset
# HuggingFace: AudioLLMs/MMAU-mini
# Paper: https://arxiv.org/abs/2410.19168
# Tasks: 27 distinct audio understanding skills across speech, sound, and music

# User template: Audio + instruction + choices
# Format choices list as newline-separated options
MMAU_USER_TEMPLATE = f"{types.AUDIO_PLACEHOLDER}\n\n{{{{instruction}}}}\n\n{{% for choice in choices %}}{{{{choice}}}}\n{{% endfor %}}"

# Base config for MMAU
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
    eval_config=types.EvalConfig(metric="voicebench_mcq"),
)

configs = [MMAU_BASE_CONFIG]
