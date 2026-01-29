import abc
import logging
import os
import tempfile
import warnings
from typing import Any, Dict, List, Optional, Sequence

import datasets as hf_datasets
import jinja2
import numpy as np
import streaming as mds
import transformers
from torch.utils import data

from ultravox.data import data_sample
from ultravox.data import text_proc
from ultravox.data import types

# TODO(juberti): set these in the environment so they don't need to be hard-coded here.
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_account.json"
os.environ["GOOGLE_CLOUD_PROJECT"] = "fixie-training"

# Silence the spurious warnings coming from the MosaicML streaming library.
logging.getLogger("streaming.base.dataset").setLevel(logging.ERROR)


def _get_messages(
    user_message: str,
    assistant_message: str,
    message_history: Optional[List[Dict[str, str]]] = None,
    sys_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Convert a user message, assistant message, and optional message history into a list of messages.
    If `sys_prompt` is set, it is prepended as a system message.
    """
    messages = []

    if sys_prompt is not None:
        messages.append({"role": "system", "content": sys_prompt})

    if message_history is not None:
        # For now, we only support chat history with user and assistant messages.
        assert all("role" in msg and "content" in msg for msg in message_history)
        assert all(msg["role"] in ["user", "assistant"] for msg in message_history)
        messages.extend(message_history)

    messages.append({"role": "user", "content": user_message})
    messages.append({"role": "assistant", "content": assistant_message})

    return messages


def _get_worker_info(length: int):
    """
    Calculate number of samples for this worker, accounting for max workers limit.
    Returns 0 if worker_id exceeds max allowed workers.
    """
    worker_id = 0
    num_workers = 1
    worker_info = data.get_worker_info()
    if worker_info is not None:
        worker_id = worker_info.id
        num_workers = worker_info.num_workers

    # Calculate samples for this worker
    worker_samples = length // num_workers
    extra_samples = length % num_workers

    # Workers with id < extra_samples get one extra sample
    if worker_id < extra_samples:
        worker_samples += 1

    return num_workers, worker_id, worker_samples


class SizedIterableDataset(abc.ABC, data.IterableDataset):
    """
    An interface for an IterableDataset that provides a length method.
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass


class VoiceDataset(SizedIterableDataset):
    """
    Base class for streaming voice datasets.
    Wraps a Hugging Face dataset or MDS-formatted dataset from GCP.
    """

    def __init__(self, args: types.VoiceDatasetArgs) -> None:
        super().__init__()
        self._args = args
        self._rng = np.random.default_rng(self._args.shuffle_seed)
        self._name = "[unset]"
        self._length = -1

    # num_samples is the total number of samples in the dataset
    def _init_dataset(
        self,
        dataset: data.Dataset,
        name: str,
        num_samples: int,
    ) -> None:
        self._dataset = dataset
        self._name = name
        self._length = num_samples

    def __len__(self):
        return self._length

    @property
    def name(self):
        return self._name

    def _load_hf_dataset(
        self,
        path: str,
        name: Optional[str] = None,
        *,
        split: Optional[str] = None,
        streaming: bool = True,
        audio_field: Optional[str] = None,
    ) -> data.Dataset:
        # HF datasets sometimes fails to download due to network issues, so retry a few times.
        dataset = hf_datasets.load_dataset(
            path,
            name,
            split=split,
            trust_remote_code=True,
            streaming=streaming,
            download_config=hf_datasets.DownloadConfig(max_retries=10),
        )
        if audio_field is not None:
            dataset = dataset.cast_column(
                audio_field, hf_datasets.Audio(sampling_rate=data_sample.SAMPLE_RATE)
            )
        if self._args.shuffle:
            if streaming:
                dataset = dataset.shuffle(
                    seed=self._args.shuffle_seed,
                    buffer_size=self._args.shuffle_buffer_size,
                )
            else:
                dataset = dataset.shuffle(seed=self._args.shuffle_seed)
        return dataset

    def _load_mds_dataset(
        self,
        path: str,
        name: Optional[str] = None,
        *,
        split: Optional[str] = None,
        batch_size: int = 1,
    ) -> data.Dataset:
        gcs_path = path.replace("/", "_")
        if name:
            gcs_path += f"/{name}"
        if split:
            gcs_path += f"/{split}"
        url = f"gs://fixie-datasets/mds/{gcs_path}"
        temp_dir = os.path.join(
            tempfile.gettempdir(), f"mds_{gcs_path.replace('/', '_')}"
        )
        return mds.StreamingDataset(
            remote=url,
            local=temp_dir,
            batch_size=batch_size,
            shuffle=self._args.shuffle,
            shuffle_seed=self._args.shuffle_seed,
        )

    def __iter__(self):
        num_workers, _, _ = _get_worker_info(self._length)
        if num_workers > 1:
            assert hasattr(
                self._dataset, "n_shards"
            ), f"{self._name} does not have n_shards attribute, which is required when num_workers ({num_workers}) > 1"
            assert (
                self._dataset.n_shards >= num_workers
            ), f"{self._name} has {self._dataset.n_shards} shards, which is less than the number of workers ({num_workers})."

        actual_length = 0
        skipped_samples = 0
        bad_samples = 0
        dataset_iter = iter(self._dataset)
        for row in dataset_iter:
            actual_length += 1
            sample = self._get_sample(row)
            if sample is None:
                print(f"Sample is None in dataset {self.name} for row {row}")
                bad_samples += 1
                continue

            input_characters = sum(len(msg["content"]) for msg in sample.messages)
            if (
                self._args.max_input_characters is not None
                and input_characters > self._args.max_input_characters
            ):
                print(
                    f"Sample has input characters longer than {self._args.max_input_characters} in dataset {self.name} for row {row}"
                )
                bad_samples += 1
                continue

            elif len(sample.messages[-1]["content"]) == 0:
                print(
                    f"Sample has empty assistant message in dataset {self.name} for row {row}"
                )
                bad_samples += 1
                continue

            if self._args.include_audio:
                if sample.audio is None:
                    print(f"Audio is None for sample {sample}")
                    bad_samples += 1
                    continue
                if sample.audio.shape[-1] == 0:
                    print(f"Audio length is 0 for sample {sample}")
                    bad_samples += 1
                    continue
                if (
                    self._args.max_audio_duration_secs > 0
                    and sample.audio.shape[-1] / data_sample.SAMPLE_RATE
                    > self._args.max_audio_duration_secs
                ):
                    skipped_samples += 1
                    continue

            yield sample

        logging.info(
            f"Extracted {actual_length} samples from {self.name} (total: {len(self)}), removed {bad_samples} bad samples, and skipped {skipped_samples} samples for exceeding max audio duration ({self._args.max_audio_duration_secs}s)."
        )

    @abc.abstractmethod
    def _get_sample(
        self, row: transformers.BatchFeature
    ) -> Optional[data_sample.VoiceSample]:
        """
        Converts a row from the dataset into a VoiceSample.
        Returns None if the sample should be skipped.
        """

    def _get_nested_field(self, row: Dict[str, Any], field_path: str) -> Any:
        """Get a field value from row, supporting dot notation for nested access.

        Examples:
            _get_nested_field(row, "id") -> row["id"]
            _get_nested_field(row, "other_attributes.task") -> row["other_attributes"]["task"]
        """
        value = row
        for key in field_path.split("."):
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        return value

    def _check_row_filter(self, row: Dict[str, Any]) -> bool:
        """Check if row passes the configured filter. Returns True if row should be included."""
        if self._config.row_filter is None:
            return True

        for field_path, expected_value in self._config.row_filter.items():
            value = self._get_nested_field(row, field_path)
            if value != expected_value:
                return False
        return True

    def _get_audio(
        self, row: transformers.BatchFeature, column_name: Optional[str] = "audio"
    ) -> np.ndarray:
        # Hugging Face datasets have an Audio object, with array and sampling_rate fields.
        # For MDS, this object is flattened into audio_array and audio_sampling_rate fields.
        if column_name in row:
            audio = row[column_name]["array"]
            sampling_rate = row[column_name]["sampling_rate"]
        elif f"{column_name}_array" in row:
            audio = row[f"{column_name}_array"]
            sampling_rate = row[f"{column_name}_sampling_rate"]
        else:
            raise ValueError("No audio field found in row.")
        assert sampling_rate == data_sample.SAMPLE_RATE
        return audio

    def _make_messages(
        self, user_content: str, assistant_content: str
    ) -> List[Dict[str, str]]:
        return _get_messages(user_content, assistant_content)

    def _make_sample(
        self,
        messages: List[Dict[str, str]],
        audio: Optional[np.ndarray] = None,
        audio_transcript: Optional[str] = None,
        label: Optional[str] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> data_sample.VoiceSample:
        if not self._args.include_audio:
            return data_sample.VoiceSample(
                messages,
                label=label,
                extra_kwargs=extra_kwargs,
            )
        return data_sample.VoiceSample(
            messages,
            audio,
            audio_transcript=audio_transcript,
            label=label,
            extra_kwargs=extra_kwargs,
        )


class GenericDataset(VoiceDataset):
    def __init__(
        self,
        args: types.VoiceDatasetArgs,
        config: types.DatasetConfig,
    ) -> None:
        assert config.splits is not None
        assert config.path is not None
        assert config.mds_batch_size is not None
        super().__init__(args)
        self._config = config
        dsets = []
        total_samples = 0
        for split in config.splits:
            if split.split == self._args.split:
                if not config.use_mds:
                    ds = self._load_hf_dataset(
                        config.path,
                        config.subset,
                        split=split.name,
                        audio_field=config.audio_field,
                    )
                else:
                    ds = self._load_mds_dataset(
                        config.path,
                        name=config.subset,
                        split=split.name,
                        batch_size=config.mds_batch_size,
                    )
                dsets.append(ds)
                total_samples += split.num_samples
        assert (
            len(dsets) > 0
        ), f"The {config.name} dataset has no {self._args.split} splits."
        dataset = ds if len(dsets) == 1 else hf_datasets.concatenate_datasets(dsets)

        dataset_name = f"{config.name}.{self._args.split.value}"

        if self._config.messages_direct_column is None:
            assert self._config.transcript_template is not None
            assert self._config.user_template is not None
            assert self._config.user_template_args is not None
            assert (
                self._config.message_history_roles is not None
                if self._config.message_history_column is not None
                else True
            ), "message_history_roles must be provided if message_history_column is provided"
            assert self._config.assistant_template is not None

        super()._init_dataset(dataset, dataset_name, total_samples)

    def __str__(self):
        return f"GenericDataset({self._config})"

    def _get_sample(self, row) -> Optional[data_sample.VoiceSample]:
        # Check row filter first - return None to skip filtered-out samples
        if not self._check_row_filter(row):
            return None

        # Setting up extra_kwargs for datasets like Voicebench
        # Supports nested field access via dot notation (e.g., "other_attributes.task")
        extra_kwargs = None
        if (
            self._config.eval_config is not None
            and self._config.eval_config.extra_kwargs_map is not None
        ):
            extra_kwargs = {
                key: self._get_nested_field(row, field_path)
                for key, field_path in self._config.eval_config.extra_kwargs_map.items()
            }

        # If the messages_direct_column is provided, we use it directly to create the messages and transcript.
        if self._config.messages_direct_column is not None:
            messages = row[self._config.messages_direct_column]
            if len(messages) == 0:
                raise ValueError("messages_direct_column is empty")

            label = (
                row[self._config.label_column]
                if self._config.label_column is not None
                else None
            )

            if not self._args.include_audio:
                return self._make_sample(
                    messages,
                    label=label,
                    extra_kwargs=extra_kwargs,
                )

            transcript = jinja2.Template(
                self._config.transcript_template,  # type: ignore[arg-type]
                undefined=jinja2.StrictUndefined,
            ).render(**row, text_proc=text_proc)
            audio = self._get_audio(row, self._config.audio_field)
            return self._make_sample(
                messages,
                audio,
                audio_transcript=transcript,
                extra_kwargs=extra_kwargs,
            )

        # Convert the dataset's message_history_column into a list of messages
        message_history = (
            text_proc.format_message_history(
                row[self._config.message_history_column],
                self._config.message_history_roles,
            )
            if self._config.message_history_column is not None
            and self._config.message_history_roles is not None
            and not self._args.ignore_message_history
            else None
        )

        try:
            user_content = jinja2.Template(
                self._config.user_template,  # type: ignore[arg-type]
                undefined=jinja2.StrictUndefined,
            ).render(
                **row,
                text_proc=text_proc,
                **self._config.user_template_args,  # type: ignore[arg-type]
            )
            assistant_content = jinja2.Template(
                self._config.assistant_template,  # type: ignore[arg-type]
                undefined=jinja2.StrictUndefined,
            ).render(**row, text_proc=text_proc)
            transcript = jinja2.Template(
                self._config.transcript_template,  # type: ignore[arg-type]
                undefined=jinja2.StrictUndefined,
            ).render(**row, text_proc=text_proc)
            system_prompt = (
                jinja2.Template(
                    self._config.system_prompt_template,  # type: ignore[arg-type]
                    undefined=jinja2.StrictUndefined,
                ).render(**row, text_proc=text_proc)
                if self._config.system_prompt_template is not None
                and not self._args.ignore_system_prompt
                else None
            )

        except jinja2.TemplateError as e:
            print(f"Error rendering template: {e}")
            print(f"user_template: {self._config.user_template}")
            print(f"assistant_template: {self._config.assistant_template}")
            print(f"transcript_template: {self._config.transcript_template}")
            print(f"system_prompt_template: {self._config.system_prompt_template}")
            print(f"sample keys: {list(row.keys())}")
            raise ValueError(
                "Template rendering failed. Make sure all keys in the template exist in the sample."
            ) from e
        if not self._args.include_audio:
            user_content = user_content.replace(
                types.AUDIO_PLACEHOLDER, f'"{transcript}"'
            )

        messages = _get_messages(
            user_content,
            assistant_content,
            message_history=message_history,
            sys_prompt=system_prompt,
        )
        audio: Optional[np.ndarray] = (  # type: ignore[no-redef]
            self._get_audio(row, self._config.audio_field)
            if self._args.include_audio
            else None
        )
        return self._make_sample(
            messages,
            audio,
            audio_transcript=transcript,
            extra_kwargs=extra_kwargs,
        )

    def get_config(self):
        return self._config


class LibriSpeechDummyDataset(GenericDataset):
    def __init__(self, args: types.VoiceDatasetArgs) -> None:
        VoiceDataset.__init__(self, args)
        # This dataset doesn't support streaming.
        dataset = self._load_hf_dataset(
            "hf-internal-testing/librispeech_asr_dummy",
            "clean",
            split="validation",
            streaming=False,
        )
        self._init_dataset(dataset, "dummy", 73)

    def __str__(self):
        return "LibriSpeechDummyDataset"

    @property
    def name(self):
        return "dummy"

    def get_config(self):
        return types.DatasetConfig(
            name="dummy",
            path="hf-internal-testing/librispeech_asr_dummy",
        )

    def _get_sample(
        self, row: transformers.BatchFeature
    ) -> Optional[data_sample.VoiceSample]:
        text = text_proc.format_asr_text(row["text"])
        user_content = "Transcribe\n"
        user_content += (
            types.AUDIO_PLACEHOLDER if self._args.include_audio else f'"{text}"'
        )
        return self._make_sample(
            self._make_messages(user_content, text),
            # some of our test models that use this dataset can only handle up to 4 seconds of audio
            self._get_audio(row, "audio")[: 4 * data_sample.SAMPLE_RATE],
            audio_transcript=text,
        )


class AIRBenchDataset(VoiceDataset):
    """
    AIR-Bench (Audio InstRuction Benchmark) dataset.
    https://huggingface.co/datasets/qyang1021/AIR-Bench-Dataset

    This dataset requires custom loading because:
    1. Metadata (questions, answers) is in a separate JSON file
    2. Audio files are in task-specific subfolders
    3. The standard HF loader doesn't properly combine them

    The dataset contains 19 foundation tasks across speech, sound, and music domains.
    """

    # Task name to domain mapping
    SPEECH_TASKS = {
        "Speech_Grounding",
        "Speaker_Gender_Recognition",
        "Speaker_Age_Prediction",
        "Speaker_Emotion_Recontion",
        "Speaker_Intent_Classification",
        "Speaker_Number_Verification",
        "Speech_Entity_Reconition",
        "Spoken_Language_Identification",
        "Synthesized_Voice_Detection",
    }
    SOUND_TASKS = {
        "Acoustic_Scene_Classification",
        "Sound_AQA",
        "Audio_Grounding",
        "vocal_sound_classification",
    }
    MUSIC_TASKS = {
        "Music_Genre_Recognition",
        "Music_Instruments_Classfication",
        "Music_Midi_Pitch_Analysis",
        "Music_Midi_Velocity_Analysis",
        "Music_Mood_Recognition",
        "Music_AQA",
    }

    def __init__(
        self,
        args: types.VoiceDatasetArgs,
        config: types.DatasetConfig,
    ) -> None:
        super().__init__(args)
        self._config = config

        # Load metadata from HuggingFace
        from huggingface_hub import hf_hub_download

        meta_file = hf_hub_download(
            repo_id="qyang1021/AIR-Bench-Dataset",
            filename="Foundation/Foundation_meta.json",
            repo_type="dataset",
        )

        import json

        with open(meta_file, "r") as f:
            self._metadata = json.load(f)

        # Apply row filter if specified
        if config.row_filter:
            self._metadata = self._filter_metadata(self._metadata, config.row_filter)

        # Shuffle if needed
        if self._args.shuffle:
            self._rng.shuffle(self._metadata)

        # Determine the answer letter for each sample (A, B, C, or D)
        for item in self._metadata:
            item["answer_letter"] = self._get_answer_letter(item)

        dataset_name = f"{config.name}.{self._args.split.value}"
        num_samples = min(len(self._metadata), config.splits[0].num_samples)
        self._init_dataset(self._metadata, dataset_name, num_samples)

    def _filter_metadata(
        self, metadata: List[Dict[str, Any]], row_filter: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Filter metadata based on row_filter config."""
        filtered = []
        for item in metadata:
            include = True
            for field, value in row_filter.items():
                if field == "_domain":
                    # Special filter for domain grouping
                    task_name = item.get("task_name", "")
                    if value == "speech" and task_name not in self.SPEECH_TASKS:
                        include = False
                    elif value == "sound" and task_name not in self.SOUND_TASKS:
                        include = False
                    elif value == "music" and task_name not in self.MUSIC_TASKS:
                        include = False
                elif item.get(field) != value:
                    include = False
            if include:
                filtered.append(item)
        return filtered

    def _get_answer_letter(self, item: Dict[str, Any]) -> str:
        """Determine which choice (A/B/C/D) matches the ground truth answer."""
        answer_gt = item.get("answer_gt", "")
        if item.get("choice_a") == answer_gt:
            return "A"
        elif item.get("choice_b") == answer_gt:
            return "B"
        elif item.get("choice_c") == answer_gt:
            return "C"
        elif item.get("choice_d") == answer_gt:
            return "D"
        # Fallback: return the raw answer
        return answer_gt

    def __str__(self):
        return f"AIRBenchDataset({self._config.name})"

    def __iter__(self):
        from huggingface_hub import hf_hub_download
        import soundfile as sf

        yielded = 0
        for item in self._metadata:
            if yielded >= self._length:
                break

            try:
                # Construct audio path: Foundation/{task_name}_{dataset_name}/{filename}
                task_name = item["task_name"]
                dataset_name = item["dataset_name"]
                audio_filename = item["path"]
                folder_name = f"{task_name}_{dataset_name}"

                # Download audio file
                audio_path = hf_hub_download(
                    repo_id="qyang1021/AIR-Bench-Dataset",
                    filename=f"Foundation/{folder_name}/{audio_filename}",
                    repo_type="dataset",
                )

                # Load audio
                audio_data, sr = sf.read(audio_path)

                # Resample if needed
                if sr != data_sample.SAMPLE_RATE:
                    import librosa

                    audio_data = librosa.resample(
                        audio_data, orig_sr=sr, target_sr=data_sample.SAMPLE_RATE
                    )

                # Convert to float32 if needed
                audio_data = audio_data.astype(np.float32)

                # Check audio duration
                if (
                    self._args.max_audio_duration_secs > 0
                    and len(audio_data) / data_sample.SAMPLE_RATE
                    > self._args.max_audio_duration_secs
                ):
                    continue

                # Create the sample
                sample = self._get_sample(item, audio_data)
                if sample is not None:
                    yielded += 1
                    yield sample

            except Exception as e:
                logging.warning(f"Error loading AIR-Bench sample: {e}")
                continue

    def _get_sample(
        self, item: Dict[str, Any], audio: np.ndarray
    ) -> Optional[data_sample.VoiceSample]:
        """Convert a metadata item + audio into a VoiceSample."""
        try:
            # Render user template
            user_content = jinja2.Template(
                self._config.user_template,
                undefined=jinja2.StrictUndefined,
            ).render(**item, text_proc=text_proc)

            # Render assistant template (the answer letter)
            assistant_content = jinja2.Template(
                self._config.assistant_template,
                undefined=jinja2.StrictUndefined,
            ).render(**item, text_proc=text_proc)

            # Build extra_kwargs for breakdown reporting
            extra_kwargs = None
            if (
                self._config.eval_config is not None
                and self._config.eval_config.extra_kwargs_map
            ):
                extra_kwargs = {
                    key: item.get(field_path)
                    for key, field_path in self._config.eval_config.extra_kwargs_map.items()
                }

            if not self._args.include_audio:
                user_content = user_content.replace(
                    types.AUDIO_PLACEHOLDER, '"[audio]"'
                )
                return self._make_sample(
                    self._make_messages(user_content, assistant_content),
                    extra_kwargs=extra_kwargs,
                )

            return self._make_sample(
                self._make_messages(user_content, assistant_content),
                audio,
                audio_transcript="",
                extra_kwargs=extra_kwargs,
            )

        except jinja2.TemplateError as e:
            logging.warning(f"Template error for AIR-Bench sample: {e}")
            return None

    def get_config(self):
        return self._config


class SLURPDataset(GenericDataset):
    """
    SLURP (Spoken Language Understanding Resource Package) dataset.
    https://huggingface.co/datasets/qmeeus/slurp

    SLURP is a benchmark for end-to-end Spoken Language Understanding with:
    - 101 intent classes across 18 domains
    - Slot filling annotations in format: [slot_type : value]
    - ~72k training samples, ~13k test samples

    This class extends GenericDataset to add intent label mapping,
    since the HuggingFace dataset stores intents as integers (0-100).
    """

    # All 101 intent labels from SLURP
    INTENT_LABELS = [
        "addcontact", "alarm_query", "alarm_remove", "alarm_set",
        "audio_volume_down", "audio_volume_mute", "audio_volume_other", "audio_volume_up",
        "calendar_query", "calendar_remove", "calendar_set", "cleaning", "coffee", "convert",
        "cooking_query", "cooking_recipe", "createoradd", "currency",
        "datetime_convert", "datetime_query", "definition",
        "email_addcontact", "email_query", "email_querycontact", "email_sendemail",
        "events", "factoid", "game",
        "general_affirm", "general_commandstop", "general_confirm", "general_dontcare",
        "general_explain", "general_greet", "general_joke", "general_negate",
        "general_praise", "general_quirky", "general_repeat", "greet",
        "hue_lightdim", "hue_lightoff", "hue_lightup",
        "iot_cleaning", "iot_coffee", "iot_hue_lightchange", "iot_hue_lightdim",
        "iot_hue_lightoff", "iot_hue_lighton", "iot_hue_lightup", "iot_wemo_off", "iot_wemo_on",
        "joke", "likeness", "lists_createoradd", "lists_query", "lists_remove", "locations",
        "music", "music_dislikeness", "music_likeness", "music_query", "music_settings",
        "news_query", "play_audiobook", "play_game", "play_music", "play_podcasts", "play_radio",
        "podcasts", "post", "qa_currency", "qa_definition", "qa_factoid", "qa_maths", "qa_stock",
        "query", "querycontact", "quirky", "radio",
        "recommendation_events", "recommendation_locations", "recommendation_movies",
        "remove", "sendemail", "set", "settings", "social_post", "social_query",
        "takeaway_order", "takeaway_query", "ticket", "traffic",
        "transport_query", "transport_taxi", "transport_ticket", "transport_traffic",
        "volume_other", "weather_query", "wemo_off", "wemo_on",
    ]

    # Domain groupings for breakdown reporting
    DOMAIN_MAP = {
        "alarm": ["alarm_query", "alarm_remove", "alarm_set"],
        "audio": ["audio_volume_down", "audio_volume_mute", "audio_volume_other", "audio_volume_up", "volume_other"],
        "calendar": ["calendar_query", "calendar_remove", "calendar_set"],
        "cooking": ["cooking_query", "cooking_recipe"],
        "datetime": ["datetime_convert", "datetime_query"],
        "email": ["email_addcontact", "email_query", "email_querycontact", "email_sendemail", "sendemail"],
        "general": ["general_affirm", "general_commandstop", "general_confirm", "general_dontcare",
                    "general_explain", "general_greet", "general_joke", "general_negate",
                    "general_praise", "general_quirky", "general_repeat", "greet", "joke", "quirky"],
        "iot": ["cleaning", "coffee", "hue_lightdim", "hue_lightoff", "hue_lightup",
                "iot_cleaning", "iot_coffee", "iot_hue_lightchange", "iot_hue_lightdim",
                "iot_hue_lightoff", "iot_hue_lighton", "iot_hue_lightup", "iot_wemo_off", "iot_wemo_on",
                "wemo_off", "wemo_on"],
        "lists": ["createoradd", "lists_createoradd", "lists_query", "lists_remove"],
        "music": ["music", "music_dislikeness", "music_likeness", "music_query", "music_settings",
                  "play_audiobook", "play_game", "play_music", "play_podcasts", "play_radio", "podcasts", "radio"],
        "news": ["news_query"],
        "qa": ["convert", "currency", "definition", "events", "factoid", "game", "likeness", "locations",
               "qa_currency", "qa_definition", "qa_factoid", "qa_maths", "qa_stock", "query"],
        "recommendation": ["recommendation_events", "recommendation_locations", "recommendation_movies"],
        "social": ["addcontact", "post", "querycontact", "social_post", "social_query"],
        "takeaway": ["takeaway_order", "takeaway_query"],
        "transport": ["ticket", "traffic", "transport_query", "transport_taxi", "transport_ticket", "transport_traffic"],
        "weather": ["weather_query"],
    }

    def _get_sample(self, row) -> Optional[data_sample.VoiceSample]:
        # Map intent ID to label name
        intent_id = row.get("intent")
        if isinstance(intent_id, int) and 0 <= intent_id < len(self.INTENT_LABELS):
            row["intent_label"] = self.INTENT_LABELS[intent_id]
        else:
            row["intent_label"] = str(intent_id)

        # Add domain for breakdown reporting
        intent_label = row["intent_label"]
        row["domain"] = self._get_domain(intent_label)

        # Call parent implementation
        return super()._get_sample(row)

    def _get_domain(self, intent_label: str) -> str:
        """Map intent label to domain for breakdown reporting."""
        for domain, intents in self.DOMAIN_MAP.items():
            if intent_label in intents:
                return domain
        return "other"


class EmptyDataset(SizedIterableDataset):
    def __init__(self, length: int = 1) -> None:
        self._length = length

    def __iter__(self):
        return iter([])

    def __len__(self):
        return self._length

    def __str__(self):
        return f"EmptyDataset(length={self._length})"

    @property
    def name(self):
        return "empty"


class InterleaveDataset(SizedIterableDataset):
    """Interleaves multiple SizedIterableDataset objects based on normalized weights."""

    def __init__(
        self,
        datasets: Sequence[SizedIterableDataset],
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Args:
            datasets: A list of SizedIterableDataset objects.
            weights: An optional list of dataset weights, i.e., the number of times it should be repeated.
            seed: Optional seed for reproducibility.
        """
        self._datasets = datasets
        if weights is not None:
            assert len(weights) == len(datasets)
        else:
            weights = [1.0] * len(datasets)
        self._weights = weights
        self._weighted_samples = [int(w * len(d)) for w, d in zip(weights, datasets)]
        self._total_samples = sum(self._weighted_samples)

    def __iter__(self):
        ds_iters = [iter(ds) for ds in self._datasets]
        ds_pos = [0] * len(ds_iters)
        num_workers, worker_id, worker_samples = _get_worker_info(self._total_samples)
        # Find the iterator that is least far along and vend from it.
        for i in range(worker_samples):
            min_fraction = 1.0
            for j in range(len(ds_iters)):
                iter_fraction = ds_pos[j] / self._weighted_samples[j]
                if iter_fraction < min_fraction:
                    min_fraction = iter_fraction
                    iter_index = j
            try:
                yield next(ds_iters[iter_index])
            except StopIteration:
                ds_iters[iter_index] = iter(self._datasets[iter_index])
                try:
                    yield next(ds_iters[iter_index])
                except StopIteration:
                    warnings.warn(
                        f"Dataset {iter_index} is empty for worker {worker_id}/{num_workers}. num_workers is likely too high. Stopping iteration."
                    )
                    break
            ds_pos[iter_index] += 1

    def __len__(self):
        return self._total_samples

    def __str__(self):
        return "+".join([f"{d}:{w:.2f}" for w, d in zip(self._weights, self._datasets)])

    @property
    def name(self):
        return "+".join([ds.name for ds in self._datasets])


class Dataproc(SizedIterableDataset):
    """Base class to preprocess a dataset of VoiceSamples."""

    def __init__(self, dataset: SizedIterableDataset) -> None:
        self._dataset = dataset

    @abc.abstractmethod
    def _process(self, sample: data_sample.VoiceSample) -> Dict[str, Any]:
        pass

    def __iter__(self):
        # Replace generator expression with a regular function that yields items
        for sample in self._dataset:
            yield self._process(sample)

    def __len__(self):
        return len(self._dataset)

    def __str__(self):
        return f"Dataproc({self._dataset})"

    @property
    def name(self):
        return self._dataset.name


class Range(SizedIterableDataset):
    """Limits the number of samples from another dataset."""

    def __init__(
        self,
        dataset: SizedIterableDataset,
        num_samples: Optional[int] = None,
    ) -> None:
        self._dataset = dataset
        self._length = num_samples or len(dataset)
        if self._length > len(dataset):
            warnings.warn(
                f"num_samples ({self._length}) exceeds dataset length ({len(dataset)}). Truncating to {len(dataset)}."
            )
            self._length = len(dataset)
        self._name = f"{dataset.name}.{self._length}"

    def __iter__(self):
        num_workers, worker_id, worker_samples = _get_worker_info(self._length)
        if worker_samples == 0:
            return iter([])
        yielded_samples = 0
        try:
            for sample in self._dataset:
                yielded_samples += 1
                yield sample
                if yielded_samples == worker_samples:
                    break
        except Exception as e:
            logging.error(
                f"Worker {worker_id}/{num_workers} failed after yielding {yielded_samples}/{worker_samples} samples, out of {self._length} total samples with error: {e}"
            )
            raise e
        if yielded_samples < worker_samples:
            logging.warn(
                f"Worker {worker_id}/{num_workers} only yielded {yielded_samples} (expected {worker_samples}) samples, out of {self._length} total samples"
            )

    def __str__(self):
        return f"Range({self._dataset}%{len(self)})"

    def __len__(self):
        return self._length

    @property
    def name(self):
        return self._name

    def get_config(self):
        if hasattr(self._dataset, "get_config"):
            return self._dataset.get_config()
        else:
            raise ValueError(f"Cannot get config for {type(self._dataset).__name__}")
